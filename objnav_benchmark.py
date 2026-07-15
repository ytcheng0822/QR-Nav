import habitat
import os
import warnings
import argparse
import csv
import cv2
import imageio
import numpy as np
import random
import time  # 用來以系統時間重置隨機種子

from cv_utils.detection_tools import *
from cv_utils.segmentation_tools import *
from tqdm import tqdm
from constants import *
from config_utils import hm3d_config
from gpt4v_planner import GPT4V_Planner
from policy_agent import Policy_Agent
from depth_estimator import DepthEstimator  # Re-perception Module 的 Monocular Depth Estimation
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# ==========================================
# Log 寫入工具
# ==========================================
LOG_FILE = "log.txt"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== Object Navigation Benchmark Log ===\n")


def print_and_log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'], 1024), cv2.COLOR_BGR2RGB)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=10)
    # 除錯用：指定只跑某一個 episode_id（可選擇搭配 --scene_id 避免不同場景出現相同 episode_id 時誤選）
    parser.add_argument("--episode_id", type=str, default=None,
                         help="只評估指定的 episode_id，例如 --episode_id 9（log 中的 'ID: 9'）")
    parser.add_argument("--scene_id", type=str, default=None,
                         help="搭配 --episode_id 使用，例如 --scene_id 6s7QHgap2fW（log 中的 '場景' 檔名，可只給前綴）")
    # 消融實驗模式（三選一）：
    #   full          完整方法（預設）
    #   no_memory     w/o Semantic Memory Queue：關閉 LLM 的歷史記憶，每步只看當下全景圖做決策
    #   no_reperception  w/o Re-perception Module：關閉深度估計 + DINO 精確測距，改由小腦 Stop 動作直接判定成功
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_memory", "no_reperception"],
        help=(
            "消融實驗模式：\n"
            "  full             完整方法（預設）\n"
            "  no_memory        w/o Semantic Memory Queue\n"
            "  no_reperception  w/o Re-perception Module"
        ),
    )
    return parser.parse_args()


# ==========================================
# 動作輔助函式
# 用來消除原本散落在主迴圈中、重複多次的「轉動 + 紀錄影像」邏輯。
# 動作代號（0~5）沿用原始 Habitat action space 的定義，數值本身未更動。
# ==========================================
def step_and_capture(env, action, episode_images, episode_topdowns, display_image=None):
    """
    執行一個動作，並把畫面與 top-down map 記錄到對應的軌跡 list 中。
    display_image: 若提供則記錄這張影像（例如疊加偵測框的 skill_image），
                    否則記錄環境回傳的原始 RGB 觀測。
    """
    obs = env.step(action)
    episode_images.append(display_image if display_image is not None else obs['rgb'])
    episode_topdowns.append(adjust_topdown(env.get_metrics()))
    return obs


def spin_and_collect_panorama(env, episode_images, episode_topdowns, obs, num_steps=11):
    """
    原地連續轉動 num_steps 次以收集全景影像（搭配起始畫面，共可組成 12 張全景視角），
    對應架構圖中 Panoramic Observations 的取得方式。
    """
    for _ in range(num_steps):
        if env.episode_over:
            break
        obs = step_and_capture(env, 3, episode_images, episode_topdowns)
    return obs


def rotate_to_goal_direction(env, episode_images, episode_topdowns, obs, goal_rotate):
    """
    依 Decision-making Module 選出的方向 (goal_rotate, 0~11，每格 30 度) 轉向該方向。
    goal_rotate <= 6 時用動作 3 連續轉動，否則改用反方向的動作 2，取兩者中步數較少者以加快轉向。
    """
    num_steps = min(11 - goal_rotate, goal_rotate + 1)
    for _ in range(num_steps):
        if env.episode_over:
            break
        action = 3 if goal_rotate <= 6 else 2
        obs = step_and_capture(env, action, episode_images, episode_topdowns)
    return obs


def cancel_heading_offset(env, episode_images, episode_topdowns, obs, heading_offset):
    """
    抵銷先前 Pixel Navigation Policy 執行 TurnLeft(4)/TurnRight(5) 動作所累積的朝向偏移，
    確保下一次全景掃描前的朝向與起始時一致。
    """
    for _ in range(abs(heading_offset)):
        if env.episode_over:
            break
        if heading_offset > 0:
            obs = step_and_capture(env, 5, episode_images, episode_topdowns)
            heading_offset -= 1
        else:
            obs = step_and_capture(env, 4, episode_images, episode_topdowns)
            heading_offset += 1
    return obs, heading_offset


def render_episode_videos(episode_dir, episode_images, episode_topdowns, current_goal_name):
    """
    將整段 episode 的觀測影像與 top-down map 寫成三支影片：
    第一人稱畫面 (fps.mp4)、top-down map (metric.mp4)、以及左右並排的對照影片 (result.mp4)。
    """
    fps_writer = imageio.get_writer("%s/fps.mp4" % episode_dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % episode_dir, fps=4)
    result_writer = imageio.get_writer("%s/result.mp4" % episode_dir, fps=4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    text_color = (255, 255, 255)  # 白色字體
    bg_color = (0, 0, 0)          # 黑色底框

    for image, topdown in zip(episode_images, episode_topdowns):
        fps_writer.append_data(image)
        topdown_writer.append_data(topdown)

        h_img, w_img = image.shape[:2]
        h_top, w_top = topdown.shape[:2]

        if h_img != h_top:
            new_w_img = int(w_img * (h_top / h_img))
            image_resized = cv2.resize(image, (new_w_img, h_top))
        else:
            image_resized = image.copy()

        topdown_copy = topdown.copy()

        text_left = f"Observations (Goal: {current_goal_name})"
        text_size_left, _ = cv2.getTextSize(text_left, font, font_scale, thickness)
        text_x_left = (image_resized.shape[1] - text_size_left[0]) // 2
        text_y_left = 60

        cv2.rectangle(image_resized, (text_x_left - 10, text_y_left - text_size_left[1] - 10),
                      (text_x_left + text_size_left[0] + 10, text_y_left + 10), bg_color, -1)
        cv2.putText(image_resized, text_left, (text_x_left, text_y_left), font, font_scale, text_color, thickness)

        text_right = "Top-Down Map"
        text_size_right, _ = cv2.getTextSize(text_right, font, font_scale, thickness)
        text_x_right = (topdown_copy.shape[1] - text_size_right[0]) // 2
        text_y_right = 60

        cv2.rectangle(topdown_copy, (text_x_right - 10, text_y_right - text_size_right[1] - 10),
                      (text_x_right + text_size_right[0] + 10, text_y_right + 10), bg_color, -1)
        cv2.putText(topdown_copy, text_right, (text_x_right, text_y_right), font, font_scale, text_color, thickness)

        combined_frame = np.concatenate((image_resized, topdown_copy), axis=1)
        result_writer.append_data(combined_frame)

    fps_writer.close()
    topdown_writer.close()
    result_writer.close()


# ==========================================
# 消融實驗模式設定
# ==========================================
args = get_args()
ablation_mode = args.ablation
ABLATION_LABELS = {
    "full":            "Full Method",
    "no_memory":       "w/o Semantic Memory Queue",
    "no_reperception": "w/o Re-perception Module",
}
print_and_log("=" * 50)
print_and_log(f"Mode：{ABLATION_LABELS[ablation_mode]}")
print_and_log("=" * 50)

# ==========================================
# 環境與模型初始化
# ==========================================
habitat_config = hm3d_config(stage='val', episodes=args.eval_episodes)
habitat_env = habitat.Env(habitat_config)
detection_model = initialize_dino_model()
segmentation_model = initialize_sam_model()

nav_planner = GPT4V_Planner(
    detection_model, segmentation_model,
    use_memory=(ablation_mode != "no_memory"),
)
nav_executor = Policy_Agent(model_path=POLICY_CHECKPOINT)

print_and_log("Loading Depth Module for Distance Measurement...")
# no_reperception 消融模式下 DepthEstimator 仍會被載入，但主迴圈中不會呼叫，
# 小腦（Pixel Nav Policy）的 Stop 動作將直接被採信為成功，不經過深度驗證。
depth_estimator = DepthEstimator(model_type='vits')
if ablation_mode == "no_reperception":
    print_and_log("⚠️  [消融] Re-perception Module 已停用：小腦 Stop 動作將直接宣告成功，不進行深度驗證。")

evaluation_metrics = []

# ==========================================
# Episodes 選取：可指定單一 episode 重跑（除錯用），否則隨機抽樣
# ==========================================
all_episodes = habitat_env.episodes

if args.episode_id is not None:
    # 除錯模式：只挑出指定的 episode_id（場景檔名只需符合就好，不用打完整路徑）
    sampled_episodes = [
        ep for ep in all_episodes
        if str(ep.episode_id) == str(args.episode_id)
        and (args.scene_id is None or args.scene_id in ep.scene_id)
    ]
    if len(sampled_episodes) == 0:
        raise ValueError(
            f"找不到 episode_id={args.episode_id}"
            + (f", scene_id 包含 '{args.scene_id}'" if args.scene_id else "")
            + " 的回合，請確認 ID / 場景檔名是否正確。"
        )
    if len(sampled_episodes) > 1:
        print_and_log(
            f"⚠️ 找到 {len(sampled_episodes)} 個 episode_id={args.episode_id} 的回合（不同場景），"
            f"全部一起重跑；若只想跑其中一個，請加上 --scene_id 篩選。"
        )
    sample_size = len(sampled_episodes)
    print_and_log(f"🎯 除錯模式：只重跑指定的 {sample_size} 個回合 (episode_id={args.episode_id}) ...")
else:
    sample_size = min(args.eval_episodes, len(all_episodes))

    # 固定隨機種子，確保消融實驗的 100 個 episode 順序完全一致、可重現
    random.seed(42)

    # 隨機抽樣（不做排序，確保每次執行、每回合都是真隨機）
    sampled_episodes = random.sample(all_episodes, sample_size)
    print_and_log(f"📊 已從資料集中隨機抽樣 {sample_size} 個回合準備測試 ...")

# 強制將環境的 episode iterator 替換為自訂的抽樣列表
try:
    habitat_env.episode_iterator = iter(sampled_episodes)
except AttributeError:
    # 兼容舊版 Habitat
    habitat_env._episode_iterator = iter(sampled_episodes)

# ==========================================
# 主評估迴圈
# ==========================================
benchmark_start_time = time.time()
for i in tqdm(range(sample_size)):

    # 自動處理跨場景切換
    obs = habitat_env.reset()

    current_ep = habitat_env.current_episode
    ep_id = current_ep.episode_id
    scene_name = os.path.basename(current_ep.scene_id)
    current_goal_name = current_ep.object_category

    episode_start_time = time.time()
    print_and_log(f"\n▶ [Episode {i+1}/{sample_size}] ID: {ep_id} | 場景: {scene_name} | 目標: {current_goal_name}")

    episode_dir = "./tmp/trajectory_%d" % i
    os.makedirs(episode_dir, exist_ok=False)

    heading_offset = 0

    nav_planner.reset(current_goal_name)
    episode_images = [obs['rgb']]
    episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]

    # 初始規劃：原地旋轉收集全景影像 -> 交給 Decision-making Module 選擇前進方向
    obs = spin_and_collect_panorama(habitat_env, episode_images, episode_topdowns, obs)
    goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan(episode_images[-12:])
    obs = rotate_to_goal_direction(habitat_env, episode_images, episode_topdowns, obs, goal_rotate)

    nav_executor.reset(goal_image, goal_mask)

    # 連續碰撞計數器（卡死偵測用）
    collision_count = 0
    # 這一輪追蹤中，DINO 最後一次「連續確認」看到目標時的距離（初始值設為 99m 代表從未看到過）。
    # 原本用布林 ever_confirmed_target_in_view 會導致：DINO 從 8m 外就確認了目標 → 之後
    # 被沙發遮住、DINO 找不到 TV 了 → front_distance 量到的其實是沙發（0.64m）→ 盲區觸地誤觸發。
    # 改成記錄「最後確認距離」之後，盲區觸地條件從「曾經看到過」收緊為
    # 「上次確認時已經夠近（≤2.5m）」，杜絕沙發/牆壁等中途障礙物被誤當成目標的問題。
    last_confirmed_target_distance = 99.0
    # 時間一致性檢查：記錄上一幀「候選」目標框的中心點與連續命中次數。
    # 單一影格的偵測結果不夠可靠（靠近地板紋理、窗簾摺痕等都可能被誤判成 plant），
    # 真正的目標在相鄰影格之間位置應該是連續、平滑移動的，因此要求同一個大致位置
    # 連續被偵測到兩幀以上，才正式採信為「看到目標」，可以濾掉這種偶發性的單幀誤判。
    pending_target_center = None
    pending_target_streak = 0

    while not habitat_env.episode_over:

        # 1. 取得深度圖（Re-perception Module - Monocular Depth Estimation）
        depth_map = depth_estimator.estimate(obs['rgb'])
        h, w = depth_map.shape

        real_target_distance = 99.0
        target_in_view = False
        x1, y1, x2, y2 = 0, 0, 0, 0

        # 前方備援距離（掃描範圍向下延伸至 h*0.9）
        front_zone = depth_map[int(h * 0.3):int(h * 0.9), int(w * 0.3):int(w * 0.7)]
        front_depths = np.sort(front_zone.flatten())
        # 取最近的 5% 像素作為前方實體的距離
        front_distance = np.mean(front_depths[:max(1, int(len(front_depths) * 0.05))])

        # 2. DINO 精確測距與防幻覺濾網
        if goal_flag:
            rgb_image = cv2.cvtColor(obs['rgb'], cv2.COLOR_BGR2RGB) if obs['rgb'].shape[-1] == 3 else obs['rgb']

            # 依目標類別動態調整偵測敏感度
            current_box_thresh = 0.2
            current_text_thresh = 0.4

            if nav_planner.object_goal == 'tv':
                current_box_thresh = 0.3
                current_text_thresh = 0.25
            elif nav_planner.object_goal == 'chair':
                current_box_thresh = 0.35  # 0.45
                current_text_thresh = 0.4  # 0.4
            elif nav_planner.object_goal == 'toilet':
                current_box_thresh = 0.35
                current_text_thresh = 0.4
            elif nav_planner.object_goal == 'plant':
                # plant 體積通常較小、容易部分入鏡（例如放在層架高處），
                # 預設的 text_threshold=0.4 偏嚴格，這裡放寬到與 make_plan() 一致的 0.25
                current_box_thresh = 0.2
                current_text_thresh = 0.25

            target_bbox = openset_detection(
                rgb_image,
                [nav_planner.object_goal],
                detection_model,
                box_threshold=current_box_thresh,
                text_threshold=current_text_thresh
            )

            if target_bbox.xyxy.shape[0] > 0:
                best_idx = np.argmax(target_bbox.confidence)
                conf = target_bbox.confidence[best_idx]
                x1, y1, x2, y2 = map(int, target_bbox.xyxy[best_idx])

                w_box = x2 - x1
                h_box = y2 - y1
                aspect_ratio = w_box / h_box if h_box > 0 else 0

                is_not_floor_rug = (y1 < h * 0.6)
                # is_widescreen 這個長寬比濾網原本是為了過濾「電視」這種明顯寬扁的物件，
                # 但舊版預設把它套用到所有類別（只有 toilet 給了例外），導致 plant 這種
                # 本來就偏窄高的物件（例如直立在層架上的盆栽）即使被 DINO 正確偵測到，
                # 也常常因為 aspect_ratio 不夠寬而被這個濾網誤刪，造成量不到距離、無法判定成功。
                # 現在改成預設不限制長寬比，只有 tv / toilet 這類形狀較固定的類別才額外加上比例限制。
                is_widescreen = True

                if nav_planner.object_goal == 'tv':
                    is_widescreen = aspect_ratio > 0.55
                elif nav_planner.object_goal == 'toilet':
                    is_widescreen = aspect_ratio > 0.3

                if conf > current_box_thresh and (w_box * h_box) > (w * h * 0.002) and is_not_floor_rug and is_widescreen:
                    box_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    image_diagonal = (w ** 2 + h ** 2) ** 0.5

                    # 與上一幀「候選」目標框的中心點距離夠近，視為同一個物體的連續觀測；
                    # 否則代表這是一個全新出現的偵測框，連續次數重新計算
                    if pending_target_center is not None:
                        center_shift = ((box_center[0] - pending_target_center[0]) ** 2
                                         + (box_center[1] - pending_target_center[1]) ** 2) ** 0.5
                    else:
                        center_shift = None

                    if center_shift is not None and center_shift < image_diagonal * 0.35:
                        pending_target_streak += 1
                    else:
                        pending_target_streak = 1
                    pending_target_center = box_center

                    # 連續兩幀以上偵測到位置相近的目標，才正式採信，避免單幀誤判
                    # （例如貼近地板紋理、窗簾摺痕等）直接觸發距離量測與成功判定
                    if pending_target_streak >= 2:
                        target_depth_region = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if target_depth_region.size > 0:
                            sorted_depths = np.sort(target_depth_region.flatten())
                            real_target_distance = np.mean(sorted_depths[:max(1, int(len(sorted_depths) * 0.1))])
                            target_in_view = True
                            last_confirmed_target_distance = real_target_distance
                else:
                    # 這一幀沒有通過基本濾網（信心值/大小/長寬比等），中斷連續確認的計數
                    pending_target_center = None
                    pending_target_streak = 0
            else:
                # 這一幀完全沒偵測到候選框，同樣中斷連續確認的計數
                pending_target_center = None
                pending_target_streak = 0

        is_collided = habitat_env.sim.previous_step_collided
        action, skill_image = nav_executor.step(obs['rgb'], is_collided)

        if is_collided:
            collision_count += 1
        else:
            collision_count = 0

        # 3. Re-perception：輕量級監督邏輯（Premature Stop Rejected）
        if goal_flag:
            if action == 0:
                if ablation_mode == "no_reperception":
                    # w/o Re-perception：不做任何深度驗證，小腦 Stop 直接視為成功
                    print_and_log("🎯 [消融 w/o Re-perception] 小腦停止，直接宣告成功！")
                    action = 0
                elif target_in_view and real_target_distance <= 2.0:
                    print_and_log(f"🎯 真正抵達 2.0m 內！(估測目標距離 {real_target_distance:.2f}m)，宣告成功！")
                    action = 0
                elif not target_in_view and last_confirmed_target_distance <= 2.5 and front_distance <= 2.0:
                    # 盲區觸地：目標曾在 2.5m 內被確認，之後因太近導致 DINO 視角丟失；
                    # 前方 2m 內還有實體代表機器人已經貼近目標，合理視為成功。
                    # 若上次確認距離 > 2.5m（例如從 8m 外確認了 TV），前方實體可能是中途障礙物，不觸發。
                    print_and_log(f"🎯 盲區觸地得分！目標曾於 {last_confirmed_target_distance:.2f}m 處確認，DINO 丟失，前方有實體 ({front_distance:.2f}m)，宣告成功！")
                    action = 0
                else:
                    dist_msg = f"{real_target_distance:.2f}m" if target_in_view else f"Lost (Front {front_distance:.2f}m)"
                    print_and_log(f"⚠️ 小腦過早放棄 (狀態: {dist_msg})。駁回 Stop，交還大腦重新規劃！")
                    action = 0
                    goal_flag = False
                    collision_count = 0

            elif collision_count > 5:
                print_and_log("🧱 偵測到連續碰撞卡死！放棄當前追蹤，交還大腦重新尋找新路徑！")
                action = 0
                goal_flag = False
                collision_count = 0

        # 在畫面上繪製測距資訊
        if goal_flag and target_in_view:
            cv2.rectangle(skill_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(skill_image, f"{nav_planner.object_goal}: {real_target_distance:.2f}m", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ==========================================
        # 環境互動與 Replanning
        # ==========================================
        if action != 0 or goal_flag:
            if action == 4:
                heading_offset += 1
            elif action == 5:
                heading_offset -= 1
            obs = step_and_capture(habitat_env, action, episode_images, episode_topdowns, display_image=skill_image)
        else:
            if habitat_env.episode_over:
                break

            # 先轉回初始朝向，再重新做一輪全景掃描 + Decision-making
            obs, heading_offset = cancel_heading_offset(habitat_env, episode_images, episode_topdowns, obs, heading_offset)
            obs = spin_and_collect_panorama(habitat_env, episode_images, episode_topdowns, obs)
            goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan(episode_images[-12:])
            obs = rotate_to_goal_direction(habitat_env, episode_images, episode_topdowns, obs, goal_rotate)

            nav_executor.reset(goal_image, goal_mask)
            collision_count = 0
            last_confirmed_target_distance = 99.0
            pending_target_center = None
            pending_target_streak = 0

    # 寫入該 episode 的影片（第一人稱 / top-down map / 並排對照）
    render_episode_videos(episode_dir, episode_images, episode_topdowns, current_goal_name)

    ep_success = habitat_env.get_metrics()['success']
    ep_spl = round(habitat_env.get_metrics()['spl'], 3)
    ep_dist = round(habitat_env.get_metrics()['distance_to_goal'], 3)
    ep_elapsed = round(time.time() - episode_start_time, 1)

    evaluation_metrics.append({
        'episode_id': ep_id,
        'scene_name': scene_name,
        'success': ep_success,
        'spl': ep_spl,
        'distance_to_goal': ep_dist,
        'object_goal': current_goal_name,
        'elapsed_sec': ep_elapsed,
    })
    write_metrics(evaluation_metrics)
    print_and_log(f"✅ Episode {ep_id} 結束。狀態儲存完畢 (SR: {ep_success}, SPL: {ep_spl}, 耗時: {ep_elapsed}s)")

# ==========================================
# 彙整最終指標
# ==========================================
num_episodes = len(evaluation_metrics)
if num_episodes > 0:
    total_success = sum(item['success'] for item in evaluation_metrics)
    total_spl = sum(item['spl'] for item in evaluation_metrics)
    total_dist = sum(item['distance_to_goal'] for item in evaluation_metrics)

    final_sr = round(total_success / num_episodes, 3)
    final_spl = round(total_spl / num_episodes, 3)
    final_dist = round(total_dist / num_episodes, 3)

    print_and_log("\n" + "=" * 50)
    print_and_log("🎉 Benchmark 測試完成！最終評估結果 (Final Metrics):")
    print_and_log(f"Total Episodes: {num_episodes}")
    print_and_log(f"Success Rate (SR): {final_sr}")
    print_and_log(f"SPL: {final_spl}")
    print_and_log(f"Average Distance to Goal: {final_dist}m")

    total_elapsed = round(time.time() - benchmark_start_time, 1)
    avg_elapsed = round(total_elapsed / num_episodes, 1)
    total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
    print_and_log(f"總花費時間: {total_elapsed_str}  (平均每回合 {avg_elapsed}s)")
    print_and_log("=" * 50 + "\n")

    evaluation_metrics.append({
        'episode_id': 'ALL',
        'scene_name': 'FINAL_AVERAGE',
        'success': final_sr,
        'spl': final_spl,
        'distance_to_goal': final_dist,
        'object_goal': 'ALL',
        'elapsed_sec': total_elapsed,
    })
    write_metrics(evaluation_metrics)