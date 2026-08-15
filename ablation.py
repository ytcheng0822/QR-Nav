import habitat
import os
import warnings
import argparse
import csv
import cv2
import imageio
import numpy as np
import random
import time

from cv_utils.detection_tools import *
from cv_utils.segmentation_tools import *
from tqdm import tqdm
from constants import *
from config_utils import hm3d_config
from gpt4o_planner import GPT4o_Planner
from policy_agent import Policy_Agent
from depth_estimator import DepthEstimator
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

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
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--episode_id", type=str, default=None,
                        help="Run only a specific episode_id, e.g. --episode_id 9")
    parser.add_argument("--scene_id", type=str, default=None,
                        help="Used with --episode_id to disambiguate across scenes")
    # Ablation mode (choose one):
    #   full             Full method (default)
    #   no_memory        w/o Semantic Memory Queue
    #   no_reperception  w/o Re-perception Module
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_memory", "no_reperception"],
        help=(
            "Ablation mode:\n"
            "  full             Full method (default)\n"
            "  no_memory        w/o Semantic Memory Queue\n"
            "  no_reperception  w/o Re-perception Module"
        ),
    )
    return parser.parse_args()


# Action codes (0-5) follow the original Habitat action space definition.
def step_and_capture(env, action, episode_images, episode_topdowns, display_image=None):
    """Execute one action and record the frame and top-down map."""
    obs = env.step(action)
    episode_images.append(display_image if display_image is not None else obs['rgb'])
    episode_topdowns.append(adjust_topdown(env.get_metrics()))
    return obs


def spin_and_collect_panorama(env, episode_images, episode_topdowns, obs, num_steps=11):
    """Rotate in place for num_steps steps to collect 12 panoramic views."""
    for _ in range(num_steps):
        if env.episode_over:
            break
        obs = step_and_capture(env, 3, episode_images, episode_topdowns)
    return obs


def rotate_to_goal_direction(env, episode_images, episode_topdowns, obs, goal_rotate):
    """Rotate toward the direction selected by the Decision-making Module.

    goal_rotate is in [0, 11] (30-degree steps). Chooses the shorter rotation
    direction between CW (action 3) and CCW (action 2).
    """
    num_steps = min(11 - goal_rotate, goal_rotate + 1)
    for _ in range(num_steps):
        if env.episode_over:
            break
        action = 3 if goal_rotate <= 6 else 2
        obs = step_and_capture(env, action, episode_images, episode_topdowns)
    return obs


def cancel_heading_offset(env, episode_images, episode_topdowns, obs, heading_offset):
    """Undo accumulated heading drift from TurnLeft(4)/TurnRight(5) actions
    so the next panoramic scan starts from a consistent heading.
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
    """Save the episode trajectory as three videos: fps.mp4, metric.mp4, result.mp4."""
    fps_writer = imageio.get_writer("%s/fps.mp4" % episode_dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % episode_dir, fps=4)
    result_writer = imageio.get_writer("%s/result.mp4" % episode_dir, fps=4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

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


# ==================================================
# Ablation Mode Setup
# ==================================================
args = get_args()
ablation_mode = args.ablation
ABLATION_LABELS = {
    "full":            "Full Method",
    "no_memory":       "w/o Semantic Memory Queue",
    "no_reperception": "w/o Re-perception Module",
}
print_and_log("=" * 50)
print_and_log(f"Mode: {ABLATION_LABELS[ablation_mode]}")
print_and_log("=" * 50)

# ==================================================
# Initialization
# ==================================================
habitat_config = hm3d_config(stage='val', episodes=args.eval_episodes)
habitat_env = habitat.Env(habitat_config)
detection_model = initialize_dino_model()
segmentation_model = initialize_sam_model()

nav_planner = GPT4o_Planner(
    detection_model, segmentation_model,
    use_memory=(ablation_mode != "no_memory"),
)
nav_executor = Policy_Agent(model_path=POLICY_CHECKPOINT)

print_and_log("Loading Depth Module for Distance Measurement...")
# DepthEstimator is always loaded; under no_reperception it is not called —
# the policy's Stop action is accepted directly without depth validation.
depth_estimator = DepthEstimator(model_type='vits')
if ablation_mode == "no_reperception":
    print_and_log("[Ablation] Re-perception Module disabled: Stop action accepted directly.")

evaluation_metrics = []

# ==================================================
# Episode Selection
# ==================================================
all_episodes = habitat_env.episodes

if args.episode_id is not None:
    # Debug mode: run a single specified episode.
    sampled_episodes = [
        ep for ep in all_episodes
        if str(ep.episode_id) == str(args.episode_id)
        and (args.scene_id is None or args.scene_id in ep.scene_id)
    ]
    if len(sampled_episodes) == 0:
        raise ValueError(
            f"Episode id={args.episode_id}"
            + (f", scene_id containing '{args.scene_id}'" if args.scene_id else "")
            + " not found. Check the ID and scene filename."
        )
    if len(sampled_episodes) > 1:
        print_and_log(
            f"Found {len(sampled_episodes)} episodes with id={args.episode_id} across different scenes. "
            f"Running all of them; add --scene_id to select one."
        )
    sample_size = len(sampled_episodes)
    print_and_log(f"Debug mode: running {sample_size} episode(s) with episode_id={args.episode_id}")
else:
    sample_size = min(args.eval_episodes, len(all_episodes))
    # Fixed seed for reproducible ablation comparisons.
    random.seed(42)
    sampled_episodes = random.sample(all_episodes, sample_size)
    print_and_log(f"Sampled {sample_size} episodes (seed=42)")

# Override the environment's episode iterator with the selected list.
try:
    habitat_env.episode_iterator = iter(sampled_episodes)
except AttributeError:
    habitat_env._episode_iterator = iter(sampled_episodes)

# ==================================================
# Main Evaluation Loop
# ==================================================
benchmark_start_time = time.time()
for i in tqdm(range(sample_size)):

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
    episode_images  = [obs['rgb']]
    episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]

    # Initial planning: spin to collect panorama, select goal direction, rotate toward it.
    obs = spin_and_collect_panorama(habitat_env, episode_images, episode_topdowns, obs)
    goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan(episode_images[-12:])
    obs = rotate_to_goal_direction(habitat_env, episode_images, episode_topdowns, obs, goal_rotate)

    nav_executor.reset(goal_image, goal_mask)

    collision_count = 0
    # Tracks the closest distance at which DINO confirmed the target this round.
    # Using a distance value (rather than a boolean) prevents false blind-spot stops
    # when the last confirmed sighting was far away (e.g. TV at 8 m blocked by a sofa).
    last_confirmed_target_distance = 99.0
    # Temporal consistency check: require the same detection box to appear in at least
    # two consecutive frames before accepting it as a valid target sighting.
    pending_target_center = None
    pending_target_streak = 0

    while not habitat_env.episode_over:

        # Step 1: Monocular depth estimation (Re-perception Module).
        depth_map = depth_estimator.estimate(obs['rgb'])
        h, w = depth_map.shape

        real_target_distance = 99.0
        target_in_view = False
        x1, y1, x2, y2 = 0, 0, 0, 0

        # Estimate the distance to the nearest obstacle in the forward-facing region.
        front_zone   = depth_map[int(h * 0.3):int(h * 0.9), int(w * 0.3):int(w * 0.7)]
        front_depths = np.sort(front_zone.flatten())
        front_distance = np.mean(front_depths[:max(1, int(len(front_depths) * 0.05))])

        # Step 2: DINO target detection and distance measurement.
        if goal_flag:
            rgb_image = cv2.cvtColor(obs['rgb'], cv2.COLOR_BGR2RGB) if obs['rgb'].shape[-1] == 3 else obs['rgb']

            current_box_thresh  = 0.2
            current_text_thresh = 0.4

            if nav_planner.object_goal == 'tv':
                current_box_thresh  = 0.3
                current_text_thresh = 0.25
            elif nav_planner.object_goal == 'chair':
                current_box_thresh  = 0.35
                current_text_thresh = 0.4
            elif nav_planner.object_goal == 'toilet':
                current_box_thresh  = 0.35
                current_text_thresh = 0.4
            elif nav_planner.object_goal == 'plant':
                # plant is small and often partially in frame; relax threshold to match make_plan().
                current_box_thresh  = 0.2
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
                # Aspect-ratio filter: only applied to categories with predictable shapes.
                # Disabled by default to avoid incorrectly rejecting tall/narrow objects like plants.
                is_widescreen = True
                if nav_planner.object_goal == 'tv':
                    is_widescreen = aspect_ratio > 0.55
                elif nav_planner.object_goal == 'toilet':
                    is_widescreen = aspect_ratio > 0.3

                if conf > current_box_thresh and (w_box * h_box) > (w * h * 0.002) and is_not_floor_rug and is_widescreen:
                    box_center     = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    image_diagonal = (w ** 2 + h ** 2) ** 0.5

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

                    # Accept the detection only after two consecutive consistent frames.
                    if pending_target_streak >= 2:
                        target_depth_region = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if target_depth_region.size > 0:
                            sorted_depths = np.sort(target_depth_region.flatten())
                            real_target_distance = np.mean(sorted_depths[:max(1, int(len(sorted_depths) * 0.1))])
                            target_in_view = True
                            last_confirmed_target_distance = real_target_distance
                else:
                    pending_target_center = None
                    pending_target_streak = 0
            else:
                pending_target_center = None
                pending_target_streak = 0

        is_collided = habitat_env.sim.previous_step_collided
        action, skill_image = nav_executor.step(obs['rgb'], is_collided)

        if is_collided:
            collision_count += 1
        else:
            collision_count = 0

        # Step 3: Re-perception stop validation (Premature Stop Rejected).
        if goal_flag:
            if action == 0:
                if ablation_mode == "no_reperception":
                    # w/o Re-perception: accept Stop directly without depth validation.
                    print_and_log("[消融 w/o Re-perception] 小腦停止，直接宣告成功！")
                    action = 0
                elif target_in_view and real_target_distance <= 2.0:
                    print_and_log(f"真正抵達 2.0m 內！(估測目標距離 {real_target_distance:.2f}m)，宣告成功！")
                    action = 0
                elif not target_in_view and last_confirmed_target_distance <= 2.5 and front_distance <= 2.0:
                    # Blind-spot stop: target was recently confirmed close but is now out of view.
                    # A solid object ahead indicates the robot has reached the target.
                    print_and_log(f"盲區觸地得分！目標曾於 {last_confirmed_target_distance:.2f}m 處確認，DINO 丟失，前方有實體 ({front_distance:.2f}m)，宣告成功！")
                    action = 0
                else:
                    dist_msg = f"{real_target_distance:.2f}m" if target_in_view else f"Lost (Front {front_distance:.2f}m)"
                    print_and_log(f"小腦過早放棄 (狀態: {dist_msg})。駁回 Stop，交還大腦重新規劃！")
                    action = 0
                    goal_flag = False
                    collision_count = 0

            elif collision_count > 5:
                print_and_log("偵測到連續碰撞卡死！放棄當前追蹤，交還大腦重新尋找新路徑！")
                action = 0
                goal_flag = False
                collision_count = 0

        # Draw detection overlay.
        if goal_flag and target_in_view:
            cv2.rectangle(skill_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(skill_image, f"{nav_planner.object_goal}: {real_target_distance:.2f}m",
                        (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Environment step and replanning.
        if action != 0 or goal_flag:
            if action == 4:
                heading_offset += 1
            elif action == 5:
                heading_offset -= 1
            obs = step_and_capture(habitat_env, action, episode_images, episode_topdowns, display_image=skill_image)
        else:
            if habitat_env.episode_over:
                break

            # Restore heading, collect new panorama, replan.
            obs, heading_offset = cancel_heading_offset(habitat_env, episode_images, episode_topdowns, obs, heading_offset)
            obs = spin_and_collect_panorama(habitat_env, episode_images, episode_topdowns, obs)
            goal_image, goal_mask, _, goal_rotate, goal_flag = nav_planner.make_plan(episode_images[-12:])
            obs = rotate_to_goal_direction(habitat_env, episode_images, episode_topdowns, obs, goal_rotate)

            nav_executor.reset(goal_image, goal_mask)
            collision_count = 0
            last_confirmed_target_distance = 99.0
            pending_target_center = None
            pending_target_streak = 0

    render_episode_videos(episode_dir, episode_images, episode_topdowns, current_goal_name)

    ep_success = habitat_env.get_metrics()['success']
    ep_spl     = round(habitat_env.get_metrics()['spl'], 3)
    ep_dist    = round(habitat_env.get_metrics()['distance_to_goal'], 3)
    ep_elapsed = round(time.time() - episode_start_time, 1)

    evaluation_metrics.append({
        'ablation':        ablation_mode,
        'episode_id':      ep_id,
        'scene_name':      scene_name,
        'success':         ep_success,
        'spl':             ep_spl,
        'distance_to_goal': ep_dist,
        'object_goal':     current_goal_name,
        'elapsed_sec':     ep_elapsed,
    })
    write_metrics(evaluation_metrics)
    print_and_log(f"Episode {ep_id} 結束。狀態儲存完畢 (SR: {ep_success}, SPL: {ep_spl}, 耗時: {ep_elapsed}s)")

# ==================================================
# Final Metrics
# ==================================================
num_episodes = len(evaluation_metrics)
if num_episodes > 0:
    total_success = sum(item['success'] for item in evaluation_metrics)
    total_spl     = sum(item['spl'] for item in evaluation_metrics)
    total_dist    = sum(item['distance_to_goal'] for item in evaluation_metrics)

    final_sr   = round(total_success / num_episodes, 3)
    final_spl  = round(total_spl / num_episodes, 3)
    final_dist = round(total_dist / num_episodes, 3)

    total_elapsed     = round(time.time() - benchmark_start_time, 1)
    avg_elapsed       = round(total_elapsed / num_episodes, 1)
    total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))

    print_and_log("\n" + "=" * 50)
    print_and_log("Benchmark 測試完成！最終評估結果 (Final Metrics):")
    print_and_log(f"Total Episodes: {num_episodes}")
    print_and_log(f"Success Rate (SR): {final_sr}")
    print_and_log(f"SPL: {final_spl}")
    print_and_log(f"Average Distance to Goal: {final_dist}m")
    print_and_log(f"總花費時間: {total_elapsed_str}  (平均每回合 {avg_elapsed}s)")
    print_and_log("=" * 50 + "\n")

    evaluation_metrics.append({
        'ablation':        ablation_mode,
        'episode_id':      'ALL',
        'scene_name':      'FINAL_AVERAGE',
        'success':         final_sr,
        'spl':             final_spl,
        'distance_to_goal': final_dist,
        'object_goal':     'ALL',
        'elapsed_sec':     total_elapsed,
    })
    write_metrics(evaluation_metrics)