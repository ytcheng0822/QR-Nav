"""
Decision-making Module + Perception & Localization Module
===========================================================
對應論文架構圖：
  - LLM Reasoning + Semantic Memory Queue  -> GPT4V_Planner.query_gpt4v()
  - Object Detection + Image Segmentation  -> GPT4V_Planner.make_plan()
"""

import numpy as np
import cv2
import ast
import collections

from llm_utils.gpt_request import gptv_response
from llm_utils.nav_prompt import GPT4V_PROMPT, GPT4V_PROMPT_NO_MEMORY
from cv_utils.detection_tools import *
from cv_utils.segmentation_tools import *


class GPT4V_Planner:
    # Semantic Memory Queue 保留的最近步數（縮短至 8 步以降低 token 消耗，
    # 仍足以涵蓋最近一輪探索，供 LLM 判斷是否重複探索同一區域）
    MEMORY_QUEUE_SIZE = 8
    # 連續多少步都還沒確認看到目標時，視為「房間級 deadlock」，
    # 在 prompt 中插入升級版的強制指令，要求優先選擇通往新區域的門/走廊
    # （與 nav_prompt.py 系統提示中既有的「連續 6 步」說法對齊，避免兩處給 LLM 不同的判斷基準）
    STUCK_STREAK_THRESHOLD = 6

    def __init__(self, dino_model, sam_model, use_memory=True, dataset='hm3d'):
        self.gptv_trajectory = []
        self.dino_model = dino_model
        self.sam_model = sam_model
        self.dataset = dataset

        if dataset == 'mp3d':
            self.detect_objects = [
                'chair', 'table', 'picture', 'cabinet', 'cushion',
                'sofa', 'bed', 'chest_of_drawers', 'plant', 'sink',
                'toilet', 'stool', 'towel', 'tv_monitor', 'shower',
                'bathtub', 'counter', 'fireplace', 'gym_equipment',
                'seating', 'clothes',
            ]
            # MP3D 部分類別視覺特徵太特殊（例如 cabinet 是古典木雕衣櫃、
            # chest_of_drawers 外觀接近 dresser），DINO 即使閾值調到最低也無法穩定偵測，
            # 對這些類別直接信任 LLM 的 Flag 判斷，跳過 make_plan 的 DINO 複驗。
            self.dino_skip_verify = {
                'cabinet', 'chest_of_drawers', 'picture', 'clothes',
                'seating', 'gym_equipment', 'fireplace', 'counter',
            }
            self.dino_prompt_override = {
                'cabinet':          'wooden cabinet . wardrobe . cupboard . dresser',
                'chest_of_drawers': 'chest of drawers . dresser . drawer',
                'gym_equipment':    'gym equipment . treadmill . exercise machine . weight rack',
                'seating':          'seating . bench . ottoman . stool',
                'clothes':          'clothes . clothing . garment . jacket . coat',
            }
            # MP3D 中這些大型家具靠近時偵測框會延伸到畫面很低的位置，
            # 不應套用「y1 < h*0.6」的地毯過濾條件。
            self.large_furniture = {
                'cabinet', 'chest_of_drawers', 'bed', 'sofa',
                'bathtub', 'counter', 'fireplace', 'gym_equipment',
            }
        else:
            # HM3D 資料集支援的目標類別清單
            self.detect_objects = ['bed', 'sofa', 'chair', 'plant', 'tv', 'toilet', 'floor']
            self.dino_prompt_override = {}
            self.large_furniture = set()
            self.dino_skip_verify = set()

        # use_memory=False 時對應消融實驗 w/o Semantic Memory Queue，
        # 此時 memory_queue 固定為空，LLM 每步只看當下全景圖決策，無法參考歷史
        self.use_memory = use_memory
        self.memory_queue = collections.deque(maxlen=self.MEMORY_QUEUE_SIZE)
        self.step_count = 0
        # 連續幾步「LLM 都判斷尚未看到目標」的計數器，用來偵測同一房間內打轉的 deadlock，
        # 並在 query_gpt4v() 的 prompt 中動態升級警告強度（見下方 STUCK_STREAK_THRESHOLD）
        self.unconfirmed_streak = 0

    def reset(self, object_goal):
        # detect_objects 裡用的名稱需與 Habitat episode 的 object_category 對齊。
        # HM3D 目標名稱是 'tv_monitor'，但 detect_objects 裡用 'tv'，需要轉換。
        # MP3D 目標名稱本來就是 'tv_monitor'，且 detect_objects 也用 'tv_monitor'，不需轉換。
        # 判斷方式：看 detect_objects 裡是否有 'tv_monitor'（MP3D）或 'tv'（HM3D）。
        if 'tv_monitor' in self.detect_objects:
            # MP3D 模式：不做任何轉換
            self.object_goal = object_goal
        else:
            # HM3D 模式：tv_monitor → tv
            self.object_goal = 'tv' if object_goal == 'tv_monitor' else object_goal

        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        self.direction_image_trajectory = []
        self.direction_mask_trajectory = []

        # 每個 episode 開始時清空 Semantic Memory Queue 與步數計數器
        self.memory_queue.clear()
        self.step_count = 0
        self.unconfirmed_streak = 0

    def concat_panoramic(self, images, angles):
        """將 6 個方向（取奇數索引，間隔 60 度）的全景影像拼接成一張 2x3 方格圖，供 LLM 一次輸入。"""
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except Exception:
            height, width = 480, 640

        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)

        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(
                copy_images[i], "Angle %d" % angles[i], (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA
            )
            row = i // 6
            col = (i // 2) % 3
            background_image[
                10 * (row + 1) + row * height: 10 * (row + 1) + row * height + height,
                10 * (col + 1) + col * width: 10 * (col + 1) + col * width + width,
                :,
            ] = copy_images[i]
        return background_image

    def make_plan(self, pano_images):
        """
        Perception & Localization Module
        依 query_gpt4v() 選出的方向影像執行 Object Detection + Image Segmentation，
        產生 Motion Control Module 所需的目標遮罩 (mask)。
        """
        direction, goal_flag = self.query_gpt4v(pano_images)
        # copy 一份，避免後續的 openset_detection / sam_masking 意外修改 pano_images 的原始資料
        direction_image = np.array(pano_images[direction])

        if goal_flag:
            if self.object_goal in self.dino_skip_verify:
                # DINO 對這類類別本來就偵測不到，直接信任 LLM 的 Flag，
                # 仍然跑一次偵測是為了給 SAM 產生遮罩；偵測不到時備援改用 floor
                detect_prompt = self.dino_prompt_override.get(self.object_goal, self.object_goal)
                bbox = openset_detection(
                    direction_image, [detect_prompt], self.dino_model,
                    box_threshold=0.1, text_threshold=0.1
                )
                if bbox.xyxy.shape[0] == 0:
                    bbox = openset_detection(
                        direction_image, ['floor'], self.dino_model,
                        box_threshold=0.2, text_threshold=0.4
                    )
            else:
                detect_prompt = self.dino_prompt_override.get(self.object_goal, self.object_goal)
                bbox = openset_detection(
                    direction_image, [detect_prompt], self.dino_model,
                    box_threshold=0.2, text_threshold=0.25
                )
                if bbox.xyxy.shape[0] == 0:
                    goal_flag = False

        if not goal_flag:
            bbox = openset_detection(
                direction_image, ['door', 'floor'], self.dino_model,
                box_threshold=0.2, text_threshold=0.4
            )

        try:
            mask = sam_masking(direction_image, bbox.xyxy, self.sam_model)
        except Exception:
            mask = np.ones_like(direction_image).mean(axis=-1)

        self.direction_image_trajectory.append(direction_image)
        self.direction_mask_trajectory.append(mask)

        # 取遮罩重心作為視覺化用的 debug 標記（畫出方框）
        debug_image = np.array(direction_image)
        debug_mask = np.zeros_like(debug_image)
        pixel_y, pixel_x = np.where(mask > 0)[0:2]
        pixel_y = int(pixel_y.mean())
        pixel_x = int(pixel_x.mean())
        debug_image = cv2.rectangle(debug_image, (pixel_x - 8, pixel_y - 8), (pixel_x + 8, pixel_y + 8), (255, 0, 0), -1)
        debug_mask = cv2.rectangle(debug_mask, (pixel_x - 8, pixel_y - 8), (pixel_x + 8, pixel_y + 8), (255, 255, 255), -1)
        debug_mask = debug_mask.mean(axis=-1)

        return direction_image, debug_mask, debug_image, direction, goal_flag

    def _build_history_prompt(self):
        """將 Semantic Memory Queue 轉成文字，提供給 LLM 作為避免重複探索的依據。
        w/o Semantic Memory Queue 模式下（use_memory=False）固定回傳空字串，
        讓 LLM 每步都從零開始決策，沒有任何歷史資訊可以參考。
        """
        if not self.use_memory:
            return "Memory disabled (ablation: w/o Semantic Memory Queue)."
        if len(self.memory_queue) == 0:
            return "No history yet. This is the first step."
        history_lines = [
            f"- Step {mem['step']}: Tried to move towards an area described as '{mem['scene_feature']}'"
            for mem in self.memory_queue
        ]
        return "\n".join(history_lines)

    def query_gpt4v(self, pano_images):
        """
        Decision-making Module
        將全景影像 + Semantic Memory 餵給 GPT-4o，選出下一步要前進的方向
        (0~11，每格 30 度) 以及是否已經看到目標物件 (goal_flag)。
        """
        self.step_count += 1
        angles = np.arange(len(pano_images)) * 30
        # 先 copy 再轉色彩，避免 BGR2RGB 的 in-place 操作汙染 pano_images 原始資料，
        # 導致後續寫入影片的影像色頻對調（例如木頭橘色變藍色）
        inference_image = cv2.cvtColor(
            np.array(self.concat_panoramic(pano_images, angles)), cv2.COLOR_BGR2RGB
        )
        cv2.imwrite("monitor-panoramic.jpg", inference_image)

        history_text = self._build_history_prompt()

        text_content = "<Target Object>:{}\n\n".format(self.object_goal)

        if self.use_memory:
            # 完整方法：提供歷史記憶 + 所有 deadlock-avoidance 指令
            text_content += f"[Exploration History (Semantic Memory of last {len(self.memory_queue)} steps)]\n{history_text}\n\n"
            text_content += "CRITICAL INSTRUCTION FOR NAVIGATION & DEADLOCK AVOIDANCE:\n"
            text_content += "1. The angles are relative to your CURRENT facing direction, which changes every step. DO NOT rely on angle numbers to avoid explored areas.\n"
            text_content += "2. Instead, visually compare your 6 current views with the descriptions in your [Exploration History].\n"
            text_content += "3. DEADLOCK AVOIDANCE: Any angle showing the same walls, dead-ends, furniture, or room already described in your history counts as explored — even if the exact view looks slightly different. If your history keeps describing the same room without confirming the target, prioritize any visible door, hallway, or opening leading to an area NOT in your history over another guess inside the current room.\n"

            if self.unconfirmed_streak >= self.STUCK_STREAK_THRESHOLD:
                text_content += (
                    f"4. ⚠️ URGENT OVERRIDE: You have made {self.unconfirmed_streak} consecutive decisions WITHOUT "
                    "finding the target. Re-read your [Exploration History] — if it keeps citing the same kind of room feature "
                    "(e.g. repeatedly mentioning 'windowsill', 'window', or similar) without ever confirming the target, you are "
                    "deadlocked in one room. You MUST now choose whichever direction shows the clearest door, hallway, or opening "
                    "leading to an area NOT mentioned in your history, even if it looks less promising than staying in the current "
                    "room. Do NOT select another direction that keeps you in the same room.\n"
                )
            system_prompt = GPT4V_PROMPT
        else:
            # w/o Semantic Memory Queue：不提供歷史，也不加入任何依賴歷史的指令
            # 只告訴 LLM 目標物件，讓它純粹依靠當下視覺判斷
            system_prompt = GPT4V_PROMPT_NO_MEMORY

        self.gptv_trajectory.append("\nInput:\n%s \n" % text_content)
        self.panoramic_trajectory.append(inference_image)

        answer = None
        raw_answer = ""
        for _ in range(10):
            try:
                raw_answer = gptv_response(text_content, inference_image, system_prompt)
                print("GPT-4o Output Response: %s" % raw_answer)
                answer_str = raw_answer[raw_answer.index("{"): raw_answer.index("}") + 1]
                answer = ast.literal_eval(answer_str)
                if 'Reason' in answer.keys() and 'Angle' in answer.keys():
                    assert int(answer['Angle']) in angles
                    break
            except Exception:
                continue

        self.gptv_trajectory.append("GPT-4o Answer:\n%s" % raw_answer)
        self.panoramic_trajectory.append(inference_image)

        try:
            # 寫入 Semantic Memory Queue：只存場景特徵（Reason），不存相對角度，
            # 因為機器人每一步的朝向都會改變，角度沒有跨步參考的意義。
            # use_memory=False（w/o Semantic Memory Queue 消融）時跳過，確保記憶始終為空。
            if self.use_memory:
                self.memory_queue.append({"step": self.step_count, "scene_feature": answer['Reason']})
            # 更新「連續未確認目標」的計數器：確認看到目標就歸零，否則累加，
            # 供下一次呼叫時動態升級 prompt 中的房間級 deadlock 警告
            self.unconfirmed_streak = 0 if answer['Flag'] else self.unconfirmed_streak + 1
            return (int(answer['Angle']) // 30) % 12, answer['Flag']
        except Exception:
            # GPT-4o 解析失敗時的保底策略：隨機選一個方向，並視為尚未看到目標
            self.unconfirmed_streak += 1
            return np.random.randint(0, 12), False