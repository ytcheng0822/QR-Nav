import numpy as np
import cv2
import ast
import collections

from llm_utils.gpt_request import gptv_response
from llm_utils.nav_prompt import GPT4o_PROMPT, GPT4o_PROMPT_NO_MEMORY
from cv_utils.detection_tools import *
from cv_utils.segmentation_tools import *

class GPT4o_Planner:
    # Number of recent steps retained in the Semantic Memory Queue.
    MEMORY_QUEUE_SIZE = 8
    # Number of consecutive steps without confirming the target before triggering
    # the URGENT OVERRIDE instruction in the prompt.
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
            # Categories whose visual appearance is too unusual for DINO to detect reliably.
            # For these, the LLM's Flag=True is trusted directly and DINO verification is skipped.
            self.dino_skip_verify = {
                'cabinet', 'chest_of_drawers', 'picture', 'clothes',
                'seating', 'gym_equipment', 'fireplace', 'counter',
            }
            # Richer natural-language prompts to improve DINO recall for hard-to-detect MP3D categories.
            self.dino_prompt_override = {
                'cabinet':          'wooden cabinet . wardrobe . cupboard . dresser',
                'chest_of_drawers': 'chest of drawers . dresser . drawer',
                'gym_equipment':    'gym equipment . treadmill . exercise machine . weight rack',
                'seating':          'seating . bench . ottoman . stool',
                'clothes':          'clothes . clothing . garment . jacket . coat',
            }
            # Large furniture whose detection box extends low in the frame when the robot is nearby.
            # The floor-rug filter (y1 < h*0.6) is disabled for these categories.
            self.large_furniture = {
                'cabinet', 'chest_of_drawers', 'bed', 'sofa',
                'bathtub', 'counter', 'fireplace', 'gym_equipment',
            }
        else:
            self.detect_objects = ['bed', 'sofa', 'chair', 'plant', 'tv', 'toilet', 'floor']
            self.dino_prompt_override = {}
            self.large_furniture = set()
            self.dino_skip_verify = set()

        # When use_memory=False (ablation: w/o Semantic Memory Queue), memory_queue stays empty
        # and the LLM makes decisions based solely on the current panoramic observation.
        self.use_memory = use_memory
        self.memory_queue = collections.deque(maxlen=self.MEMORY_QUEUE_SIZE)
        self.step_count = 0
        # Counts consecutive steps where the LLM did not confirm the target.
        # Used to escalate the deadlock-avoidance instruction in query_gpt4o().
        self.unconfirmed_streak = 0

    def reset(self, object_goal):
        # Align the object name with detect_objects.
        # HM3D uses 'tv' internally while the episode category is 'tv_monitor'; MP3D uses 'tv_monitor' for both.
        if 'tv_monitor' in self.detect_objects:
            self.object_goal = object_goal
        else:
            self.object_goal = 'tv' if object_goal == 'tv_monitor' else object_goal

        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        self.direction_image_trajectory = []
        self.direction_mask_trajectory = []

        self.memory_queue.clear()
        self.step_count = 0
        self.unconfirmed_streak = 0

    def concat_panoramic(self, images, angles):
        """Tile 6 views (odd indices, 60-degree spacing) into a 2x3 grid for LLM input."""
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
        Runs Object Detection + Image Segmentation on the direction chosen by query_gpt4o()
        to produce the pixel-level mask needed by the Motion Control Module.
        """
        direction, goal_flag = self.query_gpt4o(pano_images)
        # Copy to avoid openset_detection / sam_masking from modifying the original pano_images.
        direction_image = np.array(pano_images[direction])

        if goal_flag:
            if self.object_goal in self.dino_skip_verify:
                # Trust the LLM's Flag directly; still run detection to produce a SAM mask.
                # Fall back to 'floor' if no box is found.
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

        # Compute mask centroid and draw a small debug marker.
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
        """Serialize the Semantic Memory Queue into text for the LLM prompt.
        Returns a fixed disabled message when use_memory=False.
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

    def query_gpt4o(self, pano_images):
        """
        Decision-making Module
        Feeds the panoramic image and Semantic Memory to GPT-4o to select the next
        direction (0-11, 30-degree steps) and whether the target is visible (goal_flag).
        """
        self.step_count += 1
        angles = np.arange(len(pano_images)) * 30
        # Copy before colour conversion to avoid in-place BGR2RGB corrupting pano_images,
        # which would cause colour-channel swaps in the recorded video frames.
        inference_image = cv2.cvtColor(
            np.array(self.concat_panoramic(pano_images, angles)), cv2.COLOR_BGR2RGB
        )
        cv2.imwrite("monitor-panoramic.jpg", inference_image)

        history_text = self._build_history_prompt()

        text_content = "<Target Object>:{}\n\n".format(self.object_goal)

        if self.use_memory:
            text_content += f"[Exploration History (Semantic Memory of last {len(self.memory_queue)} steps)]\n{history_text}\n\n"
            text_content += "CRITICAL INSTRUCTION FOR NAVIGATION & DEADLOCK AVOIDANCE:\n"
            text_content += "1. The angles are relative to your CURRENT facing direction, which changes every step. DO NOT rely on angle numbers to avoid explored areas.\n"
            text_content += "2. Instead, visually compare your 6 current views with the descriptions in your [Exploration History].\n"
            text_content += "3. DEADLOCK AVOIDANCE: Any angle showing the same walls, dead-ends, furniture, or room already described in your history counts as explored — even if the exact view looks slightly different. If your history keeps describing the same room without confirming the target, prioritize any visible door, hallway, or opening leading to an area NOT in your history over another guess inside the current room.\n"

            if self.unconfirmed_streak >= self.STUCK_STREAK_THRESHOLD:
                text_content += (
                    f"4. URGENT OVERRIDE: You have made {self.unconfirmed_streak} consecutive decisions WITHOUT "
                    "finding the target. Re-read your [Exploration History] — if it keeps citing the same kind of room feature "
                    "(e.g. repeatedly mentioning 'windowsill', 'window', or similar) without ever confirming the target, you are "
                    "deadlocked in one room. You MUST now choose whichever direction shows the clearest door, hallway, or opening "
                    "leading to an area NOT mentioned in your history, even if it looks less promising than staying in the current "
                    "room. Do NOT select another direction that keeps you in the same room.\n"
                )
            system_prompt = GPT4o_PROMPT
        else:
            system_prompt = GPT4o_PROMPT_NO_MEMORY

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
            # Store only the scene description (Reason), not the angle, since the heading
            # changes every step and angle numbers have no cross-step meaning.
            # Skipped when use_memory=False to keep the queue empty.
            if self.use_memory:
                self.memory_queue.append({"step": self.step_count, "scene_feature": answer['Reason']})
            self.unconfirmed_streak = 0 if answer['Flag'] else self.unconfirmed_streak + 1
            return (int(answer['Angle']) // 30) % 12, answer['Flag']
        except Exception:
            self.unconfirmed_streak += 1
            return np.random.randint(0, 12), False