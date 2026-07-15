"""
Re-perception Module - Monocular Depth Estimation
===================================================
對應論文架構圖中「Re-perception Module」裡的 Monocular Depth Estimation 區塊。

此模組僅負責一件事：把 RGB 影像轉換成公制深度圖 (metric depth map)。
原本 SafetyDecisionModule 中「依距離自動閃避障礙物 (Dynamic Evasion)」的功能
在目前版本中並未被啟用（呼叫端永遠忽略該回傳值），因此已整併移除，
真正的「Re-perception / Premature Stop Rejected」判斷邏輯改放在
objnav_benchmark.py 的主迴圈中，集中以深度圖 + 物件偵測結果一起判斷。
"""

import torch
from depth_anything_v2.dpt import DepthAnythingV2

# Depth Anything V2 各 encoder 的模型結構設定
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}

DEFAULT_CHECKPOINT = "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"


class DepthEstimator:
    """Depth Anything V2 (Metric) 的輕量包裝，提供單目深度估計。"""

    def __init__(self, model_type='vits', checkpoint_path=DEFAULT_CHECKPOINT, device='cuda'):
        self.device = device

        print("Loading Depth Anything V2 (Metric) model...")
        self.model = DepthAnythingV2(**MODEL_CONFIGS[model_type])
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def estimate(self, rgb_image):
        """
        輸入一張 RGB 影像，回傳對應的公制深度圖 (單位：公尺)，shape 與輸入影像的 H x W 相同。
        """
        return self.model.infer_image(rgb_image)