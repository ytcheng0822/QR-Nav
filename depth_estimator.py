import torch
from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}

DEFAULT_CHECKPOINT = "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"

class DepthEstimator:
    def __init__(self, model_type='vits', checkpoint_path=DEFAULT_CHECKPOINT, device='cuda'):
        self.device = device
        print("Loading Depth Anything V2 (Metric) model...")
        self.model = DepthAnythingV2(**MODEL_CONFIGS[model_type])
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def estimate(self, rgb_image):
        return self.model.infer_image(rgb_image)