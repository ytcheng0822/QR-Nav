[README.md](https://github.com/user-attachments/files/31102156/README.md)
# QR-Nav: LLM-Based Mapless Zero-Shot Object Goal Navigation

<img width="900" alt="system architecture" src="./assets/qr-nav.png">

---

## System Overview

The system consists of four modules:

| Module | Role |
| :--- | :--- |
| **Decision-making Module** | GPT-4o reasons over panoramic observations and a Semantic Memory Queue to select the best exploration direction |
| **Perception & Localization Module** | GroundingDINO detects the target object; SAM produces a pixel-level mask as the navigation goal |
| **Motion Control Module** | PixelNav policy executes low-level navigation toward the pixel goal |
| **Re-perception Module** | Depth Anything V2 validates whether the robot has truly reached the target, rejecting premature stops |

---

## Project Structure

```text
QR-Nav
├── checkpoints/              # Pretrained model weights
│   ├── depth_anything_v2_metric_hypersim_vits.pth
│   ├── GroundingDINO_SwinB_cfg.py
│   ├── groundingdino_swinb_cogcoor.pth
│   ├── pixelnav_A.ckpt
│   ├── pixelnav_C.ckpt
│   └── sam_vit_h_4b8939.pth
├── cv_utils/
│   ├── detection_tools.py
│   └── segmentation_tools.py
├── data/                     # Scene and episode datasets (HM3D / MP3D)
├── data_utils/
│   └── geometry_tools.py
├── depth_anything_v2/
├── habitat-lab/
├── habitat-sim/
├── llm_utils/                # LLM prompting and API request interfaces
│   ├── gpt_request.py
│   └── nav_prompt.py
├── thirdparty/               # Third-party repositories
│   ├── Depth-Anything-V2
│   ├── GroundingDINO
│   └── segment-anything
├── ablation.py               # Ablation study execution script
├── config_utils.py           # Configuration utilities
├── constants.py              # Global constants and path definitions
├── depth_estimator.py        # Depth estimation wrapper
├── evaluate_policy.py        # Policy evaluation script
├── gpt4o_planner.py          # GPT-4o high-level decision planner
├── policy_agent.py           # Navigation agent implementation
├── policy_network.py         # Low-level control network architecture
└── qr-nav.py                 # Main execution entry point for ObjectNav benchmark
