# Scripts/NTU_RGBD/19_Generate_MS_Spiking_ResNet50_Membrane_Heatmap_Video.py

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.common.paths import (
    NTU_METADATA_DIR,
    NTU_RGBD_DATASET_DIR,
    NTU_RGBD_OUTPUT_DIR,
)
from Scripts.common.reproducibility import get_device
from Scripts.NTU_RGBD.datasets import build_eval_transform
from Scripts.NTU_RGBD.inference.video_generation import (
    VideoGenerationConfig,
    run_video_generation,
)
from Scripts.NTU_RGBD.models import (
    build_ms_spiking_resnet50_membrane_heatmap,
)


# ============================================================
# 1. Model-specific configuration
# ============================================================

MODEL_VERSION = "18"

NUM_STEPS = 2
BETA = 0.90
THRESHOLD = 1.0
SURROGATE_SLOPE = 25.0

DEVICE_NAME = None

MODEL_PATH = (
    NTU_RGBD_OUTPUT_DIR
    / "18_Train_MS_Spiking_ResNet50_Membrane_Heatmap_Human_Detection"
    / "best_model.pt"
)

CONFIG = VideoGenerationConfig(
    model_version=MODEL_VERSION,
    test_csv=NTU_METADATA_DIR / "test_split.csv",
    rgb_video_dir=NTU_RGBD_DATASET_DIR / "rgb_videos",
    skeleton_dir=NTU_RGBD_DATASET_DIR / "skeletons",
    output_dir=(
        NTU_RGBD_OUTPUT_DIR
        / f"19_Generate_MP4_MS_Spiking_Membrane_Model_{MODEL_VERSION}"
    ),

    image_size=224,
    heatmap_size=56,
    num_joints=25,

    person_crop=True,
    bbox_expansion=0.25,
    confidence_threshold=0.02,

    frame_stride=1,
    output_fps=None,
    skip_existing_videos=True,
    save_prediction_npz=True,
    max_test_videos=None,

    readout_type="mean_membrane",
    prediction_label=(
        f"Prediction - MS-SNN Membrane Model {MODEL_VERSION}"
    ),

    num_steps=NUM_STEPS,
    beta=BETA,
    threshold=THRESHOLD,
    surrogate_slope=SURROGATE_SLOPE,
)


# ============================================================
# 2. Model loading
# ============================================================

def extract_state_dict(
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find a model state dictionary inside the checkpoint."
    )


def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def load_model(device: torch.device) -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = build_ms_spiking_resnet50_membrane_heatmap(
        num_joints=CONFIG.num_joints,
        num_steps=NUM_STEPS,
        beta=BETA,
        threshold=THRESHOLD,
        surrogate_slope=SURROGATE_SLOPE,
        pretrained=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = remove_module_prefix(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    return model


# ============================================================
# 3. Entry point
# ============================================================

def main() -> None:
    device = get_device(
        preferred=DEVICE_NAME,
        verbose=True,
    )

    run_video_generation(
        config=CONFIG,
        model_loader=load_model,
        transform_builder=lambda image_size: build_eval_transform(
            image_size=image_size
        ),
        device=device,
    )


if __name__ == "__main__":
    main()
