# Scripts/NTU_RGBD/
# 22_Generate_SpikeYOLO_Style_ILIF_Heatmap_Video.py

from __future__ import annotations

from pathlib import Path
import sys

import torch


# ============================================================
# 1. Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# 2. Project imports
# ============================================================

from Scripts.common.paths import (
    NTU_METADATA_DIR,
    NTU_RGBD_DATASET_DIR,
    NTU_RGBD_OUTPUT_DIR,
)

from Scripts.common.reproducibility import (
    get_device,
)

from Scripts.NTU_RGBD.datasets import (
    build_eval_transform,
)

from Scripts.NTU_RGBD.inference.video_generation import (
    VideoGenerationConfig,
    run_video_generation,
)

from Scripts.NTU_RGBD.models.spikeyolo_style_ilif_heatmap import (
    build_spikeyolo_style_ilif_heatmap,
)


# ============================================================
# 3. Model configuration
# ============================================================

MODEL_VERSION = "21"

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_JOINTS = 25


# ============================================================
# 4. I-LIF configuration
# ============================================================

# These values must exactly match Model 21 training.
NUM_STEPS = 2
BETA = 0.90
THRESHOLD = 1.0
MAX_SPIKES = 4


# ============================================================
# 5. Video generation configuration
# ============================================================

# Model 21 was trained using skeleton-guided person crops.
PERSON_CROP = True

BBOX_EXPANSION = 0.25

# Raw heatmap peak threshold.
CONFIDENCE_THRESHOLD = 0.02

# 1 means every source frame is processed.
FRAME_STRIDE = 1

# None keeps the source video's effective FPS.
OUTPUT_FPS = None

# None lets get_device select the device automatically.
DEVICE_NAME = None

# Skip an existing non-empty MP4.
SKIP_EXISTING_VIDEOS = True

# Save prediction coordinates, confidence, GT and bbox data.
SAVE_PREDICTION_NPZ = True

# Use 1 or 3 for a quick test.
# Use None to process every single-person test video.
MAX_TEST_VIDEOS = 1


# ============================================================
# 6. Paths
# ============================================================

TEST_CSV = (
    NTU_METADATA_DIR
    / "test_split.csv"
)

RGB_VIDEO_DIR = (
    NTU_RGBD_DATASET_DIR
    / "rgb_videos"
)

SKELETON_DIR = (
    NTU_RGBD_DATASET_DIR
    / "skeletons"
)

MODEL_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / "21_Train_SpikeYOLO_Style_ILIF_Heatmap"
)

MODEL_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

OUTPUT_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / "22_Generate_MP4_SpikeYOLO_Style_ILIF_Model_21"
)


# ============================================================
# 7. Shared video-generation configuration
# ============================================================

CONFIG = VideoGenerationConfig(
    model_version=MODEL_VERSION,

    test_csv=TEST_CSV,

    rgb_video_dir=RGB_VIDEO_DIR,

    skeleton_dir=SKELETON_DIR,

    output_dir=OUTPUT_DIR,

    image_size=IMAGE_SIZE,

    heatmap_size=HEATMAP_SIZE,

    num_joints=NUM_JOINTS,

    person_crop=PERSON_CROP,

    bbox_expansion=BBOX_EXPANSION,

    confidence_threshold=CONFIDENCE_THRESHOLD,

    frame_stride=FRAME_STRIDE,

    output_fps=OUTPUT_FPS,

    skip_existing_videos=(
        SKIP_EXISTING_VIDEOS
    ),

    save_prediction_npz=(
        SAVE_PREDICTION_NPZ
    ),

    max_test_videos=MAX_TEST_VIDEOS,

    readout_type=(
        "mean_integer_spike_features"
    ),

    prediction_label=(
        "Prediction - SpikeYOLO-style "
        f"I-LIF Model {MODEL_VERSION}"
    ),

    num_steps=NUM_STEPS,

    beta=BETA,

    threshold=THRESHOLD,
)


# ============================================================
# 8. Checkpoint utilities
# ============================================================

def extract_state_dict(
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    """
    Extract a state dictionary from common checkpoint formats.

    Supported formats:
        checkpoint["model_state_dict"]
        checkpoint["state_dict"]
        checkpoint["model"]
        direct state dictionary
    """
    if isinstance(
        checkpoint,
        dict,
    ):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = checkpoint.get(
                key
            )

            if isinstance(
                value,
                dict,
            ):
                return value

        if checkpoint and all(
            isinstance(
                value,
                torch.Tensor,
            )
            for value in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find a model state dictionary "
        "inside the checkpoint."
    )


def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove the 'module.' prefix added by DataParallel
    or DistributedDataParallel.
    """
    cleaned: dict[
        str,
        torch.Tensor,
    ] = {}

    for key, value in state_dict.items():
        cleaned_key = key

        if cleaned_key.startswith(
            "module."
        ):
            cleaned_key = cleaned_key[
                len("module.") :
            ]

        cleaned[
            cleaned_key
        ] = value

    return cleaned


# ============================================================
# 9. Model loading
# ============================================================

def load_model(
    device: torch.device,
) -> torch.nn.Module:
    """
    Build Model 21 and load its trained checkpoint.

    Moving the model to the device and switching it to eval mode
    are handled by run_video_generation().
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model checkpoint not found: "
            f"{MODEL_PATH}"
        )

    model = (
        build_spikeyolo_style_ilif_heatmap(
            num_joints=NUM_JOINTS,
            num_steps=NUM_STEPS,
            beta=BETA,
            threshold=THRESHOLD,
            max_spikes=MAX_SPIKES,
        )
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = remove_module_prefix(
        state_dict
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model


# ============================================================
# 10. Transform builder
# ============================================================

def build_transform(
    image_size: int,
):
    """
    Build the standard NTU RGB+D evaluation transform.
    """
    return build_eval_transform(
        image_size=image_size,
    )


# ============================================================
# 11. Model information
# ============================================================

def print_model_information() -> None:
    print()
    print("=" * 72)
    print(
        "NTU RGB+D SpikeYOLO-style I-LIF "
        "heatmap video generation"
    )
    print("=" * 72)

    print(
        f"Model version:       "
        f"{MODEL_VERSION}"
    )

    print(
        f"Model path:          "
        f"{MODEL_PATH}"
    )

    print(
        "Readout type:        "
        "Mean integer spike features"
    )

    print(
        f"Image size:          "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Heatmap size:        "
        f"{HEATMAP_SIZE}"
    )

    print(
        f"Number of joints:    "
        f"{NUM_JOINTS}"
    )

    print(
        f"SNN time steps:      "
        f"{NUM_STEPS}"
    )

    print(
        f"I-LIF beta:          "
        f"{BETA}"
    )

    print(
        f"I-LIF threshold:     "
        f"{THRESHOLD}"
    )

    print(
        f"Integer range:       "
        f"0..{MAX_SPIKES}"
    )

    print(
        f"Person crop:         "
        f"{PERSON_CROP}"
    )

    print(
        f"BBox expansion:      "
        f"{BBOX_EXPANSION}"
    )

    print(
        f"Confidence threshold:"
        f" {CONFIDENCE_THRESHOLD}"
    )

    print(
        f"Frame stride:        "
        f"{FRAME_STRIDE}"
    )

    print(
        f"Maximum videos:      "
        f"{MAX_TEST_VIDEOS}"
    )

    print(
        f"Output directory:    "
        f"{OUTPUT_DIR}"
    )

    print("=" * 72)


# ============================================================
# 12. Main
# ============================================================

def main() -> None:
    print_model_information()

    device = get_device(
        preferred=DEVICE_NAME,
        verbose=True,
    )

    run_video_generation(
        config=CONFIG,
        model_loader=load_model,
        transform_builder=build_transform,
        device=device,
    )


if __name__ == "__main__":
    main()