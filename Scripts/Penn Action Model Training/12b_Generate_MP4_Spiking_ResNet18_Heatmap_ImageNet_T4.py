# Scripts/Penn Action Model Training/
# 12b_Generate_MP4_Spiking_ResNet18_Heatmap_ImageNet_T4.py

from __future__ import annotations

import importlib.util
from pathlib import Path


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_SCRIPT_PATH = (
    PROJECT_ROOT
    / "Scripts"
    / "Penn Action Model Training"
    / "12_Generate_MP4_Spiking_ResNet18_Heatmap.py"
)

TRAIN_SCRIPT_PATH = (
    PROJECT_ROOT
    / "Scripts"
    / "Penn Action Model Training"
    / "08b_Spiking_ResNet18_Heatmap_ImageNet_T4.py"
)

CKPT_PATH = (
    PROJECT_ROOT
    / "server_outputs"
    / "08b_Spiking_ResNet18_Heatmap_ImageNet_T4"
    / "best_Spiking_ResNet18_Heatmap_ImageNet_T4.pth"
)

TARGET_VIDEO_ID = "0684"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "12b_Generate_MP4_Spiking_ResNet18_Heatmap_ImageNet_T4"
)

OUTPUT_VIDEO_PATH = (
    OUTPUT_DIR
    / f"{TARGET_VIDEO_ID}_spiking_resnet18_imagenet_t4_pose.mp4"
)

PREDICTION_NPZ_PATH = (
    OUTPUT_DIR
    / (
        f"{TARGET_VIDEO_ID}_"
        "spiking_resnet18_imagenet_t4_predictions.npz"
    )
)

METADATA_PATH = (
    OUTPUT_DIR
    / (
        f"{TARGET_VIDEO_ID}_"
        "spiking_resnet18_imagenet_t4_metadata.json"
    )
)


# ============================================================
# 2. Load the original 12 script
# ============================================================

def load_base_module():
    if not BASE_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "Base MP4 generation script was not found:\n"
            f"{BASE_SCRIPT_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "generate_mp4_spiking_resnet18_base",
        BASE_SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not import the base MP4 generation script:\n"
            f"{BASE_SCRIPT_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# 3. Configure 08b experiment
# ============================================================

def configure_module(module) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.TRAIN_SCRIPT_PATH = TRAIN_SCRIPT_PATH
    module.CKPT_PATH = CKPT_PATH
    module.TARGET_VIDEO_ID = TARGET_VIDEO_ID

    module.OUTPUT_DIR = OUTPUT_DIR
    module.OUTPUT_VIDEO_PATH = OUTPUT_VIDEO_PATH
    module.PREDICTION_NPZ_PATH = PREDICTION_NPZ_PATH

    # Keep all other settings identical to script 12:
    #
    # IMAGE_SIZE
    # HEATMAP_SIZE
    # OUTPUT_FPS
    # MAX_FRAMES
    # skeleton connections
    # legend style
    # prediction and ground-truth colors
    # image normalization
    # heatmap decoding


# ============================================================
# 4. Main
# ============================================================

def main() -> None:
    print("=" * 72)
    print("12b - Spiking ResNet18 ImageNet T4 MP4 Generation")
    print("=" * 72)
    print(f"Base script    : {BASE_SCRIPT_PATH}")
    print(f"Training script: {TRAIN_SCRIPT_PATH}")
    print(f"Checkpoint     : {CKPT_PATH}")
    print(f"Target video   : {TARGET_VIDEO_ID}")
    print(f"Output folder  : {OUTPUT_DIR}")
    print("=" * 72)

    if not TRAIN_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "08b training script was not found:\n"
            f"{TRAIN_SCRIPT_PATH}\n\n"
            "Confirm that the training script filename is exactly:\n"
            "08b_Spiking_ResNet18_Heatmap_ImageNet_T4.py"
        )

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            "08b checkpoint was not found:\n"
            f"{CKPT_PATH}\n\n"
            "Expected local checkpoint folder:\n"
            f"{CKPT_PATH.parent}"
        )

    module = load_base_module()
    configure_module(module)

    # The original script creates its own metadata filename.
    # Override Path operations temporarily by replacing OUTPUT_DIR and
    # TARGET_VIDEO_ID before calling main().
    module.main()

    original_metadata_path = (
        OUTPUT_DIR
        / f"{TARGET_VIDEO_ID}_spiking_resnet18_metadata.json"
    )

    if original_metadata_path.exists():
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

        original_metadata_path.rename(
            METADATA_PATH
        )

        print(
            f"Renamed metadata: {METADATA_PATH}"
        )

    print("\n" + "=" * 72)
    print("12b generation complete.")
    print("=" * 72)
    print(f"MP4        : {OUTPUT_VIDEO_PATH}")
    print(f"Prediction : {PREDICTION_NPZ_PATH}")
    print(f"Metadata   : {METADATA_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()