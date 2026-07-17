# Scripts/Penn Action Model Training/
# 14_Generate_MP4_Spiking_ResNet50_Heatmap.py

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


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
    / "09b_Spiking_ResNet50_Heatmap_ImageNet_T4.py"
)

CKPT_PATH = (
    PROJECT_ROOT
    / "server_outputs"
    / "09b_Spiking_ResNet50_Heatmap_ImageNet_T4"
    / "best_Spiking_ResNet50_Heatmap_ImageNet_T4.pth"
)

TARGET_VIDEO_ID = "0684"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "14_Generate_MP4_Spiking_ResNet50_Heatmap"
)

OUTPUT_VIDEO_PATH = (
    OUTPUT_DIR
    / f"{TARGET_VIDEO_ID}_spiking_resnet50_imagenet_t4_pose.mp4"
)

PREDICTION_NPZ_PATH = (
    OUTPUT_DIR
    / f"{TARGET_VIDEO_ID}_spiking_resnet50_imagenet_t4_predictions.npz"
)

METADATA_PATH = (
    OUTPUT_DIR
    / f"{TARGET_VIDEO_ID}_spiking_resnet50_imagenet_t4_metadata.json"
)


# ============================================================
# 2. Import helpers
# ============================================================

def import_module_from_path(
    module_name: str,
    script_path: Path,
):
    if not script_path.exists():
        raise FileNotFoundError(
            f"Python script was not found:\n{script_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import Python script:\n{script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# 3. ResNet50 model creation
# ============================================================

def create_resnet50_model(training_module):
    """
    Create exactly the same model architecture used by 09b.

    Factory functions are attempted first because they are the safest
    way to reproduce the training configuration.
    """

    factory_names = [
        "build_model",
        "create_model",
        "get_model",
        "make_model",
    ]

    argument_sets = [
        {},
        {"num_keypoints": 13},
        {"num_joints": 13},
        {"num_classes": 13},
    ]

    for factory_name in factory_names:
        factory = getattr(
            training_module,
            factory_name,
            None,
        )

        if not callable(factory):
            continue

        for arguments in argument_sets:
            try:
                model = factory(**arguments)

                print(
                    f"Creating model with "
                    f"{factory_name}({arguments})"
                )

                return model

            except TypeError:
                continue

    class_names = [
        "SpikingResNet50Heatmap",
        "SpikingResNet50HeatmapModel",
        "SpikingResNet50PoseModel",
        "SNNResNet50Heatmap",
        "SpikingResNetHeatmap",
        "SpikingResNetPoseModel",
        "SNNHeatmapPoseModel",
        "PoseModel",
        "HeatmapModel",
    ]

    for class_name in class_names:
        model_class = getattr(
            training_module,
            class_name,
            None,
        )

        if model_class is None:
            continue

        for arguments in argument_sets:
            try:
                model = model_class(**arguments)

                print(
                    f"Creating model with "
                    f"{class_name}({arguments})"
                )

                return model

            except TypeError:
                continue

    # Final fallback: inspect all model classes declared in 09b.
    discovered_classes = []

    for name, value in vars(training_module).items():
        if not isinstance(value, type):
            continue

        if not issubclass(value, torch.nn.Module):
            continue

        if value.__module__ != training_module.__name__:
            continue

        discovered_classes.append(
            (name, value)
        )

    for class_name, model_class in discovered_classes:
        for arguments in argument_sets:
            try:
                model = model_class(**arguments)

                parameter_count = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )

                # Avoid accidentally selecting an individual block.
                # ResNet50 pose models should be much larger than 1M.
                if parameter_count > 10_000_000:
                    print(
                        "Automatically selected model class: "
                        f"{class_name}"
                    )
                    print(
                        f"Parameter count: "
                        f"{parameter_count:,}"
                    )

                    return model

            except (TypeError, ValueError):
                continue

    available_classes = [
        name
        for name, value in vars(training_module).items()
        if isinstance(value, type)
    ]

    raise RuntimeError(
        "Could not construct the 09b model.\n"
        f"Classes found in 09b: {available_classes}\n\n"
        "Add the exact 09b model class name to class_names."
    )


# ============================================================
# 4. Configure base script
# ============================================================

def configure_base_module(
    base_module,
    training_module,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_module.TRAIN_SCRIPT_PATH = TRAIN_SCRIPT_PATH
    base_module.CKPT_PATH = CKPT_PATH
    base_module.TARGET_VIDEO_ID = TARGET_VIDEO_ID

    base_module.OUTPUT_DIR = OUTPUT_DIR
    base_module.OUTPUT_VIDEO_PATH = OUTPUT_VIDEO_PATH
    base_module.PREDICTION_NPZ_PATH = PREDICTION_NPZ_PATH

    def create_model_override(_):
        return create_resnet50_model(
            training_module
        )

    # Replace script 12's ResNet18-oriented model selector.
    base_module.create_model = create_model_override


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    print("=" * 72)
    print("14 - Spiking ResNet50 ImageNet T4 MP4 Generation")
    print("=" * 72)
    print(f"Base script    : {BASE_SCRIPT_PATH}")
    print(f"Training script: {TRAIN_SCRIPT_PATH}")
    print(f"Checkpoint     : {CKPT_PATH}")
    print(f"Target video   : {TARGET_VIDEO_ID}")
    print(f"Output video   : {OUTPUT_VIDEO_PATH}")
    print(f"Prediction NPZ : {PREDICTION_NPZ_PATH}")
    print("=" * 72)

    if not BASE_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "Script 12 was not found:\n"
            f"{BASE_SCRIPT_PATH}"
        )

    if not TRAIN_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "09b training script was not found:\n"
            f"{TRAIN_SCRIPT_PATH}"
        )

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            "09b checkpoint was not found:\n"
            f"{CKPT_PATH}\n\n"
            "Expected folder:\n"
            f"{CKPT_PATH.parent}"
        )

    base_module = import_module_from_path(
        module_name="spiking_resnet50_mp4_base",
        script_path=BASE_SCRIPT_PATH,
    )

    training_module = import_module_from_path(
        module_name="spiking_resnet50_training_module",
        script_path=TRAIN_SCRIPT_PATH,
    )

    configure_base_module(
        base_module=base_module,
        training_module=training_module,
    )

    base_module.main()

    # Script 12 uses its original ResNet18 metadata filename.
    old_metadata_path = (
        OUTPUT_DIR
        / f"{TARGET_VIDEO_ID}_spiking_resnet18_metadata.json"
    )

    if old_metadata_path.exists():
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

        old_metadata_path.rename(
            METADATA_PATH
        )

    print("\n" + "=" * 72)
    print("14 generation finished.")
    print("=" * 72)
    print(f"MP4        : {OUTPUT_VIDEO_PATH}")
    print(f"Prediction : {PREDICTION_NPZ_PATH}")
    print(f"Metadata   : {METADATA_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()