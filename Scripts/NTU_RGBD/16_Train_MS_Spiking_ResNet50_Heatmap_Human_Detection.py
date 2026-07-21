# Scripts/NTU_RGBD/
# 16_Train_MS_Spiking_ResNet50_Heatmap_Human_Detection.py

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


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
# 2. Imports
# ============================================================

from Scripts.common.paths import (
    NTU_METADATA_DIR,
    NTU_RGBD_OUTPUT_DIR,
)

from Scripts.common.reproducibility import (
    seed_everything,
    get_device,
)

from Scripts.NTU_RGBD.datasets import (
    NTUFrameDataset,
    build_train_transform,
    build_eval_transform,
)

from Scripts.NTU_RGBD.models import (
    build_ms_spiking_resnet50_heatmap,
    count_parameters,
)

from Scripts.NTU_RGBD.training import (
    HeatmapMSELoss,
    run_training,
)


# ============================================================
# 3. Paths
# ============================================================

TRAIN_CSV = (
    NTU_METADATA_DIR
    / "train_split.csv"
)

VAL_CSV = (
    NTU_METADATA_DIR
    / "val_split.csv"
)

OUTPUT_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / (
        "16_Train_MS_Spiking_ResNet50_"
        "Heatmap_Human_Detection"
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. Pose configuration
# ============================================================

SEED = 42

IMAGE_SIZE = 224
HEATMAP_SIZE = 56

NUM_JOINTS = 25
HEATMAP_SIGMA = 2.0

# Full dataset.
MAX_TRAIN_VIDEOS = None
MAX_VAL_VIDEOS = None

# Use one frame every five frames.
FRAME_STRIDE = 5


# ============================================================
# 5. SNN configuration
# ============================================================

# Current implementation repeats the same RGB frame over
# multiple SNN simulation steps.
#
# This keeps the model compatible with NTUFrameDataset,
# which currently returns one frame per sample.
NUM_STEPS = 2

# LIF membrane decay.
BETA = 0.90

# LIF firing threshold.
THRESHOLD = 1.0

# Slope used by the surrogate gradient function.
SURROGATE_SLOPE = 25.0

# Load compatible ImageNet ResNet50 convolution and
# BatchNorm weights.
PRETRAINED = True


# ============================================================
# 6. Training configuration
# ============================================================

# The SNN backbone processes every batch NUM_STEPS times.
# Start with a smaller batch size than the ANN ResNet50.
BATCH_SIZE = 8

NUM_WORKERS = 4

EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PRINT_EVERY = 50


# ============================================================
# 7. Human detection / person crop configuration
# ============================================================

# The current implementation uses NTU skeleton keypoints
# to generate a person bounding box.
#
# It is not an external detector such as YOLO.
PERSON_CROP = True

# Expand the skeleton-derived bounding box by 25%.
BBOX_EXPANSION = 0.25


# ============================================================
# 8. Device configuration
# ============================================================

# Leave as None to select the best available device.
#
# Examples:
# DEVICE_NAME = "cpu"
# DEVICE_NAME = "mps"
# DEVICE_NAME = "cuda:0"
# DEVICE_NAME = "cuda:1"
DEVICE_NAME = None

PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# 9. Dataset
# ============================================================

def build_datasets() -> tuple[
    NTUFrameDataset,
    NTUFrameDataset,
]:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Train split not found: {TRAIN_CSV}\n"
            "Run 04_Create_Splits.py first."
        )

    if not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Validation split not found: {VAL_CSV}\n"
            "Run 04_Create_Splits.py first."
        )

    train_dataset = NTUFrameDataset(
        metadata_csv=TRAIN_CSV,
        transform=build_train_transform(
            image_size=IMAGE_SIZE,
        ),
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
        frame_stride=FRAME_STRIDE,
        single_person_only=True,
        max_samples=MAX_TRAIN_VIDEOS,
        skeleton_cache_size=8,
        person_crop=PERSON_CROP,
        bbox_expansion=BBOX_EXPANSION,
    )

    val_dataset = NTUFrameDataset(
        metadata_csv=VAL_CSV,
        transform=build_eval_transform(
            image_size=IMAGE_SIZE,
        ),
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
        frame_stride=FRAME_STRIDE,
        single_person_only=True,
        max_samples=MAX_VAL_VIDEOS,
        skeleton_cache_size=8,
        person_crop=PERSON_CROP,
        bbox_expansion=BBOX_EXPANSION,
    )

    return (
        train_dataset,
        val_dataset,
    )


# ============================================================
# 10. DataLoader
# ============================================================

def build_dataloaders(
    train_dataset: NTUFrameDataset,
    val_dataset: NTUFrameDataset,
) -> tuple[
    DataLoader,
    DataLoader,
]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    return (
        train_loader,
        val_loader,
    )


# ============================================================
# 11. Model shape test
# ============================================================

def validate_model_output(
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    Verify the model output shape before training.

    Expected:
        Input:
            1 x 3 x 224 x 224

        Output:
            1 x 25 x 56 x 56
    """

    model.eval()

    test_input = torch.zeros(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    with torch.no_grad():
        test_output = model(
            test_input,
        )

    expected_shape = (
        1,
        NUM_JOINTS,
        HEATMAP_SIZE,
        HEATMAP_SIZE,
    )

    received_shape = tuple(
        test_output.shape,
    )

    print()
    print("Model shape test")
    print("-" * 70)

    print(
        f"Input shape:         "
        f"{tuple(test_input.shape)}"
    )

    print(
        f"Output shape:        "
        f"{received_shape}"
    )

    print(
        f"Expected shape:      "
        f"{expected_shape}"
    )

    if received_shape != expected_shape:
        raise RuntimeError(
            "Unexpected model output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {received_shape}"
        )

    print(
        "Model shape test:   Passed"
    )

    del test_input
    del test_output

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 12. Main
# ============================================================

def main() -> None:
    seed_everything(
        seed=SEED,
        deterministic=False,
    )

    device = get_device(
        preferred=DEVICE_NAME,
        verbose=True,
    )

    print()
    print("=" * 70)
    print(
        "NTU RGB+D MS-Spiking ResNet50 Heatmap "
        "with Human Detection"
    )
    print("=" * 70)

    print(f"Train CSV:         {TRAIN_CSV}")
    print(f"Validation CSV:    {VAL_CSV}")
    print(f"Output directory:  {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Pose configuration
    # --------------------------------------------------------

    print()
    print("Pose configuration")
    print("-" * 70)

    print(f"Image size:        {IMAGE_SIZE}")
    print(f"Heatmap size:      {HEATMAP_SIZE}")
    print(f"Number of joints:  {NUM_JOINTS}")
    print(f"Heatmap sigma:     {HEATMAP_SIGMA}")

    print(f"Train videos:      {MAX_TRAIN_VIDEOS}")
    print(f"Val videos:        {MAX_VAL_VIDEOS}")
    print(f"Frame stride:      {FRAME_STRIDE}")

    # --------------------------------------------------------
    # SNN configuration
    # --------------------------------------------------------

    print()
    print("SNN configuration")
    print("-" * 70)

    print(f"Number of steps:   {NUM_STEPS}")
    print(f"LIF beta:          {BETA}")
    print(f"LIF threshold:     {THRESHOLD}")
    print(f"Surrogate slope:   {SURROGATE_SLOPE}")
    print(f"Pretrained:        {PRETRAINED}")

    print(
        "Residual type:     "
        "Pre-activation MS Residual"
    )

    print(
        "Temporal input:    "
        "Same frame repeated over SNN steps"
    )

    print(
        "Temporal readout:  "
        "Mean spike rate"
    )

    print(
        "Decoder type:      "
        "ANN deconvolution heatmap decoder"
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print()
    print("Training configuration")
    print("-" * 70)

    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Epochs:            {EPOCHS}")
    print(f"Learning rate:     {LEARNING_RATE}")
    print(f"Weight decay:      {WEIGHT_DECAY}")
    print(f"Workers:           {NUM_WORKERS}")
    print(f"Device:            {device}")

    # --------------------------------------------------------
    # Human detection configuration
    # --------------------------------------------------------

    print()
    print("Human detection configuration")
    print("-" * 70)

    print(f"Person crop:       {PERSON_CROP}")
    print(f"BBox expansion:    {BBOX_EXPANSION}")

    print(
        "BBox source:       "
        "NTU skeleton annotations"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset, val_dataset = (
        build_datasets()
    )

    train_loader, val_loader = (
        build_dataloaders(
            train_dataset,
            val_dataset,
        )
    )

    print()
    print("Dataset summary")
    print("-" * 70)

    print(
        f"Train frame samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Val frame samples:   "
        f"{len(val_dataset)}"
    )

    print(
        f"Train batches:       "
        f"{len(train_loader)}"
    )

    print(
        f"Val batches:         "
        f"{len(val_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_ms_spiking_resnet50_heatmap(
        num_joints=NUM_JOINTS,
        num_steps=NUM_STEPS,
        beta=BETA,
        threshold=THRESHOLD,
        surrogate_slope=SURROGATE_SLOPE,
        pretrained=PRETRAINED,
    )

    total_parameters, trainable_parameters = (
        count_parameters(
            model,
        )
    )

    print()
    print("Model summary")
    print("-" * 70)

    print(
        f"Total parameters:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    model = model.to(
        device,
    )

    validate_model_output(
        model=model,
        device=device,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = HeatmapMSELoss(
        reduction="mean",
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=EPOCHS,
        output_dir=OUTPUT_DIR,
        scheduler=scheduler,
        print_every=PRINT_EVERY,
        model_name=(
            "NTU RGB+D MS-Spiking ResNet50 "
            "Heatmap Human Detection"
        ),
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Training script completed")
    print("=" * 70)

    completed_epochs = len(
        history["epoch"]
    )

    print(
        f"Completed epochs: "
        f"{completed_epochs}"
    )

    if completed_epochs > 0:
        print(
            f"Final train loss: "
            f"{history['train_loss'][-1]:.6f}"
        )

        print(
            f"Final val loss:   "
            f"{history['val_loss'][-1]:.6f}"
        )

    print()
    print("Generated files:")

    for filename in (
        "best_model.pt",
        "last_model.pt",
        "training_history.csv",
        "loss_curve.png",
        "loss_curve.pdf",
    ):
        print(
            f"  {OUTPUT_DIR / filename}"
        )


if __name__ == "__main__":
    main()