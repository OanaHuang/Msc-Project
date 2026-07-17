# Scripts/NTU_RGBD/06_Train_ResNet50_Heatmap.py

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
    sys.path.insert(0, str(PROJECT_ROOT))


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
    build_resnet50_heatmap,
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
    / "06_Train_ResNet50_Heatmap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. Training configuration
# ============================================================

SEED = 42

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_JOINTS = 25
HEATMAP_SIGMA = 2.0

# Small local test configuration.
MAX_TRAIN_VIDEOS = None
MAX_VAL_VIDEOS = None

# Use one frame every 10 frames.
FRAME_STRIDE = 5

BATCH_SIZE = 32
NUM_WORKERS = 4

EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PRINT_EVERY = 50

PRETRAINED = True

# Leave as None to select the best available device automatically.
#
# Examples:
# DEVICE_NAME = "cpu"
# DEVICE_NAME = "mps"
# DEVICE_NAME = "cuda:0"
DEVICE_NAME = None

PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# 5. Dataset
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
    )

    return train_dataset, val_dataset


# ============================================================
# 6. DataLoader
# ============================================================

def build_dataloaders(
    train_dataset: NTUFrameDataset,
    val_dataset: NTUFrameDataset,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# 7. Main
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
    print("NTU RGB+D ResNet50 heatmap training")
    print("=" * 70)

    print(f"Train CSV:         {TRAIN_CSV}")
    print(f"Validation CSV:    {VAL_CSV}")
    print(f"Output directory:  {OUTPUT_DIR}")

    print()
    print("Configuration")
    print("-" * 70)

    print(f"Image size:        {IMAGE_SIZE}")
    print(f"Heatmap size:      {HEATMAP_SIZE}")
    print(f"Number of joints:  {NUM_JOINTS}")
    print(f"Heatmap sigma:     {HEATMAP_SIGMA}")

    print(f"Train videos:      {MAX_TRAIN_VIDEOS}")
    print(f"Val videos:        {MAX_VAL_VIDEOS}")
    print(f"Frame stride:      {FRAME_STRIDE}")

    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Epochs:            {EPOCHS}")
    print(f"Learning rate:     {LEARNING_RATE}")
    print(f"Weight decay:      {WEIGHT_DECAY}")
    print(f"Pretrained:        {PRETRAINED}")

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

    model = build_resnet50_heatmap(
        num_joints=NUM_JOINTS,
        pretrained=PRETRAINED,
    )

    total_parameters, trainable_parameters = (
        count_parameters(model)
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

    criterion = HeatmapMSELoss(
        reduction="mean",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    )

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
            "NTU RGB+D ResNet50 Heatmap"
        ),
    )

    print()
    print("=" * 70)
    print("Training script completed")
    print("=" * 70)

    print(
        f"Completed epochs: "
        f"{len(history['epoch'])}"
    )

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