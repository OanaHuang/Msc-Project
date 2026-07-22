# Scripts/NTU_RGBD/
# 18_Train_MS_Spiking_ResNet50_Membrane_Heatmap_Human_Detection.py

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    build_ms_spiking_resnet50_membrane_heatmap,
    count_parameters,
)
from Scripts.NTU_RGBD.training import (
    HeatmapMSELoss,
    run_training,
)


TRAIN_CSV = NTU_METADATA_DIR / "train_split.csv"
VAL_CSV = NTU_METADATA_DIR / "val_split.csv"

OUTPUT_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / (
        "18_Train_MS_Spiking_ResNet50_"
        "Membrane_Heatmap_Human_Detection"
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_JOINTS = 25
HEATMAP_SIGMA = 2.0

MAX_TRAIN_VIDEOS = None
MAX_VAL_VIDEOS = None
FRAME_STRIDE = 5

NUM_STEPS = 2
BETA = 0.90
THRESHOLD = 1.0
SURROGATE_SLOPE = 25.0
PRETRAINED = True
READOUT_TYPE = "mean_membrane"

BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 50

PERSON_CROP = True
BBOX_EXPANSION = 0.25

DEVICE_NAME = None
PIN_MEMORY = torch.cuda.is_available()


def build_datasets() -> tuple[NTUFrameDataset, NTUFrameDataset]:
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

    return train_dataset, val_dataset


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
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader


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
    print("=" * 72)
    print(
        "NTU RGB+D Model 18: MS-Spiking ResNet50 "
        "Mean Membrane Heatmap Human Detection"
    )
    print("=" * 72)

    print(f"Train CSV:          {TRAIN_CSV}")
    print(f"Validation CSV:     {VAL_CSV}")
    print(f"Output directory:   {OUTPUT_DIR}")

    print()
    print("SNN configuration")
    print("-" * 72)
    print(f"Number of steps:    {NUM_STEPS}")
    print(f"LIF beta:           {BETA}")
    print(f"LIF threshold:      {THRESHOLD}")
    print(f"Surrogate slope:    {SURROGATE_SLOPE}")
    print(f"Readout type:       {READOUT_TYPE}")
    print(f"Pretrained:         {PRETRAINED}")

    print()
    print("Training configuration")
    print("-" * 72)
    print(f"Image size:         {IMAGE_SIZE}")
    print(f"Heatmap size:       {HEATMAP_SIZE}")
    print(f"Number of joints:   {NUM_JOINTS}")
    print(f"Heatmap sigma:      {HEATMAP_SIGMA}")
    print(f"Frame stride:       {FRAME_STRIDE}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Epochs:             {EPOCHS}")
    print(f"Learning rate:      {LEARNING_RATE}")
    print(f"Weight decay:       {WEIGHT_DECAY}")
    print(f"Person crop:        {PERSON_CROP}")
    print(f"BBox expansion:     {BBOX_EXPANSION}")
    print(f"Device:             {device}")

    train_dataset, val_dataset = build_datasets()
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
    )

    print()
    print("Dataset summary")
    print("-" * 72)
    print(f"Train frame samples: {len(train_dataset)}")
    print(f"Val frame samples:   {len(val_dataset)}")
    print(f"Train batches:       {len(train_loader)}")
    print(f"Val batches:         {len(val_loader)}")

    model = build_ms_spiking_resnet50_membrane_heatmap(
        num_joints=NUM_JOINTS,
        num_steps=NUM_STEPS,
        beta=BETA,
        threshold=THRESHOLD,
        surrogate_slope=SURROGATE_SLOPE,
        pretrained=PRETRAINED,
    )

    total_parameters, trainable_parameters = count_parameters(
        model
    )

    print()
    print("Model summary")
    print("-" * 72)
    print(f"Total parameters:     {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    criterion = HeatmapMSELoss(reduction="mean")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
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
            "NTU RGB+D Model 18 MS-Spiking ResNet50 "
            "Mean Membrane Heatmap Human Detection"
        ),
    )

    print()
    print("=" * 72)
    print("Model 18 training completed")
    print("=" * 72)

    completed_epochs = len(history["epoch"])
    print(f"Completed epochs: {completed_epochs}")

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
        print(f"  {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()