# Scripts/NTU_RGBD/05_Test_Frame_Dataset.py

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Scripts.common.paths import NTU_METADATA_DIR

from Scripts.NTU_RGBD.datasets import (
    NTUFrameDataset,
    build_eval_transform,
)


TRAIN_CSV = (
    NTU_METADATA_DIR
    / "train_split.csv"
)

IMAGE_SIZE = 224
HEATMAP_SIZE = 56

FRAME_STRIDE = 10
MAX_VIDEO_SAMPLES = 5

BATCH_SIZE = 4


def main() -> None:
    dataset = NTUFrameDataset(
        metadata_csv=TRAIN_CSV,
        transform=build_eval_transform(
            image_size=IMAGE_SIZE,
        ),
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=2.0,
        frame_stride=FRAME_STRIDE,
        single_person_only=True,
        max_samples=MAX_VIDEO_SAMPLES,
    )

    print("=" * 70)
    print("Single sample test")
    print("=" * 70)

    sample = dataset[0]

    print(
        f"Image shape:      "
        f"{tuple(sample['image'].shape)}"
    )

    print(
        f"Heatmap shape:    "
        f"{tuple(sample['heatmaps'].shape)}"
    )

    print(
        f"Keypoints shape:  "
        f"{tuple(sample['keypoints'].shape)}"
    )

    print(
        f"Visibility shape: "
        f"{tuple(sample['visibility'].shape)}"
    )

    print(
        f"Sample ID:        "
        f"{sample['sample_id']}"
    )

    print(
        f"Frame index:      "
        f"{sample['frame_index']}"
    )

    print(
        f"Visible joints:   "
        f"{int(sample['visibility'].sum().item())}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    batch = next(
        iter(dataloader)
    )

    print()
    print("=" * 70)
    print("Batch test")
    print("=" * 70)

    print(
        f"Batch images:     "
        f"{tuple(batch['image'].shape)}"
    )

    print(
        f"Batch heatmaps:   "
        f"{tuple(batch['heatmaps'].shape)}"
    )

    print(
        f"Batch keypoints:  "
        f"{tuple(batch['keypoints'].shape)}"
    )

    print(
        f"Batch visibility: "
        f"{tuple(batch['visibility'].shape)}"
    )

    expected_image_shape = (
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    expected_heatmap_shape = (
        BATCH_SIZE,
        25,
        HEATMAP_SIZE,
        HEATMAP_SIZE,
    )

    assert (
        tuple(batch["image"].shape)
        == expected_image_shape
    )

    assert (
        tuple(batch["heatmaps"].shape)
        == expected_heatmap_shape
    )

    assert torch.isfinite(
        batch["image"]
    ).all()

    assert torch.isfinite(
        batch["heatmaps"]
    ).all()

    print()
    print("NTUFrameDataset test passed.")


if __name__ == "__main__":
    main()