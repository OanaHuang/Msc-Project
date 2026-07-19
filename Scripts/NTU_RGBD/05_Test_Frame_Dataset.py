# Scripts/NTU_RGBD/05_Test_Frame_Dataset.py

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
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
)

from Scripts.NTU_RGBD.datasets import (
    NTUFrameDataset,
    build_eval_transform,
)


# ============================================================
# 3. Configuration
# ============================================================

SEED = 42

METADATA_CSV = (
    NTU_METADATA_DIR
    / "train_split.csv"
)

OUTPUT_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / "05_Test_Frame_Dataset"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_JOINTS = 25
HEATMAP_SIGMA = 2.0

FRAME_STRIDE = 5

# Number of videos used for this quick test.
MAX_TEST_VIDEOS = 3

BATCH_SIZE = 4
NUM_WORKERS = 0

NUM_SAMPLES_TO_CHECK = 5

VISUALIZATION_PATH = (
    OUTPUT_DIR
    / "dataset_sample.jpg"
)


# ============================================================
# 4. Skeleton connections
# ============================================================

# NTU joint indices are converted from 1-based to 0-based.
SKELETON_EDGES = [
    (0, 1),
    (1, 20),
    (20, 2),
    (2, 3),

    (20, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 21),
    (7, 22),

    (20, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 23),
    (11, 24),

    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),

    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
]


# ============================================================
# 5. Utility functions
# ============================================================

def tensor_to_bgr_image(
    image_tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convert the normalized ImageNet tensor back to
    a BGR uint8 image for OpenCV visualization.
    """
    if image_tensor.ndim != 3:
        raise ValueError(
            "image_tensor must have shape [C, H, W]"
        )

    image = (
        image_tensor
        .detach()
        .cpu()
        .float()
        .clone()
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=image.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=image.dtype,
    ).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0.0, 1.0)

    image = (
        image.permute(1, 2, 0)
        .numpy()
        * 255.0
    ).astype(np.uint8)

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )


def draw_pose(
    image: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    visibility = np.asarray(
        visibility,
        dtype=np.float32,
    ) > 0

    for joint_a, joint_b in SKELETON_EDGES:
        if not (
            visibility[joint_a]
            and visibility[joint_b]
        ):
            continue

        point_a = tuple(
            np.round(
                keypoints[joint_a]
            ).astype(int)
        )

        point_b = tuple(
            np.round(
                keypoints[joint_b]
            ).astype(int)
        )

        cv2.line(
            output,
            point_a,
            point_b,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for joint_index, point in enumerate(
        keypoints
    ):
        if not visibility[joint_index]:
            continue

        center = tuple(
            np.round(point).astype(int)
        )

        cv2.circle(
            output,
            center,
            4,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

    return output


def validate_sample(
    sample: dict[str, object],
    sample_index: int,
) -> None:
    required_keys = {
        "image",
        "heatmaps",
        "keypoints",
        "visibility",
        "sample_id",
        "frame_index",
        "rgb_path",
    }

    missing_keys = (
        required_keys
        - set(sample.keys())
    )

    if missing_keys:
        raise KeyError(
            f"Sample {sample_index} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    image = sample["image"]
    heatmaps = sample["heatmaps"]
    keypoints = sample["keypoints"]
    visibility = sample["visibility"]

    if not isinstance(image, torch.Tensor):
        raise TypeError(
            "image must be a torch.Tensor"
        )

    if not isinstance(heatmaps, torch.Tensor):
        raise TypeError(
            "heatmaps must be a torch.Tensor"
        )

    if not isinstance(keypoints, torch.Tensor):
        raise TypeError(
            "keypoints must be a torch.Tensor"
        )

    if not isinstance(visibility, torch.Tensor):
        raise TypeError(
            "visibility must be a torch.Tensor"
        )

    expected_image_shape = (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    expected_heatmap_shape = (
        NUM_JOINTS,
        HEATMAP_SIZE,
        HEATMAP_SIZE,
    )

    expected_keypoint_shape = (
        NUM_JOINTS,
        2,
    )

    expected_visibility_shape = (
        NUM_JOINTS,
    )

    assert tuple(image.shape) == (
        expected_image_shape
    ), (
        f"Unexpected image shape: "
        f"{tuple(image.shape)}"
    )

    assert tuple(heatmaps.shape) == (
        expected_heatmap_shape
    ), (
        f"Unexpected heatmap shape: "
        f"{tuple(heatmaps.shape)}"
    )

    assert tuple(keypoints.shape) == (
        expected_keypoint_shape
    ), (
        f"Unexpected keypoint shape: "
        f"{tuple(keypoints.shape)}"
    )

    assert tuple(visibility.shape) == (
        expected_visibility_shape
    ), (
        f"Unexpected visibility shape: "
        f"{tuple(visibility.shape)}"
    )

    assert torch.isfinite(image).all()
    assert torch.isfinite(heatmaps).all()
    assert torch.isfinite(keypoints).all()
    assert torch.isfinite(visibility).all()

    rgb_path = Path(
        str(sample["rgb_path"])
    )

    assert rgb_path.exists(), (
        f"Extracted JPG does not exist: "
        f"{rgb_path}"
    )

    assert rgb_path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
    }, (
        f"Dataset is not reading an image file: "
        f"{rgb_path}"
    )


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    seed_everything(
        seed=SEED,
        deterministic=False,
    )

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: "
            f"{METADATA_CSV}\n"
            "Run 04_Create_Splits.py first."
        )

    print("=" * 70)
    print("NTU RGB+D extracted-frame Dataset test")
    print("=" * 70)

    print(f"Metadata CSV:       {METADATA_CSV}")
    print(f"Output directory:   {OUTPUT_DIR}")
    print(f"Image size:         {IMAGE_SIZE}")
    print(f"Heatmap size:       {HEATMAP_SIZE}")
    print(f"Frame stride:       {FRAME_STRIDE}")
    print(f"Test videos:        {MAX_TEST_VIDEOS}")

    dataset = NTUFrameDataset(
        metadata_csv=METADATA_CSV,
        transform=build_eval_transform(
            image_size=IMAGE_SIZE,
        ),
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,
        frame_stride=FRAME_STRIDE,
        single_person_only=True,
        max_samples=MAX_TEST_VIDEOS,
        skeleton_cache_size=4,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset contains no frame samples"
        )

    print()
    print("Dataset summary")
    print("-" * 70)

    print(
        f"Number of videos:   "
        f"{len(dataset.samples)}"
    )

    print(
        f"Number of frames:   "
        f"{len(dataset)}"
    )

    samples_to_check = min(
        NUM_SAMPLES_TO_CHECK,
        len(dataset),
    )

    print()
    print("Individual sample checks")
    print("-" * 70)

    first_sample = None

    for index in range(
        samples_to_check
    ):
        sample = dataset[index]

        validate_sample(
            sample=sample,
            sample_index=index,
        )

        if first_sample is None:
            first_sample = sample

        print(
            f"Sample {index:>2}: "
            f"id={sample['sample_id']}, "
            f"frame={sample['frame_index']}, "
            f"visible={int(sample['visibility'].sum().item())}, "
            f"path={sample['rgb_path']}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    batch = next(
        iter(loader)
    )

    print()
    print("DataLoader batch check")
    print("-" * 70)

    print(
        f"Image batch:        "
        f"{tuple(batch['image'].shape)}"
    )

    print(
        f"Heatmap batch:      "
        f"{tuple(batch['heatmaps'].shape)}"
    )

    print(
        f"Keypoint batch:     "
        f"{tuple(batch['keypoints'].shape)}"
    )

    print(
        f"Visibility batch:   "
        f"{tuple(batch['visibility'].shape)}"
    )

    assert batch["image"].ndim == 4
    assert batch["heatmaps"].ndim == 4
    assert batch["keypoints"].ndim == 3
    assert batch["visibility"].ndim == 2

    if first_sample is None:
        raise RuntimeError(
            "No sample available for visualization"
        )

    image = tensor_to_bgr_image(
        first_sample["image"]
    )

    keypoints = (
        first_sample["keypoints"]
        .detach()
        .cpu()
        .numpy()
    )

    visibility = (
        first_sample["visibility"]
        .detach()
        .cpu()
        .numpy()
    )

    visualization = draw_pose(
        image=image,
        keypoints=keypoints,
        visibility=visibility,
    )

    saved = cv2.imwrite(
        str(VISUALIZATION_PATH),
        visualization,
    )

    if not saved:
        raise RuntimeError(
            f"Could not save visualization: "
            f"{VISUALIZATION_PATH}"
        )

    print()
    print("=" * 70)
    print("Dataset test passed")
    print("=" * 70)

    print(
        f"Visualization saved: "
        f"{VISUALIZATION_PATH}"
    )

    print(
        "The Dataset is reading extracted "
        "JPG frames successfully."
    )


if __name__ == "__main__":
    main()