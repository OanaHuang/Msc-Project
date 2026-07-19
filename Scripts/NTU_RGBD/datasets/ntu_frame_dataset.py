from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from Scripts.common.paths import (
    NTU_RGBD_DATASET_DIR,
)

from Scripts.NTU_RGBD.core import (
    coordinate_visibility,
    extract_primary_pose_sequence,
    read_skeleton_file,
)

from Scripts.NTU_RGBD.datasets.person_crop import (
    crop_and_resize_person,
)

# ============================================================
# 1. Extracted frame directory
# ============================================================

NTU_EXTRACTED_FRAMES_DIR = (
    NTU_RGBD_DATASET_DIR
    / "extracted_frames"
)


# ============================================================
# 2. Heatmap generation
# ============================================================

def generate_gaussian_heatmaps(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    image_size: int = 224,
    heatmap_size: int = 56,
    sigma: float = 2.0,
) -> np.ndarray:
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    visibility = np.asarray(
        visibility,
        dtype=bool,
    )

    num_joints = keypoints.shape[0]

    heatmaps = np.zeros(
        (
            num_joints,
            heatmap_size,
            heatmap_size,
        ),
        dtype=np.float32,
    )

    scale = heatmap_size / image_size
    radius = int(3 * sigma)

    for joint_index in range(num_joints):
        if not visibility[joint_index]:
            continue

        x = keypoints[joint_index, 0] * scale
        y = keypoints[joint_index, 1] * scale

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            continue

        center_x = int(round(x))
        center_y = int(round(y))

        if not (
            0 <= center_x < heatmap_size
            and 0 <= center_y < heatmap_size
        ):
            continue

        x_min = max(
            center_x - radius,
            0,
        )

        x_max = min(
            center_x + radius + 1,
            heatmap_size,
        )

        y_min = max(
            center_y - radius,
            0,
        )

        y_max = min(
            center_y + radius + 1,
            heatmap_size,
        )

        grid_x = np.arange(
            x_min,
            x_max,
            dtype=np.float32,
        )

        grid_y = np.arange(
            y_min,
            y_max,
            dtype=np.float32,
        )

        yy, xx = np.meshgrid(
            grid_y,
            grid_x,
            indexing="ij",
        )

        gaussian = np.exp(
            -(
                (xx - x) ** 2
                + (yy - y) ** 2
            )
            / (2 * sigma ** 2)
        )

        heatmaps[
            joint_index,
            y_min:y_max,
            x_min:x_max,
        ] = gaussian

    return heatmaps


# ============================================================
# 3. Dataset
# ============================================================
class NTUFrameDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        transform=None,
        image_size=224,
        heatmap_size=56,
        sigma=2.0,
        frame_stride=1,
        single_person_only=True,
        max_samples=None,
        skeleton_cache_size=8,
        extracted_frames_dir=None,
        person_crop: bool = True,
        bbox_expansion: float = 0.25,
    ):
        super().__init__()

        self.metadata_csv = Path(metadata_csv)
        self.transform = transform
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.frame_stride = frame_stride
        self.single_person_only = single_person_only
        self.max_samples = max_samples
        self.skeleton_cache_size = skeleton_cache_size

        self.person_crop = person_crop
        self.bbox_expansion = bbox_expansion

        if self.bbox_expansion < 0:
            raise ValueError(
                "bbox_expansion must be non-negative"
            )

        if extracted_frames_dir is None:
            extracted_frames_dir = (
                NTU_RGBD_DATASET_DIR
                / "extracted_frames"
            )

        self.extracted_frames_dir = Path(
            extracted_frames_dir
        )

        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: "
                f"{self.metadata_csv}"
            )

        if not self.extracted_frames_dir.exists():
            raise FileNotFoundError(
                f"Extracted frames directory "
                f"not found: "
                f"{self.extracted_frames_dir}\n"
                "Run 05a_Extract_RGB_Frames.py first."
            )

        if frame_stride <= 0:
            raise ValueError(
                "frame_stride must be positive"
            )

        self.samples = self._load_metadata(
            single_person_only=(
                single_person_only
            ),
            max_samples=max_samples,
        )

        self.frame_index = (
            self._build_frame_index()
        )

        self._skeleton_cache = OrderedDict()

        print(
            f"NTUFrameDataset: "
            f"{len(self.samples)} videos, "
            f"{len(self.frame_index)} frames"
        )

        print(
            f"Extracted frames: "
            f"{self.extracted_frames_dir}"
        )

    def _load_metadata(
        self,
        single_person_only: bool,
        max_samples: Optional[int],
    ) -> list[dict]:
        rows = []

        with self.metadata_csv.open(
            "r",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(
                handle
            )

            for row in reader:
                if single_person_only:
                    value = str(
                        row.get(
                            "is_single_person",
                            "",
                        )
                    ).strip().lower()

                    if value not in {
                        "true",
                        "1",
                        "yes",
                    }:
                        continue

                rows.append(
                    row
                )

                if (
                    max_samples is not None
                    and len(rows) >= max_samples
                ):
                    break

        if not rows:
            raise RuntimeError(
                "No valid samples found "
                "in metadata CSV"
            )

        return rows

    def _build_frame_index(
        self,
    ) -> list[tuple[int, int]]:
        frame_index = []

        for sample_index, sample in enumerate(
            self.samples
        ):
            rgb_frames = int(
                sample["rgb_frames"]
            )

            skeleton_frames = int(
                sample["skeleton_frames"]
            )

            usable_frames = min(
                rgb_frames,
                skeleton_frames,
            )

            for frame_number in range(
                0,
                usable_frames,
                self.frame_stride,
            ):
                frame_index.append(
                    (
                        sample_index,
                        frame_number,
                    )
                )

        return frame_index

    def _load_pose_sequence(
        self,
        skeleton_path: Path,
    ) -> dict:
        cache_key = str(
            skeleton_path
        )

        if cache_key in self._skeleton_cache:
            value = self._skeleton_cache.pop(
                cache_key
            )

            self._skeleton_cache[
                cache_key
            ] = value

            return value

        sequence = read_skeleton_file(
            skeleton_path
        )

        pose_sequence = (
            extract_primary_pose_sequence(
                sequence
            )
        )

        self._skeleton_cache[
            cache_key
        ] = pose_sequence

        while (
            len(self._skeleton_cache)
            > self.skeleton_cache_size
        ):
            self._skeleton_cache.popitem(
                last=False
            )

        return pose_sequence

    def _get_frame_path(
        self,
        sample_id: str,
        frame_number: int,
    ) -> Path:
        return (
            self.extracted_frames_dir
            / sample_id
            / f"frame_{frame_number:06d}.jpg"
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.frame_index
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        sample_index, frame_number = (
            self.frame_index[index]
        )

        sample = self.samples[
            sample_index
        ]

        sample_id = str(
            sample["sample_id"]
        )

        skeleton_path = Path(
            sample["skeleton_path"]
        )

        frame_path = self._get_frame_path(
            sample_id=sample_id,
            frame_number=frame_number,
        )

        if not frame_path.exists():
            raise FileNotFoundError(
                f"Extracted frame not found: "
                f"{frame_path}"
            )

        image = cv2.imread(
            str(frame_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read extracted frame: "
                f"{frame_path}"
            )

        pose_sequence = (
            self._load_pose_sequence(
                skeleton_path
            )
        )

        keypoints = pose_sequence[
            "color_xy"
        ][frame_number].copy()

        tracking_state = pose_sequence[
            "tracking_state"
        ][frame_number].copy()

        visibility = (
            coordinate_visibility(
                keypoints,
                tracking_state=tracking_state,
                image_size=(
                    image.shape[1],
                    image.shape[0],
                ),
                include_inferred=False,
            )
        ).astype(
            np.float32
        )

        original_keypoints = keypoints.copy()
        original_visibility = visibility.copy()

        if self.person_crop:
            crop_result = crop_and_resize_person(
                image=image,
                keypoints=keypoints,
                visibility=visibility,
                output_size=self.image_size,
                expansion=self.bbox_expansion,
                make_square=True,
            )

            image = crop_result.image
            keypoints = crop_result.keypoints
            visibility = crop_result.visibility
            person_bbox = crop_result.bbox_xyxy

        else:
            person_bbox = np.array(
                [
                    0.0,
                    0.0,
                    float(image.shape[1]),
                    float(image.shape[0]),
                ],
                dtype=np.float32,
            )

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                keypoints=keypoints,
                visibility=visibility,
            )

            image_tensor = transformed[
                "image"
            ]

            keypoints_tensor = transformed[
                "keypoints"
            ]

            visibility_tensor = transformed[
                "visibility"
            ]

        else:
            original_height, original_width = (
                image.shape[:2]
            )

            image = cv2.resize(
                image,
                (
                    self.image_size,
                    self.image_size,
                ),
            )

            keypoints[:, 0] *= (
                self.image_size
                / original_width
            )

            keypoints[:, 1] *= (
                self.image_size
                / original_height
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            image_tensor = (
                torch.from_numpy(
                    np.transpose(
                        image,
                        (
                            2,
                            0,
                            1,
                        ),
                    )
                )
                .float()
                / 255.0
            )

            keypoints_tensor = (
                torch.from_numpy(
                    keypoints
                ).float()
            )

            visibility_tensor = (
                torch.from_numpy(
                    visibility
                ).float()
            )

        heatmaps = generate_gaussian_heatmaps(
            keypoints=(
                keypoints_tensor
                .detach()
                .cpu()
                .numpy()
            ),
            visibility=(
                visibility_tensor
                .detach()
                .cpu()
                .numpy()
            ),
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
        )

        return {
            "image": image_tensor,

            "heatmaps": torch.from_numpy(
                heatmaps
            ).float(),

            "keypoints": keypoints_tensor,

            "visibility": visibility_tensor,

            "sample_id": sample_id,

            "frame_index": frame_number,

            "rgb_path": str(
                frame_path
            ),

            "person_bbox": torch.from_numpy(
                person_bbox
            ).float(),

            "original_keypoints": torch.from_numpy(
                original_keypoints
            ).float(),

            "original_visibility": torch.from_numpy(
                original_visibility
            ).float(),
        }