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
        validate_pose_sequences: bool = True,
    ):
        super().__init__()

        self.metadata_csv = Path(
            metadata_csv
        )

        self.transform = transform
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.frame_stride = frame_stride
        self.single_person_only = (
            single_person_only
        )

        self.max_samples = max_samples
        self.skeleton_cache_size = (
            skeleton_cache_size
        )

        self.person_crop = person_crop
        self.bbox_expansion = (
            bbox_expansion
        )

        self.validate_pose_sequences = (
            validate_pose_sequences
        )

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
                "Extracted frames directory "
                f"not found: "
                f"{self.extracted_frames_dir}\n"
                "Run 05a_Extract_RGB_Frames.py first."
            )

        if frame_stride <= 0:
            raise ValueError(
                "frame_stride must be positive"
            )

        if skeleton_cache_size <= 0:
            raise ValueError(
                "skeleton_cache_size must be positive"
            )

        # Skeleton cache is created before metadata validation.
        # Validated pose sequences can therefore be reused later.
        self._skeleton_cache = OrderedDict()

        (
            self.samples,
            self.skipped_samples,
        ) = self._load_metadata(
            single_person_only=(
                single_person_only
            ),
            max_samples=max_samples,
        )

        self.frame_index = (
            self._build_frame_index()
        )

        if not self.frame_index:
            raise RuntimeError(
                "No valid frames remained after "
                "dataset filtering."
            )

        print()
        print("=" * 70)
        print("NTUFrameDataset")
        print("=" * 70)

        print(
            f"Metadata CSV:        "
            f"{self.metadata_csv}"
        )

        print(
            f"Valid videos:        "
            f"{len(self.samples)}"
        )

        print(
            f"Skipped videos:      "
            f"{len(self.skipped_samples)}"
        )

        print(
            f"Usable frames:       "
            f"{len(self.frame_index)}"
        )

        print(
            f"Extracted frames:    "
            f"{self.extracted_frames_dir}"
        )

        print(
            f"Person crop:         "
            f"{self.person_crop}"
        )

        print(
            f"Validate skeletons:  "
            f"{self.validate_pose_sequences}"
        )

        if self.skipped_samples:
            print()
            print("Skipped sample examples:")

            for item in self.skipped_samples[:10]:
                print(
                    f"  {item['sample_id']} | "
                    f"{item['reason']}"
                )

            if len(self.skipped_samples) > 10:
                print(
                    f"  ... and "
                    f"{len(self.skipped_samples) - 10} "
                    f"more"
                )

        print("=" * 70)
        print()

    # ========================================================
    # 4. Metadata loading and validation
    # ========================================================

    @staticmethod
    def _string_to_bool(
        value,
    ) -> bool:
        return str(
            value
        ).strip().lower() in {
            "true",
            "1",
            "yes",
        }

    @staticmethod
    def _safe_int(
        value,
        default: int = 0,
    ) -> int:
        try:
            return int(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            return default

    def _metadata_pose_is_empty(
        self,
        row: dict,
    ) -> tuple[bool, str]:
        """
        Fast metadata-level rejection.

        Returns:
            is_empty
            rejection reason
        """
        max_bodies = self._safe_int(
            row.get(
                "max_bodies",
                0,
            )
        )

        skeleton_frames = self._safe_int(
            row.get(
                "skeleton_frames",
                0,
            )
        )

        empty_frames = self._safe_int(
            row.get(
                "empty_frames",
                0,
            )
        )

        if skeleton_frames <= 0:
            return (
                True,
                "skeleton_frames <= 0",
            )

        if max_bodies <= 0:
            return (
                True,
                "max_bodies <= 0",
            )

        if empty_frames >= skeleton_frames:
            return (
                True,
                "all skeleton frames are empty",
            )

        return (
            False,
            "",
        )

    def _validate_pose_sequence(
        self,
        skeleton_path: Path,
    ) -> tuple[bool, str]:
        """
        Strictly verify that the skeleton file contains a
        primary pose sequence that the Dataset can use.

        Validation is performed once per video during Dataset
        initialisation, not once per frame.
        """
        if not skeleton_path.exists():
            return (
                False,
                f"skeleton file not found: "
                f"{skeleton_path}",
            )

        try:
            pose_sequence = (
                self._load_pose_sequence(
                    skeleton_path
                )
            )

        except Exception as error:
            return (
                False,
                f"{type(error).__name__}: "
                f"{error}",
            )

        if "color_xy" not in pose_sequence:
            return (
                False,
                "pose sequence has no color_xy",
            )

        if "tracking_state" not in pose_sequence:
            return (
                False,
                "pose sequence has no tracking_state",
            )

        color_xy = np.asarray(
            pose_sequence["color_xy"]
        )

        tracking_state = np.asarray(
            pose_sequence["tracking_state"]
        )

        if color_xy.ndim != 3:
            return (
                False,
                "color_xy has invalid shape: "
                f"{color_xy.shape}",
            )

        if tracking_state.ndim != 2:
            return (
                False,
                "tracking_state has invalid shape: "
                f"{tracking_state.shape}",
            )

        if color_xy.shape[0] <= 0:
            return (
                False,
                "pose sequence contains no frames",
            )

        if (
            tracking_state.shape[0]
            != color_xy.shape[0]
        ):
            return (
                False,
                "pose and tracking frame counts differ",
            )

        if not np.isfinite(
            color_xy
        ).any():
            return (
                False,
                "all pose coordinates are non-finite",
            )

        return (
            True,
            "",
        )

    def _load_metadata(
        self,
        single_person_only: bool,
        max_samples: Optional[int],
    ) -> tuple[
        list[dict],
        list[dict[str, str]],
    ]:
        valid_rows: list[dict] = []
        skipped_rows: list[
            dict[str, str]
        ] = []

        with self.metadata_csv.open(
            "r",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(
                handle
            )

            for row in reader:
                sample_id = str(
                    row.get(
                        "sample_id",
                        "",
                    )
                ).strip()

                if not sample_id:
                    skipped_rows.append(
                        {
                            "sample_id": "<missing>",
                            "reason": (
                                "missing sample_id"
                            ),
                        }
                    )
                    continue

                if single_person_only:
                    is_single_person = (
                        self._string_to_bool(
                            row.get(
                                "is_single_person",
                                "",
                            )
                        )
                    )

                    if not is_single_person:
                        skipped_rows.append(
                            {
                                "sample_id": (
                                    sample_id
                                ),
                                "reason": (
                                    "not single-person"
                                ),
                            }
                        )
                        continue

                is_empty, empty_reason = (
                    self._metadata_pose_is_empty(
                        row
                    )
                )

                if is_empty:
                    skipped_rows.append(
                        {
                            "sample_id": (
                                sample_id
                            ),
                            "reason": (
                                empty_reason
                            ),
                        }
                    )
                    continue

                skeleton_path = Path(
                    str(
                        row.get(
                            "skeleton_path",
                            "",
                        )
                    )
                )

                if self.validate_pose_sequences:
                    (
                        is_valid,
                        validation_reason,
                    ) = self._validate_pose_sequence(
                        skeleton_path
                    )

                    if not is_valid:
                        skipped_rows.append(
                            {
                                "sample_id": (
                                    sample_id
                                ),
                                "reason": (
                                    validation_reason
                                ),
                            }
                        )
                        continue

                valid_rows.append(
                    row
                )

                if (
                    max_samples is not None
                    and len(valid_rows)
                    >= max_samples
                ):
                    break

        if not valid_rows:
            raise RuntimeError(
                "No valid samples remained after "
                "metadata and pose validation."
            )

        return (
            valid_rows,
            skipped_rows,
        )

    # ========================================================
    # 5. Frame index
    # ========================================================

    def _build_frame_index(
        self,
    ) -> list[tuple[int, int]]:
        frame_index: list[
            tuple[int, int]
        ] = []

        valid_samples: list[dict] = []
        skipped_missing_frames: list[
            dict[str, str]
        ] = []

        for sample in self.samples:
            sample_id = str(
                sample["sample_id"]
            )

            rgb_frames = self._safe_int(
                sample.get(
                    "rgb_frames",
                    0,
                )
            )

            skeleton_frames = (
                self._safe_int(
                    sample.get(
                        "skeleton_frames",
                        0,
                    )
                )
            )

            usable_frames = min(
                rgb_frames,
                skeleton_frames,
            )

            if usable_frames <= 0:
                skipped_missing_frames.append(
                    {
                        "sample_id": sample_id,
                        "reason": (
                            "usable frame count <= 0"
                        ),
                    }
                )
                continue

            # Verify that the sample frame directory exists.
            sample_frame_dir = (
                self.extracted_frames_dir
                / sample_id
            )

            if not sample_frame_dir.exists():
                skipped_missing_frames.append(
                    {
                        "sample_id": sample_id,
                        "reason": (
                            "extracted frame directory "
                            "not found"
                        ),
                    }
                )
                continue

            new_sample_index = len(
                valid_samples
            )

            sample_frame_indices = []

            for frame_number in range(
                0,
                usable_frames,
                self.frame_stride,
            ):
                frame_path = (
                    sample_frame_dir
                    / (
                        f"frame_"
                        f"{frame_number:06d}.jpg"
                    )
                )

                # Missing individual frames are skipped.
                # They no longer terminate training.
                if not frame_path.exists():
                    continue

                sample_frame_indices.append(
                    (
                        new_sample_index,
                        frame_number,
                    )
                )

            if not sample_frame_indices:
                skipped_missing_frames.append(
                    {
                        "sample_id": sample_id,
                        "reason": (
                            "no extracted frames found"
                        ),
                    }
                )
                continue

            valid_samples.append(
                sample
            )

            frame_index.extend(
                sample_frame_indices
            )

        self.samples = valid_samples

        self.skipped_samples.extend(
            skipped_missing_frames
        )

        return frame_index

    # ========================================================
    # 6. Skeleton loading and cache
    # ========================================================

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

    # ========================================================
    # 7. Frame path
    # ========================================================

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

    # ========================================================
    # 8. Dataset interface
    # ========================================================

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

        # This should not normally happen because missing
        # frames were removed in _build_frame_index().
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
                "Could not read extracted frame: "
                f"{frame_path}"
            )

        pose_sequence = (
            self._load_pose_sequence(
                skeleton_path
            )
        )

        if (
            frame_number
            >= len(
                pose_sequence["color_xy"]
            )
        ):
            raise IndexError(
                f"Pose frame index out of range: "
                f"sample={sample_id}, "
                f"frame={frame_number}"
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

        original_keypoints = (
            keypoints.copy()
        )

        original_visibility = (
            visibility.copy()
        )

        if self.person_crop:
            crop_result = (
                crop_and_resize_person(
                    image=image,
                    keypoints=keypoints,
                    visibility=visibility,
                    output_size=(
                        self.image_size
                    ),
                    expansion=(
                        self.bbox_expansion
                    ),
                    make_square=True,
                )
            )

            image = crop_result.image
            keypoints = crop_result.keypoints
            visibility = crop_result.visibility
            person_bbox = (
                crop_result.bbox_xyxy
            )

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