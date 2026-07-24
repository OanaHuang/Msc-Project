# Scripts/NTU_RGBD/evaluation/metric_config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ============================================================
# 1. NTU RGB+D joint configuration
# ============================================================

NTU_NECK_INDEX = 2
NTU_HEAD_INDEX = 3

NTU_JOINT_NAMES = [
    "spine_base",
    "spine_mid",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "spine_shoulder",
    "left_hand_tip",
    "left_thumb",
    "right_hand_tip",
    "right_thumb",
]


# ============================================================
# 2. Metric descriptions
# ============================================================

DEFAULT_PCK_SCALE_DESCRIPTION = (
    "GT visible-joint bounding-box maximum side length"
)

DEFAULT_HEAD_SCALE_DESCRIPTION = (
    "MPII-calibrated 2D GT head-to-neck joint distance"
)


# ============================================================
# 3. Evaluation configuration
# ============================================================

@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for evaluating one model's NPZ predictions.

    Parameters
    ----------
    model_version:
        Model identifier, such as "06" or "12".

    npz_dir:
        Directory containing prediction NPZ files.

    output_dir:
        Directory used to save summary, per-video and per-joint CSVs.

    filename_pattern:
        Preferred NPZ filename pattern.

    pck_threshold:
        Threshold for the original PCK metric.

    pckh_threshold:
        Threshold for the approximate MPII-style PCKh metric.

    mpii_head_scale_factor:
        Calibration factor applied to the NTU Head-to-Neck distance.

    head_index:
        NTU Head joint index.

    neck_index:
        NTU Neck joint index.

    scale_epsilon:
        Minimum valid normalization scale.
    """

    model_version: str

    npz_dir: Path
    output_dir: Path

    filename_pattern: str

    pck_threshold: float = 0.10
    pckh_threshold: float = 0.50

    mpii_head_scale_factor: float = 1.8

    head_index: int = NTU_HEAD_INDEX
    neck_index: int = NTU_NECK_INDEX

    scale_epsilon: float = 1e-6

    pck_scale_description: str = (
        DEFAULT_PCK_SCALE_DESCRIPTION
    )

    head_scale_description: str = (
        DEFAULT_HEAD_SCALE_DESCRIPTION
    )

    joint_names: tuple[str, ...] = tuple(
        NTU_JOINT_NAMES
    )

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "npz_dir",
            Path(self.npz_dir),
        )

        object.__setattr__(
            self,
            "output_dir",
            Path(self.output_dir),
        )

        if not self.model_version:
            raise ValueError(
                "model_version cannot be empty"
            )

        if not self.filename_pattern:
            raise ValueError(
                "filename_pattern cannot be empty"
            )

        if self.pck_threshold <= 0:
            raise ValueError(
                "pck_threshold must be positive"
            )

        if self.pckh_threshold <= 0:
            raise ValueError(
                "pckh_threshold must be positive"
            )

        if self.mpii_head_scale_factor <= 0:
            raise ValueError(
                "mpii_head_scale_factor must be positive"
            )

        if self.scale_epsilon <= 0:
            raise ValueError(
                "scale_epsilon must be positive"
            )

        if self.head_index < 0:
            raise ValueError(
                "head_index must be non-negative"
            )

        if self.neck_index < 0:
            raise ValueError(
                "neck_index must be non-negative"
            )

        if self.head_index == self.neck_index:
            raise ValueError(
                "head_index and neck_index must be different"
            )

        if len(self.joint_names) == 0:
            raise ValueError(
                "joint_names cannot be empty"
            )

    @property
    def summary_csv_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / (
                f"model{self.model_version}"
                "_metrics_summary.csv"
            )
        )

    @property
    def per_video_csv_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / (
                f"model{self.model_version}"
                "_metrics_per_video.csv"
            )
        )

    @property
    def per_joint_csv_path(
        self,
    ) -> Path:
        return (
            self.output_dir
            / (
                f"model{self.model_version}"
                "_metrics_per_joint.csv"
            )
        )

    def create_output_dir(
        self,
    ) -> None:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )