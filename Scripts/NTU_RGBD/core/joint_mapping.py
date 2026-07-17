# Scripts/NTU_RGBD/core/joint_mapping.py

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import (
    NTU_JOINT_INDEX,
    NTU_JOINT_NAMES,
    PENN_13_FROM_NTU_INDICES,
    PENN_13_JOINT_NAMES,
)


# ============================================================
# 1. Generic joint selection
# ============================================================

def select_joints(
    keypoints: np.ndarray,
    joint_indices: Sequence[int],
) -> np.ndarray:
    """
    Select joints from an array whose second-last dimension
    represents joints.

    Supported examples:

        [J, 2]
        [J, 3]
        [T, J, 2]
        [N, T, J, 3]
    """
    keypoints = np.asarray(
        keypoints
    )

    if keypoints.ndim < 2:
        raise ValueError(
            "keypoints must contain a joint dimension"
        )

    num_joints = (
        keypoints.shape[-2]
    )

    indices = np.asarray(
        joint_indices,
        dtype=np.int64,
    )

    if indices.ndim != 1:
        raise ValueError(
            "joint_indices must be one-dimensional"
        )

    if np.any(indices < 0):
        raise IndexError(
            "joint indices cannot be negative"
        )

    if np.any(indices >= num_joints):
        raise IndexError(
            f"Joint index exceeds available "
            f"joint count {num_joints}"
        )

    return np.take(
        keypoints,
        indices,
        axis=-2,
    )


# ============================================================
# 2. Select joints using names
# ============================================================

def joint_names_to_indices(
    joint_names: Sequence[str],
) -> tuple[int, ...]:
    """
    Convert NTU joint names into indices.
    """
    indices: list[int] = []

    for joint_name in joint_names:
        normalized_name = (
            joint_name.strip().lower()
        )

        if (
            normalized_name
            not in NTU_JOINT_INDEX
        ):
            raise KeyError(
                f"Unknown NTU joint name: "
                f"{joint_name}"
            )

        indices.append(
            NTU_JOINT_INDEX[
                normalized_name
            ]
        )

    return tuple(indices)


def select_joints_by_name(
    keypoints: np.ndarray,
    joint_names: Sequence[str],
) -> np.ndarray:
    """
    Select NTU joints by joint names.
    """
    indices = joint_names_to_indices(
        joint_names
    )

    return select_joints(
        keypoints,
        indices,
    )


# ============================================================
# 3. Convert NTU25 to Penn-compatible 13 joints
# ============================================================

def convert_ntu25_to_penn13(
    keypoints: np.ndarray,
) -> np.ndarray:
    """
    Convert NTU's 25-joint pose into the 13-joint subset
    used by the current Penn Action pipeline.

    Input examples:
        [25, 2]
        [25, 3]
        [T, 25, 2]
        [T, 25, 3]

    Output:
        [..., 13, C]
    """
    keypoints = np.asarray(
        keypoints
    )

    if keypoints.shape[-2] != 25:
        raise ValueError(
            f"Expected 25 NTU joints, got "
            f"{keypoints.shape[-2]}"
        )

    return select_joints(
        keypoints,
        PENN_13_FROM_NTU_INDICES,
    )


def convert_ntu_visibility_to_penn13(
    visibility: np.ndarray,
) -> np.ndarray:
    """
    Convert NTU joint visibility from 25 to 13 joints.

    Input examples:
        [25]
        [T, 25]

    Output:
        [..., 13]
    """
    visibility = np.asarray(
        visibility
    )

    if visibility.shape[-1] != 25:
        raise ValueError(
            f"Expected visibility for 25 joints, "
            f"got {visibility.shape[-1]}"
        )

    return np.take(
        visibility,
        PENN_13_FROM_NTU_INDICES,
        axis=-1,
    )


# ============================================================
# 4. Joint-name utility
# ============================================================

def get_ntu_joint_name(
    joint_index: int,
) -> str:
    """
    Return the name of one NTU joint.
    """
    if not (
        0 <= joint_index
        < len(NTU_JOINT_NAMES)
    ):
        raise IndexError(
            f"Invalid NTU joint index: "
            f"{joint_index}"
        )

    return NTU_JOINT_NAMES[
        joint_index
    ]


def get_penn13_joint_name(
    joint_index: int,
) -> str:
    """
    Return the name of one Penn-compatible joint.
    """
    if not (
        0 <= joint_index
        < len(PENN_13_JOINT_NAMES)
    ):
        raise IndexError(
            f"Invalid Penn 13 joint index: "
            f"{joint_index}"
        )

    return PENN_13_JOINT_NAMES[
        joint_index
    ]