# Scripts/common/metrics.py

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


EPSILON = 1e-8


# ============================================================
# 1. Internal helpers
# ============================================================

def _as_float_array(
    array: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        array,
        dtype=np.float64,
    )


def _prepare_mask(
    mask: Optional[np.ndarray],
    target_shape: tuple,
) -> np.ndarray:
    """
    Prepare a boolean mask matching all dimensions except the
    coordinate dimension.
    """
    if mask is None:
        return np.ones(
            target_shape,
            dtype=bool,
        )

    mask = np.asarray(mask)

    try:
        mask = np.broadcast_to(
            mask,
            target_shape,
        )
    except ValueError as exc:
        raise ValueError(
            f"Mask shape {mask.shape} cannot be broadcast "
            f"to target shape {target_shape}"
        ) from exc

    return mask > 0


def euclidean_distance(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distance over the final coordinate axis.
    """
    prediction = _as_float_array(prediction)
    target = _as_float_array(target)

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape"
        )

    return np.linalg.norm(
        prediction - target,
        axis=-1,
    )


# ============================================================
# 2. PCK
# ============================================================

def compute_pck(
    prediction: np.ndarray,
    target: np.ndarray,
    normalization_length: np.ndarray | float,
    threshold: float = 0.2,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Percentage of Correct Keypoints.

    A joint is correct when:

        distance <= threshold * normalization_length

    Parameters
    ----------
    prediction:
        Shape [..., J, 2].
    target:
        Shape [..., J, 2].
    normalization_length:
        Scalar or array broadcastable to distance shape.
    threshold:
        PCK threshold.
    visibility:
        Optional mask with shape [..., J].

    Returns
    -------
    float
        PCK in percentage form, between 0 and 100.
    """
    if threshold <= 0:
        raise ValueError(
            "threshold must be positive"
        )

    distances = euclidean_distance(
        prediction,
        target,
    )

    normalization_length = _as_float_array(
        normalization_length
    )

    try:
        normalization_length = np.broadcast_to(
            normalization_length,
            distances.shape,
        )
    except ValueError as exc:
        # Common case: one normalization value per sample.
        if (
            normalization_length.ndim
            == distances.ndim - 1
        ):
            normalization_length = np.expand_dims(
                normalization_length,
                axis=-1,
            )

            normalization_length = np.broadcast_to(
                normalization_length,
                distances.shape,
            )
        else:
            raise ValueError(
                "normalization_length cannot be broadcast "
                f"to distance shape {distances.shape}"
            ) from exc

    valid = _prepare_mask(
        visibility,
        distances.shape,
    )

    valid &= np.isfinite(distances)
    valid &= np.isfinite(normalization_length)
    valid &= normalization_length > EPSILON

    if not np.any(valid):
        return float("nan")

    normalized_distance = (
        distances[valid]
        / normalization_length[valid]
    )

    correct = normalized_distance <= threshold

    return float(
        np.mean(correct) * 100.0
    )


def compute_pck_per_joint(
    prediction: np.ndarray,
    target: np.ndarray,
    normalization_length: np.ndarray | float,
    threshold: float = 0.2,
    visibility: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute PCK separately for every joint.

    Expected shape:
        [N, J, 2]
    """
    prediction = _as_float_array(prediction)
    target = _as_float_array(target)

    if prediction.ndim != 3:
        raise ValueError(
            "prediction must have shape [N, J, 2]"
        )

    num_joints = prediction.shape[1]

    results = np.full(
        num_joints,
        np.nan,
        dtype=np.float64,
    )

    if visibility is not None:
        visibility = np.asarray(visibility)

    for joint_index in range(num_joints):
        joint_visibility = None

        if visibility is not None:
            joint_visibility = visibility[:, joint_index]

        results[joint_index] = compute_pck(
            prediction=prediction[:, joint_index],
            target=target[:, joint_index],
            normalization_length=normalization_length,
            threshold=threshold,
            visibility=joint_visibility,
        )

    return results


# ============================================================
# 3. MPJPE
# ============================================================

def compute_mpjpe(
    prediction: np.ndarray,
    target: np.ndarray,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Mean Per Joint Position Error.

    Supports 2D or 3D coordinates, though MPJPE is normally used
    for 3D pose.

    Returns the value in the same unit as the input coordinates.
    """
    distances = euclidean_distance(
        prediction,
        target,
    )

    valid = _prepare_mask(
        visibility,
        distances.shape,
    )

    valid &= np.isfinite(distances)

    if not np.any(valid):
        return float("nan")

    return float(
        np.mean(distances[valid])
    )


def compute_mpjpe_per_joint(
    prediction: np.ndarray,
    target: np.ndarray,
    visibility: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute MPJPE separately for every joint.

    Expected shape:
        [N, J, C]
    """
    prediction = _as_float_array(prediction)
    target = _as_float_array(target)

    if prediction.ndim != 3:
        raise ValueError(
            "prediction must have shape [N, J, C]"
        )

    num_joints = prediction.shape[1]

    results = np.full(
        num_joints,
        np.nan,
        dtype=np.float64,
    )

    for joint_index in range(num_joints):
        joint_visibility = None

        if visibility is not None:
            joint_visibility = np.asarray(
                visibility
            )[:, joint_index]

        results[joint_index] = compute_mpjpe(
            prediction[:, joint_index],
            target[:, joint_index],
            visibility=joint_visibility,
        )

    return results


# ============================================================
# 4. Temporal derivatives
# ============================================================

def compute_velocity(
    sequence: np.ndarray,
    fps: float = 1.0,
) -> np.ndarray:
    """
    Compute first temporal derivative.

    Parameters
    ----------
    sequence:
        Shape [T, J, C].
    fps:
        Frames per second.

    Returns
    -------
    np.ndarray
        Shape [T - 1, J, C].
    """
    sequence = _as_float_array(sequence)

    if sequence.ndim < 2:
        raise ValueError(
            "sequence must include a time dimension"
        )

    if len(sequence) < 2:
        shape = (
            0,
            *sequence.shape[1:],
        )
        return np.empty(
            shape,
            dtype=np.float64,
        )

    if fps <= 0:
        raise ValueError(
            "fps must be positive"
        )

    return np.diff(
        sequence,
        axis=0,
    ) * fps


def compute_acceleration(
    sequence: np.ndarray,
    fps: float = 1.0,
) -> np.ndarray:
    """
    Compute second temporal derivative.

    Returns shape [T - 2, J, C].
    """
    velocity = compute_velocity(
        sequence,
        fps=fps,
    )

    if len(velocity) < 2:
        shape = (
            0,
            *sequence.shape[1:],
        )
        return np.empty(
            shape,
            dtype=np.float64,
        )

    return np.diff(
        velocity,
        axis=0,
    ) * fps


def compute_jerk(
    sequence: np.ndarray,
    fps: float = 1.0,
) -> np.ndarray:
    """
    Compute third temporal derivative.

    Returns shape [T - 3, J, C].
    """
    acceleration = compute_acceleration(
        sequence,
        fps=fps,
    )

    if len(acceleration) < 2:
        shape = (
            0,
            *sequence.shape[1:],
        )
        return np.empty(
            shape,
            dtype=np.float64,
        )

    return np.diff(
        acceleration,
        axis=0,
    ) * fps


# ============================================================
# 5. Temporal stability metrics
# ============================================================

def compute_velocity_error(
    prediction: np.ndarray,
    target: np.ndarray,
    fps: float = 1.0,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean velocity error between predicted and GT poses.
    """
    pred_velocity = compute_velocity(
        prediction,
        fps=fps,
    )

    target_velocity = compute_velocity(
        target,
        fps=fps,
    )

    velocity_visibility = None

    if visibility is not None:
        visibility = np.asarray(visibility)

        velocity_visibility = (
            visibility[:-1]
            & visibility[1:]
        )

    return compute_mpjpe(
        pred_velocity,
        target_velocity,
        visibility=velocity_visibility,
    )


def compute_acceleration_error(
    prediction: np.ndarray,
    target: np.ndarray,
    fps: float = 1.0,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean acceleration error between predicted and GT poses.
    """
    pred_acceleration = compute_acceleration(
        prediction,
        fps=fps,
    )

    target_acceleration = compute_acceleration(
        target,
        fps=fps,
    )

    acceleration_visibility = None

    if visibility is not None:
        visibility = np.asarray(visibility)

        acceleration_visibility = (
            visibility[:-2]
            & visibility[1:-1]
            & visibility[2:]
        )

    return compute_mpjpe(
        pred_acceleration,
        target_acceleration,
        visibility=acceleration_visibility,
    )


def compute_acceleration_jitter(
    sequence: np.ndarray,
    fps: float = 1.0,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Measure the average magnitude of predicted acceleration.

    A lower value generally indicates smoother motion, but it must
    not be interpreted alone because real fast actions naturally
    have larger acceleration.
    """
    acceleration = compute_acceleration(
        sequence,
        fps=fps,
    )

    if len(acceleration) == 0:
        return float("nan")

    magnitude = np.linalg.norm(
        acceleration,
        axis=-1,
    )

    valid = np.isfinite(magnitude)

    if visibility is not None:
        visibility = np.asarray(visibility)

        acceleration_visibility = (
            visibility[:-2]
            & visibility[1:-1]
            & visibility[2:]
        )

        valid &= acceleration_visibility

    if not np.any(valid):
        return float("nan")

    return float(
        np.mean(magnitude[valid])
    )


def compute_jerk_jitter(
    sequence: np.ndarray,
    fps: float = 1.0,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Measure mean jerk magnitude.
    """
    jerk = compute_jerk(
        sequence,
        fps=fps,
    )

    if len(jerk) == 0:
        return float("nan")

    magnitude = np.linalg.norm(
        jerk,
        axis=-1,
    )

    valid = np.isfinite(magnitude)

    if visibility is not None:
        visibility = np.asarray(visibility)

        jerk_visibility = (
            visibility[:-3]
            & visibility[1:-2]
            & visibility[2:-1]
            & visibility[3:]
        )

        valid &= jerk_visibility

    if not np.any(valid):
        return float("nan")

    return float(
        np.mean(magnitude[valid])
    )


# ============================================================
# 6. Frame-to-frame displacement
# ============================================================

def compute_mean_frame_displacement(
    sequence: np.ndarray,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean joint displacement between consecutive frames.

    Unlike physical velocity, this does not multiply by FPS.
    """
    sequence = _as_float_array(sequence)

    displacement = np.diff(
        sequence,
        axis=0,
    )

    if len(displacement) == 0:
        return float("nan")

    magnitude = np.linalg.norm(
        displacement,
        axis=-1,
    )

    valid = np.isfinite(magnitude)

    if visibility is not None:
        visibility = np.asarray(visibility)

        valid &= (
            visibility[:-1]
            & visibility[1:]
        )

    if not np.any(valid):
        return float("nan")

    return float(
        np.mean(magnitude[valid])
    )


# ============================================================
# 7. Metric summary
# ============================================================

def summarize_pose_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    normalization_length: Optional[np.ndarray] = None,
    visibility: Optional[np.ndarray] = None,
    pck_thresholds: Sequence[float] = (
        0.05,
        0.10,
        0.20,
        0.50,
    ),
) -> dict:
    """
    Compute a compact dictionary of pose metrics.
    """
    metrics = {
        "mpjpe": compute_mpjpe(
            prediction,
            target,
            visibility=visibility,
        )
    }

    if normalization_length is not None:
        for threshold in pck_thresholds:
            metric_name = f"pck_{threshold:.2f}"

            metrics[metric_name] = compute_pck(
                prediction=prediction,
                target=target,
                normalization_length=normalization_length,
                threshold=threshold,
                visibility=visibility,
            )

    return metrics