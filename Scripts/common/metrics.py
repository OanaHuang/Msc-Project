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
    """
    Convert input to a NumPy float64 array.
    """
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

    Parameters
    ----------
    mask:
        Optional visibility or validity mask.
    target_shape:
        Desired output shape.

    Returns
    -------
    np.ndarray
        Boolean mask with shape target_shape.
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


def _broadcast_normalization_length(
    normalization_length: np.ndarray | float,
    target_shape: tuple,
) -> np.ndarray:
    """
    Broadcast a normalization length to the distance-array shape.

    Common supported inputs include:

        scalar
        [N]
        [N, 1]
        [N, J]

    Parameters
    ----------
    normalization_length:
        Scalar or array containing one or more normalization values.
    target_shape:
        Shape of the distance array, usually [N, J].

    Returns
    -------
    np.ndarray
        Broadcast normalization array with shape target_shape.
    """
    normalization_length = _as_float_array(
        normalization_length
    )

    try:
        return np.broadcast_to(
            normalization_length,
            target_shape,
        )
    except ValueError as original_exc:
        # Common case:
        # normalization_length shape [N]
        # target shape [N, J]
        if (
            normalization_length.ndim
            == len(target_shape) - 1
        ):
            expanded = np.expand_dims(
                normalization_length,
                axis=-1,
            )

            try:
                return np.broadcast_to(
                    expanded,
                    target_shape,
                )
            except ValueError as exc:
                raise ValueError(
                    "normalization_length with shape "
                    f"{normalization_length.shape} cannot be "
                    f"broadcast to target shape {target_shape}"
                ) from exc

        raise ValueError(
            "normalization_length with shape "
            f"{normalization_length.shape} cannot be broadcast "
            f"to target shape {target_shape}"
        ) from original_exc


def euclidean_distance(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distance over the final coordinate axis.

    Examples
    --------
    Input shape:
        [N, J, 2]

    Output shape:
        [N, J]
    """
    prediction = _as_float_array(prediction)
    target = _as_float_array(target)

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape, "
            f"but received {prediction.shape} and {target.shape}"
        )

    if prediction.shape[-1] not in (2, 3):
        raise ValueError(
            "The final dimension should normally contain "
            "2D or 3D coordinates"
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

    A joint is counted as correct when:

        Euclidean distance <= threshold * normalization_length

    This is a generic normalized keypoint metric. The meaning of
    the metric depends on the supplied normalization length.

    Examples
    --------
    Bounding-box normalization:
        normalization_length = person bounding-box size

    Torso normalization:
        normalization_length = torso length

    Head normalization:
        normalization_length = head length

    Parameters
    ----------
    prediction:
        Predicted coordinates with shape [..., J, C].
    target:
        Ground-truth coordinates with shape [..., J, C].
    normalization_length:
        Scalar or array broadcastable to the distance shape.
    threshold:
        PCK threshold.
    visibility:
        Optional mask with shape [..., J].

    Returns
    -------
    float
        PCK percentage between 0 and 100.
    """
    if threshold <= 0:
        raise ValueError(
            "threshold must be positive"
        )

    distances = euclidean_distance(
        prediction,
        target,
    )

    normalization_length = _broadcast_normalization_length(
        normalization_length,
        distances.shape,
    )

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

    Expected input shape:
        prediction: [N, J, C]
        target:     [N, J, C]
        visibility: [N, J]

    Returns
    -------
    np.ndarray
        Shape [J], containing PCK percentages.
    """
    prediction = _as_float_array(prediction)
    target = _as_float_array(target)

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape"
        )

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

    if visibility is not None:
        visibility = np.asarray(visibility)

        if visibility.shape != prediction.shape[:-1]:
            try:
                visibility = np.broadcast_to(
                    visibility,
                    prediction.shape[:-1],
                )
            except ValueError as exc:
                raise ValueError(
                    f"visibility shape {visibility.shape} cannot "
                    f"match joint shape {prediction.shape[:-1]}"
                ) from exc

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
# 3. PCKh
# ============================================================

def compute_pckh(
    prediction: np.ndarray,
    target: np.ndarray,
    head_length: np.ndarray | float,
    threshold: float = 0.5,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Percentage of Correct Keypoints normalized by head size.

    A joint is counted as correct when:

        Euclidean distance <= threshold * head_length

    For PCKh@0.5:

        threshold = 0.5

    Parameters
    ----------
    prediction:
        Predicted coordinates with shape [..., J, C].
    target:
        Ground-truth coordinates with shape [..., J, C].
    head_length:
        Head normalization length.

        Common shapes:
            scalar
            [N]
            [N, 1]
            [N, J]

        For video data, this may also be one value per frame.
    threshold:
        PCKh threshold. Default is 0.5.
    visibility:
        Optional mask with shape [..., J].

    Returns
    -------
    float
        PCKh percentage between 0 and 100.

    Notes
    -----
    The exact interpretation of PCKh depends on how head_length
    is defined.

    MPII-style PCKh normally uses a head bounding-box-derived
    head size. If head-to-neck joint distance is used instead,
    this should be documented clearly in the experiment report.
    """
    return compute_pck(
        prediction=prediction,
        target=target,
        normalization_length=head_length,
        threshold=threshold,
        visibility=visibility,
    )


def compute_pckh_per_joint(
    prediction: np.ndarray,
    target: np.ndarray,
    head_length: np.ndarray | float,
    threshold: float = 0.5,
    visibility: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute PCKh separately for every joint.

    Expected input shape:
        prediction: [N, J, C]
        target:     [N, J, C]
        visibility: [N, J]
        head_length:[N], [N, 1], or scalar

    Returns
    -------
    np.ndarray
        Shape [J], containing per-joint PCKh percentages.
    """
    return compute_pck_per_joint(
        prediction=prediction,
        target=target,
        normalization_length=head_length,
        threshold=threshold,
        visibility=visibility,
    )


# ============================================================
# 4. MPJPE
# ============================================================

def compute_mpjpe(
    prediction: np.ndarray,
    target: np.ndarray,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Mean Per Joint Position Error.

    Supports 2D or 3D coordinates, although MPJPE is commonly
    used for 3D pose estimation.

    Parameters
    ----------
    prediction:
        Predicted coordinates.
    target:
        Ground-truth coordinates.
    visibility:
        Optional joint validity mask.

    Returns
    -------
    float
        Mean error in the same unit as the input coordinates.
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

    Returns
    -------
    np.ndarray
        Shape [J].
    """
    prediction = _as_float_array(prediction)
    target = _as_float_array(target)

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape"
        )

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

    if visibility is not None:
        visibility = np.asarray(visibility)

    for joint_index in range(num_joints):
        joint_visibility = None

        if visibility is not None:
            joint_visibility = (
                visibility[:, joint_index]
            )

        results[joint_index] = compute_mpjpe(
            prediction[:, joint_index],
            target[:, joint_index],
            visibility=joint_visibility,
        )

    return results


# ============================================================
# 5. Temporal derivatives
# ============================================================

def compute_velocity(
    sequence: np.ndarray,
    fps: float = 1.0,
) -> np.ndarray:
    """
    Compute the first temporal derivative.

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
    Compute the second temporal derivative.

    Parameters
    ----------
    sequence:
        Shape [T, J, C].
    fps:
        Frames per second.

    Returns
    -------
    np.ndarray
        Shape [T - 2, J, C].
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
    Compute the third temporal derivative.

    Parameters
    ----------
    sequence:
        Shape [T, J, C].
    fps:
        Frames per second.

    Returns
    -------
    np.ndarray
        Shape [T - 3, J, C].
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
# 6. Temporal stability metrics
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
        visibility = np.asarray(
            visibility
        ).astype(bool)

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
        visibility = np.asarray(
            visibility
        ).astype(bool)

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

    A lower value generally indicates smoother motion.

    This metric should not be interpreted alone because genuine
    fast movements naturally contain larger acceleration.
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
        visibility = np.asarray(
            visibility
        ).astype(bool)

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

    A lower value generally indicates smoother changes in
    acceleration.
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
        visibility = np.asarray(
            visibility
        ).astype(bool)

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
# 7. Frame-to-frame displacement
# ============================================================

def compute_mean_frame_displacement(
    sequence: np.ndarray,
    visibility: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean joint displacement between consecutive frames.

    Unlike physical velocity, this function does not multiply by
    the video FPS.

    Parameters
    ----------
    sequence:
        Shape [T, J, C].
    visibility:
        Optional shape [T, J].

    Returns
    -------
    float
        Mean frame-to-frame displacement.
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
        visibility = np.asarray(
            visibility
        ).astype(bool)

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
# 8. Metric summary
# ============================================================

def summarize_pose_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    normalization_length: Optional[np.ndarray] = None,
    head_length: Optional[np.ndarray] = None,
    visibility: Optional[np.ndarray] = None,
    pck_thresholds: Sequence[float] = (
        0.05,
        0.10,
        0.20,
        0.50,
    ),
    pckh_thresholds: Sequence[float] = (
        0.50,
    ),
) -> dict:
    """
    Compute a compact dictionary of pose-estimation metrics.

    Parameters
    ----------
    prediction:
        Predicted keypoint coordinates.
    target:
        Ground-truth keypoint coordinates.
    normalization_length:
        Normalization scale used for generic PCK.
        Set this to None when generic PCK is not required.
    head_length:
        Head normalization scale used for PCKh.
        Set this to None when PCKh is not required.
    visibility:
        Optional joint visibility mask.
    pck_thresholds:
        Thresholds used for generic PCK.
    pckh_thresholds:
        Thresholds used for PCKh.

    Returns
    -------
    dict
        Metric names and values.

    Example output
    --------------
    {
        "mpjpe": 4.25,
        "pck_0.10": 93.50,
        "pckh_0.50": 88.30,
    }
    """
    metrics = {
        "mpjpe": compute_mpjpe(
            prediction=prediction,
            target=target,
            visibility=visibility,
        )
    }

    if normalization_length is not None:
        for threshold in pck_thresholds:
            if threshold <= 0:
                raise ValueError(
                    "All PCK thresholds must be positive"
                )

            metric_name = (
                f"pck_{threshold:.2f}"
            )

            metrics[metric_name] = compute_pck(
                prediction=prediction,
                target=target,
                normalization_length=normalization_length,
                threshold=threshold,
                visibility=visibility,
            )

    if head_length is not None:
        for threshold in pckh_thresholds:
            if threshold <= 0:
                raise ValueError(
                    "All PCKh thresholds must be positive"
                )

            metric_name = (
                f"pckh_{threshold:.2f}"
            )

            metrics[metric_name] = compute_pckh(
                prediction=prediction,
                target=target,
                head_length=head_length,
                threshold=threshold,
                visibility=visibility,
            )

    return metrics