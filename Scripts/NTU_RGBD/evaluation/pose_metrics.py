# Scripts/NTU_RGBD/evaluation/pose_metrics.py

from __future__ import annotations

import numpy as np

from .metric_config import EvaluationConfig


# ============================================================
# 1. Basic coordinate helpers
# ============================================================

def compute_distances(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distance between predicted and GT joints.

    Parameters
    ----------
    predictions:
        Array with shape [T, J, 2].

    ground_truth:
        Array with shape [T, J, 2].

    Returns
    -------
    np.ndarray
        Joint distances with shape [T, J].
    """
    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float32,
    )

    if predictions.shape != ground_truth.shape:
        raise ValueError(
            "predictions and ground_truth must have "
            "the same shape"
        )

    if predictions.ndim != 3:
        raise ValueError(
            "predictions must have shape [T, J, 2]"
        )

    if predictions.shape[-1] != 2:
        raise ValueError(
            "the final coordinate dimension must be 2"
        )

    return np.linalg.norm(
        predictions - ground_truth,
        axis=-1,
    ).astype(
        np.float32
    )


def compute_valid_mask(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    """
    Create a boolean mask for valid GT-visible keypoints.

    A keypoint is valid when:

    - GT visibility is greater than zero
    - predicted x/y coordinates are finite
    - GT x/y coordinates are finite

    Returns
    -------
    np.ndarray
        Boolean array with shape [T, J].
    """
    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float32,
    )

    visibility = np.asarray(
        visibility,
        dtype=np.float32,
    )

    expected_visibility_shape = (
        predictions.shape[0],
        predictions.shape[1],
    )

    if predictions.shape != ground_truth.shape:
        raise ValueError(
            "predictions and ground_truth must have "
            "the same shape"
        )

    if visibility.shape != expected_visibility_shape:
        raise ValueError(
            "visibility must have shape [T, J]"
        )

    visible = np.asarray(
        visibility > 0,
        dtype=bool,
    )

    valid_prediction = np.asarray(
        np.isfinite(
            predictions
        ).all(axis=-1),
        dtype=bool,
    )

    valid_ground_truth = np.asarray(
        np.isfinite(
            ground_truth
        ).all(axis=-1),
        dtype=bool,
    )

    return np.asarray(
        visible
        & valid_prediction
        & valid_ground_truth,
        dtype=bool,
    )


# ============================================================
# 2. Normalization scales
# ============================================================

def compute_pck_reference_lengths(
    ground_truth: np.ndarray,
    visibility: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Compute the original PCK reference length for every frame.

    The reference length is the maximum side of the bounding box
    formed by all visible GT joints:

        reference_length = max(width, height)

    Returns
    -------
    np.ndarray
        Shape [T]. Invalid frames contain NaN.
    """
    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float32,
    )

    visibility = np.asarray(
        visibility,
        dtype=np.float32,
    )

    if ground_truth.ndim != 3:
        raise ValueError(
            "ground_truth must have shape [T, J, 2]"
        )

    if ground_truth.shape[-1] != 2:
        raise ValueError(
            "ground_truth final dimension must be 2"
        )

    if visibility.shape != ground_truth.shape[:2]:
        raise ValueError(
            "visibility must have shape [T, J]"
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive"
        )

    num_frames = ground_truth.shape[0]

    reference_lengths = np.full(
        (num_frames,),
        np.nan,
        dtype=np.float32,
    )

    for frame_index in range(
        num_frames
    ):
        frame_keypoints = ground_truth[
            frame_index
        ]

        frame_visible = (
            visibility[frame_index] > 0
        )

        frame_finite = np.isfinite(
            frame_keypoints
        ).all(axis=-1)

        frame_mask = np.asarray(
            frame_visible
            & frame_finite,
            dtype=bool,
        )

        if int(frame_mask.sum()) < 2:
            continue

        valid_points = frame_keypoints[
            frame_mask
        ]

        width = float(
            valid_points[:, 0].max()
            - valid_points[:, 0].min()
        )

        height = float(
            valid_points[:, 1].max()
            - valid_points[:, 1].min()
        )

        reference_length = max(
            width,
            height,
        )

        if (
            np.isfinite(reference_length)
            and reference_length > epsilon
        ):
            reference_lengths[
                frame_index
            ] = np.float32(
                reference_length
            )

    return reference_lengths


def compute_head_lengths(
    ground_truth: np.ndarray,
    visibility: np.ndarray,
    config: EvaluationConfig,
) -> np.ndarray:
    """
    Compute the calibrated 2D GT Head-to-Neck scale.

    Formula:

        head_scale =
            mpii_head_scale_factor
            * ||GT_head - GT_neck||_2

    This approximates an MPII-style head scale but does not use
    the official MPII head bounding-box annotation.

    Returns
    -------
    np.ndarray
        Shape [T]. Invalid frames contain NaN.
    """
    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float32,
    )

    visibility = np.asarray(
        visibility,
        dtype=np.float32,
    )

    if ground_truth.ndim != 3:
        raise ValueError(
            "ground_truth must have shape [T, J, 2]"
        )

    if ground_truth.shape[-1] != 2:
        raise ValueError(
            "ground_truth final dimension must be 2"
        )

    if visibility.shape != ground_truth.shape[:2]:
        raise ValueError(
            "visibility must have shape [T, J]"
        )

    num_frames = ground_truth.shape[0]
    num_joints = ground_truth.shape[1]

    if not 0 <= config.head_index < num_joints:
        raise IndexError(
            f"Invalid head_index: "
            f"{config.head_index}"
        )

    if not 0 <= config.neck_index < num_joints:
        raise IndexError(
            f"Invalid neck_index: "
            f"{config.neck_index}"
        )

    head_points = ground_truth[
        :,
        config.head_index,
        :,
    ]

    neck_points = ground_truth[
        :,
        config.neck_index,
        :,
    ]

    head_visible = (
        visibility[:, config.head_index] > 0
    )

    neck_visible = (
        visibility[:, config.neck_index] > 0
    )

    head_finite = np.isfinite(
        head_points
    ).all(axis=-1)

    neck_finite = np.isfinite(
        neck_points
    ).all(axis=-1)

    raw_head_neck_distances = np.linalg.norm(
        head_points - neck_points,
        axis=-1,
    )

    distance_valid = (
        np.isfinite(
            raw_head_neck_distances
        )
        & (
            raw_head_neck_distances
            > config.scale_epsilon
        )
    )

    valid_mask = np.asarray(
        head_visible
        & neck_visible
        & head_finite
        & neck_finite
        & distance_valid,
        dtype=bool,
    )

    head_lengths = np.full(
        (num_frames,),
        np.nan,
        dtype=np.float32,
    )

    calibrated_lengths = (
        raw_head_neck_distances
        * float(
            config.mpii_head_scale_factor
        )
    ).astype(
        np.float32
    )

    head_lengths[
        valid_mask
    ] = calibrated_lengths[
        valid_mask
    ]

    return head_lengths


# ============================================================
# 3. Coordinate and pixel-error metrics
# ============================================================

def compute_coordinate_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute coordinate-level MSE, RMSE and MAE.

    MSE and MAE are calculated over valid x/y coordinate values,
    rather than over Euclidean joint distance.
    """
    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    if predictions.shape != ground_truth.shape:
        raise ValueError(
            "predictions and ground_truth must have "
            "the same shape"
        )

    if valid_mask.shape != predictions.shape[:2]:
        raise ValueError(
            "valid_mask must have shape [T, J]"
        )

    coordinate_error = (
        predictions - ground_truth
    )

    valid_error = coordinate_error[
        valid_mask
    ]

    if valid_error.size == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mse = float(
        np.mean(
            valid_error ** 2
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = float(
        np.mean(
            np.abs(valid_error)
        )
    )

    return mse, rmse, mae


def compute_pixel_errors(
    distances: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float]:
    """
    Compute mean and median Euclidean joint error in pixels.
    """
    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    if distances.shape != valid_mask.shape:
        raise ValueError(
            "distances and valid_mask must have "
            "the same shape"
        )

    valid_distances = distances[
        valid_mask
    ]

    if valid_distances.size == 0:
        return (
            np.nan,
            np.nan,
        )

    return (
        float(
            valid_distances.mean()
        ),
        float(
            np.median(
                valid_distances
            )
        ),
    )


# ============================================================
# 4. Normalized mean error
# ============================================================

def compute_normalized_mean_error(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    reference_lengths: np.ndarray,
    epsilon: float,
) -> float:
    """
    Compute normalized mean error.

    Formula:

        NME = mean(
            joint_distance / reference_length
        )
    """
    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    reference_lengths = np.asarray(
        reference_lengths,
        dtype=np.float32,
    )

    if distances.shape != valid_mask.shape:
        raise ValueError(
            "distances and valid_mask must have "
            "the same shape"
        )

    if reference_lengths.shape != (
        distances.shape[0],
    ):
        raise ValueError(
            "reference_lengths must have shape [T]"
        )

    expanded_reference_lengths = np.broadcast_to(
        reference_lengths[:, None],
        distances.shape,
    )

    valid_reference_mask = (
        np.isfinite(
            expanded_reference_lengths
        )
        & (
            expanded_reference_lengths
            > epsilon
        )
    )

    final_mask = np.asarray(
        valid_mask
        & valid_reference_mask,
        dtype=bool,
    )

    if not np.any(final_mask):
        return np.nan

    normalized_distances = (
        distances[final_mask]
        / expanded_reference_lengths[
            final_mask
        ]
    )

    return float(
        np.mean(
            normalized_distances
        )
    )


def compute_pck_normalized_nme(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    reference_lengths: np.ndarray,
    config: EvaluationConfig,
) -> float:
    """
    Compute NME using the original PCK body reference scale.
    """
    return compute_normalized_mean_error(
        distances=distances,
        valid_mask=valid_mask,
        reference_lengths=reference_lengths,
        epsilon=config.scale_epsilon,
    )


def compute_head_normalized_nme(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    head_lengths: np.ndarray,
    config: EvaluationConfig,
) -> float:
    """
    Compute NME using the calibrated Head-to-Neck scale.
    """
    return compute_normalized_mean_error(
        distances=distances,
        valid_mask=valid_mask,
        reference_lengths=head_lengths,
        epsilon=config.scale_epsilon,
    )


# ============================================================
# 5. PCK and PCKh
# ============================================================

def compute_threshold_accuracy(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    reference_lengths: np.ndarray,
    threshold: float,
    epsilon: float,
) -> float:
    """
    Compute percentage of joints within a normalized threshold.

    A keypoint is correct when:

        distance <= threshold * reference_length

    Returns
    -------
    float
        Percentage in the range [0, 100].
    """
    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    reference_lengths = np.asarray(
        reference_lengths,
        dtype=np.float32,
    )

    if threshold <= 0:
        raise ValueError(
            "threshold must be positive"
        )

    if distances.shape != valid_mask.shape:
        raise ValueError(
            "distances and valid_mask must have "
            "the same shape"
        )

    if reference_lengths.shape != (
        distances.shape[0],
    ):
        raise ValueError(
            "reference_lengths must have shape [T]"
        )

    frame_thresholds = (
        threshold
        * reference_lengths
    )[:, None]

    expanded_thresholds = np.broadcast_to(
        frame_thresholds,
        distances.shape,
    )

    valid_threshold_mask = (
        np.isfinite(
            expanded_thresholds
        )
        & (
            expanded_thresholds
            > epsilon
        )
    )

    final_mask = np.asarray(
        valid_mask
        & valid_threshold_mask,
        dtype=bool,
    )

    if not np.any(final_mask):
        return np.nan

    correct = (
        distances[final_mask]
        <= expanded_thresholds[
            final_mask
        ]
    )

    return float(
        np.mean(correct) * 100.0
    )


def compute_pck(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    reference_lengths: np.ndarray,
    config: EvaluationConfig,
) -> float:
    """
    Compute original PCK using the configured PCK threshold.
    """
    return compute_threshold_accuracy(
        distances=distances,
        valid_mask=valid_mask,
        reference_lengths=reference_lengths,
        threshold=config.pck_threshold,
        epsilon=config.scale_epsilon,
    )


def compute_pckh(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    head_lengths: np.ndarray,
    config: EvaluationConfig,
) -> float:
    """
    Compute approximate MPII-style PCKh using the configured
    calibrated Head-to-Neck scale.
    """
    return compute_threshold_accuracy(
        distances=distances,
        valid_mask=valid_mask,
        reference_lengths=head_lengths,
        threshold=config.pckh_threshold,
        epsilon=config.scale_epsilon,
    )


# ============================================================
# 6. Scale summaries
# ============================================================

def summarize_reference_lengths(
    reference_lengths: np.ndarray,
    epsilon: float,
) -> dict[str, float | int]:
    """
    Summarize valid frame-level normalization scales.
    """
    reference_lengths = np.asarray(
        reference_lengths,
        dtype=np.float32,
    )

    if reference_lengths.ndim != 1:
        raise ValueError(
            "reference_lengths must have shape [T]"
        )

    valid_mask = (
        np.isfinite(
            reference_lengths
        )
        & (
            reference_lengths > epsilon
        )
    )

    valid_values = reference_lengths[
        valid_mask
    ]

    total_frames = int(
        reference_lengths.shape[0]
    )

    valid_frames = int(
        valid_mask.sum()
    )

    invalid_frames = (
        total_frames - valid_frames
    )

    if valid_values.size == 0:
        mean_length = np.nan
        median_length = np.nan
        minimum_length = np.nan
        maximum_length = np.nan

    else:
        mean_length = float(
            valid_values.mean()
        )

        median_length = float(
            np.median(
                valid_values
            )
        )

        minimum_length = float(
            valid_values.min()
        )

        maximum_length = float(
            valid_values.max()
        )

    valid_ratio = (
        float(
            valid_frames
            / max(total_frames, 1)
            * 100.0
        )
    )

    return {
        "valid_frames": valid_frames,
        "invalid_frames": invalid_frames,
        "valid_frame_ratio_percent": (
            valid_ratio
        ),
        "mean_length_px": mean_length,
        "median_length_px": median_length,
        "minimum_length_px": minimum_length,
        "maximum_length_px": maximum_length,
    }


# ============================================================
# 7. Visibility metrics
# ============================================================

def compute_visibility_metrics(
    ground_truth_visibility: np.ndarray,
    predicted_visibility: np.ndarray | None,
) -> dict[str, float]:
    """
    Compute visibility precision, recall and F1.

    If predicted visibility is unavailable, all values are NaN.
    """
    ground_truth_visibility = np.asarray(
        ground_truth_visibility,
        dtype=np.float32,
    )

    if predicted_visibility is None:
        return {
            "visibility_precision_percent": np.nan,
            "visibility_recall_percent": np.nan,
            "visibility_f1_percent": np.nan,
        }

    predicted_visibility = np.asarray(
        predicted_visibility,
        dtype=np.float32,
    )

    if (
        ground_truth_visibility.shape
        != predicted_visibility.shape
    ):
        raise ValueError(
            "ground_truth_visibility and "
            "predicted_visibility must have the same shape"
        )

    gt_visible = (
        ground_truth_visibility > 0
    )

    pred_visible = (
        predicted_visibility > 0
    )

    true_positive = int(
        np.sum(
            gt_visible
            & pred_visible
        )
    )

    false_positive = int(
        np.sum(
            ~gt_visible
            & pred_visible
        )
    )

    false_negative = int(
        np.sum(
            gt_visible
            & ~pred_visible
        )
    )

    precision = (
        true_positive
        / max(
            true_positive
            + false_positive,
            1,
        )
    )

    recall = (
        true_positive
        / max(
            true_positive
            + false_negative,
            1,
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / max(
            precision + recall,
            1e-12,
        )
    )

    return {
        "visibility_precision_percent": (
            precision * 100.0
        ),
        "visibility_recall_percent": (
            recall * 100.0
        ),
        "visibility_f1_percent": (
            f1 * 100.0
        ),
    }


# ============================================================
# 8. Confidence summary
# ============================================================

def compute_mean_confidence(
    confidences: np.ndarray | None,
) -> float:
    """
    Compute the mean of all finite confidence values.
    """
    if confidences is None:
        return np.nan

    confidences = np.asarray(
        confidences,
        dtype=np.float32,
    )

    valid_confidences = confidences[
        np.isfinite(
            confidences
        )
    ]

    if valid_confidences.size == 0:
        return np.nan

    return float(
        valid_confidences.mean()
    )