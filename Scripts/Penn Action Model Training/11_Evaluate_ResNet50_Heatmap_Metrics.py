# Scripts/Penn Action Model Training/
# 11_Evaluate_ResNet50_Heatmap_Metrics.py

from pathlib import Path
import csv

import numpy as np


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GT_NPZ_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Penn_Action"
    / "penn_action_processed.npz"
)

PRED_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "10_Generate_MP4_ResNet50_Heatmap"
    / "0684_ResNet50_Pred_vs_GT.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "11_Evaluate_ResNet50_Heatmap_Metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "resnet50_heatmap_metrics_summary.csv"
)

PER_JOINT_CSV = (
    OUTPUT_DIR
    / "resnet50_heatmap_metrics_per_joint.csv"
)

PER_FRAME_CSV = (
    OUTPUT_DIR
    / "resnet50_heatmap_metrics_per_frame.csv"
)

# PCK thresholds based on visible keypoint bounding-box scale
PCK_THRESHOLDS = [
    0.05,
    0.10,
    0.20,
    0.50,
]

# Per-joint table uses this threshold
PER_JOINT_PCK_THRESHOLD = 0.10


# ============================================================
# 2. Keypoint definition
# ============================================================

JOINT_NAMES = [
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hip",
    "Right_Hip",
    "Left_Knee",
    "Right_Knee",
    "Left_Ankle",
    "Right_Ankle",
]


# ============================================================
# 3. Basic utilities
# ============================================================

def safe_video_id_to_str(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    value = str(value)

    if value.isdigit():
        return value.zfill(4)

    return value


def read_scalar_string(array_value):
    """
    Convert NPZ scalar/string array safely to Python string.
    """
    value = np.asarray(array_value)

    if value.ndim == 0:
        value = value.item()

    return safe_video_id_to_str(value)


def compute_visible_mask(
    visibility,
    predictions,
    ground_truth,
):
    """
    Evaluate only visible and finite keypoints.

    visibility:
        [T, K]

    predictions:
        [T, K, 2]

    ground_truth:
        [T, K, 2]
    """

    visible = visibility > 0

    valid_predictions = (
        np.isfinite(predictions)
        .all(axis=-1)
    )

    valid_ground_truth = (
        np.isfinite(ground_truth)
        .all(axis=-1)
    )

    return (
        visible
        & valid_predictions
        & valid_ground_truth
    )


def euclidean_distance(
    predictions,
    ground_truth,
):
    """
    Returns Euclidean keypoint error.

    Shape:
        [T, K]
    """

    return np.linalg.norm(
        predictions - ground_truth,
        axis=-1,
    )


# ============================================================
# 4. PCK reference scale
# ============================================================

def estimate_reference_length(
    ground_truth,
    visibility,
):
    """
    Penn Action does not provide MPII-style head size.

    Per-frame reference length:
        max(width, height)

    where width and height come from the bounding box of
    visible ground-truth keypoints.

    Returns:
        [T]
    """

    num_frames = ground_truth.shape[0]

    reference_lengths = np.full(
        num_frames,
        np.nan,
        dtype=np.float32,
    )

    for frame_index in range(
        num_frames
    ):
        visible = (
            visibility[frame_index] > 0
        )

        finite = (
            np.isfinite(
                ground_truth[frame_index]
            )
            .all(axis=-1)
        )

        mask = visible & finite

        if mask.sum() < 2:
            continue

        x_values = ground_truth[
            frame_index,
            mask,
            0,
        ]

        y_values = ground_truth[
            frame_index,
            mask,
            1,
        ]

        width = (
            x_values.max()
            - x_values.min()
        )

        height = (
            y_values.max()
            - y_values.min()
        )

        scale = max(
            float(width),
            float(height),
        )

        if scale > 1.0:
            reference_lengths[
                frame_index
            ] = scale

    return reference_lengths


# ============================================================
# 5. Error metrics
# ============================================================

def compute_mse(
    predictions,
    ground_truth,
    mask,
):
    """
    Coordinate MSE over valid x/y values.
    """

    coordinate_error = (
        predictions - ground_truth
    ) ** 2

    valid_error = coordinate_error[
        mask
    ]

    if valid_error.size == 0:
        return np.nan

    return float(
        valid_error.mean()
    )


def compute_rmse(
    predictions,
    ground_truth,
    mask,
):
    mse = compute_mse(
        predictions,
        ground_truth,
        mask,
    )

    if np.isnan(mse):
        return np.nan

    return float(
        np.sqrt(mse)
    )


def compute_mae(
    predictions,
    ground_truth,
    mask,
):
    coordinate_error = np.abs(
        predictions - ground_truth
    )

    valid_error = coordinate_error[
        mask
    ]

    if valid_error.size == 0:
        return np.nan

    return float(
        valid_error.mean()
    )


def compute_mean_pixel_error(
    distances,
    mask,
):
    valid_distances = distances[
        mask
    ]

    if valid_distances.size == 0:
        return np.nan

    return float(
        valid_distances.mean()
    )


def compute_median_pixel_error(
    distances,
    mask,
):
    valid_distances = distances[
        mask
    ]

    if valid_distances.size == 0:
        return np.nan

    return float(
        np.median(
            valid_distances
        )
    )


# ============================================================
# 6. PCK
# ============================================================

def compute_pck_with_frame_reference(
    distances,
    mask,
    reference_lengths,
    threshold_ratio,
):
    """
    threshold_pixels[t] =
        threshold_ratio * reference_lengths[t]
    """

    threshold_pixels = (
        threshold_ratio
        * reference_lengths
    )

    threshold_matrix = np.broadcast_to(
        threshold_pixels[:, None],
        distances.shape,
    )

    valid_reference = np.broadcast_to(
        np.isfinite(
            threshold_pixels
        )[:, None],
        distances.shape,
    )

    final_mask = (
        mask
        & valid_reference
    )

    valid_distances = distances[
        final_mask
    ]

    valid_thresholds = threshold_matrix[
        final_mask
    ]

    if valid_distances.size == 0:
        return np.nan

    correct = (
        valid_distances
        < valid_thresholds
    )

    return float(
        correct.mean()
        * 100.0
    )


# ============================================================
# 7. Temporal metrics
# ============================================================

def compute_temporal_velocity(
    predictions,
    visibility,
):
    """
    Mean frame-to-frame displacement.

    Lower values indicate less movement, but this metric
    mixes true human motion and prediction jitter.
    """

    if len(predictions) < 2:
        return np.nan

    total_velocity = 0.0
    total_count = 0

    for frame_index in range(
        1,
        len(predictions),
    ):
        valid = (
            (visibility[frame_index] > 0)
            & (
                visibility[
                    frame_index - 1
                ] > 0
            )
            & np.isfinite(
                predictions[frame_index]
            ).all(axis=-1)
            & np.isfinite(
                predictions[
                    frame_index - 1
                ]
            ).all(axis=-1)
        )

        if valid.sum() == 0:
            continue

        velocity = (
            predictions[
                frame_index,
                valid,
            ]
            - predictions[
                frame_index - 1,
                valid,
            ]
        )

        velocity_norm = np.linalg.norm(
            velocity,
            axis=1,
        )

        total_velocity += float(
            velocity_norm.sum()
        )

        total_count += int(
            velocity_norm.size
        )

    if total_count == 0:
        return np.nan

    return float(
        total_velocity
        / total_count
    )


def compute_temporal_acceleration(
    predictions,
    visibility,
):
    """
    Second-order temporal difference:

        acceleration[t] =
            pred[t]
            - 2 * pred[t-1]
            + pred[t-2]

    Lower values generally indicate smoother predictions.
    """

    if len(predictions) < 3:
        return np.nan

    total_acceleration = 0.0
    total_count = 0

    for frame_index in range(
        2,
        len(predictions),
    ):
        valid = (
            (visibility[frame_index] > 0)
            & (
                visibility[
                    frame_index - 1
                ] > 0
            )
            & (
                visibility[
                    frame_index - 2
                ] > 0
            )
            & np.isfinite(
                predictions[frame_index]
            ).all(axis=-1)
            & np.isfinite(
                predictions[
                    frame_index - 1
                ]
            ).all(axis=-1)
            & np.isfinite(
                predictions[
                    frame_index - 2
                ]
            ).all(axis=-1)
        )

        if valid.sum() == 0:
            continue

        acceleration = (
            predictions[
                frame_index,
                valid,
            ]
            - 2.0
            * predictions[
                frame_index - 1,
                valid,
            ]
            + predictions[
                frame_index - 2,
                valid,
            ]
        )

        acceleration_norm = np.linalg.norm(
            acceleration,
            axis=1,
        )

        total_acceleration += float(
            acceleration_norm.sum()
        )

        total_count += int(
            acceleration_norm.size
        )

    if total_count == 0:
        return np.nan

    return float(
        total_acceleration
        / total_count
    )


def compute_gt_relative_acceleration_error(
    predictions,
    ground_truth,
    visibility,
):
    """
    Compare predicted acceleration with GT acceleration.

    This is more informative than prediction acceleration alone,
    because large real actions can legitimately have large
    acceleration.

    Lower is better.
    """

    if len(predictions) < 3:
        return np.nan

    total_error = 0.0
    total_count = 0

    for frame_index in range(
        2,
        len(predictions),
    ):
        valid = (
            (visibility[frame_index] > 0)
            & (
                visibility[
                    frame_index - 1
                ] > 0
            )
            & (
                visibility[
                    frame_index - 2
                ] > 0
            )
            & np.isfinite(
                predictions[frame_index]
            ).all(axis=-1)
            & np.isfinite(
                predictions[
                    frame_index - 1
                ]
            ).all(axis=-1)
            & np.isfinite(
                predictions[
                    frame_index - 2
                ]
            ).all(axis=-1)
            & np.isfinite(
                ground_truth[frame_index]
            ).all(axis=-1)
            & np.isfinite(
                ground_truth[
                    frame_index - 1
                ]
            ).all(axis=-1)
            & np.isfinite(
                ground_truth[
                    frame_index - 2
                ]
            ).all(axis=-1)
        )

        if valid.sum() == 0:
            continue

        predicted_acceleration = (
            predictions[
                frame_index,
                valid,
            ]
            - 2.0
            * predictions[
                frame_index - 1,
                valid,
            ]
            + predictions[
                frame_index - 2,
                valid,
            ]
        )

        ground_truth_acceleration = (
            ground_truth[
                frame_index,
                valid,
            ]
            - 2.0
            * ground_truth[
                frame_index - 1,
                valid,
            ]
            + ground_truth[
                frame_index - 2,
                valid,
            ]
        )

        acceleration_error = (
            predicted_acceleration
            - ground_truth_acceleration
        )

        acceleration_error_norm = (
            np.linalg.norm(
                acceleration_error,
                axis=1,
            )
        )

        total_error += float(
            acceleration_error_norm.sum()
        )

        total_count += int(
            acceleration_error_norm.size
        )

    if total_count == 0:
        return np.nan

    return float(
        total_error
        / total_count
    )


# ============================================================
# 8. Per-joint metrics
# ============================================================

def compute_per_joint_metrics(
    predictions,
    ground_truth,
    visibility,
    keypoint_names,
    pck_threshold_ratio,
):
    num_keypoints = (
        predictions.shape[1]
    )

    distances = euclidean_distance(
        predictions,
        ground_truth,
    )

    mask = compute_visible_mask(
        visibility,
        predictions,
        ground_truth,
    )

    reference_lengths = (
        estimate_reference_length(
            ground_truth,
            visibility,
        )
    )

    thresholds = (
        pck_threshold_ratio
        * reference_lengths
    )

    results = []

    for joint_index in range(
        num_keypoints
    ):
        joint_mask = mask[
            :,
            joint_index,
        ]

        valid_distances = distances[
            :,
            joint_index,
        ][joint_mask]

        visible_count = int(
            valid_distances.size
        )

        if visible_count == 0:
            mean_error = np.nan
            median_error = np.nan
            rmse = np.nan
            pck = np.nan

        else:
            mean_error = float(
                valid_distances.mean()
            )

            median_error = float(
                np.median(
                    valid_distances
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        valid_distances ** 2
                    )
                )
            )

            valid_reference = np.isfinite(
                thresholds
            )

            final_mask = (
                joint_mask
                & valid_reference
            )

            joint_distances = distances[
                :,
                joint_index,
            ][final_mask]

            joint_thresholds = thresholds[
                final_mask
            ]

            if joint_distances.size == 0:
                pck = np.nan
            else:
                pck = float(
                    (
                        joint_distances
                        < joint_thresholds
                    ).mean()
                    * 100.0
                )

        joint_name = (
            keypoint_names[joint_index]
            if joint_index
            < len(keypoint_names)
            else f"Joint_{joint_index}"
        )

        results.append({
            "joint_index":
                joint_index,

            "joint_name":
                joint_name,

            "visible_count":
                visible_count,

            "mean_pixel_error":
                mean_error,

            "median_pixel_error":
                median_error,

            "RMSE_pixel":
                rmse,

            f"PCK@{pck_threshold_ratio}":
                pck,
        })

    return results


# ============================================================
# 9. Per-frame metrics
# ============================================================

def compute_per_frame_metrics(
    frame_indices,
    predictions,
    ground_truth,
    visibility,
    reference_lengths,
):
    distances = euclidean_distance(
        predictions,
        ground_truth,
    )

    mask = compute_visible_mask(
        visibility,
        predictions,
        ground_truth,
    )

    rows = []

    for frame_position in range(
        len(predictions)
    ):
        frame_mask = mask[
            frame_position
        ]

        valid_distances = distances[
            frame_position
        ][frame_mask]

        if valid_distances.size == 0:
            mean_error = np.nan
            median_error = np.nan
        else:
            mean_error = float(
                valid_distances.mean()
            )

            median_error = float(
                np.median(
                    valid_distances
                )
            )

        row = {
            "frame_index":
                int(
                    frame_indices[
                        frame_position
                    ]
                ),

            "visible_keypoints":
                int(
                    frame_mask.sum()
                ),

            "reference_length":
                float(
                    reference_lengths[
                        frame_position
                    ]
                )
                if np.isfinite(
                    reference_lengths[
                        frame_position
                    ]
                )
                else np.nan,

            "mean_pixel_error":
                mean_error,

            "median_pixel_error":
                median_error,
        }

        for threshold in PCK_THRESHOLDS:
            reference_length = (
                reference_lengths[
                    frame_position
                ]
            )

            if (
                valid_distances.size == 0
                or not np.isfinite(
                    reference_length
                )
            ):
                pck = np.nan
            else:
                threshold_pixels = (
                    threshold
                    * reference_length
                )

                pck = float(
                    (
                        valid_distances
                        < threshold_pixels
                    ).mean()
                    * 100.0
                )

            row[f"PCK@{threshold}"] = pck

        rows.append(row)

    return rows


# ============================================================
# 10. Prediction loading
# ============================================================

def load_prediction_file(
    prediction_path,
):
    with np.load(
        prediction_path,
        allow_pickle=True,
    ) as prediction_data:

        print(
            "\nPrediction keys:",
            prediction_data.files,
        )

        video_id = read_scalar_string(
            prediction_data["video_id"]
        )

        frame_indices = (
            prediction_data[
                "frame_indices"
            ]
            .astype(np.int64)
        )

        predictions = (
            prediction_data[
                "pred_keypoints"
            ]
            .astype(np.float32)
        )

        predicted_scores = (
            prediction_data[
                "pred_scores"
            ]
            .astype(np.float32)
            if "pred_scores"
            in prediction_data.files
            else None
        )

        saved_ground_truth = (
            prediction_data[
                "gt_keypoints"
            ]
            .astype(np.float32)
            if "gt_keypoints"
            in prediction_data.files
            else None
        )

        saved_visibility = (
            prediction_data[
                "gt_visibility"
            ]
            .astype(np.float32)
            if "gt_visibility"
            in prediction_data.files
            else None
        )

        video_split = (
            read_scalar_string(
                prediction_data[
                    "video_split"
                ]
            )
            if "video_split"
            in prediction_data.files
            else "unknown"
        )

        model_name = (
            read_scalar_string(
                prediction_data[
                    "model_name"
                ]
            )
            if "model_name"
            in prediction_data.files
            else "ResNet50_Heatmap_Baseline"
        )

    return {
        "video_id":
            video_id,

        "frame_indices":
            frame_indices,

        "predictions":
            predictions,

        "predicted_scores":
            predicted_scores,

        "saved_ground_truth":
            saved_ground_truth,

        "saved_visibility":
            saved_visibility,

        "video_split":
            video_split,

        "model_name":
            model_name,
    }


# ============================================================
# 11. Match predictions with GT dataset
# ============================================================

def load_and_match_ground_truth(
    gt_npz_path,
    video_id,
    prediction_frame_indices,
):
    with np.load(
        gt_npz_path,
        allow_pickle=True,
    ) as gt_data:

        gt_video_ids = np.asarray([
            safe_video_id_to_str(value)
            for value in gt_data[
                "video_ids"
            ]
        ])

        gt_frame_indices = (
            gt_data[
                "frame_indices"
            ]
            .astype(np.int64)
        )

        gt_keypoints = (
            gt_data[
                "keypoints"
            ]
            .astype(np.float32)
        )

        gt_visibility = (
            gt_data[
                "visibility"
            ]
            .astype(np.float32)
        )

    lookup = {}

    for dataset_index in range(
        len(gt_video_ids)
    ):
        key = (
            gt_video_ids[
                dataset_index
            ],
            int(
                gt_frame_indices[
                    dataset_index
                ]
            ),
        )

        lookup[key] = dataset_index

    matched_prediction_positions = []
    matched_ground_truth = []
    matched_visibility = []
    matched_frame_indices = []

    for prediction_position, frame_index in enumerate(
        prediction_frame_indices
    ):
        key = (
            video_id,
            int(frame_index),
        )

        if key not in lookup:
            continue

        gt_index = lookup[key]

        matched_prediction_positions.append(
            prediction_position
        )

        matched_ground_truth.append(
            gt_keypoints[gt_index]
        )

        matched_visibility.append(
            gt_visibility[gt_index]
        )

        matched_frame_indices.append(
            int(frame_index)
        )

    if len(
        matched_prediction_positions
    ) == 0:
        raise RuntimeError(
            "No matched frames were found between "
            "predictions and ground truth."
        )

    return {
        "prediction_positions":
            np.asarray(
                matched_prediction_positions,
                dtype=np.int64,
            ),

        "ground_truth":
            np.stack(
                matched_ground_truth,
                axis=0,
            ).astype(np.float32),

        "visibility":
            np.stack(
                matched_visibility,
                axis=0,
            ).astype(np.float32),

        "frame_indices":
            np.asarray(
                matched_frame_indices,
                dtype=np.int64,
            ),
    }


# ============================================================
# 12. CSV saving
# ============================================================

def save_single_row_csv(
    path,
    row,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(row)


def save_rows_csv(
    path,
    rows,
):
    if len(rows) == 0:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 13. Main
# ============================================================

def main():
    print("=" * 72)
    print("Evaluate ResNet50 Heatmap Metrics")
    print("=" * 72)
    print(f"Project root    : {PROJECT_ROOT}")
    print(f"GT dataset      : {GT_NPZ_PATH}")
    print(f"Prediction file : {PRED_PATH}")
    print(f"Output dir      : {OUTPUT_DIR}")

    if not GT_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"GT NPZ not found:\n"
            f"{GT_NPZ_PATH}"
        )

    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PRED_PATH}\n\n"
            "Run "
            "10_Generate_MP4_ResNet50_Heatmap.py "
            "first."
        )

    prediction_info = (
        load_prediction_file(
            PRED_PATH
        )
    )

    video_id = prediction_info[
        "video_id"
    ]

    prediction_frame_indices = (
        prediction_info[
            "frame_indices"
        ]
    )

    all_predictions = (
        prediction_info[
            "predictions"
        ]
    )

    print(
        f"\nModel           : "
        f"{prediction_info['model_name']}"
    )

    print(
        f"Video ID        : "
        f"{video_id}"
    )

    print(
        f"Video split     : "
        f"{prediction_info['video_split']}"
    )

    print(
        f"Predicted frames: "
        f"{len(prediction_frame_indices)}"
    )

    print(
        f"Prediction shape: "
        f"{all_predictions.shape}"
    )

    matched = load_and_match_ground_truth(
        gt_npz_path=GT_NPZ_PATH,
        video_id=video_id,
        prediction_frame_indices=(
            prediction_frame_indices
        ),
    )

    prediction_positions = (
        matched[
            "prediction_positions"
        ]
    )

    predictions = all_predictions[
        prediction_positions
    ]

    ground_truth = matched[
        "ground_truth"
    ]

    visibility = matched[
        "visibility"
    ]

    frame_indices = matched[
        "frame_indices"
    ]

    print(
        f"Matched frames  : "
        f"{len(predictions)}"
    )

    if (
        predictions.shape
        != ground_truth.shape
    ):
        raise ValueError(
            "Prediction and GT shapes do not match:\n"
            f"Prediction: {predictions.shape}\n"
            f"GT: {ground_truth.shape}"
        )

    # ========================================================
    # 14. Compute summary metrics
    # ========================================================

    distances = euclidean_distance(
        predictions,
        ground_truth,
    )

    mask = compute_visible_mask(
        visibility,
        predictions,
        ground_truth,
    )

    reference_lengths = (
        estimate_reference_length(
            ground_truth,
            visibility,
        )
    )

    visible_count = int(
        mask.sum()
    )

    total_count = int(
        mask.size
    )

    mse = compute_mse(
        predictions,
        ground_truth,
        mask,
    )

    rmse = compute_rmse(
        predictions,
        ground_truth,
        mask,
    )

    mae = compute_mae(
        predictions,
        ground_truth,
        mask,
    )

    mean_pixel_error = (
        compute_mean_pixel_error(
            distances,
            mask,
        )
    )

    median_pixel_error = (
        compute_median_pixel_error(
            distances,
            mask,
        )
    )

    mean_velocity = (
        compute_temporal_velocity(
            predictions,
            visibility,
        )
    )

    mean_acceleration = (
        compute_temporal_acceleration(
            predictions,
            visibility,
        )
    )

    gt_relative_acceleration_error = (
        compute_gt_relative_acceleration_error(
            predictions,
            ground_truth,
            visibility,
        )
    )

    summary = {
        "method":
            "ResNet50_Heatmap_Baseline",

        "video_id":
            video_id,

        "video_split":
            prediction_info[
                "video_split"
            ],

        "num_frames":
            int(
                predictions.shape[0]
            ),

        "num_keypoints":
            int(
                predictions.shape[1]
            ),

        "visible_keypoints":
            visible_count,

        "total_keypoints":
            total_count,

        "visibility_ratio_percent":
            (
                visible_count
                / max(total_count, 1)
                * 100.0
            ),

        "MSE_coordinate":
            mse,

        "RMSE_coordinate":
            rmse,

        "MAE_coordinate":
            mae,

        "mean_pixel_error":
            mean_pixel_error,

        "median_pixel_error":
            median_pixel_error,

        "mean_temporal_velocity":
            mean_velocity,

        "mean_temporal_acceleration":
            mean_acceleration,

        "gt_relative_acceleration_error":
            gt_relative_acceleration_error,
    }

    for threshold in PCK_THRESHOLDS:
        pck_value = (
            compute_pck_with_frame_reference(
                distances=distances,
                mask=mask,
                reference_lengths=(
                    reference_lengths
                ),
                threshold_ratio=threshold,
            )
        )

        summary[
            f"PCK@{threshold}"
        ] = pck_value

    per_joint_results = (
        compute_per_joint_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            visibility=visibility,
            keypoint_names=JOINT_NAMES,
            pck_threshold_ratio=(
                PER_JOINT_PCK_THRESHOLD
            ),
        )
    )

    per_frame_results = (
        compute_per_frame_metrics(
            frame_indices=frame_indices,
            predictions=predictions,
            ground_truth=ground_truth,
            visibility=visibility,
            reference_lengths=(
                reference_lengths
            ),
        )
    )

    # ========================================================
    # 15. Print results
    # ========================================================

    print("\n" + "=" * 72)
    print("ResNet50 Heatmap Baseline Metrics")
    print("=" * 72)

    print(
        f"Video ID             : "
        f"{video_id}"
    )

    print(
        f"Video split          : "
        f"{prediction_info['video_split']}"
    )

    print(
        f"Frames               : "
        f"{predictions.shape[0]}"
    )

    print(
        f"Keypoints            : "
        f"{predictions.shape[1]}"
    )

    print(
        f"Visible keypoints    : "
        f"{visible_count}/{total_count}"
    )

    print(
        f"Visibility ratio     : "
        f"{summary['visibility_ratio_percent']:.2f}%"
    )

    print(
        f"MSE coordinate       : "
        f"{mse:.4f}"
    )

    print(
        f"RMSE coordinate      : "
        f"{rmse:.4f}"
    )

    print(
        f"MAE coordinate       : "
        f"{mae:.4f}"
    )

    print(
        f"Mean pixel error     : "
        f"{mean_pixel_error:.4f}"
    )

    print(
        f"Median pixel error   : "
        f"{median_pixel_error:.4f}"
    )

    print(
        f"Temporal velocity    : "
        f"{mean_velocity:.4f}"
    )

    print(
        f"Temporal acceleration: "
        f"{mean_acceleration:.4f}"
    )

    print(
        f"GT-relative accel err: "
        f"{gt_relative_acceleration_error:.4f}"
    )

    for threshold in PCK_THRESHOLDS:
        value = summary[
            f"PCK@{threshold}"
        ]

        print(
            f"PCK@{threshold:<4}            : "
            f"{value:.2f}%"
        )

    print("\nPer-joint results:")

    for result in per_joint_results:
        print(
            f"{result['joint_index']:02d} "
            f"{result['joint_name']:<16} | "
            f"count="
            f"{result['visible_count']:<5} | "
            f"mean="
            f"{result['mean_pixel_error']:.2f} | "
            f"median="
            f"{result['median_pixel_error']:.2f} | "
            f"PCK@"
            f"{PER_JOINT_PCK_THRESHOLD}="
            f"{result[f'PCK@{PER_JOINT_PCK_THRESHOLD}']:.2f}%"
        )

    # ========================================================
    # 16. Save results
    # ========================================================

    save_single_row_csv(
        SUMMARY_CSV,
        summary,
    )

    save_rows_csv(
        PER_JOINT_CSV,
        per_joint_results,
    )

    save_rows_csv(
        PER_FRAME_CSV,
        per_frame_results,
    )

    print("\nEvaluation finished.")

    print("\nSaved files:")
    print(
        f"  Summary   : {SUMMARY_CSV}"
    )
    print(
        f"  Per joint : {PER_JOINT_CSV}"
    )
    print(
        f"  Per frame : {PER_FRAME_CSV}"
    )


if __name__ == "__main__":
    main()