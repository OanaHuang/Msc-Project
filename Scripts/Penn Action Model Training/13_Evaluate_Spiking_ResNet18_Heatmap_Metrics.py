# Scripts/Penn Action Model Training/
# 13_Evaluate_Spiking_ResNet18_Heatmap_Metrics.py

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_VIDEO_ID = "0684"

PRED_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "12_Generate_MP4_Spiking_ResNet18_Heatmap"
    / f"{TARGET_VIDEO_ID}_spiking_resnet18_predictions.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "13_Evaluate_Spiking_ResNet18_Heatmap_Metrics"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV_PATH = (
    OUTPUT_DIR
    / "spiking_resnet18_metrics_summary.csv"
)

PER_JOINT_CSV_PATH = (
    OUTPUT_DIR
    / "spiking_resnet18_metrics_per_joint.csv"
)

PLOT_PATH = (
    OUTPUT_DIR
    / "spiking_resnet18_pck_per_joint.png"
)

PCK_THRESHOLD = 0.1

# Exclude samples whose person scale is extremely small.
MIN_PERSON_SCALE = 1e-6


JOINT_NAMES = [
    "Head",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
]


# ============================================================
# 2. Data loading
# ============================================================

def load_prediction_data() -> dict[str, np.ndarray]:
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            "Prediction file was not found:\n"
            f"{PRED_PATH}\n\n"
            "Run code 12 first."
        )

    data = np.load(
        PRED_PATH,
        allow_pickle=True,
    )

    required_keys = [
        "predicted_xy",
        "ground_truth_xy",
        "visibility",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in data.files
    ]

    if missing_keys:
        raise KeyError(
            "Prediction file is missing required arrays.\n"
            f"Missing keys: {missing_keys}\n"
            f"Available keys: {data.files}"
        )

    result = {
        "predicted_xy": np.asarray(
            data["predicted_xy"],
            dtype=np.float32,
        ),
        "ground_truth_xy": np.asarray(
            data["ground_truth_xy"],
            dtype=np.float32,
        ),
        "visibility": np.asarray(
            data["visibility"],
            dtype=np.float32,
        ),
    }

    optional_keys = [
        "confidence",
        "sample_indices",
        "image_paths",
        "video_id",
        "checkpoint_path",
        "checkpoint_epoch",
        "checkpoint_best_val_loss",
    ]

    for key in optional_keys:
        if key in data.files:
            result[key] = data[key]

    data.close()

    return result


# ============================================================
# 3. Validation
# ============================================================

def validate_shapes(
    predicted_xy: np.ndarray,
    ground_truth_xy: np.ndarray,
    visibility: np.ndarray,
) -> None:
    if predicted_xy.ndim != 3:
        raise ValueError(
            "predicted_xy must have shape [N, J, 2], "
            f"but received {predicted_xy.shape}"
        )

    if ground_truth_xy.ndim != 3:
        raise ValueError(
            "ground_truth_xy must have shape [N, J, 2], "
            f"but received {ground_truth_xy.shape}"
        )

    if visibility.ndim != 2:
        raise ValueError(
            "visibility must have shape [N, J], "
            f"but received {visibility.shape}"
        )

    if predicted_xy.shape != ground_truth_xy.shape:
        raise ValueError(
            "Prediction and ground-truth shapes do not match.\n"
            f"Prediction  : {predicted_xy.shape}\n"
            f"Ground truth: {ground_truth_xy.shape}"
        )

    expected_visibility_shape = predicted_xy.shape[:2]

    if visibility.shape != expected_visibility_shape:
        raise ValueError(
            "Visibility shape does not match predictions.\n"
            f"Visibility : {visibility.shape}\n"
            f"Expected   : {expected_visibility_shape}"
        )

    if predicted_xy.shape[-1] != 2:
        raise ValueError(
            "The final coordinate dimension must contain x and y."
        )

    if predicted_xy.shape[1] != len(JOINT_NAMES):
        raise ValueError(
            "Unexpected number of joints.\n"
            f"Prediction joints: {predicted_xy.shape[1]}\n"
            f"Expected joints  : {len(JOINT_NAMES)}"
        )


# ============================================================
# 4. Metric utilities
# ============================================================

def calculate_person_scale(
    ground_truth_xy: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    """
    Calculate one person scale for each frame.

    Scale definition:
        max(width of visible GT joints,
            height of visible GT joints)

    Input:
        ground_truth_xy: [N, J, 2]
        visibility:      [N, J]

    Return:
        person_scale:    [N]
    """

    num_frames = ground_truth_xy.shape[0]

    person_scale = np.zeros(
        num_frames,
        dtype=np.float32,
    )

    for frame_index in range(num_frames):
        visible_mask = visibility[frame_index] > 0

        visible_points = ground_truth_xy[
            frame_index,
            visible_mask,
        ]

        if len(visible_points) < 2:
            person_scale[frame_index] = 0.0
            continue

        min_xy = np.min(
            visible_points,
            axis=0,
        )

        max_xy = np.max(
            visible_points,
            axis=0,
        )

        width = max_xy[0] - min_xy[0]
        height = max_xy[1] - min_xy[1]

        person_scale[frame_index] = max(
            width,
            height,
        )

    return person_scale


def calculate_distances(
    predicted_xy: np.ndarray,
    ground_truth_xy: np.ndarray,
) -> np.ndarray:
    """
    Euclidean distance in original image pixel space.

    Return:
        distances: [N, J]
    """

    difference = predicted_xy - ground_truth_xy

    distances = np.linalg.norm(
        difference,
        axis=-1,
    )

    return distances.astype(np.float32)


def calculate_normalized_errors(
    distances: np.ndarray,
    person_scale: np.ndarray,
) -> np.ndarray:
    normalized_errors = np.full(
        distances.shape,
        np.nan,
        dtype=np.float32,
    )

    valid_scale_mask = person_scale > MIN_PERSON_SCALE

    normalized_errors[valid_scale_mask] = (
        distances[valid_scale_mask]
        / person_scale[valid_scale_mask, None]
    )

    return normalized_errors


def build_valid_mask(
    visibility: np.ndarray,
    person_scale: np.ndarray,
    predicted_xy: np.ndarray,
    ground_truth_xy: np.ndarray,
) -> np.ndarray:
    visible_mask = visibility > 0

    scale_mask = (
        person_scale[:, None] > MIN_PERSON_SCALE
    )

    finite_prediction_mask = np.isfinite(
        predicted_xy
    ).all(axis=-1)

    finite_ground_truth_mask = np.isfinite(
        ground_truth_xy
    ).all(axis=-1)

    valid_mask = (
        visible_mask
        & scale_mask
        & finite_prediction_mask
        & finite_ground_truth_mask
    )

    return valid_mask


def safe_mean(
    values: np.ndarray,
) -> float:
    if values.size == 0:
        return float("nan")

    return float(
        np.mean(values)
    )


def safe_percentage(
    correct_count: int,
    total_count: int,
) -> float:
    if total_count == 0:
        return float("nan")

    return 100.0 * correct_count / total_count


# ============================================================
# 5. Metric calculation
# ============================================================

def calculate_metrics(
    predicted_xy: np.ndarray,
    ground_truth_xy: np.ndarray,
    visibility: np.ndarray,
) -> tuple[dict, list[dict]]:
    person_scale = calculate_person_scale(
        ground_truth_xy=ground_truth_xy,
        visibility=visibility,
    )

    distances = calculate_distances(
        predicted_xy=predicted_xy,
        ground_truth_xy=ground_truth_xy,
    )

    normalized_errors = calculate_normalized_errors(
        distances=distances,
        person_scale=person_scale,
    )

    valid_mask = build_valid_mask(
        visibility=visibility,
        person_scale=person_scale,
        predicted_xy=predicted_xy,
        ground_truth_xy=ground_truth_xy,
    )

    correct_mask = (
        normalized_errors <= PCK_THRESHOLD
    ) & valid_mask

    total_valid_joints = int(
        np.sum(valid_mask)
    )

    total_correct_joints = int(
        np.sum(correct_mask)
    )

    overall_pck = safe_percentage(
        correct_count=total_correct_joints,
        total_count=total_valid_joints,
    )

    overall_mean_pixel_error = safe_mean(
        distances[valid_mask]
    )

    overall_mean_normalized_error = safe_mean(
        normalized_errors[valid_mask]
    )

    valid_frames = np.any(
        valid_mask,
        axis=1,
    )

    summary = {
        "model": "Spiking ResNet18 Heatmap",
        "video_id": str(TARGET_VIDEO_ID),
        "pck_threshold": PCK_THRESHOLD,
        "number_of_frames": int(
            predicted_xy.shape[0]
        ),
        "valid_frames": int(
            np.sum(valid_frames)
        ),
        "total_visible_joints": total_valid_joints,
        "correct_joints": total_correct_joints,
        "mean_pixel_error": overall_mean_pixel_error,
        "mean_normalized_error": overall_mean_normalized_error,
        "mean_pck_percent": overall_pck,
    }

    per_joint_results = []

    for joint_index, joint_name in enumerate(JOINT_NAMES):
        joint_valid_mask = valid_mask[
            :,
            joint_index,
        ]

        joint_correct_mask = correct_mask[
            :,
            joint_index,
        ]

        valid_count = int(
            np.sum(joint_valid_mask)
        )

        correct_count = int(
            np.sum(joint_correct_mask)
        )

        joint_pck = safe_percentage(
            correct_count=correct_count,
            total_count=valid_count,
        )

        joint_pixel_error = safe_mean(
            distances[
                joint_valid_mask,
                joint_index,
            ]
        )

        joint_normalized_error = safe_mean(
            normalized_errors[
                joint_valid_mask,
                joint_index,
            ]
        )

        per_joint_results.append(
            {
                "joint_index": joint_index,
                "joint_name": joint_name,
                "valid_count": valid_count,
                "correct_count": correct_count,
                "mean_pixel_error": joint_pixel_error,
                "mean_normalized_error": joint_normalized_error,
                "pck_percent": joint_pck,
            }
        )

    return summary, per_joint_results


# ============================================================
# 6. CSV output
# ============================================================

def write_summary_csv(
    summary: dict,
) -> None:
    fieldnames = [
        "model",
        "video_id",
        "pck_threshold",
        "number_of_frames",
        "valid_frames",
        "total_visible_joints",
        "correct_joints",
        "mean_pixel_error",
        "mean_normalized_error",
        "mean_pck_percent",
    ]

    with SUMMARY_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerow(summary)


def write_per_joint_csv(
    per_joint_results: list[dict],
) -> None:
    fieldnames = [
        "joint_index",
        "joint_name",
        "valid_count",
        "correct_count",
        "mean_pixel_error",
        "mean_normalized_error",
        "pck_percent",
    ]

    with PER_JOINT_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(per_joint_results)


# ============================================================
# 7. Plot
# ============================================================

def create_per_joint_pck_plot(
    per_joint_results: list[dict],
    overall_pck: float,
) -> None:
    joint_names = [
        result["joint_name"]
        for result in per_joint_results
    ]

    pck_values = np.asarray(
        [
            result["pck_percent"]
            for result in per_joint_results
        ],
        dtype=np.float32,
    )

    x_positions = np.arange(
        len(joint_names)
    )

    figure, axis = plt.subplots(
        figsize=(14, 7),
    )

    bars = axis.bar(
        x_positions,
        pck_values,
    )

    axis.axhline(
        overall_pck,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean PCK: {overall_pck:.2f}%",
    )

    axis.set_title(
        f"Spiking ResNet18 Heatmap Per-Joint PCK@{PCK_THRESHOLD}",
        fontsize=15,
        pad=15,
    )

    axis.set_xlabel(
        "Joint",
        fontsize=12,
    )

    axis.set_ylabel(
        f"PCK@{PCK_THRESHOLD} (%)",
        fontsize=12,
    )

    axis.set_xticks(
        x_positions,
    )

    axis.set_xticklabels(
        joint_names,
        rotation=35,
        ha="right",
    )

    axis.set_ylim(
        0,
        105,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.legend(
        loc="lower right",
    )

    for bar, value in zip(
        bars,
        pck_values,
    ):
        if np.isnan(value):
            label = "N/A"
            label_height = 1.0
        else:
            label = f"{value:.1f}%"
            label_height = float(value) + 1.0

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            label_height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()

    figure.savefig(
        PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 8. Console output
# ============================================================

def format_metric(
    value: float,
    decimals: int = 4,
) -> str:
    if np.isnan(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def print_results(
    summary: dict,
    per_joint_results: list[dict],
) -> None:
    print("\n" + "=" * 82)
    print("Spiking ResNet18 Heatmap Evaluation Results")
    print("=" * 82)

    print(f"Video ID                  : {summary['video_id']}")
    print(f"Frames                    : {summary['number_of_frames']}")
    print(f"Valid frames              : {summary['valid_frames']}")
    print(f"Visible joints            : {summary['total_visible_joints']}")
    print(f"PCK threshold             : {summary['pck_threshold']}")
    print(
        "Mean pixel error          : "
        f"{format_metric(summary['mean_pixel_error'], 3)} px"
    )
    print(
        "Mean normalized error     : "
        f"{format_metric(summary['mean_normalized_error'], 4)}"
    )
    print(
        f"Mean PCK@{PCK_THRESHOLD:<4}           : "
        f"{format_metric(summary['mean_pck_percent'], 2)}%"
    )

    print("\n" + "-" * 82)

    header = (
        f"{'Joint':<20}"
        f"{'Valid':>9}"
        f"{'Correct':>11}"
        f"{'Pixel Err':>13}"
        f"{'Norm Err':>12}"
        f"{'PCK (%)':>12}"
    )

    print(header)
    print("-" * 82)

    for result in per_joint_results:
        print(
            f"{result['joint_name']:<20}"
            f"{result['valid_count']:>9d}"
            f"{result['correct_count']:>11d}"
            f"{format_metric(result['mean_pixel_error'], 3):>13}"
            f"{format_metric(result['mean_normalized_error'], 4):>12}"
            f"{format_metric(result['pck_percent'], 2):>12}"
        )

    print("=" * 82)


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    print("=" * 72)
    print("Evaluate Spiking ResNet18 Heatmap Predictions")
    print("=" * 72)
    print(f"Prediction file : {PRED_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"PCK threshold   : {PCK_THRESHOLD}")
    print("=" * 72)

    prediction_data = load_prediction_data()

    predicted_xy = prediction_data[
        "predicted_xy"
    ]

    ground_truth_xy = prediction_data[
        "ground_truth_xy"
    ]

    visibility = prediction_data[
        "visibility"
    ]

    validate_shapes(
        predicted_xy=predicted_xy,
        ground_truth_xy=ground_truth_xy,
        visibility=visibility,
    )

    summary, per_joint_results = calculate_metrics(
        predicted_xy=predicted_xy,
        ground_truth_xy=ground_truth_xy,
        visibility=visibility,
    )

    write_summary_csv(summary)

    write_per_joint_csv(
        per_joint_results
    )

    create_per_joint_pck_plot(
        per_joint_results=per_joint_results,
        overall_pck=summary["mean_pck_percent"],
    )

    print_results(
        summary=summary,
        per_joint_results=per_joint_results,
    )

    print("\nSaved files:")
    print(f"Summary CSV  : {SUMMARY_CSV_PATH}")
    print(f"Per-joint CSV: {PER_JOINT_CSV_PATH}")
    print(f"Plot         : {PLOT_PATH}")


if __name__ == "__main__":
    main()