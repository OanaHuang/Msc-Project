# Scripts/Penn Action Model Training/
# 15_Evaluate_Spiking_ResNet50_Heatmap_Metrics.py

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_NPZ_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "14_Generate_MP4_Spiking_ResNet50_Heatmap"
    / "0684_spiking_resnet50_imagenet_t4_predictions.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "15_Evaluate_Spiking_ResNet50_Heatmap_Metrics"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV_PATH = (
    OUTPUT_DIR
    / "spiking_resnet50_heatmap_metrics_summary.csv"
)

PER_JOINT_CSV_PATH = (
    OUTPUT_DIR
    / "spiking_resnet50_heatmap_metrics_per_joint.csv"
)

PER_FRAME_CSV_PATH = (
    OUTPUT_DIR
    / "spiking_resnet50_heatmap_metrics_per_frame.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR
    / "spiking_resnet50_heatmap_metrics_summary.json"
)

PCK_THRESHOLDS = [
    0.05,
    0.10,
    0.20,
    0.50,
]

DEFAULT_JOINT_NAMES = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


# ============================================================
# 2. Utilities
# ============================================================

def decode_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.ndarray) and value.ndim == 0:
        return decode_string(value.item())

    return str(value)


def safe_mean(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = np.isfinite(values)

    if not np.any(finite):
        return float("nan")

    return float(
        np.mean(values[finite])
    )


def safe_percentage(
    correct: np.ndarray,
    valid: np.ndarray,
) -> float:
    correct = np.asarray(
        correct,
        dtype=bool,
    )

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    valid_count = int(valid.sum())

    if valid_count == 0:
        return float("nan")

    return float(
        correct[valid].mean() * 100.0
    )


# ============================================================
# 3. PCK normalization
# ============================================================

def calculate_person_scale(
    ground_truth_xy: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    """
    Calculate a per-frame person scale from the visible GT joints.

    Scale = max(
        visible joint bounding-box width,
        visible joint bounding-box height
    )

    This produces normalized PCK thresholds such as:
        PCK@0.05
        PCK@0.10
        PCK@0.20
        PCK@0.50
    """

    frame_count = ground_truth_xy.shape[0]

    scales = np.full(
        frame_count,
        np.nan,
        dtype=np.float32,
    )

    for frame_index in range(frame_count):
        frame_visibility = (
            visibility[frame_index] > 0
        )

        frame_points = ground_truth_xy[
            frame_index,
            frame_visibility,
        ]

        if len(frame_points) < 2:
            continue

        finite = np.isfinite(
            frame_points
        ).all(axis=1)

        frame_points = frame_points[finite]

        if len(frame_points) < 2:
            continue

        minimum_xy = frame_points.min(
            axis=0
        )

        maximum_xy = frame_points.max(
            axis=0
        )

        box_width = float(
            maximum_xy[0] - minimum_xy[0]
        )

        box_height = float(
            maximum_xy[1] - minimum_xy[1]
        )

        scale = max(
            box_width,
            box_height,
        )

        if scale > 1e-6:
            scales[frame_index] = scale

    return scales


# ============================================================
# 4. Load predictions
# ============================================================

def load_prediction_data():
    if not PREDICTION_NPZ_PATH.exists():
        raise FileNotFoundError(
            "Prediction NPZ was not found:\n"
            f"{PREDICTION_NPZ_PATH}\n\n"
            "Run script 14 first."
        )

    data = np.load(
        PREDICTION_NPZ_PATH,
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
        data.close()

        raise KeyError(
            "Prediction NPZ is missing required arrays.\n"
            f"Missing: {missing_keys}\n"
            f"Available: {data.files}"
        )

    predicted_xy = np.asarray(
        data["predicted_xy"],
        dtype=np.float32,
    )

    ground_truth_xy = np.asarray(
        data["ground_truth_xy"],
        dtype=np.float32,
    )

    visibility = np.asarray(
        data["visibility"],
        dtype=np.float32,
    )

    if predicted_xy.shape != ground_truth_xy.shape:
        data.close()

        raise ValueError(
            "Prediction and ground-truth shapes do not match.\n"
            f"Prediction shape : {predicted_xy.shape}\n"
            f"Ground truth    : {ground_truth_xy.shape}"
        )

    if predicted_xy.ndim != 3 or predicted_xy.shape[-1] != 2:
        data.close()

        raise ValueError(
            "Expected coordinate shape [frames, joints, 2], "
            f"received {predicted_xy.shape}."
        )

    if visibility.shape != predicted_xy.shape[:2]:
        data.close()

        raise ValueError(
            "Visibility shape does not match coordinates.\n"
            f"Visibility : {visibility.shape}\n"
            f"Coordinates: {predicted_xy.shape}"
        )

    if "joint_names" in data.files:
        joint_names = [
            decode_string(value)
            for value in data["joint_names"]
        ]
    else:
        joint_names = DEFAULT_JOINT_NAMES.copy()

    if "image_paths" in data.files:
        image_paths = [
            decode_string(value)
            for value in data["image_paths"]
        ]
    else:
        image_paths = [
            ""
            for _ in range(predicted_xy.shape[0])
        ]

    if "sample_indices" in data.files:
        sample_indices = np.asarray(
            data["sample_indices"],
            dtype=np.int64,
        )
    else:
        sample_indices = np.arange(
            predicted_xy.shape[0],
            dtype=np.int64,
        )

    video_id = (
        decode_string(data["video_id"])
        if "video_id" in data.files
        else "unknown"
    )

    data.close()

    return {
        "predicted_xy": predicted_xy,
        "ground_truth_xy": ground_truth_xy,
        "visibility": visibility,
        "joint_names": joint_names,
        "image_paths": image_paths,
        "sample_indices": sample_indices,
        "video_id": video_id,
    }


# ============================================================
# 5. Metric calculation
# ============================================================

def calculate_metrics(
    predicted_xy: np.ndarray,
    ground_truth_xy: np.ndarray,
    visibility: np.ndarray,
):
    finite_prediction = np.isfinite(
        predicted_xy
    ).all(axis=-1)

    finite_ground_truth = np.isfinite(
        ground_truth_xy
    ).all(axis=-1)

    valid_joint = (
        (visibility > 0)
        & finite_prediction
        & finite_ground_truth
    )

    pixel_error = np.linalg.norm(
        predicted_xy - ground_truth_xy,
        axis=-1,
    )

    person_scale = calculate_person_scale(
        ground_truth_xy=ground_truth_xy,
        visibility=visibility,
    )

    valid_scale = (
        np.isfinite(person_scale)
        & (person_scale > 1e-6)
    )

    normalized_error = np.full_like(
        pixel_error,
        np.nan,
        dtype=np.float32,
    )

    normalizable = (
        valid_joint
        & valid_scale[:, None]
    )

    normalized_error[normalizable] = (
        pixel_error[normalizable]
        / np.broadcast_to(
            person_scale[:, None],
            pixel_error.shape,
        )[normalizable]
    )

    overall = {
        "valid_joint_count": int(
            valid_joint.sum()
        ),
        "mean_pixel_error": safe_mean(
            pixel_error[valid_joint]
        ),
        "normalized_mean_error": safe_mean(
            normalized_error[normalizable]
        ),
    }

    pck_results = {}

    for threshold in PCK_THRESHOLDS:
        correct = (
            normalized_error <= threshold
        )

        pck_value = safe_percentage(
            correct=correct,
            valid=normalizable,
        )

        pck_results[threshold] = pck_value

        overall[
            f"pck_{threshold:.2f}"
        ] = pck_value

    return {
        "valid_joint": valid_joint,
        "pixel_error": pixel_error,
        "person_scale": person_scale,
        "valid_scale": valid_scale,
        "normalized_error": normalized_error,
        "normalizable": normalizable,
        "pck_results": pck_results,
        "overall": overall,
    }


# ============================================================
# 6. Save summary
# ============================================================

def save_summary_csv(
    video_id: str,
    metrics: dict,
) -> None:
    overall = metrics["overall"]

    fieldnames = [
        "model",
        "video_id",
        "epochs",
        "valid_joint_count",
        "mean_pixel_error",
        "normalized_mean_error",
    ]

    fieldnames.extend(
        [
            f"PCK@{threshold:.2f}"
            for threshold in PCK_THRESHOLDS
        ]
    )

    row = {
        "model": (
            "Spiking ResNet50 Heatmap "
            "ImageNet T4"
        ),
        "video_id": video_id,
        "epochs": 20,
        "valid_joint_count": overall[
            "valid_joint_count"
        ],
        "mean_pixel_error": overall[
            "mean_pixel_error"
        ],
        "normalized_mean_error": overall[
            "normalized_mean_error"
        ],
    }

    for threshold in PCK_THRESHOLDS:
        row[
            f"PCK@{threshold:.2f}"
        ] = overall[
            f"pck_{threshold:.2f}"
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
        writer.writerow(row)


# ============================================================
# 7. Save per-joint metrics
# ============================================================

def save_per_joint_csv(
    joint_names: list[str],
    metrics: dict,
) -> None:
    valid_joint = metrics["valid_joint"]
    pixel_error = metrics["pixel_error"]
    normalized_error = metrics[
        "normalized_error"
    ]
    normalizable = metrics["normalizable"]

    fieldnames = [
        "joint_index",
        "joint_name",
        "valid_count",
        "mean_pixel_error",
        "normalized_mean_error",
    ]

    fieldnames.extend(
        [
            f"PCK@{threshold:.2f}"
            for threshold in PCK_THRESHOLDS
        ]
    )

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

        for joint_index, joint_name in enumerate(joint_names):
            joint_valid = valid_joint[
                :,
                joint_index,
            ]

            joint_normalizable = normalizable[
                :,
                joint_index,
            ]

            row = {
                "joint_index": joint_index,
                "joint_name": joint_name,
                "valid_count": int(
                    joint_valid.sum()
                ),
                "mean_pixel_error": safe_mean(
                    pixel_error[
                        joint_valid,
                        joint_index,
                    ]
                ),
                "normalized_mean_error": safe_mean(
                    normalized_error[
                        joint_normalizable,
                        joint_index,
                    ]
                ),
            }

            for threshold in PCK_THRESHOLDS:
                joint_correct = (
                    normalized_error[
                        :,
                        joint_index,
                    ]
                    <= threshold
                )

                row[
                    f"PCK@{threshold:.2f}"
                ] = safe_percentage(
                    correct=joint_correct,
                    valid=joint_normalizable,
                )

            writer.writerow(row)


# ============================================================
# 8. Save per-frame metrics
# ============================================================

def save_per_frame_csv(
    image_paths: list[str],
    sample_indices: np.ndarray,
    metrics: dict,
) -> None:
    valid_joint = metrics["valid_joint"]
    pixel_error = metrics["pixel_error"]
    person_scale = metrics["person_scale"]
    normalized_error = metrics[
        "normalized_error"
    ]
    normalizable = metrics["normalizable"]

    fieldnames = [
        "frame_position",
        "sample_index",
        "image_path",
        "person_scale",
        "valid_joint_count",
        "mean_pixel_error",
        "normalized_mean_error",
    ]

    fieldnames.extend(
        [
            f"PCK@{threshold:.2f}"
            for threshold in PCK_THRESHOLDS
        ]
    )

    with PER_FRAME_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for frame_index in range(
            valid_joint.shape[0]
        ):
            frame_valid = valid_joint[
                frame_index
            ]

            frame_normalizable = normalizable[
                frame_index
            ]

            row = {
                "frame_position": frame_index,
                "sample_index": int(
                    sample_indices[frame_index]
                ),
                "image_path": image_paths[
                    frame_index
                ],
                "person_scale": float(
                    person_scale[frame_index]
                ),
                "valid_joint_count": int(
                    frame_valid.sum()
                ),
                "mean_pixel_error": safe_mean(
                    pixel_error[
                        frame_index,
                        frame_valid,
                    ]
                ),
                "normalized_mean_error": safe_mean(
                    normalized_error[
                        frame_index,
                        frame_normalizable,
                    ]
                ),
            }

            for threshold in PCK_THRESHOLDS:
                frame_correct = (
                    normalized_error[
                        frame_index
                    ]
                    <= threshold
                )

                row[
                    f"PCK@{threshold:.2f}"
                ] = safe_percentage(
                    correct=frame_correct,
                    valid=frame_normalizable,
                )

            writer.writerow(row)


# ============================================================
# 9. Save JSON
# ============================================================

def save_summary_json(
    video_id: str,
    frame_count: int,
    metrics: dict,
) -> None:
    summary = {
        "model": (
            "Spiking ResNet50 Heatmap "
            "ImageNet T4"
        ),
        "source_script": (
            "09b_Spiking_ResNet50_"
            "Heatmap_ImageNet_T4.py"
        ),
        "prediction_file": str(
            PREDICTION_NPZ_PATH
        ),
        "video_id": video_id,
        "epochs": 20,
        "frame_count": frame_count,
        "normalization": (
            "Maximum side length of the "
            "visible ground-truth joint bounding box"
        ),
        "metrics": metrics["overall"],
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


# ============================================================
# 10. Print results
# ============================================================

def print_results(
    video_id: str,
    frame_count: int,
    metrics: dict,
) -> None:
    overall = metrics["overall"]

    print("\n" + "=" * 72)
    print("Spiking ResNet50 Heatmap Metrics")
    print("=" * 72)
    print(f"Video ID              : {video_id}")
    print(f"Frames                : {frame_count}")
    print(f"Valid joint samples   : {overall['valid_joint_count']}")
    print(
        "Mean pixel error       : "
        f"{overall['mean_pixel_error']:.4f}"
    )
    print(
        "Normalized mean error  : "
        f"{overall['normalized_mean_error']:.6f}"
    )

    for threshold in PCK_THRESHOLDS:
        value = overall[
            f"pck_{threshold:.2f}"
        ]

        print(
            f"PCK@{threshold:.2f}"
            f"{' ' * max(1, 18 - len(f'PCK@{threshold:.2f}'))}: "
            f"{value:.2f}%"
        )

    print("-" * 72)
    print(f"Summary CSV  : {SUMMARY_CSV_PATH}")
    print(f"Per-joint CSV: {PER_JOINT_CSV_PATH}")
    print(f"Per-frame CSV: {PER_FRAME_CSV_PATH}")
    print(f"Summary JSON : {SUMMARY_JSON_PATH}")
    print("=" * 72)


# ============================================================
# 11. Main
# ============================================================

def main() -> None:
    print("=" * 72)
    print("15 - Evaluate Spiking ResNet50 Heatmap Metrics")
    print("=" * 72)
    print(f"Prediction: {PREDICTION_NPZ_PATH}")
    print(f"Output    : {OUTPUT_DIR}")
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

    joint_names = prediction_data[
        "joint_names"
    ]

    image_paths = prediction_data[
        "image_paths"
    ]

    sample_indices = prediction_data[
        "sample_indices"
    ]

    video_id = prediction_data[
        "video_id"
    ]

    metrics = calculate_metrics(
        predicted_xy=predicted_xy,
        ground_truth_xy=ground_truth_xy,
        visibility=visibility,
    )

    save_summary_csv(
        video_id=video_id,
        metrics=metrics,
    )

    save_per_joint_csv(
        joint_names=joint_names,
        metrics=metrics,
    )

    save_per_frame_csv(
        image_paths=image_paths,
        sample_indices=sample_indices,
        metrics=metrics,
    )

    save_summary_json(
        video_id=video_id,
        frame_count=predicted_xy.shape[0],
        metrics=metrics,
    )

    print_results(
        video_id=video_id,
        frame_count=predicted_xy.shape[0],
        metrics=metrics,
    )


if __name__ == "__main__":
    main()