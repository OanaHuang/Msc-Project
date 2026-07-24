# Scripts/NTU_RGBD/evaluation/metric_runner.py

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metric_config import EvaluationConfig
from .npz_loader import (
    find_prediction_files,
    load_prediction_file,
)
from .pose_metrics import (
    compute_coordinate_metrics,
    compute_distances,
    compute_head_lengths,
    compute_head_normalized_nme,
    compute_mean_confidence,
    compute_pck,
    compute_pck_normalized_nme,
    compute_pck_reference_lengths,
    compute_pckh,
    compute_pixel_errors,
    compute_threshold_accuracy,
    compute_valid_mask,
    compute_visibility_metrics,
    summarize_reference_lengths,
)


# ============================================================
# 1. Array helpers
# ============================================================

def concatenate_arrays(
    arrays: list[np.ndarray],
) -> np.ndarray:
    """
    Concatenate frame-level arrays from multiple videos.
    """
    if not arrays:
        raise ValueError(
            "Cannot concatenate an empty array list."
        )

    return np.concatenate(
        arrays,
        axis=0,
    )


def create_empty_aggregate(
) -> dict[str, list[np.ndarray]]:
    """
    Create the shared array container used for overall metrics.
    """
    return {
        "predictions": [],
        "ground_truth": [],
        "visibility": [],
        "distances": [],
        "valid_mask": [],
        "pck_reference_lengths": [],
        "head_lengths": [],
    }


# ============================================================
# 2. One-video evaluation
# ============================================================

def evaluate_one_video(
    item: dict[str, object],
    npz_path: Path,
    config: EvaluationConfig,
) -> tuple[
    dict[str, object],
    dict[str, np.ndarray],
]:
    """
    Evaluate one prediction NPZ file.

    Returns
    -------
    tuple
        First item:
            Dictionary containing per-video metrics.

        Second item:
            Arrays required for later overall and per-joint
            aggregation.
    """
    predictions = np.asarray(
        item["predictions"],
        dtype=np.float32,
    )

    ground_truth = np.asarray(
        item["ground_truth"],
        dtype=np.float32,
    )

    visibility = np.asarray(
        item["visibility"],
        dtype=np.float32,
    )

    predicted_visibility = item[
        "predicted_visibility"
    ]

    confidences = item[
        "confidences"
    ]

    distances = compute_distances(
        predictions=predictions,
        ground_truth=ground_truth,
    )

    valid_mask = compute_valid_mask(
        predictions=predictions,
        ground_truth=ground_truth,
        visibility=visibility,
    )

    pck_reference_lengths = (
        compute_pck_reference_lengths(
            ground_truth=ground_truth,
            visibility=visibility,
            epsilon=config.scale_epsilon,
        )
    )

    head_lengths = compute_head_lengths(
        ground_truth=ground_truth,
        visibility=visibility,
        config=config,
    )

    mse, rmse, mae = (
        compute_coordinate_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            valid_mask=valid_mask,
        )
    )

    (
        mean_pixel_error,
        median_pixel_error,
    ) = compute_pixel_errors(
        distances=distances,
        valid_mask=valid_mask,
    )

    pck_normalized_nme = (
        compute_pck_normalized_nme(
            distances=distances,
            valid_mask=valid_mask,
            reference_lengths=(
                pck_reference_lengths
            ),
            config=config,
        )
    )

    head_normalized_nme = (
        compute_head_normalized_nme(
            distances=distances,
            valid_mask=valid_mask,
            head_lengths=head_lengths,
            config=config,
        )
    )

    pck_score = compute_pck(
        distances=distances,
        valid_mask=valid_mask,
        reference_lengths=(
            pck_reference_lengths
        ),
        config=config,
    )

    pckh_score = compute_pckh(
        distances=distances,
        valid_mask=valid_mask,
        head_lengths=head_lengths,
        config=config,
    )

    pck_scale_summary = (
        summarize_reference_lengths(
            reference_lengths=(
                pck_reference_lengths
            ),
            epsilon=config.scale_epsilon,
        )
    )

    head_scale_summary = (
        summarize_reference_lengths(
            reference_lengths=head_lengths,
            epsilon=config.scale_epsilon,
        )
    )

    num_frames = int(
        predictions.shape[0]
    )

    num_joints = int(
        predictions.shape[1]
    )

    visible_keypoints = int(
        valid_mask.sum()
    )

    total_keypoints = int(
        valid_mask.size
    )

    result: dict[str, object] = {
        "sample_id": item[
            "sample_id"
        ],
        "model_version": item[
            "model_version"
        ],
        "npz_file": npz_path.name,

        "num_frames": num_frames,
        "num_joints": num_joints,

        "visible_keypoints": (
            visible_keypoints
        ),
        "total_keypoints": (
            total_keypoints
        ),
        "visibility_ratio_percent": (
            visible_keypoints
            / max(total_keypoints, 1)
            * 100.0
        ),

        "pck_scale_definition": (
            config.pck_scale_description
        ),
        "pck_threshold": (
            config.pck_threshold
        ),

        "valid_pck_reference_frames": (
            pck_scale_summary[
                "valid_frames"
            ]
        ),
        "invalid_pck_reference_frames": (
            pck_scale_summary[
                "invalid_frames"
            ]
        ),
        "valid_pck_reference_frame_ratio_percent": (
            pck_scale_summary[
                "valid_frame_ratio_percent"
            ]
        ),
        "mean_pck_reference_length_px": (
            pck_scale_summary[
                "mean_length_px"
            ]
        ),
        "median_pck_reference_length_px": (
            pck_scale_summary[
                "median_length_px"
            ]
        ),
        "minimum_pck_reference_length_px": (
            pck_scale_summary[
                "minimum_length_px"
            ]
        ),
        "maximum_pck_reference_length_px": (
            pck_scale_summary[
                "maximum_length_px"
            ]
        ),

        "head_scale_definition": (
            config.head_scale_description
        ),
        "mpii_head_scale_factor": (
            config.mpii_head_scale_factor
        ),
        "pckh_threshold": (
            config.pckh_threshold
        ),

        "valid_head_frames": (
            head_scale_summary[
                "valid_frames"
            ]
        ),
        "invalid_head_frames": (
            head_scale_summary[
                "invalid_frames"
            ]
        ),
        "valid_head_frame_ratio_percent": (
            head_scale_summary[
                "valid_frame_ratio_percent"
            ]
        ),
        "mean_head_length_px": (
            head_scale_summary[
                "mean_length_px"
            ]
        ),
        "median_head_length_px": (
            head_scale_summary[
                "median_length_px"
            ]
        ),
        "minimum_head_length_px": (
            head_scale_summary[
                "minimum_length_px"
            ]
        ),
        "maximum_head_length_px": (
            head_scale_summary[
                "maximum_length_px"
            ]
        ),

        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,

        "mean_pixel_error": (
            mean_pixel_error
        ),
        "median_pixel_error": (
            median_pixel_error
        ),

        "PCK_normalized_NME": (
            pck_normalized_nme
        ),
        "head_normalized_NME": (
            head_normalized_nme
        ),

        f"PCK@{config.pck_threshold}": (
            pck_score
        ),
        f"PCKh@{config.pckh_threshold}": (
            pckh_score
        ),
    }

    visibility_metrics = (
        compute_visibility_metrics(
            ground_truth_visibility=(
                visibility
            ),
            predicted_visibility=(
                predicted_visibility
            ),
        )
    )

    result.update(
        visibility_metrics
    )

    mean_confidence = (
        compute_mean_confidence(
            confidences=confidences,
        )
    )

    if np.isfinite(
        mean_confidence
    ):
        result[
            "mean_confidence"
        ] = mean_confidence

    arrays = {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "visibility": visibility,
        "distances": distances,
        "valid_mask": valid_mask,
        "pck_reference_lengths": (
            pck_reference_lengths
        ),
        "head_lengths": head_lengths,
    }

    return result, arrays


# ============================================================
# 3. Overall evaluation
# ============================================================

def compute_overall_metrics(
    aggregate: dict[str, list[np.ndarray]],
    successful_files: int,
    failed_files: int,
    config: EvaluationConfig,
) -> dict[str, object]:
    """
    Compute metrics over all successfully loaded videos.

    All frame arrays are concatenated before metric calculation,
    so the result is weighted by the number of valid keypoints,
    rather than averaging video-level percentages.
    """
    predictions = concatenate_arrays(
        aggregate["predictions"]
    )

    ground_truth = concatenate_arrays(
        aggregate["ground_truth"]
    )

    distances = concatenate_arrays(
        aggregate["distances"]
    )

    valid_mask = concatenate_arrays(
        aggregate["valid_mask"]
    )

    pck_reference_lengths = (
        concatenate_arrays(
            aggregate[
                "pck_reference_lengths"
            ]
        )
    )

    head_lengths = concatenate_arrays(
        aggregate["head_lengths"]
    )

    mse, rmse, mae = (
        compute_coordinate_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            valid_mask=valid_mask,
        )
    )

    (
        mean_pixel_error,
        median_pixel_error,
    ) = compute_pixel_errors(
        distances=distances,
        valid_mask=valid_mask,
    )

    pck_normalized_nme = (
        compute_pck_normalized_nme(
            distances=distances,
            valid_mask=valid_mask,
            reference_lengths=(
                pck_reference_lengths
            ),
            config=config,
        )
    )

    head_normalized_nme = (
        compute_head_normalized_nme(
            distances=distances,
            valid_mask=valid_mask,
            head_lengths=head_lengths,
            config=config,
        )
    )

    pck_score = compute_pck(
        distances=distances,
        valid_mask=valid_mask,
        reference_lengths=(
            pck_reference_lengths
        ),
        config=config,
    )

    pckh_score = compute_pckh(
        distances=distances,
        valid_mask=valid_mask,
        head_lengths=head_lengths,
        config=config,
    )

    pck_scale_summary = (
        summarize_reference_lengths(
            reference_lengths=(
                pck_reference_lengths
            ),
            epsilon=config.scale_epsilon,
        )
    )

    head_scale_summary = (
        summarize_reference_lengths(
            reference_lengths=head_lengths,
            epsilon=config.scale_epsilon,
        )
    )

    visible_keypoints = int(
        valid_mask.sum()
    )

    total_keypoints = int(
        valid_mask.size
    )

    return {
        "model_version": (
            config.model_version
        ),
        "metrics": (
            f"PCK@{config.pck_threshold} and "
            f"PCKh@{config.pckh_threshold}"
        ),

        "pck_scale_definition": (
            config.pck_scale_description
        ),
        "pck_threshold": (
            config.pck_threshold
        ),

        "head_scale_definition": (
            config.head_scale_description
        ),
        "mpii_head_scale_factor": (
            config.mpii_head_scale_factor
        ),
        "pckh_threshold": (
            config.pckh_threshold
        ),

        "successful_npz_files": (
            successful_files
        ),
        "failed_npz_files": (
            failed_files
        ),

        "num_frames": int(
            predictions.shape[0]
        ),
        "num_joints": int(
            predictions.shape[1]
        ),

        "visible_keypoints": (
            visible_keypoints
        ),
        "total_keypoints": (
            total_keypoints
        ),
        "visibility_ratio_percent": (
            visible_keypoints
            / max(total_keypoints, 1)
            * 100.0
        ),

        "valid_pck_reference_frames": (
            pck_scale_summary[
                "valid_frames"
            ]
        ),
        "invalid_pck_reference_frames": (
            pck_scale_summary[
                "invalid_frames"
            ]
        ),
        "valid_pck_reference_frame_ratio_percent": (
            pck_scale_summary[
                "valid_frame_ratio_percent"
            ]
        ),
        "mean_pck_reference_length_px": (
            pck_scale_summary[
                "mean_length_px"
            ]
        ),
        "median_pck_reference_length_px": (
            pck_scale_summary[
                "median_length_px"
            ]
        ),
        "minimum_pck_reference_length_px": (
            pck_scale_summary[
                "minimum_length_px"
            ]
        ),
        "maximum_pck_reference_length_px": (
            pck_scale_summary[
                "maximum_length_px"
            ]
        ),

        "valid_head_frames": (
            head_scale_summary[
                "valid_frames"
            ]
        ),
        "invalid_head_frames": (
            head_scale_summary[
                "invalid_frames"
            ]
        ),
        "valid_head_frame_ratio_percent": (
            head_scale_summary[
                "valid_frame_ratio_percent"
            ]
        ),
        "mean_head_length_px": (
            head_scale_summary[
                "mean_length_px"
            ]
        ),
        "median_head_length_px": (
            head_scale_summary[
                "median_length_px"
            ]
        ),
        "minimum_head_length_px": (
            head_scale_summary[
                "minimum_length_px"
            ]
        ),
        "maximum_head_length_px": (
            head_scale_summary[
                "maximum_length_px"
            ]
        ),

        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,

        "mean_pixel_error": (
            mean_pixel_error
        ),
        "median_pixel_error": (
            median_pixel_error
        ),

        "PCK_normalized_NME": (
            pck_normalized_nme
        ),
        "head_normalized_NME": (
            head_normalized_nme
        ),

        f"PCK@{config.pck_threshold}": (
            pck_score
        ),
        f"PCKh@{config.pckh_threshold}": (
            pckh_score
        ),
    }


# ============================================================
# 4. Per-joint evaluation
# ============================================================

def compute_per_joint_metrics(
    aggregate: dict[str, list[np.ndarray]],
    config: EvaluationConfig,
) -> list[dict[str, object]]:
    """
    Compute PCK, PCKh, NME and pixel errors for each joint.
    """
    distances = concatenate_arrays(
        aggregate["distances"]
    )

    valid_mask = concatenate_arrays(
        aggregate["valid_mask"]
    )

    pck_reference_lengths = (
        concatenate_arrays(
            aggregate[
                "pck_reference_lengths"
            ]
        )
    )

    head_lengths = concatenate_arrays(
        aggregate["head_lengths"]
    )

    valid_pck_scale_mask = (
        np.isfinite(
            pck_reference_lengths
        )
        & (
            pck_reference_lengths
            > config.scale_epsilon
        )
    )

    valid_head_scale_mask = (
        np.isfinite(
            head_lengths
        )
        & (
            head_lengths
            > config.scale_epsilon
        )
    )

    num_joints = int(
        distances.shape[1]
    )

    results: list[
        dict[str, object]
    ] = []

    for joint_index in range(
        num_joints
    ):
        if joint_index < len(
            config.joint_names
        ):
            joint_name = (
                config.joint_names[
                    joint_index
                ]
            )

        else:
            joint_name = (
                f"joint_{joint_index}"
            )

        joint_distances = distances[
            :,
            joint_index,
        ]

        joint_valid_mask = valid_mask[
            :,
            joint_index,
        ]

        visible_distances = (
            joint_distances[
                joint_valid_mask
            ]
        )

        if visible_distances.size == 0:
            mean_pixel_error = np.nan
            median_pixel_error = np.nan

        else:
            mean_pixel_error = float(
                visible_distances.mean()
            )

            median_pixel_error = float(
                np.median(
                    visible_distances
                )
            )

        # ----------------------------------------------------
        # Original PCK
        # ----------------------------------------------------

        pck_mask = np.asarray(
            joint_valid_mask
            & valid_pck_scale_mask,
            dtype=bool,
        )

        pck_distances = joint_distances[
            pck_mask
        ]

        joint_pck_scales = (
            pck_reference_lengths[
                pck_mask
            ]
        )

        if pck_distances.size == 0:
            pck_normalized_nme = np.nan
            pck_score = np.nan

        else:
            pck_normalized_distances = (
                pck_distances
                / joint_pck_scales
            )

            pck_normalized_nme = float(
                np.mean(
                    pck_normalized_distances
                )
            )

            pck_score = float(
                np.mean(
                    pck_normalized_distances
                    <= config.pck_threshold
                )
                * 100.0
            )

        # ----------------------------------------------------
        # Approximate MPII-style PCKh
        # ----------------------------------------------------

        pckh_mask = np.asarray(
            joint_valid_mask
            & valid_head_scale_mask,
            dtype=bool,
        )

        pckh_distances = (
            joint_distances[
                pckh_mask
            ]
        )

        joint_head_scales = (
            head_lengths[
                pckh_mask
            ]
        )

        if pckh_distances.size == 0:
            head_normalized_nme = np.nan
            pckh_score = np.nan

        else:
            head_normalized_distances = (
                pckh_distances
                / joint_head_scales
            )

            head_normalized_nme = float(
                np.mean(
                    head_normalized_distances
                )
            )

            pckh_score = float(
                np.mean(
                    head_normalized_distances
                    <= config.pckh_threshold
                )
                * 100.0
            )

        results.append(
            {
                "joint_index": (
                    joint_index
                ),
                "joint_name": (
                    joint_name
                ),

                "visible_count": int(
                    visible_distances.size
                ),

                "mean_pixel_error": (
                    mean_pixel_error
                ),
                "median_pixel_error": (
                    median_pixel_error
                ),

                "valid_pck_count": int(
                    pck_distances.size
                ),
                "PCK_normalized_NME": (
                    pck_normalized_nme
                ),
                (
                    f"PCK@"
                    f"{config.pck_threshold}"
                ): pck_score,

                "valid_pckh_count": int(
                    pckh_distances.size
                ),
                "head_normalized_NME": (
                    head_normalized_nme
                ),
                (
                    f"PCKh@"
                    f"{config.pckh_threshold}"
                ): pckh_score,
            }
        )

    return results


# ============================================================
# 5. Console output helpers
# ============================================================

def print_evaluation_header(
    config: EvaluationConfig,
    npz_paths: list[Path],
) -> None:
    """
    Print model and metric configuration.
    """
    print("=" * 72)

    print(
        f"NTU RGB+D Model "
        f"{config.model_version} "
        "Metrics Evaluation"
    )

    print("=" * 72)

    print(
        f"NPZ directory: {config.npz_dir}"
    )

    print(
        f"NPZ files:     {len(npz_paths)}"
    )

    print(
        f"Output dir:    {config.output_dir}"
    )

    print(
        f"PCK metric:    "
        f"PCK@{config.pck_threshold}"
    )

    print(
        f"PCK scale:     "
        f"{config.pck_scale_description}"
    )

    print(
        f"PCKh metric:   "
        f"PCKh@{config.pckh_threshold}"
    )

    print(
        f"Head scale:    "
        f"{config.head_scale_description}"
    )

    print(
        f"Head factor:   "
        f"{config.mpii_head_scale_factor}"
    )

    print("=" * 72)


def print_video_result(
    result: dict[str, object],
    config: EvaluationConfig,
) -> None:
    """
    Print a compact one-video result line.
    """
    pck_key = (
        f"PCK@{config.pck_threshold}"
    )

    pckh_key = (
        f"PCKh@{config.pckh_threshold}"
    )

    print(
        "  "
        f"{pck_key}="
        f"{float(result[pck_key]):.2f}% | "
        f"{pckh_key}="
        f"{float(result[pckh_key]):.2f}% | "
        f"Mean error="
        f"{float(result['mean_pixel_error']):.2f}px | "
        f"Body scale="
        f"{float(result['mean_pck_reference_length_px']):.2f}px | "
        f"Head scale="
        f"{float(result['mean_head_length_px']):.2f}px"
    )


def print_overall_result(
    summary: dict[str, object],
    config: EvaluationConfig,
) -> None:
    """
    Print overall metrics after all videos are processed.
    """
    pck_key = (
        f"PCK@{config.pck_threshold}"
    )

    pckh_key = (
        f"PCKh@{config.pckh_threshold}"
    )

    print()

    print("=" * 72)

    print(
        f"Overall Model "
        f"{config.model_version} Metrics"
    )

    print("=" * 72)

    print(
        f"Successful NPZ files: "
        f"{summary['successful_npz_files']}"
    )

    print(
        f"Failed NPZ files:     "
        f"{summary['failed_npz_files']}"
    )

    print(
        f"Frames:               "
        f"{summary['num_frames']}"
    )

    print(
        f"Visible keypoints:    "
        f"{summary['visible_keypoints']}"
    )

    print(
        f"Mean PCK scale:       "
        f"{float(summary['mean_pck_reference_length_px']):.4f}px"
    )

    print(
        f"Median PCK scale:     "
        f"{float(summary['median_pck_reference_length_px']):.4f}px"
    )

    print(
        f"Valid head frames:    "
        f"{summary['valid_head_frames']}"
    )

    print(
        f"Invalid head frames:  "
        f"{summary['invalid_head_frames']}"
    )

    print(
        f"Valid head ratio:     "
        f"{float(summary['valid_head_frame_ratio_percent']):.2f}%"
    )

    print(
        f"Mean head scale:      "
        f"{float(summary['mean_head_length_px']):.4f}px"
    )

    print(
        f"Median head scale:    "
        f"{float(summary['median_head_length_px']):.4f}px"
    )

    print(
        f"MSE:                  "
        f"{float(summary['MSE']):.4f}"
    )

    print(
        f"RMSE:                 "
        f"{float(summary['RMSE']):.4f}"
    )

    print(
        f"MAE:                  "
        f"{float(summary['MAE']):.4f}"
    )

    print(
        f"Mean pixel error:     "
        f"{float(summary['mean_pixel_error']):.4f}px"
    )

    print(
        f"Median pixel error:   "
        f"{float(summary['median_pixel_error']):.4f}px"
    )

    print(
        f"PCK-normalized NME:   "
        f"{float(summary['PCK_normalized_NME']):.6f}"
    )

    print(
        f"Head-normalized NME:  "
        f"{float(summary['head_normalized_NME']):.6f}"
    )

    print(
        f"{pck_key}:              "
        f"{float(summary[pck_key]):.2f}%"
    )

    print(
        f"{pckh_key}:             "
        f"{float(summary[pckh_key]):.2f}%"
    )


# ============================================================
# 6. NPZ processing
# ============================================================

def process_prediction_files(
    config: EvaluationConfig,
) -> tuple[
    list[dict[str, object]],
    dict[str, list[np.ndarray]],
    int,
]:
    """
    Load and evaluate all prediction files for one model.

    Returns
    -------
    tuple
        per_video_results:
            Successful per-video result dictionaries.

        aggregate:
            Arrays for overall and per-joint metrics.

        failed_files:
            Number of NPZ files that failed evaluation.
    """
    npz_paths = find_prediction_files(
        config=config
    )

    print_evaluation_header(
        config=config,
        npz_paths=npz_paths,
    )

    per_video_results: list[
        dict[str, object]
    ] = []

    aggregate = create_empty_aggregate()

    failed_files = 0

    for file_index, npz_path in enumerate(
        npz_paths,
        start=1,
    ):
        print(
            f"[{file_index}/{len(npz_paths)}] "
            f"{npz_path.name}"
        )

        try:
            item = load_prediction_file(
                npz_path=npz_path,
                config=config,
            )

            (
                video_result,
                arrays,
            ) = evaluate_one_video(
                item=item,
                npz_path=npz_path,
                config=config,
            )

            per_video_results.append(
                video_result
            )

            for key in aggregate:
                aggregate[key].append(
                    arrays[key]
                )

            print_video_result(
                result=video_result,
                config=config,
            )

        except Exception as error:
            failed_files += 1

            print(
                "  Failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    if not per_video_results:
        raise RuntimeError(
            "所有 NPZ 文件均评估失败。"
        )

    return (
        per_video_results,
        aggregate,
        failed_files,
    )