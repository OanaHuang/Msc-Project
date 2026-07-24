# Scripts/NTU_RGBD/evaluation/run_evaluation.py

from __future__ import annotations

from .csv_writer import (
    print_saved_csv_paths,
    save_evaluation_csvs,
)

from .metric_config import (
    EvaluationConfig,
)

from .metric_runner import (
    compute_overall_metrics,
    compute_per_joint_metrics,
    print_overall_result,
    process_prediction_files,
)


def run_npz_evaluation(
    config: EvaluationConfig,
) -> dict[str, object]:
    """
    Run the complete NPZ evaluation pipeline for one model.

    The pipeline performs:

    1. Prediction-file discovery
    2. Per-video evaluation
    3. Overall metric aggregation
    4. Per-joint metric calculation
    5. CSV output
    6. Console summary

    Parameters
    ----------
    config:
        Evaluation configuration for one model.

    Returns
    -------
    dict[str, object]
        Overall metric summary.
    """
    config.create_output_dir()

    (
        per_video_results,
        aggregate,
        failed_files,
    ) = process_prediction_files(
        config=config,
    )

    summary = compute_overall_metrics(
        aggregate=aggregate,
        successful_files=len(
            per_video_results
        ),
        failed_files=failed_files,
        config=config,
    )

    per_joint_results = (
        compute_per_joint_metrics(
            aggregate=aggregate,
            config=config,
        )
    )

    save_evaluation_csvs(
        summary=summary,
        per_video_results=(
            per_video_results
        ),
        per_joint_results=(
            per_joint_results
        ),
        summary_csv_path=(
            config.summary_csv_path
        ),
        per_video_csv_path=(
            config.per_video_csv_path
        ),
        per_joint_csv_path=(
            config.per_joint_csv_path
        ),
    )

    print_overall_result(
        summary=summary,
        config=config,
    )

    print_saved_csv_paths(
        summary_csv_path=(
            config.summary_csv_path
        ),
        per_video_csv_path=(
            config.per_video_csv_path
        ),
        per_joint_csv_path=(
            config.per_joint_csv_path
        ),
    )

    return summary