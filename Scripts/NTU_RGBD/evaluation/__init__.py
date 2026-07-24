# Scripts/NTU_RGBD/evaluation/__init__.py

from .csv_writer import (
    print_saved_csv_paths,
    save_csv,
    save_evaluation_csvs,
)

from .metric_config import (
    DEFAULT_HEAD_SCALE_DESCRIPTION,
    DEFAULT_PCK_SCALE_DESCRIPTION,
    NTU_HEAD_INDEX,
    NTU_JOINT_NAMES,
    NTU_NECK_INDEX,
    EvaluationConfig,
)

from .metric_runner import (
    compute_overall_metrics,
    compute_per_joint_metrics,
    evaluate_one_video,
    print_evaluation_header,
    print_overall_result,
    print_video_result,
    process_prediction_files,
)

from .npz_loader import (
    find_prediction_files,
    load_prediction_file,
    read_scalar,
    validate_prediction_shapes,
)

from .pose_metrics import (
    compute_coordinate_metrics,
    compute_distances,
    compute_head_lengths,
    compute_head_normalized_nme,
    compute_mean_confidence,
    compute_normalized_mean_error,
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

from .run_evaluation import (
    run_npz_evaluation,
)


__all__ = [
    # ========================================================
    # Configuration
    # ========================================================
    "EvaluationConfig",
    "NTU_HEAD_INDEX",
    "NTU_NECK_INDEX",
    "NTU_JOINT_NAMES",
    "DEFAULT_PCK_SCALE_DESCRIPTION",
    "DEFAULT_HEAD_SCALE_DESCRIPTION",

    # ========================================================
    # Main evaluation entry
    # ========================================================
    "run_npz_evaluation",

    # ========================================================
    # NPZ loading
    # ========================================================
    "read_scalar",
    "validate_prediction_shapes",
    "load_prediction_file",
    "find_prediction_files",

    # ========================================================
    # Pose metrics
    # ========================================================
    "compute_distances",
    "compute_valid_mask",
    "compute_pck_reference_lengths",
    "compute_head_lengths",
    "compute_coordinate_metrics",
    "compute_pixel_errors",
    "compute_normalized_mean_error",
    "compute_pck_normalized_nme",
    "compute_head_normalized_nme",
    "compute_threshold_accuracy",
    "compute_pck",
    "compute_pckh",
    "summarize_reference_lengths",
    "compute_visibility_metrics",
    "compute_mean_confidence",

    # ========================================================
    # Evaluation runner
    # ========================================================
    "evaluate_one_video",
    "compute_overall_metrics",
    "compute_per_joint_metrics",
    "process_prediction_files",
    "print_evaluation_header",
    "print_video_result",
    "print_overall_result",

    # ========================================================
    # CSV output
    # ========================================================
    "save_csv",
    "save_evaluation_csvs",
    "print_saved_csv_paths",
]