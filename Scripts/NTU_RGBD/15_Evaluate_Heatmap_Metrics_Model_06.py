# Scripts/NTU_RGBD/15_Evaluate_Heatmap_Metrics_Model_06.py

from __future__ import annotations

from pathlib import Path

from evaluation import (
    EvaluationConfig,
    run_npz_evaluation,
)


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

MODEL_VERSION = "06"

NPZ_DIR = (
    PROJECT_ROOT
    / "server_outputs"
    / "NTU_RGBD"
    / "13_Generate_MP4_Heatmap_Model_06"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "NTU_RGBD"
    / "15_Evaluate_Heatmap_Metrics_Model_06"
)


# ============================================================
# 2. Evaluation configuration
# ============================================================

CONFIG = EvaluationConfig(
    model_version=MODEL_VERSION,

    npz_dir=NPZ_DIR,
    output_dir=OUTPUT_DIR,

    filename_pattern=(
        "*_predictions_model_06.npz"
    ),

    # Original PCK:
    # normalized by the visible GT joint
    # bounding-box maximum side length.
    pck_threshold=0.10,

    # Approximate MPII-style PCKh.
    pckh_threshold=0.50,

    # Calibrated Head-to-Neck scale factor.
    # Keep this consistent with Model 12 and
    # ntu_frame_dataset.py.
    mpii_head_scale_factor=1.8,
)


# ============================================================
# 3. Main
# ============================================================

def main() -> None:
    run_npz_evaluation(
        config=CONFIG,
    )


if __name__ == "__main__":
    main()