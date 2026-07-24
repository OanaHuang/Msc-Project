# Scripts/NTU_RGBD/training/__init__.py

from .losses import (
    HeatmapMSELoss,
    temporal_velocity_loss,
    temporal_acceleration_loss,
)

from .trainer import (
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
    save_training_history_csv,
    plot_loss_curves,
    run_training,
)

from .evaluator import (
    heatmaps_to_keypoints,
    compute_batch_pixel_error,

    # Generic PCK
    compute_batch_pck,
    compute_pck_per_joint,
    compute_torso_normalization,

    # PCKh
    compute_batch_pckh,
    compute_pckh_per_joint,

    # Full evaluation
    evaluate_heatmap_model,
)


__all__ = [
    # Losses
    "HeatmapMSELoss",
    "temporal_velocity_loss",
    "temporal_acceleration_loss",

    # Training
    "train_one_epoch",
    "validate_one_epoch",
    "save_checkpoint",
    "save_training_history_csv",
    "plot_loss_curves",
    "run_training",

    # Heatmap decoding
    "heatmaps_to_keypoints",

    # Pixel error
    "compute_batch_pixel_error",

    # Generic PCK
    "compute_batch_pck",
    "compute_pck_per_joint",
    "compute_torso_normalization",

    # PCKh
    "compute_batch_pckh",
    "compute_pckh_per_joint",

    # Full evaluation
    "evaluate_heatmap_model",
]