# Scripts/NTU_RGBD/
# 24_Run_SpikeYOLO_Ablation_Pipeline_Test.py

from __future__ import annotations

from pathlib import Path
import json
import sys

import torch
from torch.utils.data import DataLoader


# ============================================================
# 1. Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# 2. Project imports
# ============================================================

from Scripts.common.paths import (
    NTU_METADATA_DIR,
    NTU_RGBD_DATASET_DIR,
    NTU_RGBD_OUTPUT_DIR,
)

from Scripts.common.reproducibility import (
    seed_everything,
    get_device,
)

from Scripts.NTU_RGBD.datasets import (
    NTUFrameDataset,
    build_train_transform,
    build_eval_transform,
)

from Scripts.NTU_RGBD.models import (
    count_parameters,
)

from Scripts.NTU_RGBD.models.spikeyolo_style_ilif_heatmap_experiment import (
    build_spikeyolo_style_ilif_heatmap_experiment,
)

from Scripts.NTU_RGBD.training import (
    HeatmapMSELoss,
    run_training,
)

from Scripts.NTU_RGBD.inference.video_generation import (
    VideoGenerationConfig,
    run_video_generation,
)

from evaluation import (
    EvaluationConfig,
    run_npz_evaluation,
)


# ============================================================
# 3. Test experiment identity
# ============================================================

EXPERIMENT_ID = "M21-B0-TEST"

MODEL_VERSION = "24_test"


# ============================================================
# 4. Experiment configuration
# ============================================================

# Baseline configuration equivalent to the original Model 21.
EXPERIMENT_CONFIG = {
    "fusion_type": "concat",
    "readout_type": "mean",
    "decoder_type": "default",
    "backbone_variant": "default",
    "num_steps": 2,
    "max_spikes": 4,
}


# ============================================================
# 5. Dataset and heatmap configuration
# ============================================================

SEED = 42

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_JOINTS = 25
HEATMAP_SIGMA = 2.0

FRAME_STRIDE = 5

PERSON_CROP = True
BBOX_EXPANSION = 0.25

# None uses the complete training/validation split.
#
# For a faster pipeline test, use small values such as:
# MAX_TRAIN_SAMPLES = 256
# MAX_VAL_SAMPLES = 64
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None


# ============================================================
# 6. SNN configuration
# ============================================================

BETA = 0.90
THRESHOLD = 1.0


# ============================================================
# 7. Training configuration
# ============================================================

EPOCHS = 1

BATCH_SIZE = 8
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PRINT_EVERY = 50

DEVICE_NAME = None
PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# 8. Video-generation configuration
# ============================================================

# Generate only one test video.
MAX_TEST_VIDEOS = 1

# Process every frame in the selected video.
VIDEO_FRAME_STRIDE = 1

OUTPUT_FPS = None

CONFIDENCE_THRESHOLD = 0.02

# Always regenerate during the pipeline test.
SKIP_EXISTING_VIDEOS = False

SAVE_PREDICTION_NPZ = True


# ============================================================
# 9. Evaluation configuration
# ============================================================

PCK_THRESHOLD = 0.10
PCKH_THRESHOLD = 0.50

MPII_HEAD_SCALE_FACTOR = 1.8


# ============================================================
# 10. Input paths
# ============================================================

TRAIN_CSV = (
    NTU_METADATA_DIR
    / "train_split.csv"
)

VAL_CSV = (
    NTU_METADATA_DIR
    / "val_split.csv"
)

TEST_CSV = (
    NTU_METADATA_DIR
    / "test_split.csv"
)

RGB_VIDEO_DIR = (
    NTU_RGBD_DATASET_DIR
    / "rgb_videos"
)

SKELETON_DIR = (
    NTU_RGBD_DATASET_DIR
    / "skeletons"
)


# ============================================================
# 11. Output paths
# ============================================================

PIPELINE_OUTPUT_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / "24_SpikeYOLO_Ablation_Pipeline_Test"
    / EXPERIMENT_ID
)

TRAINING_OUTPUT_DIR = (
    PIPELINE_OUTPUT_DIR
    / "training"
)

VIDEO_OUTPUT_DIR = (
    PIPELINE_OUTPUT_DIR
    / "video_generation"
)

METRICS_OUTPUT_DIR = (
    PIPELINE_OUTPUT_DIR
    / "metrics"
)

CONFIG_PATH = (
    PIPELINE_OUTPUT_DIR
    / "experiment_config.json"
)

STATUS_PATH = (
    PIPELINE_OUTPUT_DIR
    / "pipeline_status.json"
)

MODEL_PATH = (
    TRAINING_OUTPUT_DIR
    / "best_model.pt"
)


# ============================================================
# 12. Directory preparation
# ============================================================

def prepare_output_directories() -> None:
    for directory in (
        PIPELINE_OUTPUT_DIR,
        TRAINING_OUTPUT_DIR,
        VIDEO_OUTPUT_DIR,
        METRICS_OUTPUT_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 13. Configuration and status saving
# ============================================================

def save_experiment_configuration() -> None:
    configuration = {
        "experiment_id": EXPERIMENT_ID,
        "model_version": MODEL_VERSION,

        "model": {
            "num_joints": NUM_JOINTS,
            "num_steps": EXPERIMENT_CONFIG[
                "num_steps"
            ],
            "beta": BETA,
            "threshold": THRESHOLD,
            "max_spikes": EXPERIMENT_CONFIG[
                "max_spikes"
            ],
            "fusion_type": EXPERIMENT_CONFIG[
                "fusion_type"
            ],
            "readout_type": EXPERIMENT_CONFIG[
                "readout_type"
            ],
            "decoder_type": EXPERIMENT_CONFIG[
                "decoder_type"
            ],
            "backbone_variant": EXPERIMENT_CONFIG[
                "backbone_variant"
            ],
        },

        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "frame_stride": FRAME_STRIDE,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_val_samples": MAX_VAL_SAMPLES,
        },

        "video_generation": {
            "max_test_videos": MAX_TEST_VIDEOS,
            "frame_stride": VIDEO_FRAME_STRIDE,
            "confidence_threshold": (
                CONFIDENCE_THRESHOLD
            ),
        },

        "evaluation": {
            "pck_threshold": PCK_THRESHOLD,
            "pckh_threshold": PCKH_THRESHOLD,
            "mpii_head_scale_factor": (
                MPII_HEAD_SCALE_FACTOR
            ),
        },
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=4,
        )


def save_pipeline_status(
    training_completed: bool,
    video_generation_completed: bool,
    evaluation_completed: bool,
) -> None:
    status = {
        "experiment_id": EXPERIMENT_ID,
        "training_completed": (
            training_completed
        ),
        "video_generation_completed": (
            video_generation_completed
        ),
        "evaluation_completed": (
            evaluation_completed
        ),
    }

    with STATUS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            status,
            file,
            indent=4,
        )


# ============================================================
# 14. Dataset validation
# ============================================================

def validate_input_paths() -> None:
    required_paths = {
        "Train CSV": TRAIN_CSV,
        "Validation CSV": VAL_CSV,
        "Test CSV": TEST_CSV,
        "RGB video directory": RGB_VIDEO_DIR,
        "Skeleton directory": SKELETON_DIR,
    }

    missing_paths = []

    for name, path in required_paths.items():
        if not path.exists():
            missing_paths.append(
                f"{name}: {path}"
            )

    if missing_paths:
        message = "\n".join(
            missing_paths
        )

        raise FileNotFoundError(
            "Required NTU RGB+D paths are missing:\n"
            f"{message}"
        )


# ============================================================
# 15. Dataset construction
# ============================================================

def build_datasets(
) -> tuple[
    NTUFrameDataset,
    NTUFrameDataset,
]:
    train_dataset = NTUFrameDataset(
        metadata_csv=TRAIN_CSV,

        transform=build_train_transform(
            image_size=IMAGE_SIZE,
        ),

        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,

        frame_stride=FRAME_STRIDE,

        single_person_only=True,

        max_samples=MAX_TRAIN_SAMPLES,

        skeleton_cache_size=8,

        person_crop=PERSON_CROP,
        bbox_expansion=BBOX_EXPANSION,
    )

    val_dataset = NTUFrameDataset(
        metadata_csv=VAL_CSV,

        transform=build_eval_transform(
            image_size=IMAGE_SIZE,
        ),

        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=HEATMAP_SIGMA,

        frame_stride=FRAME_STRIDE,

        single_person_only=True,

        max_samples=MAX_VAL_SAMPLES,

        skeleton_cache_size=8,

        person_crop=PERSON_CROP,
        bbox_expansion=BBOX_EXPANSION,
    )

    return (
        train_dataset,
        val_dataset,
    )


def build_dataloaders(
    train_dataset: NTUFrameDataset,
    val_dataset: NTUFrameDataset,
) -> tuple[
    DataLoader,
    DataLoader,
]:
    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        drop_last=False,

        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        drop_last=False,

        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    return (
        train_loader,
        val_loader,
    )


# ============================================================
# 16. Model construction
# ============================================================

def build_experiment_model(
) -> torch.nn.Module:
    return (
        build_spikeyolo_style_ilif_heatmap_experiment(
            num_joints=NUM_JOINTS,

            num_steps=EXPERIMENT_CONFIG[
                "num_steps"
            ],

            beta=BETA,

            threshold=THRESHOLD,

            max_spikes=EXPERIMENT_CONFIG[
                "max_spikes"
            ],

            fusion_type=EXPERIMENT_CONFIG[
                "fusion_type"
            ],

            readout_type=EXPERIMENT_CONFIG[
                "readout_type"
            ],

            decoder_type=EXPERIMENT_CONFIG[
                "decoder_type"
            ],

            backbone_variant=EXPERIMENT_CONFIG[
                "backbone_variant"
            ],
        )
    )


# ============================================================
# 17. Checkpoint utilities
# ============================================================

def extract_state_dict(
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    if isinstance(
        checkpoint,
        dict,
    ):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = checkpoint.get(
                key
            )

            if isinstance(
                value,
                dict,
            ):
                return value

        if checkpoint and all(
            isinstance(
                value,
                torch.Tensor,
            )
            for value in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find a model state dictionary "
        "inside the checkpoint."
    )


def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        cleaned_key = key

        if cleaned_key.startswith(
            "module."
        ):
            cleaned_key = cleaned_key[
                len("module.") :
            ]

        cleaned_state_dict[
            cleaned_key
        ] = value

    return cleaned_state_dict


# ============================================================
# 18. Training stage
# ============================================================

def run_training_stage(
    device: torch.device,
) -> None:
    print()
    print("=" * 72)
    print("Stage 1/3: Training")
    print("=" * 72)

    train_dataset, val_dataset = (
        build_datasets()
    )

    train_loader, val_loader = (
        build_dataloaders(
            train_dataset,
            val_dataset,
        )
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    model = build_experiment_model()

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    print(
        f"Total parameters: {total_parameters:,}"
    )

    print(
        "Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    criterion = HeatmapMSELoss(
        reduction="mean",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    )

    history = run_training(
        model=model,

        train_loader=train_loader,
        val_loader=val_loader,

        optimizer=optimizer,
        criterion=criterion,

        device=device,

        epochs=EPOCHS,

        output_dir=TRAINING_OUTPUT_DIR,

        scheduler=scheduler,

        print_every=PRINT_EVERY,

        model_name=(
            "NTU RGB+D Model 21 "
            f"Ablation {EXPERIMENT_ID}"
        ),
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Training completed, but the best "
            f"checkpoint was not found: {MODEL_PATH}"
        )

    completed_epochs = len(
        history.get(
            "epoch",
            [],
        )
    )

    print()
    print(
        f"Completed epochs: {completed_epochs}"
    )

    if completed_epochs > 0:
        print(
            "Final train loss: "
            f"{history['train_loss'][-1]:.6f}"
        )

        print(
            "Final validation loss: "
            f"{history['val_loss'][-1]:.6f}"
        )

    print(
        f"Best model: {MODEL_PATH}"
    )


# ============================================================
# 19. Video model loader
# ============================================================

def load_trained_model(
    device: torch.device,
) -> torch.nn.Module:
    del device

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    model = build_experiment_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = remove_module_prefix(
        state_dict
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model


# ============================================================
# 20. Video transform builder
# ============================================================

def build_video_transform(
    image_size: int,
):
    return build_eval_transform(
        image_size=image_size,
    )


# ============================================================
# 21. Video-generation stage
# ============================================================

def run_video_generation_stage(
    device: torch.device,
) -> None:
    print()
    print("=" * 72)
    print("Stage 2/3: Video generation")
    print("=" * 72)

    video_config = VideoGenerationConfig(
        model_version=MODEL_VERSION,

        test_csv=TEST_CSV,

        rgb_video_dir=RGB_VIDEO_DIR,

        skeleton_dir=SKELETON_DIR,

        output_dir=VIDEO_OUTPUT_DIR,

        image_size=IMAGE_SIZE,

        heatmap_size=HEATMAP_SIZE,

        num_joints=NUM_JOINTS,

        person_crop=PERSON_CROP,

        bbox_expansion=BBOX_EXPANSION,

        confidence_threshold=(
            CONFIDENCE_THRESHOLD
        ),

        frame_stride=VIDEO_FRAME_STRIDE,

        output_fps=OUTPUT_FPS,

        skip_existing_videos=(
            SKIP_EXISTING_VIDEOS
        ),

        save_prediction_npz=(
            SAVE_PREDICTION_NPZ
        ),

        max_test_videos=MAX_TEST_VIDEOS,

        readout_type=(
            EXPERIMENT_CONFIG[
                "readout_type"
            ]
        ),

        prediction_label=(
            "Prediction - "
            f"{EXPERIMENT_ID}"
        ),

        num_steps=EXPERIMENT_CONFIG[
            "num_steps"
        ],

        beta=BETA,

        threshold=THRESHOLD,
    )

    run_video_generation(
        config=video_config,

        model_loader=load_trained_model,

        transform_builder=(
            build_video_transform
        ),

        device=device,
    )

    prediction_files = list(
        VIDEO_OUTPUT_DIR.glob(
            "*_predictions_*.npz"
        )
    )

    if not prediction_files:
        raise FileNotFoundError(
            "Video generation completed, but no "
            "prediction NPZ file was found in: "
            f"{VIDEO_OUTPUT_DIR}"
        )

    print()
    print(
        "Generated prediction files:"
    )

    for path in prediction_files:
        print(
            f"  {path}"
        )


# ============================================================
# 22. Evaluation stage
# ============================================================

def run_evaluation_stage() -> None:
    print()
    print("=" * 72)
    print("Stage 3/3: Metrics evaluation")
    print("=" * 72)

    evaluation_config = EvaluationConfig(
        model_version=MODEL_VERSION,

        npz_dir=VIDEO_OUTPUT_DIR,

        output_dir=METRICS_OUTPUT_DIR,

        filename_pattern=(
            f"*_predictions_model_"
            f"{MODEL_VERSION}.npz"
        ),

        pck_threshold=PCK_THRESHOLD,

        pckh_threshold=PCKH_THRESHOLD,

        mpii_head_scale_factor=(
            MPII_HEAD_SCALE_FACTOR
        ),
    )

    run_npz_evaluation(
        config=evaluation_config,
    )

    print()
    print(
        f"Metrics directory: {METRICS_OUTPUT_DIR}"
    )


# ============================================================
# 23. Pipeline summary
# ============================================================

def print_pipeline_summary() -> None:
    print()
    print("=" * 72)
    print("Pipeline test completed")
    print("=" * 72)

    print(
        f"Experiment:       {EXPERIMENT_ID}"
    )

    print(
        f"Checkpoint:       {MODEL_PATH}"
    )

    print(
        f"Video/NPZ output: {VIDEO_OUTPUT_DIR}"
    )

    print(
        f"Metrics output:   {METRICS_OUTPUT_DIR}"
    )

    print(
        f"Configuration:    {CONFIG_PATH}"
    )

    print(
        f"Status:           {STATUS_PATH}"
    )

    print("=" * 72)


# ============================================================
# 24. Main
# ============================================================

def main() -> None:
    prepare_output_directories()

    save_experiment_configuration()

    save_pipeline_status(
        training_completed=False,
        video_generation_completed=False,
        evaluation_completed=False,
    )

    validate_input_paths()

    seed_everything(
        seed=SEED,
        deterministic=False,
    )

    device = get_device(
        preferred=DEVICE_NAME,
        verbose=True,
    )

    print()
    print("=" * 72)
    print(
        "NTU RGB+D Model 21 "
        "Ablation Pipeline Test"
    )
    print("=" * 72)

    print(
        f"Experiment ID:       {EXPERIMENT_ID}"
    )

    print(
        f"Epochs:              {EPOCHS}"
    )

    print(
        f"Test videos:         {MAX_TEST_VIDEOS}"
    )

    print(
        f"Fusion type:         "
        f"{EXPERIMENT_CONFIG['fusion_type']}"
    )

    print(
        f"Readout type:        "
        f"{EXPERIMENT_CONFIG['readout_type']}"
    )

    print(
        f"Decoder type:        "
        f"{EXPERIMENT_CONFIG['decoder_type']}"
    )

    print(
        f"Backbone variant:    "
        f"{EXPERIMENT_CONFIG['backbone_variant']}"
    )

    print(
        f"Number of steps:     "
        f"{EXPERIMENT_CONFIG['num_steps']}"
    )

    print(
        f"Maximum spikes:      "
        f"{EXPERIMENT_CONFIG['max_spikes']}"
    )

    print(
        f"Output directory:    "
        f"{PIPELINE_OUTPUT_DIR}"
    )

    try:
        run_training_stage(
            device=device,
        )

        save_pipeline_status(
            training_completed=True,
            video_generation_completed=False,
            evaluation_completed=False,
        )

        run_video_generation_stage(
            device=device,
        )

        save_pipeline_status(
            training_completed=True,
            video_generation_completed=True,
            evaluation_completed=False,
        )

        run_evaluation_stage()

        save_pipeline_status(
            training_completed=True,
            video_generation_completed=True,
            evaluation_completed=True,
        )

    except Exception:
        print()
        print("=" * 72)
        print("Pipeline test failed")
        print("=" * 72)

        print(
            f"Check partial outputs in: "
            f"{PIPELINE_OUTPUT_DIR}"
        )

        raise

    print_pipeline_summary()


if __name__ == "__main__":
    main()