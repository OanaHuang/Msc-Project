# Scripts/NTU_RGBD/25_Run_SpikeYOLO_Ablation_Experiments.py

from __future__ import annotations

from pathlib import Path
import csv
import json
import re
import sys
import traceback

import torch
from torch.utils.data import DataLoader


# ============================================================
# 1. Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
# 3. Pipeline identity
# ============================================================

PIPELINE_NUMBER = "25"
PIPELINE_NAME = "25_SpikeYOLO_Ablation_Experiments"


# ============================================================
# 4. Experiment definitions
# ============================================================

EXPERIMENTS = {
    "M21-B0": {
        "name": "Baseline",
        "fusion_type": "concat",
        "readout_type": "mean",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 4,
    },
    "M21-S1": {
        "name": "Stage2 only",
        "fusion_type": "stage2_only",
        "readout_type": "mean",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 4,
    },
    "M21-S2": {
        "name": "Stage3 only",
        "fusion_type": "stage3_only",
        "readout_type": "mean",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 4,
    },
    "M21-N1": {
        "name": "Binary I-LIF",
        "fusion_type": "concat",
        "readout_type": "mean",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 1,
    },
    "M21-D2": {
        "name": "I-LIF D=2",
        "fusion_type": "concat",
        "readout_type": "mean",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 2,
    },
    "M21-D8": {
        "name": "I-LIF D=8",
        "fusion_type": "concat",
        "readout_type": "mean",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 8,
    },
    "M21-T1": {
        "name": "Single time step",
        "fusion_type": "concat",
        "readout_type": "mean",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 1,
        "max_spikes": 4,
    },
    "M21-T4": {
        "name": "Four time steps",
        "fusion_type": "concat",
        "readout_type": "mean",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 4,
        "max_spikes": 4,
    },
    "M21-R1": {
        "name": "Sum readout",
        "fusion_type": "concat",
        "readout_type": "sum",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 4,
    },
    "M21-R2": {
        "name": "Last-step readout",
        "fusion_type": "concat",
        "readout_type": "last",
        "decoder_type": "default",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 4,
    },
    "M21-C1": {
        "name": "No refinement",
        "fusion_type": "concat",
        "readout_type": "mean",
        "decoder_type": "no_refine",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 4,
    },
    "M21-C2": {
        "name": "Bilinear decoder",
        "fusion_type": "concat",
        "readout_type": "mean",
        "decoder_type": "bilinear",
        "backbone_variant": "default",
        "num_steps": 2,
        "max_spikes": 4,
    },
    "M21-F1": {
        "name": "Shallow backbone",
        "fusion_type": "concat",
        "readout_type": "mean",
        "decoder_type": "default",
        "backbone_variant": "shallow",
        "num_steps": 2,
        "max_spikes": 4,
    },
}

EXPERIMENTS_TO_RUN = ['M21-T4', 'M21-R1', 'M21-R2', 'M21-C1', 'M21-C2', 'M21-F1']


# ============================================================
# 5. Stage controls
# ============================================================

RUN_TRAINING = True
RUN_VIDEO_GENERATION = True
RUN_EVALUATION = True

# Re-running the script skips completed stages when their expected
# output files already exist.
SKIP_COMPLETED_STAGES = True

# Continue with the next experiment if one experiment fails.
CONTINUE_AFTER_FAILURE = True


# ============================================================
# 6. Dataset and heatmap configuration
# ============================================================

SEED = 42

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_JOINTS = 25
HEATMAP_SIGMA = 2.0

TRAIN_FRAME_STRIDE = 5

PERSON_CROP = True
BBOX_EXPANSION = 0.25

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None


# ============================================================
# 7. SNN configuration
# ============================================================

BETA = 0.90
THRESHOLD = 1.0


# ============================================================
# 8. Training configuration
# ============================================================

EPOCHS = 20

BATCH_SIZE = 8
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PRINT_EVERY = 50

DEVICE_NAME = None
PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# 9. Video-generation configuration
# ============================================================

# None processes every single-person test video.
MAX_TEST_VIDEOS = None

VIDEO_FRAME_STRIDE = 1
OUTPUT_FPS = None

CONFIDENCE_THRESHOLD = 0.02

SKIP_EXISTING_VIDEOS = True
SAVE_PREDICTION_NPZ = True


# ============================================================
# 10. Evaluation configuration
# ============================================================

PCK_THRESHOLD = 0.10
PCKH_THRESHOLD = 0.50
MPII_HEAD_SCALE_FACTOR = 1.8


# ============================================================
# 11. Input paths
# ============================================================

TRAIN_CSV = NTU_METADATA_DIR / "train_split.csv"
VAL_CSV = NTU_METADATA_DIR / "val_split.csv"
TEST_CSV = NTU_METADATA_DIR / "test_split.csv"

RGB_VIDEO_DIR = NTU_RGBD_DATASET_DIR / "rgb_videos"
SKELETON_DIR = NTU_RGBD_DATASET_DIR / "skeletons"


# ============================================================
# 12. Pipeline output paths
# ============================================================

PIPELINE_ROOT = (
    NTU_RGBD_OUTPUT_DIR
    / PIPELINE_NAME
)

SUMMARY_ALL_PATH = (
    PIPELINE_ROOT
    / "summary_all.csv"
)

PIPELINE_STATUS_PATH = (
    PIPELINE_ROOT
    / "pipeline_status.json"
)


# ============================================================
# 13. General helpers
# ============================================================

def clean_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("@", "_at_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def experiment_model_version(
    experiment_id: str,
) -> str:
    return clean_column_name(experiment_id)


def get_experiment_paths(
    experiment_id: str,
) -> dict[str, Path]:
    experiment_root = (
        PIPELINE_ROOT
        / experiment_id
    )

    training_dir = (
        experiment_root
        / "training"
    )

    video_dir = (
        experiment_root
        / "video_generation"
    )

    metrics_dir = (
        experiment_root
        / "metrics"
    )

    return {
        "root": experiment_root,
        "training": training_dir,
        "video": video_dir,
        "metrics": metrics_dir,
        "config": (
            experiment_root
            / "experiment_config.json"
        ),
        "status": (
            experiment_root
            / "pipeline_status.json"
        ),
        "error": (
            experiment_root
            / "error.txt"
        ),
        "model": (
            training_dir
            / "best_model.pt"
        ),
    }


def ensure_experiment_directories(
    paths: dict[str, Path],
) -> None:
    for key in (
        "root",
        "training",
        "video",
        "metrics",
    ):
        paths[key].mkdir(
            parents=True,
            exist_ok=True,
        )


def write_json(
    path: Path,
    data: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def update_experiment_status(
    paths: dict[str, Path],
    **updates,
) -> None:
    status = load_json(
        paths["status"]
    )

    status.update(
        updates
    )

    write_json(
        paths["status"],
        status,
    )


def save_experiment_configuration(
    experiment_id: str,
    experiment_config: dict,
    paths: dict[str, Path],
) -> None:
    configuration = {
        "pipeline_number": PIPELINE_NUMBER,
        "pipeline_name": PIPELINE_NAME,
        "experiment_id": experiment_id,
        "experiment_name": (
            experiment_config["name"]
        ),
        "model": {
            "num_joints": NUM_JOINTS,
            "num_steps": (
                experiment_config[
                    "num_steps"
                ]
            ),
            "beta": BETA,
            "threshold": THRESHOLD,
            "max_spikes": (
                experiment_config[
                    "max_spikes"
                ]
            ),
            "fusion_type": (
                experiment_config[
                    "fusion_type"
                ]
            ),
            "readout_type": (
                experiment_config[
                    "readout_type"
                ]
            ),
            "decoder_type": (
                experiment_config[
                    "decoder_type"
                ]
            ),
            "backbone_variant": (
                experiment_config[
                    "backbone_variant"
                ]
            ),
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "frame_stride": (
                TRAIN_FRAME_STRIDE
            ),
            "max_train_samples": (
                MAX_TRAIN_SAMPLES
            ),
            "max_val_samples": (
                MAX_VAL_SAMPLES
            ),
        },
        "video_generation": {
            "max_test_videos": (
                MAX_TEST_VIDEOS
            ),
            "frame_stride": (
                VIDEO_FRAME_STRIDE
            ),
            "confidence_threshold": (
                CONFIDENCE_THRESHOLD
            ),
        },
        "evaluation": {
            "pck_threshold": (
                PCK_THRESHOLD
            ),
            "pckh_threshold": (
                PCKH_THRESHOLD
            ),
            "mpii_head_scale_factor": (
                MPII_HEAD_SCALE_FACTOR
            ),
        },
    }

    write_json(
        paths["config"],
        configuration,
    )


def validate_input_paths() -> None:
    required_paths = {
        "Train CSV": TRAIN_CSV,
        "Validation CSV": VAL_CSV,
        "Test CSV": TEST_CSV,
        "RGB video directory": (
            RGB_VIDEO_DIR
        ),
        "Skeleton directory": (
            SKELETON_DIR
        ),
    }

    missing = [
        f"{name}: {path}"
        for name, path
        in required_paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required NTU RGB+D paths are missing:\n"
            + "\n".join(missing)
        )


def validate_experiment_assignment() -> None:
    unknown = [
        experiment_id
        for experiment_id
        in EXPERIMENTS_TO_RUN
        if experiment_id
        not in EXPERIMENTS
    ]

    if unknown:
        raise KeyError(
            "Unknown experiment IDs: "
            + ", ".join(unknown)
        )

    if len(
        set(EXPERIMENTS_TO_RUN)
    ) != len(EXPERIMENTS_TO_RUN):
        raise ValueError(
            "EXPERIMENTS_TO_RUN contains "
            "duplicate experiment IDs."
        )


# ============================================================
# 14. Dataset and DataLoader construction
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
        frame_stride=(
            TRAIN_FRAME_STRIDE
        ),
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
        frame_stride=(
            TRAIN_FRAME_STRIDE
        ),
        single_person_only=True,
        max_samples=MAX_VAL_SAMPLES,
        skeleton_cache_size=8,
        person_crop=PERSON_CROP,
        bbox_expansion=BBOX_EXPANSION,
    )

    return train_dataset, val_dataset


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

    return train_loader, val_loader


# ============================================================
# 15. Model construction and checkpoint loading
# ============================================================

def build_experiment_model(
    experiment_config: dict,
) -> torch.nn.Module:
    return (
        build_spikeyolo_style_ilif_heatmap_experiment(
            num_joints=NUM_JOINTS,
            num_steps=(
                experiment_config[
                    "num_steps"
                ]
            ),
            beta=BETA,
            threshold=THRESHOLD,
            max_spikes=(
                experiment_config[
                    "max_spikes"
                ]
            ),
            fusion_type=(
                experiment_config[
                    "fusion_type"
                ]
            ),
            readout_type=(
                experiment_config[
                    "readout_type"
                ]
            ),
            decoder_type=(
                experiment_config[
                    "decoder_type"
                ]
            ),
            backbone_variant=(
                experiment_config[
                    "backbone_variant"
                ]
            ),
        )
    )


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
            value = checkpoint.get(key)

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
            for value
            in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find a model state "
        "dictionary in the checkpoint."
    )


def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


def make_model_loader(
    experiment_config: dict,
    model_path: Path,
):
    def load_model(
        device: torch.device,
    ) -> torch.nn.Module:
        del device

        if not model_path.exists():
            raise FileNotFoundError(
                "Checkpoint not found: "
                f"{model_path}"
            )

        model = build_experiment_model(
            experiment_config
        )

        checkpoint = torch.load(
            model_path,
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

    return load_model


def build_video_transform(
    image_size: int,
):
    return build_eval_transform(
        image_size=image_size,
    )


# ============================================================
# 16. Completion checks
# ============================================================

def find_prediction_npz_files(
    video_dir: Path,
) -> list[Path]:
    return sorted(
        video_dir.glob(
            "*_predictions_*.npz"
        )
    )


def find_metrics_summary_file(
    metrics_dir: Path,
) -> Path | None:
    files = sorted(
        metrics_dir.glob(
            "*_metrics_summary.csv"
        )
    )

    return files[0] if files else None


def find_metrics_per_joint_file(
    metrics_dir: Path,
) -> Path | None:
    files = sorted(
        metrics_dir.glob(
            "*_metrics_per_joint.csv"
        )
    )

    return files[0] if files else None


def training_is_complete(
    paths: dict[str, Path],
) -> bool:
    return (
        paths["model"].exists()
        and paths["model"].stat().st_size > 0
    )


def video_generation_is_complete(
    paths: dict[str, Path],
) -> bool:
    return bool(
        find_prediction_npz_files(
            paths["video"]
        )
    )


def evaluation_is_complete(
    paths: dict[str, Path],
) -> bool:
    return (
        find_metrics_summary_file(
            paths["metrics"]
        )
        is not None
        and find_metrics_per_joint_file(
            paths["metrics"]
        )
        is not None
    )


# ============================================================
# 17. Training stage
# ============================================================

def run_training_stage(
    experiment_id: str,
    experiment_config: dict,
    paths: dict[str, Path],
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> None:
    if (
        SKIP_COMPLETED_STAGES
        and training_is_complete(paths)
    ):
        print(
            "Training already completed; "
            "skipping."
        )

        update_experiment_status(
            paths,
            training_completed=True,
        )

        return

    print()
    print("-" * 72)
    print(
        f"Training {experiment_id}"
    )
    print("-" * 72)

    seed_everything(
        seed=SEED,
        deterministic=False,
    )

    model = build_experiment_model(
        experiment_config
    )

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    print(
        "Total parameters:     "
        f"{total_parameters:,}"
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
        output_dir=paths["training"],
        scheduler=scheduler,
        print_every=PRINT_EVERY,
        model_name=(
            "NTU RGB+D Model 21 "
            f"Ablation {experiment_id}"
        ),
    )

    if not training_is_complete(paths):
        raise FileNotFoundError(
            "Training finished, but the best "
            "checkpoint was not found: "
            f"{paths['model']}"
        )

    update_experiment_status(
        paths,
        training_completed=True,
        completed_epochs=len(
            history.get(
                "epoch",
                [],
            )
        ),
        total_parameters=(
            total_parameters
        ),
        trainable_parameters=(
            trainable_parameters
        ),
    )


# ============================================================
# 18. Video-generation stage
# ============================================================

def run_video_generation_stage(
    experiment_id: str,
    experiment_config: dict,
    paths: dict[str, Path],
    device: torch.device,
) -> None:
    if (
        SKIP_COMPLETED_STAGES
        and video_generation_is_complete(
            paths
        )
    ):
        print(
            "Prediction NPZ files already "
            "exist; skipping video generation."
        )

        update_experiment_status(
            paths,
            video_generation_completed=True,
        )

        return

    print()
    print("-" * 72)
    print(
        "Generating video and predictions "
        f"for {experiment_id}"
    )
    print("-" * 72)

    model_version = (
        experiment_model_version(
            experiment_id
        )
    )

    video_config = VideoGenerationConfig(
        model_version=model_version,
        test_csv=TEST_CSV,
        rgb_video_dir=RGB_VIDEO_DIR,
        skeleton_dir=SKELETON_DIR,
        output_dir=paths["video"],
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        num_joints=NUM_JOINTS,
        person_crop=PERSON_CROP,
        bbox_expansion=BBOX_EXPANSION,
        confidence_threshold=(
            CONFIDENCE_THRESHOLD
        ),
        frame_stride=(
            VIDEO_FRAME_STRIDE
        ),
        output_fps=OUTPUT_FPS,
        skip_existing_videos=(
            SKIP_EXISTING_VIDEOS
        ),
        save_prediction_npz=(
            SAVE_PREDICTION_NPZ
        ),
        max_test_videos=(
            MAX_TEST_VIDEOS
        ),
        readout_type=(
            experiment_config[
                "readout_type"
            ]
        ),
        prediction_label=(
            "Prediction - "
            f"{experiment_id}"
        ),
        num_steps=(
            experiment_config[
                "num_steps"
            ]
        ),
        beta=BETA,
        threshold=THRESHOLD,
    )

    run_video_generation(
        config=video_config,
        model_loader=make_model_loader(
            experiment_config,
            paths["model"],
        ),
        transform_builder=(
            build_video_transform
        ),
        device=device,
    )

    prediction_files = (
        find_prediction_npz_files(
            paths["video"]
        )
    )

    if not prediction_files:
        raise FileNotFoundError(
            "Video generation finished, but "
            "no prediction NPZ file was found "
            f"in {paths['video']}"
        )

    update_experiment_status(
        paths,
        video_generation_completed=True,
        prediction_npz_count=len(
            prediction_files
        ),
    )


# ============================================================
# 19. Evaluation stage
# ============================================================

def run_evaluation_stage(
    experiment_id: str,
    paths: dict[str, Path],
) -> None:
    if (
        SKIP_COMPLETED_STAGES
        and evaluation_is_complete(paths)
    ):
        print(
            "Metric files already exist; "
            "skipping evaluation."
        )

        update_experiment_status(
            paths,
            evaluation_completed=True,
        )

        return

    print()
    print("-" * 72)
    print(
        f"Evaluating {experiment_id}"
    )
    print("-" * 72)

    model_version = (
        experiment_model_version(
            experiment_id
        )
    )

    evaluation_config = EvaluationConfig(
        model_version=model_version,
        npz_dir=paths["video"],
        output_dir=paths["metrics"],
        filename_pattern=(
            "*_predictions_model_"
            f"{model_version}.npz"
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

    if not evaluation_is_complete(paths):
        raise FileNotFoundError(
            "Evaluation finished, but the "
            "summary or per-joint CSV was not "
            f"found in {paths['metrics']}"
        )

    update_experiment_status(
        paths,
        evaluation_completed=True,
    )


# ============================================================
# 20. summary_all.csv
# ============================================================

def read_first_csv_row(
    csv_path: Path,
) -> dict[str, str]:
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        row = next(reader, None)

    if row is None:
        raise RuntimeError(
            "CSV contains no data rows: "
            f"{csv_path}"
        )

    return {
        str(key): (
            "" if value is None else value
        )
        for key, value in row.items()
    }


def flatten_per_joint_csv(
    csv_path: Path,
) -> dict[str, str]:
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        raise RuntimeError(
            "Per-joint CSV contains no data: "
            f"{csv_path}"
        )

    identifier_candidates = (
        "joint_name",
        "joint",
        "name",
        "joint_id",
        "joint_index",
        "index",
    )

    flattened: dict[str, str] = {}

    for row_index, row in enumerate(rows):
        joint_value = None
        identifier_key = None

        for candidate in (
            identifier_candidates
        ):
            if (
                candidate in row
                and row[candidate]
                not in (None, "")
            ):
                identifier_key = candidate
                joint_value = row[candidate]
                break

        if joint_value is None:
            joint_value = (
                f"joint_{row_index}"
            )

        joint_name = clean_column_name(
            joint_value
        )

        if not joint_name:
            joint_name = (
                f"joint_{row_index}"
            )

        for key, value in row.items():
            if key == identifier_key:
                continue

            if value is None:
                value = ""

            metric_name = (
                clean_column_name(key)
            )

            if not metric_name:
                continue

            flattened[
                f"{joint_name}_{metric_name}"
            ] = value

    return flattened


def read_existing_summary_rows(
) -> list[dict[str, str]]:
    if not SUMMARY_ALL_PATH.exists():
        return []

    with SUMMARY_ALL_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def write_summary_rows(
    rows: list[dict[str, str]],
) -> None:
    if not rows:
        return

    preferred_columns = [
        "experiment_id",
        "experiment_name",
        "pipeline_number",
        "fusion_type",
        "readout_type",
        "decoder_type",
        "backbone_variant",
        "num_steps",
        "max_spikes",
    ]

    discovered_columns: set[str] = set()

    for row in rows:
        discovered_columns.update(
            row.keys()
        )

    remaining_columns = sorted(
        discovered_columns.difference(
            preferred_columns
        )
    )

    fieldnames = [
        column
        for column in preferred_columns
        if column in discovered_columns
    ]

    fieldnames.extend(
        remaining_columns
    )

    SUMMARY_ALL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        SUMMARY_ALL_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    temporary_path.replace(
        SUMMARY_ALL_PATH
    )


def update_summary_all(
    experiment_id: str,
    experiment_config: dict,
    paths: dict[str, Path],
) -> None:
    summary_csv = (
        find_metrics_summary_file(
            paths["metrics"]
        )
    )

    per_joint_csv = (
        find_metrics_per_joint_file(
            paths["metrics"]
        )
    )

    if summary_csv is None:
        raise FileNotFoundError(
            "Metrics summary CSV not found "
            f"for {experiment_id}."
        )

    if per_joint_csv is None:
        raise FileNotFoundError(
            "Per-joint metrics CSV not found "
            f"for {experiment_id}."
        )

    summary_data = (
        read_first_csv_row(
            summary_csv
        )
    )

    joint_data = (
        flatten_per_joint_csv(
            per_joint_csv
        )
    )

    combined_row = {
        "experiment_id": experiment_id,
        "experiment_name": (
            experiment_config["name"]
        ),
        "pipeline_number": (
            PIPELINE_NUMBER
        ),
        "fusion_type": (
            experiment_config[
                "fusion_type"
            ]
        ),
        "readout_type": (
            experiment_config[
                "readout_type"
            ]
        ),
        "decoder_type": (
            experiment_config[
                "decoder_type"
            ]
        ),
        "backbone_variant": (
            experiment_config[
                "backbone_variant"
            ]
        ),
        "num_steps": str(
            experiment_config[
                "num_steps"
            ]
        ),
        "max_spikes": str(
            experiment_config[
                "max_spikes"
            ]
        ),
        **summary_data,
        **joint_data,
    }

    existing_rows = (
        read_existing_summary_rows()
    )

    rows_by_experiment = {
        row.get(
            "experiment_id",
            "",
        ): row
        for row in existing_rows
        if row.get(
            "experiment_id",
            ""
        )
    }

    rows_by_experiment[
        experiment_id
    ] = combined_row

    ordered_rows = []

    for assigned_id in (
        EXPERIMENTS_TO_RUN
    ):
        if assigned_id in rows_by_experiment:
            ordered_rows.append(
                rows_by_experiment[
                    assigned_id
                ]
            )

    extra_ids = sorted(
        set(rows_by_experiment).difference(
            EXPERIMENTS_TO_RUN
        )
    )

    for extra_id in extra_ids:
        ordered_rows.append(
            rows_by_experiment[
                extra_id
            ]
        )

    write_summary_rows(
        ordered_rows
    )

    update_experiment_status(
        paths,
        summary_all_updated=True,
    )

    print(
        "Updated combined summary: "
        f"{SUMMARY_ALL_PATH}"
    )


# ============================================================
# 21. Single-experiment pipeline
# ============================================================

def run_single_experiment(
    experiment_id: str,
    experiment_config: dict,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> bool:
    paths = get_experiment_paths(
        experiment_id
    )

    ensure_experiment_directories(
        paths
    )

    save_experiment_configuration(
        experiment_id,
        experiment_config,
        paths,
    )

    update_experiment_status(
        paths,
        experiment_id=experiment_id,
        experiment_name=(
            experiment_config["name"]
        ),
        training_completed=(
            training_is_complete(paths)
        ),
        video_generation_completed=(
            video_generation_is_complete(
                paths
            )
        ),
        evaluation_completed=(
            evaluation_is_complete(
                paths
            )
        ),
        summary_all_updated=False,
        failed=False,
    )

    print()
    print("=" * 72)
    print(
        f"{experiment_id}: "
        f"{experiment_config['name']}"
    )
    print("=" * 72)
    print(
        "Configuration: "
        f"fusion={experiment_config['fusion_type']}, "
        f"readout={experiment_config['readout_type']}, "
        f"decoder={experiment_config['decoder_type']}, "
        f"backbone={experiment_config['backbone_variant']}, "
        f"T={experiment_config['num_steps']}, "
        f"D={experiment_config['max_spikes']}"
    )

    try:
        if RUN_TRAINING:
            run_training_stage(
                experiment_id=experiment_id,
                experiment_config=(
                    experiment_config
                ),
                paths=paths,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
            )

        if RUN_VIDEO_GENERATION:
            if not training_is_complete(
                paths
            ):
                raise RuntimeError(
                    "Video generation requires "
                    "a completed checkpoint."
                )

            run_video_generation_stage(
                experiment_id=experiment_id,
                experiment_config=(
                    experiment_config
                ),
                paths=paths,
                device=device,
            )

        if RUN_EVALUATION:
            if not video_generation_is_complete(
                paths
            ):
                raise RuntimeError(
                    "Evaluation requires at least "
                    "one prediction NPZ file."
                )

            run_evaluation_stage(
                experiment_id=experiment_id,
                paths=paths,
            )

        if evaluation_is_complete(paths):
            update_summary_all(
                experiment_id=experiment_id,
                experiment_config=(
                    experiment_config
                ),
                paths=paths,
            )

        update_experiment_status(
            paths,
            failed=False,
            pipeline_completed=(
                training_is_complete(paths)
                and video_generation_is_complete(
                    paths
                )
                and evaluation_is_complete(
                    paths
                )
            ),
        )

        if paths["error"].exists():
            paths["error"].unlink()

        print(
            f"Completed {experiment_id}."
        )

        return True

    except Exception:
        error_text = traceback.format_exc()

        paths["error"].write_text(
            error_text,
            encoding="utf-8",
        )

        update_experiment_status(
            paths,
            failed=True,
            pipeline_completed=False,
            error_file=str(
                paths["error"]
            ),
        )

        print()
        print(
            f"FAILED: {experiment_id}"
        )
        print(error_text)

        if not CONTINUE_AFTER_FAILURE:
            raise

        return False

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 22. Pipeline status
# ============================================================

def save_pipeline_status(
    completed: list[str],
    failed: list[str],
) -> None:
    pending = [
        experiment_id
        for experiment_id
        in EXPERIMENTS_TO_RUN
        if experiment_id
        not in completed
        and experiment_id
        not in failed
    ]

    write_json(
        PIPELINE_STATUS_PATH,
        {
            "pipeline_number": (
                PIPELINE_NUMBER
            ),
            "pipeline_name": (
                PIPELINE_NAME
            ),
            "assigned_experiments": (
                EXPERIMENTS_TO_RUN
            ),
            "completed_experiments": (
                completed
            ),
            "failed_experiments": (
                failed
            ),
            "pending_experiments": (
                pending
            ),
            "summary_all": str(
                SUMMARY_ALL_PATH
            ),
        },
    )


# ============================================================
# 23. Main
# ============================================================

def main() -> None:
    validate_experiment_assignment()
    validate_input_paths()

    PIPELINE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print(PIPELINE_NAME)
    print("=" * 72)
    print(
        f"Device: {device}"
    )
    print(
        f"Epochs per experiment: {EPOCHS}"
    )
    print(
        "Assigned experiments:"
    )

    for experiment_id in (
        EXPERIMENTS_TO_RUN
    ):
        print(
            f"  {experiment_id} - "
            f"{EXPERIMENTS[experiment_id]['name']}"
        )

    print(
        f"Output root: {PIPELINE_ROOT}"
    )
    print(
        f"Combined summary: {SUMMARY_ALL_PATH}"
    )
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

    completed: list[str] = []
    failed: list[str] = []

    save_pipeline_status(
        completed,
        failed,
    )

    for experiment_id in (
        EXPERIMENTS_TO_RUN
    ):
        success = run_single_experiment(
            experiment_id=experiment_id,
            experiment_config=(
                EXPERIMENTS[
                    experiment_id
                ]
            ),
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        if success:
            completed.append(
                experiment_id
            )
        else:
            failed.append(
                experiment_id
            )

        save_pipeline_status(
            completed,
            failed,
        )

    print()
    print("=" * 72)
    print(
        f"{PIPELINE_NAME} finished"
    )
    print("=" * 72)
    print(
        "Completed: "
        + (
            ", ".join(completed)
            if completed
            else "None"
        )
    )
    print(
        "Failed:    "
        + (
            ", ".join(failed)
            if failed
            else "None"
        )
    )
    print(
        f"Summary:   {SUMMARY_ALL_PATH}"
    )
    print("=" * 72)

    if failed:
        raise RuntimeError(
            "One or more experiments failed: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
