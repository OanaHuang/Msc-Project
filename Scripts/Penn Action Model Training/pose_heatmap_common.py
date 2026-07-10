from pathlib import Path
import csv
import json
import random
import time

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NPZ_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Penn_Action"
    / "penn_action_processed.npz"
)

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_KEYPOINTS = 13

BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-4
NUM_WORKERS = 4

SIGMA = 2.0
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAX_VIDEOS = None
PRINT_EVERY_BATCHES = 50


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


def safe_video_id_to_str(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    value = str(value)

    if value.isdigit():
        return value.zfill(4)

    return value


def resolve_image_path(path_value):
    original_path = str(path_value).strip()
    processed_path = original_path

    if "Penn_Action" in processed_path:
        processed_path = (
            processed_path
            .split("Penn_Action")[-1]
            .lstrip("/")
        )

    candidates = [
        Path(processed_path),
        PROJECT_ROOT / processed_path,
        (
            PROJECT_ROOT
            / "Datasets"
            / "Penn_Action"
            / processed_path
        ),
        (
            PROJECT_ROOT
            / "Datasets"
            / "Penn_Action"
            / "frames"
            / processed_path
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Image not found.\n"
        f"Original path: {original_path}\n"
        f"Processed path: {processed_path}"
    )


def make_gaussian_heatmap(
    x,
    y,
    heatmap_size,
    sigma,
):
    grid_y, grid_x = np.meshgrid(
        np.arange(heatmap_size),
        np.arange(heatmap_size),
        indexing="ij",
    )

    heatmap = np.exp(
        -(
            (grid_x - x) ** 2
            + (grid_y - y) ** 2
        )
        / (2 * sigma ** 2)
    )

    return heatmap.astype(np.float32)


def create_video_level_split(video_ids):
    unique_videos = sorted(set(video_ids))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(unique_videos)

    if MAX_VIDEOS is not None:
        unique_videos = unique_videos[:MAX_VIDEOS]

    num_videos = len(unique_videos)

    train_end = int(
        num_videos * TRAIN_RATIO
    )

    val_end = train_end + int(
        num_videos * VAL_RATIO
    )

    train_videos = unique_videos[:train_end]
    val_videos = unique_videos[train_end:val_end]
    test_videos = unique_videos[val_end:]

    train_set = set(train_videos)
    val_set = set(val_videos)
    test_set = set(test_videos)

    train_indices = [
        index
        for index, video_id in enumerate(video_ids)
        if video_id in train_set
    ]

    val_indices = [
        index
        for index, video_id in enumerate(video_ids)
        if video_id in val_set
    ]

    test_indices = [
        index
        for index, video_id in enumerate(video_ids)
        if video_id in test_set
    ]

    return {
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def save_split_files(
    split,
    split_json_path,
    split_csv_path,
):
    split_for_json = {
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,

        "num_train_videos":
            len(split["train_videos"]),

        "num_val_videos":
            len(split["val_videos"]),

        "num_test_videos":
            len(split["test_videos"]),

        "num_train_frames":
            len(split["train_indices"]),

        "num_val_frames":
            len(split["val_indices"]),

        "num_test_frames":
            len(split["test_indices"]),

        "train_videos":
            split["train_videos"],

        "val_videos":
            split["val_videos"],

        "test_videos":
            split["test_videos"],
    }

    with split_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_for_json,
            file,
            indent=4,
        )

    rows = [
        {
            "split": "train",
            "num_videos":
                len(split["train_videos"]),
            "num_frames":
                len(split["train_indices"]),
            "video_ids":
                " ".join(split["train_videos"]),
        },
        {
            "split": "val",
            "num_videos":
                len(split["val_videos"]),
            "num_frames":
                len(split["val_indices"]),
            "video_ids":
                " ".join(split["val_videos"]),
        },
        {
            "split": "test",
            "num_videos":
                len(split["test_videos"]),
            "num_frames":
                len(split["test_indices"]),
            "video_ids":
                " ".join(split["test_videos"]),
        },
    ]

    with split_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "num_videos",
                "num_frames",
                "video_ids",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


class PennActionHeatmapDataset(Dataset):

    def __init__(
        self,
        npz_path,
        indices,
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=SIGMA,
    ):
        # 将数组真正复制到内存，关闭 NPZ 文件。
        # 这样多 worker 不会共享同一个压缩文件句柄。
        with np.load(
            npz_path,
            allow_pickle=True,
        ) as data:
            self.image_paths = (
                data["image_paths"].copy()
            )

            self.keypoints = (
                data["keypoints"]
                .astype(np.float32)
                .copy()
            )

            self.visibility = (
                data["visibility"]
                .astype(np.float32)
                .copy()
            )

        self.indices = list(indices)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        self.transform = T.Compose([
            T.Resize(
                (image_size, image_size)
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]

        image_path = resolve_image_path(
            self.image_paths[real_index]
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = (
                image.size
            )

            image_tensor = self.transform(
                image
            )

        keypoints = (
            self.keypoints[real_index]
            .copy()
        )

        visibility = (
            self.visibility[real_index]
            .copy()
        )

        heatmaps = np.zeros(
            (
                NUM_KEYPOINTS,
                self.heatmap_size,
                self.heatmap_size,
            ),
            dtype=np.float32,
        )

        target_weights = np.zeros(
            (
                NUM_KEYPOINTS,
                1,
                1,
            ),
            dtype=np.float32,
        )

        for joint_index in range(
            NUM_KEYPOINTS
        ):
            if visibility[joint_index] <= 0:
                continue

            x, y = keypoints[joint_index]

            if (
                x < 0
                or y < 0
                or x >= original_width
                or y >= original_height
            ):
                continue

            heatmap_x = (
                x
                / original_width
                * self.heatmap_size
            )

            heatmap_y = (
                y
                / original_height
                * self.heatmap_size
            )

            heatmaps[joint_index] = (
                make_gaussian_heatmap(
                    heatmap_x,
                    heatmap_y,
                    self.heatmap_size,
                    self.sigma,
                )
            )

            target_weights[
                joint_index,
                0,
                0,
            ] = 1.0

        return {
            "image":
                image_tensor,

            "heatmaps":
                torch.from_numpy(
                    heatmaps
                ),

            "target_weights":
                torch.from_numpy(
                    target_weights
                ),
        }


def heatmap_mse_loss(
    prediction,
    target,
    target_weights,
):
    loss = (
        prediction - target
    ) ** 2

    loss = loss * target_weights

    visible_count = (
        target_weights.sum()
    )

    if visible_count <= 0:
        return loss.mean()

    # 同时除以 heatmap 像素数，避免 loss 数值过大
    return (
        loss.sum()
        / (
            visible_count
            * prediction.shape[-2]
            * prediction.shape[-1]
        )
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    epoch,
):
    model.train()

    total_loss = 0.0
    total_batches = 0

    epoch_start = time.perf_counter()

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):
        images = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        heatmaps = batch["heatmaps"].to(
            DEVICE,
            non_blocking=True,
        )

        target_weights = batch[
            "target_weights"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predicted_heatmaps = model(
            images
        )

        loss = heatmap_mse_loss(
            predicted_heatmaps,
            heatmaps,
            target_weights,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss: "
                f"{loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (
            batch_index
            % PRINT_EVERY_BATCHES
            == 0
            or batch_index
            == len(dataloader)
        ):
            elapsed = (
                time.perf_counter()
                - epoch_start
            )

            average_batch_time = (
                elapsed / batch_index
            )

            remaining_batches = (
                len(dataloader)
                - batch_index
            )

            eta_minutes = (
                average_batch_time
                * remaining_batches
                / 60
            )

            print(
                f"  Epoch {epoch:03d} | "
                f"Batch "
                f"{batch_index:04d}/"
                f"{len(dataloader):04d} | "
                f"Loss={loss.item():.6f} | "
                f"ETA={eta_minutes:.1f} min",
                flush=True,
            )

    return (
        total_loss
        / max(total_batches, 1)
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        images = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        heatmaps = batch["heatmaps"].to(
            DEVICE,
            non_blocking=True,
        )

        target_weights = batch[
            "target_weights"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        predicted_heatmaps = model(
            images
        )

        loss = heatmap_mse_loss(
            predicted_heatmaps,
            heatmaps,
            target_weights,
        )

        total_loss += loss.item()
        total_batches += 1

    return (
        total_loss
        / max(total_batches, 1)
    )


def run_training(
    model,
    method_name,
    output_dir,
    best_model_name,
    last_model_name,
    extra_config=None,
):
    set_seed()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        output_dir
        / best_model_name
    )

    last_model_path = (
        output_dir
        / last_model_name
    )

    split_json_path = (
        output_dir
        / "video_level_split.json"
    )

    split_csv_path = (
        output_dir
        / "video_level_split_summary.csv"
    )

    print("=" * 72)
    print(method_name)
    print("=" * 72)
    print(f"Device      : {DEVICE}")
    print(f"Dataset     : {NPZ_PATH}")
    print(f"Output      : {output_dir}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Workers     : {NUM_WORKERS}")
    print("=" * 72)

    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"NPZ not found: {NPZ_PATH}"
        )

    with np.load(
        NPZ_PATH,
        allow_pickle=True,
    ) as data:
        video_ids = np.asarray([
            safe_video_id_to_str(value)
            for value in data["video_ids"]
        ])

    split = create_video_level_split(
        video_ids
    )

    save_split_files(
        split,
        split_json_path,
        split_csv_path,
    )

    print("\nVideo-level split:")
    print(
        f"Train videos : "
        f"{len(split['train_videos'])}"
    )
    print(
        f"Val videos   : "
        f"{len(split['val_videos'])}"
    )
    print(
        f"Test videos  : "
        f"{len(split['test_videos'])}"
    )
    print(
        f"Train frames : "
        f"{len(split['train_indices'])}"
    )
    print(
        f"Val frames   : "
        f"{len(split['val_indices'])}"
    )
    print(
        f"Test frames  : "
        f"{len(split['test_indices'])}"
    )

    train_dataset = (
        PennActionHeatmapDataset(
            NPZ_PATH,
            split["train_indices"],
        )
    )

    val_dataset = (
        PennActionHeatmapDataset(
            NPZ_PATH,
            split["val_indices"],
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    model = model.to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )

    best_val_loss = float("inf")

    print("\nTraining starts...")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        print(
            f"\nEpoch [{epoch}/{EPOCHS}]"
        )

        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
        )

        val_loss = validate_one_epoch(
            model,
            val_loader,
        )

        scheduler.step(val_loss)

        elapsed_minutes = (
            time.perf_counter()
            - epoch_start
        ) / 60

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train Loss={train_loss:.6f} | "
            f"Val Loss={val_loss:.6f} | "
            f"Time={elapsed_minutes:.2f} min"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "method": method_name,
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_val_loss":
                    best_val_loss,
                "image_size":
                    IMAGE_SIZE,
                "heatmap_size":
                    HEATMAP_SIZE,
                "num_keypoints":
                    NUM_KEYPOINTS,
                "sigma":
                    SIGMA,
                "split_type":
                    "video_level",
                "train_videos":
                    split["train_videos"],
                "val_videos":
                    split["val_videos"],
                "test_videos":
                    split["test_videos"],
            }

            if extra_config is not None:
                checkpoint.update(
                    extra_config
                )

            torch.save(
                checkpoint,
                best_model_path,
            )

            print(
                f"  New best model saved: "
                f"{best_model_path}"
            )

    final_checkpoint = {
        "method": method_name,
        "epoch": EPOCHS,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "best_val_loss":
            best_val_loss,
        "image_size":
            IMAGE_SIZE,
        "heatmap_size":
            HEATMAP_SIZE,
        "num_keypoints":
            NUM_KEYPOINTS,
        "sigma":
            SIGMA,
        "split_type":
            "video_level",
        "train_videos":
            split["train_videos"],
        "val_videos":
            split["val_videos"],
        "test_videos":
            split["test_videos"],
    }

    if extra_config is not None:
        final_checkpoint.update(
            extra_config
        )

    torch.save(
        final_checkpoint,
        last_model_path,
    )

    print("\nTraining finished.")
    print(
        f"Best val loss: "
        f"{best_val_loss:.6f}"
    )
    print(
        f"Best model: {best_model_path}"
    )
    print(
        f"Last model: {last_model_path}"
    )