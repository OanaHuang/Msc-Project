# Scripts/Penn Action Model Training/04_ResNet_Heatmap_Baseline.py

from pathlib import Path
import random
import csv
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NPZ_PATH = PROJECT_ROOT / "Datasets" / "Penn_Action" / "penn_action_processed.npz"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "04_ResNet_Heatmap_Baseline"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "best_ResNet_Heatmap_Baseline.pth"
LAST_MODEL_PATH = OUTPUT_DIR / "last_ResNet_Heatmap_Baseline.pth"

SPLIT_JSON_PATH = OUTPUT_DIR / "video_level_split.json"
SPLIT_CSV_PATH = OUTPUT_DIR / "video_level_split_summary.csv"

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_KEYPOINTS = 13

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 4

SIGMA = 2.0
RANDOM_SEED = 42

# Video-level split ratio
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# None = use all videos
# If you want quick debug, set for example MAX_VIDEOS = 50
MAX_VIDEOS = None


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# 3. Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# 4. Helper Functions
# ============================================================

def safe_video_id_to_str(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8")

    x = str(x)

    if x.isdigit():
        return x.zfill(4)

    return x


def resolve_image_path(p):
    p_original = str(p).strip()
    p = p_original

    if "Penn_Action" in p:
        p = p.split("Penn_Action")[-1].lstrip("/")

    candidates = [
        Path(p),
        PROJECT_ROOT / p,
        PROJECT_ROOT / "Datasets" / "Penn_Action" / p,
        PROJECT_ROOT / "Datasets" / "Penn_Action" / "frames" / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Image not found.\n"
        f"Original path: {p_original}\n"
        f"Processed path: {p}"
    )


def make_gaussian_heatmap(x, y, heatmap_size, sigma):
    """
    x, y are coordinates on heatmap scale.
    Return: [heatmap_size, heatmap_size]
    """
    grid_y, grid_x = np.meshgrid(
        np.arange(heatmap_size),
        np.arange(heatmap_size),
        indexing="ij",
    )

    heatmap = np.exp(
        -((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2)
    )

    return heatmap.astype(np.float32)


def create_video_level_split(video_ids):
    """
    Split unique video IDs into train / val / test.
    No video can appear in more than one split.
    """
    unique_videos = sorted(list(set(video_ids)))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(unique_videos)

    if MAX_VIDEOS is not None:
        unique_videos = unique_videos[:MAX_VIDEOS]

    num_videos = len(unique_videos)

    train_end = int(num_videos * TRAIN_RATIO)
    val_end = train_end + int(num_videos * VAL_RATIO)

    train_videos = unique_videos[:train_end]
    val_videos = unique_videos[train_end:val_end]
    test_videos = unique_videos[val_end:]

    train_set = set(train_videos)
    val_set = set(val_videos)
    test_set = set(test_videos)

    train_indices = [
        i for i, vid in enumerate(video_ids)
        if vid in train_set
    ]

    val_indices = [
        i for i, vid in enumerate(video_ids)
        if vid in val_set
    ]

    test_indices = [
        i for i, vid in enumerate(video_ids)
        if vid in test_set
    ]

    return {
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def save_split_files(split):
    split_for_json = {
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "num_train_videos": len(split["train_videos"]),
        "num_val_videos": len(split["val_videos"]),
        "num_test_videos": len(split["test_videos"]),
        "num_train_frames": len(split["train_indices"]),
        "num_val_frames": len(split["val_indices"]),
        "num_test_frames": len(split["test_indices"]),
        "train_videos": split["train_videos"],
        "val_videos": split["val_videos"],
        "test_videos": split["test_videos"],
    }

    with open(SPLIT_JSON_PATH, "w") as f:
        json.dump(split_for_json, f, indent=4)

    rows = [
        {
            "split": "train",
            "num_videos": len(split["train_videos"]),
            "num_frames": len(split["train_indices"]),
            "video_ids": " ".join(split["train_videos"]),
        },
        {
            "split": "val",
            "num_videos": len(split["val_videos"]),
            "num_frames": len(split["val_indices"]),
            "video_ids": " ".join(split["val_videos"]),
        },
        {
            "split": "test",
            "num_videos": len(split["test_videos"]),
            "num_frames": len(split["test_indices"]),
            "video_ids": " ".join(split["test_videos"]),
        },
    ]

    with open(SPLIT_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "num_videos", "num_frames", "video_ids"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 5. Dataset
# ============================================================

class PennActionHeatmapDataset(Dataset):
    def __init__(
        self,
        npz_path,
        indices,
        image_size=224,
        heatmap_size=56,
        sigma=2.0,
    ):
        self.data = np.load(npz_path, allow_pickle=True)

        self.image_paths = self.data["image_paths"]
        self.keypoints = self.data["keypoints"].astype(np.float32)
        self.visibility = self.data["visibility"].astype(np.float32)

        self.indices = list(indices)

        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        img_path = resolve_image_path(self.image_paths[real_idx])

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img_tensor = self.transform(img)

        kpts = self.keypoints[real_idx].copy()
        vis = self.visibility[real_idx].copy()

        heatmaps = np.zeros(
            (NUM_KEYPOINTS, self.heatmap_size, self.heatmap_size),
            dtype=np.float32,
        )

        target_weights = np.zeros(
            (NUM_KEYPOINTS, 1, 1),
            dtype=np.float32,
        )

        for j in range(NUM_KEYPOINTS):
            if vis[j] <= 0:
                continue

            x, y = kpts[j]

            if x < 0 or y < 0 or x >= orig_w or y >= orig_h:
                continue

            heatmap_x = x / orig_w * self.heatmap_size
            heatmap_y = y / orig_h * self.heatmap_size

            heatmaps[j] = make_gaussian_heatmap(
                heatmap_x,
                heatmap_y,
                self.heatmap_size,
                self.sigma,
            )

            target_weights[j, 0, 0] = 1.0

        return {
            "image": img_tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "target_weights": torch.from_numpy(target_weights),
        }


# ============================================================
# 6. Model
# ============================================================

class ResNet18HeatmapBaseline(nn.Module):
    def __init__(self, num_keypoints=13):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove avgpool and fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # input 224 -> [B, 512, 7, 7]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )
        # 7 -> 14 -> 28 -> 56

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x


# ============================================================
# 7. Loss
# ============================================================

def heatmap_mse_loss(pred, target, target_weights):
    """
    pred: [B, K, H, W]
    target: [B, K, H, W]
    target_weights: [B, K, 1, 1]
    """
    loss = (pred - target) ** 2
    loss = loss * target_weights

    visible_count = target_weights.sum()

    if visible_count <= 0:
        return loss.mean()

    return loss.sum() / visible_count


# ============================================================
# 8. Train / Validate
# ============================================================

def train_one_epoch(model, dataloader, optimizer):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(DEVICE)
        heatmaps = batch["heatmaps"].to(DEVICE)
        target_weights = batch["target_weights"].to(DEVICE)

        optimizer.zero_grad()

        pred_heatmaps = model(images)

        loss = heatmap_mse_loss(
            pred_heatmaps,
            heatmaps,
            target_weights,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / max(total_batches, 1)


def validate_one_epoch(model, dataloader):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            heatmaps = batch["heatmaps"].to(DEVICE)
            target_weights = batch["target_weights"].to(DEVICE)

            pred_heatmaps = model(images)

            loss = heatmap_mse_loss(
                pred_heatmaps,
                heatmaps,
                target_weights,
            )

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


# ============================================================
# 9. Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"NPZ path: {NPZ_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ not found: {NPZ_PATH}")

    data = np.load(NPZ_PATH, allow_pickle=True)
    video_ids = np.array([safe_video_id_to_str(v) for v in data["video_ids"]])

    split = create_video_level_split(video_ids)
    save_split_files(split)

    train_videos = set(split["train_videos"])
    val_videos = set(split["val_videos"])
    test_videos = set(split["test_videos"])

    assert len(train_videos & val_videos) == 0
    assert len(train_videos & test_videos) == 0
    assert len(val_videos & test_videos) == 0

    print("\n========== Video-level Split ==========")
    print(f"Train videos: {len(split['train_videos'])}")
    print(f"Val videos: {len(split['val_videos'])}")
    print(f"Test videos: {len(split['test_videos'])}")

    print(f"Train frames: {len(split['train_indices'])}")
    print(f"Val frames: {len(split['val_indices'])}")
    print(f"Test frames: {len(split['test_indices'])}")

    print(f"\nSplit saved to:")
    print(SPLIT_JSON_PATH)
    print(SPLIT_CSV_PATH)

    train_dataset = PennActionHeatmapDataset(
        NPZ_PATH,
        indices=split["train_indices"],
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=SIGMA,
    )

    val_dataset = PennActionHeatmapDataset(
        NPZ_PATH,
        indices=split["val_indices"],
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=SIGMA,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = ResNet18HeatmapBaseline(num_keypoints=NUM_KEYPOINTS)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    best_val_loss = float("inf")

    print("\n========== Training ==========")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "method": "ResNet18_Heatmap_Baseline_VideoSplit",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "image_size": IMAGE_SIZE,
                    "heatmap_size": HEATMAP_SIZE,
                    "num_keypoints": NUM_KEYPOINTS,
                    "sigma": SIGMA,
                    "split_type": "video_level",
                    "train_videos": split["train_videos"],
                    "val_videos": split["val_videos"],
                    "test_videos": split["test_videos"],
                    "train_ratio": TRAIN_RATIO,
                    "val_ratio": VAL_RATIO,
                    "test_ratio": TEST_RATIO,
                    "random_seed": RANDOM_SEED,
                },
                BEST_MODEL_PATH,
            )

            print(f"  New best model saved to: {BEST_MODEL_PATH}")

    torch.save(
        {
            "method": "ResNet18_Heatmap_Baseline_VideoSplit",
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "image_size": IMAGE_SIZE,
            "heatmap_size": HEATMAP_SIZE,
            "num_keypoints": NUM_KEYPOINTS,
            "sigma": SIGMA,
            "split_type": "video_level",
            "train_videos": split["train_videos"],
            "val_videos": split["val_videos"],
            "test_videos": split["test_videos"],
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "random_seed": RANDOM_SEED,
        },
        LAST_MODEL_PATH,
    )

    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Last model saved to: {LAST_MODEL_PATH}")


if __name__ == "__main__":
    main()