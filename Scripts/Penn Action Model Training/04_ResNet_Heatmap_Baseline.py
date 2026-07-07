# Scripts/Penn Action Model Training/04_ResNet_Heatmap_Baseline.py

from pathlib import Path
import random
import math

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
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

IMAGE_SIZE = 224
HEATMAP_SIZE = 56

NUM_KEYPOINTS = 13

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
VAL_RATIO = 0.2
NUM_WORKERS = 0

# 先用 30000，和你之前 ResNet baseline 对齐
MAX_SAMPLES = 30000

SIGMA = 2.0

RANDOM_SEED = 42


# ============================================================
# 2. Device
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
# 3. Helper Functions
# ============================================================

def safe_video_id_to_str(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8")

    x = str(x)

    if x.isdigit():
        return x.zfill(4)

    return x


def resolve_image_path(project_root, p):
    p_original = str(p).strip()
    p = p_original

    if "Penn_Action" in p:
        p = p.split("Penn_Action")[-1].lstrip("/")

    candidates = [
        Path(p),
        project_root / p,
        project_root / "Datasets" / "Penn_Action" / p,
        project_root / "Datasets" / "Penn_Action" / "frames" / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Image not found: {p_original}")


def generate_gaussian_heatmap(center_x, center_y, heatmap_size, sigma):
    """
    Generate one 2D Gaussian heatmap.
    center_x, center_y are in heatmap coordinate system.
    """
    x = np.arange(0, heatmap_size, 1, np.float32)
    y = np.arange(0, heatmap_size, 1, np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    heatmap = np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2)
    )

    return heatmap.astype(np.float32)


def heatmaps_to_coordinates(heatmaps, image_size=224, heatmap_size=56):
    """
    heatmaps: [K, H, W]
    return coordinates in resized image space: [K, 2]
    """
    num_keypoints = heatmaps.shape[0]
    coords = np.zeros((num_keypoints, 2), dtype=np.float32)

    for k in range(num_keypoints):
        hm = heatmaps[k]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)

        coords[k, 0] = x * image_size / heatmap_size
        coords[k, 1] = y * image_size / heatmap_size

    return coords


# ============================================================
# 4. Dataset
# ============================================================

class PennActionHeatmapDataset(Dataset):
    def __init__(
        self,
        npz_path,
        project_root,
        image_size=224,
        heatmap_size=56,
        sigma=2.0,
        max_samples=None,
    ):
        super().__init__()

        data = np.load(npz_path, allow_pickle=True)

        self.image_paths = data["image_paths"]
        self.keypoints = data["keypoints"].astype(np.float32)
        self.visibility = data["visibility"].astype(np.float32)

        self.project_root = project_root
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        self.indices = list(range(len(self.image_paths)))

        if max_samples is not None:
            self.indices = self.indices[:max_samples]

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

    def __getitem__(self, i):
        idx = self.indices[i]

        img_path = resolve_image_path(self.project_root, self.image_paths[idx])
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size

        kpts = self.keypoints[idx].copy()
        vis = self.visibility[idx].copy()

        image_tensor = self.transform(img)

        target_heatmaps = np.zeros(
            (NUM_KEYPOINTS, self.heatmap_size, self.heatmap_size),
            dtype=np.float32
        )

        target_weights = np.zeros(
            (NUM_KEYPOINTS, 1, 1),
            dtype=np.float32
        )

        for j in range(NUM_KEYPOINTS):
            if vis[j] <= 0:
                continue

            x, y = kpts[j]

            if x < 0 or y < 0 or x >= orig_w or y >= orig_h:
                continue

            heatmap_x = x / orig_w * self.heatmap_size
            heatmap_y = y / orig_h * self.heatmap_size

            target_heatmaps[j] = generate_gaussian_heatmap(
                center_x=heatmap_x,
                center_y=heatmap_y,
                heatmap_size=self.heatmap_size,
                sigma=self.sigma,
            )

            target_weights[j, 0, 0] = 1.0

        return {
            "image": image_tensor,
            "heatmaps": torch.from_numpy(target_heatmaps),
            "weights": torch.from_numpy(target_weights),
        }


# ============================================================
# 5. Model
# ============================================================

class ResNet18HeatmapBaseline(nn.Module):
    def __init__(self, num_keypoints=13):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove avgpool and fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Output: [B, 512, 7, 7] for 224x224 input

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

            nn.Conv2d(64, num_keypoints, kernel_size=1)
        )
        # 7 -> 14 -> 28 -> 56

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x


# ============================================================
# 6. Loss
# ============================================================

def masked_heatmap_mse_loss(pred_heatmaps, target_heatmaps, target_weights):
    """
    pred_heatmaps: [B, K, H, W]
    target_heatmaps: [B, K, H, W]
    target_weights: [B, K, 1, 1]
    """
    loss = (pred_heatmaps - target_heatmaps) ** 2
    loss = loss * target_weights

    visible_count = target_weights.sum()

    if visible_count.item() == 0:
        return loss.mean()

    return loss.sum() / visible_count


# ============================================================
# 7. Train / Validate
# ============================================================

def train_one_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        heatmaps = batch["heatmaps"].to(DEVICE)
        weights = batch["weights"].to(DEVICE)

        preds = model(images)

        loss = masked_heatmap_mse_loss(preds, heatmaps, weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def validate(model, loader):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        heatmaps = batch["heatmaps"].to(DEVICE)
        weights = batch["weights"].to(DEVICE)

        preds = model(images)

        loss = masked_heatmap_mse_loss(preds, heatmaps, weights)

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


# ============================================================
# 8. Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"NPZ path: {NPZ_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ not found: {NPZ_PATH}")

    dataset = PennActionHeatmapDataset(
        npz_path=NPZ_PATH,
        project_root=PROJECT_ROOT,
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        sigma=SIGMA,
        max_samples=MAX_SAMPLES,
    )

    print(f"Dataset size: {len(dataset)}")

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Train size: {train_size}")
    print(f"Val size: {val_size}")

    model = ResNet18HeatmapBaseline(num_keypoints=NUM_KEYPOINTS)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "image_size": IMAGE_SIZE,
                "heatmap_size": HEATMAP_SIZE,
                "num_keypoints": NUM_KEYPOINTS,
                "sigma": SIGMA,
                "method": "ResNet18 Heatmap Baseline",
            }, BEST_MODEL_PATH)

            print(f"Best model saved: {BEST_MODEL_PATH}")

    torch.save({
        "epoch": EPOCHS - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "image_size": IMAGE_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "num_keypoints": NUM_KEYPOINTS,
        "sigma": SIGMA,
        "method": "ResNet18 Heatmap Baseline",
    }, LAST_MODEL_PATH)

    print("\nTraining finished.")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Best model saved in:\n{BEST_MODEL_PATH}")
    print(f"Last model saved in:\n{LAST_MODEL_PATH}")


if __name__ == "__main__":
    main()