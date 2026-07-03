# Scripts/05_train_f2f_coordreg_baseline.py

from pathlib import Path
import random

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NPZ_PATH = PROJECT_ROOT / "Datasets" / "Penn_Action" / "penn_action_processed.npz"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "f2f_coordreg_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Head-only baseline
HEAD_INDEX = 0

# Stronger training
MAX_SAMPLES = 30000

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
VAL_RATIO = 0.2
NUM_WORKERS = 0


# ============================================================
# 2. Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
print(f"Using device: {DEVICE}")


# ============================================================
# 3. Dataset
# ============================================================

class PennActionHeadDataset(Dataset):
    def __init__(
        self,
        npz_path,
        project_root,
        image_size=224,
        head_index=0,
        max_samples=None
    ):
        self.npz_path = Path(npz_path)
        self.project_root = Path(project_root)
        self.image_size = image_size
        self.head_index = head_index

        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)

        print("\nLoaded npz:")
        print(self.npz_path)
        print("Available keys:", data.files)

        self.image_paths = data["image_paths"]
        all_keypoints = data["keypoints"].astype(np.float32)
        all_visibility = data["visibility"].astype(np.float32)

        if all_keypoints.ndim != 3 or all_keypoints.shape[2] != 2:
            raise ValueError(f"Expected keypoints shape [N, K, 2], got {all_keypoints.shape}")

        if head_index >= all_keypoints.shape[1]:
            raise ValueError(
                f"HEAD_INDEX={head_index} out of range. "
                f"Dataset has {all_keypoints.shape[1]} keypoints."
            )

        # Only use head keypoint
        self.keypoints = all_keypoints[:, head_index:head_index + 1, :]   # [N, 1, 2]
        self.visibility = all_visibility[:, head_index:head_index + 1]    # [N, 1]

        print(f"Original keypoints shape: {all_keypoints.shape}")
        print(f"Using HEAD_INDEX: {head_index}")
        print(f"Head keypoints shape: {self.keypoints.shape}")
        print(f"Total samples before limit: {len(self.image_paths)}")

        if max_samples is not None:
            max_samples = min(max_samples, len(self.image_paths))
            self.image_paths = self.image_paths[:max_samples]
            self.keypoints = self.keypoints[:max_samples]
            self.visibility = self.visibility[:max_samples]
            print(f"Debug mode: using first {len(self.image_paths)} samples")
        else:
            print(f"Full mode: using all {len(self.image_paths)} samples")

        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def _resolve_image_path(self, p):
        p = str(p).strip()
        path = Path(p)

        candidates = [
            path,
            self.project_root / path,
            self.project_root / "Datasets" / "Penn_Action" / path,
            self.project_root / "Datasets" / "Penn_Action" / "frames" / path,
        ]

        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError(
            f"Image not found.\n"
            f"Original path: {p}\n"
            f"Tried:\n" + "\n".join([f"  {c}" for c in candidates])
        )

    def __getitem__(self, idx):
        img_path = self._resolve_image_path(self.image_paths[idx])

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        keypoints = self.keypoints[idx].copy()      # [1, 2]
        visibility = self.visibility[idx].copy()    # [1]

        # Resize image and scale coordinate
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        image = self.transform(image)

        return {
            "image": image,
            "keypoints": torch.tensor(keypoints, dtype=torch.float32),
            "visibility": torch.tensor(visibility, dtype=torch.float32),
        }


# ============================================================
# 4. Model
# ============================================================

class ResNet18HeadRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 1, 2)
        return x


# ============================================================
# 5. Loss / MSE
# ============================================================

def masked_mse_loss(pred, target, visibility):
    """
    pred: [B, 1, 2]
    target: [B, 1, 2]
    visibility: [B, 1]
    """
    visibility = visibility.unsqueeze(-1)  # [B, 1, 1]

    error = (pred - target) ** 2
    error = error * visibility

    denom = visibility.sum() * 2.0
    denom = torch.clamp(denom, min=1.0)

    mse = error.sum() / denom
    return mse


# ============================================================
# 6. Train / Evaluate
# ============================================================

def train_one_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(DEVICE)
        keypoints = batch["keypoints"].to(DEVICE)
        visibility = batch["visibility"].to(DEVICE)

        pred = model(images)
        loss = masked_mse_loss(pred, keypoints, visibility)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss/MSE: {loss.item():.4f}"
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_mse = 0.0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        keypoints = batch["keypoints"].to(DEVICE)
        visibility = batch["visibility"].to(DEVICE)

        pred = model(images)
        mse = masked_mse_loss(pred, keypoints, visibility)

        total_mse += mse.item()

    return total_mse / len(loader)


# ============================================================
# 7. Main
# ============================================================

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = PennActionHeadDataset(
        npz_path=NPZ_PATH,
        project_root=PROJECT_ROOT,
        image_size=IMAGE_SIZE,
        head_index=HEAD_INDEX,
        max_samples=MAX_SAMPLES
    )

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    model = ResNet18HeadRegressor().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mse = float("inf")

    print("\nStart training Head-only Baseline")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"NPZ path: {NPZ_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Head index: {HEAD_INDEX}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch:03d}/{EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_mse = evaluate(model, val_loader)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] Summary | "
            f"Train Loss/MSE: {train_loss:.4f} | "
            f"Val MSE: {val_mse:.4f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse

            best_path = OUTPUT_DIR / "best_f2f_coordreg_resnet18.pth"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "head_index": HEAD_INDEX,
                    "num_keypoints": 1,
                    "best_val_mse": best_val_mse,
                    "epoch": epoch,
                    "model_name": "ResNet18HeadRegressor",
                    "method": "Head-only Frame-by-frame Coordinate Regression",
                },
                best_path
            )

            print(f"Saved best model to: {best_path}")

    last_path = OUTPUT_DIR / "last_f2f_coordreg_resnet18.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": IMAGE_SIZE,
            "head_index": HEAD_INDEX,
            "num_keypoints": 1,
            "best_val_mse": best_val_mse,
            "epoch": EPOCHS,
            "model_name": "ResNet18HeadRegressor",
            "method": "Head-only Frame-by-frame Coordinate Regression",
        },
        last_path
    )

    print("\nTraining finished.")
    print(f"Best Val MSE: {best_val_mse:.4f}")
    print(f"Last model saved to: {last_path}")


if __name__ == "__main__":
    main()