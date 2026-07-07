    # Scripts/Penn Action model training/01_Resnet_with_ImageNet.py

from pathlib import Path
import random
import time

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

# Colab Drive path
PROJECT_ROOT = Path("/content/drive/MyDrive/MSc Project")

# Local fallback
# This file is inside:
# Scripts/Penn Action model training/
# parents[2] goes back to project root
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

NPZ_PATH = PROJECT_ROOT / "Datasets" / "Penn_Action" / "penn_action_processed.npz"

# Output path:
# server_output/PennAction modeltraining/01_Resnet_with_ImageNet/
OUTPUT_DIR = (
    PROJECT_ROOT
    / "server_output"
    / "PennAction modeltraining"
    / "01_Resnet_with_ImageNet"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
VAL_RATIO = 0.2
NUM_WORKERS = 2

# Use 30000 first. Later you can change it to None.
MAX_SAMPLES = 30000
# MAX_SAMPLES = None

USE_PRETRAINED = True


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
print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. Dataset
# ============================================================

class PennActionAllKeypointsDataset(Dataset):
    def __init__(
        self,
        npz_path,
        project_root,
        image_size=224,
        max_samples=None
    ):
        self.npz_path = Path(npz_path)
        self.project_root = Path(project_root)
        self.image_size = image_size

        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)

        print("\nLoaded npz:")
        print(self.npz_path)
        print("Available keys:", data.files)

        self.image_paths = data["image_paths"]
        self.keypoints = data["keypoints"].astype(np.float32)       # [N, K, 2]
        self.visibility = data["visibility"].astype(np.float32)     # [N, K]

        if self.keypoints.ndim != 3 or self.keypoints.shape[2] != 2:
            raise ValueError(
                f"Expected keypoints shape [N, K, 2], got {self.keypoints.shape}"
            )

        if self.visibility.ndim != 2:
            raise ValueError(
                f"Expected visibility shape [N, K], got {self.visibility.shape}"
            )

        if self.keypoints.shape[:2] != self.visibility.shape:
            raise ValueError(
                f"Shape mismatch: keypoints {self.keypoints.shape}, "
                f"visibility {self.visibility.shape}"
            )

        self.num_keypoints = self.keypoints.shape[1]

        print(f"Keypoints shape: {self.keypoints.shape}")
        print(f"Visibility shape: {self.visibility.shape}")
        print(f"Number of keypoints: {self.num_keypoints}")
        print(f"Total samples before limit: {len(self.image_paths)}")

        if max_samples is not None:
            max_samples = min(max_samples, len(self.image_paths))

            # Random sampling is better than taking the first N samples
            rng = np.random.RandomState(42)
            selected_indices = rng.permutation(len(self.image_paths))[:max_samples]

            self.image_paths = self.image_paths[selected_indices]
            self.keypoints = self.keypoints[selected_indices]
            self.visibility = self.visibility[selected_indices]

            print(f"Using random {len(self.image_paths)} samples")
        else:
            print(f"Using all {len(self.image_paths)} samples")

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
        p_original = str(p).strip()
        p = p_original

        # If npz stores Mac absolute path, keep only the part after Penn_Action
        if "Penn_Action" in p:
            p = p.split("Penn_Action")[-1].lstrip("/")

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
            f"Original path: {p_original}\n"
            f"Processed path: {p}\n"
            f"Tried:\n" + "\n".join([f"  {c}" for c in candidates])
        )

    def __getitem__(self, idx):
        img_path = self._resolve_image_path(self.image_paths[idx])

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        keypoints = self.keypoints[idx].copy()        # [K, 2]
        visibility = self.visibility[idx].copy()      # [K]

        # Resize image and scale coordinates to IMAGE_SIZE x IMAGE_SIZE
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

class ResNetWithImageNetAllKeypointRegressor(nn.Module):
    def __init__(self, num_keypoints, use_pretrained=True):
        super().__init__()

        self.num_keypoints = num_keypoints

        if use_pretrained:
            print("Using ImageNet-pretrained ResNet18 backbone")
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
        else:
            print("Using ResNet18 backbone from scratch")
            self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features

        # Replace ImageNet classification head:
        # Original: 512 -> 1000 classes
        # New: 512 -> 512 -> K*2 coordinates
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_keypoints * 2)
        )

    def forward(self, x):
        x = self.backbone(x)                         # [B, K * 2]
        x = x.view(-1, self.num_keypoints, 2)        # [B, K, 2]
        return x


# ============================================================
# 5. Loss
# ============================================================

def masked_mse_loss(pred, target, visibility):
    """
    pred: [B, K, 2]
    target: [B, K, 2]
    visibility: [B, K]
    """
    visibility = visibility.unsqueeze(-1)  # [B, K, 1]

    error = (pred - target) ** 2
    error = error * visibility

    denom = visibility.sum() * 2.0
    denom = torch.clamp(denom, min=1.0)

    return error.sum() / denom


# ============================================================
# 6. Train / Evaluate
# ============================================================

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()

    total_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(DEVICE, non_blocking=True)
        keypoints = batch["keypoints"].to(DEVICE, non_blocking=True)
        visibility = batch["visibility"].to(DEVICE, non_blocking=True)

        pred = model(images)
        loss = masked_mse_loss(pred, keypoints, visibility)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch} | "
                f"Batch [{batch_idx + 1}/{len(loader)}] | "
                f"Loss/MSE: {loss.item():.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_mse = 0.0

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        keypoints = batch["keypoints"].to(DEVICE, non_blocking=True)
        visibility = batch["visibility"].to(DEVICE, non_blocking=True)

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

    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)

    print("\nProject root:", PROJECT_ROOT)
    print("NPZ path:", NPZ_PATH)
    print("Output dir:", OUTPUT_DIR)

    dataset = PennActionAllKeypointsDataset(
        npz_path=NPZ_PATH,
        project_root=PROJECT_ROOT,
        image_size=IMAGE_SIZE,
        max_samples=MAX_SAMPLES
    )

    num_keypoints = dataset.num_keypoints

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    pin_memory = True if DEVICE.type == "cuda" else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    model = ResNetWithImageNetAllKeypointRegressor(
        num_keypoints=num_keypoints,
        use_pretrained=USE_PRETRAINED
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mse = float("inf")

    print("\nStart training: ResNet_with_ImageNet")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Use ImageNet pretrained: {USE_PRETRAINED}")
    print(f"Number of keypoints: {num_keypoints}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch:03d}/{EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        val_mse = evaluate(model, val_loader)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] Summary | "
            f"Train Loss/MSE: {train_loss:.4f} | "
            f"Val MSE: {val_mse:.4f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse

            best_path = OUTPUT_DIR / "best_Resnet_with_ImageNet.pth"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "num_keypoints": num_keypoints,
                    "best_val_mse": best_val_mse,
                    "epoch": epoch,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "use_pretrained": USE_PRETRAINED,
                    "model_name": "ResNetWithImageNetAllKeypointRegressor",
                    "method": "ResNet18 with ImageNet pretrained weights for all-keypoint coordinate regression",
                    "output_dir": str(OUTPUT_DIR),
                },
                best_path
            )

            print(f"Saved best model to: {best_path}")

    last_path = OUTPUT_DIR / "last_Resnet_with_ImageNet.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": IMAGE_SIZE,
            "num_keypoints": num_keypoints,
            "best_val_mse": best_val_mse,
            "epoch": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "use_pretrained": USE_PRETRAINED,
            "model_name": "ResNetWithImageNetAllKeypointRegressor",
            "method": "ResNet18 with ImageNet pretrained weights for all-keypoint coordinate regression",
            "output_dir": str(OUTPUT_DIR),
        },
        last_path
    )

    print("\nTraining finished.")
    print(f"Best Val MSE: {best_val_mse:.4f}")
    print(f"Best model saved in: {OUTPUT_DIR / 'best_Resnet_with_ImageNet.pth'}")
    print(f"Last model saved in: {last_path}")


if __name__ == "__main__":
    main()