from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


# ============================================================
# 1. Device 选择：CUDA / Apple Silicon MPS / CPU
# ============================================================

def get_device():
    """
    自动选择训练设备：
    1. 如果有 NVIDIA GPU，用 CUDA
    2. 如果是 Apple Silicon，比如 M1/M2/M3，用 MPS
    3. 否则用 CPU
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU:", torch.cuda.get_device_name(0))

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU: MPS")

    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


# ============================================================
# 2. MPII Dataset
# ============================================================

class MPIIDataset(Dataset):
    def __init__(self, mpii_root, npz_path, image_size=(256, 256)):
        self.mpii_root = Path(mpii_root)
        self.npz_path = Path(npz_path)
        self.image_size = image_size

        if not self.mpii_root.exists():
            raise FileNotFoundError(f"MPII root not found: {self.mpii_root}")

        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)

        self.image_paths = data["image_paths"]
        self.keypoints = data["keypoints"].astype(np.float32)
        self.visibility = data["visibility"].astype(np.float32)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        print("\nMPIIDataset loaded.")
        print("Samples:", len(self.image_paths))
        print("Keypoints:", self.keypoints.shape)
        print("Visibility:", self.visibility.shape)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.mpii_root / self.image_paths[idx]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size

        keypoints = self.keypoints[idx].copy()
        visibility = self.visibility[idx].copy()

        new_width, new_height = self.image_size

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        # resize 图片后，同步缩放关键点坐标
        keypoints[:, 0] = keypoints[:, 0] * scale_x
        keypoints[:, 1] = keypoints[:, 1] * scale_y

        # 把坐标归一化到 0~1，训练更稳定
        keypoints[:, 0] = keypoints[:, 0] / new_width
        keypoints[:, 1] = keypoints[:, 1] / new_height

        image = self.transform(image)

        keypoints = torch.from_numpy(keypoints).float()
        visibility = torch.from_numpy(visibility).float()

        return image, keypoints, visibility


# ============================================================
# 3. 最简单 CNN baseline
# ============================================================

class SimplePoseCNN(nn.Module):
    def __init__(self, num_joints=16):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 256 -> 128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 128 -> 64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 64 -> 32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_joints * 2),
            nn.Sigmoid()
        )

        self.num_joints = num_joints

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        x = x.view(-1, self.num_joints, 2)
        return x


# ============================================================
# 4. Loss：只计算 visible joints
# ============================================================

def pose_loss(pred_keypoints, target_keypoints, visibility):
    """
    pred_keypoints:   [B, 16, 2]
    target_keypoints: [B, 16, 2]
    visibility:       [B, 16]

    因为有些关键点可能不可见，所以 loss 只计算 visibility = 1 的点。
    """

    visibility = visibility.unsqueeze(-1)  # [B, 16, 1]

    squared_error = (pred_keypoints - target_keypoints) ** 2
    visible_error = squared_error * visibility

    loss = visible_error.sum() / (visibility.sum() * 2 + 1e-8)

    return loss


# ============================================================
# 5. 训练主程序
# ============================================================

def main():
    MPII_ROOT = "/Users/oanahuang/Desktop/MSc Project/Datasets/MPII"
    NPZ_PATH = "/Users/oanahuang/Desktop/MSc Project/Datasets/MPII/Annotations/mpii_processed.npz"

    device = get_device()

    dataset = MPIIDataset(
        mpii_root=MPII_ROOT,
        npz_path=NPZ_PATH,
        image_size=(256, 256)
    )

    # 先只用一小部分数据测试训练流程
    small_size = min(1000, len(dataset))
    rest_size = len(dataset) - small_size

    small_dataset, _ = random_split(
        dataset,
        [small_size, rest_size],
        generator=torch.Generator().manual_seed(42)
    )

    dataloader = DataLoader(
        small_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    model = SimplePoseCNN(num_joints=16).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 3

    print("\nModel:")
    print(model)

    print("\nStart training baseline...")
    print("Training samples:", small_size)
    print("Batch size:", 8)
    print("Epochs:", num_epochs)

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0

        for batch_idx, (images, keypoints, visibility) in enumerate(dataloader):
            images = images.to(device)
            keypoints = keypoints.to(device)
            visibility = visibility.to(device)

            pred_keypoints = model(images)

            loss = pose_loss(pred_keypoints, keypoints, visibility)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Average Loss: {avg_loss:.6f}"
        )

    save_path = Path(MPII_ROOT) / "Scripts" / "simple_pose_cnn.pth"
    torch.save(model.state_dict(), save_path)

    print("\nTraining finished.")
    print("Model saved to:", save_path)


if __name__ == "__main__":
    main()