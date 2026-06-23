from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


# ============================================================
# 1. PyTorch Dataset 类
# ============================================================

class MPIIDataset(Dataset):
    """
    读取已经处理好的 MPII npz 文件。

    每次返回:
        image:      torch.Tensor, shape = [3, H, W]
        keypoints:  torch.Tensor, shape = [16, 2]
        visibility: torch.Tensor, shape = [16]
    """

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
        self.image_names = data["image_names"]
        self.keypoints = data["keypoints"].astype(np.float32)
        self.visibility = data["visibility"].astype(np.float32)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        print("MPIIDataset loaded successfully.")
        print("Number of samples:", len(self.image_paths))
        print("Image paths shape:", self.image_paths.shape)
        print("Keypoints shape:", self.keypoints.shape)
        print("Visibility shape:", self.visibility.shape)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ------------------------------------------------------------
        # 1. 读取图片
        # ------------------------------------------------------------
        img_path = self.mpii_root / self.image_paths[idx]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        original_width, original_height = image.size

        # ------------------------------------------------------------
        # 2. 读取关键点和可见性
        # ------------------------------------------------------------
        keypoints = self.keypoints[idx].copy()
        visibility = self.visibility[idx].copy()

        # ------------------------------------------------------------
        # 3. 图片 resize 后，关键点坐标也要同步缩放
        # ------------------------------------------------------------
        new_width, new_height = self.image_size

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        keypoints[:, 0] = keypoints[:, 0] * scale_x
        keypoints[:, 1] = keypoints[:, 1] * scale_y

        # ------------------------------------------------------------
        # 4. 图片转成 tensor
        # ------------------------------------------------------------
        image = self.transform(image)

        keypoints = torch.from_numpy(keypoints).float()
        visibility = torch.from_numpy(visibility).float()

        return image, keypoints, visibility


# ============================================================
# 2. 可视化检查函数
# ============================================================

def show_sample(image_tensor, keypoints, visibility, title="MPII sample"):
    """
    把 Dataset 读取出来的 tensor 再画出来检查。
    """

    # image tensor: [3, H, W] -> [H, W, 3]
    image_np = image_tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)

    keypoints_np = keypoints.numpy()
    visibility_np = visibility.numpy()

    for joint_id in range(16):
        if visibility_np[joint_id] > 0:
            x, y = keypoints_np[joint_id]
            plt.scatter(x, y)
            plt.text(x + 3, y + 3, str(joint_id), fontsize=8)

    plt.title(title)
    plt.axis("off")
    plt.show()


# ============================================================
# 3. 主测试程序
# ============================================================

if __name__ == "__main__":

    MPII_ROOT = "/Users/oanahuang/Desktop/MSc Project/Datasets/MPII"
    NPZ_PATH = "/Users/oanahuang/Desktop/MSc Project/Datasets/MPII/Annotations/mpii_processed.npz"

    # ------------------------------------------------------------
    # 1. 创建 Dataset
    # ------------------------------------------------------------
    dataset = MPIIDataset(
        mpii_root=MPII_ROOT,
        npz_path=NPZ_PATH,
        image_size=(256, 256)
    )

    print("\nDataset length:", len(dataset))

    # ------------------------------------------------------------
    # 2. 读取单个样本
    # ------------------------------------------------------------
    sample_idx = 20

    image, keypoints, visibility = dataset[sample_idx]

    print("\nSingle sample:")
    print("Sample index:", sample_idx)
    print("Image tensor shape:", image.shape)
    print("Keypoints tensor shape:", keypoints.shape)
    print("Visibility tensor shape:", visibility.shape)
    print("Visible joints:", visibility.sum().item())

    print("\nKeypoints:")
    print(keypoints)

    print("\nVisibility:")
    print(visibility)

    # 可视化检查
    show_sample(
        image,
        keypoints,
        visibility,
        title=f"MPII PyTorch Dataset sample {sample_idx}"
    )

    # ------------------------------------------------------------
    # 3. 创建 DataLoader
    # ------------------------------------------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    images_batch, keypoints_batch, visibility_batch = next(iter(dataloader))

    print("\nBatch sample:")
    print("Images batch shape:", images_batch.shape)
    print("Keypoints batch shape:", keypoints_batch.shape)
    print("Visibility batch shape:", visibility_batch.shape)

    print("\nPyTorch Dataset and DataLoader test finished successfully.")