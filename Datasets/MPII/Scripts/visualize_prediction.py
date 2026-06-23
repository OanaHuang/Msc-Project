from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


# ============================================================
# 1. Device
# ============================================================

def get_device():
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
# 2. Dataset
# ============================================================

class MPIIDataset(Dataset):
    def __init__(self, mpii_root, npz_path, image_size=(256, 256)):
        self.mpii_root = Path(mpii_root)
        self.npz_path = Path(npz_path)
        self.image_size = image_size

        data = np.load(self.npz_path, allow_pickle=True)

        self.image_paths = data["image_paths"]
        self.image_names = data["image_names"]
        self.keypoints = data["keypoints"].astype(np.float32)
        self.visibility = data["visibility"].astype(np.float32)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.mpii_root / self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size

        keypoints = self.keypoints[idx].copy()
        visibility = self.visibility[idx].copy()

        new_width, new_height = self.image_size

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        keypoints[:, 0] = keypoints[:, 0] * scale_x
        keypoints[:, 1] = keypoints[:, 1] * scale_y

        # 归一化到 0~1，和训练时保持一致
        keypoints[:, 0] = keypoints[:, 0] / new_width
        keypoints[:, 1] = keypoints[:, 1] / new_height

        image_tensor = self.transform(image)

        keypoints = torch.from_numpy(keypoints).float()
        visibility = torch.from_numpy(visibility).float()

        return image_tensor, keypoints, visibility, self.image_names[idx]


# ============================================================
# 3. Model：必须和 train_baseline.py 里一模一样
# ============================================================

class SimplePoseCNN(nn.Module):
    def __init__(self, num_joints=16):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

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
# 4. 可视化预测结果
# ============================================================

def visualize_sample(image_tensor, gt_keypoints, pred_keypoints, visibility, image_name):
    """
    gt_keypoints 和 pred_keypoints 都是 0~1 坐标。
    画图前要乘回 256。
    """

    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    gt = gt_keypoints.cpu().numpy().copy()
    pred = pred_keypoints.cpu().numpy().copy()
    vis = visibility.cpu().numpy()

    gt[:, 0] *= 256
    gt[:, 1] *= 256

    pred[:, 0] *= 256
    pred[:, 1] *= 256

    plt.figure(figsize=(7, 7))
    plt.imshow(image_np)

    for joint_id in range(16):
        if vis[joint_id] > 0:
            gx, gy = gt[joint_id]
            px, py = pred[joint_id]

            # ground truth
            plt.scatter(gx, gy, marker="o")
            plt.text(gx + 3, gy + 3, f"G{joint_id}", fontsize=8)

            # prediction
            plt.scatter(px, py, marker="x")
            plt.text(px + 3, py + 3, f"P{joint_id}", fontsize=8)

    plt.title(f"Prediction vs Ground Truth: {image_name}")
    plt.axis("off")
    plt.show()


# ============================================================
# 5. 主程序
# ============================================================

def main():
    MPII_ROOT = "/Users/oanahuang/Desktop/MSc Project/Datasets/MPII"
    NPZ_PATH = "/Users/oanahuang/Desktop/MSc Project/Datasets/MPII/Annotations/mpii_processed.npz"
    MODEL_PATH = "/Users/oanahuang/Desktop/MSc Project/Datasets/MPII/Scripts/simple_pose_cnn.pth"

    device = get_device()

    dataset = MPIIDataset(
        mpii_root=MPII_ROOT,
        npz_path=NPZ_PATH,
        image_size=(256, 256)
    )

    model = SimplePoseCNN(num_joints=16).to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    # 换这个数字可以看不同图片
    sample_idx = 20

    image, gt_keypoints, visibility, image_name = dataset[sample_idx]

    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_keypoints = model(image_input)[0]

    print("Image name:", image_name)
    print("Image tensor shape:", image.shape)
    print("GT keypoints shape:", gt_keypoints.shape)
    print("Pred keypoints shape:", pred_keypoints.shape)
    print("Visibility shape:", visibility.shape)

    visualize_sample(
        image_tensor=image,
        gt_keypoints=gt_keypoints,
        pred_keypoints=pred_keypoints.cpu(),
        visibility=visibility,
        image_name=image_name
    )


if __name__ == "__main__":
    main()