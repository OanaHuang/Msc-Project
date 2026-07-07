# Scripts/Penn Action Model Training/03_Evaluate_Metrics.py

from pathlib import Path
import csv
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NPZ_PATH = PROJECT_ROOT / "Datasets" / "Penn_Action" / "penn_action_processed.npz"

CKPT_PATH = (
    PROJECT_ROOT
    / "server_outputs"
    / "01_Resnet_with_ImageNet"
    / "best_Resnet_with_ImageNet.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "03_Evaluate_Metrics"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "metrics_summary.csv"
PCK_BY_PART_CSV = OUTPUT_DIR / "metrics_pck_by_part.csv"
PCK_BY_JOINT_CSV = OUTPUT_DIR / "metrics_pck_by_joint.csv"

# None = evaluate all videos
TARGET_VIDEO_IDS = None

# Quick test: 10 or 100
# Final full evaluation: None
MAX_VIDEOS = 100

PCK_THRESHOLD = 0.2


# ============================================================
# 2. Keypoint Definition
# ============================================================

JOINT_NAMES = [
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hip",
    "Right_Hip",
    "Left_Knee",
    "Right_Knee",
    "Left_Ankle",
    "Right_Ankle",
]

PART_GROUPS = {
    "Head": [0],
    "Sho": [1, 2],
    "Elb": [3, 4],
    "Wri": [5, 6],
    "Hip": [7, 8],
    "Knee": [9, 10],
    "Ank": [11, 12],
}


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
# 4. Model
# ============================================================

class ResNetWithImageNetAllKeypointRegressor(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_keypoints * 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, self.num_keypoints, 2)
        return x


# ============================================================
# 5. Helper Functions
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

    path = Path(p)

    candidates = [
        path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "Datasets" / "Penn_Action" / path,
        PROJECT_ROOT / "Datasets" / "Penn_Action" / "frames" / path,
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Image not found: {p_original}")


def compute_person_scale(gt_kpts, visibility):
    visible = visibility > 0

    if visible.sum() < 2:
        return None

    pts = gt_kpts[visible]

    width = pts[:, 0].max() - pts[:, 0].min()
    height = pts[:, 1].max() - pts[:, 1].min()

    scale = max(width, height)

    if scale <= 1:
        return None

    return float(scale)


def predict_video(model, transform, image_paths, keypoints, visibility, frame_indices, indices, image_size):
    order = np.argsort(frame_indices[indices])
    indices = indices[order]

    preds = []
    gts = []
    viss = []

    with torch.no_grad():
        for idx in indices:
            img_path = resolve_image_path(image_paths[idx])

            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            x = transform(img).unsqueeze(0).to(DEVICE)

            pred = model(x)[0].detach().cpu().numpy()

            pred[:, 0] *= orig_w / image_size
            pred[:, 1] *= orig_h / image_size

            preds.append(pred)
            gts.append(keypoints[idx])
            viss.append(visibility[idx])

    return np.stack(preds), np.stack(gts), np.stack(viss)


def update_pck_counts(preds, gts, viss, correct, total):
    """
    Updates correct and total count for each keypoint.
    """
    num_frames, num_keypoints, _ = preds.shape

    for t in range(num_frames):
        pred = preds[t]
        gt = gts[t]
        vis = viss[t]

        scale = compute_person_scale(gt, vis)

        if scale is None:
            continue

        threshold = PCK_THRESHOLD * scale

        for j in range(num_keypoints):
            if vis[j] <= 0:
                continue

            dist = np.linalg.norm(pred[j] - gt[j])

            total[j] += 1

            if dist < threshold:
                correct[j] += 1


def compute_part_pck_percent(correct, total):
    result = {}

    all_correct = 0
    all_total = 0

    for part_name, joint_ids in PART_GROUPS.items():
        part_correct = correct[joint_ids].sum()
        part_total = total[joint_ids].sum()

        if part_total > 0:
            result[part_name] = float(part_correct / part_total * 100.0)
        else:
            result[part_name] = np.nan

        all_correct += part_correct
        all_total += part_total

    if all_total > 0:
        result["Mean"] = float(all_correct / all_total * 100.0)
    else:
        result["Mean"] = np.nan

    return result


def compute_joint_pck_percent(correct, total):
    result = {}

    for i, name in enumerate(JOINT_NAMES):
        if total[i] > 0:
            result[name] = float(correct[i] / total[i] * 100.0)
        else:
            result[name] = np.nan

    return result


def compute_mean_pixel_error_and_accel(all_preds, all_gts, all_viss):
    total_dist = 0.0
    total_points = 0

    total_accel = 0.0
    total_accel_points = 0

    for preds, gts, viss in zip(all_preds, all_gts, all_viss):
        # Mean Pixel Error
        visible = viss > 0
        diff = preds - gts
        dist = np.linalg.norm(diff, axis=2)

        total_dist += dist[visible].sum()
        total_points += visible.sum()

        # Accel
        if len(preds) >= 3:
            for t in range(2, len(preds)):
                valid = (
                    (viss[t] > 0)
                    & (viss[t - 1] > 0)
                    & (viss[t - 2] > 0)
                )

                if valid.sum() == 0:
                    continue

                accel = preds[t, valid] - 2 * preds[t - 1, valid] + preds[t - 2, valid]
                accel_norm = np.linalg.norm(accel, axis=1)

                total_accel += accel_norm.sum()
                total_accel_points += len(accel_norm)

    mean_pixel_error = total_dist / total_points if total_points > 0 else np.nan
    mean_accel = total_accel / total_accel_points if total_accel_points > 0 else np.nan

    return float(mean_pixel_error), float(mean_accel)


# ============================================================
# 6. Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ not found: {NPZ_PATH}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    print(f"NPZ path: {NPZ_PATH}")
    print(f"Checkpoint path: {CKPT_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")

    data = np.load(NPZ_PATH, allow_pickle=True)

    image_paths = data["image_paths"]
    keypoints = data["keypoints"]
    visibility = data["visibility"]
    video_ids = np.array([safe_video_id_to_str(v) for v in data["video_ids"]])
    frame_indices = data["frame_indices"]

    all_video_ids = sorted(list(set(video_ids)))

    if TARGET_VIDEO_IDS is None:
        selected_video_ids = all_video_ids
    else:
        selected_video_ids = [str(v).zfill(4) for v in TARGET_VIDEO_IDS]

    if MAX_VIDEOS is not None:
        selected_video_ids = selected_video_ids[:MAX_VIDEOS]

    print(f"Total available videos: {len(all_video_ids)}")
    print(f"Videos to evaluate: {len(selected_video_ids)}")
    print(f"First 10 selected videos: {selected_video_ids[:10]}")

    print("\nLoading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    image_size = ckpt["image_size"]
    num_keypoints = ckpt["num_keypoints"]

    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Image size: {image_size}")
    print(f"Number of keypoints: {num_keypoints}")
    print(f"Best Val MSE: {ckpt.get('best_val_mse', 'unknown')}")

    model = ResNetWithImageNetAllKeypointRegressor(num_keypoints)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    correct = np.zeros(num_keypoints)
    total = np.zeros(num_keypoints)

    all_preds = []
    all_gts = []
    all_viss = []

    start = time.time()

    for i, video_id in enumerate(selected_video_ids):
        indices = np.where(video_ids == video_id)[0]

        if len(indices) == 0:
            continue

        print(f"\n[{i + 1}/{len(selected_video_ids)}] Evaluating video {video_id}, frames={len(indices)}")

        preds, gts, viss = predict_video(
            model=model,
            transform=transform,
            image_paths=image_paths,
            keypoints=keypoints,
            visibility=visibility,
            frame_indices=frame_indices,
            indices=indices,
            image_size=image_size,
        )

        update_pck_counts(preds, gts, viss, correct, total)

        all_preds.append(preds)
        all_gts.append(gts)
        all_viss.append(viss)

    part_pck = compute_part_pck_percent(correct, total)
    joint_pck = compute_joint_pck_percent(correct, total)
    mean_pixel_error, mean_accel = compute_mean_pixel_error_and_accel(all_preds, all_gts, all_viss)

    # Save summary
    summary = {
        "method": "ResNet18_CoordReg_ImageNet",
        "num_videos": len(selected_video_ids),
        "mean_pixel_error": mean_pixel_error,
        "mean_pck_0.2_percent": part_pck["Mean"],
        "mean_accel": mean_accel,
    }

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # Save paper-style PCK by part
    part_row = {
        "method": "ResNet18_CoordReg_ImageNet",
        "num_videos": len(selected_video_ids),
        "Head": part_pck["Head"],
        "Sho": part_pck["Sho"],
        "Elb": part_pck["Elb"],
        "Wri": part_pck["Wri"],
        "Hip": part_pck["Hip"],
        "Knee": part_pck["Knee"],
        "Ank": part_pck["Ank"],
        "Mean": part_pck["Mean"],
    }

    with open(PCK_BY_PART_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(part_row.keys()))
        writer.writeheader()
        writer.writerow(part_row)

    # Save PCK by individual joint
    joint_row = {
        "method": "ResNet18_CoordReg_ImageNet",
        "num_videos": len(selected_video_ids),
    }
    joint_row.update(joint_pck)

    with open(PCK_BY_JOINT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(joint_row.keys()))
        writer.writeheader()
        writer.writerow(joint_row)

    elapsed = time.time() - start

    print("\nEvaluation finished.")
    print(f"Elapsed time: {elapsed:.2f}s")

    print("\nSaved files:")
    print(f"  {SUMMARY_CSV}")
    print(f"  {PCK_BY_PART_CSV}")
    print(f"  {PCK_BY_JOINT_CSV}")

    print("\nPaper-style PCK by part:")
    print(
        f"Head={part_pck['Head']:.2f}, "
        f"Sho={part_pck['Sho']:.2f}, "
        f"Elb={part_pck['Elb']:.2f}, "
        f"Wri={part_pck['Wri']:.2f}, "
        f"Hip={part_pck['Hip']:.2f}, "
        f"Knee={part_pck['Knee']:.2f}, "
        f"Ank={part_pck['Ank']:.2f}, "
        f"Mean={part_pck['Mean']:.2f}"
    )

    print("\nMain summary:")
    print(f"Mean Pixel Error: {mean_pixel_error:.2f}")
    print(f"PCK@0.2: {part_pck['Mean']:.2f}")
    print(f"Accel: {mean_accel:.2f}")


if __name__ == "__main__":
    main()