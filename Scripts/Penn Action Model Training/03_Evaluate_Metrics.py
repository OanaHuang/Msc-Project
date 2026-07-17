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
PER_JOINT_CSV = OUTPUT_DIR / "metrics_per_joint.csv"

# None = evaluate all videos
TARGET_VIDEO_IDS = ["0684"]

# Quick test: 10 or 100
# Final full evaluation: None
MAX_VIDEOS = None

# PCK thresholds based on visible keypoint bounding-box scale
PCK_THRESHOLDS = [0.05, 0.10, 0.20, 0.50]


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

        # The checkpoint already contains trained weights.
        # Here weights=None is correct because we load model_state_dict below.
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

    raise FileNotFoundError(
        f"Image not found.\n"
        f"Original path: {p_original}\n"
        f"Processed path: {p}\n"
        f"Tried:\n" + "\n".join([f"  {c}" for c in candidates])
    )


def compute_visible_mask(visibility, pred, gt):
    """
    Evaluate only visible and valid keypoints.

    visibility: [T, K]
    pred:       [T, K, 2]
    gt:         [T, K, 2]
    """
    visible = visibility > 0
    valid_pred = ~np.isnan(pred).any(axis=-1)
    valid_gt = ~np.isnan(gt).any(axis=-1)

    return visible & valid_pred & valid_gt


def euclidean_distance(pred, gt):
    """
    pred: [T, K, 2]
    gt:   [T, K, 2]

    return:
        dist: [T, K]
    """
    return np.linalg.norm(pred - gt, axis=-1)


def estimate_reference_length(gt, visibility):
    """
    Estimate per-frame reference length for PCK.

    Penn Action does not provide MPII-style head size.
    Therefore, this script uses the visible keypoint bounding-box scale.

    reference length = max(width, height) of visible GT keypoint box

    return:
        ref_lengths: [T]
    """
    T, K, _ = gt.shape
    ref_lengths = np.zeros(T, dtype=np.float32)

    for t in range(T):
        visible = visibility[t] > 0
        valid = ~np.isnan(gt[t]).any(axis=-1)
        mask = visible & valid

        if mask.sum() < 2:
            ref_lengths[t] = np.nan
            continue

        xs = gt[t, mask, 0]
        ys = gt[t, mask, 1]

        width = xs.max() - xs.min()
        height = ys.max() - ys.min()

        scale = max(width, height)

        if scale <= 1:
            ref_lengths[t] = np.nan
        else:
            ref_lengths[t] = scale

    return ref_lengths


def compute_mse(pred, gt, mask):
    """
    MSE over visible keypoints and both x/y coordinate values.
    """
    error = (pred - gt) ** 2
    error = error[mask]

    if error.size == 0:
        return np.nan

    return float(error.mean())


def compute_rmse(pred, gt, mask):
    mse = compute_mse(pred, gt, mask)

    if np.isnan(mse):
        return np.nan

    return float(np.sqrt(mse))


def compute_mae(pred, gt, mask):
    """
    MAE over visible keypoints and both x/y coordinate values.
    """
    error = np.abs(pred - gt)
    error = error[mask]

    if error.size == 0:
        return np.nan

    return float(error.mean())


def compute_mean_pixel_error(dist, mask):
    """
    Mean Euclidean keypoint distance in pixels.
    """
    valid_dist = dist[mask]

    if valid_dist.size == 0:
        return np.nan

    return float(valid_dist.mean())


def compute_pck_with_frame_reference(dist, mask, ref_lengths, threshold_ratio):
    """
    PCK using per-frame reference length.

    threshold_pixels[t] = threshold_ratio * ref_lengths[t]
    """
    threshold_pixels = threshold_ratio * ref_lengths
    threshold_pixels = threshold_pixels[:, None]

    valid_ref = ~np.isnan(threshold_pixels)
    final_mask = mask & valid_ref

    valid_dist = dist[final_mask]
    valid_thresholds = np.broadcast_to(threshold_pixels, dist.shape)[final_mask]

    if valid_dist.size == 0:
        return np.nan

    correct = valid_dist < valid_thresholds

    return float(correct.mean() * 100.0)


def compute_temporal_accel(all_preds, all_viss):
    """
    Mean acceleration of predicted keypoints.

    accel[t] = pred[t] - 2 * pred[t-1] + pred[t-2]

    This is useful for measuring temporal jitter.
    Lower value usually means smoother prediction.
    """
    total_accel = 0.0
    total_points = 0

    for preds, viss in zip(all_preds, all_viss):
        if len(preds) < 3:
            continue

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
            total_points += len(accel_norm)

    if total_points == 0:
        return np.nan

    return float(total_accel / total_points)


def compute_per_joint_metrics(all_preds, all_gts, all_viss, keypoint_names, pck_threshold_ratio=0.1):
    """
    Compute per-joint:
    - visible count
    - mean pixel error
    - median pixel error
    - PCK@0.1
    """
    num_keypoints = all_preds[0].shape[1]

    joint_distances = [[] for _ in range(num_keypoints)]
    joint_correct = np.zeros(num_keypoints, dtype=np.float32)
    joint_total = np.zeros(num_keypoints, dtype=np.float32)

    for preds, gts, viss in zip(all_preds, all_gts, all_viss):
        dist = euclidean_distance(preds, gts)
        mask = compute_visible_mask(viss, preds, gts)
        ref_lengths = estimate_reference_length(gts, viss)
        thresholds = pck_threshold_ratio * ref_lengths

        T, K = dist.shape

        for t in range(T):
            if np.isnan(thresholds[t]):
                continue

            for k in range(K):
                if not mask[t, k]:
                    continue

                d = dist[t, k]

                joint_distances[k].append(float(d))
                joint_total[k] += 1

                if d < thresholds[t]:
                    joint_correct[k] += 1

    results = []

    for k in range(num_keypoints):
        distances = np.array(joint_distances[k], dtype=np.float32)

        if distances.size == 0:
            mean_error = np.nan
            median_error = np.nan
            pck = np.nan
            count = 0
        else:
            mean_error = float(distances.mean())
            median_error = float(np.median(distances))
            count = int(distances.size)

            if joint_total[k] > 0:
                pck = float(joint_correct[k] / joint_total[k] * 100.0)
            else:
                pck = np.nan

        name = keypoint_names[k] if k < len(keypoint_names) else f"joint_{k}"

        results.append({
            "joint_index": k,
            "joint_name": name,
            "visible_count": count,
            "mean_pixel_error": mean_error,
            "median_pixel_error": median_error,
            f"PCK@{pck_threshold_ratio}": pck,
        })

    return results


def predict_video(model, transform, image_paths, keypoints, visibility, frame_indices, indices, image_size):
    """
    Run frame-by-frame inference for one video.
    """
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

            # Convert from resized image coordinate system back to original frame size
            pred[:, 0] *= orig_w / image_size
            pred[:, 1] *= orig_h / image_size

            preds.append(pred)
            gts.append(keypoints[idx])
            viss.append(visibility[idx])

    return np.stack(preds), np.stack(gts), np.stack(viss)


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
    keypoints = data["keypoints"].astype(np.float32)
    visibility = data["visibility"].astype(np.float32)
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

        all_preds.append(preds)
        all_gts.append(gts)
        all_viss.append(viss)

    elapsed = time.time() - start

    if len(all_preds) == 0:
        raise RuntimeError("No videos were evaluated. Please check TARGET_VIDEO_IDS and dataset paths.")

    # ========================================================
    # 7. Compute global metrics
    # ========================================================

    all_pred_array = np.concatenate(all_preds, axis=0)
    all_gt_array = np.concatenate(all_gts, axis=0)
    all_vis_array = np.concatenate(all_viss, axis=0)

    dist = euclidean_distance(all_pred_array, all_gt_array)
    mask = compute_visible_mask(all_vis_array, all_pred_array, all_gt_array)
    ref_lengths = estimate_reference_length(all_gt_array, all_vis_array)

    visible_count = int(mask.sum())
    total_count = int(mask.size)

    mse = compute_mse(all_pred_array, all_gt_array, mask)
    rmse = compute_rmse(all_pred_array, all_gt_array, mask)
    mae = compute_mae(all_pred_array, all_gt_array, mask)
    mean_pixel_error = compute_mean_pixel_error(dist, mask)
    mean_accel = compute_temporal_accel(all_preds, all_viss)

    summary = {
        "method": "ResNet18_CoordReg_ImageNet",
        "num_videos": len(selected_video_ids),
        "num_frames": int(all_pred_array.shape[0]),
        "num_keypoints": int(all_pred_array.shape[1]),
        "visible_keypoints": visible_count,
        "total_keypoints": total_count,
        "visibility_ratio_percent": visible_count / max(total_count, 1) * 100.0,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "mean_pixel_error": mean_pixel_error,
        "mean_accel": mean_accel,
        "elapsed_time_sec": elapsed,
    }

    for thr in PCK_THRESHOLDS:
        pck_value = compute_pck_with_frame_reference(
            dist=dist,
            mask=mask,
            ref_lengths=ref_lengths,
            threshold_ratio=thr,
        )
        summary[f"PCK@{thr}"] = pck_value

    per_joint_results = compute_per_joint_metrics(
        all_preds=all_preds,
        all_gts=all_gts,
        all_viss=all_viss,
        keypoint_names=JOINT_NAMES,
        pck_threshold_ratio=0.2,
    )

    # ========================================================
    # 8. Print results
    # ========================================================

    print("\n==============================")
    print("ResNet18 ImageNet Coordinate Regression Metrics")
    print("==============================")
    print(f"Videos: {len(selected_video_ids)}")
    print(f"Frames: {all_pred_array.shape[0]}")
    print(f"Keypoints: {all_pred_array.shape[1]}")
    print(f"Visible keypoints: {visible_count}/{total_count}")
    print(f"Visibility ratio: {summary['visibility_ratio_percent']:.2f}%")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Mean pixel error: {mean_pixel_error:.4f}")
    print(f"Mean accel: {mean_accel:.4f}")

    for thr in PCK_THRESHOLDS:
        value = summary[f"PCK@{thr}"]
        print(f"PCK@{thr}: {value:.2f}%")

    print("\nPer-joint results:")
    for r in per_joint_results:
        print(
            f"{r['joint_index']:02d} {r['joint_name']:<15} | "
            f"count={r['visible_count']:<5} | "
            f"mean_error={r['mean_pixel_error']:.2f} | "
            f"median_error={r['median_pixel_error']:.2f} | "
            f"PCK@0.2={r['PCK@0.2']:.2f}%"
        )

    # ========================================================
    # 9. Save CSV files
    # ========================================================

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with open(PER_JOINT_CSV, "w", newline="") as f:
        fieldnames = list(per_joint_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_joint_results)

    print("\nEvaluation finished.")
    print(f"Elapsed time: {elapsed:.2f}s")

    print("\nSaved files:")
    print(f"  {SUMMARY_CSV}")
    print(f"  {PER_JOINT_CSV}")


if __name__ == "__main__":
    main()