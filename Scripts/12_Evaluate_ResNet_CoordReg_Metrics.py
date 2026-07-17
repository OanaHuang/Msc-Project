# Scripts/12_Evaluate_ResNet_CoordReg_Metrics.py

from pathlib import Path
import csv
import numpy as np


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "f2f_all_coordreg_v1"
    / "predictions"
    / "0684_all_keypoints_predictions_smallvis.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "f2f_all_coordreg_v1"
    / "metrics"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV_PATH = OUTPUT_DIR / "0024_resnet_coordreg_metrics_summary.csv"
PER_JOINT_CSV_PATH = OUTPUT_DIR / "0024_resnet_coordreg_per_joint_metrics.csv"


# Penn Action 13-keypoint common order assumption
KEYPOINT_NAMES = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


# PCK thresholds relative to image size
# 0.2 means 20% of reference length
PCK_THRESHOLDS = [0.05, 0.10, 0.20, 0.50]


# ============================================================
# 2. Metric helpers
# ============================================================

def euclidean_distance(pred, gt):
    """
    pred: [T, K, 2]
    gt:   [T, K, 2]

    return:
        dist: [T, K]
    """
    return np.linalg.norm(pred - gt, axis=-1)


def compute_visible_mask(visibility, pred, gt):
    """
    Only evaluate visible and valid keypoints.
    visibility: [T, K]
    pred: [T, K, 2]
    gt: [T, K, 2]
    """
    visible = visibility > 0

    valid_pred = ~np.isnan(pred).any(axis=-1)
    valid_gt = ~np.isnan(gt).any(axis=-1)

    return visible & valid_pred & valid_gt


def compute_mse(pred, gt, mask):
    """
    MSE over visible keypoints.
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
    MAE over x and y coordinate values.
    """
    error = np.abs(pred - gt)
    error = error[mask]

    if error.size == 0:
        return np.nan

    return float(error.mean())


def compute_mean_pixel_error(dist, mask):
    """
    Mean Euclidean distance in pixels.
    """
    valid_dist = dist[mask]

    if valid_dist.size == 0:
        return np.nan

    return float(valid_dist.mean())


def compute_pck(dist, mask, threshold_pixels):
    """
    PCK = percentage of visible keypoints with distance < threshold.
    """
    valid_dist = dist[mask]

    if valid_dist.size == 0:
        return np.nan

    correct = valid_dist < threshold_pixels
    return float(correct.mean() * 100.0)


def estimate_reference_length(gt, visibility):
    """
    Estimate a per-frame reference length for PCK.

    Since Penn Action does not directly provide MPII-style head size,
    this script uses the image/body scale approximation from visible keypoints.

    Method:
        For each frame, compute bounding box of visible ground truth keypoints.
        Reference length = max(width, height) of the visible keypoint box.

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

        ref_lengths[t] = max(width, height)

    return ref_lengths


def compute_pck_with_frame_reference(dist, mask, ref_lengths, threshold_ratio):
    """
    PCK using per-frame reference length.

    threshold_pixels[t] = threshold_ratio * ref_lengths[t]
    """
    T, K = dist.shape

    threshold_pixels = threshold_ratio * ref_lengths
    threshold_pixels = threshold_pixels[:, None]  # [T, 1]

    valid_ref = ~np.isnan(threshold_pixels)
    final_mask = mask & valid_ref

    valid_dist = dist[final_mask]
    valid_thresholds = np.broadcast_to(threshold_pixels, dist.shape)[final_mask]

    if valid_dist.size == 0:
        return np.nan

    correct = valid_dist < valid_thresholds
    return float(correct.mean() * 100.0)


def compute_per_joint_metrics(pred, gt, visibility, keypoint_names, pck_threshold_ratio=0.1):
    """
    Compute per-joint mean error and PCK.
    """
    dist = euclidean_distance(pred, gt)
    mask = compute_visible_mask(visibility, pred, gt)
    ref_lengths = estimate_reference_length(gt, visibility)

    results = []

    T, K = dist.shape

    for k in range(K):
        joint_mask = mask[:, k]

        valid_dist = dist[:, k][joint_mask]

        if valid_dist.size == 0:
            mean_error = np.nan
            median_error = np.nan
            pck = np.nan
            count = 0
        else:
            mean_error = float(valid_dist.mean())
            median_error = float(np.median(valid_dist))

            thresholds = pck_threshold_ratio * ref_lengths
            valid_ref = ~np.isnan(thresholds)

            final_mask = joint_mask & valid_ref
            valid_joint_dist = dist[:, k][final_mask]
            valid_joint_thresholds = thresholds[final_mask]

            if valid_joint_dist.size == 0:
                pck = np.nan
            else:
                pck = float((valid_joint_dist < valid_joint_thresholds).mean() * 100.0)

            count = int(valid_dist.size)

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


# ============================================================
# 3. Main
# ============================================================

def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PRED_PATH}\n\n"
            f"Please run Scripts/08_infer_video_f2f_all_coordreg_baseline.py first."
        )

    data = np.load(PRED_PATH, allow_pickle=True)

    print(f"Loaded prediction file: {PRED_PATH}")
    print("Available keys:", data.files)

    pred_keypoints = data["pred_keypoints"].astype(np.float32)
    gt_keypoints = data["gt_keypoints"].astype(np.float32)
    visibility = data["visibility"].astype(np.float32)

    video_id = str(data["video_id"]) if "video_id" in data.files else "unknown"
    method = str(data["method"]) if "method" in data.files else "ResNet Coordinate Regression"

    print(f"Video ID: {video_id}")
    print(f"Method: {method}")
    print(f"Prediction shape: {pred_keypoints.shape}")
    print(f"Ground truth shape: {gt_keypoints.shape}")
    print(f"Visibility shape: {visibility.shape}")

    # ----------------------------
    # Basic checks
    # ----------------------------
    if pred_keypoints.shape != gt_keypoints.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_keypoints.shape}, gt {gt_keypoints.shape}"
        )

    if visibility.shape != pred_keypoints.shape[:2]:
        raise ValueError(
            f"Visibility shape mismatch: visibility {visibility.shape}, "
            f"expected {pred_keypoints.shape[:2]}"
        )

    T, K, _ = pred_keypoints.shape

    # ----------------------------
    # Main metrics
    # ----------------------------
    dist = euclidean_distance(pred_keypoints, gt_keypoints)
    mask = compute_visible_mask(visibility, pred_keypoints, gt_keypoints)

    visible_count = int(mask.sum())
    total_count = int(mask.size)

    mse = compute_mse(pred_keypoints, gt_keypoints, mask)
    rmse = compute_rmse(pred_keypoints, gt_keypoints, mask)
    mae = compute_mae(pred_keypoints, gt_keypoints, mask)
    mean_pixel_error = compute_mean_pixel_error(dist, mask)

    ref_lengths = estimate_reference_length(gt_keypoints, visibility)

    summary = {
        "video_id": video_id,
        "method": method,
        "num_frames": T,
        "num_keypoints": K,
        "visible_keypoints": visible_count,
        "total_keypoints": total_count,
        "visibility_ratio_percent": visible_count / max(total_count, 1) * 100.0,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "mean_pixel_error": mean_pixel_error,
    }

    for thr in PCK_THRESHOLDS:
        pck_value = compute_pck_with_frame_reference(
            dist=dist,
            mask=mask,
            ref_lengths=ref_lengths,
            threshold_ratio=thr
        )
        summary[f"PCK@{thr}"] = pck_value

    # Also keep original video_mse from 08 script if available
    if "video_mse" in data.files:
        summary["video_mse_from_inference_script"] = float(data["video_mse"])

    # ----------------------------
    # Per-joint metrics
    # ----------------------------
    per_joint_results = compute_per_joint_metrics(
        pred=pred_keypoints,
        gt=gt_keypoints,
        visibility=visibility,
        keypoint_names=KEYPOINT_NAMES,
        pck_threshold_ratio=0.2,
    )

    # ----------------------------
    # Print results
    # ----------------------------
    print("\n==============================")
    print("ResNet Coordinate Regression Metrics")
    print("==============================")
    print(f"Video ID: {video_id}")
    print(f"Method: {method}")
    print(f"Frames: {T}")
    print(f"Keypoints: {K}")
    print(f"Visible keypoints: {visible_count}/{total_count}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Mean pixel error: {mean_pixel_error:.4f}")

    for thr in PCK_THRESHOLDS:
        value = summary[f"PCK@{thr}"]
        print(f"PCK@{thr}: {value:.2f}%")

    print("\nPer-joint results:")
    for r in per_joint_results:
        print(
            f"{r['joint_index']:02d} {r['joint_name']:<15} | "
            f"count={r['visible_count']:<4} | "
            f"mean_error={r['mean_pixel_error']:.2f} | "
            f"PCK@0.2={r['PCK@0.2']:.2f}%"
        )

    # ----------------------------
    # Save summary CSV
    # ----------------------------
    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # ----------------------------
    # Save per-joint CSV
    # ----------------------------
    with open(PER_JOINT_CSV_PATH, "w", newline="") as f:
        fieldnames = list(per_joint_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_joint_results)

    print("\nSaved summary metrics to:")
    print(SUMMARY_CSV_PATH)

    print("Saved per-joint metrics to:")
    print(PER_JOINT_CSV_PATH)


if __name__ == "__main__":
    main()