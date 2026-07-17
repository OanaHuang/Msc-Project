# Scripts/Penn Action Model Training/06_Evaluate_Heatmap_Metrics.py

from pathlib import Path
import csv

import numpy as np


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NPZ_PATH = PROJECT_ROOT / "Datasets" / "Penn_Action" / "penn_action_processed.npz"

PRED_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "05_Generate_MP4_Heatmap"
    / "0684_heatmap_predictions.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "06_Evaluate_Heatmap_Metrics"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "heatmap_metrics_summary.csv"
PER_JOINT_CSV = OUTPUT_DIR / "heatmap_metrics_per_joint.csv"

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
# 3. Helper Functions
# ============================================================

def safe_video_id_to_str(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8")

    x = str(x)

    if x.isdigit():
        return x.zfill(4)

    return x


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


def compute_temporal_accel(preds, viss):
    """
    Mean acceleration of predicted keypoints.

    accel[t] = pred[t] - 2 * pred[t-1] + pred[t-2]

    Lower value usually means smoother prediction.
    """
    total_accel = 0.0
    total_count = 0

    if len(preds) < 3:
        return np.nan

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
        total_count += len(accel_norm)

    if total_count == 0:
        return np.nan

    return float(total_accel / total_count)


def compute_per_joint_metrics(preds, gts, viss, keypoint_names, pck_threshold_ratio=0.2):
    """
    Compute per-joint:
    - visible count
    - mean pixel error
    - median pixel error
    - PCK@0.2
    """
    num_keypoints = preds.shape[1]

    dist = euclidean_distance(preds, gts)
    mask = compute_visible_mask(viss, preds, gts)
    ref_lengths = estimate_reference_length(gts, viss)
    thresholds = pck_threshold_ratio * ref_lengths

    results = []

    for k in range(num_keypoints):
        joint_mask = mask[:, k]

        valid_distances = dist[:, k][joint_mask]

        if valid_distances.size == 0:
            mean_error = np.nan
            median_error = np.nan
            pck = np.nan
            count = 0
        else:
            mean_error = float(valid_distances.mean())
            median_error = float(np.median(valid_distances))
            count = int(valid_distances.size)

            valid_ref = ~np.isnan(thresholds)
            final_mask = joint_mask & valid_ref

            joint_dist = dist[:, k][final_mask]
            joint_thresholds = thresholds[final_mask]

            if joint_dist.size == 0:
                pck = np.nan
            else:
                pck = float((joint_dist < joint_thresholds).mean() * 100.0)

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
# 4. Main
# ============================================================

def main():
    print("Evaluating heatmap predictions...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"GT NPZ path: {NPZ_PATH}")
    print(f"Prediction path: {PRED_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"GT NPZ not found: {NPZ_PATH}")

    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PRED_PATH}\n"
            "Please run 05_Generate_MP4_Heatmap.py first."
        )

    gt_data = np.load(NPZ_PATH, allow_pickle=True)
    pred_data = np.load(PRED_PATH, allow_pickle=True)

    print("\nPrediction keys:", pred_data.files)

    video_id = safe_video_id_to_str(str(pred_data["video_id"]))
    pred_frame_indices = pred_data["frame_indices"].astype(int)
    pred_keypoints = pred_data["pred_keypoints"].astype(np.float32)

    gt_video_ids = np.array([safe_video_id_to_str(v) for v in gt_data["video_ids"]])
    gt_frame_indices = gt_data["frame_indices"].astype(int)
    gt_keypoints = gt_data["keypoints"].astype(np.float32)
    gt_visibility = gt_data["visibility"].astype(np.float32)

    print(f"\nTarget video: {video_id}")
    print(f"Predicted frames: {len(pred_frame_indices)}")
    print(f"Prediction shape: {pred_keypoints.shape}")

    matched_preds = []
    matched_gts = []
    matched_viss = []
    matched_frame_indices = []

    for i, frame_idx in enumerate(pred_frame_indices):
        matches = np.where(
            (gt_video_ids == video_id)
            & (gt_frame_indices == frame_idx)
        )[0]

        if len(matches) == 0:
            continue

        gt_idx = matches[0]

        matched_preds.append(pred_keypoints[i])
        matched_gts.append(gt_keypoints[gt_idx])
        matched_viss.append(gt_visibility[gt_idx])
        matched_frame_indices.append(frame_idx)

    if len(matched_preds) == 0:
        raise RuntimeError("No matched frames found between predictions and ground truth.")

    preds = np.stack(matched_preds, axis=0).astype(np.float32)
    gts = np.stack(matched_gts, axis=0).astype(np.float32)
    viss = np.stack(matched_viss, axis=0).astype(np.float32)

    print(f"Matched frames: {len(preds)}")

    # ========================================================
    # 5. Compute metrics
    # ========================================================

    dist = euclidean_distance(preds, gts)
    mask = compute_visible_mask(viss, preds, gts)
    ref_lengths = estimate_reference_length(gts, viss)

    visible_count = int(mask.sum())
    total_count = int(mask.size)

    mse = compute_mse(preds, gts, mask)
    rmse = compute_rmse(preds, gts, mask)
    mae = compute_mae(preds, gts, mask)
    mean_pixel_error = compute_mean_pixel_error(dist, mask)
    mean_accel = compute_temporal_accel(preds, viss)

    summary = {
        "method": "ResNet18_Heatmap_Baseline",
        "video_id": video_id,
        "num_frames": int(preds.shape[0]),
        "num_keypoints": int(preds.shape[1]),
        "visible_keypoints": visible_count,
        "total_keypoints": total_count,
        "visibility_ratio_percent": visible_count / max(total_count, 1) * 100.0,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "mean_pixel_error": mean_pixel_error,
        "mean_accel": mean_accel,
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
        preds=preds,
        gts=gts,
        viss=viss,
        keypoint_names=JOINT_NAMES,
        pck_threshold_ratio=0.1,
    )

    # ========================================================
    # 6. Print results
    # ========================================================

    print("\n==============================")
    print("ResNet18 Heatmap Baseline Metrics")
    print("==============================")
    print(f"Video ID: {video_id}")
    print(f"Frames: {preds.shape[0]}")
    print(f"Keypoints: {preds.shape[1]}")
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
            f"PCK@0.1={r['PCK@0.1']:.2f}%"
        )

    # ========================================================
    # 7. Save CSV files
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

    print("\nSaved files:")
    print(f"  {SUMMARY_CSV}")
    print(f"  {PER_JOINT_CSV}")


if __name__ == "__main__":
    main()