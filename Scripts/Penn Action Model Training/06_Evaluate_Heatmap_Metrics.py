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
    / "0011_heatmap_predictions.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "06_Evaluate_Heatmap_Metrics"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "heatmap_metrics_summary.csv"
PCK_BY_PART_CSV = OUTPUT_DIR / "heatmap_pck_by_part.csv"
PCK_BY_JOINT_CSV = OUTPUT_DIR / "heatmap_pck_by_joint.csv"

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
# 3. Helper Functions
# ============================================================

def safe_video_id_to_str(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8")

    x = str(x)

    if x.isdigit():
        return x.zfill(4)

    return x


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


def compute_mean_pixel_error(preds, gts, viss):
    total_dist = 0.0
    total_points = 0

    for t in range(len(preds)):
        visible = viss[t] > 0

        if visible.sum() == 0:
            continue

        dist = np.linalg.norm(preds[t, visible] - gts[t, visible], axis=1)

        total_dist += dist.sum()
        total_points += len(dist)

    if total_points == 0:
        return np.nan

    return total_dist / total_points


def compute_pck(preds, gts, viss):
    num_keypoints = preds.shape[1]

    correct = np.zeros(num_keypoints, dtype=np.float64)
    total = np.zeros(num_keypoints, dtype=np.float64)

    for t in range(len(preds)):
        scale = compute_person_scale(gts[t], viss[t])

        if scale is None:
            continue

        threshold = PCK_THRESHOLD * scale

        for j in range(num_keypoints):
            if viss[t, j] <= 0:
                continue

            dist = np.linalg.norm(preds[t, j] - gts[t, j])

            total[j] += 1

            if dist < threshold:
                correct[j] += 1

    joint_pck = {}

    for j, name in enumerate(JOINT_NAMES):
        if total[j] > 0:
            joint_pck[name] = correct[j] / total[j] * 100.0
        else:
            joint_pck[name] = np.nan

    part_pck = {}

    all_correct = 0.0
    all_total = 0.0

    for part_name, joint_ids in PART_GROUPS.items():
        part_correct = correct[joint_ids].sum()
        part_total = total[joint_ids].sum()

        if part_total > 0:
            part_pck[part_name] = part_correct / part_total * 100.0
        else:
            part_pck[part_name] = np.nan

        all_correct += part_correct
        all_total += part_total

    if all_total > 0:
        part_pck["Mean"] = all_correct / all_total * 100.0
    else:
        part_pck["Mean"] = np.nan

    return joint_pck, part_pck


def compute_accel(preds, viss):
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

    return total_accel / total_count


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

    video_id = safe_video_id_to_str(str(pred_data["video_id"]))
    pred_frame_indices = pred_data["frame_indices"].astype(int)
    pred_keypoints = pred_data["pred_keypoints"].astype(np.float32)

    gt_video_ids = np.array([safe_video_id_to_str(v) for v in gt_data["video_ids"]])
    gt_frame_indices = gt_data["frame_indices"].astype(int)
    gt_keypoints = gt_data["keypoints"].astype(np.float32)
    gt_visibility = gt_data["visibility"].astype(np.float32)

    print(f"\nTarget video: {video_id}")
    print(f"Predicted frames: {len(pred_frame_indices)}")

    matched_preds = []
    matched_gts = []
    matched_viss = []

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

    if len(matched_preds) == 0:
        raise RuntimeError("No matched frames found between predictions and ground truth.")

    preds = np.stack(matched_preds, axis=0)
    gts = np.stack(matched_gts, axis=0)
    viss = np.stack(matched_viss, axis=0)

    print(f"Matched frames: {len(preds)}")

    mean_pixel_error = compute_mean_pixel_error(preds, gts, viss)
    joint_pck, part_pck = compute_pck(preds, gts, viss)
    mean_accel = compute_accel(preds, viss)

    summary = {
        "method": "ResNet18_Heatmap_Baseline",
        "video_id": video_id,
        "num_frames": len(preds),
        "mean_pixel_error": mean_pixel_error,
        "mean_pck_0.2_percent": part_pck["Mean"],
        "mean_accel": mean_accel,
    }

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    part_row = {
        "method": "ResNet18_Heatmap_Baseline",
        "video_id": video_id,
        "num_frames": len(preds),
        "Head_percent": part_pck["Head"],
        "Sho_percent": part_pck["Sho"],
        "Elb_percent": part_pck["Elb"],
        "Wri_percent": part_pck["Wri"],
        "Hip_percent": part_pck["Hip"],
        "Knee_percent": part_pck["Knee"],
        "Ank_percent": part_pck["Ank"],
        "Mean_percent": part_pck["Mean"],
    }

    with open(PCK_BY_PART_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(part_row.keys()))
        writer.writeheader()
        writer.writerow(part_row)

    joint_row = {
        "method": "ResNet18_Heatmap_Baseline",
        "video_id": video_id,
        "num_frames": len(preds),
    }

    for name in JOINT_NAMES:
        joint_row[f"{name}_percent"] = joint_pck[name]

    with open(PCK_BY_JOINT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(joint_row.keys()))
        writer.writeheader()
        writer.writerow(joint_row)

    print("\nEvaluation finished.")

    print("\nSummary:")
    print(f"Mean Pixel Error: {mean_pixel_error:.2f}")
    print(f"PCK@0.2: {part_pck['Mean']:.2f}%")
    print(f"Accel: {mean_accel:.2f}")

    print("\nPCK by part:")
    print(
        f"Head={part_pck['Head']:.2f}%, "
        f"Sho={part_pck['Sho']:.2f}%, "
        f"Elb={part_pck['Elb']:.2f}%, "
        f"Wri={part_pck['Wri']:.2f}%, "
        f"Hip={part_pck['Hip']:.2f}%, "
        f"Knee={part_pck['Knee']:.2f}%, "
        f"Ank={part_pck['Ank']:.2f}%, "
        f"Mean={part_pck['Mean']:.2f}%"
    )

    print("\nSaved files:")
    print(SUMMARY_CSV)
    print(PCK_BY_PART_CSV)
    print(PCK_BY_JOINT_CSV)


if __name__ == "__main__":
    main()