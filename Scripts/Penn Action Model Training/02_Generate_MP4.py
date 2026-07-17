# Scripts/Penn Action Model Training/02_Generate_MP4.py

from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

import cv2


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NPZ_PATH = PROJECT_ROOT / "Datasets" / "Penn_Action" / "penn_action_processed.npz"

# Use the best model downloaded from server
CKPT_PATH = (
    PROJECT_ROOT
    / "server_outputs"
    / "01_Resnet_with_ImageNet"
    / "best_Resnet_with_ImageNet.pth"
)

# Generate MP4 and frame comparisons directly under this folder
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "02_Generate_MP4"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Change this video id if needed
TARGET_VIDEO_ID = "0684"

FPS = 30

# Per-frame comparison image folder
FRAME_COMPARE_DIR = OUTPUT_DIR / f"{TARGET_VIDEO_ID}_frame_comparisons"
FRAME_COMPARE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1.1 Visualization Config
# ============================================================

SHOW_GT = True
SHOW_KEYPOINT_INDEX = False
SHOW_TITLE = True

PRED_POINT_RADIUS = 2
PRED_LINE_THICKNESS = 1

GT_POINT_RADIUS = 3
GT_LINE_THICKNESS = 1

TEXT_SCALE = 0.45
TEXT_THICKNESS = 1


# Penn Action 13-keypoint order:
# 0 head, 1 left shoulder, 2 right shoulder, 3 left elbow, 4 right elbow,
# 5 left wrist, 6 right wrist, 7 left hip, 8 right hip,
# 9 left knee, 10 right knee, 11 left ankle, 12 right ankle
SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8),
    (7, 8),
    (7, 9), (9, 11),
    (8, 10), (10, 12),
]


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


# ============================================================
# 3. Model
# ============================================================

class ResNetWithImageNetAllKeypointRegressor(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()

        self.num_keypoints = num_keypoints

        # During inference, do not download ImageNet weights again.
        # The trained weights are loaded from the checkpoint.
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
# 4. Helper Functions
# ============================================================

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


def safe_video_id_to_str(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8")

    x = str(x)

    if x.isdigit():
        return x.zfill(4)

    return x


def draw_keypoints(frame_bgr, pred_kpts, gt_kpts=None, visibility=None):
    """
    Red = prediction
    Green = ground truth
    """
    img = frame_bgr.copy()

    # ----------------------------
    # Draw predicted skeleton
    # ----------------------------
    for a, b in SKELETON:
        if a < len(pred_kpts) and b < len(pred_kpts):
            xa, ya = pred_kpts[a]
            xb, yb = pred_kpts[b]

            if (
                not np.isnan(xa)
                and not np.isnan(ya)
                and not np.isnan(xb)
                and not np.isnan(yb)
            ):
                cv2.line(
                    img,
                    (int(xa), int(ya)),
                    (int(xb), int(yb)),
                    (0, 0, 255),
                    PRED_LINE_THICKNESS
                )

    # ----------------------------
    # Draw predicted keypoints
    # ----------------------------
    for i, (x, y) in enumerate(pred_kpts):
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(
                img,
                (int(x), int(y)),
                PRED_POINT_RADIUS,
                (0, 0, 255),
                -1
            )

            if SHOW_KEYPOINT_INDEX:
                cv2.putText(
                    img,
                    str(i),
                    (int(x) + 3, int(y) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    TEXT_SCALE,
                    (0, 0, 255),
                    TEXT_THICKNESS,
                    cv2.LINE_AA
                )

    # ----------------------------
    # Draw ground truth
    # ----------------------------
    if SHOW_GT and gt_kpts is not None and visibility is not None:
        for a, b in SKELETON:
            if a < len(gt_kpts) and b < len(gt_kpts):
                if visibility[a] > 0 and visibility[b] > 0:
                    xa, ya = gt_kpts[a]
                    xb, yb = gt_kpts[b]

                    if (
                        not np.isnan(xa)
                        and not np.isnan(ya)
                        and not np.isnan(xb)
                        and not np.isnan(yb)
                    ):
                        cv2.line(
                            img,
                            (int(xa), int(ya)),
                            (int(xb), int(yb)),
                            (0, 255, 0),
                            GT_LINE_THICKNESS
                        )

        for i, (x, y) in enumerate(gt_kpts):
            if visibility[i] > 0:
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(
                        img,
                        (int(x), int(y)),
                        GT_POINT_RADIUS,
                        (0, 255, 0),
                        1
                    )

    # ----------------------------
    # Draw title
    # ----------------------------
    if SHOW_TITLE:
        cv2.putText(
            img,
            "Red: Prediction | Green: Ground Truth",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            (255, 255, 255),
            TEXT_THICKNESS,
            cv2.LINE_AA
        )

    return img


def compute_video_mse(pred_kpts, gt_kpts, visibility):
    """
    pred_kpts: [T, K, 2]
    gt_kpts: [T, K, 2]
    visibility: [T, K]
    """
    visibility = visibility.astype(np.float32)

    error = (pred_kpts - gt_kpts) ** 2
    error = error * visibility[:, :, None]

    denom = max(visibility.sum() * 2.0, 1.0)

    return error.sum() / denom


# ============================================================
# 5. Main
# ============================================================

def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ not found: {NPZ_PATH}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found:\n{CKPT_PATH}\n\n"
            f"Please make sure best_Resnet_with_ImageNet.pth is inside:\n"
            f"server_outputs/01_Resnet_with_ImageNet/"
        )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"NPZ path: {NPZ_PATH}")
    print(f"Checkpoint path: {CKPT_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Frame comparison dir: {FRAME_COMPARE_DIR}")

    data = np.load(NPZ_PATH, allow_pickle=True)

    print(f"\nLoaded npz: {NPZ_PATH}")
    print("Available keys:", data.files)

    image_paths = data["image_paths"]
    all_keypoints = data["keypoints"]
    all_visibility = data["visibility"]
    video_ids = data["video_ids"]
    frame_indices = data["frame_indices"]

    video_ids_str = np.array([safe_video_id_to_str(v) for v in video_ids])

    unique_videos = sorted(list(set(video_ids_str)))
    print(f"Total videos: {len(unique_videos)}")
    print(f"First 10 video ids: {unique_videos[:10]}")

    target_video_id = str(TARGET_VIDEO_ID).zfill(4)
    print(f"Target video id: {target_video_id}")

    indices = np.where(video_ids_str == target_video_id)[0]

    if len(indices) == 0:
        raise ValueError(
            f"No frames found for video id {target_video_id}.\n"
            f"Available examples: {unique_videos[:20]}"
        )

    order = np.argsort(frame_indices[indices])
    indices = indices[order]

    print(f"Frames in target video: {len(indices)}")

    print(f"\nLoading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    image_size = ckpt["image_size"]
    num_keypoints = ckpt["num_keypoints"]

    print(f"Checkpoint method: {ckpt.get('method', 'unknown')}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint image size: {image_size}")
    print(f"Number of keypoints: {num_keypoints}")
    print(f"Best Val MSE: {ckpt.get('best_val_mse', 'unknown')}")

    model = ResNetWithImageNetAllKeypointRegressor(
        num_keypoints=num_keypoints
    )

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

    pred_keypoints = []
    gt_keypoints = []
    vis_keypoints = []

    output_video_path = OUTPUT_DIR / f"{target_video_id}_Resnet_with_ImageNet_pred.mp4"
    video_writer = None

    print("\nStart inference, MP4 generation, and frame comparison export...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_path = resolve_image_path(image_paths[idx])

            pil_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            pred_resized = model(input_tensor)[0].detach().cpu().numpy()

            pred_original = pred_resized.copy()
            pred_original[:, 0] *= orig_w / image_size
            pred_original[:, 1] *= orig_h / image_size

            gt = all_keypoints[idx]
            vis = all_visibility[idx]

            pred_keypoints.append(pred_original)
            gt_keypoints.append(gt)
            vis_keypoints.append(vis)

            frame_bgr = cv2.imread(str(img_path))
            if frame_bgr is None:
                raise RuntimeError(f"cv2 failed to read image: {img_path}")

            drawn = draw_keypoints(
                frame_bgr=frame_bgr,
                pred_kpts=pred_original,
                gt_kpts=gt,
                visibility=vis
            )

            # Save per-frame comparison image
            frame_id = int(frame_indices[idx])
            frame_compare_path = FRAME_COMPARE_DIR / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(frame_compare_path), drawn)

            # Write frame to MP4
            if video_writer is None:
                h, w = drawn.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    FPS,
                    (w, h)
                )

            video_writer.write(drawn)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(indices)} frames")

    if video_writer is not None:
        video_writer.release()

    pred_keypoints = np.stack(pred_keypoints, axis=0)
    gt_keypoints = np.stack(gt_keypoints, axis=0)
    vis_keypoints = np.stack(vis_keypoints, axis=0)

    video_mse = compute_video_mse(pred_keypoints, gt_keypoints, vis_keypoints)

    print("\nGeneration finished.")
    print(f"Video MSE: {video_mse:.4f}")
    print("Saved MP4 to:")
    print(output_video_path)
    print("Saved frame comparisons to:")
    print(FRAME_COMPARE_DIR)
    print(f"Prediction shape: {pred_keypoints.shape}")


if __name__ == "__main__":
    main()