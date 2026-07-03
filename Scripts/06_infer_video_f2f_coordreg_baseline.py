# Scripts/06_infer_video_f2f_coordreg_baseline.py

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NPZ_PATH = PROJECT_ROOT / "Datasets" / "Penn_Action" / "penn_action_processed.npz"

CKPT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "f2f_coordreg_v1"
    / "best_f2f_coordreg_resnet18.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "f2f_coordreg_v1"
PRED_DIR = OUTPUT_DIR / "predictions"
VIDEO_DIR = OUTPUT_DIR / "videos"

PRED_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VIDEO_ID = "0077"
FPS = 60


# ============================================================
# 2. Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
print(f"Using device: {DEVICE}")


# ============================================================
# 3. Model
# ============================================================

class ResNet18HeadRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 1, 2)
        return x


# ============================================================
# 4. Helpers
# ============================================================

def resolve_image_path(p):
    p = str(p).strip()
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
        f"Original path: {p}\n"
        f"Tried:\n" + "\n".join([f"  {c}" for c in candidates])
    )


def safe_video_id_to_str(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8")

    x = str(x)

    if x.isdigit():
        return x.zfill(4)

    return x


def draw_head(frame_bgr, pred_head, gt_head=None, visible=True):
    """
    Red solid dot = predicted head
    Green circle = ground truth head
    """
    img = frame_bgr.copy()

    px, py = pred_head

    if not np.isnan(px) and not np.isnan(py):
        cv2.circle(img, (int(px), int(py)), 8, (0, 0, 255), -1)
        cv2.putText(
            img,
            "Pred",
            (int(px) + 8, int(py) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    if gt_head is not None and visible:
        gx, gy = gt_head

        if not np.isnan(gx) and not np.isnan(gy):
            cv2.circle(img, (int(gx), int(gy)), 8, (0, 255, 0), 2)
            cv2.putText(
                img,
                "GT",
                (int(gx) + 8, int(gy) + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    return img


def compute_video_mse(pred_heads, gt_heads, visibility):
    """
    pred_heads: [T, 2]
    gt_heads: [T, 2]
    visibility: [T]
    """
    visibility = visibility.astype(np.float32)

    error = (pred_heads - gt_heads) ** 2   # [T, 2]
    error = error * visibility[:, None]

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
            f"Checkpoint not found: {CKPT_PATH}\n"
            f"Please run Scripts/05_train_f2f_coordreg_baseline.py first."
        )

    data = np.load(NPZ_PATH, allow_pickle=True)

    print(f"Loaded npz: {NPZ_PATH}")
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

    # Sort frames by frame index
    order = np.argsort(frame_indices[indices])
    indices = indices[order]

    print(f"Frames in target video: {len(indices)}")

    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    image_size = ckpt["image_size"]
    head_index = ckpt.get("head_index", 0)

    print(f"Checkpoint method: {ckpt.get('method', 'unknown')}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint image size: {image_size}")
    print(f"Best Val MSE: {ckpt.get('best_val_mse', 'unknown')}")
    print(f"Head index: {head_index}")

    model = ResNet18HeadRegressor()
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

    pred_heads = []
    gt_heads = []
    vis_heads = []
    frame_paths = []
    used_frame_indices = []

    output_video_path = VIDEO_DIR / f"{target_video_id}_baseline_pred.mp4"
    video_writer = None

    print("\nStart inference...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_path = resolve_image_path(image_paths[idx])

            pil_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            pred_resized = model(input_tensor)[0, 0].detach().cpu().numpy()

            # Convert coordinate back to original image size
            pred_original = pred_resized.copy()
            pred_original[0] *= orig_w / image_size
            pred_original[1] *= orig_h / image_size

            gt_head = all_keypoints[idx, head_index, :]
            vis_head = all_visibility[idx, head_index]

            pred_heads.append(pred_original)
            gt_heads.append(gt_head)
            vis_heads.append(vis_head)
            frame_paths.append(str(img_path))
            used_frame_indices.append(int(frame_indices[idx]))

            frame_bgr = cv2.imread(str(img_path))
            if frame_bgr is None:
                raise RuntimeError(f"cv2 failed to read image: {img_path}")

            drawn = draw_head(
                frame_bgr=frame_bgr,
                pred_head=pred_original,
                gt_head=gt_head,
                visible=bool(vis_head > 0)
            )

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

    pred_heads = np.stack(pred_heads, axis=0)
    gt_heads = np.stack(gt_heads, axis=0)
    vis_heads = np.array(vis_heads)

    video_mse = compute_video_mse(pred_heads, gt_heads, vis_heads)

    output_pred_path = PRED_DIR / f"{target_video_id}_baseline_predictions.npz"

    np.savez_compressed(
        output_pred_path,
        video_id=target_video_id,
        frame_paths=np.array(frame_paths),
        frame_indices=np.array(used_frame_indices),
        pred_head=pred_heads,
        gt_head=gt_heads,
        visibility=vis_heads,
        video_mse=video_mse,
        checkpoint=str(CKPT_PATH),
        image_size=image_size,
        head_index=head_index,
        method="Head-only Frame-by-frame Coordinate Regression"
    )

    print("\nInference finished.")
    print(f"Video MSE: {video_mse:.4f}")
    print(f"Saved predictions to:")
    print(output_pred_path)
    print(f"Saved prediction video to:")
    print(output_video_path)
    print(f"Prediction shape: {pred_heads.shape}")


if __name__ == "__main__":
    main()