# Scripts/Penn Action Model Training/05_Generate_MP4_Heatmap.py

from pathlib import Path
import json

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

# 优先使用服务器训练输出
CKPT_PATH_SERVER = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "04_ResNet_Heatmap_Baseline"
    / "best_ResNet_Heatmap_Baseline.pth"
)

# 如果你把模型下载回本地，也支持这个路径
CKPT_PATH_LOCAL = (
    PROJECT_ROOT
    / "server_outputs"
    / "04_ResNet_Heatmap_Baseline"
    / "best_ResNet_Heatmap_Baseline.pth"
)

# video-level split json，优先找服务器 outputs，再找本地 server_outputs
SPLIT_JSON_SERVER = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "04_ResNet_Heatmap_Baseline"
    / "video_level_split.json"
)

SPLIT_JSON_LOCAL = (
    PROJECT_ROOT
    / "server_outputs"
    / "04_ResNet_Heatmap_Baseline"
    / "video_level_split.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "05_Generate_MP4_Heatmap"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 从 video_level_split.json 的 test_videos 里选一个
TARGET_VIDEO_ID = "0684"

FPS = 30

SHOW_GT = True
SHOW_KEYPOINT_INDEX = False
SHOW_TITLE = True

PRED_POINT_RADIUS = 3
PRED_LINE_THICKNESS = 2

GT_POINT_RADIUS = 2
GT_LINE_THICKNESS = 1

TEXT_SCALE = 0.45
TEXT_THICKNESS = 1


# ============================================================
# 2. Penn Action Skeleton
# ============================================================

# Penn Action 13 keypoints:
# 0 head
# 1 left shoulder
# 2 right shoulder
# 3 left elbow
# 4 right elbow
# 5 left wrist
# 6 right wrist
# 7 left hip
# 8 right hip
# 9 left knee
# 10 right knee
# 11 left ankle
# 12 right ankle

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

class ResNet18HeatmapBaseline(nn.Module):
    def __init__(self, num_keypoints=13):
        super().__init__()

        resnet = models.resnet18(weights=None)

        # Remove avgpool and fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # 224 input -> [B, 512, 7, 7]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_keypoints, kernel_size=1)
        )
        # 7 -> 14 -> 28 -> 56

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
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
    )


def choose_checkpoint_path():
    if CKPT_PATH_SERVER.exists():
        return CKPT_PATH_SERVER

    if CKPT_PATH_LOCAL.exists():
        return CKPT_PATH_LOCAL

    raise FileNotFoundError(
        "Heatmap checkpoint not found.\n\n"
        f"Tried:\n"
        f"1. {CKPT_PATH_SERVER}\n"
        f"2. {CKPT_PATH_LOCAL}\n\n"
        "Make sure 04_ResNet_Heatmap_Baseline.py has finished training."
    )


def choose_split_json_path():
    if SPLIT_JSON_SERVER.exists():
        return SPLIT_JSON_SERVER

    if SPLIT_JSON_LOCAL.exists():
        return SPLIT_JSON_LOCAL

    return None


def check_video_split(video_id):
    split_path = choose_split_json_path()

    if split_path is None:
        print("\nSplit JSON not found. Skipping split check.")
        return

    with open(split_path, "r") as f:
        split = json.load(f)

    train_videos = set(split.get("train_videos", []))
    val_videos = set(split.get("val_videos", []))
    test_videos = set(split.get("test_videos", []))

    print(f"\nSplit JSON path: {split_path}")

    if video_id in train_videos:
        print(f"WARNING: Video {video_id} is in TRAIN set.")
        print("This is not suitable for final unseen-video testing.")
    elif video_id in val_videos:
        print(f"Video {video_id} is in VAL set.")
        print("This is okay for validation visualisation, but not final test.")
    elif video_id in test_videos:
        print(f"Video {video_id} is in TEST set.")
        print("This is suitable for unseen-video testing.")
    else:
        print(f"Video {video_id} is not found in train/val/test split lists.")


def heatmaps_to_coords_original(heatmaps, orig_w, orig_h):
    """
    heatmaps: [K, H, W]
    return coords in original image coordinate system: [K, 2]
    """
    num_keypoints, heatmap_h, heatmap_w = heatmaps.shape

    coords = np.zeros((num_keypoints, 2), dtype=np.float32)
    scores = np.zeros((num_keypoints,), dtype=np.float32)

    for k in range(num_keypoints):
        hm = heatmaps[k]

        y, x = np.unravel_index(np.argmax(hm), hm.shape)

        score = hm[y, x]

        coords[k, 0] = x / heatmap_w * orig_w
        coords[k, 1] = y / heatmap_h * orig_h
        scores[k] = score

    return coords, scores


def draw_pose(
    frame,
    pred_kpts,
    pred_scores=None,
    gt_kpts=None,
    gt_vis=None,
    title=None,
):
    """
    frame: BGR image
    pred_kpts: [13, 2]
    gt_kpts: [13, 2]
    """

    out = frame.copy()

    # Draw GT skeleton first
    if SHOW_GT and gt_kpts is not None and gt_vis is not None:
        for a, b in SKELETON:
            if gt_vis[a] > 0 and gt_vis[b] > 0:
                pt1 = tuple(gt_kpts[a].astype(int))
                pt2 = tuple(gt_kpts[b].astype(int))
                cv2.line(out, pt1, pt2, (0, 255, 0), GT_LINE_THICKNESS)

        for j, p in enumerate(gt_kpts):
            if gt_vis[j] > 0:
                x, y = int(p[0]), int(p[1])
                cv2.circle(out, (x, y), GT_POINT_RADIUS, (0, 255, 0), -1)

    # Draw prediction skeleton
    for a, b in SKELETON:
        pt1 = tuple(pred_kpts[a].astype(int))
        pt2 = tuple(pred_kpts[b].astype(int))
        cv2.line(out, pt1, pt2, (0, 0, 255), PRED_LINE_THICKNESS)

    for j, p in enumerate(pred_kpts):
        x, y = int(p[0]), int(p[1])
        cv2.circle(out, (x, y), PRED_POINT_RADIUS, (0, 0, 255), -1)

        if SHOW_KEYPOINT_INDEX:
            cv2.putText(
                out,
                str(j),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                (255, 255, 255),
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    if SHOW_TITLE and title is not None:
        cv2.putText(
            out,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


# ============================================================
# 6. Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"NPZ path: {NPZ_PATH}")

    ckpt_path = choose_checkpoint_path()
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Output dir: {OUTPUT_DIR}")

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ not found: {NPZ_PATH}")

    data = np.load(NPZ_PATH, allow_pickle=True)

    image_paths = data["image_paths"]
    keypoints = data["keypoints"].astype(np.float32)
    visibility = data["visibility"].astype(np.float32)
    video_ids = np.array([safe_video_id_to_str(v) for v in data["video_ids"]])
    frame_indices = data["frame_indices"]

    target_video_id = safe_video_id_to_str(TARGET_VIDEO_ID)

    check_video_split(target_video_id)

    indices = np.where(video_ids == target_video_id)[0]

    if len(indices) == 0:
        raise ValueError(f"Video not found: {target_video_id}")

    indices = indices[np.argsort(frame_indices[indices])]

    print(f"\nTarget video: {target_video_id}")
    print(f"Number of frames: {len(indices)}")

    print("\nLoading checkpoint...")

    # Important for newer PyTorch versions:
    # This checkpoint is created by our own training script and contains
    # model_state_dict, optimizer_state_dict, split lists and numpy scalars.
    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE,
        weights_only=False,
    )

    image_size = ckpt["image_size"]
    heatmap_size = ckpt["heatmap_size"]
    num_keypoints = ckpt["num_keypoints"]

    print(f"Checkpoint method: {ckpt.get('method', 'unknown')}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Best val loss: {ckpt.get('best_val_loss', 'unknown')}")
    print(f"Image size: {image_size}")
    print(f"Heatmap size: {heatmap_size}")
    print(f"Num keypoints: {num_keypoints}")

    model = ResNet18HeatmapBaseline(num_keypoints=num_keypoints)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Prepare first frame
    first_img_path = resolve_image_path(image_paths[indices[0]])
    first_img = Image.open(first_img_path).convert("RGB")
    orig_w, orig_h = first_img.size

    output_video_path = OUTPUT_DIR / f"{target_video_id}_heatmap_pose.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        FPS,
        (orig_w, orig_h)
    )

    pred_all = []
    score_all = []
    frame_id_all = []

    print("\nRunning heatmap inference...")

    with torch.no_grad():
        for count, idx in enumerate(indices):
            img_path = resolve_image_path(image_paths[idx])

            pil_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            x = transform(pil_img).unsqueeze(0).to(DEVICE)

            pred_heatmaps = model(x)[0].detach().cpu().numpy()

            pred_kpts, pred_scores = heatmaps_to_coords_original(
                pred_heatmaps,
                orig_w=orig_w,
                orig_h=orig_h,
            )

            frame_rgb = np.array(pil_img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            title = f"Video {target_video_id} | Frame {int(frame_indices[idx])} | Heatmap"

            drawn = draw_pose(
                frame=frame_bgr,
                pred_kpts=pred_kpts,
                pred_scores=pred_scores,
                gt_kpts=keypoints[idx],
                gt_vis=visibility[idx],
                title=title,
            )

            writer.write(drawn)

            pred_all.append(pred_kpts)
            score_all.append(pred_scores)
            frame_id_all.append(int(frame_indices[idx]))

            if (count + 1) % 20 == 0:
                print(f"Processed {count + 1}/{len(indices)} frames")

    writer.release()

    pred_all = np.stack(pred_all, axis=0)
    score_all = np.stack(score_all, axis=0)
    frame_id_all = np.array(frame_id_all)

    pred_path = OUTPUT_DIR / f"{target_video_id}_heatmap_predictions.npz"

    np.savez_compressed(
        pred_path,
        video_id=target_video_id,
        frame_indices=frame_id_all,
        pred_keypoints=pred_all,
        pred_scores=score_all,
    )

    print("\nFinished.")
    print(f"Output video saved to:\n{output_video_path}")
    print(f"Predictions saved to:\n{pred_path}")


if __name__ == "__main__":
    main()