# Scripts/11_Test_External_Video_Heatmap.py

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 外部测试视频路径
INPUT_VIDEO_PATH = (
    PROJECT_ROOT
    / "external_simple_test"
    / "data"
    / "squat.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "11_Test_External_Video_Heatmap"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_DIR = OUTPUT_DIR / "frames"
FRAME_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO_PATH = OUTPUT_DIR / "external_heatmap_pose.mp4"
PREDICTION_NPZ_PATH = OUTPUT_DIR / "external_heatmap_predictions.npz"

# 服务器训练完后下载到本地的模型路径
CKPT_PATH_LOCAL = (
    PROJECT_ROOT
    / "server_outputs"
    / "04_ResNet_Heatmap_Baseline"
    / "best_ResNet_Heatmap_Baseline.pth"
)

# 本地 outputs 里的模型路径
CKPT_PATH_OUTPUTS = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "04_ResNet_Heatmap_Baseline"
    / "best_ResNet_Heatmap_Baseline.pth"
)

# None = 全部帧
MAX_FRAMES = None

# 1 = 每帧都取
FRAME_STRIDE = 1

OUTPUT_FPS = 20
SHOW_KEYPOINT_INDEX = False


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

        # Input: 224 x 224
        # ResNet feature: 7 x 7
        # Decoder output: 56 x 56
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

            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x


# ============================================================
# 5. Helper Functions
# ============================================================

def choose_checkpoint_path():
    if CKPT_PATH_LOCAL.exists():
        return CKPT_PATH_LOCAL

    if CKPT_PATH_OUTPUTS.exists():
        return CKPT_PATH_OUTPUTS

    raise FileNotFoundError(
        "Heatmap checkpoint not found.\n\n"
        f"Tried:\n"
        f"1. {CKPT_PATH_LOCAL}\n"
        f"2. {CKPT_PATH_OUTPUTS}\n\n"
        "Please download best_ResNet_Heatmap_Baseline.pth first."
    )


def clear_old_frames(frame_dir):
    for old_file in frame_dir.glob("*.jpg"):
        old_file.unlink()


def extract_frames_from_video(video_path, frame_dir):
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    clear_old_frames(frame_dir)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {video_path}")
    print(f"Original FPS: {original_fps}")
    print(f"Total frames in video: {total_video_frames}")

    saved_paths = []
    read_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if read_count % FRAME_STRIDE == 0:
            frame_path = frame_dir / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(frame_path)
            saved_count += 1

            if MAX_FRAMES is not None and saved_count >= MAX_FRAMES:
                break

        read_count += 1

    cap.release()

    print(f"Extracted frames: {len(saved_paths)}")
    print(f"Frame dir: {frame_dir}")

    return saved_paths, original_fps


def heatmaps_to_coords_original(heatmaps, orig_w, orig_h):
    """
    heatmaps: [K, H, W]

    return:
        coords: [K, 2] in original image coordinates
        scores: [K]
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


def draw_pose(frame_bgr, pred_kpts, frame_idx):
    out = frame_bgr.copy()

    # Draw skeleton lines
    for a, b in SKELETON:
        pt1 = tuple(pred_kpts[a].astype(int))
        pt2 = tuple(pred_kpts[b].astype(int))
        cv2.line(out, pt1, pt2, (0, 0, 255), 2)

    # Draw keypoints
    for j, p in enumerate(pred_kpts):
        x, y = int(p[0]), int(p[1])
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)

        if SHOW_KEYPOINT_INDEX:
            cv2.putText(
                out,
                str(j),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        out,
        f"External Video | Frame {frame_idx} | Heatmap Model",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def load_checkpoint_safely(ckpt_path, device):
    """
    PyTorch 2.6+ changed torch.load default behavior:
    weights_only=True by default.

    Our checkpoint was generated by our own training script and contains
    extra information such as image_size, heatmap_size, epoch, best_val_loss,
    so we explicitly use weights_only=False.
    """
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = [
        "model_state_dict",
        "image_size",
        "heatmap_size",
        "num_keypoints",
    ]

    for key in required_keys:
        if key not in ckpt:
            raise KeyError(
                f"Checkpoint is missing key: {key}\n"
                f"Checkpoint path: {ckpt_path}\n"
                f"Available keys: {list(ckpt.keys())}"
            )

    return ckpt


# ============================================================
# 6. Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Project root: {PROJECT_ROOT}")

    ckpt_path = choose_checkpoint_path()
    print(f"Checkpoint path: {ckpt_path}")

    # Step 1: Extract frames
    print("\nStep 1: Extracting frames...")
    frame_paths, original_fps = extract_frames_from_video(
        video_path=INPUT_VIDEO_PATH,
        frame_dir=FRAME_DIR,
    )

    if len(frame_paths) == 0:
        raise RuntimeError("No frames extracted from video.")

    # Step 2: Load model
    print("\nStep 2: Loading heatmap model...")

    ckpt = load_checkpoint_safely(
        ckpt_path=ckpt_path,
        device=DEVICE,
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

    # Step 3: Prepare video writer
    first_img = Image.open(frame_paths[0]).convert("RGB")
    orig_w, orig_h = first_img.size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        fourcc,
        OUTPUT_FPS,
        (orig_w, orig_h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {OUTPUT_VIDEO_PATH}")

    pred_all = []
    score_all = []

    # Step 4: Inference
    print("\nStep 3: Running heatmap inference...")

    with torch.no_grad():
        for i, frame_path in enumerate(frame_paths):
            pil_img = Image.open(frame_path).convert("RGB")
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

            drawn = draw_pose(
                frame_bgr=frame_bgr,
                pred_kpts=pred_kpts,
                frame_idx=i,
            )

            writer.write(drawn)

            pred_all.append(pred_kpts)
            score_all.append(pred_scores)

            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(frame_paths)} frames")

    writer.release()

    pred_all = np.stack(pred_all, axis=0)
    score_all = np.stack(score_all, axis=0)

    np.savez_compressed(
        PREDICTION_NPZ_PATH,
        input_video=str(INPUT_VIDEO_PATH),
        frame_paths=np.array([str(p) for p in frame_paths]),
        pred_keypoints=pred_all,
        pred_scores=score_all,
    )

    print("\nFinished.")
    print(f"Frames saved to:")
    print(FRAME_DIR)

    print(f"\nOutput video saved to:")
    print(OUTPUT_VIDEO_PATH)

    print(f"\nPredictions saved to:")
    print(PREDICTION_NPZ_PATH)


if __name__ == "__main__":
    main()