# Scripts/10_export_penn_video_for_vitpose_test.py

from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NPZ_PATH = PROJECT_ROOT / "Datasets" / "Penn_Action" / "penn_action_processed.npz"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "10_ViTPose_Manual_Test_Video"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VIDEO_ID = "0054"

# 建议先导出短一点，方便上传到 HuggingFace demo
MAX_FRAMES = 80

FPS = 30

RAW_OUTPUT_VIDEO = OUTPUT_DIR / f"{TARGET_VIDEO_ID}_raw_no_skeleton.mp4"
H264_OUTPUT_VIDEO = OUTPUT_DIR / f"{TARGET_VIDEO_ID}_raw_no_skeleton_h264.mp4"


# ============================================================
# 2. Helper Functions
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

    candidates = [
        Path(p),
        PROJECT_ROOT / p,
        PROJECT_ROOT / "Datasets" / "Penn_Action" / p,
        PROJECT_ROOT / "Datasets" / "Penn_Action" / "frames" / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Image not found: {p_original}")


def make_h264_video(input_path, output_path):
    """
    Convert OpenCV mp4v video to browser-friendly H.264 mp4.
    Requires ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-r", str(FPS),
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"H.264 video saved: {output_path}")
    except FileNotFoundError:
        print("\nffmpeg not found. Skipped H.264 conversion.")
        print("You can install ffmpeg with:")
        print("  brew install ffmpeg")
    except subprocess.CalledProcessError:
        print("\nffmpeg conversion failed. Raw mp4 is still saved.")


# ============================================================
# 3. Main
# ============================================================

def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"NPZ path: {NPZ_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ not found: {NPZ_PATH}")

    data = np.load(NPZ_PATH, allow_pickle=True)

    image_paths = data["image_paths"]
    video_ids = np.array([safe_video_id_to_str(v) for v in data["video_ids"]])
    frame_indices = data["frame_indices"]

    indices = np.where(video_ids == TARGET_VIDEO_ID)[0]

    if len(indices) == 0:
        raise ValueError(f"Video not found: {TARGET_VIDEO_ID}")

    indices = indices[np.argsort(frame_indices[indices])]

    if MAX_FRAMES is not None:
        indices = indices[:MAX_FRAMES]

    print(f"\nExporting Penn Action raw video")
    print(f"Video ID: {TARGET_VIDEO_ID}")
    print(f"Frames: {len(indices)}")
    print("No skeleton / no keypoints will be drawn.")

    first_img_path = resolve_image_path(image_paths[indices[0]])
    first_img = Image.open(first_img_path).convert("RGB")
    width, height = first_img.size

    print(f"Video size: {width} x {height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(RAW_OUTPUT_VIDEO),
        fourcc,
        FPS,
        (width, height)
    )

    for i, idx in enumerate(indices):
        img_path = resolve_image_path(image_paths[idx])

        img = Image.open(img_path).convert("RGB")
        frame = np.array(img)

        # RGB -> BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 注意：这里没有画任何骨骼、关键点、文字
        writer.write(frame)

        if (i + 1) % 20 == 0:
            print(f"Written {i + 1}/{len(indices)} frames")

    writer.release()

    print("\nRaw video saved:")
    print(RAW_OUTPUT_VIDEO)

    print("\nConverting to H.264 for web demo...")
    make_h264_video(RAW_OUTPUT_VIDEO, H264_OUTPUT_VIDEO)

    print("\nFinished.")
    print("Upload this file to ViTPose demo:")
    print(H264_OUTPUT_VIDEO)


if __name__ == "__main__":
    main()