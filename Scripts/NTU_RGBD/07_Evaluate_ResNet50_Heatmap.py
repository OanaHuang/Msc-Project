# Scripts/NTU_RGBD/07_Generate_MP4_ResNet50_Heatmap.py

from __future__ import annotations

from pathlib import Path
import csv
import sys

import cv2
import numpy as np
import torch


# ============================================================
# 1. Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. Project imports
# ============================================================

from Scripts.NTU_RGBD.core import (
    read_skeleton_file,
    extract_primary_pose_sequence,
)

from Scripts.NTU_RGBD.models import (
    build_resnet50_heatmap,
)


# ============================================================
# 3. Paths
# ============================================================

TEST_CSV = (
    PROJECT_ROOT
    / "Datasets"
    / "NTU_RGBD"
    / "metadata"
    / "test_split.csv"
)

# Use the model downloaded from the server.
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "server_outputs"
    / "NTU_RGBD"
    / "06_Train_ResNet50_Heatmap"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "NTU_RGBD"
    / "07_Generate_MP4_ResNet50_Heatmap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. Configuration
# ============================================================

# None means use the first sample in test_split.csv.
# Example:
# TARGET_SAMPLE_ID = "S001C001P001R001A001"
TARGET_SAMPLE_ID = None

IMAGE_SIZE = 224
NUM_JOINTS = 25

# 1 means process every frame.
FRAME_STRIDE = 1

# None means process the complete video.
MAX_FRAMES = None

# None means preserve original video resolution.
OUTPUT_WIDTH = None

# None means preserve original video FPS.
OUTPUT_FPS = None

JOINT_RADIUS = 4
BONE_THICKNESS = 2

# OpenCV uses BGR.
GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)

# None means automatically select CUDA, MPS, or CPU.
DEVICE_NAME = None


# ============================================================
# 5. ImageNet normalization
# ============================================================

IMAGENET_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ============================================================
# 6. NTU 25-joint skeleton connections
# ============================================================

NTU_EDGES = (
    # Spine and head
    (0, 1),
    (1, 20),
    (20, 2),
    (2, 3),

    # Left arm
    (20, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 21),
    (7, 22),

    # Right arm
    (20, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 23),
    (11, 24),

    # Left leg
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),

    # Right leg
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
)


# ============================================================
# 7. Device
# ============================================================

def get_device() -> torch.device:
    if DEVICE_NAME is not None:
        return torch.device(DEVICE_NAME)

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# 8. Select sample
# ============================================================

def select_sample() -> dict[str, str]:
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {TEST_CSV}"
        )

    with TEST_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if not rows:
        raise RuntimeError(
            "test_split.csv contains no samples"
        )

    if TARGET_SAMPLE_ID is None:
        return rows[0]

    for row in rows:
        if (
            row.get("sample_id")
            == TARGET_SAMPLE_ID
        ):
            return row

    raise ValueError(
        f"Sample ID not found in test split: "
        f"{TARGET_SAMPLE_ID}"
    )


# ============================================================
# 9. Model input
# ============================================================

def prepare_image(
    frame: np.ndarray,
) -> torch.Tensor:
    if frame is None or frame.size == 0:
        raise ValueError(
            "Input frame is empty"
        )

    resized = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    image = (
        rgb.astype(np.float32)
        / 255.0
    )

    image = (
        image - IMAGENET_MEAN
    ) / IMAGENET_STD

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    tensor = torch.from_numpy(
        image
    ).float()

    return tensor.unsqueeze(0)


# ============================================================
# 10. Heatmap decoding
# ============================================================

def decode_heatmaps(
    heatmaps: torch.Tensor,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    """
    Convert heatmaps with shape [1, J, H, W]
    into original-video coordinates.
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            "heatmaps must have shape [B, J, H, W]"
        )

    if heatmaps.shape[0] != 1:
        raise ValueError(
            "This video script expects batch size 1"
        )

    heatmaps = heatmaps[0]

    num_joints = heatmaps.shape[0]
    heatmap_height = heatmaps.shape[1]
    heatmap_width = heatmaps.shape[2]

    flattened = heatmaps.reshape(
        num_joints,
        -1,
    )

    indices = torch.argmax(
        flattened,
        dim=1,
    )

    x = (
        indices % heatmap_width
    ).float()

    y = torch.div(
        indices,
        heatmap_width,
        rounding_mode="floor",
    ).float()

    x *= (
        original_width
        / heatmap_width
    )

    y *= (
        original_height
        / heatmap_height
    )

    keypoints = torch.stack(
        (x, y),
        dim=1,
    )

    return (
        keypoints
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# 11. Drawing
# ============================================================

def valid_point(
    point: np.ndarray,
    width: int,
    height: int,
) -> bool:
    point = np.asarray(point)

    if point.shape != (2,):
        return False

    x = float(point[0])
    y = float(point[1])

    return (
        np.isfinite(x)
        and np.isfinite(y)
        and 0 <= x < width
        and 0 <= y < height
    )


def draw_pose(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    visibility: np.ndarray | None = None,
) -> None:
    height, width = frame.shape[:2]

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if keypoints.shape != (
        NUM_JOINTS,
        2,
    ):
        raise ValueError(
            f"Expected keypoints shape "
            f"({NUM_JOINTS}, 2), "
            f"but got {keypoints.shape}"
        )

    if visibility is None:
        visibility = np.ones(
            NUM_JOINTS,
            dtype=bool,
        )
    else:
        visibility = np.asarray(
            visibility
        ).reshape(-1)

        if visibility.shape[0] != NUM_JOINTS:
            raise ValueError(
                f"Expected visibility length "
                f"{NUM_JOINTS}, "
                f"but got {visibility.shape[0]}"
            )

        visibility = visibility.astype(
            bool
        )

    # Draw bones.
    for start_index, end_index in NTU_EDGES:
        if not (
            visibility[start_index]
            and visibility[end_index]
        ):
            continue

        start = keypoints[start_index]
        end = keypoints[end_index]

        if not (
            valid_point(
                start,
                width,
                height,
            )
            and valid_point(
                end,
                width,
                height,
            )
        ):
            continue

        start_point = (
            int(round(start[0])),
            int(round(start[1])),
        )

        end_point = (
            int(round(end[0])),
            int(round(end[1])),
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            color,
            BONE_THICKNESS,
            cv2.LINE_AA,
        )

    # Draw joints.
    for joint_index, point in enumerate(
        keypoints
    ):
        if not visibility[joint_index]:
            continue

        if not valid_point(
            point,
            width,
            height,
        ):
            continue

        center = (
            int(round(point[0])),
            int(round(point[1])),
        )

        cv2.circle(
            frame,
            center,
            JOINT_RADIUS,
            color,
            -1,
            cv2.LINE_AA,
        )


def draw_legend(
    frame: np.ndarray,
    sample_id: str,
    frame_index: int,
) -> None:
    cv2.rectangle(
        frame,
        (10, 10),
        (335, 92),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        f"Sample: {sample_id}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Frame: {frame_index}",
        (20, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "GT",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GT_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Prediction",
        (75, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        PRED_COLOR,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# 12. Checkpoint
# ============================================================

def load_model(
    device: torch.device,
) -> torch.nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    model = build_resnet50_heatmap(
        num_joints=NUM_JOINTS,
        pretrained=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# 13. Main
# ============================================================

@torch.no_grad()
def main() -> None:
    sample = select_sample()

    sample_id = sample["sample_id"]

    rgb_path = Path(
        sample["rgb_path"]
    )

    skeleton_path = Path(
        sample["skeleton_path"]
    )

    if not rgb_path.exists():
        raise FileNotFoundError(
            f"RGB video not found: "
            f"{rgb_path}"
        )

    if not skeleton_path.exists():
        raise FileNotFoundError(
            f"Skeleton file not found: "
            f"{skeleton_path}"
        )

    device = get_device()

    print("=" * 70)
    print(
        "NTU RGB+D GT vs prediction "
        "video generation"
    )
    print("=" * 70)
    print(f"Sample ID:   {sample_id}")
    print(f"RGB video:   {rgb_path}")
    print(f"Skeleton:    {skeleton_path}")
    print(f"Checkpoint:  {CHECKPOINT_PATH}")
    print(f"Device:      {device}")

    model = load_model(
        device
    )

    skeleton_data = read_skeleton_file(
        skeleton_path
    )

    pose_sequence = (
        extract_primary_pose_sequence(
            skeleton_data
        )
    )

    if "color_xy" not in pose_sequence:
        raise KeyError(
            "pose_sequence does not contain "
            "'color_xy'"
        )

    if "tracking_state" not in pose_sequence:
        raise KeyError(
            "pose_sequence does not contain "
            "'tracking_state'"
        )

    gt_sequence = np.asarray(
        pose_sequence["color_xy"],
        dtype=np.float32,
    )

    tracking_state_sequence = np.asarray(
        pose_sequence["tracking_state"]
    )

    if gt_sequence.ndim != 3:
        raise ValueError(
            f"Expected GT sequence shape "
            f"[T, J, 2], got "
            f"{gt_sequence.shape}"
        )

    if tracking_state_sequence.ndim != 2:
        raise ValueError(
            f"Expected tracking state shape "
            f"[T, J], got "
            f"{tracking_state_sequence.shape}"
        )

    capture = cv2.VideoCapture(
        str(rgb_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{rgb_path}"
        )

    original_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    original_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    original_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    video_frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    usable_frames = min(
        video_frame_count,
        gt_sequence.shape[0],
        tracking_state_sequence.shape[0],
    )

    if MAX_FRAMES is not None:
        usable_frames = min(
            usable_frames,
            MAX_FRAMES,
        )

    if OUTPUT_WIDTH is None:
        output_width = original_width
        output_height = original_height
    else:
        output_width = int(
            OUTPUT_WIDTH
        )

        output_height = int(
            round(
                original_height
                * output_width
                / original_width
            )
        )

    output_fps = (
        original_fps
        if OUTPUT_FPS is None
        else float(OUTPUT_FPS)
    )

    if output_fps <= 0:
        output_fps = 30.0

    output_path = (
        OUTPUT_DIR
        / f"{sample_id}_gt_vs_prediction.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        output_fps / FRAME_STRIDE,
        (
            output_width,
            output_height,
        ),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    written_frames = 0

    try:
        for frame_index in range(
            usable_frames
        ):
            success, frame = capture.read()

            if not success:
                print(
                    f"Stopped early at frame "
                    f"{frame_index}: "
                    f"video read failed"
                )
                break

            if (
                frame_index
                % FRAME_STRIDE
                != 0
            ):
                continue

            model_input = prepare_image(
                frame
            ).to(
                device,
                non_blocking=True,
            )

            predicted_heatmaps = model(
                model_input
            )

            predicted_keypoints = (
                decode_heatmaps(
                    heatmaps=predicted_heatmaps,
                    original_width=original_width,
                    original_height=original_height,
                )
            )

            gt_keypoints = (
                gt_sequence[
                    frame_index
                ].copy()
            )

            # NTU tracking state:
            # 0 = not tracked
            # 1 = inferred
            # 2 = tracked
            gt_visibility = (
                tracking_state_sequence[
                    frame_index
                ] == 2
            )

            draw_pose(
                frame=frame,
                keypoints=gt_keypoints,
                color=GT_COLOR,
                visibility=gt_visibility,
            )

            draw_pose(
                frame=frame,
                keypoints=predicted_keypoints,
                color=PRED_COLOR,
                visibility=None,
            )

            draw_legend(
                frame=frame,
                sample_id=sample_id,
                frame_index=frame_index,
            )

            if (
                output_width
                != original_width
                or output_height
                != original_height
            ):
                frame = cv2.resize(
                    frame,
                    (
                        output_width,
                        output_height,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )

            writer.write(frame)

            written_frames += 1

            if written_frames % 50 == 0:
                print(
                    f"Written frames: "
                    f"{written_frames}"
                )

    finally:
        capture.release()
        writer.release()

    if written_frames == 0:
        raise RuntimeError(
            "No frames were written to the output video"
        )

    print()
    print("=" * 70)
    print("Video generation finished")
    print("=" * 70)
    print(
        f"Frames written: "
        f"{written_frames}"
    )
    print(
        f"Output video:   "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()