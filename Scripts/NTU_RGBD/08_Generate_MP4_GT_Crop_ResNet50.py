# Scripts/NTU_RGBD/08_Generate_MP4_GT_Crop_ResNet50.py

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
    / "08_Generate_MP4_GT_Crop_ResNet50"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. Configuration
# ============================================================

# None means use the first valid test sample.
TARGET_SAMPLE_ID = None

IMAGE_SIZE = 224
NUM_JOINTS = 25

FRAME_STRIDE = 1
MAX_FRAMES = None

OUTPUT_WIDTH = None
OUTPUT_FPS = None

# Bounding-box configuration.
BBOX_PADDING = 0.20
MIN_BBOX_SIZE = 64
MIN_TRACKED_JOINTS = 5

# Smooth only the crop box, not the predicted skeleton.
# Higher means more weight on the current box.
BBOX_SMOOTHING_ALPHA = 0.25

JOINT_RADIUS = 4
BONE_THICKNESS = 2
BBOX_THICKNESS = 2

# OpenCV uses BGR.
GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)
BBOX_COLOR = (0, 255, 255)

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
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(
            "test_split.csv contains no samples"
        )

    if TARGET_SAMPLE_ID is None:
        return rows[0]

    for row in rows:
        if row.get("sample_id") == TARGET_SAMPLE_ID:
            return row

    raise ValueError(
        f"Sample ID not found in test split: "
        f"{TARGET_SAMPLE_ID}"
    )


# ============================================================
# 9. Bounding-box utilities
# ============================================================

def clamp_value(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(value, maximum),
    )


def calculate_gt_bbox(
    keypoints: np.ndarray,
    tracking_state: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    """
    Calculate a square bounding box from tracked GT joints.

    Returns:
        [x1, y1, x2, y2] as float32, or None when the frame
        does not contain enough tracked joints.
    """
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    tracking_state = np.asarray(
        tracking_state
    ).reshape(-1)

    if keypoints.shape != (NUM_JOINTS, 2):
        raise ValueError(
            f"Expected keypoints shape "
            f"({NUM_JOINTS}, 2), "
            f"but got {keypoints.shape}"
        )

    if tracking_state.shape[0] != NUM_JOINTS:
        raise ValueError(
            f"Expected {NUM_JOINTS} tracking states, "
            f"but got {tracking_state.shape[0]}"
        )

    visible = tracking_state == 2

    finite = np.isfinite(
        keypoints
    ).all(axis=1)

    inside_image = (
        (keypoints[:, 0] >= 0)
        & (keypoints[:, 0] < image_width)
        & (keypoints[:, 1] >= 0)
        & (keypoints[:, 1] < image_height)
    )

    valid = (
        visible
        & finite
        & inside_image
    )

    if int(valid.sum()) < MIN_TRACKED_JOINTS:
        return None

    valid_points = keypoints[valid]

    x_min = float(
        valid_points[:, 0].min()
    )

    x_max = float(
        valid_points[:, 0].max()
    )

    y_min = float(
        valid_points[:, 1].min()
    )

    y_max = float(
        valid_points[:, 1].max()
    )

    person_width = max(
        x_max - x_min,
        1.0,
    )

    person_height = max(
        y_max - y_min,
        1.0,
    )

    # Apply padding before making the crop square.
    padded_width = (
        person_width
        * (1.0 + 2.0 * BBOX_PADDING)
    )

    padded_height = (
        person_height
        * (1.0 + 2.0 * BBOX_PADDING)
    )

    side_length = max(
        padded_width,
        padded_height,
        float(MIN_BBOX_SIZE),
    )

    center_x = (
        x_min + x_max
    ) / 2.0

    center_y = (
        y_min + y_max
    ) / 2.0

    x1 = center_x - side_length / 2.0
    y1 = center_y - side_length / 2.0
    x2 = center_x + side_length / 2.0
    y2 = center_y + side_length / 2.0

    # Shift the box when it extends outside the frame.
    if x1 < 0:
        x2 -= x1
        x1 = 0.0

    if y1 < 0:
        y2 -= y1
        y1 = 0.0

    if x2 > image_width:
        shift = x2 - image_width
        x1 -= shift
        x2 = float(image_width)

    if y2 > image_height:
        shift = y2 - image_height
        y1 -= shift
        y2 = float(image_height)

    x1 = clamp_value(
        x1,
        0.0,
        float(image_width - 1),
    )

    y1 = clamp_value(
        y1,
        0.0,
        float(image_height - 1),
    )

    x2 = clamp_value(
        x2,
        x1 + 1.0,
        float(image_width),
    )

    y2 = clamp_value(
        y2,
        y1 + 1.0,
        float(image_height),
    )

    return np.asarray(
        [x1, y1, x2, y2],
        dtype=np.float32,
    )


def smooth_bbox(
    current_bbox: np.ndarray,
    previous_bbox: np.ndarray | None,
) -> np.ndarray:
    if previous_bbox is None:
        return current_bbox.copy()

    alpha = float(
        BBOX_SMOOTHING_ALPHA
    )

    return (
        alpha * current_bbox
        + (1.0 - alpha) * previous_bbox
    ).astype(np.float32)


def bbox_to_integer(
    bbox: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    x1_i = int(
        np.floor(x1)
    )

    y1_i = int(
        np.floor(y1)
    )

    x2_i = int(
        np.ceil(x2)
    )

    y2_i = int(
        np.ceil(y2)
    )

    x1_i = max(
        0,
        min(x1_i, image_width - 1),
    )

    y1_i = max(
        0,
        min(y1_i, image_height - 1),
    )

    x2_i = max(
        x1_i + 1,
        min(x2_i, image_width),
    )

    y2_i = max(
        y1_i + 1,
        min(y2_i, image_height),
    )

    return (
        x1_i,
        y1_i,
        x2_i,
        y2_i,
    )


# ============================================================
# 10. Model input
# ============================================================

def prepare_crop(
    frame: np.ndarray,
    bbox: np.ndarray,
) -> tuple[
    torch.Tensor,
    tuple[int, int, int, int],
]:
    """
    Crop the person region and prepare a normalized model input.
    """
    image_height, image_width = frame.shape[:2]

    x1, y1, x2, y2 = bbox_to_integer(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
    )

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        raise RuntimeError(
            f"Empty crop generated from bbox: "
            f"{x1, y1, x2, y2}"
        )

    resized = cv2.resize(
        crop,
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

    return (
        tensor.unsqueeze(0),
        (x1, y1, x2, y2),
    )


# ============================================================
# 11. Heatmap decoding
# ============================================================

def decode_heatmaps_to_crop(
    heatmaps: torch.Tensor,
    crop_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Decode heatmaps and map predictions directly back to the
    original-video coordinate system.
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            "heatmaps must have shape [B, J, H, W]"
        )

    if heatmaps.shape[0] != 1:
        raise ValueError(
            "This script expects batch size 1"
        )

    heatmaps = heatmaps[0]

    num_joints = heatmaps.shape[0]
    heatmap_height = heatmaps.shape[1]
    heatmap_width = heatmaps.shape[2]

    if num_joints != NUM_JOINTS:
        raise ValueError(
            f"Expected {NUM_JOINTS} joints, "
            f"but model returned {num_joints}"
        )

    flattened = heatmaps.reshape(
        num_joints,
        -1,
    )

    indices = torch.argmax(
        flattened,
        dim=1,
    )

    heatmap_x = (
        indices % heatmap_width
    ).float()

    heatmap_y = torch.div(
        indices,
        heatmap_width,
        rounding_mode="floor",
    ).float()

    x1, y1, x2, y2 = crop_bbox

    crop_width = float(
        x2 - x1
    )

    crop_height = float(
        y2 - y1
    )

    # Use the centre of the selected heatmap cell.
    crop_x = (
        heatmap_x + 0.5
    ) * (
        crop_width / heatmap_width
    )

    crop_y = (
        heatmap_y + 0.5
    ) * (
        crop_height / heatmap_height
    )

    original_x = crop_x + float(x1)
    original_y = crop_y + float(y1)

    keypoints = torch.stack(
        (
            original_x,
            original_y,
        ),
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
# 12. Drawing
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
        ).reshape(-1).astype(bool)

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

        cv2.line(
            frame,
            (
                int(round(start[0])),
                int(round(start[1])),
            ),
            (
                int(round(end[0])),
                int(round(end[1])),
            ),
            color,
            BONE_THICKNESS,
            cv2.LINE_AA,
        )

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

        cv2.circle(
            frame,
            (
                int(round(point[0])),
                int(round(point[1])),
            ),
            JOINT_RADIUS,
            color,
            -1,
            cv2.LINE_AA,
        )


def draw_bbox(
    frame: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = crop_bbox

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2 - 1, y2 - 1),
        BBOX_COLOR,
        BBOX_THICKNESS,
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
        (410, 112),
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
        (20, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "GT",
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GT_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Prediction",
        (72, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        PRED_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "GT crop box",
        (190, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BBOX_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "B1 diagnostic: full-frame trained model",
        (20, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# 13. Checkpoint
# ============================================================

def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    return cleaned_state_dict


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
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]
    else:
        state_dict = checkpoint

    state_dict = remove_module_prefix(
        state_dict
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# 14. Main
# ============================================================

@torch.no_grad()
def main() -> None:
    sample = select_sample()

    sample_id = sample["sample_id"]
    rgb_path = Path(sample["rgb_path"])
    skeleton_path = Path(
        sample["skeleton_path"]
    )

    if not rgb_path.exists():
        raise FileNotFoundError(
            f"RGB video not found: {rgb_path}"
        )

    if not skeleton_path.exists():
        raise FileNotFoundError(
            f"Skeleton file not found: "
            f"{skeleton_path}"
        )

    device = get_device()

    print("=" * 72)
    print(
        "NTU RGB+D B1 GT-crop diagnostic video"
    )
    print("=" * 72)
    print(f"Sample ID:    {sample_id}")
    print(f"RGB video:    {rgb_path}")
    print(f"Skeleton:     {skeleton_path}")
    print(f"Checkpoint:   {CHECKPOINT_PATH}")
    print(f"Device:       {device}")
    print(f"Padding:      {BBOX_PADDING}")
    print(
        f"Box smoothing alpha: "
        f"{BBOX_SMOOTHING_ALPHA}"
    )

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

    gt_sequence = np.asarray(
        pose_sequence["color_xy"],
        dtype=np.float32,
    )

    tracking_state_sequence = np.asarray(
        pose_sequence["tracking_state"]
    )

    capture = cv2.VideoCapture(
        str(rgb_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {rgb_path}"
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
        / (
            f"{sample_id}_"
            f"gt_crop_gt_vs_prediction.mp4"
        )
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

    previous_bbox: np.ndarray | None = None
    written_frames = 0
    skipped_frames = 0

    try:
        for frame_index in range(
            usable_frames
        ):
            success, frame = capture.read()

            if not success:
                print(
                    f"Video read stopped at "
                    f"frame {frame_index}"
                )
                break

            if (
                frame_index
                % FRAME_STRIDE
                != 0
            ):
                continue

            gt_keypoints = gt_sequence[
                frame_index
            ].copy()

            tracking_state = (
                tracking_state_sequence[
                    frame_index
                ]
            )

            gt_visibility = (
                tracking_state == 2
            )

            current_bbox = calculate_gt_bbox(
                keypoints=gt_keypoints,
                tracking_state=tracking_state,
                image_width=original_width,
                image_height=original_height,
            )

            if current_bbox is None:
                if previous_bbox is None:
                    skipped_frames += 1
                    continue

                current_bbox = (
                    previous_bbox.copy()
                )

            smoothed_bbox = smooth_bbox(
                current_bbox=current_bbox,
                previous_bbox=previous_bbox,
            )

            previous_bbox = (
                smoothed_bbox.copy()
            )

            (
                model_input,
                crop_bbox,
            ) = prepare_crop(
                frame=frame,
                bbox=smoothed_bbox,
            )

            model_input = model_input.to(
                device,
                non_blocking=True,
            )

            predicted_heatmaps = model(
                model_input
            )

            predicted_keypoints = (
                decode_heatmaps_to_crop(
                    heatmaps=predicted_heatmaps,
                    crop_bbox=crop_bbox,
                )
            )

            # Yellow: box used as model input.
            draw_bbox(
                frame=frame,
                crop_bbox=crop_bbox,
            )

            # Green: NTU GT.
            draw_pose(
                frame=frame,
                keypoints=gt_keypoints,
                color=GT_COLOR,
                visibility=gt_visibility,
            )

            # Red: model prediction.
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
                output_width != original_width
                or output_height != original_height
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
    print("=" * 72)
    print("Video generation finished")
    print("=" * 72)
    print(
        f"Frames written: {written_frames}"
    )
    print(
        f"Frames skipped: {skipped_frames}"
    )
    print(
        f"Output video:   {output_path}"
    )


if __name__ == "__main__":
    main()