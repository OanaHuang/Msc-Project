# Scripts/NTU_RGBD/08_Visualize_Penn_Model_on_NTU.py

from __future__ import annotations

from pathlib import Path
import csv
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


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

PENN_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "server_outputs"
    / "07_ResNet50_Heatmap_Baseline"
    / "best_ResNet50_Heatmap_Baseline.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "NTU_RGBD"
    / "08_Visualize_Penn_Model_on_NTU"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. Configuration
# ============================================================

# None means use the first sample in test_split.csv.
#
# Example:
# TARGET_SAMPLE_ID = "S001C001P001R001A001"
TARGET_SAMPLE_ID = None

IMAGE_SIZE = 224
NUM_PENN_JOINTS = 13

# Process every frame.
FRAME_STRIDE = 1

# None means process the entire video.
MAX_FRAMES = None

# None means preserve original video resolution.
OUTPUT_WIDTH = None

# None means use original video FPS.
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
# 6. Penn 13-joint definition
# ============================================================

# Penn joint order used by the trained model:
#
# 0  = head
# 1  = left shoulder
# 2  = right shoulder
# 3  = left elbow
# 4  = right elbow
# 5  = left wrist
# 6  = right wrist
# 7  = left hip
# 8  = right hip
# 9  = left knee
# 10 = right knee
# 11 = left ankle
# 12 = right ankle

PENN_EDGES = (
    # Upper body
    (0, 1),
    (0, 2),
    (1, 2),

    # Left arm
    (1, 3),
    (3, 5),

    # Right arm
    (2, 4),
    (4, 6),

    # Torso
    (1, 7),
    (2, 8),
    (7, 8),

    # Left leg
    (7, 9),
    (9, 11),

    # Right leg
    (8, 10),
    (10, 12),
)


# ============================================================
# 7. NTU-to-Penn joint mapping
# ============================================================

# NTU 25-joint indexing:
#
# 0  = spine base
# 1  = spine mid
# 2  = neck
# 3  = head
# 4  = left shoulder
# 5  = left elbow
# 6  = left wrist
# 7  = left hand
# 8  = right shoulder
# 9  = right elbow
# 10 = right wrist
# 11 = right hand
# 12 = left hip
# 13 = left knee
# 14 = left ankle
# 15 = left foot
# 16 = right hip
# 17 = right knee
# 18 = right ankle
# 19 = right foot
# 20 = spine shoulder
# 21 = left hand tip
# 22 = left thumb
# 23 = right hand tip
# 24 = right thumb

# Penn index -> NTU index
PENN_TO_NTU = np.asarray(
    [
        3,   # Penn head           <- NTU head
        4,   # Penn left shoulder  <- NTU left shoulder
        8,   # Penn right shoulder <- NTU right shoulder
        5,   # Penn left elbow     <- NTU left elbow
        9,   # Penn right elbow    <- NTU right elbow
        6,   # Penn left wrist     <- NTU left wrist
        10,  # Penn right wrist    <- NTU right wrist
        12,  # Penn left hip       <- NTU left hip
        16,  # Penn right hip      <- NTU right hip
        13,  # Penn left knee      <- NTU left knee
        17,  # Penn right knee     <- NTU right knee
        14,  # Penn left ankle     <- NTU left ankle
        18,  # Penn right ankle    <- NTU right ankle
    ],
    dtype=np.int64,
)


# ============================================================
# 8. Penn model
# ============================================================

class ResNet50HeatmapBaseline(nn.Module):

    def __init__(
        self,
        num_keypoints: int = NUM_PENN_JOINTS,
    ) -> None:
        super().__init__()

        # No ImageNet download is required during inference because
        # the complete trained checkpoint will be loaded afterward.
        resnet = models.resnet50(
            weights=None
        )

        # Input:
        # [B, 3, 224, 224]
        #
        # Output:
        # [B, 2048, 7, 7]
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # Output:
        # [B, 13, 56, 56]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                2048,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                num_keypoints,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(
            images
        )

        heatmaps = self.decoder(
            features
        )

        return heatmaps


# ============================================================
# 9. Device
# ============================================================

def get_device() -> torch.device:
    if DEVICE_NAME is not None:
        return torch.device(
            DEVICE_NAME
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# 10. Select NTU sample
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
# 11. Model input
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
# 12. Heatmap decoding
# ============================================================

def decode_heatmaps(
    heatmaps: torch.Tensor,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    """
    Decode [1, 13, H, W] heatmaps into coordinates in
    the original NTU video resolution.
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            "Heatmaps must have shape [B, J, H, W]"
        )

    if heatmaps.shape[0] != 1:
        raise ValueError(
            "This script expects batch size 1"
        )

    heatmaps = heatmaps[0]

    num_joints = heatmaps.shape[0]
    heatmap_height = heatmaps.shape[1]
    heatmap_width = heatmaps.shape[2]

    if num_joints != NUM_PENN_JOINTS:
        raise ValueError(
            f"Expected {NUM_PENN_JOINTS} heatmaps, "
            f"but received {num_joints}"
        )

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
# 13. NTU GT mapping
# ============================================================

def map_ntu_to_penn(
    ntu_keypoints: np.ndarray,
    ntu_tracking_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert NTU 25-joint GT into the Penn 13-joint layout.

    Only NTU joints with tracking_state == 2 are marked visible.
    """
    ntu_keypoints = np.asarray(
        ntu_keypoints,
        dtype=np.float32,
    )

    ntu_tracking_state = np.asarray(
        ntu_tracking_state
    ).reshape(-1)

    if ntu_keypoints.shape != (25, 2):
        raise ValueError(
            f"Expected NTU keypoints shape (25, 2), "
            f"but got {ntu_keypoints.shape}"
        )

    if ntu_tracking_state.shape[0] != 25:
        raise ValueError(
            f"Expected 25 tracking states, "
            f"but got {ntu_tracking_state.shape[0]}"
        )

    penn_keypoints = ntu_keypoints[
        PENN_TO_NTU
    ].copy()

    penn_visibility = (
        ntu_tracking_state[
            PENN_TO_NTU
        ] == 2
    )

    return (
        penn_keypoints,
        penn_visibility,
    )


# ============================================================
# 14. Drawing
# ============================================================

def valid_point(
    point: np.ndarray,
    width: int,
    height: int,
) -> bool:
    point = np.asarray(
        point
    )

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
        NUM_PENN_JOINTS,
        2,
    ):
        raise ValueError(
            f"Expected keypoints shape "
            f"({NUM_PENN_JOINTS}, 2), "
            f"but got {keypoints.shape}"
        )

    if visibility is None:
        visibility = np.ones(
            NUM_PENN_JOINTS,
            dtype=bool,
        )
    else:
        visibility = np.asarray(
            visibility
        ).reshape(-1).astype(bool)

        if (
            visibility.shape[0]
            != NUM_PENN_JOINTS
        ):
            raise ValueError(
                f"Expected visibility length "
                f"{NUM_PENN_JOINTS}, "
                f"but got {visibility.shape[0]}"
            )

    # Draw bones.
    for start_index, end_index in PENN_EDGES:
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
        (425, 100),
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
        "NTU GT",
        (20, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GT_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Penn prediction",
        (115, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        PRED_COLOR,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# 15. Checkpoint loading
# ============================================================

def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove 'module.' added by DataParallel when necessary.
    """
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_key = key[
                len("module.") :
            ]
        else:
            cleaned_key = key

        cleaned_state_dict[
            cleaned_key
        ] = value

    return cleaned_state_dict


def load_model(
    device: torch.device,
) -> nn.Module:
    if not PENN_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Penn checkpoint not found: "
            f"{PENN_CHECKPOINT_PATH}"
        )

    model = ResNet50HeatmapBaseline(
        num_keypoints=NUM_PENN_JOINTS,
    )

    checkpoint = torch.load(
        PENN_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Unsupported checkpoint format"
        )

    possible_state_dict_keys = (
        "model_state_dict",
        "state_dict",
        "model",
    )

    state_dict = None

    for key in possible_state_dict_keys:
        if (
            key in checkpoint
            and isinstance(
                checkpoint[key],
                dict,
            )
        ):
            state_dict = checkpoint[key]
            break

    # The checkpoint itself may already be a raw state_dict.
    if state_dict is None:
        state_dict = checkpoint

    state_dict = remove_module_prefix(
        state_dict
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# 16. Main
# ============================================================

@torch.no_grad()
def main() -> None:
    sample = select_sample()

    sample_id = sample[
        "sample_id"
    ]

    rgb_path = Path(
        sample["rgb_path"]
    )

    skeleton_path = Path(
        sample["skeleton_path"]
    )

    if not rgb_path.exists():
        raise FileNotFoundError(
            f"NTU RGB video not found: "
            f"{rgb_path}"
        )

    if not skeleton_path.exists():
        raise FileNotFoundError(
            f"NTU skeleton file not found: "
            f"{skeleton_path}"
        )

    device = get_device()

    print("=" * 72)
    print(
        "Penn ResNet50 model on NTU RGB+D"
    )
    print("=" * 72)
    print(f"Sample ID:       {sample_id}")
    print(f"NTU video:       {rgb_path}")
    print(f"NTU skeleton:    {skeleton_path}")
    print(f"Penn checkpoint: {PENN_CHECKPOINT_PATH}")
    print(f"Device:          {device}")

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

    ntu_gt_sequence = np.asarray(
        pose_sequence["color_xy"],
        dtype=np.float32,
    )

    tracking_state_sequence = np.asarray(
        pose_sequence["tracking_state"]
    )

    if ntu_gt_sequence.ndim != 3:
        raise ValueError(
            f"Expected GT sequence shape "
            f"[T, 25, 2], but got "
            f"{ntu_gt_sequence.shape}"
        )

    if tracking_state_sequence.ndim != 2:
        raise ValueError(
            f"Expected tracking-state shape "
            f"[T, 25], but got "
            f"{tracking_state_sequence.shape}"
        )

    capture = cv2.VideoCapture(
        str(rgb_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open NTU video: "
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
        ntu_gt_sequence.shape[0],
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
            f"penn_model_on_ntu.mp4"
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

    written_frames = 0

    try:
        for frame_index in range(
            usable_frames
        ):
            success, frame = capture.read()

            if not success:
                print(
                    f"Video reading stopped at "
                    f"frame {frame_index}"
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

            ntu_keypoints = (
                ntu_gt_sequence[
                    frame_index
                ]
            )

            ntu_tracking_state = (
                tracking_state_sequence[
                    frame_index
                ]
            )

            (
                penn_gt_keypoints,
                penn_gt_visibility,
            ) = map_ntu_to_penn(
                ntu_keypoints=ntu_keypoints,
                ntu_tracking_state=ntu_tracking_state,
            )

            # Green: mapped NTU GT.
            draw_pose(
                frame=frame,
                keypoints=penn_gt_keypoints,
                color=GT_COLOR,
                visibility=penn_gt_visibility,
            )

            # Red: Penn model prediction.
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

            writer.write(
                frame
            )

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
            "No frames were written"
        )

    print()
    print("=" * 72)
    print("Video generation finished")
    print("=" * 72)
    print(
        f"Frames written: {written_frames}"
    )
    print(
        f"Output video:   {output_path}"
    )


if __name__ == "__main__":
    main()