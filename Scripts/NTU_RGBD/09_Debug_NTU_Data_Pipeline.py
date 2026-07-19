# Scripts/NTU_RGBD/09_Debug_NTU_Data_Pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv
import math

import cv2
import numpy as np


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 修改为你的 NTU RGB 视频文件夹
NTU_RGB_DIR = (
    PROJECT_ROOT
    / "Datasets"
    / "NTU_RGBD"
    / "rgb_videos"
)

# 修改为你的 NTU Skeleton 文件夹
NTU_SKELETON_DIR = (
    PROJECT_ROOT
    / "Datasets"
    / "NTU_RGBD"
    / "skeletons"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "NTU_RGBD"
    / "09_Debug_NTU_Data_Pipeline"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_OUTPUT_DIR = OUTPUT_DIR / "frames"
FRAME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEATMAP_OUTPUT_DIR = OUTPUT_DIR / "heatmaps"
HEATMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = OUTPUT_DIR / "debug_report.csv"
OUTPUT_VIDEO = OUTPUT_DIR / "ntu_pipeline_debug.mp4"


# ------------------------------------------------------------
# Sample selection
# ------------------------------------------------------------

# None：自动寻找第一个同时存在 RGB 和 Skeleton 的样本
#
# 或者写成：
# SAMPLE_ID = "S001C001P001R001A001"
SAMPLE_ID: Optional[str] = "S015C003P016R002A058"

# 最多检查多少帧
MAX_FRAMES = 300

# 每隔多少帧保存一张静态调试图
SAVE_EVERY_N_FRAMES = 20

# 是否输出调试视频
SAVE_VIDEO = True

# 是否固定跟踪第一个选中的 body ID
#
# True:
#   第一帧选出主体后，后续优先使用相同 body ID。
#
# False:
#   每帧重新选择面积最大的 body。
LOCK_BODY_ID = True


# ------------------------------------------------------------
# Crop / heatmap settings
# ------------------------------------------------------------

MODEL_INPUT_WIDTH = 224
MODEL_INPUT_HEIGHT = 224

HEATMAP_WIDTH = 56
HEATMAP_HEIGHT = 56

HEATMAP_SIGMA = 2.0

# 人体框向外扩展比例
BBOX_EXPANSION = 0.25

# 至少需要多少个有效关节点才处理这一帧
MIN_VALID_JOINTS = 8


# ------------------------------------------------------------
# NTU joint definitions
# ------------------------------------------------------------

NUM_JOINTS = 25

JOINT_NAMES = [
    "SpineBase",          # 0
    "SpineMid",           # 1
    "Neck",               # 2
    "Head",               # 3
    "ShoulderLeft",       # 4
    "ElbowLeft",          # 5
    "WristLeft",          # 6
    "HandLeft",           # 7
    "ShoulderRight",      # 8
    "ElbowRight",         # 9
    "WristRight",         # 10
    "HandRight",          # 11
    "HipLeft",            # 12
    "KneeLeft",           # 13
    "AnkleLeft",          # 14
    "FootLeft",           # 15
    "HipRight",           # 16
    "KneeRight",          # 17
    "AnkleRight",         # 18
    "FootRight",          # 19
    "SpineShoulder",      # 20
    "HandTipLeft",        # 21
    "ThumbLeft",          # 22
    "HandTipRight",       # 23
    "ThumbRight",         # 24
]

# NTU 25-joint skeleton connections
SKELETON_EDGES = [
    (0, 1),
    (1, 20),
    (20, 2),
    (2, 3),

    (20, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 21),
    (6, 22),

    (20, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 23),
    (10, 24),

    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),

    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
]


# ============================================================
# 2. Data structures
# ============================================================

@dataclass
class Joint:
    x: float
    y: float
    z: float

    depth_x: float
    depth_y: float

    color_x: float
    color_y: float

    orientation_w: float
    orientation_x: float
    orientation_y: float
    orientation_z: float

    tracking_state: int


@dataclass
class Body:
    body_id: str
    joints: list[Joint]


@dataclass
class SkeletonFrame:
    bodies: list[Body]


# ============================================================
# 3. NTU skeleton parser
# ============================================================

def parse_joint_line(line: str) -> Joint:
    values = line.strip().split()

    if len(values) < 12:
        raise ValueError(
            f"Invalid NTU joint line. "
            f"Expected at least 12 values, got {len(values)}:\n{line}"
        )

    return Joint(
        x=float(values[0]),
        y=float(values[1]),
        z=float(values[2]),

        depth_x=float(values[3]),
        depth_y=float(values[4]),

        color_x=float(values[5]),
        color_y=float(values[6]),

        orientation_w=float(values[7]),
        orientation_x=float(values[8]),
        orientation_y=float(values[9]),
        orientation_z=float(values[10]),

        tracking_state=int(float(values[11])),
    )


def load_ntu_skeleton(
    skeleton_path: Path,
) -> list[SkeletonFrame]:
    frames: list[SkeletonFrame] = []

    with skeleton_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        first_line = file.readline()

        if not first_line:
            raise ValueError(
                f"Empty skeleton file: {skeleton_path}"
            )

        num_frames = int(first_line.strip())

        for frame_index in range(num_frames):
            line = file.readline()

            if not line:
                raise EOFError(
                    f"Unexpected end of skeleton file at "
                    f"frame {frame_index}."
                )

            num_bodies = int(line.strip())
            bodies: list[Body] = []

            for _ in range(num_bodies):
                body_info_line = file.readline().strip()
                body_info = body_info_line.split()

                if not body_info:
                    raise ValueError(
                        f"Missing body information at "
                        f"frame {frame_index}."
                    )

                body_id = body_info[0]

                num_joints_line = file.readline().strip()
                num_joints = int(num_joints_line)

                joints: list[Joint] = []

                for _ in range(num_joints):
                    joint_line = file.readline()
                    joints.append(
                        parse_joint_line(joint_line)
                    )

                bodies.append(
                    Body(
                        body_id=body_id,
                        joints=joints,
                    )
                )

            frames.append(
                SkeletonFrame(
                    bodies=bodies,
                )
            )

    return frames


# ============================================================
# 4. File matching
# ============================================================

def normalise_sample_id(path: Path) -> str:
    """
    Examples:

    S001C001P001R001A001_rgb.avi
        -> S001C001P001R001A001

    S001C001P001R001A001.skeleton
        -> S001C001P001R001A001
    """
    name = path.stem

    suffixes = [
        "_rgb",
        "_color",
        "_skeleton",
    ]

    for suffix in suffixes:
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]

    return name


def find_files_recursive(
    root: Path,
    extensions: tuple[str, ...],
) -> list[Path]:
    if not root.exists():
        return []

    results: list[Path] = []

    for extension in extensions:
        results.extend(root.rglob(f"*{extension}"))
        results.extend(root.rglob(f"*{extension.upper()}"))

    return sorted(set(results))


def find_matching_sample() -> tuple[str, Path, Path]:
    rgb_extensions = (
        ".avi",
        ".mp4",
        ".mkv",
        ".mov",
    )

    rgb_files = find_files_recursive(
        NTU_RGB_DIR,
        rgb_extensions,
    )

    skeleton_files = find_files_recursive(
        NTU_SKELETON_DIR,
        (".skeleton",),
    )

    if not rgb_files:
        raise FileNotFoundError(
            f"No RGB videos found under:\n{NTU_RGB_DIR}"
        )

    if not skeleton_files:
        raise FileNotFoundError(
            f"No skeleton files found under:\n"
            f"{NTU_SKELETON_DIR}"
        )

    rgb_by_id = {
        normalise_sample_id(path): path
        for path in rgb_files
    }

    skeleton_by_id = {
        normalise_sample_id(path): path
        for path in skeleton_files
    }

    if SAMPLE_ID is not None:
        if SAMPLE_ID not in rgb_by_id:
            raise FileNotFoundError(
                f"RGB video not found for sample:\n"
                f"{SAMPLE_ID}"
            )

        if SAMPLE_ID not in skeleton_by_id:
            raise FileNotFoundError(
                f"Skeleton file not found for sample:\n"
                f"{SAMPLE_ID}"
            )

        return (
            SAMPLE_ID,
            rgb_by_id[SAMPLE_ID],
            skeleton_by_id[SAMPLE_ID],
        )

    common_ids = sorted(
        set(rgb_by_id.keys())
        & set(skeleton_by_id.keys())
    )

    if not common_ids:
        print("\nFirst RGB sample IDs:")
        for key in list(rgb_by_id.keys())[:10]:
            print(f"  {key}")

        print("\nFirst Skeleton sample IDs:")
        for key in list(skeleton_by_id.keys())[:10]:
            print(f"  {key}")

        raise RuntimeError(
            "No matching RGB/Skeleton sample IDs found."
        )

    sample_id = common_ids[0]

    return (
        sample_id,
        rgb_by_id[sample_id],
        skeleton_by_id[sample_id],
    )


# ============================================================
# 5. Joint validation and body selection
# ============================================================

def is_finite_number(value: float) -> bool:
    return math.isfinite(value)


def joint_is_valid_color(
    joint: Joint,
    image_width: int,
    image_height: int,
) -> bool:
    x = joint.color_x
    y = joint.color_y

    if not is_finite_number(x):
        return False

    if not is_finite_number(y):
        return False

    if x < 0 or x >= image_width:
        return False

    if y < 0 or y >= image_height:
        return False

    # NTU tracking state:
    # 0 = not tracked
    # 1 = inferred
    # 2 = tracked
    if joint.tracking_state <= 0:
        return False

    return True


def body_to_color_keypoints(
    body: Body,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros(
        (NUM_JOINTS, 2),
        dtype=np.float32,
    )

    valid = np.zeros(
        NUM_JOINTS,
        dtype=bool,
    )

    available_joints = min(
        NUM_JOINTS,
        len(body.joints),
    )

    for joint_index in range(available_joints):
        joint = body.joints[joint_index]

        keypoints[joint_index, 0] = joint.color_x
        keypoints[joint_index, 1] = joint.color_y

        valid[joint_index] = joint_is_valid_color(
            joint,
            image_width,
            image_height,
        )

    return keypoints, valid


def calculate_keypoint_bbox(
    keypoints: np.ndarray,
    valid: np.ndarray,
) -> Optional[tuple[float, float, float, float]]:
    points = keypoints[valid]

    if len(points) == 0:
        return None

    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))

    return x1, y1, x2, y2


def bbox_area(
    bbox: Optional[tuple[float, float, float, float]],
) -> float:
    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def select_body(
    frame: SkeletonFrame,
    image_width: int,
    image_height: int,
    preferred_body_id: Optional[str],
) -> Optional[Body]:
    if not frame.bodies:
        return None

    # First try the locked body ID
    if preferred_body_id is not None:
        for body in frame.bodies:
            if body.body_id == preferred_body_id:
                keypoints, valid = body_to_color_keypoints(
                    body,
                    image_width,
                    image_height,
                )

                if int(valid.sum()) >= MIN_VALID_JOINTS:
                    return body

    # Otherwise choose the body with the largest visible bbox
    best_body: Optional[Body] = None
    best_score = -1.0

    for body in frame.bodies:
        keypoints, valid = body_to_color_keypoints(
            body,
            image_width,
            image_height,
        )

        valid_count = int(valid.sum())

        if valid_count < MIN_VALID_JOINTS:
            continue

        bbox = calculate_keypoint_bbox(
            keypoints,
            valid,
        )

        area = bbox_area(bbox)

        # Prioritise valid joint count, then area
        score = valid_count * 1_000_000.0 + area

        if score > best_score:
            best_score = score
            best_body = body

    return best_body


# ============================================================
# 6. Bounding box and coordinate transforms
# ============================================================

def expand_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    expansion: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    expanded_width = width * (1.0 + expansion * 2.0)
    expanded_height = height * (1.0 + expansion * 2.0)

    new_x1 = int(round(center_x - expanded_width / 2.0))
    new_y1 = int(round(center_y - expanded_height / 2.0))
    new_x2 = int(round(center_x + expanded_width / 2.0))
    new_y2 = int(round(center_y + expanded_height / 2.0))

    new_x1 = max(0, min(new_x1, image_width - 1))
    new_y1 = max(0, min(new_y1, image_height - 1))

    new_x2 = max(new_x1 + 1, min(new_x2, image_width))
    new_y2 = max(new_y1 + 1, min(new_y2, image_height))

    return new_x1, new_y1, new_x2, new_y2


def transform_keypoints_to_crop(
    keypoints: np.ndarray,
    valid: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = crop_bbox

    crop_width = x2 - x1
    crop_height = y2 - y1

    transformed = keypoints.copy().astype(
        np.float32
    )

    transformed[:, 0] = (
        (transformed[:, 0] - x1)
        * target_width
        / crop_width
    )

    transformed[:, 1] = (
        (transformed[:, 1] - y1)
        * target_height
        / crop_height
    )

    transformed_valid = valid.copy()

    transformed_valid &= (
        transformed[:, 0] >= 0
    )
    transformed_valid &= (
        transformed[:, 0] < target_width
    )
    transformed_valid &= (
        transformed[:, 1] >= 0
    )
    transformed_valid &= (
        transformed[:, 1] < target_height
    )

    return transformed, transformed_valid


def transform_crop_to_original(
    crop_keypoints: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
) -> np.ndarray:
    x1, y1, x2, y2 = crop_bbox

    original_width = x2 - x1
    original_height = y2 - y1

    restored = crop_keypoints.copy().astype(
        np.float32
    )

    restored[:, 0] = (
        restored[:, 0]
        * original_width
        / crop_width
        + x1
    )

    restored[:, 1] = (
        restored[:, 1]
        * original_height
        / crop_height
        + y1
    )

    return restored


# ============================================================
# 7. Heatmap creation and decoding
# ============================================================

def draw_gaussian(
    heatmap: np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float,
) -> None:
    height, width = heatmap.shape

    radius = int(3.0 * sigma)

    x0 = int(round(center_x))
    y0 = int(round(center_y))

    left = max(0, x0 - radius)
    right = min(width - 1, x0 + radius)

    top = max(0, y0 - radius)
    bottom = min(height - 1, y0 + radius)

    if left > right or top > bottom:
        return

    x_values = np.arange(
        left,
        right + 1,
        dtype=np.float32,
    )

    y_values = np.arange(
        top,
        bottom + 1,
        dtype=np.float32,
    )

    grid_x, grid_y = np.meshgrid(
        x_values,
        y_values,
    )

    gaussian = np.exp(
        -(
            (grid_x - center_x) ** 2
            + (grid_y - center_y) ** 2
        )
        / (2.0 * sigma ** 2)
    )

    current_region = heatmap[
        top:bottom + 1,
        left:right + 1,
    ]

    np.maximum(
        current_region,
        gaussian,
        out=current_region,
    )


def create_heatmaps(
    input_keypoints: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    heatmaps = np.zeros(
        (
            NUM_JOINTS,
            HEATMAP_HEIGHT,
            HEATMAP_WIDTH,
        ),
        dtype=np.float32,
    )

    scale_x = (
        HEATMAP_WIDTH
        / MODEL_INPUT_WIDTH
    )

    scale_y = (
        HEATMAP_HEIGHT
        / MODEL_INPUT_HEIGHT
    )

    for joint_index in range(NUM_JOINTS):
        if not valid[joint_index]:
            continue

        heatmap_x = (
            input_keypoints[joint_index, 0]
            * scale_x
        )

        heatmap_y = (
            input_keypoints[joint_index, 1]
            * scale_y
        )

        draw_gaussian(
            heatmaps[joint_index],
            heatmap_x,
            heatmap_y,
            HEATMAP_SIGMA,
        )

    return heatmaps


def decode_heatmaps(
    heatmaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_joints = heatmaps.shape[0]

    decoded = np.zeros(
        (num_joints, 2),
        dtype=np.float32,
    )

    confidences = np.zeros(
        num_joints,
        dtype=np.float32,
    )

    for joint_index in range(num_joints):
        heatmap = heatmaps[joint_index]

        flat_index = int(np.argmax(heatmap))
        y, x = np.unravel_index(
            flat_index,
            heatmap.shape,
        )

        decoded[joint_index, 0] = (
            (x + 0.5)
            * MODEL_INPUT_WIDTH
            / HEATMAP_WIDTH
        )

        decoded[joint_index, 1] = (
            (y + 0.5)
            * MODEL_INPUT_HEIGHT
            / HEATMAP_HEIGHT
        )

        confidences[joint_index] = float(
            heatmap[y, x]
        )

    return decoded, confidences


# ============================================================
# 8. Visualisation
# ============================================================

def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    valid: np.ndarray,
    point_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    output = image.copy()

    for joint_a, joint_b in SKELETON_EDGES:
        if not valid[joint_a]:
            continue

        if not valid[joint_b]:
            continue

        point_a = (
            int(round(keypoints[joint_a, 0])),
            int(round(keypoints[joint_a, 1])),
        )

        point_b = (
            int(round(keypoints[joint_b, 0])),
            int(round(keypoints[joint_b, 1])),
        )

        cv2.line(
            output,
            point_a,
            point_b,
            (0, 255, 0),
            line_thickness,
            cv2.LINE_AA,
        )

    for joint_index in range(NUM_JOINTS):
        if not valid[joint_index]:
            continue

        point = (
            int(round(keypoints[joint_index, 0])),
            int(round(keypoints[joint_index, 1])),
        )

        cv2.circle(
            output,
            point,
            point_radius,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            str(joint_index),
            (
                point[0] + 4,
                point[1] - 4,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    output = image.copy()

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3,
        cv2.LINE_AA,
    )

    return output


def add_title(
    image: np.ndarray,
    title: str,
) -> np.ndarray:
    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 34),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def create_heatmap_overlay(
    crop_image: np.ndarray,
    heatmaps: np.ndarray,
) -> np.ndarray:
    combined = np.max(
        heatmaps,
        axis=0,
    )

    combined = cv2.resize(
        combined,
        (
            MODEL_INPUT_WIDTH,
            MODEL_INPUT_HEIGHT,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    combined_uint8 = np.clip(
        combined * 255.0,
        0,
        255,
    ).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(
        combined_uint8,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        crop_image,
        0.55,
        heatmap_color,
        0.45,
        0,
    )

    return overlay


def resize_with_padding(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    source_height, source_width = image.shape[:2]

    scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    resized_width = max(
        1,
        int(round(source_width * scale)),
    )

    resized_height = max(
        1,
        int(round(source_height * scale)),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.zeros(
        (
            target_height,
            target_width,
            3,
        ),
        dtype=np.uint8,
    )

    x_offset = (
        target_width - resized_width
    ) // 2

    y_offset = (
        target_height - resized_height
    ) // 2

    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized

    return canvas


def build_debug_panel(
    original_with_gt: np.ndarray,
    original_with_bbox: np.ndarray,
    crop_with_gt: np.ndarray,
    crop_with_decoded: np.ndarray,
    heatmap_overlay: np.ndarray,
    frame_index: int,
    body_id: str,
    valid_joint_count: int,
    mean_decode_error: float,
) -> np.ndarray:
    panel_width = 640
    panel_height = 360

    original_gt_panel = resize_with_padding(
        add_title(
            original_with_gt,
            "1. Original RGB + NTU color coordinates",
        ),
        panel_width,
        panel_height,
    )

    bbox_panel = resize_with_padding(
        add_title(
            original_with_bbox,
            "2. Skeleton-derived person crop",
        ),
        panel_width,
        panel_height,
    )

    crop_gt_panel = resize_with_padding(
        add_title(
            crop_with_gt,
            "3. Resized crop + transformed GT",
        ),
        panel_width,
        panel_height,
    )

    crop_decode_panel = resize_with_padding(
        add_title(
            crop_with_decoded,
            "4. Heatmap argmax decoded skeleton",
        ),
        panel_width,
        panel_height,
    )

    heatmap_panel = resize_with_padding(
        add_title(
            heatmap_overlay,
            "5. Combined ground-truth heatmaps",
        ),
        panel_width,
        panel_height,
    )

    information_panel = np.zeros(
        (
            panel_height,
            panel_width,
            3,
        ),
        dtype=np.uint8,
    )

    information_lines = [
        "NTU data pipeline diagnostics",
        f"Frame index: {frame_index}",
        f"Selected body ID: {body_id}",
        f"Valid color joints: {valid_joint_count}/{NUM_JOINTS}",
        (
            "Mean heatmap decode error: "
            f"{mean_decode_error:.3f} input pixels"
        ),
        "",
        "Expected:",
        "- Skeleton follows the person in original RGB",
        "- Blue bbox covers the full body",
        "- Crop skeleton remains correctly aligned",
        "- Decoded skeleton nearly overlaps crop GT",
        "",
        "Failure interpretation:",
        "- Original GT wrong: color coordinates/frame match",
        "- Crop GT wrong: crop coordinate transform",
        "- Decode wrong: heatmap encode/decode",
    ]

    y = 35

    for line in information_lines:
        cv2.putText(
            information_panel,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 25

    top_row = np.hstack([
        original_gt_panel,
        bbox_panel,
        crop_gt_panel,
    ])

    bottom_row = np.hstack([
        crop_decode_panel,
        heatmap_panel,
        information_panel,
    ])

    return np.vstack([
        top_row,
        bottom_row,
    ])


# ============================================================
# 9. Diagnostics
# ============================================================

def calculate_decode_error(
    original_keypoints: np.ndarray,
    decoded_keypoints: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
    if int(valid.sum()) == 0:
        return float("nan"), float("nan")

    errors = np.linalg.norm(
        original_keypoints[valid]
        - decoded_keypoints[valid],
        axis=1,
    )

    return (
        float(np.mean(errors)),
        float(np.max(errors)),
    )


def inspect_coordinate_statistics(
    skeleton_frames: list[SkeletonFrame],
    video_width: int,
    video_height: int,
) -> None:
    color_values = []
    depth_values = []
    invalid_color_count = 0
    total_joint_count = 0

    for frame in skeleton_frames[
        :min(MAX_FRAMES, len(skeleton_frames))
    ]:
        for body in frame.bodies:
            for joint in body.joints:
                total_joint_count += 1

                if (
                    math.isfinite(joint.color_x)
                    and math.isfinite(joint.color_y)
                ):
                    color_values.append([
                        joint.color_x,
                        joint.color_y,
                    ])
                else:
                    invalid_color_count += 1

                if (
                    math.isfinite(joint.depth_x)
                    and math.isfinite(joint.depth_y)
                ):
                    depth_values.append([
                        joint.depth_x,
                        joint.depth_y,
                    ])

    print("\nCoordinate statistics")
    print("-" * 70)
    print(
        f"RGB resolution: "
        f"{video_width} x {video_height}"
    )
    print(
        f"Total inspected joints: "
        f"{total_joint_count}"
    )
    print(
        f"Non-finite color coordinates: "
        f"{invalid_color_count}"
    )

    if color_values:
        color_array = np.asarray(
            color_values,
            dtype=np.float32,
        )

        print(
            "colorX range: "
            f"{color_array[:, 0].min():.2f} "
            f"to {color_array[:, 0].max():.2f}"
        )

        print(
            "colorY range: "
            f"{color_array[:, 1].min():.2f} "
            f"to {color_array[:, 1].max():.2f}"
        )

    if depth_values:
        depth_array = np.asarray(
            depth_values,
            dtype=np.float32,
        )

        print(
            "depthX range: "
            f"{depth_array[:, 0].min():.2f} "
            f"to {depth_array[:, 0].max():.2f}"
        )

        print(
            "depthY range: "
            f"{depth_array[:, 1].min():.2f} "
            f"to {depth_array[:, 1].max():.2f}"
        )


# ============================================================
# 10. Main
# ============================================================

def main() -> None:
    print("=" * 70)
    print("NTU RGB + Skeleton Data Pipeline Debugger")
    print("=" * 70)

    sample_id, rgb_path, skeleton_path = (
        find_matching_sample()
    )

    print(f"\nSample ID:\n  {sample_id}")
    print(f"\nRGB video:\n  {rgb_path}")
    print(f"\nSkeleton file:\n  {skeleton_path}")

    skeleton_frames = load_ntu_skeleton(
        skeleton_path
    )

    capture = cv2.VideoCapture(
        str(rgb_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open RGB video:\n{rgb_path}"
        )

    video_frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    video_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    video_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    video_fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    if video_fps <= 0:
        video_fps = 30.0

    skeleton_frame_count = len(
        skeleton_frames
    )

    print("\nFrame alignment")
    print("-" * 70)
    print(
        f"RGB frame count:      "
        f"{video_frame_count}"
    )
    print(
        f"Skeleton frame count: "
        f"{skeleton_frame_count}"
    )

    frame_count_difference = (
        video_frame_count
        - skeleton_frame_count
    )

    print(
        f"Difference:           "
        f"{frame_count_difference}"
    )

    if frame_count_difference == 0:
        print(
            "Frame count check: PASS"
        )
    else:
        print(
            "Frame count check: WARNING"
        )
        print(
            "RGB and Skeleton frame counts differ. "
            "This may indicate frame-alignment problems."
        )

    inspect_coordinate_statistics(
        skeleton_frames,
        video_width,
        video_height,
    )

    total_frames_to_process = min(
        video_frame_count,
        skeleton_frame_count,
        MAX_FRAMES,
    )

    video_writer = None

    locked_body_id: Optional[str] = None

    report_rows: list[dict] = []

    processed_count = 0
    skipped_no_body = 0
    skipped_few_joints = 0
    skipped_invalid_bbox = 0

    print("\nProcessing frames")
    print("-" * 70)

    for frame_index in range(
        total_frames_to_process
    ):
        success, frame_bgr = capture.read()

        if not success:
            print(
                f"Stopped: failed to read RGB frame "
                f"{frame_index}."
            )
            break

        skeleton_frame = skeleton_frames[
            frame_index
        ]

        preferred_body_id = (
            locked_body_id
            if LOCK_BODY_ID
            else None
        )

        body = select_body(
            skeleton_frame,
            video_width,
            video_height,
            preferred_body_id,
        )

        if body is None:
            skipped_no_body += 1

            report_rows.append({
                "frame_index": frame_index,
                "status": "no_valid_body",
                "body_id": "",
                "num_bodies": len(
                    skeleton_frame.bodies
                ),
                "valid_joint_count": 0,
                "bbox_x1": "",
                "bbox_y1": "",
                "bbox_x2": "",
                "bbox_y2": "",
                "mean_decode_error": "",
                "max_decode_error": "",
            })

            continue

        if LOCK_BODY_ID and locked_body_id is None:
            locked_body_id = body.body_id

            print(
                f"Locked body ID: "
                f"{locked_body_id}"
            )

        keypoints_original, valid_original = (
            body_to_color_keypoints(
                body,
                video_width,
                video_height,
            )
        )

        valid_joint_count = int(
            valid_original.sum()
        )

        if valid_joint_count < MIN_VALID_JOINTS:
            skipped_few_joints += 1
            continue

        raw_bbox = calculate_keypoint_bbox(
            keypoints_original,
            valid_original,
        )

        if raw_bbox is None:
            skipped_invalid_bbox += 1
            continue

        crop_bbox = expand_bbox(
            raw_bbox,
            video_width,
            video_height,
            BBOX_EXPANSION,
        )

        x1, y1, x2, y2 = crop_bbox

        crop_bgr = frame_bgr[
            y1:y2,
            x1:x2,
        ]

        if crop_bgr.size == 0:
            skipped_invalid_bbox += 1
            continue

        crop_resized = cv2.resize(
            crop_bgr,
            (
                MODEL_INPUT_WIDTH,
                MODEL_INPUT_HEIGHT,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        keypoints_crop, valid_crop = (
            transform_keypoints_to_crop(
                keypoints_original,
                valid_original,
                crop_bbox,
                MODEL_INPUT_WIDTH,
                MODEL_INPUT_HEIGHT,
            )
        )

        heatmaps = create_heatmaps(
            keypoints_crop,
            valid_crop,
        )

        decoded_crop, confidences = (
            decode_heatmaps(
                heatmaps
            )
        )

        decoded_valid = (
            valid_crop
            & (confidences > 0)
        )

        mean_decode_error, max_decode_error = (
            calculate_decode_error(
                keypoints_crop,
                decoded_crop,
                decoded_valid,
            )
        )

        decoded_original = (
            transform_crop_to_original(
                decoded_crop,
                crop_bbox,
                MODEL_INPUT_WIDTH,
                MODEL_INPUT_HEIGHT,
            )
        )

        original_with_gt = draw_skeleton(
            frame_bgr,
            keypoints_original,
            valid_original,
            point_radius=5,
            line_thickness=3,
        )

        original_with_bbox = draw_bbox(
            original_with_gt,
            crop_bbox,
        )

        crop_with_gt = draw_skeleton(
            crop_resized,
            keypoints_crop,
            valid_crop,
            point_radius=4,
            line_thickness=2,
        )

        crop_with_decoded = draw_skeleton(
            crop_resized,
            decoded_crop,
            decoded_valid,
            point_radius=4,
            line_thickness=2,
        )

        heatmap_overlay = create_heatmap_overlay(
            crop_resized,
            heatmaps,
        )

        debug_panel = build_debug_panel(
            original_with_gt=original_with_gt,
            original_with_bbox=original_with_bbox,
            crop_with_gt=crop_with_gt,
            crop_with_decoded=crop_with_decoded,
            heatmap_overlay=heatmap_overlay,
            frame_index=frame_index,
            body_id=body.body_id,
            valid_joint_count=valid_joint_count,
            mean_decode_error=mean_decode_error,
        )

        if SAVE_VIDEO:
            if video_writer is None:
                panel_height, panel_width = (
                    debug_panel.shape[:2]
                )

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                video_writer = cv2.VideoWriter(
                    str(OUTPUT_VIDEO),
                    fourcc,
                    video_fps,
                    (
                        panel_width,
                        panel_height,
                    ),
                )

                if not video_writer.isOpened():
                    raise RuntimeError(
                        f"Could not create video:\n"
                        f"{OUTPUT_VIDEO}"
                    )

            video_writer.write(
                debug_panel
            )

        if (
            frame_index % SAVE_EVERY_N_FRAMES
            == 0
        ):
            frame_path = (
                FRAME_OUTPUT_DIR
                / f"frame_{frame_index:05d}.jpg"
            )

            cv2.imwrite(
                str(frame_path),
                debug_panel,
            )

            combined_heatmap = np.max(
                heatmaps,
                axis=0,
            )

            heatmap_path = (
                HEATMAP_OUTPUT_DIR
                / f"heatmap_{frame_index:05d}.npy"
            )

            np.save(
                heatmap_path,
                combined_heatmap,
            )

        report_rows.append({
            "frame_index": frame_index,
            "status": "processed",
            "body_id": body.body_id,
            "num_bodies": len(
                skeleton_frame.bodies
            ),
            "valid_joint_count": valid_joint_count,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "mean_decode_error": (
                f"{mean_decode_error:.6f}"
            ),
            "max_decode_error": (
                f"{max_decode_error:.6f}"
            ),
        })

        processed_count += 1

        if (
            processed_count == 1
            or processed_count % 50 == 0
        ):
            print(
                f"Processed {processed_count} frames | "
                f"video frame {frame_index} | "
                f"valid joints {valid_joint_count} | "
                f"decode error "
                f"{mean_decode_error:.3f}px"
            )

    capture.release()

    if video_writer is not None:
        video_writer.release()

    fieldnames = [
        "frame_index",
        "status",
        "body_id",
        "num_bodies",
        "valid_joint_count",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "mean_decode_error",
        "max_decode_error",
    ]

    with REPORT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(report_rows)

    processed_errors = []

    for row in report_rows:
        if row["status"] != "processed":
            continue

        value = row["mean_decode_error"]

        if value == "":
            continue

        processed_errors.append(
            float(value)
        )

    print("\n" + "=" * 70)
    print("Debug summary")
    print("=" * 70)

    print(
        f"Frames requested:       "
        f"{total_frames_to_process}"
    )

    print(
        f"Frames processed:       "
        f"{processed_count}"
    )

    print(
        f"Skipped, no body:       "
        f"{skipped_no_body}"
    )

    print(
        f"Skipped, few joints:    "
        f"{skipped_few_joints}"
    )

    print(
        f"Skipped, invalid bbox:  "
        f"{skipped_invalid_bbox}"
    )

    if processed_errors:
        print(
            "Average heatmap "
            f"encode/decode error: "
            f"{np.mean(processed_errors):.4f} px"
        )

        print(
            "Maximum frame-average "
            f"decode error: "
            f"{np.max(processed_errors):.4f} px"
        )

    print(f"\nReport:\n  {REPORT_CSV}")

    print(
        f"\nDebug frames:\n  "
        f"{FRAME_OUTPUT_DIR}"
    )

    if SAVE_VIDEO:
        print(
            f"\nDebug video:\n  "
            f"{OUTPUT_VIDEO}"
        )

    print("\nHow to interpret the result")
    print("-" * 70)
    print(
        "1. If skeleton is wrong on the original RGB frame:"
    )
    print(
        "   Check RGB/Skeleton matching, frame indexing, "
        "and colorX/colorY."
    )

    print(
        "2. If original skeleton is correct but crop "
        "skeleton is wrong:"
    )
    print(
        "   Check bbox cropping and coordinate transforms."
    )

    print(
        "3. If crop GT is correct but decoded skeleton "
        "is wrong:"
    )
    print(
        "   Check heatmap creation and decoding."
    )

    print(
        "4. If all five panels are correct:"
    )
    print(
        "   The basic data pipeline is likely correct. "
        "Next run a small-set overfitting test."
    )


if __name__ == "__main__":
    main()