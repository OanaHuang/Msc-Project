# Scripts/NTU_RGBD/10_Check_Skeleton_RGB_Alignment.py

from __future__ import annotations

from pathlib import Path
import csv
import math
from typing import Optional

import cv2
import numpy as np

from core.skeleton_reader import (
    NTUBody,
    NTUSkeletonSequence,
    read_skeleton_file,
)


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_ID = "S015C001P015R001A023"

RGB_ROOT = (
    PROJECT_ROOT
    / "Datasets"
    / "NTU_RGBD"
    / "rgb_videos"
)

SKELETON_ROOT = (
    PROJECT_ROOT
    / "Datasets"
    / "NTU_RGBD"
    / "skeletons"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "NTU_RGBD"
    / "10_Check_Skeleton_RGB_Alignment"
    / SAMPLE_ID
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FRAME_DIR = OUTPUT_DIR / "sample_frames"
FRAME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALIGNMENT_VIDEO_PATH = (
    OUTPUT_DIR
    / f"{SAMPLE_ID}_alignment.mp4"
)

OFFSET_VIDEO_PATH = (
    OUTPUT_DIR
    / f"{SAMPLE_ID}_offset_comparison.mp4"
)

REPORT_CSV_PATH = (
    OUTPUT_DIR
    / f"{SAMPLE_ID}_alignment_report.csv"
)


# 最多处理多少帧
# None 表示处理完整视频
MAX_FRAMES: Optional[int] = None

# 是否只绘制 tracking_state == 2 的可靠关节点
#
# False:
#   state 1 和 state 2 都显示，但样式不同。
#
# True:
#   只显示 state 2。
ONLY_TRACKED_JOINTS = False

# 是否绘制关节点编号
DRAW_JOINT_INDEX = True

# 是否生成时间偏移对比视频
GENERATE_OFFSET_VIDEO = True

# 测试 Skeleton 时间偏移
FRAME_OFFSETS = (-2, -1, 0, 1, 2)

# 输出视频缩放比例
OUTPUT_SCALE = 0.75

# 抽样保存多少张静态图
NUM_SAMPLE_FRAMES = 12

# 边界框扩展比例
BBOX_EXPANSION = 0.15


# ============================================================
# 2. NTU joint definitions
# ============================================================

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

SKELETON_EDGES = [
    # Trunk
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
    (6, 22),

    # Right arm
    (20, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 23),
    (10, 24),

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
]


# ============================================================
# 3. File discovery
# ============================================================

def normalise_sample_id(
    path: Path,
) -> str:
    name = path.stem

    suffixes = (
        "_rgb",
        "_color",
        "_skeleton",
    )

    for suffix in suffixes:
        if name.lower().endswith(
            suffix
        ):
            name = name[
                :-len(suffix)
            ]

    return name


def find_files_recursive(
    root: Path,
    extensions: tuple[str, ...],
) -> list[Path]:
    if not root.exists():
        return []

    paths: list[Path] = []

    for extension in extensions:
        paths.extend(
            root.rglob(
                f"*{extension}"
            )
        )

        paths.extend(
            root.rglob(
                f"*{extension.upper()}"
            )
        )

    return sorted(
        set(paths)
    )


def find_sample_files() -> tuple[Path, Path]:
    rgb_files = find_files_recursive(
        RGB_ROOT,
        (
            ".avi",
            ".mp4",
            ".mkv",
            ".mov",
        ),
    )

    skeleton_files = (
        find_files_recursive(
            SKELETON_ROOT,
            (".skeleton",),
        )
    )

    rgb_by_id = {
        normalise_sample_id(path): path
        for path in rgb_files
    }

    skeleton_by_id = {
        normalise_sample_id(path): path
        for path in skeleton_files
    }

    if SAMPLE_ID not in rgb_by_id:
        raise FileNotFoundError(
            "Could not find RGB video for:\n"
            f"{SAMPLE_ID}\n\n"
            f"Search root:\n{RGB_ROOT}"
        )

    if SAMPLE_ID not in skeleton_by_id:
        raise FileNotFoundError(
            "Could not find Skeleton file for:\n"
            f"{SAMPLE_ID}\n\n"
            f"Search root:\n{SKELETON_ROOT}"
        )

    return (
        rgb_by_id[SAMPLE_ID],
        skeleton_by_id[SAMPLE_ID],
    )


# ============================================================
# 4. Body selection
# ============================================================

def count_body_presence(
    sequence: NTUSkeletonSequence,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for frame in sequence.frames:
        for body in frame.bodies:
            counts[body.body_id] = (
                counts.get(
                    body.body_id,
                    0,
                )
                + 1
            )

    return counts


def calculate_body_motion(
    sequence: NTUSkeletonSequence,
    body_id: str,
) -> float:
    previous_center: Optional[
        np.ndarray
    ] = None

    total_motion = 0.0

    for frame in sequence.frames:
        body = next(
            (
                candidate
                for candidate
                in frame.bodies
                if candidate.body_id
                == body_id
            ),
            None,
        )

        if body is None:
            continue

        arrays = body.joint_arrays()

        color_xy = arrays[
            "color_xy"
        ]

        tracking_state = arrays[
            "tracking_state"
        ]

        valid = (
            np.isfinite(
                color_xy
            ).all(axis=1)
            & (tracking_state > 0)
        )

        if not np.any(valid):
            continue

        center = np.mean(
            color_xy[valid],
            axis=0,
        )

        if previous_center is not None:
            total_motion += float(
                np.linalg.norm(
                    center
                    - previous_center
                )
            )

        previous_center = center

    return total_motion


def choose_primary_body_id(
    sequence: NTUSkeletonSequence,
) -> str:
    presence_counts = (
        count_body_presence(
            sequence
        )
    )

    if not presence_counts:
        raise RuntimeError(
            "No bodies were found in the "
            "Skeleton sequence."
        )

    ranking: list[
        tuple[int, float, str]
    ] = []

    for body_id, presence in (
        presence_counts.items()
    ):
        motion = calculate_body_motion(
            sequence,
            body_id,
        )

        ranking.append(
            (
                presence,
                motion,
                body_id,
            )
        )

    ranking.sort(
        reverse=True
    )

    print("\nBody ranking")
    print("-" * 70)

    for index, (
        presence,
        motion,
        body_id,
    ) in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"body_id={body_id} | "
            f"frames={presence} | "
            f"motion={motion:.2f}"
        )

    selected_body_id = ranking[
        0
    ][2]

    print(
        "\nSelected body ID:\n"
        f"  {selected_body_id}"
    )

    return selected_body_id


def get_body_for_frame(
    sequence: NTUSkeletonSequence,
    frame_index: int,
    body_id: str,
) -> Optional[NTUBody]:
    if (
        frame_index < 0
        or frame_index
        >= sequence.num_frames
    ):
        return None

    frame = sequence.frames[
        frame_index
    ]

    return next(
        (
            body
            for body in frame.bodies
            if body.body_id
            == body_id
        ),
        None,
    )


# ============================================================
# 5. Skeleton validation
# ============================================================

def body_to_arrays(
    body: NTUBody,
    image_width: int,
    image_height: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    arrays = body.joint_arrays()

    color_xy = arrays[
        "color_xy"
    ].astype(
        np.float32
    )

    tracking_state = arrays[
        "tracking_state"
    ].astype(
        np.int8
    )

    inside_image = (
        np.isfinite(
            color_xy
        ).all(axis=1)
        & (
            color_xy[:, 0]
            >= 0
        )
        & (
            color_xy[:, 0]
            < image_width
        )
        & (
            color_xy[:, 1]
            >= 0
        )
        & (
            color_xy[:, 1]
            < image_height
        )
    )

    if ONLY_TRACKED_JOINTS:
        valid = (
            inside_image
            & (
                tracking_state == 2
            )
        )

    else:
        valid = (
            inside_image
            & (
                tracking_state > 0
            )
        )

    return (
        color_xy,
        tracking_state,
        valid,
    )


def calculate_bbox(
    keypoints: np.ndarray,
    valid: np.ndarray,
    image_width: int,
    image_height: int,
) -> Optional[
    tuple[int, int, int, int]
]:
    points = keypoints[
        valid
    ]

    if len(points) == 0:
        return None

    x1 = float(
        points[:, 0].min()
    )

    y1 = float(
        points[:, 1].min()
    )

    x2 = float(
        points[:, 0].max()
    )

    y2 = float(
        points[:, 1].max()
    )

    width = max(
        1.0,
        x2 - x1,
    )

    height = max(
        1.0,
        y2 - y1,
    )

    x1 -= width * BBOX_EXPANSION
    x2 += width * BBOX_EXPANSION

    y1 -= height * BBOX_EXPANSION
    y2 += height * BBOX_EXPANSION

    x1_int = max(
        0,
        int(round(x1)),
    )

    y1_int = max(
        0,
        int(round(y1)),
    )

    x2_int = min(
        image_width - 1,
        int(round(x2)),
    )

    y2_int = min(
        image_height - 1,
        int(round(y2)),
    )

    if (
        x2_int <= x1_int
        or y2_int <= y1_int
    ):
        return None

    return (
        x1_int,
        y1_int,
        x2_int,
        y2_int,
    )


# ============================================================
# 6. Drawing
# ============================================================

def state_point_colour(
    tracking_state: int,
) -> tuple[int, int, int]:
    # BGR
    if tracking_state == 2:
        # Tracked: red
        return (0, 0, 255)

    if tracking_state == 1:
        # Inferred: orange
        return (0, 165, 255)

    # Not tracked
    return (128, 128, 128)


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    tracking_state: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    # Draw bones first
    for joint_a, joint_b in (
        SKELETON_EDGES
    ):
        if not (
            valid[joint_a]
            and valid[joint_b]
        ):
            continue

        point_a = (
            int(round(
                keypoints[
                    joint_a,
                    0,
                ]
            )),
            int(round(
                keypoints[
                    joint_a,
                    1,
                ]
            )),
        )

        point_b = (
            int(round(
                keypoints[
                    joint_b,
                    0,
                ]
            )),
            int(round(
                keypoints[
                    joint_b,
                    1,
                ]
            )),
        )

        both_tracked = (
            tracking_state[
                joint_a
            ] == 2
            and tracking_state[
                joint_b
            ] == 2
        )

        if both_tracked:
            line_colour = (
                0,
                255,
                0,
            )
        else:
            line_colour = (
                0,
                255,
                255,
            )

        cv2.line(
            output,
            point_a,
            point_b,
            line_colour,
            2,
            cv2.LINE_AA,
        )

    # Draw joints
    for joint_index in range(
        NUM_JOINTS
    ):
        if not valid[
            joint_index
        ]:
            continue

        x = int(round(
            keypoints[
                joint_index,
                0,
            ]
        ))

        y = int(round(
            keypoints[
                joint_index,
                1,
            ]
        ))

        state = int(
            tracking_state[
                joint_index
            ]
        )

        colour = (
            state_point_colour(
                state
            )
        )

        if state == 2:
            cv2.circle(
                output,
                (x, y),
                5,
                colour,
                -1,
                cv2.LINE_AA,
            )

        elif state == 1:
            cv2.circle(
                output,
                (x, y),
                6,
                colour,
                2,
                cv2.LINE_AA,
            )

            cv2.line(
                output,
                (x - 4, y - 4),
                (x + 4, y + 4),
                colour,
                1,
                cv2.LINE_AA,
            )

            cv2.line(
                output,
                (x - 4, y + 4),
                (x + 4, y - 4),
                colour,
                1,
                cv2.LINE_AA,
            )

        if DRAW_JOINT_INDEX:
            cv2.putText(
                output,
                str(joint_index),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return output


def draw_information(
    image: np.ndarray,
    frame_index: int,
    skeleton_frame_index: int,
    body_id: str,
    num_bodies: int,
    tracked_count: int,
    inferred_count: int,
    rgb_frame_count: int,
    skeleton_frame_count: int,
) -> np.ndarray:
    output = image.copy()

    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (output.shape[1], 90),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        output,
        0.35,
        0,
        output,
    )

    first_line = (
        f"Sample: {SAMPLE_ID} | "
        f"RGB frame: {frame_index} | "
        f"Skeleton frame: "
        f"{skeleton_frame_index}"
    )

    second_line = (
        f"Body ID: {body_id} | "
        f"Bodies in frame: "
        f"{num_bodies} | "
        f"Tracked: {tracked_count} | "
        f"Inferred: {inferred_count}"
    )

    third_line = (
        f"RGB frames: "
        f"{rgb_frame_count} | "
        f"Skeleton frames: "
        f"{skeleton_frame_count} | "
        "Red=Tracked, Orange=Inferred"
    )

    cv2.putText(
        output,
        first_line,
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        second_line,
        (15, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        third_line,
        (15, 79),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def draw_body_bbox(
    image: np.ndarray,
    bbox: Optional[
        tuple[int, int, int, int]
    ],
) -> np.ndarray:
    output = image.copy()

    if bbox is None:
        return output

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return output


def resize_output(
    image: np.ndarray,
) -> np.ndarray:
    if OUTPUT_SCALE == 1.0:
        return image

    width = int(round(
        image.shape[1]
        * OUTPUT_SCALE
    ))

    height = int(round(
        image.shape[0]
        * OUTPUT_SCALE
    ))

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# 7. Static sample frames
# ============================================================

def build_sample_indices(
    total_frames: int,
) -> set[int]:
    if total_frames <= 0:
        return set()

    sample_count = min(
        NUM_SAMPLE_FRAMES,
        total_frames,
    )

    indices = np.linspace(
        0,
        total_frames - 1,
        sample_count,
    )

    return {
        int(round(index))
        for index in indices
    }


# ============================================================
# 8. Offset comparison panel
# ============================================================

def add_panel_title(
    image: np.ndarray,
    title: str,
) -> np.ndarray:
    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 38),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        title,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def resize_panel(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    source_height, source_width = (
        image.shape[:2]
    )

    scale = min(
        width / source_width,
        height / source_height,
    )

    resized_width = max(
        1,
        int(round(
            source_width * scale
        )),
    )

    resized_height = max(
        1,
        int(round(
            source_height * scale
        )),
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    offset_x = (
        width - resized_width
    ) // 2

    offset_y = (
        height - resized_height
    ) // 2

    canvas[
        offset_y:
        offset_y + resized_height,
        offset_x:
        offset_x + resized_width,
    ] = resized

    return canvas


def create_offset_panel(
    rgb_frame: np.ndarray,
    rgb_frame_index: int,
    sequence: NTUSkeletonSequence,
    body_id: str,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    panel_width = 480
    panel_height = 300

    panels: list[
        np.ndarray
    ] = []

    for offset in FRAME_OFFSETS:
        skeleton_index = (
            rgb_frame_index
            + offset
        )

        panel = rgb_frame.copy()

        body = get_body_for_frame(
            sequence,
            skeleton_index,
            body_id,
        )

        if body is not None:
            (
                keypoints,
                tracking_state,
                valid,
            ) = body_to_arrays(
                body,
                image_width,
                image_height,
            )

            panel = draw_skeleton(
                panel,
                keypoints,
                tracking_state,
                valid,
            )

        title = (
            f"RGB[{rgb_frame_index}] + "
            f"Skeleton[{skeleton_index}] "
            f"(offset {offset:+d})"
        )

        panel = add_panel_title(
            panel,
            title,
        )

        panel = resize_panel(
            panel,
            panel_width,
            panel_height,
        )

        panels.append(
            panel
        )

    # Five panels:
    # first row = -2, -1, 0
    # second row = +1, +2, explanation
    blank_panel = np.zeros(
        (
            panel_height,
            panel_width,
            3,
        ),
        dtype=np.uint8,
    )

    explanation_lines = [
        "Temporal alignment check",
        "",
        "Compare the five panels.",
        "The best alignment should",
        "normally be offset +0.",
        "",
        "If +1 or -1 repeatedly",
        "looks better, RGB and",
        "Skeleton may be shifted.",
    ]

    y = 45

    for line in explanation_lines:
        cv2.putText(
            blank_panel,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 30

    top_row = np.hstack(
        panels[:3]
    )

    bottom_row = np.hstack([
        panels[3],
        panels[4],
        blank_panel,
    ])

    return np.vstack([
        top_row,
        bottom_row,
    ])


# ============================================================
# 9. Video writer
# ============================================================

def create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        frame_size,
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create video:\n"
            f"{output_path}"
        )

    return writer


# ============================================================
# 10. Main
# ============================================================

def main() -> None:
    print("=" * 70)
    print("NTU RGB-Skeleton Alignment Check")
    print("=" * 70)

    rgb_path, skeleton_path = (
        find_sample_files()
    )

    print(
        f"\nSample ID:\n  {SAMPLE_ID}"
    )

    print(
        f"\nRGB video:\n  {rgb_path}"
    )

    print(
        "\nSkeleton file:\n"
        f"  {skeleton_path}"
    )

    sequence = read_skeleton_file(
        skeleton_path
    )

    selected_body_id = (
        choose_primary_body_id(
            sequence
        )
    )

    capture = cv2.VideoCapture(
        str(rgb_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open RGB video:\n"
            f"{rgb_path}"
        )

    rgb_frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    image_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    image_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 0:
        fps = 30.0

    skeleton_frame_count = (
        sequence.num_frames
    )

    total_frames = min(
        rgb_frame_count,
        skeleton_frame_count,
    )

    if MAX_FRAMES is not None:
        total_frames = min(
            total_frames,
            MAX_FRAMES,
        )

    print("\nFrame information")
    print("-" * 70)

    print(
        f"RGB resolution:         "
        f"{image_width} x {image_height}"
    )

    print(
        f"RGB frame count:        "
        f"{rgb_frame_count}"
    )

    print(
        f"Skeleton frame count:   "
        f"{skeleton_frame_count}"
    )

    print(
        f"Difference:             "
        f"{rgb_frame_count - skeleton_frame_count}"
    )

    print(
        f"Frames to process:      "
        f"{total_frames}"
    )

    sample_indices = (
        build_sample_indices(
            total_frames
        )
    )

    output_width = int(round(
        image_width
        * OUTPUT_SCALE
    ))

    output_height = int(round(
        image_height
        * OUTPUT_SCALE
    ))

    alignment_writer = (
        create_video_writer(
            ALIGNMENT_VIDEO_PATH,
            fps,
            (
                output_width,
                output_height,
            ),
        )
    )

    offset_writer: Optional[
        cv2.VideoWriter
    ] = None

    report_rows: list[dict] = []

    missing_body_frames = 0

    print("\nProcessing")
    print("-" * 70)

    for frame_index in range(
        total_frames
    ):
        success, rgb_frame = (
            capture.read()
        )

        if not success:
            print(
                "Stopped because RGB frame "
                f"{frame_index} could not "
                "be read."
            )
            break

        skeleton_frame = (
            sequence.frames[
                frame_index
            ]
        )

        body = get_body_for_frame(
            sequence,
            frame_index,
            selected_body_id,
        )

        tracked_count = 0
        inferred_count = 0
        valid_count = 0
        bbox = None

        visualisation = (
            rgb_frame.copy()
        )

        if body is not None:
            (
                keypoints,
                tracking_state,
                valid,
            ) = body_to_arrays(
                body,
                image_width,
                image_height,
            )

            tracked_count = int(
                np.sum(
                    tracking_state == 2
                )
            )

            inferred_count = int(
                np.sum(
                    tracking_state == 1
                )
            )

            valid_count = int(
                valid.sum()
            )

            bbox = calculate_bbox(
                keypoints,
                valid,
                image_width,
                image_height,
            )

            visualisation = draw_skeleton(
                visualisation,
                keypoints,
                tracking_state,
                valid,
            )

            visualisation = draw_body_bbox(
                visualisation,
                bbox,
            )

        else:
            missing_body_frames += 1

            cv2.putText(
                visualisation,
                "SELECTED BODY NOT PRESENT",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        visualisation = draw_information(
            visualisation,
            frame_index=frame_index,
            skeleton_frame_index=(
                frame_index
            ),
            body_id=selected_body_id,
            num_bodies=len(
                skeleton_frame.bodies
            ),
            tracked_count=(
                tracked_count
            ),
            inferred_count=(
                inferred_count
            ),
            rgb_frame_count=(
                rgb_frame_count
            ),
            skeleton_frame_count=(
                skeleton_frame_count
            ),
        )

        output_frame = resize_output(
            visualisation
        )

        alignment_writer.write(
            output_frame
        )

        if frame_index in sample_indices:
            frame_path = (
                FRAME_DIR
                / (
                    f"frame_"
                    f"{frame_index:05d}.jpg"
                )
            )

            cv2.imwrite(
                str(frame_path),
                visualisation,
            )

        if GENERATE_OFFSET_VIDEO:
            offset_panel = (
                create_offset_panel(
                    rgb_frame=rgb_frame,
                    rgb_frame_index=(
                        frame_index
                    ),
                    sequence=sequence,
                    body_id=(
                        selected_body_id
                    ),
                    image_width=(
                        image_width
                    ),
                    image_height=(
                        image_height
                    ),
                )
            )

            if offset_writer is None:
                panel_height, panel_width = (
                    offset_panel.shape[:2]
                )

                offset_writer = (
                    create_video_writer(
                        OFFSET_VIDEO_PATH,
                        fps,
                        (
                            panel_width,
                            panel_height,
                        ),
                    )
                )

            offset_writer.write(
                offset_panel
            )

        report_rows.append({
            "frame_index": frame_index,
            "selected_body_id": (
                selected_body_id
            ),
            "body_present": (
                body is not None
            ),
            "num_bodies_in_frame": len(
                skeleton_frame.bodies
            ),
            "tracked_joint_count": (
                tracked_count
            ),
            "inferred_joint_count": (
                inferred_count
            ),
            "valid_joint_count": (
                valid_count
            ),
            "bbox_x1": (
                ""
                if bbox is None
                else bbox[0]
            ),
            "bbox_y1": (
                ""
                if bbox is None
                else bbox[1]
            ),
            "bbox_x2": (
                ""
                if bbox is None
                else bbox[2]
            ),
            "bbox_y2": (
                ""
                if bbox is None
                else bbox[3]
            ),
        })

        if (
            frame_index == 0
            or (frame_index + 1)
            % 50 == 0
        ):
            print(
                f"Processed "
                f"{frame_index + 1}/"
                f"{total_frames} frames | "
                f"tracked={tracked_count} | "
                f"inferred={inferred_count} | "
                f"bodies="
                f"{len(skeleton_frame.bodies)}"
            )

    capture.release()
    alignment_writer.release()

    if offset_writer is not None:
        offset_writer.release()

    fieldnames = [
        "frame_index",
        "selected_body_id",
        "body_present",
        "num_bodies_in_frame",
        "tracked_joint_count",
        "inferred_joint_count",
        "valid_joint_count",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]

    with REPORT_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            report_rows
        )

    tracked_values = [
        row[
            "tracked_joint_count"
        ]
        for row in report_rows
        if row[
            "body_present"
        ]
    ]

    inferred_values = [
        row[
            "inferred_joint_count"
        ]
        for row in report_rows
        if row[
            "body_present"
        ]
    ]

    print("\n" + "=" * 70)
    print("Alignment check summary")
    print("=" * 70)

    print(
        f"Processed frames:       "
        f"{len(report_rows)}"
    )

    print(
        f"Missing-body frames:    "
        f"{missing_body_frames}"
    )

    if tracked_values:
        print(
            "Average tracked joints: "
            f"{np.mean(tracked_values):.2f}"
            f"/{NUM_JOINTS}"
        )

    if inferred_values:
        print(
            "Average inferred joints:"
            f" {np.mean(inferred_values):.2f}"
            f"/{NUM_JOINTS}"
        )

    print(
        "\nAlignment video:\n  "
        f"{ALIGNMENT_VIDEO_PATH}"
    )

    if GENERATE_OFFSET_VIDEO:
        print(
            "\nOffset comparison video:\n  "
            f"{OFFSET_VIDEO_PATH}"
        )

    print(
        "\nSample frames:\n  "
        f"{FRAME_DIR}"
    )

    print(
        "\nCSV report:\n  "
        f"{REPORT_CSV_PATH}"
    )

    print("\nInterpretation")
    print("-" * 70)

    print(
        "1. In the alignment video, check "
        "whether the skeleton follows the "
        "same person throughout the video."
    )

    print(
        "2. Check whether shoulders, elbows, "
        "hips, knees and ankles remain on "
        "the correct body regions."
    )

    print(
        "3. Orange joints are inferred by "
        "Kinect and may be less reliable."
    )

    print(
        "4. In the offset video, offset +0 "
        "should usually align best."
    )

    print(
        "5. If offset +1 or -1 consistently "
        "looks better, there may be a frame "
        "alignment error."
    )


if __name__ == "__main__":
    main()