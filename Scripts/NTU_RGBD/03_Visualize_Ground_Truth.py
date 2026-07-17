# Scripts/NTU_RGBD/03_Visualize_Ground_Truth.py

from __future__ import annotations

from pathlib import Path
import csv
import sys

import cv2
import numpy as np


# ============================================================
# 1. Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. Imports
# ============================================================

from Scripts.common.paths import (
    NTU_METADATA_DIR,
    NTU_RGBD_OUTPUT_DIR,
)

from Scripts.common.pose_visualization import (
    draw_skeleton,
    add_frame_information,
    add_pose_legend,
)

from Scripts.NTU_RGBD.core import (
    NTU_SKELETON_EDGES,
    coordinate_visibility,
    extract_primary_pose_sequence,
    read_skeleton_file,
)


# ============================================================
# 3. Config
# ============================================================

MATCHED_CSV_PATH = (
    NTU_METADATA_DIR
    / "matched_samples.csv"
)

OUTPUT_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / "03_Visualize_Ground_Truth"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# None means use the first valid sample in matched_samples.csv.
TARGET_SAMPLE_ID = "S015C001P007R001A010"

# Start and end use zero-based frame indices.
START_FRAME = 0
END_FRAME = None

FRAME_STRIDE = 1

# Output video size. Set None to keep the original RGB resolution.
OUTPUT_WIDTH = 960

JOINT_RADIUS = 4
BONE_THICKNESS = 2

# OpenCV BGR.
GT_COLOR = (0, 255, 0)

OUTPUT_FPS = None


# ============================================================
# 4. Metadata
# ============================================================

def load_matched_samples(
    csv_path: Path,
) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Matched CSV not found: {csv_path}\n"
            "Run 02_Match_RGB_Skeleton.py first."
        )

    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if not rows:
        raise RuntimeError(
            f"No rows found in {csv_path}"
        )

    return rows


def select_sample(
    rows: list[dict[str, str]],
    target_sample_id: str | None,
) -> dict[str, str]:
    if target_sample_id is None:
        return rows[0]

    for row in rows:
        if row["sample_id"] == target_sample_id:
            return row

    raise ValueError(
        f"Sample ID not found in matched CSV: "
        f"{target_sample_id}"
    )


# ============================================================
# 5. Video output helper
# ============================================================

def resize_frame_and_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    target_width: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if target_width is None:
        return frame, keypoints

    original_height, original_width = frame.shape[:2]

    if target_width <= 0:
        raise ValueError(
            "OUTPUT_WIDTH must be positive or None"
        )

    scale = target_width / original_width

    target_height = int(
        round(original_height * scale)
    )

    resized_frame = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )

    resized_keypoints = (
        keypoints.astype(np.float32).copy()
    )

    resized_keypoints[:, 0] *= scale
    resized_keypoints[:, 1] *= scale

    return resized_frame, resized_keypoints


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    rows = load_matched_samples(
        MATCHED_CSV_PATH
    )

    sample = select_sample(
        rows,
        TARGET_SAMPLE_ID,
    )

    sample_id = sample["sample_id"]
    rgb_path = Path(sample["rgb_path"])
    skeleton_path = Path(
        sample["skeleton_path"]
    )

    print("=" * 70)
    print("NTU RGB+D ground-truth visualization")
    print("=" * 70)
    print(f"Sample ID:      {sample_id}")
    print(f"RGB video:      {rgb_path}")
    print(f"Skeleton file:  {skeleton_path}")

    skeleton_sequence = read_skeleton_file(
        skeleton_path
    )

    pose_sequence = (
        extract_primary_pose_sequence(
            skeleton_sequence
        )
    )

    capture = cv2.VideoCapture(
        str(rgb_path)
    )

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            f"Could not open RGB video: "
            f"{rgb_path}"
        )

    rgb_frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    rgb_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
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

    usable_frame_count = min(
        rgb_frame_count,
        skeleton_sequence.num_frames,
    )

    actual_end_frame = (
        usable_frame_count
        if END_FRAME is None
        else min(
            END_FRAME,
            usable_frame_count,
        )
    )

    if not (
        0 <= START_FRAME
        < actual_end_frame
    ):
        capture.release()

        raise ValueError(
            f"Invalid frame range: "
            f"{START_FRAME} to {actual_end_frame}"
        )

    output_fps = (
        rgb_fps / FRAME_STRIDE
        if OUTPUT_FPS is None
        else OUTPUT_FPS
    )

    if output_fps <= 0:
        output_fps = 30.0

    if OUTPUT_WIDTH is None:
        output_size = (
            original_width,
            original_height,
        )
    else:
        scale = (
            OUTPUT_WIDTH
            / original_width
        )

        output_size = (
            OUTPUT_WIDTH,
            int(
                round(
                    original_height
                    * scale
                )
            ),
        )

    output_video_path = (
        OUTPUT_DIR
        / f"{sample_id}_gt_pose.mp4"
    )

    first_frame_path = (
        OUTPUT_DIR
        / f"{sample_id}_first_frame.jpg"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        output_fps,
        output_size,
    )

    if not writer.isOpened():
        capture.release()
        writer.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{output_video_path}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        START_FRAME,
    )

    written_frames = 0

    try:
        for frame_index in range(
            START_FRAME,
            actual_end_frame,
        ):
            success, frame = capture.read()

            if not success or frame is None:
                print(
                    f"Stopped: could not read "
                    f"frame {frame_index}"
                )
                break

            if (
                frame_index - START_FRAME
            ) % FRAME_STRIDE != 0:
                continue

            keypoints = pose_sequence[
                "color_xy"
            ][frame_index].copy()

            tracking_state = pose_sequence[
                "tracking_state"
            ][frame_index].copy()

            visibility = (
                coordinate_visibility(
                    keypoints,
                    tracking_state=(
                        tracking_state
                    ),
                    image_size=(
                        frame.shape[1],
                        frame.shape[0],
                    ),
                    include_inferred=True,
                )
            )

            frame, keypoints = (
                resize_frame_and_keypoints(
                    frame,
                    keypoints,
                    OUTPUT_WIDTH,
                )
            )

            output_frame = draw_skeleton(
                image=frame,
                keypoints=keypoints,
                skeleton_edges=(
                    NTU_SKELETON_EDGES
                ),
                visibility=visibility,
                joint_color=GT_COLOR,
                bone_color=GT_COLOR,
                joint_radius=(
                    JOINT_RADIUS
                ),
                bone_thickness=(
                    BONE_THICKNESS
                ),
                copy_image=True,
            )

            output_frame = (
                add_pose_legend(
                    output_frame,
                    entries=[
                        (
                            "Ground Truth",
                            GT_COLOR,
                        )
                    ],
                )
            )

            output_frame = (
                add_frame_information(
                    output_frame,
                    frame_index=(
                        frame_index
                    ),
                    sample_id=sample_id,
                    fps=rgb_fps,
                    origin=(20, 100),
                )
            )

            writer.write(
                output_frame
            )

            if written_frames == 0:
                cv2.imwrite(
                    str(first_frame_path),
                    output_frame,
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

    print()
    print("=" * 70)
    print("Visualization finished")
    print("=" * 70)
    print(
        f"Written frames: "
        f"{written_frames}"
    )
    print(
        f"Output video: "
        f"{output_video_path}"
    )
    print(
        f"Preview image: "
        f"{first_frame_path}"
    )


if __name__ == "__main__":
    main()