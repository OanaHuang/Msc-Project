# Scripts/NTU_RGBD/core/rgb_reader.py

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from .filename_parser import get_sample_id


# ============================================================
# 1. Open RGB video
# ============================================================

def open_rgb_video(
    video_path: str | Path,
) -> cv2.VideoCapture:
    """
    Open an NTU RGB video.

    The caller must release the returned VideoCapture object.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"RGB video does not exist: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            f"OpenCV could not open RGB video: {video_path}"
        )

    return capture


# ============================================================
# 2. Locate RGB video
# ============================================================

def find_rgb_video(
    rgb_root: str | Path,
    sample_id: str,
    recursive: bool = True,
) -> Path:
    """
    Find an RGB video using an NTU sample ID.

    Accepted filenames include:

        S001C001P001R001A001_rgb.avi
        S001C001P001R001A001.avi
        S001C001P001R001A001_rgb.mp4
    """
    rgb_root = Path(rgb_root)

    if not rgb_root.exists():
        raise FileNotFoundError(
            f"RGB root does not exist: {rgb_root}"
        )

    canonical_id = get_sample_id(
        sample_id
    )

    filename_patterns = (
        f"{canonical_id}_rgb.avi",
        f"{canonical_id}.avi",
        f"{canonical_id}_rgb.mp4",
        f"{canonical_id}.mp4",
    )

    matches: list[Path] = []

    for pattern in filename_patterns:
        if recursive:
            matches.extend(
                rgb_root.rglob(pattern)
            )
        else:
            matches.extend(
                rgb_root.glob(pattern)
            )

    matches = sorted(
        set(matches)
    )

    if not matches:
        raise FileNotFoundError(
            f"No RGB video found for sample "
            f"{canonical_id} under {rgb_root}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple RGB videos found for "
            f"{canonical_id}: {matches}"
        )

    return matches[0]


# ============================================================
# 3. Read metadata
# ============================================================

def get_rgb_video_info(
    video_path: str | Path,
) -> dict[str, int | float | str]:
    """
    Read RGB video metadata.

    Returns
    -------
    dict
        Contains path, sample ID, frame count, FPS,
        width, height, and duration.
    """
    video_path = Path(video_path)

    capture = open_rgb_video(
        video_path
    )

    try:
        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        duration_seconds = (
            frame_count / fps
            if fps > 0
            else 0.0
        )

        return {
            "path": str(video_path),
            "sample_id": get_sample_id(
                video_path.name
            ),
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration_seconds,
        }

    finally:
        capture.release()


# ============================================================
# 4. Read one frame
# ============================================================

def read_rgb_frame(
    video_path: str | Path,
    frame_index: int,
    convert_to_rgb: bool = False,
) -> np.ndarray:
    """
    Read one frame from an RGB video.

    Parameters
    ----------
    frame_index:
        Zero-based frame index.
    convert_to_rgb:
        Convert OpenCV BGR output to RGB.
    """
    if frame_index < 0:
        raise ValueError(
            "frame_index must be non-negative"
        )

    capture = open_rgb_video(
        video_path
    )

    try:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = capture.read()

        if not success or frame is None:
            raise RuntimeError(
                f"Could not read frame {frame_index} "
                f"from {video_path}"
            )

        if convert_to_rgb:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

        return frame

    finally:
        capture.release()


# ============================================================
# 5. Iterate through frames
# ============================================================

def iterate_rgb_frames(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    frame_stride: int = 1,
    convert_to_rgb: bool = False,
) -> Generator[
    Tuple[int, np.ndarray],
    None,
    None,
]:
    """
    Iterate through an NTU RGB video without loading it all
    into memory.
    """
    if start_frame < 0:
        raise ValueError(
            "start_frame must be non-negative"
        )

    if frame_stride <= 0:
        raise ValueError(
            "frame_stride must be positive"
        )

    if (
        end_frame is not None
        and end_frame <= start_frame
    ):
        raise ValueError(
            "end_frame must be greater than start_frame"
        )

    capture = open_rgb_video(
        video_path
    )

    try:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame,
        )

        frame_index = start_frame

        while True:
            if (
                end_frame is not None
                and frame_index >= end_frame
            ):
                break

            success, frame = capture.read()

            if not success or frame is None:
                break

            relative_index = (
                frame_index - start_frame
            )

            if relative_index % frame_stride == 0:
                if convert_to_rgb:
                    frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )

                yield frame_index, frame

            frame_index += 1

    finally:
        capture.release()


# ============================================================
# 6. Validation
# ============================================================

def validate_rgb_video(
    video_path: str | Path,
    test_first_frame: bool = True,
) -> tuple[bool, str]:
    """
    Check whether an RGB video can be opened and read.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return (
            False,
            f"File does not exist: {video_path}",
        )

    try:
        info = get_rgb_video_info(
            video_path
        )
    except Exception as error:
        return False, str(error)

    if info["frame_count"] <= 0:
        return (
            False,
            "Video reports zero frames",
        )

    if info["fps"] <= 0:
        return (
            False,
            "Video reports invalid FPS",
        )

    if (
        info["width"] <= 0
        or info["height"] <= 0
    ):
        return (
            False,
            "Video reports invalid resolution",
        )

    if test_first_frame:
        try:
            frame = read_rgb_frame(
                video_path,
                frame_index=0,
            )
        except Exception as error:
            return False, str(error)

        if frame.size == 0:
            return (
                False,
                "First frame is empty",
            )

    message = (
        f"Valid RGB video | "
        f"frames={info['frame_count']} | "
        f"fps={info['fps']:.2f} | "
        f"resolution="
        f"{info['width']}x{info['height']} | "
        f"duration="
        f"{info['duration_seconds']:.2f}s"
    )

    return True, message