# Scripts/common/video_utils.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 1. Video information structure
# ============================================================

@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float

    @property
    def frame_size(self) -> Tuple[int, int]:
        """
        OpenCV frame size format: (width, height).
        """
        return self.width, self.height


# ============================================================
# 2. Open video
# ============================================================

def open_video(video_path: Path | str) -> cv2.VideoCapture:
    """
    Open a video and verify that OpenCV can read it.

    The caller is responsible for calling cap.release().
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file does not exist: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"OpenCV could not open video: {video_path}"
        )

    return cap


# ============================================================
# 3. Read video metadata
# ============================================================

def get_video_info(video_path: Path | str) -> VideoInfo:
    """
    Read video metadata.

    Returns
    -------
    VideoInfo
        Frame count, FPS, resolution, and duration.
    """
    video_path = Path(video_path)
    cap = open_video(video_path)

    try:
        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if fps > 0:
            duration_seconds = frame_count / fps
        else:
            duration_seconds = 0.0

        return VideoInfo(
            path=video_path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
        )

    finally:
        cap.release()


# ============================================================
# 4. Read one frame
# ============================================================

def read_video_frame(
    video_path: Path | str,
    frame_index: int,
    convert_to_rgb: bool = False,
) -> np.ndarray:
    """
    Read a specific frame from a video.

    Parameters
    ----------
    video_path:
        Input video.
    frame_index:
        Zero-based frame index.
    convert_to_rgb:
        Convert OpenCV BGR image to RGB.

    Returns
    -------
    np.ndarray
        Frame with shape [H, W, 3].
    """
    if frame_index < 0:
        raise ValueError(
            f"frame_index must be non-negative, got {frame_index}"
        )

    cap = open_video(video_path)

    try:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = cap.read()

        if not success or frame is None:
            raise RuntimeError(
                f"Could not read frame {frame_index} "
                f"from video: {video_path}"
            )

        if convert_to_rgb:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

        return frame

    finally:
        cap.release()


# ============================================================
# 5. Iterate through video
# ============================================================

def iterate_video_frames(
    video_path: Path | str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    frame_stride: int = 1,
    convert_to_rgb: bool = False,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Iterate through video frames without loading the entire video.

    Yields
    ------
    tuple[int, np.ndarray]
        Zero-based frame index and frame image.
    """
    if start_frame < 0:
        raise ValueError(
            "start_frame must be non-negative"
        )

    if frame_stride <= 0:
        raise ValueError(
            "frame_stride must be greater than zero"
        )

    if end_frame is not None and end_frame <= start_frame:
        raise ValueError(
            "end_frame must be greater than start_frame"
        )

    cap = open_video(video_path)

    try:
        cap.set(
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

            success, frame = cap.read()

            if not success or frame is None:
                break

            if (
                frame_index - start_frame
            ) % frame_stride == 0:

                if convert_to_rgb:
                    frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )

                yield frame_index, frame

            frame_index += 1

    finally:
        cap.release()


# ============================================================
# 6. Video writer
# ============================================================

def create_video_writer(
    output_path: Path | str,
    fps: float,
    frame_size: Tuple[int, int],
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    """
    Create an OpenCV video writer.

    Parameters
    ----------
    output_path:
        Output video path.
    fps:
        Output frames per second.
    frame_size:
        Tuple in OpenCV order: (width, height).
    codec:
        Four-character OpenCV codec.

    Returns
    -------
    cv2.VideoWriter
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fps <= 0:
        raise ValueError(
            f"fps must be positive, got {fps}"
        )

    width, height = frame_size

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid frame size: {frame_size}"
        )

    if len(codec) != 4:
        raise ValueError(
            "codec must contain exactly four characters"
        )

    fourcc = cv2.VideoWriter_fourcc(*codec)

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Could not create video writer: {output_path}"
        )

    return writer


# ============================================================
# 7. Video validation
# ============================================================

def validate_video(
    video_path: Path | str,
    test_first_frame: bool = True,
) -> Tuple[bool, str]:
    """
    Validate that a video exists and can be read.

    Returns
    -------
    tuple[bool, str]
        Validation result and human-readable message.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return False, f"File not found: {video_path}"

    try:
        info = get_video_info(video_path)
    except Exception as exc:
        return False, str(exc)

    if info.frame_count <= 0:
        return False, "Video reports zero frames"

    if info.width <= 0 or info.height <= 0:
        return False, "Video has invalid resolution"

    if info.fps <= 0:
        return False, "Video has invalid FPS"

    if test_first_frame:
        try:
            frame = read_video_frame(
                video_path,
                frame_index=0,
            )
        except Exception as exc:
            return False, str(exc)

        if frame.size == 0:
            return False, "First frame is empty"

    message = (
        f"Valid video | "
        f"frames={info.frame_count}, "
        f"fps={info.fps:.2f}, "
        f"size={info.width}x{info.height}, "
        f"duration={info.duration_seconds:.2f}s"
    )

    return True, message


# ============================================================
# 8. Resize utilities
# ============================================================

def resize_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize frame.

    Parameters
    ----------
    frame:
        Input image.
    target_size:
        OpenCV order: (width, height).
    """
    if frame is None or frame.size == 0:
        raise ValueError(
            "Input frame is empty"
        )

    width, height = target_size

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid target size: {target_size}"
        )

    return cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def scale_keypoints(
    keypoints: np.ndarray,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Scale 2D keypoints between image resolutions.

    Parameters
    ----------
    keypoints:
        Array with shape [..., 2].
    source_size:
        (width, height).
    target_size:
        (width, height).
    """
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if keypoints.shape[-1] != 2:
        raise ValueError(
            "keypoints must have final dimension of size 2"
        )

    source_width, source_height = source_size
    target_width, target_height = target_size

    if source_width <= 0 or source_height <= 0:
        raise ValueError(
            f"Invalid source size: {source_size}"
        )

    scale_x = target_width / source_width
    scale_y = target_height / source_height

    scaled = keypoints.copy()
    scaled[..., 0] *= scale_x
    scaled[..., 1] *= scale_y

    return scaled


if __name__ == "__main__":
    print(
        "video_utils.py contains shared video "
        "reading and writing functions."
    )