# Scripts/common/pose_visualization.py

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


Color = Tuple[int, int, int]


# ============================================================
# 1. Validation helpers
# ============================================================

def _validate_keypoints(
    keypoints: np.ndarray,
) -> np.ndarray:
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if keypoints.ndim != 2:
        raise ValueError(
            "keypoints must have shape [num_joints, 2]"
        )

    if keypoints.shape[1] != 2:
        raise ValueError(
            "keypoints must have shape [num_joints, 2]"
        )

    return keypoints


def _prepare_visibility(
    visibility: Optional[np.ndarray],
    num_joints: int,
) -> np.ndarray:
    if visibility is None:
        return np.ones(
            num_joints,
            dtype=bool,
        )

    visibility = np.asarray(
        visibility,
    ).reshape(-1)

    if len(visibility) != num_joints:
        raise ValueError(
            "visibility length must match number of joints"
        )

    return visibility > 0


def _valid_point(
    point: np.ndarray,
    visible: bool,
    image_width: int,
    image_height: int,
) -> bool:
    if not visible:
        return False

    x, y = point

    if not np.isfinite(x) or not np.isfinite(y):
        return False

    return (
        0 <= x < image_width
        and 0 <= y < image_height
    )


# ============================================================
# 2. Draw keypoints
# ============================================================

def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    color: Color = (0, 255, 0),
    radius: int = 4,
    thickness: int = -1,
    copy_image: bool = True,
) -> np.ndarray:
    """
    Draw 2D pose keypoints on an image.

    OpenCV uses BGR colors.
    """
    if image is None or image.size == 0:
        raise ValueError(
            "Input image is empty"
        )

    keypoints = _validate_keypoints(keypoints)

    visible = _prepare_visibility(
        visibility,
        num_joints=len(keypoints),
    )

    output = (
        image.copy()
        if copy_image
        else image
    )

    height, width = output.shape[:2]

    for joint_index, point in enumerate(keypoints):
        if not _valid_point(
            point,
            visible[joint_index],
            width,
            height,
        ):
            continue

        x, y = np.rint(point).astype(int)

        cv2.circle(
            output,
            (x, y),
            radius,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    return output


# ============================================================
# 3. Draw skeleton
# ============================================================

def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    skeleton_edges: Sequence[Tuple[int, int]],
    visibility: Optional[np.ndarray] = None,
    joint_color: Color = (0, 255, 0),
    bone_color: Color = (0, 255, 0),
    joint_radius: int = 4,
    bone_thickness: int = 2,
    copy_image: bool = True,
) -> np.ndarray:
    """
    Draw keypoints and skeleton edges.
    """
    if image is None or image.size == 0:
        raise ValueError(
            "Input image is empty"
        )

    keypoints = _validate_keypoints(keypoints)

    visible = _prepare_visibility(
        visibility,
        num_joints=len(keypoints),
    )

    output = (
        image.copy()
        if copy_image
        else image
    )

    height, width = output.shape[:2]

    # Draw bones first so joints remain visible on top.
    for start_joint, end_joint in skeleton_edges:
        if not (
            0 <= start_joint < len(keypoints)
            and 0 <= end_joint < len(keypoints)
        ):
            raise IndexError(
                f"Invalid skeleton edge: "
                f"({start_joint}, {end_joint})"
            )

        start_point = keypoints[start_joint]
        end_point = keypoints[end_joint]

        start_valid = _valid_point(
            start_point,
            visible[start_joint],
            width,
            height,
        )

        end_valid = _valid_point(
            end_point,
            visible[end_joint],
            width,
            height,
        )

        if not start_valid or not end_valid:
            continue

        x1, y1 = np.rint(
            start_point
        ).astype(int)

        x2, y2 = np.rint(
            end_point
        ).astype(int)

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            bone_color,
            bone_thickness,
            lineType=cv2.LINE_AA,
        )

    output = draw_keypoints(
        image=output,
        keypoints=keypoints,
        visibility=visible,
        color=joint_color,
        radius=joint_radius,
        thickness=-1,
        copy_image=False,
    )

    return output


# ============================================================
# 4. Draw GT and prediction
# ============================================================

def draw_gt_and_prediction(
    image: np.ndarray,
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
    skeleton_edges: Sequence[Tuple[int, int]],
    gt_visibility: Optional[np.ndarray] = None,
    pred_visibility: Optional[np.ndarray] = None,
    gt_color: Color = (0, 255, 0),
    pred_color: Color = (0, 0, 255),
    joint_radius: int = 4,
    bone_thickness: int = 2,
    add_legend: bool = True,
) -> np.ndarray:
    """
    Draw ground truth and prediction on the same frame.

    Default:
    - Ground truth: green
    - Prediction: red
    """
    output = image.copy()

    output = draw_skeleton(
        image=output,
        keypoints=gt_keypoints,
        skeleton_edges=skeleton_edges,
        visibility=gt_visibility,
        joint_color=gt_color,
        bone_color=gt_color,
        joint_radius=joint_radius,
        bone_thickness=bone_thickness,
        copy_image=False,
    )

    output = draw_skeleton(
        image=output,
        keypoints=pred_keypoints,
        skeleton_edges=skeleton_edges,
        visibility=pred_visibility,
        joint_color=pred_color,
        bone_color=pred_color,
        joint_radius=joint_radius,
        bone_thickness=bone_thickness,
        copy_image=False,
    )

    if add_legend:
        output = add_pose_legend(
            output,
            entries=[
                ("Ground Truth", gt_color),
                ("Prediction", pred_color),
            ],
        )

    return output


# ============================================================
# 5. Add legend
# ============================================================

def add_pose_legend(
    image: np.ndarray,
    entries: Sequence[Tuple[str, Color]],
    origin: Tuple[int, int] = (20, 30),
    line_spacing: int = 28,
    marker_radius: int = 6,
    font_scale: float = 0.65,
    font_thickness: int = 2,
    background: bool = True,
) -> np.ndarray:
    """
    Add a pose-color legend to an image.
    """
    output = image.copy()

    start_x, start_y = origin

    if background and entries:
        max_text_width = 0

        for label, _ in entries:
            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )
            max_text_width = max(
                max_text_width,
                text_size[0],
            )

        box_width = max_text_width + 60
        box_height = line_spacing * len(entries) + 14

        overlay = output.copy()

        cv2.rectangle(
            overlay,
            (start_x - 10, start_y - 22),
            (
                start_x - 10 + box_width,
                start_y - 22 + box_height,
            ),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.addWeighted(
            overlay,
            0.55,
            output,
            0.45,
            0,
            output,
        )

    for index, (label, color) in enumerate(entries):
        y = start_y + index * line_spacing

        cv2.circle(
            output,
            (start_x, y - 5),
            marker_radius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            output,
            label,
            (start_x + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return output


# ============================================================
# 6. Draw frame information
# ============================================================

def add_frame_information(
    image: np.ndarray,
    frame_index: Optional[int] = None,
    sample_id: Optional[str] = None,
    fps: Optional[float] = None,
    origin: Tuple[int, int] = (20, 30),
) -> np.ndarray:
    """
    Add frame metadata to the image.
    """
    output = image.copy()

    labels = []

    if sample_id is not None:
        labels.append(f"Sample: {sample_id}")

    if frame_index is not None:
        labels.append(f"Frame: {frame_index}")

    if fps is not None:
        labels.append(f"FPS: {fps:.2f}")

    for line_index, label in enumerate(labels):
        x = origin[0]
        y = origin[1] + line_index * 26

        cv2.putText(
            output,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return output


# ============================================================
# 7. Bounding box from keypoints
# ============================================================

def keypoints_to_bbox(
    keypoints: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    padding: float = 0.1,
    image_size: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Estimate a bounding box from visible keypoints.

    Returns
    -------
    tuple or None
        (x_min, y_min, x_max, y_max)
    """
    keypoints = _validate_keypoints(keypoints)

    visible = _prepare_visibility(
        visibility,
        num_joints=len(keypoints),
    )

    finite = np.isfinite(keypoints).all(axis=1)
    valid = visible & finite

    valid_keypoints = keypoints[valid]

    if len(valid_keypoints) == 0:
        return None

    x_min = float(valid_keypoints[:, 0].min())
    y_min = float(valid_keypoints[:, 1].min())
    x_max = float(valid_keypoints[:, 0].max())
    y_max = float(valid_keypoints[:, 1].max())

    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)

    x_padding = width * padding
    y_padding = height * padding

    x_min -= x_padding
    y_min -= y_padding
    x_max += x_padding
    y_max += y_padding

    if image_size is not None:
        image_width, image_height = image_size

        x_min = np.clip(
            x_min,
            0,
            image_width - 1,
        )
        x_max = np.clip(
            x_max,
            0,
            image_width - 1,
        )

        y_min = np.clip(
            y_min,
            0,
            image_height - 1,
        )
        y_max = np.clip(
            y_max,
            0,
            image_height - 1,
        )

    return (
        int(round(x_min)),
        int(round(y_min)),
        int(round(x_max)),
        int(round(y_max)),
    )


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Color = (255, 0, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """
    Draw a bounding box.
    """
    output = image.copy()

    x_min, y_min, x_max, y_max = bbox

    cv2.rectangle(
        output,
        (x_min, y_min),
        (x_max, y_max),
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )

    if label:
        cv2.putText(
            output,
            label,
            (x_min, max(y_min - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    return output