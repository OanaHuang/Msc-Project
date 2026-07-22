# Scripts/NTU_RGBD/17_Generate_MS_Spiking_ResNet50_Heatmap_Video.py

from __future__ import annotations

from pathlib import Path
import csv
import re
import sys
import traceback

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

from Scripts.common.paths import (
    NTU_METADATA_DIR,
    NTU_RGBD_DATASET_DIR,
    NTU_RGBD_OUTPUT_DIR,
)

from Scripts.common.reproducibility import (
    get_device,
)

from Scripts.NTU_RGBD.core import (
    coordinate_visibility,
    extract_primary_pose_sequence,
    read_skeleton_file,
)

from Scripts.NTU_RGBD.datasets.person_crop import (
    crop_and_resize_person,
)

from Scripts.NTU_RGBD.datasets import (
    build_eval_transform,
)

from Scripts.NTU_RGBD.models import (
    build_ms_spiking_resnet50_heatmap,
)


# ============================================================
# 3. Configuration
# ============================================================

# Trained MS-Residual Spiking ResNet50 model.
MODEL_VERSION = "16"

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_JOINTS = 25

# SNN settings must match the training script.
NUM_STEPS = 2
BETA = 0.90
THRESHOLD = 1.0
SURROGATE_SLOPE = 25.0

# Model 16 was trained with skeleton-guided person crops.
PERSON_CROP = True
BBOX_EXPANSION = 0.25

# Raw heatmap peak threshold.
CONFIDENCE_THRESHOLD = 0.02

# 1 = process every frame.
FRAME_STRIDE = 1

# None = retain the source video's effective FPS.
OUTPUT_FPS = None

DEVICE_NAME = None

# Existing non-empty MP4 files are skipped.
SKIP_EXISTING_VIDEOS = True

# Keep False when generating all 344 videos to save disk space.
SAVE_PREDICTION_NPZ = True

# None = all test videos.
# Change to 3 for a quick test.
MAX_TEST_VIDEOS = None


# ============================================================
# 4. Paths
# ============================================================

TEST_CSV = (
    NTU_METADATA_DIR
    / "test_split.csv"
)

RGB_VIDEO_DIR = (
    NTU_RGBD_DATASET_DIR
    / "rgb_videos"
)

SKELETON_DIR = (
    NTU_RGBD_DATASET_DIR
    / "skeletons"
)

MODEL_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / (
        "16_Train_MS_Spiking_ResNet50_"
        "Heatmap_Human_Detection"
    )
)

MODEL_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

OUTPUT_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / f"17_Generate_MP4_MS_Spiking_Model_{MODEL_VERSION}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "generation_summary.csv"
)


# ============================================================
# 5. NTU sample ID pattern
# ============================================================

# Example:
# S001C001P001R001A001
#
# The actual RGB files on the server use lowercase names such as:
# s001c001p005r002a001_rgb.avi
#
# re.IGNORECASE makes both forms match.
NTU_SAMPLE_ID_PATTERN = re.compile(
    r"S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}",
    re.IGNORECASE,
)


def extract_sample_id_from_text(
    value: str,
) -> str | None:
    """
    Extract and normalise an NTU sample ID.

    Examples:
        s001c001p005r002a001_rgb.avi
        -> S001C001P005R002A001

        /path/to/S015C001P017R002A022.skeleton
        -> S015C001P017R002A022
    """
    match = NTU_SAMPLE_ID_PATTERN.search(
        str(value)
    )

    if match is None:
        return None

    return match.group(0).upper()


# ============================================================
# 6. NTU skeleton connections
# ============================================================

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
    (7, 22),

    (20, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 23),
    (11, 24),

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
# 7. Test split loading
# ============================================================

def load_test_rows(
    csv_path: Path,
) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Test split not found: {csv_path}"
        )

    rows: list[dict[str, str]] = []

    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            is_single_person = str(
                row.get(
                    "is_single_person",
                    "",
                )
            ).strip().lower()

            if is_single_person not in {
                "true",
                "1",
                "yes",
            }:
                continue

            raw_sample_id = str(
                row.get(
                    "sample_id",
                    "",
                )
            )

            sample_id = extract_sample_id_from_text(
                raw_sample_id
            )

            if sample_id is None:
                print(
                    "Skipping row with invalid "
                    f"sample_id: {raw_sample_id}"
                )
                continue

            row["sample_id"] = sample_id
            rows.append(row)

    if not rows:
        raise RuntimeError(
            "No valid single-person test samples "
            "were found."
        )

    if MAX_TEST_VIDEOS is not None:
        rows = rows[:MAX_TEST_VIDEOS]

    return rows


# ============================================================
# 8. File indexing
# ============================================================

def build_file_index(
    root: Path,
    allowed_suffixes: set[str],
) -> dict[str, Path]:
    """
    Build:
        normalised NTU sample ID -> file path

    Matching is independent of filename case.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Directory not found: {root}"
        )

    allowed_suffixes = {
        suffix.lower()
        for suffix in allowed_suffixes
    }

    index: dict[str, Path] = {}
    ignored = 0
    duplicates = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_suffixes:
            continue

        sample_id = extract_sample_id_from_text(
            path.name
        )

        if sample_id is None:
            # Try the full path as a fallback.
            sample_id = extract_sample_id_from_text(
                str(path)
            )

        if sample_id is None:
            ignored += 1
            continue

        if sample_id in index:
            duplicates += 1
            continue

        index[sample_id] = path

    print(
        f"  Indexed files: {len(index)}"
    )

    if ignored:
        print(
            f"  Ignored files without NTU ID: "
            f"{ignored}"
        )

    if duplicates:
        print(
            f"  Duplicate sample IDs ignored: "
            f"{duplicates}"
        )

    return index


def find_indexed_file(
    sample_id: str,
    file_index: dict[str, Path],
    file_type: str,
) -> Path:
    normalised_id = extract_sample_id_from_text(
        sample_id
    )

    if normalised_id is None:
        raise ValueError(
            f"Invalid NTU sample ID: {sample_id}"
        )

    path = file_index.get(
        normalised_id
    )

    if path is not None:
        return path

    raise FileNotFoundError(
        f"{file_type} not found for sample: "
        f"{normalised_id}"
    )


# ============================================================
# 9. Checkpoint loading
# ============================================================

def extract_state_dict(
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find a model state dictionary "
        "inside the checkpoint."
    )


def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def load_model(
    model_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: "
            f"{model_path}"
        )

    model = build_ms_spiking_resnet50_heatmap(
        num_joints=NUM_JOINTS,
        num_steps=NUM_STEPS,
        beta=BETA,
        threshold=THRESHOLD,
        surrogate_slope=SURROGATE_SLOPE,
        pretrained=False,
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = remove_module_prefix(
        state_dict
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# 10. Heatmap decoding
# ============================================================

def decode_heatmaps(
    heatmaps: torch.Tensor,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        heatmaps: [1, J, H, W]

    Returns:
        keypoints: [J, 2], model-input coordinates
        confidence: [J], raw heatmap peaks
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            "heatmaps must have shape [B, J, H, W]"
        )

    heatmaps_np = (
        heatmaps[0]
        .detach()
        .cpu()
        .numpy()
    )

    num_joints, height, width = (
        heatmaps_np.shape
    )

    keypoints = np.zeros(
        (num_joints, 2),
        dtype=np.float32,
    )

    confidence = np.zeros(
        (num_joints,),
        dtype=np.float32,
    )

    scale_x = image_size / width
    scale_y = image_size / height

    for joint_index in range(num_joints):
        heatmap = heatmaps_np[joint_index]

        flat_index = int(
            np.argmax(heatmap)
        )

        y, x = np.unravel_index(
            flat_index,
            heatmap.shape,
        )

        confidence[joint_index] = float(
            heatmap[y, x]
        )

        keypoints[joint_index, 0] = (
            (x + 0.5) * scale_x
        )

        keypoints[joint_index, 1] = (
            (y + 0.5) * scale_y
        )

    return keypoints, confidence


# ============================================================
# 11. Coordinate conversion
# ============================================================

def map_crop_keypoints_to_original(
    crop_keypoints: np.ndarray,
    bbox_xyxy: np.ndarray,
    input_size: int,
) -> np.ndarray:
    x1, y1, x2, y2 = (
        bbox_xyxy.astype(np.float32)
    )

    crop_width = max(
        x2 - x1,
        1.0,
    )

    crop_height = max(
        y2 - y1,
        1.0,
    )

    original_keypoints = (
        crop_keypoints.copy()
    )

    original_keypoints[:, 0] = (
        x1
        + crop_keypoints[:, 0]
        * crop_width
        / input_size
    )

    original_keypoints[:, 1] = (
        y1
        + crop_keypoints[:, 1]
        * crop_height
        / input_size
    )

    return original_keypoints


def map_resized_keypoints_to_original(
    resized_keypoints: np.ndarray,
    original_width: int,
    original_height: int,
    input_size: int,
) -> np.ndarray:
    original_keypoints = (
        resized_keypoints.copy()
    )

    original_keypoints[:, 0] *= (
        original_width / input_size
    )

    original_keypoints[:, 1] *= (
        original_height / input_size
    )

    return original_keypoints


# ============================================================
# 12. Drawing
# ============================================================

def point_is_valid(
    point: np.ndarray,
    image_width: int,
    image_height: int,
) -> bool:
    x = float(point[0])
    y = float(point[1])

    return (
        np.isfinite(x)
        and np.isfinite(y)
        and 0 <= x < image_width
        and 0 <= y < image_height
    )


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    point_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    point_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    output = image.copy()

    image_height, image_width = (
        output.shape[:2]
    )

    visibility_bool = np.asarray(
        visibility,
        dtype=bool,
    )

    for joint_a, joint_b in SKELETON_EDGES:
        if not (
            visibility_bool[joint_a]
            and visibility_bool[joint_b]
        ):
            continue

        point_a = keypoints[joint_a]
        point_b = keypoints[joint_b]

        if not point_is_valid(
            point_a,
            image_width,
            image_height,
        ):
            continue

        if not point_is_valid(
            point_b,
            image_width,
            image_height,
        ):
            continue

        cv2.line(
            output,
            tuple(
                np.round(
                    point_a
                ).astype(int)
            ),
            tuple(
                np.round(
                    point_b
                ).astype(int)
            ),
            line_color,
            line_thickness,
            cv2.LINE_AA,
        )

    for joint_index, point in enumerate(
        keypoints
    ):
        if not visibility_bool[joint_index]:
            continue

        if not point_is_valid(
            point,
            image_width,
            image_height,
        ):
            continue

        cv2.circle(
            output,
            tuple(
                np.round(
                    point
                ).astype(int)
            ),
            point_radius,
            point_color,
            -1,
            cv2.LINE_AA,
        )

    return output


def draw_bbox(
    image: np.ndarray,
    bbox_xyxy: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    x1, y1, x2, y2 = (
        np.round(
            bbox_xyxy
        ).astype(int)
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def draw_legend(
    image: np.ndarray,
    model_version: str,
) -> np.ndarray:
    output = image.copy()
    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (340, 82),
        (0, 0, 0),
        -1,
    )

    output = cv2.addWeighted(
        overlay,
        0.55,
        output,
        0.45,
        0,
    )

    # GT: green.
    cv2.line(
        output,
        (25, 32),
        (55, 32),
        (0, 180, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        (40, 32),
        5,
        (0, 255, 0),
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        "Ground Truth",
        (68, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Prediction: red points and yellow lines.
    cv2.line(
        output,
        (25, 64),
        (55, 64),
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        (40, 64),
        5,
        (0, 0, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"Prediction - MS-SNN Model {model_version}",
        (68, 71),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# 13. Generate one sample video
# ============================================================

def generate_sample_video(
    row: dict[str, str],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    rgb_video_index: dict[str, Path],
    skeleton_index: dict[str, Path],
) -> dict[str, object]:
    sample_id = str(
        row["sample_id"]
    ).upper()

    output_video_path = (
        OUTPUT_DIR
        / (
            f"{sample_id}_"
            f"gt_prediction_model_{MODEL_VERSION}.mp4"
        )
    )

    output_npz_path = (
        OUTPUT_DIR
        / (
            f"{sample_id}_"
            f"predictions_model_{MODEL_VERSION}.npz"
        )
    )

    if (
        SKIP_EXISTING_VIDEOS
        and output_video_path.exists()
        and output_video_path.stat().st_size > 0
    ):
        return {
            "sample_id": sample_id,
            "status": "skipped_existing",
            "processed_frames": "",
            "rgb_video": "",
            "skeleton_file": "",
            "output_video": str(
                output_video_path
            ),
            "output_npz": (
                str(output_npz_path)
                if output_npz_path.exists()
                else ""
            ),
            "error": "",
        }

    rgb_video_path = find_indexed_file(
        sample_id=sample_id,
        file_index=rgb_video_index,
        file_type="RGB video",
    )

    skeleton_path = find_indexed_file(
        sample_id=sample_id,
        file_index=skeleton_index,
        file_type="Skeleton file",
    )

    sequence = read_skeleton_file(
        skeleton_path
    )

    pose_sequence = (
        extract_primary_pose_sequence(
            sequence
        )
    )

    pose_frames = len(
        pose_sequence["color_xy"]
    )

    capture = cv2.VideoCapture(
        str(rgb_video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open RGB video: "
            f"{rgb_video_path}"
        )

    writer: cv2.VideoWriter | None = None

    saved_frame_indices: list[int] = []
    saved_predictions: list[np.ndarray] = []
    saved_confidences: list[np.ndarray] = []
    saved_predicted_visibility: list[np.ndarray] = []
    saved_ground_truth: list[np.ndarray] = []
    saved_ground_truth_visibility: list[np.ndarray] = []
    saved_bboxes: list[np.ndarray] = []

    try:
        source_fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        if (
            not np.isfinite(source_fps)
            or source_fps <= 0
        ):
            source_fps = 30.0

        if OUTPUT_FPS is None:
            output_fps = (
                source_fps / FRAME_STRIDE
            )
        else:
            output_fps = float(
                OUTPUT_FPS
            )

        frame_number = 0

        while frame_number < pose_frames:
            success, frame = capture.read()

            if not success:
                break

            current_frame_number = (
                frame_number
            )

            frame_number += 1

            if (
                current_frame_number
                % FRAME_STRIDE
                != 0
            ):
                continue

            frame_height, frame_width = (
                frame.shape[:2]
            )

            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(
                        *"mp4v"
                    ),
                    output_fps,
                    (
                        frame_width,
                        frame_height,
                    ),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "Could not create video writer: "
                        f"{output_video_path}"
                    )

            gt_keypoints = pose_sequence[
                "color_xy"
            ][current_frame_number].copy()

            tracking_state = pose_sequence[
                "tracking_state"
            ][current_frame_number].copy()

            gt_visibility = (
                coordinate_visibility(
                    gt_keypoints,
                    tracking_state=tracking_state,
                    image_size=(
                        frame_width,
                        frame_height,
                    ),
                    include_inferred=False,
                )
            ).astype(np.float32)

            if PERSON_CROP:
                crop_result = (
                    crop_and_resize_person(
                        image=frame,
                        keypoints=gt_keypoints,
                        visibility=gt_visibility,
                        output_size=IMAGE_SIZE,
                        expansion=BBOX_EXPANSION,
                        make_square=True,
                    )
                )

                model_image = (
                    crop_result.image
                )

                model_keypoints = (
                    crop_result.keypoints
                )

                model_visibility = (
                    crop_result.visibility
                )

                person_bbox = (
                    crop_result.bbox_xyxy
                )

            else:
                model_image = frame.copy()

                model_keypoints = (
                    gt_keypoints.copy()
                )

                model_visibility = (
                    gt_visibility.copy()
                )

                person_bbox = np.array(
                    [
                        0.0,
                        0.0,
                        float(frame_width),
                        float(frame_height),
                    ],
                    dtype=np.float32,
                )

            transformed = transform(
                image=model_image,
                keypoints=model_keypoints,
                visibility=model_visibility,
            )

            image_tensor = (
                transformed["image"]
                .unsqueeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.inference_mode():
                predicted_heatmaps = model(
                    image_tensor
                )

            (
                predicted_model_keypoints,
                confidence,
            ) = decode_heatmaps(
                predicted_heatmaps,
                image_size=IMAGE_SIZE,
            )

            predicted_visibility = (
                confidence
                >= CONFIDENCE_THRESHOLD
            ).astype(np.float32)

            if PERSON_CROP:
                predicted_original_keypoints = (
                    map_crop_keypoints_to_original(
                        crop_keypoints=(
                            predicted_model_keypoints
                        ),
                        bbox_xyxy=person_bbox,
                        input_size=IMAGE_SIZE,
                    )
                )

            else:
                predicted_original_keypoints = (
                    map_resized_keypoints_to_original(
                        resized_keypoints=(
                            predicted_model_keypoints
                        ),
                        original_width=(
                            frame_width
                        ),
                        original_height=(
                            frame_height
                        ),
                        input_size=IMAGE_SIZE,
                    )
                )

            # GT: green.
            comparison_frame = draw_skeleton(
                image=frame,
                keypoints=gt_keypoints,
                visibility=gt_visibility,
                point_color=(0, 255, 0),
                line_color=(0, 180, 0),
            )

            # Prediction: red points, yellow lines.
            comparison_frame = draw_skeleton(
                image=comparison_frame,
                keypoints=(
                    predicted_original_keypoints
                ),
                visibility=predicted_visibility,
                point_color=(0, 0, 255),
                line_color=(0, 255, 255),
            )

            if PERSON_CROP:
                comparison_frame = draw_bbox(
                    image=comparison_frame,
                    bbox_xyxy=person_bbox,
                )

            comparison_frame = draw_legend(
                image=comparison_frame,
                model_version=MODEL_VERSION,
            )

            cv2.putText(
                comparison_frame,
                f"Sample: {sample_id}",
                (
                    20,
                    frame_height - 50,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                comparison_frame,
                f"Frame: {current_frame_number}",
                (
                    20,
                    frame_height - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(
                comparison_frame
            )

            saved_frame_indices.append(
                current_frame_number
            )

            if SAVE_PREDICTION_NPZ:
                saved_predictions.append(
                    predicted_original_keypoints
                )

                saved_confidences.append(
                    confidence
                )

                saved_predicted_visibility.append(
                    predicted_visibility
                )

                saved_ground_truth.append(
                    gt_keypoints
                )

                saved_ground_truth_visibility.append(
                    gt_visibility
                )

                saved_bboxes.append(
                    person_bbox
                )

    finally:
        capture.release()

        if writer is not None:
            writer.release()

    if not saved_frame_indices:
        if output_video_path.exists():
            output_video_path.unlink()

        raise RuntimeError(
            "No frames were successfully processed."
        )

    if SAVE_PREDICTION_NPZ:
        np.savez_compressed(
            output_npz_path,

            sample_id=np.array(
                sample_id
            ),

            model_version=np.array(
                MODEL_VERSION
            ),

            frame_indices=np.asarray(
                saved_frame_indices,
                dtype=np.int32,
            ),

            predictions=np.asarray(
                saved_predictions,
                dtype=np.float32,
            ),

            confidences=np.asarray(
                saved_confidences,
                dtype=np.float32,
            ),

            predicted_visibility=np.asarray(
                saved_predicted_visibility,
                dtype=np.float32,
            ),

            ground_truth=np.asarray(
                saved_ground_truth,
                dtype=np.float32,
            ),

            ground_truth_visibility=np.asarray(
                saved_ground_truth_visibility,
                dtype=np.float32,
            ),

            person_bboxes=np.asarray(
                saved_bboxes,
                dtype=np.float32,
            ),

            person_crop=np.array(
                PERSON_CROP
            ),

            bbox_expansion=np.array(
                BBOX_EXPANSION,
                dtype=np.float32,
            ),

            image_size=np.array(
                IMAGE_SIZE,
                dtype=np.int32,
            ),

            heatmap_size=np.array(
                HEATMAP_SIZE,
                dtype=np.int32,
            ),

            num_steps=np.array(
                NUM_STEPS,
                dtype=np.int32,
            ),

            beta=np.array(
                BETA,
                dtype=np.float32,
            ),

            threshold=np.array(
                THRESHOLD,
                dtype=np.float32,
            ),

            surrogate_slope=np.array(
                SURROGATE_SLOPE,
                dtype=np.float32,
            ),
        )

    return {
        "sample_id": sample_id,
        "status": "completed",
        "processed_frames": len(
            saved_frame_indices
        ),
        "rgb_video": str(
            rgb_video_path
        ),
        "skeleton_file": str(
            skeleton_path
        ),
        "output_video": str(
            output_video_path
        ),
        "output_npz": (
            str(output_npz_path)
            if SAVE_PREDICTION_NPZ
            else ""
        ),
        "error": "",
    }


# ============================================================
# 14. Summary CSV
# ============================================================

SUMMARY_FIELDS = [
    "sample_id",
    "status",
    "processed_frames",
    "rgb_video",
    "skeleton_file",
    "output_video",
    "output_npz",
    "error",
]


def write_summary(
    records: list[dict[str, object]],
) -> None:
    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()
        writer.writerows(records)


# ============================================================
# 15. Main
# ============================================================

def main() -> None:
    device = get_device(
        preferred=DEVICE_NAME,
        verbose=True,
    )

    rows = load_test_rows(
        TEST_CSV
    )

    print()
    print("=" * 72)
    print("NTU RGB+D MS-Spiking ResNet50 video generation")
    print("=" * 72)
    print(f"Model version:       {MODEL_VERSION}")
    print(f"Model path:          {MODEL_PATH}")
    print(f"RGB video directory: {RGB_VIDEO_DIR}")
    print(f"Skeleton directory:  {SKELETON_DIR}")
    print(f"Test videos:         {len(rows)}")
    print(f"Person crop:         {PERSON_CROP}")
    print(f"SNN time steps:      {NUM_STEPS}")
    print(f"LIF beta:            {BETA}")
    print(f"LIF threshold:       {THRESHOLD}")
    print(f"Frame stride:        {FRAME_STRIDE}")
    print(f"Skip existing:       {SKIP_EXISTING_VIDEOS}")
    print(f"Save NPZ:            {SAVE_PREDICTION_NPZ}")
    print(f"Output directory:    {OUTPUT_DIR}")
    print("=" * 72)

    print()
    print("Building RGB video index...")

    rgb_video_index = build_file_index(
        root=RGB_VIDEO_DIR,
        allowed_suffixes={
            ".avi",
            ".mp4",
            ".mov",
            ".mkv",
        },
    )

    print()
    print("Building skeleton index...")

    skeleton_index = build_file_index(
        root=SKELETON_DIR,
        allowed_suffixes={
            ".skeleton",
        },
    )

    # Check coverage before loading the model.
    test_sample_ids = {
        str(row["sample_id"]).upper()
        for row in rows
    }

    missing_rgb = sorted(
        test_sample_ids
        - set(rgb_video_index.keys())
    )

    missing_skeleton = sorted(
        test_sample_ids
        - set(skeleton_index.keys())
    )

    print()
    print("=" * 72)
    print("Test-set file coverage")
    print("=" * 72)
    print(
        f"Test sample IDs:     "
        f"{len(test_sample_ids)}"
    )
    print(
        f"Missing RGB videos:  "
        f"{len(missing_rgb)}"
    )
    print(
        f"Missing skeletons:   "
        f"{len(missing_skeleton)}"
    )

    if missing_rgb:
        print(
            "First missing RGB IDs: "
            + ", ".join(
                missing_rgb[:10]
            )
        )

    if missing_skeleton:
        print(
            "First missing skeleton IDs: "
            + ", ".join(
                missing_skeleton[:10]
            )
        )

    print("=" * 72)

    print()
    print("Loading model once...")

    model = load_model(
        model_path=MODEL_PATH,
        device=device,
    )

    transform = build_eval_transform(
        image_size=IMAGE_SIZE,
    )

    records: list[dict[str, object]] = []

    completed = 0
    skipped = 0
    failed = 0

    for row_index, row in enumerate(
        rows,
        start=1,
    ):
        sample_id = str(
            row["sample_id"]
        ).upper()

        print()
        print(
            f"[{row_index}/{len(rows)}] "
            f"{sample_id}"
        )

        try:
            record = generate_sample_video(
                row=row,
                model=model,
                transform=transform,
                device=device,
                rgb_video_index=(
                    rgb_video_index
                ),
                skeleton_index=(
                    skeleton_index
                ),
            )

            status = str(
                record["status"]
            )

            if status == "completed":
                completed += 1

                print(
                    "  Completed: "
                    f"{record['processed_frames']} "
                    "frames"
                )

            elif status == "skipped_existing":
                skipped += 1

                print(
                    "  Skipped: output MP4 "
                    "already exists"
                )

            records.append(
                record
            )

        except Exception as error:
            failed += 1

            error_text = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                f"  Failed: {error_text}"
            )

            traceback.print_exc()

            records.append(
                {
                    "sample_id": sample_id,
                    "status": "failed",
                    "processed_frames": "",
                    "rgb_video": "",
                    "skeleton_file": "",
                    "output_video": "",
                    "output_npz": "",
                    "error": error_text,
                }
            )

        # Save progress after every sample.
        write_summary(
            records
        )

        print(
            f"  Progress: "
            f"completed={completed}, "
            f"skipped={skipped}, "
            f"failed={failed}"
        )

    print()
    print("=" * 72)
    print("Batch video generation finished")
    print("=" * 72)
    print(f"Total test videos: {len(rows)}")
    print(f"Completed:         {completed}")
    print(f"Skipped existing:  {skipped}")
    print(f"Failed:            {failed}")
    print(f"Summary CSV:       {SUMMARY_CSV}")
    print(f"Output directory:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()