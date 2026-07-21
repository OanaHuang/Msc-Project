# Scripts/NTU_RGBD/13_Generate_MP4_Heatmap.py

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
# 2. Imports
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
    build_resnet50_heatmap,
)


# ============================================================
# 3. Configuration
# ============================================================

# Choose:
# "06" = full-frame ResNet50
# "12" = skeleton-guided person crop ResNet50
MODEL_VERSION = "12"

# Select one video from the test split.
# Set to None to automatically use the first valid test video.
SAMPLE_ID = None

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_JOINTS = 25

PERSON_CROP = MODEL_VERSION == "12"
BBOX_EXPANSION = 0.25

CONFIDENCE_THRESHOLD = 0.02

# Use every frame when generating the video.
FRAME_STRIDE = 1

# Output FPS. Set None to use the original metadata FPS if available.
OUTPUT_FPS = 30.0

DEVICE_NAME = None


# ============================================================
# 4. Paths
# ============================================================

TEST_CSV = (
    NTU_METADATA_DIR
    / "test_split.csv"
)

EXTRACTED_FRAMES_DIR = (
    NTU_RGBD_DATASET_DIR
    / "extracted_frames"
)

if MODEL_VERSION == "06":
    MODEL_DIR = (
        NTU_RGBD_OUTPUT_DIR
        / "06_Train_ResNet50_Heatmap"
    )

elif MODEL_VERSION == "12":
    MODEL_DIR = (
        NTU_RGBD_OUTPUT_DIR
        / "12_Train_ResNet50_Heatmap_Human_Detection"
    )
    
else:
    raise ValueError(
        "MODEL_VERSION must be '06' or '12'"
    )

MODEL_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

OUTPUT_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / f"13_Generate_MP4_Heatmap_Model_{MODEL_VERSION}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_VIDEO_PATH = (
    OUTPUT_DIR
    / f"ntu_prediction_model_{MODEL_VERSION}.mp4"
)

OUTPUT_NPZ_PATH = (
    OUTPUT_DIR
    / f"ntu_predictions_model_{MODEL_VERSION}.npz"
)


# ============================================================
# 5. NTU skeleton connections
# ============================================================

# Zero-based NTU RGB+D 25-joint skeleton edges.
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
# 6. Metadata
# ============================================================

def load_test_row(
    csv_path: Path,
    requested_sample_id: str | None,
) -> dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Test split not found: {csv_path}"
        )

    rows = []

    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            single_person_value = str(
                row.get(
                    "is_single_person",
                    "",
                )
            ).strip().lower()

            if single_person_value not in {
                "true",
                "1",
                "yes",
            }:
                continue

            rows.append(row)

    if not rows:
        raise RuntimeError(
            "No valid single-person samples "
            "were found in the test split."
        )

    if requested_sample_id is None:
        return rows[0]

    for row in rows:
        if str(row["sample_id"]) == requested_sample_id:
            return row

    raise ValueError(
        f"Sample ID not found in test split: "
        f"{requested_sample_id}"
    )


# ============================================================
# 7. Checkpoint loading
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

        if all(
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
    cleaned = {}

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

    model = build_resnet50_heatmap(
        num_joints=NUM_JOINTS,
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
# 8. Heatmap decoding
# ============================================================

def decode_heatmaps(
    heatmaps: torch.Tensor,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        heatmaps: [1, J, H, W]

    Returns:
        keypoints: [J, 2] in model-input coordinates
        confidence: [J]
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

        confidence[joint_index] = (
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
# 9. Coordinate conversion
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
# 10. Drawing
# ============================================================

def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    point_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    output = image.copy()

    visibility = np.asarray(
        visibility,
        dtype=bool,
    )

    for joint_a, joint_b in SKELETON_EDGES:
        if not (
            visibility[joint_a]
            and visibility[joint_b]
        ):
            continue

        point_a = tuple(
            np.round(
                keypoints[joint_a]
            ).astype(int)
        )

        point_b = tuple(
            np.round(
                keypoints[joint_b]
            ).astype(int)
        )

        cv2.line(
            output,
            point_a,
            point_b,
            line_color,
            2,
            cv2.LINE_AA,
        )

    for joint_index, point in enumerate(
        keypoints
    ):
        if not visibility[joint_index]:
            continue

        center = tuple(
            np.round(point).astype(int)
        )

        cv2.circle(
            output,
            center,
            4,
            point_color,
            -1,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        point_color,
        2,
        cv2.LINE_AA,
    )

    return output


def draw_bbox(
    image: np.ndarray,
    bbox_xyxy: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    x1, y1, x2, y2 = (
        bbox_xyxy.astype(int)
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# 11. Main
# ============================================================

def main() -> None:
    device = get_device(
        preferred=DEVICE_NAME,
        verbose=True,
    )

    row = load_test_row(
        csv_path=TEST_CSV,
        requested_sample_id=SAMPLE_ID,
    )

    sample_id = str(
        row["sample_id"]
    )

    skeleton_path = Path(
        row["skeleton_path"]
    )

    rgb_frames = int(
        row["rgb_frames"]
    )

    skeleton_frames = int(
        row["skeleton_frames"]
    )

    usable_frames = min(
        rgb_frames,
        skeleton_frames,
    )

    sample_frame_dir = (
        EXTRACTED_FRAMES_DIR
        / sample_id
    )

    if not sample_frame_dir.exists():
        raise FileNotFoundError(
            f"Extracted frame directory not found: "
            f"{sample_frame_dir}"
        )

    print()
    print("=" * 70)
    print("NTU RGB+D heatmap video generation")
    print("=" * 70)
    print(f"Model version:      {MODEL_VERSION}")
    print(f"Model path:         {MODEL_PATH}")
    print(f"Sample ID:          {sample_id}")
    print(f"Skeleton path:      {skeleton_path}")
    print(f"Frame directory:    {sample_frame_dir}")
    print(f"Usable frames:      {usable_frames}")
    print(f"Person crop:        {PERSON_CROP}")
    print(f"BBox expansion:     {BBOX_EXPANSION}")
    print(f"Output video:       {OUTPUT_VIDEO_PATH}")

    model = load_model(
        model_path=MODEL_PATH,
        device=device,
    )

    transform = build_eval_transform(
        image_size=IMAGE_SIZE,
    )

    sequence = read_skeleton_file(
        skeleton_path
    )

    pose_sequence = (
        extract_primary_pose_sequence(
            sequence
        )
    )

    first_frame_path = (
        sample_frame_dir
        / "frame_000000.jpg"
    )

    first_frame = cv2.imread(
        str(first_frame_path),
        cv2.IMREAD_COLOR,
    )

    if first_frame is None:
        raise RuntimeError(
            f"Could not read first frame: "
            f"{first_frame_path}"
        )

    original_height, original_width = (
        first_frame.shape[:2]
    )

    video_width = original_width * 2
    video_height = original_height

    fps = (
        float(OUTPUT_FPS)
        if OUTPUT_FPS is not None
        else 30.0
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            video_width,
            video_height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video writer: "
            f"{OUTPUT_VIDEO_PATH}"
        )

    saved_frame_indices = []
    saved_predictions = []
    saved_confidences = []
    saved_ground_truth = []
    saved_visibility = []
    saved_bboxes = []

    try:
        for frame_number in range(
            0,
            usable_frames,
            FRAME_STRIDE,
        ):
            frame_path = (
                sample_frame_dir
                / f"frame_{frame_number:06d}.jpg"
            )

            frame = cv2.imread(
                str(frame_path),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                print(
                    f"Skipping unreadable frame: "
                    f"{frame_path}"
                )
                continue

            gt_keypoints = pose_sequence[
                "color_xy"
            ][frame_number].copy()

            tracking_state = pose_sequence[
                "tracking_state"
            ][frame_number].copy()

            gt_visibility = (
                coordinate_visibility(
                    gt_keypoints,
                    tracking_state=tracking_state,
                    image_size=(
                        frame.shape[1],
                        frame.shape[0],
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
                        expansion=(
                            BBOX_EXPANSION
                        ),
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
                        float(frame.shape[1]),
                        float(frame.shape[0]),
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
                .to(device)
            )

            with torch.inference_mode():
                predicted_heatmaps = model(
                    image_tensor
                )

            predicted_model_keypoints, confidence = (
                decode_heatmaps(
                    predicted_heatmaps,
                    image_size=IMAGE_SIZE,
                )
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
                            frame.shape[1]
                        ),
                        original_height=(
                            frame.shape[0]
                        ),
                        input_size=IMAGE_SIZE,
                    )
                )

            prediction_panel = (
                draw_skeleton(
                    image=frame,
                    keypoints=(
                        predicted_original_keypoints
                    ),
                    visibility=(
                        predicted_visibility
                    ),
                    point_color=(0, 0, 255),
                    line_color=(0, 255, 255),
                    label=(
                        f"Prediction - Model "
                        f"{MODEL_VERSION}"
                    ),
                )
            )

            if PERSON_CROP:
                prediction_panel = (
                    draw_bbox(
                        prediction_panel,
                        person_bbox,
                    )
                )

            ground_truth_panel = (
                draw_skeleton(
                    image=frame,
                    keypoints=gt_keypoints,
                    visibility=gt_visibility,
                    point_color=(0, 255, 0),
                    line_color=(255, 255, 0),
                    label="Ground Truth",
                )
            )

            comparison_frame = np.concatenate(
                [
                    prediction_panel,
                    ground_truth_panel,
                ],
                axis=1,
            )

            cv2.putText(
                comparison_frame,
                f"Frame: {frame_number}",
                (
                    comparison_frame.shape[1]
                    // 2
                    - 90,
                    comparison_frame.shape[0]
                    - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(
                comparison_frame
            )

            saved_frame_indices.append(
                frame_number
            )

            saved_predictions.append(
                predicted_original_keypoints
            )

            saved_confidences.append(
                confidence
            )

            saved_ground_truth.append(
                gt_keypoints
            )

            saved_visibility.append(
                gt_visibility
            )

            saved_bboxes.append(
                person_bbox
            )

            if (
                len(saved_frame_indices) % 50
                == 0
            ):
                print(
                    f"Processed "
                    f"{len(saved_frame_indices)} "
                    f"frames"
                )

    finally:
        writer.release()

    if not saved_predictions:
        raise RuntimeError(
            "No frames were successfully processed."
        )

    np.savez_compressed(
        OUTPUT_NPZ_PATH,
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
        ground_truth=np.asarray(
            saved_ground_truth,
            dtype=np.float32,
        ),
        visibility=np.asarray(
            saved_visibility,
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
    )

    print()
    print("=" * 70)
    print("Video generation completed")
    print("=" * 70)
    print(
        f"Processed frames: "
        f"{len(saved_frame_indices)}"
    )
    print(
        f"Video: "
        f"{OUTPUT_VIDEO_PATH}"
    )
    print(
        f"Predictions: "
        f"{OUTPUT_NPZ_PATH}"
    )


if __name__ == "__main__":
    main()