from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import csv
import re
import traceback

import cv2
import numpy as np
import torch

from Scripts.NTU_RGBD.core import (
    coordinate_visibility,
    extract_primary_pose_sequence,
    read_skeleton_file,
)
from Scripts.NTU_RGBD.datasets.person_crop import crop_and_resize_person


NTU_SAMPLE_ID_PATTERN = re.compile(
    r"S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}",
    re.IGNORECASE,
)

DEFAULT_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
)

SUMMARY_FIELDS = (
    "sample_id",
    "status",
    "processed_frames",
    "rgb_video",
    "skeleton_file",
    "output_video",
    "output_npz",
    "error",
)


@dataclass(frozen=True)
class VideoGenerationConfig:
    """Configuration shared by all NTU pose-video generation scripts."""

    model_version: str
    test_csv: Path
    rgb_video_dir: Path
    skeleton_dir: Path
    output_dir: Path

    image_size: int = 224
    heatmap_size: int = 56
    num_joints: int = 25

    person_crop: bool = True
    bbox_expansion: float = 0.25
    confidence_threshold: float = 0.02

    frame_stride: int = 1
    output_fps: float | None = None
    skip_existing_videos: bool = True
    save_prediction_npz: bool = True
    max_test_videos: int | None = None

    readout_type: str = "heatmap"
    prediction_label: str = "Prediction"
    skeleton_edges: Sequence[tuple[int, int]] = DEFAULT_SKELETON_EDGES

    # Optional metadata written into the NPZ.
    num_steps: int | None = None
    beta: float | None = None
    threshold: float | None = None
    surrogate_slope: float | None = None

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1.")
        if self.image_size < 1 or self.heatmap_size < 1:
            raise ValueError("image_size and heatmap_size must be positive.")
        if self.num_joints < 1:
            raise ValueError("num_joints must be positive.")
        if self.max_test_videos is not None and self.max_test_videos < 1:
            raise ValueError("max_test_videos must be None or positive.")

    @property
    def summary_csv(self) -> Path:
        return self.output_dir / "generation_summary.csv"


def extract_sample_id_from_text(value: str) -> str | None:
    match = NTU_SAMPLE_ID_PATTERN.search(str(value))
    return None if match is None else match.group(0).upper()


def load_test_rows(config: VideoGenerationConfig) -> list[dict[str, str]]:
    if not config.test_csv.exists():
        raise FileNotFoundError(f"Test split not found: {config.test_csv}")

    rows: list[dict[str, str]] = []
    with config.test_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            single_person = str(row.get("is_single_person", "")).strip().lower()
            if single_person not in {"true", "1", "yes"}:
                continue

            raw_sample_id = str(row.get("sample_id", ""))
            sample_id = extract_sample_id_from_text(raw_sample_id)
            if sample_id is None:
                print(f"Skipping invalid sample_id: {raw_sample_id}")
                continue

            row["sample_id"] = sample_id
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid single-person test samples were found.")

    if config.max_test_videos is not None:
        rows = rows[: config.max_test_videos]

    return rows


def build_file_index(
    root: Path,
    allowed_suffixes: set[str],
) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    suffixes = {suffix.lower() for suffix in allowed_suffixes}
    index: dict[str, Path] = {}
    ignored = 0
    duplicates = 0

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue

        sample_id = (
            extract_sample_id_from_text(path.name)
            or extract_sample_id_from_text(str(path))
        )
        if sample_id is None:
            ignored += 1
            continue
        if sample_id in index:
            duplicates += 1
            continue
        index[sample_id] = path

    print(f"  Indexed files: {len(index)}")
    if ignored:
        print(f"  Ignored files without NTU ID: {ignored}")
    if duplicates:
        print(f"  Duplicate sample IDs ignored: {duplicates}")

    return index


def find_indexed_file(
    sample_id: str,
    file_index: dict[str, Path],
    file_type: str,
) -> Path:
    normalised_id = extract_sample_id_from_text(sample_id)
    if normalised_id is None:
        raise ValueError(f"Invalid NTU sample ID: {sample_id}")

    path = file_index.get(normalised_id)
    if path is None:
        raise FileNotFoundError(f"{file_type} not found for sample: {normalised_id}")
    return path


def decode_heatmaps(
    heatmaps: torch.Tensor,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if heatmaps.ndim != 4 or heatmaps.shape[0] != 1:
        raise ValueError("heatmaps must have shape [1, J, H, W].")

    heatmaps_np = heatmaps[0].detach().cpu().numpy()
    num_joints, height, width = heatmaps_np.shape

    keypoints = np.zeros((num_joints, 2), dtype=np.float32)
    confidence = np.zeros(num_joints, dtype=np.float32)

    for joint_index, heatmap in enumerate(heatmaps_np):
        flat_index = int(np.argmax(heatmap))
        y, x = np.unravel_index(flat_index, heatmap.shape)
        confidence[joint_index] = float(heatmap[y, x])
        keypoints[joint_index] = (
            (x + 0.5) * image_size / width,
            (y + 0.5) * image_size / height,
        )

    return keypoints, confidence


def map_crop_keypoints_to_original(
    crop_keypoints: np.ndarray,
    bbox_xyxy: np.ndarray,
    input_size: int,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    crop_width = max(x2 - x1, 1.0)
    crop_height = max(y2 - y1, 1.0)

    output = crop_keypoints.copy()
    output[:, 0] = x1 + crop_keypoints[:, 0] * crop_width / input_size
    output[:, 1] = y1 + crop_keypoints[:, 1] * crop_height / input_size
    return output


def map_resized_keypoints_to_original(
    resized_keypoints: np.ndarray,
    original_width: int,
    original_height: int,
    input_size: int,
) -> np.ndarray:
    output = resized_keypoints.copy()
    output[:, 0] *= original_width / input_size
    output[:, 1] *= original_height / input_size
    return output


def point_is_valid(
    point: np.ndarray,
    image_width: int,
    image_height: int,
) -> bool:
    x, y = float(point[0]), float(point[1])
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
    skeleton_edges: Sequence[tuple[int, int]],
    point_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    point_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    output = image.copy()
    image_height, image_width = output.shape[:2]
    visibility_bool = np.asarray(visibility, dtype=bool)

    for joint_a, joint_b in skeleton_edges:
        if joint_a >= len(keypoints) or joint_b >= len(keypoints):
            continue
        if not (visibility_bool[joint_a] and visibility_bool[joint_b]):
            continue

        point_a = keypoints[joint_a]
        point_b = keypoints[joint_b]
        if not point_is_valid(point_a, image_width, image_height):
            continue
        if not point_is_valid(point_b, image_width, image_height):
            continue

        cv2.line(
            output,
            tuple(np.round(point_a).astype(int)),
            tuple(np.round(point_b).astype(int)),
            line_color,
            line_thickness,
            cv2.LINE_AA,
        )

    for joint_index, point in enumerate(keypoints):
        if joint_index >= len(visibility_bool) or not visibility_bool[joint_index]:
            continue
        if not point_is_valid(point, image_width, image_height):
            continue
        cv2.circle(
            output,
            tuple(np.round(point).astype(int)),
            point_radius,
            point_color,
            -1,
            cv2.LINE_AA,
        )

    return output


def draw_bbox(image: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    output = image.copy()
    x1, y1, x2, y2 = np.round(bbox_xyxy).astype(int)
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
    prediction_label: str,
) -> np.ndarray:
    output = image.copy()
    overlay = output.copy()

    panel_width = min(max(340, 80 + len(prediction_label) * 12), output.shape[1] - 20)
    cv2.rectangle(overlay, (10, 10), (panel_width, 82), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.55, output, 0.45, 0)

    cv2.line(output, (25, 32), (55, 32), (0, 180, 0), 3, cv2.LINE_AA)
    cv2.circle(output, (40, 32), 5, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(
        output, "Ground Truth", (68, 39),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
    )

    cv2.line(output, (25, 64), (55, 64), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(output, (40, 64), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        output, prediction_label, (68, 71),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA,
    )
    return output


def _npz_optional_metadata(config: VideoGenerationConfig) -> dict[str, np.ndarray]:
    metadata: dict[str, np.ndarray] = {}
    optional_values = {
        "num_steps": config.num_steps,
        "beta": config.beta,
        "threshold": config.threshold,
        "surrogate_slope": config.surrogate_slope,
    }
    for key, value in optional_values.items():
        if value is not None:
            metadata[key] = np.asarray(value)
    return metadata


def generate_sample_video(
    row: dict[str, str],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    rgb_video_index: dict[str, Path],
    skeleton_index: dict[str, Path],
    config: VideoGenerationConfig,
) -> dict[str, object]:
    sample_id = str(row["sample_id"]).upper()
    output_video_path = (
        config.output_dir
        / f"{sample_id}_gt_prediction_model_{config.model_version}.mp4"
    )
    output_npz_path = (
        config.output_dir
        / f"{sample_id}_predictions_model_{config.model_version}.npz"
    )

    if (
        config.skip_existing_videos
        and output_video_path.exists()
        and output_video_path.stat().st_size > 0
    ):
        return {
            "sample_id": sample_id,
            "status": "skipped_existing",
            "processed_frames": "",
            "rgb_video": "",
            "skeleton_file": "",
            "output_video": str(output_video_path),
            "output_npz": str(output_npz_path) if output_npz_path.exists() else "",
            "error": "",
        }

    rgb_video_path = find_indexed_file(sample_id, rgb_video_index, "RGB video")
    skeleton_path = find_indexed_file(sample_id, skeleton_index, "Skeleton file")

    sequence = read_skeleton_file(skeleton_path)
    pose_sequence = extract_primary_pose_sequence(sequence)
    pose_frames = len(pose_sequence["color_xy"])

    capture = cv2.VideoCapture(str(rgb_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open RGB video: {rgb_video_path}")

    writer: cv2.VideoWriter | None = None
    frame_indices: list[int] = []
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    predicted_visibilities: list[np.ndarray] = []
    ground_truths: list[np.ndarray] = []
    ground_truth_visibilities: list[np.ndarray] = []
    bboxes: list[np.ndarray] = []

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        output_fps = (
            source_fps / config.frame_stride
            if config.output_fps is None
            else float(config.output_fps)
        )

        frame_number = 0
        while frame_number < pose_frames:
            success, frame = capture.read()
            if not success:
                break

            current_frame = frame_number
            frame_number += 1
            if current_frame % config.frame_stride != 0:
                continue

            frame_height, frame_width = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (frame_width, frame_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not create video writer: {output_video_path}"
                    )

            gt_keypoints = pose_sequence["color_xy"][current_frame].copy()
            tracking_state = pose_sequence["tracking_state"][current_frame].copy()
            gt_visibility = coordinate_visibility(
                gt_keypoints,
                tracking_state=tracking_state,
                image_size=(frame_width, frame_height),
                include_inferred=False,
            ).astype(np.float32)

            if config.person_crop:
                crop_result = crop_and_resize_person(
                    image=frame,
                    keypoints=gt_keypoints,
                    visibility=gt_visibility,
                    output_size=config.image_size,
                    expansion=config.bbox_expansion,
                    make_square=True,
                )
                model_image = crop_result.image
                model_keypoints = crop_result.keypoints
                model_visibility = crop_result.visibility
                person_bbox = crop_result.bbox_xyxy
            else:
                model_image = frame.copy()
                model_keypoints = gt_keypoints.copy()
                model_visibility = gt_visibility.copy()
                person_bbox = np.asarray(
                    [0.0, 0.0, float(frame_width), float(frame_height)],
                    dtype=np.float32,
                )

            transformed = transform(
                image=model_image,
                keypoints=model_keypoints,
                visibility=model_visibility,
            )
            image_tensor = transformed["image"].unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            with torch.inference_mode():
                predicted_heatmaps = model(image_tensor)

            predicted_model_keypoints, confidence = decode_heatmaps(
                predicted_heatmaps,
                image_size=config.image_size,
            )
            predicted_visibility = (
                confidence >= config.confidence_threshold
            ).astype(np.float32)

            if config.person_crop:
                predicted_original_keypoints = map_crop_keypoints_to_original(
                    predicted_model_keypoints,
                    person_bbox,
                    config.image_size,
                )
            else:
                predicted_original_keypoints = map_resized_keypoints_to_original(
                    predicted_model_keypoints,
                    frame_width,
                    frame_height,
                    config.image_size,
                )

            comparison_frame = draw_skeleton(
                frame,
                gt_keypoints,
                gt_visibility,
                config.skeleton_edges,
                point_color=(0, 255, 0),
                line_color=(0, 180, 0),
            )
            comparison_frame = draw_skeleton(
                comparison_frame,
                predicted_original_keypoints,
                predicted_visibility,
                config.skeleton_edges,
                point_color=(0, 0, 255),
                line_color=(0, 255, 255),
            )

            if config.person_crop:
                comparison_frame = draw_bbox(comparison_frame, person_bbox)

            comparison_frame = draw_legend(
                comparison_frame,
                config.prediction_label,
            )
            cv2.putText(
                comparison_frame,
                f"Sample: {sample_id}",
                (20, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                comparison_frame,
                f"Frame: {current_frame}",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(comparison_frame)
            frame_indices.append(current_frame)

            if config.save_prediction_npz:
                predictions.append(predicted_original_keypoints)
                confidences.append(confidence)
                predicted_visibilities.append(predicted_visibility)
                ground_truths.append(gt_keypoints)
                ground_truth_visibilities.append(gt_visibility)
                bboxes.append(person_bbox)

    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if not frame_indices:
        if output_video_path.exists():
            output_video_path.unlink()
        raise RuntimeError("No frames were successfully processed.")

    if config.save_prediction_npz:
        np.savez_compressed(
            output_npz_path,
            sample_id=np.asarray(sample_id),
            model_version=np.asarray(config.model_version),
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            predictions=np.asarray(predictions, dtype=np.float32),
            confidences=np.asarray(confidences, dtype=np.float32),
            predicted_visibility=np.asarray(
                predicted_visibilities, dtype=np.float32
            ),
            ground_truth=np.asarray(ground_truths, dtype=np.float32),
            ground_truth_visibility=np.asarray(
                ground_truth_visibilities, dtype=np.float32
            ),
            person_bboxes=np.asarray(bboxes, dtype=np.float32),
            person_crop=np.asarray(config.person_crop),
            bbox_expansion=np.asarray(config.bbox_expansion, dtype=np.float32),
            image_size=np.asarray(config.image_size, dtype=np.int32),
            heatmap_size=np.asarray(config.heatmap_size, dtype=np.int32),
            readout_type=np.asarray(config.readout_type),
            **_npz_optional_metadata(config),
        )

    return {
        "sample_id": sample_id,
        "status": "completed",
        "processed_frames": len(frame_indices),
        "rgb_video": str(rgb_video_path),
        "skeleton_file": str(skeleton_path),
        "output_video": str(output_video_path),
        "output_npz": str(output_npz_path) if config.save_prediction_npz else "",
        "error": "",
    }


def write_summary(
    records: list[dict[str, object]],
    summary_csv: Path,
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _print_coverage(
    rows: list[dict[str, str]],
    rgb_video_index: dict[str, Path],
    skeleton_index: dict[str, Path],
) -> None:
    sample_ids = {str(row["sample_id"]).upper() for row in rows}
    missing_rgb = sorted(sample_ids - set(rgb_video_index))
    missing_skeleton = sorted(sample_ids - set(skeleton_index))

    print("\n" + "=" * 72)
    print("Test-set file coverage")
    print("=" * 72)
    print(f"Test sample IDs:     {len(sample_ids)}")
    print(f"Missing RGB videos:  {len(missing_rgb)}")
    print(f"Missing skeletons:   {len(missing_skeleton)}")
    if missing_rgb:
        print("First missing RGB IDs: " + ", ".join(missing_rgb[:10]))
    if missing_skeleton:
        print("First missing skeleton IDs: " + ", ".join(missing_skeleton[:10]))
    print("=" * 72)


def run_video_generation(
    *,
    config: VideoGenerationConfig,
    model_loader: Callable[[torch.device], torch.nn.Module],
    transform_builder: Callable[[int], object],
    device: torch.device,
) -> list[dict[str, object]]:
    """
    Run the complete batch-generation pipeline.

    A model-specific entry script only needs to provide:
      1. VideoGenerationConfig
      2. model_loader(device)
      3. transform_builder(image_size)
      4. device
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_test_rows(config)

    print("\n" + "=" * 72)
    print("NTU RGB+D pose video generation")
    print("=" * 72)
    print(f"Model version:       {config.model_version}")
    print(f"Readout type:        {config.readout_type}")
    print(f"Test videos:         {len(rows)}")
    print(f"Person crop:         {config.person_crop}")
    print(f"Frame stride:        {config.frame_stride}")
    print(f"Skip existing:       {config.skip_existing_videos}")
    print(f"Save NPZ:            {config.save_prediction_npz}")
    print(f"Output directory:    {config.output_dir}")
    print("=" * 72)

    print("\nBuilding RGB video index...")
    rgb_video_index = build_file_index(
        config.rgb_video_dir,
        {".avi", ".mp4", ".mov", ".mkv"},
    )

    print("\nBuilding skeleton index...")
    skeleton_index = build_file_index(
        config.skeleton_dir,
        {".skeleton"},
    )

    _print_coverage(rows, rgb_video_index, skeleton_index)

    print("\nLoading model once...")
    model = model_loader(device)
    model.to(device)
    model.eval()
    transform = transform_builder(config.image_size)

    records: list[dict[str, object]] = []
    completed = skipped = failed = 0

    for row_index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"]).upper()
        print(f"\n[{row_index}/{len(rows)}] {sample_id}")

        try:
            record = generate_sample_video(
                row=row,
                model=model,
                transform=transform,
                device=device,
                rgb_video_index=rgb_video_index,
                skeleton_index=skeleton_index,
                config=config,
            )
            status = str(record["status"])
            if status == "completed":
                completed += 1
                print(f"  Completed: {record['processed_frames']} frames")
            else:
                skipped += 1
                print("  Skipped: output MP4 already exists")
            records.append(record)

        except Exception as error:
            failed += 1
            error_text = f"{type(error).__name__}: {error}"
            print(f"  Failed: {error_text}")
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

        write_summary(records, config.summary_csv)
        print(
            f"  Progress: completed={completed}, "
            f"skipped={skipped}, failed={failed}"
        )

    print("\n" + "=" * 72)
    print("Batch video generation finished")
    print("=" * 72)
    print(f"Total test videos: {len(rows)}")
    print(f"Completed:         {completed}")
    print(f"Skipped existing:  {skipped}")
    print(f"Failed:            {failed}")
    print(f"Summary CSV:       {config.summary_csv}")
    print(f"Output directory:  {config.output_dir}")

    return records
