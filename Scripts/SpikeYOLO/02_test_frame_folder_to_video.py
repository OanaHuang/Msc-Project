from pathlib import Path
import sys
import csv
import time

import cv2
import numpy as np


# ============================================================
# 1. Paths and Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPIKEYOLO_ROOT = (
    PROJECT_ROOT
    / "External"
    / "SpikeYOLO"
)

WEIGHTS_PATH = (
    SPIKEYOLO_ROOT
    / "weights"
    / "best.pt"
)

FRAME_DIR = (
    PROJECT_ROOT
    / "Datasets"
    / "NTU_RGBD"
    / "extracted_frames"
    / "S015C001P007R001A001"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "SpikeYOLO"
    / "02_frame_folder_to_video"
    / FRAME_DIR.name
)

ANNOTATED_FRAME_DIR = OUTPUT_DIR / "annotated_frames"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / f"{FRAME_DIR.name}_spikeyolo.mp4"
CSV_PATH = OUTPUT_DIR / "detection_results.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

CONF_THRESHOLD = 0.25
VIDEO_FPS = 30.0

# Mac本地运行使用CPU。
# 传到CUDA服务器后可以改成 0。
DEVICE = "cpu"


# ============================================================
# 2. Import local SpikeYOLO
# ============================================================

sys.path.insert(0, str(SPIKEYOLO_ROOT))

from ultralytics import YOLO  # noqa: E402


# ============================================================
# 3. Helpers
# ============================================================

def get_frame_paths(frame_dir: Path) -> list[Path]:
    frame_paths = sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not frame_paths:
        raise FileNotFoundError(
            f"No image frames found in:\n{frame_dir}"
        )

    return frame_paths


def create_video_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video:\n{output_path}"
        )

    return writer


# ============================================================
# 4. Main
# ============================================================

def main():
    if not SPIKEYOLO_ROOT.exists():
        raise FileNotFoundError(
            f"SpikeYOLO repository not found:\n{SPIKEYO_ROOT}"
        )

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"SpikeYOLO weights not found:\n{WEIGHTS_PATH}"
        )

    if not FRAME_DIR.exists():
        raise FileNotFoundError(
            f"NTU frame directory not found:\n{FRAME_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    frame_paths = get_frame_paths(FRAME_DIR)

    print(f"SpikeYOLO root: {SPIKEYOLO_ROOT}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Input frame directory: {FRAME_DIR}")
    print(f"Total frames: {len(frame_paths)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    first_frame = cv2.imread(str(frame_paths[0]))

    if first_frame is None:
        raise RuntimeError(
            f"Could not read first frame:\n{frame_paths[0]}"
        )

    frame_height, frame_width = first_frame.shape[:2]

    video_writer = create_video_writer(
        output_path=OUTPUT_VIDEO_PATH,
        width=frame_width,
        height=frame_height,
        fps=VIDEO_FPS,
    )

    model = YOLO(str(WEIGHTS_PATH))

    detected_frame_count = 0
    total_person_detections = 0
    inference_times_ms = []
    csv_rows = []

    try:
        for frame_index, frame_path in enumerate(
            frame_paths,
            start=1,
        ):
            start_time = time.perf_counter()

            results = model(
                str(frame_path),
                conf=CONF_THRESHOLD,
                classes=[0],
                device=DEVICE,
                verbose=False,
            )

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            inference_times_ms.append(elapsed_ms)

            result = results[0]

            if result.boxes is None:
                person_count = 0
                max_confidence = 0.0
            else:
                person_count = len(result.boxes)

                if person_count > 0:
                    max_confidence = float(
                        result.boxes.conf.max().cpu().item()
                    )
                else:
                    max_confidence = 0.0

            if person_count > 0:
                detected_frame_count += 1
                total_person_detections += person_count

            annotated_frame = result.plot()

            if annotated_frame.shape[:2] != (
                frame_height,
                frame_width,
            ):
                annotated_frame = cv2.resize(
                    annotated_frame,
                    (frame_width, frame_height),
                )

            annotated_frame_path = (
                ANNOTATED_FRAME_DIR / frame_path.name
            )

            success = cv2.imwrite(
                str(annotated_frame_path),
                annotated_frame,
            )

            if not success:
                raise RuntimeError(
                    f"Could not save annotated frame:\n"
                    f"{annotated_frame_path}"
                )

            video_writer.write(annotated_frame)

            csv_rows.append(
                {
                    "frame_index": frame_index - 1,
                    "frame_name": frame_path.name,
                    "person_count": person_count,
                    "max_confidence": max_confidence,
                    "inference_time_ms": elapsed_ms,
                }
            )

            print(
                f"[{frame_index:04d}/{len(frame_paths):04d}] "
                f"{frame_path.name} | "
                f"persons={person_count} | "
                f"max_conf={max_confidence:.4f} | "
                f"time={elapsed_ms:.1f} ms"
            )

    finally:
        video_writer.release()

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_index",
                "frame_name",
                "person_count",
                "max_confidence",
                "inference_time_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    detection_rate = (
        detected_frame_count / len(frame_paths)
    )

    average_inference_ms = float(
        np.mean(inference_times_ms)
    )

    estimated_fps = (
        1000.0 / average_inference_ms
        if average_inference_ms > 0
        else 0.0
    )

    print()
    print("=" * 60)
    print("SpikeYOLO NTU Folder Test Complete")
    print("=" * 60)
    print(f"Total frames: {len(frame_paths)}")
    print(
        f"Frames with person detection: "
        f"{detected_frame_count}"
    )
    print(f"Frame detection rate: {detection_rate:.4f}")
    print(
        f"Total person detections: "
        f"{total_person_detections}"
    )
    print(
        f"Average inference time: "
        f"{average_inference_ms:.2f} ms/frame"
    )
    print(f"Estimated inference FPS: {estimated_fps:.2f}")
    print(f"Annotated frames:\n{ANNOTATED_FRAME_DIR}")
    print(f"Output video:\n{OUTPUT_VIDEO_PATH}")
    print(f"Detection CSV:\n{CSV_PATH}")


if __name__ == "__main__":
    main()