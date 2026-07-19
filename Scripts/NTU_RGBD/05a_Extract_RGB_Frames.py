# Scripts/NTU_RGBD/05a_Extract_RGB_Frames.py

from __future__ import annotations

from pathlib import Path
import csv
import sys
import time

import cv2


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
)


# ============================================================
# 3. Configuration
# ============================================================

METADATA_CSV_FILES = (
    NTU_METADATA_DIR / "train_split.csv",
    NTU_METADATA_DIR / "val_split.csv",
    NTU_METADATA_DIR / "test_split.csv",
)

OUTPUT_ROOT = (
    NTU_RGBD_DATASET_DIR
    / "extracted_frames"

)

# Extract every frame.
#
# 1 = every frame
# 5 = frames 0, 5, 10, 15, ...
EXTRACT_STRIDE = 1

JPEG_QUALITY = 90

# Skip a video if its output folder already contains
# the expected final frame.
SKIP_COMPLETED = True

# Use None to process every video.
MAX_VIDEOS = None

PRINT_EVERY = 10


# ============================================================
# 4. Read metadata
# ============================================================

def load_video_samples() -> list[dict[str, str]]:
    samples_by_id: dict[str, dict[str, str]] = {}

    for csv_path in METADATA_CSV_FILES:
        if not csv_path.exists():
            print(
                f"Metadata CSV not found, skipping: "
                f"{csv_path}"
            )
            continue

        with csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            required_columns = {
                "sample_id",
                "rgb_path",
            }

            if reader.fieldnames is None:
                raise RuntimeError(
                    f"CSV has no header: {csv_path}"
                )

            missing_columns = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing_columns:
                raise RuntimeError(
                    f"Missing columns in {csv_path}: "
                    f"{sorted(missing_columns)}"
                )

            for row in reader:
                sample_id = str(
                    row["sample_id"]
                ).strip()

                rgb_path = str(
                    row["rgb_path"]
                ).strip()

                if not sample_id or not rgb_path:
                    continue

                samples_by_id[sample_id] = {
                    "sample_id": sample_id,
                    "rgb_path": rgb_path,
                }

    samples = list(
        samples_by_id.values()
    )

    samples.sort(
        key=lambda item: item["sample_id"]
    )

    if MAX_VIDEOS is not None:
        samples = samples[:MAX_VIDEOS]

    if not samples:
        raise RuntimeError(
            "No RGB video samples were found in "
            "the metadata CSV files."
        )

    return samples


# ============================================================
# 5. Frame extraction
# ============================================================

def expected_frame_numbers(
    total_frames: int,
) -> list[int]:
    return list(
        range(
            0,
            total_frames,
            EXTRACT_STRIDE,
        )
    )


def frame_path(
    output_dir: Path,
    frame_number: int,
) -> Path:
    return (
        output_dir
        / f"frame_{frame_number:06d}.jpg"
    )


def is_video_complete(
    output_dir: Path,
    total_frames: int,
) -> bool:
    if total_frames <= 0:
        return False

    frame_numbers = expected_frame_numbers(
        total_frames
    )

    if not frame_numbers:
        return False

    first_path = frame_path(
        output_dir,
        frame_numbers[0],
    )

    last_path = frame_path(
        output_dir,
        frame_numbers[-1],
    )

    return (
        first_path.exists()
        and last_path.exists()
    )


def extract_video_frames(
    sample_id: str,
    rgb_path: Path,
) -> tuple[int, int]:
    if not rgb_path.exists():
        raise FileNotFoundError(
            f"RGB video not found: {rgb_path}"
        )

    output_dir = (
        OUTPUT_ROOT
        / sample_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(
        str(rgb_path)
    )

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            f"Could not open video: {rgb_path}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if (
        SKIP_COMPLETED
        and is_video_complete(
            output_dir,
            total_frames,
        )
    ):
        capture.release()

        expected_count = len(
            expected_frame_numbers(
                total_frames
            )
        )

        return expected_count, 0

    saved_count = 0
    frame_number = 0

    try:
        while True:
            success, image = capture.read()

            if not success or image is None:
                break

            if (
                frame_number
                % EXTRACT_STRIDE
                == 0
            ):
                output_path = frame_path(
                    output_dir,
                    frame_number,
                )

                write_success = cv2.imwrite(
                    str(output_path),
                    image,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        JPEG_QUALITY,
                    ],
                )

                if not write_success:
                    raise RuntimeError(
                        f"Could not write frame: "
                        f"{output_path}"
                    )

                saved_count += 1

            frame_number += 1

    finally:
        capture.release()

    return saved_count, saved_count


# ============================================================
# 6. Manifest
# ============================================================

def write_manifest(
    rows: list[dict[str, object]],
) -> Path:
    manifest_path = (
        OUTPUT_ROOT
        / "extracted_frames_manifest.csv"
    )

    fieldnames = [
        "sample_id",
        "rgb_path",
        "frame_directory",
        "saved_frames",
        "newly_written_frames",
        "status",
        "error",
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


# ============================================================
# 7. Main
# ============================================================

def main() -> None:
    if EXTRACT_STRIDE <= 0:
        raise ValueError(
            "EXTRACT_STRIDE must be positive"
        )

    if not 1 <= JPEG_QUALITY <= 100:
        raise ValueError(
            "JPEG_QUALITY must be between 1 and 100"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = load_video_samples()

    print("=" * 70)
    print("NTU RGB+D RGB frame extraction")
    print("=" * 70)

    print(f"Output root:       {OUTPUT_ROOT}")
    print(f"Number of videos:  {len(samples)}")
    print(f"Frame stride:      {EXTRACT_STRIDE}")
    print(f"JPEG quality:      {JPEG_QUALITY}")
    print(f"Skip completed:    {SKIP_COMPLETED}")

    start_time = time.perf_counter()

    manifest_rows: list[
        dict[str, object]
    ] = []

    successful_videos = 0
    failed_videos = 0
    total_saved_frames = 0
    total_new_frames = 0

    for video_index, sample in enumerate(
        samples,
        start=1,
    ):
        sample_id = sample["sample_id"]

        rgb_path = Path(
            sample["rgb_path"]
        )

        output_dir = (
            OUTPUT_ROOT
            / sample_id
        )

        try:
            saved_frames, newly_written = (
                extract_video_frames(
                    sample_id=sample_id,
                    rgb_path=rgb_path,
                )
            )

            successful_videos += 1
            total_saved_frames += saved_frames
            total_new_frames += newly_written

            status = (
                "skipped"
                if newly_written == 0
                else "extracted"
            )

            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "rgb_path": str(rgb_path),
                    "frame_directory": str(
                        output_dir
                    ),
                    "saved_frames": saved_frames,
                    "newly_written_frames": (
                        newly_written
                    ),
                    "status": status,
                    "error": "",
                }
            )

        except Exception as error:
            failed_videos += 1

            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "rgb_path": str(rgb_path),
                    "frame_directory": str(
                        output_dir
                    ),
                    "saved_frames": 0,
                    "newly_written_frames": 0,
                    "status": "failed",
                    "error": str(error),
                }
            )

            print()
            print(
                f"[ERROR] {sample_id}: "
                f"{error}"
            )

        if (
            video_index == 1
            or video_index % PRINT_EVERY == 0
            or video_index == len(samples)
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"[{video_index:>5}/"
                f"{len(samples)}] "
                f"success={successful_videos}, "
                f"failed={failed_videos}, "
                f"new_frames={total_new_frames:,}, "
                f"time={elapsed / 60:.1f} min"
            )

    manifest_path = write_manifest(
        manifest_rows
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 70)
    print("Frame extraction completed")
    print("=" * 70)

    print(
        f"Successful videos: "
        f"{successful_videos}"
    )

    print(
        f"Failed videos:     "
        f"{failed_videos}"
    )

    print(
        f"Total frame files: "
        f"{total_saved_frames:,}"
    )

    print(
        f"New frames written:"
        f" {total_new_frames:,}"
    )

    print(
        f"Elapsed time:      "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Manifest:          "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()