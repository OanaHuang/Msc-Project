# Scripts/NTU_RGBD/01_Check_Dataset.py

from __future__ import annotations

from pathlib import Path
import sys


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
    NTU_RGB_DIR,
    NTU_SKELETON_DIR,
    NTU_RGBD_OUTPUT_DIR,
)

from Scripts.NTU_RGBD.core import (
    get_sample_id,
    get_rgb_video_info,
    index_rgb_files,
    index_skeleton_files,
    read_skeleton_summary,
    validate_rgb_video,
)


# ============================================================
# 3. Config
# ============================================================

MAX_RGB_FILES_TO_CHECK = 10
MAX_SKELETON_FILES_TO_CHECK = 10

OUTPUT_DIR = (
    NTU_RGBD_OUTPUT_DIR
    / "01_Check_Dataset"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = (
    OUTPUT_DIR
    / "dataset_check_report.txt"
)


# ============================================================
# 4. Helpers
# ============================================================

def write_line(
    handle,
    text: str = "",
) -> None:
    print(text)
    handle.write(text + "\n")


def check_rgb_files(
    rgb_index: dict[str, Path],
    report_handle,
) -> tuple[int, int]:
    valid_count = 0
    invalid_count = 0

    write_line(
        report_handle,
        "\n"
        + "=" * 70,
    )

    write_line(
        report_handle,
        "RGB video check",
    )

    write_line(
        report_handle,
        "=" * 70,
    )

    sample_ids = sorted(
        rgb_index.keys()
    )

    files_to_check = sample_ids[
        :MAX_RGB_FILES_TO_CHECK
    ]

    if not files_to_check:
        write_line(
            report_handle,
            "No RGB videos found.",
        )

        return 0, 0

    for index, sample_id in enumerate(
        files_to_check,
        start=1,
    ):
        video_path = rgb_index[
            sample_id
        ]

        write_line(
            report_handle,
            f"\n[{index}/{len(files_to_check)}]",
        )

        write_line(
            report_handle,
            f"Sample ID: {sample_id}",
        )

        write_line(
            report_handle,
            f"Path: {video_path}",
        )

        is_valid, message = (
            validate_rgb_video(
                video_path,
                test_first_frame=True,
            )
        )

        write_line(
            report_handle,
            f"Validation: {message}",
        )

        if not is_valid:
            invalid_count += 1
            continue

        valid_count += 1

        info = get_rgb_video_info(
            video_path
        )

        write_line(
            report_handle,
            (
                f"Frames: {info['frame_count']} | "
                f"FPS: {info['fps']:.2f} | "
                f"Resolution: "
                f"{info['width']}x{info['height']} | "
                f"Duration: "
                f"{info['duration_seconds']:.2f}s"
            ),
        )

    return valid_count, invalid_count


def check_skeleton_files(
    skeleton_index: dict[str, Path],
    report_handle,
) -> tuple[int, int]:
    valid_count = 0
    invalid_count = 0

    write_line(
        report_handle,
        "\n"
        + "=" * 70,
    )

    write_line(
        report_handle,
        "Skeleton file check",
    )

    write_line(
        report_handle,
        "=" * 70,
    )

    sample_ids = sorted(
        skeleton_index.keys()
    )

    files_to_check = sample_ids[
        :MAX_SKELETON_FILES_TO_CHECK
    ]

    if not files_to_check:
        write_line(
            report_handle,
            "No skeleton files found.",
        )

        return 0, 0

    for index, sample_id in enumerate(
        files_to_check,
        start=1,
    ):
        skeleton_path = (
            skeleton_index[
                sample_id
            ]
        )

        write_line(
            report_handle,
            f"\n[{index}/{len(files_to_check)}]",
        )

        write_line(
            report_handle,
            f"Sample ID: {sample_id}",
        )

        write_line(
            report_handle,
            f"Path: {skeleton_path}",
        )

        try:
            summary = (
                read_skeleton_summary(
                    skeleton_path
                )
            )

        except Exception as error:
            invalid_count += 1

            write_line(
                report_handle,
                f"Validation failed: {error}",
            )

            continue

        valid_count += 1

        write_line(
            report_handle,
            (
                f"Frames: "
                f"{summary['num_frames']} | "
                f"Max bodies: "
                f"{summary['max_bodies']} | "
                f"Empty frames: "
                f"{summary['empty_frames']} | "
                f"Single person: "
                f"{summary['is_single_person']}"
            ),
        )

    return valid_count, invalid_count


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    print("=" * 70)
    print("NTU RGB+D dataset check")
    print("=" * 70)

    print(f"RGB directory:      {NTU_RGB_DIR}")
    print(f"Skeleton directory: {NTU_SKELETON_DIR}")
    print(f"Report path:        {REPORT_PATH}")

    rgb_index = index_rgb_files(
        NTU_RGB_DIR,
        recursive=True,
    )

    skeleton_index = (
        index_skeleton_files(
            NTU_SKELETON_DIR,
            recursive=True,
        )
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_handle:

        write_line(
            report_handle,
            "=" * 70,
        )

        write_line(
            report_handle,
            "NTU RGB+D dataset check report",
        )

        write_line(
            report_handle,
            "=" * 70,
        )

        write_line(
            report_handle,
            f"RGB directory: {NTU_RGB_DIR}",
        )

        write_line(
            report_handle,
            (
                "Skeleton directory: "
                f"{NTU_SKELETON_DIR}"
            ),
        )

        write_line(
            report_handle,
            f"Total RGB videos: {len(rgb_index)}",
        )

        write_line(
            report_handle,
            (
                "Total skeleton files: "
                f"{len(skeleton_index)}"
            ),
        )

        rgb_valid, rgb_invalid = (
            check_rgb_files(
                rgb_index,
                report_handle,
            )
        )

        skeleton_valid, skeleton_invalid = (
            check_skeleton_files(
                skeleton_index,
                report_handle,
            )
        )

        write_line(
            report_handle,
            "\n"
            + "=" * 70,
        )

        write_line(
            report_handle,
            "Summary",
        )

        write_line(
            report_handle,
            "=" * 70,
        )

        write_line(
            report_handle,
            (
                f"Checked RGB videos: "
                f"{rgb_valid + rgb_invalid}"
            ),
        )

        write_line(
            report_handle,
            f"Valid RGB videos: {rgb_valid}",
        )

        write_line(
            report_handle,
            f"Invalid RGB videos: {rgb_invalid}",
        )

        write_line(
            report_handle,
            (
                f"Checked skeleton files: "
                f"{skeleton_valid + skeleton_invalid}"
            ),
        )

        write_line(
            report_handle,
            (
                "Valid skeleton files: "
                f"{skeleton_valid}"
            ),
        )

        write_line(
            report_handle,
            (
                "Invalid skeleton files: "
                f"{skeleton_invalid}"
            ),
        )

    print()
    print("=" * 70)
    print("Dataset check finished.")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()