# Scripts/NTU_RGBD/02_Match_RGB_Skeleton.py

from __future__ import annotations

from pathlib import Path
import csv
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
    NTU_METADATA_DIR,
)

from Scripts.NTU_RGBD.core import (
    get_rgb_video_info,
    match_rgb_and_skeleton,
    read_skeleton_summary,
)


# ============================================================
# 3. Config
# ============================================================

MAX_SAMPLES = None

FRAME_COUNT_TOLERANCE = 2

MATCHED_CSV_PATH = (
    NTU_METADATA_DIR
    / "matched_samples.csv"
)

MISSING_RGB_PATH = (
    NTU_METADATA_DIR
    / "missing_rgb.txt"
)

MISSING_SKELETON_PATH = (
    NTU_METADATA_DIR
    / "missing_skeleton.txt"
)

SUMMARY_PATH = (
    NTU_METADATA_DIR
    / "matching_summary.txt"
)


# ============================================================
# 4. Helpers
# ============================================================

def save_text_list(
    output_path: Path,
    values: list[str],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for value in values:
            handle.write(
                value + "\n"
            )


def inspect_matched_sample(
    sample: dict,
) -> dict:
    rgb_path = Path(
        sample["rgb_path"]
    )

    skeleton_path = Path(
        sample["skeleton_path"]
    )

    rgb_info = get_rgb_video_info(
        rgb_path
    )

    skeleton_summary = (
        read_skeleton_summary(
            skeleton_path
        )
    )

    rgb_frames = int(
        rgb_info["frame_count"]
    )

    skeleton_frames = int(
        skeleton_summary[
            "num_frames"
        ]
    )

    frame_difference = abs(
        rgb_frames
        - skeleton_frames
    )

    row = dict(sample)

    row.update({
        "rgb_frames": rgb_frames,
        "skeleton_frames": (
            skeleton_frames
        ),
        "frame_difference": (
            frame_difference
        ),
        "frame_count_match": (
            frame_difference
            <= FRAME_COUNT_TOLERANCE
        ),
        "fps": rgb_info["fps"],
        "width": rgb_info["width"],
        "height": rgb_info["height"],
        "max_bodies": (
            skeleton_summary[
                "max_bodies"
            ]
        ),
        "empty_frames": (
            skeleton_summary[
                "empty_frames"
            ]
        ),
        "is_single_person": (
            skeleton_summary[
                "is_single_person"
            ]
        ),
    })

    return row


def save_matched_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_id",
        "setup",
        "camera",
        "performer",
        "replication",
        "action",
        "rgb_path",
        "skeleton_path",
        "rgb_frames",
        "skeleton_frames",
        "frame_difference",
        "frame_count_match",
        "fps",
        "width",
        "height",
        "max_bodies",
        "empty_frames",
        "is_single_person",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    NTU_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Match NTU RGB videos and skeleton files")
    print("=" * 70)

    print(f"RGB directory:      {NTU_RGB_DIR}")
    print(f"Skeleton directory: {NTU_SKELETON_DIR}")
    print()

    result = match_rgb_and_skeleton(
        rgb_root=NTU_RGB_DIR,
        skeleton_root=(
            NTU_SKELETON_DIR
        ),
        recursive=True,
    )

    matched_samples = result[
        "matched"
    ]

    missing_rgb = result[
        "missing_rgb"
    ]

    missing_skeleton = result[
        "missing_skeleton"
    ]

    if MAX_SAMPLES is not None:
        matched_samples = (
            matched_samples[
                :MAX_SAMPLES
            ]
        )

    print(
        f"RGB videos found: "
        f"{result['num_rgb']}"
    )

    print(
        f"Skeleton files found: "
        f"{result['num_skeleton']}"
    )

    print(
        f"Matched sample IDs: "
        f"{result['num_matched']}"
    )

    print(
        f"Missing RGB: "
        f"{len(missing_rgb)}"
    )

    print(
        f"Missing skeleton: "
        f"{len(missing_skeleton)}"
    )

    print()
    print("Inspecting matched samples...")

    inspected_rows: list[dict] = []

    failed_samples: list[
        tuple[str, str]
    ] = []

    total = len(
        matched_samples
    )

    for index, sample in enumerate(
        matched_samples,
        start=1,
    ):
        sample_id = sample[
            "sample_id"
        ]

        try:
            row = inspect_matched_sample(
                sample
            )

            inspected_rows.append(
                row
            )

            print(
                f"[{index}/{total}] "
                f"{sample_id} | "
                f"RGB={row['rgb_frames']} | "
                f"Skeleton="
                f"{row['skeleton_frames']} | "
                f"Diff="
                f"{row['frame_difference']} | "
                f"Single="
                f"{row['is_single_person']}"
            )

        except Exception as error:
            failed_samples.append(
                (
                    sample_id,
                    str(error),
                )
            )

            print(
                f"[{index}/{total}] "
                f"{sample_id} | FAILED | "
                f"{error}"
            )

    save_matched_csv(
        inspected_rows,
        MATCHED_CSV_PATH,
    )

    save_text_list(
        MISSING_RGB_PATH,
        missing_rgb,
    )

    save_text_list(
        MISSING_SKELETON_PATH,
        missing_skeleton,
    )

    frame_match_count = sum(
        bool(row["frame_count_match"])
        for row in inspected_rows
    )

    single_person_count = sum(
        bool(row["is_single_person"])
        for row in inspected_rows
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "NTU RGB+D matching summary\n"
        )

        handle.write(
            "=" * 70 + "\n"
        )

        handle.write(
            f"RGB videos: "
            f"{result['num_rgb']}\n"
        )

        handle.write(
            f"Skeleton files: "
            f"{result['num_skeleton']}\n"
        )

        handle.write(
            f"Matched IDs: "
            f"{result['num_matched']}\n"
        )

        handle.write(
            f"Inspected matched samples: "
            f"{len(inspected_rows)}\n"
        )

        handle.write(
            f"Frame-count matches: "
            f"{frame_match_count}\n"
        )

        handle.write(
            f"Single-person samples: "
            f"{single_person_count}\n"
        )

        handle.write(
            f"Missing RGB: "
            f"{len(missing_rgb)}\n"
        )

        handle.write(
            f"Missing skeleton: "
            f"{len(missing_skeleton)}\n"
        )

        handle.write(
            f"Failed inspections: "
            f"{len(failed_samples)}\n"
        )

        if failed_samples:
            handle.write(
                "\nFailed samples:\n"
            )

            for sample_id, error in (
                failed_samples
            ):
                handle.write(
                    f"{sample_id}: "
                    f"{error}\n"
                )

    print()
    print("=" * 70)
    print("Matching finished")
    print("=" * 70)

    print(
        f"Matched CSV: "
        f"{MATCHED_CSV_PATH}"
    )

    print(
        f"Missing RGB list: "
        f"{MISSING_RGB_PATH}"
    )

    print(
        "Missing skeleton list: "
        f"{MISSING_SKELETON_PATH}"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()