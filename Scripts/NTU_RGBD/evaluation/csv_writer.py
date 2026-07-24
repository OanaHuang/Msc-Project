# Scripts/NTU_RGBD/evaluation/csv_writer.py

from __future__ import annotations

from pathlib import Path
import csv


# ============================================================
# 1. Generic CSV writer
# ============================================================

def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """
    Save a list of dictionaries to a CSV file.

    Column order follows the first occurrence of each key
    across all rows.

    Parameters
    ----------
    path:
        Output CSV path.

    rows:
        List of row dictionaries.
    """
    path = Path(
        path
    )

    if not rows:
        raise ValueError(
            f"没有可保存的数据：{path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# 2. Evaluation CSV group
# ============================================================

def save_evaluation_csvs(
    summary: dict[str, object],
    per_video_results: list[dict[str, object]],
    per_joint_results: list[dict[str, object]],
    summary_csv_path: Path,
    per_video_csv_path: Path,
    per_joint_csv_path: Path,
) -> None:
    """
    Save all evaluation result CSV files.

    Files
    -----
    summary_csv_path:
        One-row overall result.

    per_video_csv_path:
        One row per evaluated NPZ/video.

    per_joint_csv_path:
        One row per joint.
    """
    if not summary:
        raise ValueError(
            "summary cannot be empty"
        )

    if not per_video_results:
        raise ValueError(
            "per_video_results cannot be empty"
        )

    if not per_joint_results:
        raise ValueError(
            "per_joint_results cannot be empty"
        )

    save_csv(
        path=summary_csv_path,
        rows=[summary],
    )

    save_csv(
        path=per_video_csv_path,
        rows=per_video_results,
    )

    save_csv(
        path=per_joint_csv_path,
        rows=per_joint_results,
    )


# ============================================================
# 3. Console helper
# ============================================================

def print_saved_csv_paths(
    summary_csv_path: Path,
    per_video_csv_path: Path,
    per_joint_csv_path: Path,
) -> None:
    """
    Print the saved evaluation file paths.
    """
    print()
    print("Saved files:")

    print(
        Path(
            summary_csv_path
        )
    )

    print(
        Path(
            per_video_csv_path
        )
    )

    print(
        Path(
            per_joint_csv_path
        )
    )

    print("=" * 72)