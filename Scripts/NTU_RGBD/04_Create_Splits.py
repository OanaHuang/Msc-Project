# Scripts/NTU_RGBD/04_Create_Splits.py

from __future__ import annotations

from pathlib import Path
import csv
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Scripts.common.paths import NTU_METADATA_DIR


# ============================================================
# Config
# ============================================================

INPUT_CSV = NTU_METADATA_DIR / "matched_samples.csv"

TRAIN_CSV = NTU_METADATA_DIR / "train_split.csv"
VAL_CSV = NTU_METADATA_DIR / "val_split.csv"
TEST_CSV = NTU_METADATA_DIR / "test_split.csv"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SINGLE_PERSON_ONLY = True
FRAME_COUNT_MATCH_ONLY = True


def string_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    return rows


def save_rows(
    output_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def main() -> None:
    rows = load_rows(INPUT_CSV)

    if not rows:
        raise RuntimeError(
            "matched_samples.csv contains no samples"
        )

    filtered_rows = []

    for row in rows:
        if (
            SINGLE_PERSON_ONLY
            and not string_to_bool(
                row["is_single_person"]
            )
        ):
            continue

        if (
            FRAME_COUNT_MATCH_ONLY
            and not string_to_bool(
                row["frame_count_match"]
            )
        ):
            continue

        filtered_rows.append(row)

    if not filtered_rows:
        raise RuntimeError(
            "No valid samples remained after filtering"
        )

    # Important:
    # split at video/sample level, not at frame level.
    random_generator = random.Random(SEED)
    random_generator.shuffle(filtered_rows)

    total = len(filtered_rows)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_rows = filtered_rows[:train_end]
    val_rows = filtered_rows[train_end:val_end]
    test_rows = filtered_rows[val_end:]

    fieldnames = list(filtered_rows[0].keys())

    save_rows(
        TRAIN_CSV,
        train_rows,
        fieldnames,
    )

    save_rows(
        VAL_CSV,
        val_rows,
        fieldnames,
    )

    save_rows(
        TEST_CSV,
        test_rows,
        fieldnames,
    )

    print("=" * 70)
    print("NTU RGB+D split creation finished")
    print("=" * 70)

    print(f"Total valid samples: {total}")
    print(f"Train samples:       {len(train_rows)}")
    print(f"Validation samples:  {len(val_rows)}")
    print(f"Test samples:        {len(test_rows)}")

    print()
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val CSV:   {VAL_CSV}")
    print(f"Test CSV:  {TEST_CSV}")


if __name__ == "__main__":
    main()