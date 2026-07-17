# Scripts/NTU_RGBD/core/sample_matcher.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .filename_parser import (
    NTUSampleID,
    get_sample_id,
    parse_ntu_filename,
)


RGB_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
}

SKELETON_EXTENSION = ".skeleton"


# ============================================================
# 1. Index RGB files
# ============================================================

def index_rgb_files(
    rgb_root: str | Path,
    recursive: bool = True,
) -> dict[str, Path]:
    """
    Create an index:

        sample_id -> RGB video path
    """
    rgb_root = Path(rgb_root)

    if not rgb_root.exists():
        raise FileNotFoundError(
            f"RGB directory does not exist: {rgb_root}"
        )

    iterator: Iterable[Path]

    if recursive:
        iterator = rgb_root.rglob("*")
    else:
        iterator = rgb_root.glob("*")

    index: dict[str, Path] = {}

    for path in iterator:
        if not path.is_file():
            continue

        if path.suffix.lower() not in RGB_EXTENSIONS:
            continue

        try:
            sample_id = get_sample_id(
                path.name
            )
        except ValueError:
            continue

        if sample_id in index:
            raise RuntimeError(
                f"Duplicate RGB sample ID "
                f"{sample_id}:\n"
                f"  {index[sample_id]}\n"
                f"  {path}"
            )

        index[sample_id] = path

    return index


# ============================================================
# 2. Index skeleton files
# ============================================================

def index_skeleton_files(
    skeleton_root: str | Path,
    recursive: bool = True,
) -> dict[str, Path]:
    """
    Create an index:

        sample_id -> skeleton path
    """
    skeleton_root = Path(
        skeleton_root
    )

    if not skeleton_root.exists():
        raise FileNotFoundError(
            f"Skeleton directory does not exist: "
            f"{skeleton_root}"
        )

    if recursive:
        iterator = skeleton_root.rglob(
            f"*{SKELETON_EXTENSION}"
        )
    else:
        iterator = skeleton_root.glob(
            f"*{SKELETON_EXTENSION}"
        )

    index: dict[str, Path] = {}

    for path in iterator:
        if not path.is_file():
            continue

        try:
            sample_id = get_sample_id(
                path.name
            )
        except ValueError:
            continue

        if sample_id in index:
            raise RuntimeError(
                f"Duplicate skeleton sample ID "
                f"{sample_id}:\n"
                f"  {index[sample_id]}\n"
                f"  {path}"
            )

        index[sample_id] = path

    return index


# ============================================================
# 3. Match RGB and skeleton files
# ============================================================

def match_rgb_and_skeleton(
    rgb_root: str | Path,
    skeleton_root: str | Path,
    recursive: bool = True,
) -> dict[str, object]:
    """
    Match RGB videos and skeleton files by NTU sample ID.

    Returns
    -------
    dict
        matched:
            list of metadata dictionaries

        missing_rgb:
            sample IDs that have skeleton but no RGB video

        missing_skeleton:
            sample IDs that have RGB video but no skeleton
    """
    rgb_index = index_rgb_files(
        rgb_root,
        recursive=recursive,
    )

    skeleton_index = index_skeleton_files(
        skeleton_root,
        recursive=recursive,
    )

    rgb_ids = set(
        rgb_index.keys()
    )

    skeleton_ids = set(
        skeleton_index.keys()
    )

    matched_ids = sorted(
        rgb_ids & skeleton_ids
    )

    missing_rgb = sorted(
        skeleton_ids - rgb_ids
    )

    missing_skeleton = sorted(
        rgb_ids - skeleton_ids
    )

    matched: list[dict] = []

    for sample_id in matched_ids:
        parsed = parse_ntu_filename(
            sample_id
        )

        row = parsed.to_dict()

        row.update({
            "rgb_path": str(
                rgb_index[sample_id]
            ),
            "skeleton_path": str(
                skeleton_index[sample_id]
            ),
        })

        matched.append(row)

    return {
        "matched": matched,
        "missing_rgb": missing_rgb,
        "missing_skeleton": missing_skeleton,
        "num_rgb": len(rgb_index),
        "num_skeleton": len(
            skeleton_index
        ),
        "num_matched": len(matched),
    }


# ============================================================
# 4. Filter indexed samples
# ============================================================

def filter_samples(
    samples: list[dict],
    setups: set[int] | None = None,
    cameras: set[int] | None = None,
    performers: set[int] | None = None,
    actions: set[int] | None = None,
    replications: set[int] | None = None,
) -> list[dict]:
    """
    Filter matched sample metadata.
    """
    filtered: list[dict] = []

    for sample in samples:
        if (
            setups is not None
            and sample["setup"] not in setups
        ):
            continue

        if (
            cameras is not None
            and sample["camera"] not in cameras
        ):
            continue

        if (
            performers is not None
            and sample["performer"]
            not in performers
        ):
            continue

        if (
            actions is not None
            and sample["action"] not in actions
        ):
            continue

        if (
            replications is not None
            and sample["replication"]
            not in replications
        ):
            continue

        filtered.append(sample)

    return filtered


# ============================================================
# 5. Build one metadata row
# ============================================================

def build_sample_metadata(
    rgb_path: str | Path,
    skeleton_path: str | Path,
) -> dict:
    """
    Build one matched sample row from two paths.
    """
    rgb_path = Path(rgb_path)
    skeleton_path = Path(
        skeleton_path
    )

    rgb_id = get_sample_id(
        rgb_path.name
    )

    skeleton_id = get_sample_id(
        skeleton_path.name
    )

    if rgb_id != skeleton_id:
        raise ValueError(
            f"RGB and skeleton IDs do not match:\n"
            f"RGB:      {rgb_id}\n"
            f"Skeleton: {skeleton_id}"
        )

    parsed: NTUSampleID = (
        parse_ntu_filename(rgb_id)
    )

    row = parsed.to_dict()

    row.update({
        "rgb_path": str(rgb_path),
        "skeleton_path": str(
            skeleton_path
        ),
    })

    return row