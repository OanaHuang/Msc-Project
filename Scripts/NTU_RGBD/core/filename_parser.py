# Scripts/NTU_RGBD/core/filename_parser.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


# ============================================================
# 1. NTU filename pattern
# ============================================================

NTU_FILENAME_PATTERN = re.compile(
    r"(?P<sample_id>"
    r"S(?P<setup>\d{3})"
    r"C(?P<camera>\d{3})"
    r"P(?P<performer>\d{3})"
    r"R(?P<replication>\d{3})"
    r"A(?P<action>\d{3})"
    r")",
    flags=re.IGNORECASE,
)


# ============================================================
# 2. Parsed sample structure
# ============================================================

@dataclass(frozen=True, order=True)
class NTUSampleID:
    """
    Structured representation of one NTU RGB+D sample ID.
    """

    setup: int
    camera: int
    performer: int
    replication: int
    action: int

    def __post_init__(self) -> None:
        values = {
            "setup": self.setup,
            "camera": self.camera,
            "performer": self.performer,
            "replication": self.replication,
            "action": self.action,
        }

        for name, value in values.items():
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be int, got {type(value).__name__}"
                )

            if not 0 <= value <= 999:
                raise ValueError(
                    f"{name} must be between 0 and 999, got {value}"
                )

    @property
    def sample_id(self) -> str:
        """
        Return canonical NTU sample ID.

        Example:
            S001C001P001R001A001
        """
        return (
            f"S{self.setup:03d}"
            f"C{self.camera:03d}"
            f"P{self.performer:03d}"
            f"R{self.replication:03d}"
            f"A{self.action:03d}"
        )

    def to_dict(self) -> dict[str, int | str]:
        """
        Convert the parsed sample into metadata format.
        """
        return {
            "sample_id": self.sample_id,
            "setup": self.setup,
            "camera": self.camera,
            "performer": self.performer,
            "replication": self.replication,
            "action": self.action,
        }

    def __str__(self) -> str:
        return self.sample_id


# ============================================================
# 3. Parse filename
# ============================================================

def parse_ntu_filename(
    filename: str | Path,
) -> NTUSampleID:
    """
    Parse an NTU RGB+D filename or sample ID.

    Supported examples:

        S001C001P001R001A001
        S001C001P001R001A001_rgb.avi
        S001C001P001R001A001.skeleton
        /some/path/S001C001P001R001A001_rgb.avi

    Returns
    -------
    NTUSampleID
        Parsed setup, camera, performer, replication, and action.
    """
    filename = Path(filename).name

    match = NTU_FILENAME_PATTERN.search(
        filename
    )

    if match is None:
        raise ValueError(
            f"Could not parse NTU sample ID from filename: {filename}"
        )

    return NTUSampleID(
        setup=int(match.group("setup")),
        camera=int(match.group("camera")),
        performer=int(match.group("performer")),
        replication=int(match.group("replication")),
        action=int(match.group("action")),
    )


# ============================================================
# 4. Sample ID helper
# ============================================================

def get_sample_id(
    filename: str | Path,
) -> str:
    """
    Extract the canonical sample ID from a filename.

    Example:

        S001C001P001R001A001_rgb.avi

    becomes:

        S001C001P001R001A001
    """
    return parse_ntu_filename(
        filename
    ).sample_id


def is_ntu_sample_filename(
    filename: str | Path,
) -> bool:
    """
    Check whether a filename contains a valid NTU sample ID.
    """
    filename = Path(filename).name

    return (
        NTU_FILENAME_PATTERN.search(
            filename
        )
        is not None
    )


# ============================================================
# 5. Build filenames
# ============================================================

def build_rgb_filename(
    sample: NTUSampleID | str,
    extension: str = ".avi",
) -> str:
    """
    Build a standard NTU RGB video filename.

    Example:

        S001C001P001R001A001_rgb.avi
    """
    if isinstance(
        sample,
        NTUSampleID,
    ):
        sample_id = sample.sample_id
    else:
        sample_id = get_sample_id(
            sample
        )

    if not extension.startswith("."):
        extension = f".{extension}"

    return (
        f"{sample_id}_rgb"
        f"{extension.lower()}"
    )


def build_skeleton_filename(
    sample: NTUSampleID | str,
) -> str:
    """
    Build a standard NTU skeleton filename.

    Example:

        S001C001P001R001A001.skeleton
    """
    if isinstance(
        sample,
        NTUSampleID,
    ):
        sample_id = sample.sample_id
    else:
        sample_id = get_sample_id(
            sample
        )

    return f"{sample_id}.skeleton"


# ============================================================
# 6. Split helpers
# ============================================================

def is_same_sample(
    first_filename: str | Path,
    second_filename: str | Path,
) -> bool:
    """
    Check whether two files belong to the same NTU sample.

    Useful for checking an RGB/skeleton pair.
    """
    return (
        get_sample_id(first_filename)
        == get_sample_id(second_filename)
    )


def belongs_to_setup(
    filename: str | Path,
    setup_id: int,
) -> bool:
    """
    Check whether a sample belongs to a specific setup.
    """
    return (
        parse_ntu_filename(
            filename
        ).setup
        == setup_id
    )


def belongs_to_action(
    filename: str | Path,
    action_id: int,
) -> bool:
    """
    Check whether a sample belongs to a specific action.
    """
    return (
        parse_ntu_filename(
            filename
        ).action
        == action_id
    )


def belongs_to_performer(
    filename: str | Path,
    performer_id: int,
) -> bool:
    """
    Check whether a sample belongs to a specific performer.
    """
    return (
        parse_ntu_filename(
            filename
        ).performer
        == performer_id
    )


# ============================================================
# 7. Local test
# ============================================================

if __name__ == "__main__":
    test_filename = (
        "S001C001P001R001A001_rgb.avi"
    )

    sample = parse_ntu_filename(
        test_filename
    )

    print("=" * 70)
    print("NTU filename parser test")
    print("=" * 70)

    print(f"Filename:    {test_filename}")
    print(f"Sample ID:   {sample.sample_id}")
    print(f"Setup:       {sample.setup}")
    print(f"Camera:      {sample.camera}")
    print(f"Performer:   {sample.performer}")
    print(f"Replication: {sample.replication}")
    print(f"Action:      {sample.action}")

    print(
        "RGB filename:",
        build_rgb_filename(sample),
    )

    print(
        "Skeleton filename:",
        build_skeleton_filename(sample),
    )

    print("=" * 70)