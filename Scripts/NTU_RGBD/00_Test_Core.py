# Scripts/NTU_RGBD/00_Test_Core.py

from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from Scripts.NTU_RGBD.core import (
    parse_ntu_filename,
    get_sample_id,
    NTU_JOINT_NAMES,
    NTU_SKELETON_EDGES,
)


def main():
    filename = (
        "S001C001P001R001A001_rgb.avi"
    )

    sample = parse_ntu_filename(
        filename
    )

    print("=" * 70)
    print("NTU core test")
    print("=" * 70)

    print(
        f"Sample ID: {sample.sample_id}"
    )

    print(
        f"Setup: {sample.setup}"
    )

    print(
        f"Camera: {sample.camera}"
    )

    print(
        f"Performer: {sample.performer}"
    )

    print(
        f"Replication: {sample.replication}"
    )

    print(
        f"Action: {sample.action}"
    )

    print(
        f"Number of joints: "
        f"{len(NTU_JOINT_NAMES)}"
    )

    print(
        f"Number of edges: "
        f"{len(NTU_SKELETON_EDGES)}"
    )

    print(
        f"get_sample_id: "
        f"{get_sample_id(filename)}"
    )

    print("=" * 70)
    print("Core import test passed.")


if __name__ == "__main__":
    main()