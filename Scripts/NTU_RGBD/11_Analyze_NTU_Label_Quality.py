# Scripts/NTU_RGBD/11_Analyze_NTU_Label_Quality.py

from __future__ import annotations

from pathlib import Path
import csv
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from core.skeleton_reader import (
    NTUSkeletonSequence,
    read_skeleton_file,
)


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SKELETON_ROOT = (
    PROJECT_ROOT
    / "Datasets"
    / "NTU_RGBD"
    / "skeletons"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "NTU_RGBD"
    / "11_Analyze_NTU_Label_Quality"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PER_JOINT_CSV = (
    OUTPUT_DIR
    / "ntu_label_quality_per_joint.csv"
)

PER_VIDEO_CSV = (
    OUTPUT_DIR
    / "ntu_label_quality_per_video.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "ntu_label_quality_summary.txt"
)

TRACKING_STATE_PLOT = (
    OUTPUT_DIR
    / "ntu_tracking_state_by_joint.png"
)

INFERRED_RATIO_PLOT = (
    OUTPUT_DIR
    / "ntu_inferred_ratio_by_joint.png"
)

VIDEO_QUALITY_PLOT = (
    OUTPUT_DIR
    / "ntu_video_quality_distribution.png"
)


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

# None:
#   分析全部 skeleton 文件
#
# 例如：
# MAX_FILES = 100
#   随机分析 100 个视频
MAX_FILES: int | None = 100

RANDOM_SEED = 42

# False:
#   每个视频统计所有 body
#
# True:
#   每个视频只统计出现帧数最多的主体
#
# 你的单人姿态估计任务建议设为 True
PRIMARY_BODY_ONLY = True

# 是否跳过没有 body 的帧
SKIP_EMPTY_FRAMES = True


# ============================================================
# 2. Joint definitions
# ============================================================

NUM_JOINTS = 25

JOINT_NAMES = [
    "SpineBase",          # 0
    "SpineMid",           # 1
    "Neck",               # 2
    "Head",               # 3
    "ShoulderLeft",       # 4
    "ElbowLeft",          # 5
    "WristLeft",          # 6
    "HandLeft",           # 7
    "ShoulderRight",      # 8
    "ElbowRight",         # 9
    "WristRight",         # 10
    "HandRight",          # 11
    "HipLeft",            # 12
    "KneeLeft",           # 13
    "AnkleLeft",          # 14
    "FootLeft",           # 15
    "HipRight",           # 16
    "KneeRight",          # 17
    "AnkleRight",         # 18
    "FootRight",          # 19
    "SpineShoulder",      # 20
    "HandTipLeft",        # 21
    "ThumbLeft",          # 22
    "HandTipRight",       # 23
    "ThumbRight",         # 24
]


# ============================================================
# 3. File discovery
# ============================================================

def find_skeleton_files(
    root: Path,
) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(
            f"Skeleton root does not exist:\n{root}"
        )

    files = sorted(
        set(
            list(root.rglob("*.skeleton"))
            + list(root.rglob("*.SKELETON"))
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No .skeleton files found under:\n{root}"
        )

    return files


def sample_files(
    files: list[Path],
) -> list[Path]:
    if MAX_FILES is None:
        return files

    sample_count = min(
        MAX_FILES,
        len(files),
    )

    generator = random.Random(
        RANDOM_SEED
    )

    return sorted(
        generator.sample(
            files,
            sample_count,
        )
    )


# ============================================================
# 4. Body selection
# ============================================================

def count_body_presence(
    sequence: NTUSkeletonSequence,
) -> dict[str, int]:
    presence: dict[str, int] = {}

    for frame in sequence.frames:
        for body in frame.bodies:
            presence[body.body_id] = (
                presence.get(
                    body.body_id,
                    0,
                )
                + 1
            )

    return presence


def choose_primary_body_id(
    sequence: NTUSkeletonSequence,
) -> str | None:
    presence = count_body_presence(
        sequence
    )

    if not presence:
        return None

    return max(
        presence.items(),
        key=lambda item: item[1],
    )[0]


# ============================================================
# 5. Per-video analysis
# ============================================================

def analyse_sequence(
    sequence: NTUSkeletonSequence,
) -> dict:
    joint_state_counts = np.zeros(
        (
            NUM_JOINTS,
            3,
        ),
        dtype=np.int64,
    )

    selected_body_id = None

    if PRIMARY_BODY_ONLY:
        selected_body_id = (
            choose_primary_body_id(
                sequence
            )
        )

    total_frames = sequence.num_frames
    nonempty_frames = 0
    analysed_body_frames = 0
    missing_primary_body_frames = 0

    tracked_joints_per_frame: list[int] = []
    inferred_joints_per_frame: list[int] = []
    not_tracked_joints_per_frame: list[int] = []

    body_count_per_frame: list[int] = []

    for frame in sequence.frames:
        body_count = len(
            frame.bodies
        )

        body_count_per_frame.append(
            body_count
        )

        if body_count == 0:
            if SKIP_EMPTY_FRAMES:
                continue

        else:
            nonempty_frames += 1

        if PRIMARY_BODY_ONLY:
            body = next(
                (
                    candidate
                    for candidate in frame.bodies
                    if candidate.body_id
                    == selected_body_id
                ),
                None,
            )

            if body is None:
                missing_primary_body_frames += 1
                continue

            bodies_to_analyse = [body]

        else:
            bodies_to_analyse = (
                frame.bodies
            )

        for body in bodies_to_analyse:
            arrays = body.joint_arrays()

            tracking_state = arrays[
                "tracking_state"
            ].astype(np.int64)

            if len(tracking_state) < NUM_JOINTS:
                continue

            tracking_state = (
                tracking_state[:NUM_JOINTS]
            )

            tracked_count = int(
                np.sum(
                    tracking_state == 2
                )
            )

            inferred_count = int(
                np.sum(
                    tracking_state == 1
                )
            )

            not_tracked_count = int(
                np.sum(
                    tracking_state == 0
                )
            )

            tracked_joints_per_frame.append(
                tracked_count
            )

            inferred_joints_per_frame.append(
                inferred_count
            )

            not_tracked_joints_per_frame.append(
                not_tracked_count
            )

            analysed_body_frames += 1

            for joint_index in range(
                NUM_JOINTS
            ):
                state = int(
                    tracking_state[
                        joint_index
                    ]
                )

                if state not in (
                    0,
                    1,
                    2,
                ):
                    state = 0

                joint_state_counts[
                    joint_index,
                    state,
                ] += 1

    total_joint_observations = int(
        joint_state_counts.sum()
    )

    tracked_total = int(
        joint_state_counts[:, 2].sum()
    )

    inferred_total = int(
        joint_state_counts[:, 1].sum()
    )

    not_tracked_total = int(
        joint_state_counts[:, 0].sum()
    )

    if total_joint_observations > 0:
        tracked_ratio = (
            tracked_total
            / total_joint_observations
        )

        inferred_ratio = (
            inferred_total
            / total_joint_observations
        )

        not_tracked_ratio = (
            not_tracked_total
            / total_joint_observations
        )

    else:
        tracked_ratio = 0.0
        inferred_ratio = 0.0
        not_tracked_ratio = 0.0

    return {
        "selected_body_id": (
            selected_body_id
        ),

        "total_frames": total_frames,

        "nonempty_frames": (
            nonempty_frames
        ),

        "analysed_body_frames": (
            analysed_body_frames
        ),

        "missing_primary_body_frames": (
            missing_primary_body_frames
        ),

        "max_bodies": max(
            body_count_per_frame,
            default=0,
        ),

        "mean_bodies_per_frame": (
            float(
                np.mean(
                    body_count_per_frame
                )
            )
            if body_count_per_frame
            else 0.0
        ),

        "mean_tracked_joints": (
            float(
                np.mean(
                    tracked_joints_per_frame
                )
            )
            if tracked_joints_per_frame
            else 0.0
        ),

        "mean_inferred_joints": (
            float(
                np.mean(
                    inferred_joints_per_frame
                )
            )
            if inferred_joints_per_frame
            else 0.0
        ),

        "mean_not_tracked_joints": (
            float(
                np.mean(
                    not_tracked_joints_per_frame
                )
            )
            if not_tracked_joints_per_frame
            else 0.0
        ),

        "tracked_ratio": tracked_ratio,

        "inferred_ratio": inferred_ratio,

        "not_tracked_ratio": (
            not_tracked_ratio
        ),

        "joint_state_counts": (
            joint_state_counts
        ),
    }


# ============================================================
# 6. CSV writers
# ============================================================

def write_per_video_csv(
    rows: list[dict],
) -> None:
    fieldnames = [
        "sample_id",
        "skeleton_path",
        "status",
        "selected_body_id",
        "total_frames",
        "nonempty_frames",
        "analysed_body_frames",
        "missing_primary_body_frames",
        "max_bodies",
        "mean_bodies_per_frame",
        "mean_tracked_joints",
        "mean_inferred_joints",
        "mean_not_tracked_joints",
        "tracked_ratio",
        "inferred_ratio",
        "not_tracked_ratio",
        "error",
    ]

    with PER_VIDEO_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_per_joint_csv(
    global_counts: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []

    for joint_index in range(
        NUM_JOINTS
    ):
        not_tracked_count = int(
            global_counts[
                joint_index,
                0,
            ]
        )

        inferred_count = int(
            global_counts[
                joint_index,
                1,
            ]
        )

        tracked_count = int(
            global_counts[
                joint_index,
                2,
            ]
        )

        total_count = (
            not_tracked_count
            + inferred_count
            + tracked_count
        )

        if total_count > 0:
            tracked_ratio = (
                tracked_count
                / total_count
            )

            inferred_ratio = (
                inferred_count
                / total_count
            )

            not_tracked_ratio = (
                not_tracked_count
                / total_count
            )

        else:
            tracked_ratio = 0.0
            inferred_ratio = 0.0
            not_tracked_ratio = 0.0

        reliability_score = (
            tracked_ratio
            + 0.5 * inferred_ratio
        )

        rows.append({
            "joint_index": joint_index,
            "joint_name": (
                JOINT_NAMES[
                    joint_index
                ]
            ),
            "total_observations": (
                total_count
            ),
            "tracked_count": (
                tracked_count
            ),
            "inferred_count": (
                inferred_count
            ),
            "not_tracked_count": (
                not_tracked_count
            ),
            "tracked_ratio": (
                tracked_ratio
            ),
            "inferred_ratio": (
                inferred_ratio
            ),
            "not_tracked_ratio": (
                not_tracked_ratio
            ),
            "reliability_score": (
                reliability_score
            ),
        })

    fieldnames = [
        "joint_index",
        "joint_name",
        "total_observations",
        "tracked_count",
        "inferred_count",
        "not_tracked_count",
        "tracked_ratio",
        "inferred_ratio",
        "not_tracked_ratio",
        "reliability_score",
    ]

    with PER_JOINT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    return rows


# ============================================================
# 7. Plots
# ============================================================

def plot_tracking_states(
    per_joint_rows: list[dict],
) -> None:
    labels = [
        (
            f"{row['joint_index']}\n"
            f"{row['joint_name']}"
        )
        for row in per_joint_rows
    ]

    tracked = np.asarray([
        row["tracked_ratio"] * 100.0
        for row in per_joint_rows
    ])

    inferred = np.asarray([
        row["inferred_ratio"] * 100.0
        for row in per_joint_rows
    ])

    not_tracked = np.asarray([
        row["not_tracked_ratio"] * 100.0
        for row in per_joint_rows
    ])

    x = np.arange(
        len(labels)
    )

    figure, axis = plt.subplots(
        figsize=(18, 8)
    )

    axis.bar(
        x,
        tracked,
        label="Tracked",
    )

    axis.bar(
        x,
        inferred,
        bottom=tracked,
        label="Inferred",
    )

    axis.bar(
        x,
        not_tracked,
        bottom=tracked + inferred,
        label="Not tracked",
    )

    axis.set_title(
        "NTU RGB+D Tracking State by Joint"
    )

    axis.set_ylabel(
        "Percentage (%)"
    )

    axis.set_xlabel(
        "Joint"
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.set_ylim(
        0,
        100,
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        TRACKING_STATE_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_inferred_ratio(
    per_joint_rows: list[dict],
) -> None:
    sorted_rows = sorted(
        per_joint_rows,
        key=lambda row: row[
            "inferred_ratio"
        ],
        reverse=True,
    )

    labels = [
        (
            f"{row['joint_index']} "
            f"{row['joint_name']}"
        )
        for row in sorted_rows
    ]

    values = [
        row["inferred_ratio"] * 100.0
        for row in sorted_rows
    ]

    figure, axis = plt.subplots(
        figsize=(14, 8)
    )

    axis.barh(
        labels,
        values,
    )

    axis.invert_yaxis()

    axis.set_title(
        "NTU RGB+D Inferred-Joint Ratio"
    )

    axis.set_xlabel(
        "Inferred observations (%)"
    )

    axis.set_ylabel(
        "Joint"
    )

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        INFERRED_RATIO_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_video_quality(
    per_video_rows: list[dict],
) -> None:
    successful_rows = [
        row
        for row in per_video_rows
        if row["status"] == "ok"
    ]

    tracked_ratios = [
        float(
            row["tracked_ratio"]
        ) * 100.0
        for row in successful_rows
    ]

    inferred_ratios = [
        float(
            row["inferred_ratio"]
        ) * 100.0
        for row in successful_rows
    ]

    if not tracked_ratios:
        return

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.scatter(
        tracked_ratios,
        inferred_ratios,
        alpha=0.65,
    )

    axis.set_title(
        "Per-video NTU Skeleton Quality"
    )

    axis.set_xlabel(
        "Tracked joints (%)"
    )

    axis.set_ylabel(
        "Inferred joints (%)"
    )

    axis.grid(
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        VIDEO_QUALITY_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# 8. Summary
# ============================================================

def create_summary(
    files: list[Path],
    per_video_rows: list[dict],
    per_joint_rows: list[dict],
) -> str:
    successful_rows = [
        row
        for row in per_video_rows
        if row["status"] == "ok"
    ]

    failed_rows = [
        row
        for row in per_video_rows
        if row["status"] != "ok"
    ]

    if successful_rows:
        overall_tracked = float(
            np.mean([
                float(
                    row["tracked_ratio"]
                )
                for row in successful_rows
            ])
        )

        overall_inferred = float(
            np.mean([
                float(
                    row["inferred_ratio"]
                )
                for row in successful_rows
            ])
        )

        overall_not_tracked = float(
            np.mean([
                float(
                    row[
                        "not_tracked_ratio"
                    ]
                )
                for row in successful_rows
            ])
        )

        mean_tracked_joints = float(
            np.mean([
                float(
                    row[
                        "mean_tracked_joints"
                    ]
                )
                for row in successful_rows
            ])
        )

    else:
        overall_tracked = 0.0
        overall_inferred = 0.0
        overall_not_tracked = 0.0
        mean_tracked_joints = 0.0

    worst_inferred = sorted(
        per_joint_rows,
        key=lambda row: row[
            "inferred_ratio"
        ],
        reverse=True,
    )[:8]

    lowest_reliability = sorted(
        per_joint_rows,
        key=lambda row: row[
            "reliability_score"
        ],
    )[:8]

    lines = [
        "=" * 72,
        "NTU RGB+D Skeleton Label Quality Summary",
        "=" * 72,
        "",
        f"Skeleton root: {SKELETON_ROOT}",
        f"Requested files: {len(files)}",
        f"Successful files: {len(successful_rows)}",
        f"Failed files: {len(failed_rows)}",
        f"Primary body only: {PRIMARY_BODY_ONLY}",
        "",
        "Overall tracking-state statistics",
        "-" * 72,
        (
            f"Mean tracked ratio: "
            f"{overall_tracked * 100:.2f}%"
        ),
        (
            f"Mean inferred ratio: "
            f"{overall_inferred * 100:.2f}%"
        ),
        (
            f"Mean not-tracked ratio: "
            f"{overall_not_tracked * 100:.2f}%"
        ),
        (
            f"Mean tracked joints per body frame: "
            f"{mean_tracked_joints:.2f}/{NUM_JOINTS}"
        ),
        "",
        "Joints with highest inferred ratio",
        "-" * 72,
    ]

    for row in worst_inferred:
        lines.append(
            (
                f"{row['joint_index']:>2} "
                f"{row['joint_name']:<18} "
                f"inferred="
                f"{row['inferred_ratio'] * 100:>6.2f}% "
                f"tracked="
                f"{row['tracked_ratio'] * 100:>6.2f}%"
            )
        )

    lines.extend([
        "",
        "Joints with lowest reliability score",
        "-" * 72,
    ])

    for row in lowest_reliability:
        lines.append(
            (
                f"{row['joint_index']:>2} "
                f"{row['joint_name']:<18} "
                f"score="
                f"{row['reliability_score']:.4f} "
                f"tracked="
                f"{row['tracked_ratio'] * 100:>6.2f}% "
                f"inferred="
                f"{row['inferred_ratio'] * 100:>6.2f}%"
            )
        )

    lines.extend([
        "",
        "Interpretation",
        "-" * 72,
        (
            "Tracked joints have tracking_state == 2."
        ),
        (
            "Inferred joints have tracking_state == 1 "
            "and may provide less reliable supervision."
        ),
        (
            "Reliability score = tracked ratio "
            "+ 0.5 * inferred ratio."
        ),
        (
            "A high inferred ratio does not directly "
            "measure pixel error, but indicates that the "
            "joint was frequently estimated rather than "
            "fully tracked by Kinect."
        ),
    ])

    return "\n".join(
        lines
    )


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    print("=" * 72)
    print("NTU RGB+D Skeleton Label Quality Analysis")
    print("=" * 72)

    all_files = find_skeleton_files(
        SKELETON_ROOT
    )

    selected_files = sample_files(
        all_files
    )

    print(
        f"\nSkeleton files found: "
        f"{len(all_files)}"
    )

    print(
        f"Files selected:        "
        f"{len(selected_files)}"
    )

    print(
        f"Primary body only:     "
        f"{PRIMARY_BODY_ONLY}"
    )

    global_joint_counts = np.zeros(
        (
            NUM_JOINTS,
            3,
        ),
        dtype=np.int64,
    )

    per_video_rows: list[dict] = []

    error_counter: Counter[str] = Counter()

    for file_index, skeleton_path in enumerate(
        selected_files,
        start=1,
    ):
        sample_id = skeleton_path.stem

        try:
            sequence = read_skeleton_file(
                skeleton_path
            )

            result = analyse_sequence(
                sequence
            )

            global_joint_counts += result[
                "joint_state_counts"
            ]

            row = {
                "sample_id": sample_id,
                "skeleton_path": str(
                    skeleton_path
                ),
                "status": "ok",
                "selected_body_id": (
                    result[
                        "selected_body_id"
                    ]
                    or ""
                ),
                "total_frames": (
                    result[
                        "total_frames"
                    ]
                ),
                "nonempty_frames": (
                    result[
                        "nonempty_frames"
                    ]
                ),
                "analysed_body_frames": (
                    result[
                        "analysed_body_frames"
                    ]
                ),
                "missing_primary_body_frames": (
                    result[
                        "missing_primary_body_frames"
                    ]
                ),
                "max_bodies": (
                    result[
                        "max_bodies"
                    ]
                ),
                "mean_bodies_per_frame": (
                    f"{result['mean_bodies_per_frame']:.6f}"
                ),
                "mean_tracked_joints": (
                    f"{result['mean_tracked_joints']:.6f}"
                ),
                "mean_inferred_joints": (
                    f"{result['mean_inferred_joints']:.6f}"
                ),
                "mean_not_tracked_joints": (
                    f"{result['mean_not_tracked_joints']:.6f}"
                ),
                "tracked_ratio": (
                    f"{result['tracked_ratio']:.8f}"
                ),
                "inferred_ratio": (
                    f"{result['inferred_ratio']:.8f}"
                ),
                "not_tracked_ratio": (
                    f"{result['not_tracked_ratio']:.8f}"
                ),
                "error": "",
            }

        except Exception as error:
            error_name = type(
                error
            ).__name__

            error_counter[
                error_name
            ] += 1

            row = {
                "sample_id": sample_id,
                "skeleton_path": str(
                    skeleton_path
                ),
                "status": "error",
                "selected_body_id": "",
                "total_frames": "",
                "nonempty_frames": "",
                "analysed_body_frames": "",
                "missing_primary_body_frames": "",
                "max_bodies": "",
                "mean_bodies_per_frame": "",
                "mean_tracked_joints": "",
                "mean_inferred_joints": "",
                "mean_not_tracked_joints": "",
                "tracked_ratio": "",
                "inferred_ratio": "",
                "not_tracked_ratio": "",
                "error": (
                    f"{error_name}: {error}"
                ),
            }

        per_video_rows.append(
            row
        )

        if (
            file_index == 1
            or file_index % 10 == 0
            or file_index
            == len(selected_files)
        ):
            print(
                f"Processed "
                f"{file_index}/"
                f"{len(selected_files)}"
            )

    write_per_video_csv(
        per_video_rows
    )

    per_joint_rows = (
        write_per_joint_csv(
            global_joint_counts
        )
    )

    plot_tracking_states(
        per_joint_rows
    )

    plot_inferred_ratio(
        per_joint_rows
    )

    plot_video_quality(
        per_video_rows
    )

    summary = create_summary(
        selected_files,
        per_video_rows,
        per_joint_rows,
    )

    with SUMMARY_TXT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            summary
        )

    print("\n")
    print(summary)

    if error_counter:
        print("\nErrors")
        print("-" * 72)

        for error_name, count in (
            error_counter.most_common()
        ):
            print(
                f"{error_name}: {count}"
            )

    print("\nOutputs")
    print("-" * 72)

    print(
        f"Per-joint CSV:\n  "
        f"{PER_JOINT_CSV}"
    )

    print(
        f"\nPer-video CSV:\n  "
        f"{PER_VIDEO_CSV}"
    )

    print(
        f"\nSummary:\n  "
        f"{SUMMARY_TXT}"
    )

    print(
        f"\nTracking-state plot:\n  "
        f"{TRACKING_STATE_PLOT}"
    )

    print(
        f"\nInferred-ratio plot:\n  "
        f"{INFERRED_RATIO_PLOT}"
    )

    print(
        f"\nVideo-quality plot:\n  "
        f"{VIDEO_QUALITY_PLOT}"
    )


if __name__ == "__main__":
    main()