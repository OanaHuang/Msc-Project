# Scripts/10_visualize_ntu_skeleton.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 从 NTU_RGBD_60 目录开始递归查找 .skeleton 文件
SKELETON_DIR = PROJECT_ROOT / "Datasets" / "NTU_RGBD_60"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ntu_skeleton_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# None 表示自动选择第一个 skeleton 文件
TARGET_FILE = None
# TARGET_FILE = "S001C001P001R001A001.skeleton"

FPS = 20


# ============================================================
# 2. NTU Skeleton Bones
# ============================================================

# NTU joint 原始编号是 1-25
# Python index 是 0-24
NTU_BONES = [
    (0, 1),      # spine base -> spine mid
    (1, 20),     # spine mid -> spine shoulder
    (20, 2),     # spine shoulder -> neck
    (2, 3),      # neck -> head

    (20, 4),     # spine shoulder -> left shoulder
    (4, 5),      # left shoulder -> left elbow
    (5, 6),      # left elbow -> left wrist
    (6, 7),      # left wrist -> left hand
    (7, 21),     # left hand -> left hand tip
    (6, 22),     # left wrist -> left thumb

    (20, 8),     # spine shoulder -> right shoulder
    (8, 9),      # right shoulder -> right elbow
    (9, 10),     # right elbow -> right wrist
    (10, 11),    # right wrist -> right hand
    (11, 23),    # right hand -> right hand tip
    (10, 24),    # right wrist -> right thumb

    (0, 12),     # spine base -> left hip
    (12, 13),    # left hip -> left knee
    (13, 14),    # left knee -> left ankle
    (14, 15),    # left ankle -> left foot

    (0, 16),     # spine base -> right hip
    (16, 17),    # right hip -> right knee
    (17, 18),    # right knee -> right ankle
    (18, 19),    # right ankle -> right foot
]


# ============================================================
# 3. Read NTU Skeleton File
# ============================================================

def read_skeleton_file(file_path):
    """
    Read one NTU RGB+D .skeleton file.

    Return:
        data shape = (num_frames, max_bodies, 25, 3)
    """

    with open(file_path, "r") as f:
        num_frames = int(f.readline())
        all_frames = []

        for _ in range(num_frames):
            num_bodies = int(f.readline())
            bodies = []

            for _ in range(num_bodies):
                # body information line
                f.readline()

                num_joints = int(f.readline())
                joints = []

                for _ in range(num_joints):
                    joint_info = f.readline().split()

                    # NTU joint format:
                    # x y z depthX depthY colorX colorY orientationW orientationX orientationY orientationZ trackingState
                    x = float(joint_info[0])
                    y = float(joint_info[1])
                    z = float(joint_info[2])

                    joints.append([x, y, z])

                bodies.append(joints)

            all_frames.append(bodies)

    max_bodies = max(len(frame) for frame in all_frames)

    data = np.zeros((num_frames, max_bodies, 25, 3), dtype=np.float32)

    for t, frame in enumerate(all_frames):
        for b, body in enumerate(frame):
            if len(body) == 25:
                data[t, b, :, :] = np.array(body, dtype=np.float32)

    return data


# ============================================================
# 4. Choose Main Body
# ============================================================

def choose_main_body(data):
    """
    Choose body 0 as main body.

    Input:
        data shape = (T, num_bodies, 25, 3)

    Output:
        sequence shape = (T, 25, 3)
    """

    return data[:, 0, :, :]


# ============================================================
# 5. Axis Limit Helper
# ============================================================

def get_axis_limits(sequence):
    """
    sequence shape = (T, 25, 3)
    """

    valid = sequence.reshape(-1, 3)

    # 去掉全 0 的无效点，避免画面比例被拉坏
    valid = valid[~np.all(valid == 0, axis=1)]

    if len(valid) == 0:
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.0
        return center, radius

    x_min, y_min, z_min = valid.min(axis=0)
    x_max, y_max, z_max = valid.max(axis=0)

    center = np.array([
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        (z_min + z_max) / 2,
    ])

    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    )

    radius = max_range / 2

    if radius == 0:
        radius = 1.0

    return center, radius


# ============================================================
# 6. Visualize Skeleton
# ============================================================

def visualize_skeleton(sequence, output_path, title):
    """
    Save skeleton sequence as MP4.

    sequence shape = (T, 25, 3)
    """

    num_frames = sequence.shape[0]
    center, radius = get_axis_limits(sequence)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        ax.clear()

        joints = sequence[frame_idx]

        # 为了显示更自然，用 x, z, y 显示
        xs = joints[:, 0]
        ys = joints[:, 2]
        zs = joints[:, 1]

        ax.scatter(xs, ys, zs, s=20)

        for i, j in NTU_BONES:
            # 如果两个点都是 0，跳过
            if np.all(joints[i] == 0) or np.all(joints[j] == 0):
                continue

            ax.plot(
                [joints[i, 0], joints[j, 0]],
                [joints[i, 2], joints[j, 2]],
                [joints[i, 1], joints[j, 1]],
                linewidth=2,
            )

        ax.set_title(f"{title}\nFrame {frame_idx + 1}/{num_frames}")

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)

        ax.view_init(elev=15, azim=-70)

    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / FPS,
        repeat=True,
    )

    print(f"Saving MP4 to: {output_path}")

    writer = FFMpegWriter(
        fps=FPS,
        metadata={"artist": "NTU Skeleton Visualization"},
        bitrate=1800,
    )

    ani.save(output_path, writer=writer)

    plt.close(fig)

    print("Saved successfully.")


# ============================================================
# 7. Main
# ============================================================

def main():
    skeleton_files = sorted(SKELETON_DIR.rglob("*.skeleton"))

    print("=" * 60)
    print("NTU Skeleton Visualization")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Skeleton directory: {SKELETON_DIR}")
    print(f"Number of skeleton files: {len(skeleton_files)}")

    if len(skeleton_files) == 0:
        print("No .skeleton files found.")
        return

    if TARGET_FILE is None:
        skeleton_path = skeleton_files[0]
    else:
        matches = [p for p in skeleton_files if p.name == TARGET_FILE]
        if len(matches) == 0:
            print(f"Target file not found: {TARGET_FILE}")
            return
        skeleton_path = matches[0]

    print(f"Reading: {skeleton_path}")

    data = read_skeleton_file(skeleton_path)
    sequence = choose_main_body(data)

    print(f"Raw data shape: {data.shape}")
    print(f"Main body sequence shape: {sequence.shape}")

    output_path = OUTPUT_DIR / f"{skeleton_path.stem}_skeleton.mp4"

    visualize_skeleton(
        sequence=sequence,
        output_path=output_path,
        title=skeleton_path.stem,
    )


if __name__ == "__main__":
    main()