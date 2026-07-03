from pathlib import Path
import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Datasets" / "Penn_Action"
OUTPUT = DATASET / "penn_action_processed.npz"

frames_dir = DATASET / "frames"
labels_dir = DATASET / "labels"


def read_scalar(value, default=None):
    try:
        return value.item()
    except Exception:
        try:
            return value[0][0]
        except Exception:
            return default


def read_action(value):
    """
    Penn Action 的 action 可能是字符串数组，不一定是数字。
    """
    try:
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    except Exception:
        try:
            item = value[0][0]
            if isinstance(item, bytes):
                return item.decode("utf-8")
            return str(item)
        except Exception:
            return "unknown"


image_paths = []
keypoints = []
visibility_list = []
actions = []
video_ids = []
frame_indices = []
train_flags = []

label_files = sorted(labels_dir.glob("*.mat"))

print("Dataset:", DATASET)
print("Label files:", len(label_files))

for label_file in label_files:
    video_id = label_file.stem
    frame_folder = frames_dir / video_id

    if not frame_folder.exists():
        print(f"[Skip] Missing frame folder: {video_id}")
        continue

    mat = sio.loadmat(label_file)

    if "x" not in mat or "y" not in mat or "visibility" not in mat:
        print(f"[Skip] Missing x/y/visibility in: {label_file.name}")
        continue

    x = mat["x"].astype(np.float32)
    y = mat["y"].astype(np.float32)
    visibility = mat["visibility"].astype(np.float32)

    action = read_action(mat["action"]) if "action" in mat else "unknown"

    try:
        train = int(read_scalar(mat["train"], -1)) if "train" in mat else -1
    except Exception:
        train = -1

    num_frames, num_joints = x.shape

    for i in range(num_frames):
        img_path = frame_folder / f"{i + 1:06d}.jpg"

        if not img_path.exists():
            continue

        kp = np.stack([x[i], y[i]], axis=1).astype(np.float32)

        image_paths.append(str(img_path.relative_to(DATASET)))
        keypoints.append(kp)
        visibility_list.append(visibility[i].astype(np.float32))
        actions.append(action)
        video_ids.append(video_id)
        frame_indices.append(i)
        train_flags.append(train)

image_paths = np.array(image_paths)
keypoints = np.array(keypoints, dtype=np.float32)
visibility_list = np.array(visibility_list, dtype=np.float32)
actions = np.array(actions, dtype=str)
video_ids = np.array(video_ids)
frame_indices = np.array(frame_indices, dtype=np.int64)
train_flags = np.array(train_flags, dtype=np.int64)

np.savez_compressed(
    OUTPUT,
    image_paths=image_paths,
    keypoints=keypoints,
    visibility=visibility_list,
    actions=actions,
    video_ids=video_ids,
    frame_indices=frame_indices,
    train=train_flags,
)

print("\nSaved:", OUTPUT)
print("Total frames:", len(image_paths))
print("Keypoints shape:", keypoints.shape)
print("Visibility shape:", visibility_list.shape)
print("Actions shape:", actions.shape)
print("Unique actions:", sorted(set(actions.tolist())))
print("Train flags:", sorted(set(train_flags.tolist())))