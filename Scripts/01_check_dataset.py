from pathlib import Path
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Datasets" / "Penn_Action"

frames_dir = DATASET / "frames"
labels_dir = DATASET / "labels"

print("Dataset path:", DATASET)
print("frames exists:", frames_dir.exists())
print("labels exists:", labels_dir.exists())

frame_folders = sorted([p for p in frames_dir.iterdir() if p.is_dir()])
label_files = sorted(labels_dir.glob("*.mat"))

print("Number of video folders:", len(frame_folders))
print("Number of label files:", len(label_files))

sample_video = frame_folders[0]
sample_label = labels_dir / f"{sample_video.name}.mat"

print("\nSample video:", sample_video.name)
print("Number of frames:", len(list(sample_video.glob("*.jpg"))))
print("Label file exists:", sample_label.exists())

mat = sio.loadmat(sample_label)
print("\nMAT keys:")
for k in mat.keys():
    if not k.startswith("__"):
        v = mat[k]
        print(k, type(v), getattr(v, "shape", None))