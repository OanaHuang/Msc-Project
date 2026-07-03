from pathlib import Path
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Datasets" / "Penn_Action"

video_id = "0051"
frame_idx = 0

frame_path = DATASET / "frames" / video_id / f"{frame_idx + 1:06d}.jpg"
label_path = DATASET / "labels" / f"{video_id}.mat"

img = Image.open(frame_path).convert("RGB")
mat = sio.loadmat(label_path)

x = mat["x"]
y = mat["y"]
visibility = mat["visibility"]

# Penn Action 13 keypoints:
# 0 head, 1 left_shoulder, 2 right_shoulder,
# 3 left_elbow, 4 right_elbow,
# 5 left_wrist, 6 right_wrist,
# 7 left_hip, 8 right_hip,
# 9 left_knee, 10 right_knee,
# 11 left_ankle, 12 right_ankle

skeleton = [
    (0, 1), (0, 2),          # head to shoulders
    (1, 3), (3, 5),          # left arm
    (2, 4), (4, 6),          # right arm
    (1, 7), (2, 8),          # torso
    (7, 8),                  # hips
    (7, 9), (9, 11),         # left leg
    (8, 10), (10, 12),       # right leg
]

plt.figure(figsize=(8, 6))
plt.imshow(img)

# draw skeleton lines
for j1, j2 in skeleton:
    if visibility[frame_idx, j1] > 0 and visibility[frame_idx, j2] > 0:
        plt.plot(
            [x[frame_idx, j1], x[frame_idx, j2]],
            [y[frame_idx, j1], y[frame_idx, j2]],
            linewidth=2,
        )

# draw keypoints
for j in range(x.shape[1]):
    if visibility[frame_idx, j] > 0:
        plt.scatter(x[frame_idx, j], y[frame_idx, j], s=35)
        plt.text(x[frame_idx, j], y[frame_idx, j], str(j), fontsize=9)

plt.title(f"Penn Action Video {video_id}, Frame {frame_idx + 1}")
plt.axis("off")
plt.show()