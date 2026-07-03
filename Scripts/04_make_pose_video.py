from pathlib import Path
import cv2
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Datasets" / "Penn_Action"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

video_id = "0011"
fps = 30

frames_dir = DATASET / "frames" / video_id
label_path = DATASET / "labels" / f"{video_id}.mat"
output_path = OUTPUT_DIR / f"penn_action_{video_id}_pose.mp4"

mat = sio.loadmat(label_path)
x = mat["x"]
y = mat["y"]
visibility = mat["visibility"]

skeleton = [
    (0, 1), (0, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8),
    (7, 8),
    (7, 9), (9, 11),
    (8, 10), (10, 12),
]

frame_files = sorted(frames_dir.glob("*.jpg"))

first = cv2.imread(str(frame_files[0]))
height, width = first.shape[:2]

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

for i, frame_file in enumerate(frame_files):
    frame = cv2.imread(str(frame_file))

    if i >= x.shape[0]:
        break

    for j1, j2 in skeleton:
        if visibility[i, j1] > 0 and visibility[i, j2] > 0:
            p1 = (int(x[i, j1]), int(y[i, j1]))
            p2 = (int(x[i, j2]), int(y[i, j2]))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

    for j in range(x.shape[1]):
        if visibility[i, j] > 0:
            px, py = int(x[i, j]), int(y[i, j])
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(j),
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    writer.write(frame)

writer.release()

print("Saved video:", output_path)