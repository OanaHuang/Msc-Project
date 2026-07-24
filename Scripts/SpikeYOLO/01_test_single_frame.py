from pathlib import Path
import sys

from PIL import Image


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPIKEYOLO_ROOT = (
    PROJECT_ROOT
    / "External"
    / "SpikeYOLO"
)

WEIGHTS_PATH = (
    SPIKEYOLO_ROOT
    / "weights"
    / "best.pt"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "NTU_RGBD"
    / "extracted_frames"
    / "S015C001P007R001A001"
    / "frame_000000.jpg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "SpikeYOLO"
    / "01_single_frame"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Import SpikeYOLO local ultralytics
# ============================================================

sys.path.insert(0, str(SPIKEYOLO_ROOT))

from ultralytics import YOLO  # noqa: E402


# ============================================================
# 3. Main
# ============================================================

def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"SpikeYOLO weights not found:\n{WEIGHTS_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"NTU test frame not found:\n{IMAGE_PATH}"
        )

    print(f"SpikeYOLO root: {SPIKEYOLO_ROOT}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Input image: {IMAGE_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    model = YOLO(str(WEIGHTS_PATH))

    results = model(
        str(IMAGE_PATH),
        conf=0.25,
        classes=[0],
    )

    if not results:
        print("No result returned.")
        return

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("No person detected.")
    else:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        print(f"Detected persons: {len(boxes)}")

        for index, (box, score, class_id) in enumerate(
            zip(boxes, scores, classes),
            start=1,
        ):
            x1, y1, x2, y2 = box

            print(
                f"Person {index}: "
                f"bbox=({x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f}), "
                f"confidence={score:.4f}, "
                f"class_id={int(class_id)}"
            )

    plotted_bgr = result.plot()
    plotted_rgb = plotted_bgr[:, :, ::-1]

    output_path = OUTPUT_DIR / "result.jpg"
    Image.fromarray(plotted_rgb).save(output_path)

    print(f"Saved result to:\n{output_path}")


if __name__ == "__main__":
    main()