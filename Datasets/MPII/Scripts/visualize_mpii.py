from pathlib import Path
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

MPII_ROOT = Path("/Users/oanahuang/Desktop/MSc Project/Datasets/MPII")

IMG_DIR = MPII_ROOT / "images"
MAT_PATH = MPII_ROOT / "Annotations" / "mpii_human_pose_v1_u12_1.mat"

data = sio.loadmat(MAT_PATH, struct_as_record=False, squeeze_me=True)

release = data["RELEASE"]
annolist = release.annolist

print("Total annotations:", len(annolist))


def to_list(x):
    if isinstance(x, np.ndarray):
        return x.flatten().tolist()
    else:
        return [x]


# 这里改数字：0 是第一张，1 是第二张，20 是第 21 张
TARGET_VALID_SAMPLE = 20

valid_count = 12

for i, anno in enumerate(annolist):

    if not hasattr(anno, "annorect"):
        continue

    persons = to_list(anno.annorect)

    for person in persons:

        if not hasattr(person, "annopoints"):
            continue

        if not hasattr(person.annopoints, "point"):
            continue

        img_name = anno.image.name
        img_path = IMG_DIR / img_name

        if not img_path.exists():
            continue

        if valid_count != TARGET_VALID_SAMPLE:
            valid_count += 1
            continue

        print("Found valid sample!")
        print("Valid sample number:", valid_count)
        print("Annotation index:", i)
        print("Image name:", img_name)
        print("Image path:", img_path)

        img = Image.open(img_path)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)

        points = to_list(person.annopoints.point)

        for p in points:
            x = p.x
            y = p.y
            joint_id = p.id

            plt.scatter(x, y)
            plt.text(x + 3, y + 3, str(joint_id), fontsize=8)

        plt.title(f"MPII sample {valid_count}: {img_name}")
        plt.axis("off")
        plt.show()

        raise SystemExit

print("No valid annotated sample found.")