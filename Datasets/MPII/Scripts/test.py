from pathlib import Path
import scipy.io as sio

MPII_ROOT = Path("/Users/oanahuang/Desktop/MSc Project/Datasets/MPII")

IMG_DIR = MPII_ROOT / "images"
MAT_PATH = MPII_ROOT / "Annotations" / "mpii_human_pose_v1_u12_1.mat"

print("MPII root:", MPII_ROOT)
print("Image dir:", IMG_DIR)
print("Annotation path:", MAT_PATH)

print("Image dir exists:", IMG_DIR.exists())
print("Annotation file exists:", MAT_PATH.exists())

if IMG_DIR.exists():
    jpg_files = list(IMG_DIR.glob("*.jpg"))
    print("Number of jpg images:", len(jpg_files))
else:
    print("ERROR: images folder not found")

if not MAT_PATH.exists():
    print("ERROR: annotation .mat file not found")
    raise FileNotFoundError(MAT_PATH)

data = sio.loadmat(MAT_PATH, struct_as_record=False, squeeze_me=True)

print("MAT keys:", data.keys())

release = data["RELEASE"]
annolist = release.annolist

print("Number of annotations:", len(annolist))
print("First image name:", annolist[0].image.name)

first_img = IMG_DIR / annolist[0].image.name
print("First image path:", first_img)
print("First image exists:", first_img.exists())

print("MPII dataset check finished successfully.")