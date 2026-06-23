from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


MPII_ROOT = Path("/Users/oanahuang/Desktop/MSc Project/Datasets/MPII")
NPZ_PATH = MPII_ROOT / "Annotations" / "mpii_processed.npz"

data = np.load(NPZ_PATH, allow_pickle=True)

image_paths = data["image_paths"]
image_names = data["image_names"]
keypoints = data["keypoints"]
visibility = data["visibility"]
anno_indices = data["anno_indices"]
person_indices = data["person_indices"]

print("Loaded:", NPZ_PATH)

print("\nArray shapes:")
print("image_paths:", image_paths.shape)
print("image_names:", image_names.shape)
print("keypoints:", keypoints.shape)
print("visibility:", visibility.shape)
print("anno_indices:", anno_indices.shape)
print("person_indices:", person_indices.shape)

# 你可以改这个数字来检查不同样本
idx = 18

img_path = MPII_ROOT / image_paths[idx]

print("\nSample index:", idx)
print("Image path:", img_path)
print("Image exists:", img_path.exists())
print("Image name:", image_names[idx])
print("Annotation index:", anno_indices[idx])
print("Person index:", person_indices[idx])
print("Keypoints shape for this sample:", keypoints[idx].shape)
print("Visibility shape for this sample:", visibility[idx].shape)
print("Visible joints:", visibility[idx].sum())

print("\nKeypoints:")
print(keypoints[idx])

print("\nVisibility:")
print(visibility[idx])

img = Image.open(img_path)

plt.figure(figsize=(8, 8))
plt.imshow(img)

for joint_id in range(16):
    if visibility[idx, joint_id] > 0:
        x, y = keypoints[idx, joint_id]
        plt.scatter(x, y)
        plt.text(x + 3, y + 3, str(joint_id), fontsize=8)

plt.title(f"NPZ check sample {idx}: {image_names[idx]}")
plt.axis("off")
plt.show()