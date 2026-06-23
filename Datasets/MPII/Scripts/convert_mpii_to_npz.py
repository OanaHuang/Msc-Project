from pathlib import Path
import numpy as np
import scipy.io as sio


# ============================================================
# 1. 设置 MPII 数据集路径
# ============================================================

MPII_ROOT = Path("/Users/oanahuang/Desktop/MSc Project/Datasets/MPII")

IMG_DIR = MPII_ROOT / "images"
MAT_PATH = MPII_ROOT / "Annotations" / "mpii_human_pose_v1_u12_1.mat"

OUT_PATH = MPII_ROOT / "Annotations" / "mpii_processed.npz"


# ============================================================
# 2. 工具函数：把 MATLAB 读取出来的对象统一转成 list
# ============================================================

def to_list(x):
    """
    scipy.io.loadmat 读取 MATLAB struct 后，
    有时候对象是单个 object，有时候是 numpy.ndarray。
    这个函数统一把它变成 Python list，方便后面遍历。
    """
    if isinstance(x, np.ndarray):
        return x.flatten().tolist()
    return [x]


# ============================================================
# 3. 从一个 person 标注里提取 16 个关键点
# ============================================================

def extract_keypoints(person):
    """
    从 MPII 的一个 person 标注中提取关键点。

    返回:
        keypoints: shape = (16, 2)
            每一行是一个关节点的 [x, y]

        visibility: shape = (16,)
            1 表示这个点有效/可见
            0 表示这个点缺失/不可用
    """

    keypoints = np.zeros((16, 2), dtype=np.float32)
    visibility = np.zeros((16,), dtype=np.float32)

    if not hasattr(person, "annopoints"):
        return keypoints, visibility

    if not hasattr(person.annopoints, "point"):
        return keypoints, visibility

    points = to_list(person.annopoints.point)

    for p in points:
        if not hasattr(p, "id"):
            continue

        joint_id = int(p.id)

        if joint_id < 0 or joint_id >= 16:
            continue

        if not hasattr(p, "x") or not hasattr(p, "y"):
            continue

        keypoints[joint_id, 0] = float(p.x)
        keypoints[joint_id, 1] = float(p.y)

        # MPII 有些点有 is_visible，有些点没有
        # 没有 is_visible 时，我们先默认它是有效点
        if hasattr(p, "is_visible"):
            try:
                visibility[joint_id] = float(p.is_visible)
            except Exception:
                visibility[joint_id] = 1.0
        else:
            visibility[joint_id] = 1.0

    return keypoints, visibility


# ============================================================
# 4. 主程序：读取 .mat，转换成 npz
# ============================================================

def main():
    print("MPII root:", MPII_ROOT)
    print("Image dir:", IMG_DIR)
    print("MAT path:", MAT_PATH)
    print("Output path:", OUT_PATH)

    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")

    if not MAT_PATH.exists():
        raise FileNotFoundError(f"Annotation file not found: {MAT_PATH}")

    print("\nLoading .mat annotation file...")
    data = sio.loadmat(MAT_PATH, struct_as_record=False, squeeze_me=True)

    if "RELEASE" not in data:
        raise KeyError("Cannot find RELEASE in .mat file")

    release = data["RELEASE"]
    annolist = release.annolist

    print("Number of image annotations:", len(annolist))

    image_paths = []
    image_names = []
    keypoints_list = []
    visibility_list = []
    anno_indices = []
    person_indices = []

    total_persons = 0
    valid_persons = 0
    missing_images = 0
    no_keypoint_persons = 0

    print("\nConverting annotations...")

    for anno_idx, anno in enumerate(annolist):

        if not hasattr(anno, "image"):
            continue

        if not hasattr(anno.image, "name"):
            continue

        img_name = str(anno.image.name)
        img_path = IMG_DIR / img_name

        if not img_path.exists():
            missing_images += 1
            continue

        if not hasattr(anno, "annorect"):
            continue

        persons = to_list(anno.annorect)

        for person_idx, person in enumerate(persons):
            total_persons += 1

            keypoints, visibility = extract_keypoints(person)

            # 至少有一个有效关键点，才保留这个 person
            if visibility.sum() == 0:
                no_keypoint_persons += 1
                continue

            # 保存相对路径，而不是绝对路径
            # 这样以后移动 MPII 文件夹也更方便
            relative_img_path = str(Path("images") / img_name)

            image_paths.append(relative_img_path)
            image_names.append(img_name)
            keypoints_list.append(keypoints)
            visibility_list.append(visibility)
            anno_indices.append(anno_idx)
            person_indices.append(person_idx)

            valid_persons += 1

        if anno_idx % 1000 == 0:
            print(f"Processed {anno_idx}/{len(annolist)} annotations...")

    print("\nConversion finished.")

    print("Total person instances found:", total_persons)
    print("Valid person samples:", valid_persons)
    print("Missing images:", missing_images)
    print("Persons without keypoints:", no_keypoint_persons)

    if valid_persons == 0:
        raise RuntimeError("No valid person samples found. Please check the .mat file and image folder.")

    # ============================================================
    # 5. 转成 NumPy array
    # ============================================================

    image_paths = np.array(image_paths, dtype=object)
    image_names = np.array(image_names, dtype=object)
    keypoints_array = np.stack(keypoints_list).astype(np.float32)
    visibility_array = np.stack(visibility_list).astype(np.float32)
    anno_indices = np.array(anno_indices, dtype=np.int32)
    person_indices = np.array(person_indices, dtype=np.int32)

    print("\nFinal array shapes:")
    print("image_paths:", image_paths.shape)
    print("image_names:", image_names.shape)
    print("keypoints:", keypoints_array.shape)
    print("visibility:", visibility_array.shape)
    print("anno_indices:", anno_indices.shape)
    print("person_indices:", person_indices.shape)

    # ============================================================
    # 6. 保存成 .npz 文件
    # ============================================================

    np.savez_compressed(
        OUT_PATH,
        image_paths=image_paths,
        image_names=image_names,
        keypoints=keypoints_array,
        visibility=visibility_array,
        anno_indices=anno_indices,
        person_indices=person_indices,
    )

    print("\nSaved processed MPII annotations to:")
    print(OUT_PATH)

    print("\nMPII .mat -> .npz conversion successful.")


if __name__ == "__main__":
    main()