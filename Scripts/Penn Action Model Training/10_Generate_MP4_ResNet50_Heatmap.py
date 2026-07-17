# Scripts/Penn Action Model Training/10_Generate_MP4_ResNet50_Heatmap.py

from pathlib import Path
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

import cv2


# ============================================================
# 1. Config
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NPZ_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Penn_Action"
    / "penn_action_processed.npz"
)


# ----------------------------
# ResNet50 checkpoint
# ----------------------------

# 服务器训练输出
CKPT_PATH_SERVER = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "07_ResNet50_Heatmap_Baseline"
    / "best_ResNet50_Heatmap_Baseline.pth"
)

# 下载到本地后的模型
CKPT_PATH_LOCAL = (
    PROJECT_ROOT
    / "server_outputs"
    / "07_ResNet50_Heatmap_Baseline"
    / "best_ResNet50_Heatmap_Baseline.pth"
)


# ----------------------------
# Video-level split JSON
# ----------------------------

SPLIT_JSON_SERVER = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "07_ResNet50_Heatmap_Baseline"
    / "video_level_split.json"
)

SPLIT_JSON_LOCAL = (
    PROJECT_ROOT
    / "server_outputs"
    / "07_ResNet50_Heatmap_Baseline"
    / "video_level_split.json"
)


# ----------------------------
# Output
# ----------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "10_Generate_MP4_ResNet50_Heatmap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------
# Video
# ----------------------------

# 从 test_videos 中选择一个视频
TARGET_VIDEO_ID = "0684"

FPS = 30


# ----------------------------
# Visualisation
# ----------------------------

SHOW_GT = True
SHOW_KEYPOINT_INDEX = False
SHOW_TITLE = True

PRED_POINT_RADIUS = 3
PRED_LINE_THICKNESS = 2

GT_POINT_RADIUS = 2
GT_LINE_THICKNESS = 1

TEXT_SCALE = 0.45
TEXT_THICKNESS = 1


# ============================================================
# 2. Penn Action Skeleton
# ============================================================

# Penn Action 13 keypoints:
#
# 0  head
# 1  left shoulder
# 2  right shoulder
# 3  left elbow
# 4  right elbow
# 5  left wrist
# 6  right wrist
# 7  left hip
# 8  right hip
# 9  left knee
# 10 right knee
# 11 left ankle
# 12 right ankle

SKELETON = [
    (0, 1),
    (0, 2),

    (1, 3),
    (3, 5),

    (2, 4),
    (4, 6),

    (1, 7),
    (2, 8),

    (7, 8),

    (7, 9),
    (9, 11),

    (8, 10),
    (10, 12),
]


# ============================================================
# 3. Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# 4. Model
# ============================================================

class ResNet50HeatmapBaseline(nn.Module):

    def __init__(
        self,
        num_keypoints=13,
    ):
        super().__init__()

        # Checkpoint 中已经包含模型权重，
        # 所以不需要重新下载 ImageNet weights。
        resnet = models.resnet50(
            weights=None
        )

        # Remove avgpool and fc
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # 224 input:
        # backbone output = [B, 2048, 7, 7]
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                2048,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                num_keypoints,
                kernel_size=1,
            ),
        )

        # 7 -> 14 -> 28 -> 56

    def forward(self, images):
        features = self.backbone(
            images
        )

        heatmaps = self.decoder(
            features
        )

        return heatmaps


# ============================================================
# 5. Helper Functions
# ============================================================

def safe_video_id_to_str(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    value = str(value)

    if value.isdigit():
        return value.zfill(4)

    return value


def resolve_image_path(path_value):
    original_path = str(
        path_value
    ).strip()

    processed_path = original_path

    if "Penn_Action" in processed_path:
        processed_path = (
            processed_path
            .split("Penn_Action")[-1]
            .lstrip("/")
        )

    path = Path(
        processed_path
    )

    candidates = [
        path,

        PROJECT_ROOT / path,

        (
            PROJECT_ROOT
            / "Datasets"
            / "Penn_Action"
            / path
        ),

        (
            PROJECT_ROOT
            / "Datasets"
            / "Penn_Action"
            / "frames"
            / path
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Image not found.\n"
        f"Original path: {original_path}\n"
        f"Processed path: {processed_path}\n"
    )


def choose_checkpoint_path():
    if CKPT_PATH_SERVER.exists():
        return CKPT_PATH_SERVER

    if CKPT_PATH_LOCAL.exists():
        return CKPT_PATH_LOCAL

    raise FileNotFoundError(
        "ResNet50 heatmap checkpoint not found.\n\n"
        f"Tried:\n"
        f"1. {CKPT_PATH_SERVER}\n"
        f"2. {CKPT_PATH_LOCAL}\n\n"
        "Make sure 07_ResNet50_Heatmap_Baseline.py "
        "has finished training, or download the model "
        "to server_outputs."
    )


def choose_split_json_path():
    if SPLIT_JSON_SERVER.exists():
        return SPLIT_JSON_SERVER

    if SPLIT_JSON_LOCAL.exists():
        return SPLIT_JSON_LOCAL

    return None


def check_video_split(video_id):
    split_path = (
        choose_split_json_path()
    )

    if split_path is None:
        print(
            "\nSplit JSON not found. "
            "Skipping split check."
        )

        return "unknown"

    with split_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        split = json.load(file)

    train_videos = {
        safe_video_id_to_str(value)
        for value in split.get(
            "train_videos",
            [],
        )
    }

    val_videos = {
        safe_video_id_to_str(value)
        for value in split.get(
            "val_videos",
            [],
        )
    }

    test_videos = {
        safe_video_id_to_str(value)
        for value in split.get(
            "test_videos",
            [],
        )
    }

    print(
        f"\nSplit JSON path: "
        f"{split_path}"
    )

    if video_id in train_videos:
        print(
            f"WARNING: Video {video_id} "
            f"is in TRAIN set."
        )

        print(
            "This is not suitable for "
            "final unseen-video testing."
        )

        return "train"

    if video_id in val_videos:
        print(
            f"Video {video_id} "
            f"is in VAL set."
        )

        print(
            "This is okay for validation "
            "visualisation, but not final testing."
        )

        return "val"

    if video_id in test_videos:
        print(
            f"Video {video_id} "
            f"is in TEST set."
        )

        print(
            "This is suitable for "
            "unseen-video testing."
        )

        return "test"

    print(
        f"Video {video_id} is not found "
        f"in train/val/test split lists."
    )

    return "unknown"


# ============================================================
# 6. Heatmap Decoding
# ============================================================

def heatmaps_to_coords_original(
    heatmaps,
    orig_w,
    orig_h,
):
    """
    Args:
        heatmaps:
            [K, H, W]

    Returns:
        coords:
            [K, 2], original image coordinates

        scores:
            [K], maximum heatmap values
    """

    num_keypoints = (
        heatmaps.shape[0]
    )

    heatmap_h = (
        heatmaps.shape[1]
    )

    heatmap_w = (
        heatmaps.shape[2]
    )

    coords = np.zeros(
        (num_keypoints, 2),
        dtype=np.float32,
    )

    scores = np.zeros(
        num_keypoints,
        dtype=np.float32,
    )

    for keypoint_index in range(
        num_keypoints
    ):
        heatmap = heatmaps[
            keypoint_index
        ]

        y, x = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape,
        )

        score = heatmap[
            y,
            x,
        ]

        coords[
            keypoint_index,
            0,
        ] = (
            x
            / float(heatmap_w)
            * orig_w
        )

        coords[
            keypoint_index,
            1,
        ] = (
            y
            / float(heatmap_h)
            * orig_h
        )

        scores[
            keypoint_index
        ] = score

    return coords, scores


# ============================================================
# 7. Drawing
# ============================================================

def valid_point(
    point,
    width,
    height,
):
    x = float(point[0])
    y = float(point[1])

    return (
        np.isfinite(x)
        and np.isfinite(y)
        and 0 <= x < width
        and 0 <= y < height
    )


def draw_pose(
    frame,
    pred_kpts,
    pred_scores=None,
    gt_kpts=None,
    gt_vis=None,
    title=None,
):
    """
    Display style matches the original
    05_Generate_MP4_Heatmap.py.

    Green:
        Ground Truth

    Red:
        Prediction
    """

    output = frame.copy()

    frame_height, frame_width = (
        output.shape[:2]
    )

    # --------------------------------------------------------
    # Draw Ground Truth first
    # --------------------------------------------------------

    if (
        SHOW_GT
        and gt_kpts is not None
        and gt_vis is not None
    ):
        for joint_a, joint_b in SKELETON:

            valid_a = (
                gt_vis[joint_a] > 0
                and valid_point(
                    gt_kpts[joint_a],
                    frame_width,
                    frame_height,
                )
            )

            valid_b = (
                gt_vis[joint_b] > 0
                and valid_point(
                    gt_kpts[joint_b],
                    frame_width,
                    frame_height,
                )
            )

            if not (
                valid_a and valid_b
            ):
                continue

            point_a = tuple(
                np.round(
                    gt_kpts[joint_a]
                ).astype(int)
            )

            point_b = tuple(
                np.round(
                    gt_kpts[joint_b]
                ).astype(int)
            )

            cv2.line(
                output,
                point_a,
                point_b,
                (0, 255, 0),
                GT_LINE_THICKNESS,
                cv2.LINE_AA,
            )

        for joint_index, point in enumerate(
            gt_kpts
        ):
            if (
                gt_vis[joint_index] <= 0
                or not valid_point(
                    point,
                    frame_width,
                    frame_height,
                )
            ):
                continue

            x, y = np.round(
                point
            ).astype(int)

            cv2.circle(
                output,
                (x, y),
                GT_POINT_RADIUS,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Draw Prediction
    # --------------------------------------------------------

    for joint_a, joint_b in SKELETON:

        valid_a = valid_point(
            pred_kpts[joint_a],
            frame_width,
            frame_height,
        )

        valid_b = valid_point(
            pred_kpts[joint_b],
            frame_width,
            frame_height,
        )

        if not (
            valid_a and valid_b
        ):
            continue

        point_a = tuple(
            np.round(
                pred_kpts[joint_a]
            ).astype(int)
        )

        point_b = tuple(
            np.round(
                pred_kpts[joint_b]
            ).astype(int)
        )

        cv2.line(
            output,
            point_a,
            point_b,
            (0, 0, 255),
            PRED_LINE_THICKNESS,
            cv2.LINE_AA,
        )

    for joint_index, point in enumerate(
        pred_kpts
    ):
        if not valid_point(
            point,
            frame_width,
            frame_height,
        ):
            continue

        x, y = np.round(
            point
        ).astype(int)

        cv2.circle(
            output,
            (x, y),
            PRED_POINT_RADIUS,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        if SHOW_KEYPOINT_INDEX:
            cv2.putText(
                output,
                str(joint_index),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                (255, 255, 255),
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Draw Title
    # --------------------------------------------------------

    if (
        SHOW_TITLE
        and title is not None
    ):
        cv2.putText(
            output,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# 8. Main
# ============================================================

def main():
    print(
        f"Using device: {DEVICE}"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"NPZ path: {NPZ_PATH}"
    )

    checkpoint_path = (
        choose_checkpoint_path()
    )

    print(
        f"Checkpoint path: "
        f"{checkpoint_path}"
    )

    print(
        f"Output dir: {OUTPUT_DIR}"
    )

    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"NPZ not found: "
            f"{NPZ_PATH}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    with np.load(
        NPZ_PATH,
        allow_pickle=True,
    ) as data:

        image_paths = (
            data["image_paths"]
            .copy()
        )

        keypoints = (
            data["keypoints"]
            .astype(np.float32)
            .copy()
        )

        visibility = (
            data["visibility"]
            .astype(np.float32)
            .copy()
        )

        video_ids = np.asarray([
            safe_video_id_to_str(value)
            for value in data[
                "video_ids"
            ]
        ])

        if "frame_indices" in data.files:
            frame_indices = (
                data["frame_indices"]
                .astype(np.int64)
                .copy()
            )
        else:
            frame_indices = np.arange(
                len(video_ids),
                dtype=np.int64,
            )

    target_video_id = (
        safe_video_id_to_str(
            TARGET_VIDEO_ID
        )
    )

    video_split = check_video_split(
        target_video_id
    )

    indices = np.where(
        video_ids == target_video_id
    )[0]

    if len(indices) == 0:
        raise ValueError(
            f"Video not found: "
            f"{target_video_id}"
        )

    indices = indices[
        np.argsort(
            frame_indices[
                indices
            ]
        )
    ]

    print(
        f"\nTarget video: "
        f"{target_video_id}"
    )

    print(
        f"Video split: "
        f"{video_split}"
    )

    print(
        f"Number of frames: "
        f"{len(indices)}"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    image_size = int(
        checkpoint.get(
            "image_size",
            224,
        )
    )

    heatmap_size = int(
        checkpoint.get(
            "heatmap_size",
            56,
        )
    )

    num_keypoints = int(
        checkpoint.get(
            "num_keypoints",
            13,
        )
    )

    print(
        f"Checkpoint method: "
        f"{checkpoint.get('method', 'unknown')}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best val loss: "
        f"{checkpoint.get('best_val_loss', 'unknown')}"
    )

    print(
        f"Image size: "
        f"{image_size}"
    )

    print(
        f"Heatmap size: "
        f"{heatmap_size}"
    )

    print(
        f"Num keypoints: "
        f"{num_keypoints}"
    )

    model = ResNet50HeatmapBaseline(
        num_keypoints=num_keypoints
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    transform = T.Compose([
        T.Resize(
            (image_size, image_size)
        ),

        T.ToTensor(),

        T.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    # --------------------------------------------------------
    # Prepare VideoWriter
    # --------------------------------------------------------

    first_image_path = resolve_image_path(
        image_paths[
            indices[0]
        ]
    )

    with Image.open(
        first_image_path
    ) as first_image:

        first_image = first_image.convert(
            "RGB"
        )

        output_width, output_height = (
            first_image.size
        )

    output_video_path = (
        OUTPUT_DIR
        / (
            f"{target_video_id}_"
            f"ResNet50_heatmap_pose.mp4"
        )
    )

    prediction_path = (
        OUTPUT_DIR
        / (
            f"{target_video_id}_"
            f"ResNet50_Pred_vs_GT.npz"
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        FPS,
        (
            output_width,
            output_height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not open VideoWriter:\n"
            f"{output_video_path}"
        )

    # --------------------------------------------------------
    # Prediction storage
    # --------------------------------------------------------

    pred_all = []
    score_all = []
    gt_all = []
    visibility_all = []
    frame_id_all = []

    print(
        "\nRunning ResNet50 heatmap inference..."
    )

    try:
        with torch.inference_mode():

            for count, dataset_index in enumerate(
                indices,
                start=1,
            ):
                image_path = resolve_image_path(
                    image_paths[
                        dataset_index
                    ]
                )

                with Image.open(
                    image_path
                ) as pil_image:

                    pil_image = pil_image.convert(
                        "RGB"
                    )

                    original_width, original_height = (
                        pil_image.size
                    )

                    input_tensor = transform(
                        pil_image
                    ).unsqueeze(0).to(
                        DEVICE
                    )

                    frame_rgb = np.asarray(
                        pil_image
                    ).copy()

                predicted_heatmaps = model(
                    input_tensor
                )[0].detach().cpu().numpy()

                predicted_keypoints, predicted_scores = (
                    heatmaps_to_coords_original(
                        predicted_heatmaps,
                        orig_w=original_width,
                        orig_h=original_height,
                    )
                )

                ground_truth_keypoints = (
                    keypoints[
                        dataset_index
                    ].copy()
                )

                ground_truth_visibility = (
                    visibility[
                        dataset_index
                    ].copy()
                )

                frame_bgr = cv2.cvtColor(
                    frame_rgb,
                    cv2.COLOR_RGB2BGR,
                )

                title = (
                    f"Video {target_video_id} | "
                    f"Frame "
                    f"{int(frame_indices[dataset_index])} | "
                    f"ResNet50 Heatmap"
                )

                drawn_frame = draw_pose(
                    frame=frame_bgr,
                    pred_kpts=(
                        predicted_keypoints
                    ),
                    pred_scores=(
                        predicted_scores
                    ),
                    gt_kpts=(
                        ground_truth_keypoints
                    ),
                    gt_vis=(
                        ground_truth_visibility
                    ),
                    title=title,
                )

                # Ensure fixed video size
                if (
                    drawn_frame.shape[1]
                    != output_width
                    or drawn_frame.shape[0]
                    != output_height
                ):
                    drawn_frame = cv2.resize(
                        drawn_frame,
                        (
                            output_width,
                            output_height,
                        ),
                        interpolation=cv2.INTER_LINEAR,
                    )

                writer.write(
                    drawn_frame
                )

                pred_all.append(
                    predicted_keypoints
                )

                score_all.append(
                    predicted_scores
                )

                gt_all.append(
                    ground_truth_keypoints
                )

                visibility_all.append(
                    ground_truth_visibility
                )

                frame_id_all.append(
                    int(
                        frame_indices[
                            dataset_index
                        ]
                    )
                )

                if (
                    count % 20 == 0
                    or count == len(indices)
                ):
                    print(
                        f"Processed "
                        f"{count}/"
                        f"{len(indices)} frames",
                        flush=True,
                    )

    finally:
        writer.release()

    # --------------------------------------------------------
    # Save predictions and GT
    # --------------------------------------------------------

    pred_all = np.asarray(
        pred_all,
        dtype=np.float32,
    )

    score_all = np.asarray(
        score_all,
        dtype=np.float32,
    )

    gt_all = np.asarray(
        gt_all,
        dtype=np.float32,
    )

    visibility_all = np.asarray(
        visibility_all,
        dtype=np.float32,
    )

    frame_id_all = np.asarray(
        frame_id_all,
        dtype=np.int64,
    )

    np.savez_compressed(
        prediction_path,

        video_id=target_video_id,

        video_split=video_split,

        frame_indices=frame_id_all,

        pred_keypoints=pred_all,

        pred_scores=score_all,

        gt_keypoints=gt_all,

        gt_visibility=visibility_all,

        model_name=(
            "ResNet50_Heatmap_Baseline"
        ),

        checkpoint=str(
            checkpoint_path
        ),

        image_size=image_size,

        heatmap_size=heatmap_size,
    )

    print(
        "\nFinished."
    )

    print(
        "Output video saved to:\n"
        f"{output_video_path}"
    )

    print(
        "Predictions saved to:\n"
        f"{prediction_path}"
    )

    print(
        "\nDisplay colours:"
    )

    print(
        "  Green = Ground Truth"
    )

    print(
        "  Red   = ResNet50 Prediction"
    )


if __name__ == "__main__":
    main()