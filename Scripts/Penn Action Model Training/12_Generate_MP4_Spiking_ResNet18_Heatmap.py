# Scripts/Penn Action Model Training/
# 12_Generate_MP4_Spiking_ResNet18_Heatmap.py

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


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

TRAIN_SCRIPT_PATH = (
    PROJECT_ROOT
    / "Scripts"
    / "Penn Action Model Training"
    / "08_Spiking_ResNet18_Heatmap.py"
)

CKPT_PATH = (
    PROJECT_ROOT
    / "server_outputs"
    / "08_Spiking_ResNet18_Heatmap"
    / "best_Spiking_ResNet18_Heatmap.pth"
)

# Use a video from the test split.
TARGET_VIDEO_ID = "0684"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "12_Generate_MP4_Spiking_ResNet18_Heatmap"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO_PATH = (
    OUTPUT_DIR
    / f"{TARGET_VIDEO_ID}_spiking_resnet18_pose.mp4"
)

PREDICTION_NPZ_PATH = (
    OUTPUT_DIR
    / f"{TARGET_VIDEO_ID}_spiking_resnet18_predictions.npz"
)

IMAGE_SIZE = 224
HEATMAP_SIZE = 56

# Keep the same visualization speed as the previous MP4 scripts.
OUTPUT_FPS = 30.0

# Set to None to process the complete video.
MAX_FRAMES = None

# Joint drawing settings.
JOINT_RADIUS = 4
JOINT_THICKNESS = -1
BONE_THICKNESS = 2

# OpenCV uses BGR.
PREDICTION_COLOR = (0, 0, 255)
GROUND_TRUTH_COLOR = (0, 255, 0)

# Prediction confidence below this value will not be drawn.
# Heatmap models trained with MSE can have relatively small peak values,
# so keep this at 0.0 unless visual inspection shows false detections.
CONFIDENCE_THRESHOLD = 0.0


# Penn Action 13-joint order:
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

JOINT_NAMES = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON_CONNECTIONS = [
    # Head and shoulders
    (0, 1),
    (0, 2),
    (1, 2),

    # Left arm
    (1, 3),
    (3, 5),

    # Right arm
    (2, 4),
    (4, 6),

    # Torso
    (1, 7),
    (2, 8),
    (7, 8),

    # Left leg
    (7, 9),
    (9, 11),

    # Right leg
    (8, 10),
    (10, 12),
]


# ============================================================
# 2. Device
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# 3. General utilities
# ============================================================

def decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.ndarray) and value.ndim == 0:
        return decode_string(value.item())

    return str(value)


def find_npz_key(
    data: Any,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    available = set(data.files)

    for key in candidates:
        if key in available:
            return key

    if required:
        raise KeyError(
            "Could not find the required array in the NPZ.\n"
            f"Tried keys: {candidates}\n"
            f"Available keys: {sorted(available)}"
        )

    return None


def infer_video_id_from_path(path_value: Any) -> str:
    path = Path(decode_string(path_value))

    # Typical Penn Action format:
    # frames/0684/000001.jpg
    return path.parent.name


def resolve_image_path(path_value: Any) -> Path:
    raw_path = Path(decode_string(path_value))

    candidates = [
        raw_path,
        PROJECT_ROOT / raw_path,
        NPZ_PATH.parent / raw_path,
        NPZ_PATH.parent.parent / raw_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate image file.\n"
        f"Stored path: {raw_path}\n"
        "Tried:\n"
        + "\n".join(str(path) for path in candidates)
    )


def frame_number_from_path(path_value: Any) -> int:
    path = Path(decode_string(path_value))

    try:
        return int(path.stem)
    except ValueError:
        digits = "".join(character for character in path.stem if character.isdigit())

        if digits:
            return int(digits)

        return 0


# ============================================================
# 4. Import model from training script
# ============================================================

def load_training_module():
    if not TRAIN_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "08 training script was not found:\n"
            f"{TRAIN_SCRIPT_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "spiking_resnet18_training_module",
        TRAIN_SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import training script:\n{TRAIN_SCRIPT_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format. Expected a dictionary."
        )

    possible_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "network",
        "net",
    ]

    for key in possible_keys:
        value = checkpoint.get(key)

        if isinstance(value, dict):
            return value

    # The checkpoint itself may already be the model state_dict.
    if checkpoint and all(
        isinstance(key, str) for key in checkpoint.keys()
    ):
        tensor_values = [
            value
            for value in checkpoint.values()
            if torch.is_tensor(value)
        ]

        if tensor_values:
            return checkpoint

    raise KeyError(
        "Could not find model weights in checkpoint.\n"
        f"Available checkpoint keys: {list(checkpoint.keys())}"
    )


def remove_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = {}

    prefixes = [
        "module.",
        "model.",
        "_orig_mod.",
    ]

    for key, value in state_dict.items():
        clean_key = key

        prefix_removed = True

        while prefix_removed:
            prefix_removed = False

            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    prefix_removed = True

        cleaned[clean_key] = value

    return cleaned


def create_model(training_module):
    # First try factory functions.
    factory_names = [
        "build_model",
        "create_model",
        "get_model",
        "make_model",
    ]

    for factory_name in factory_names:
        factory = getattr(training_module, factory_name, None)

        if callable(factory):
            print(f"Creating model with {factory_name}()")

            try:
                return factory()
            except TypeError:
                # Some factory functions require the number of joints.
                try:
                    return factory(num_keypoints=13)
                except TypeError:
                    try:
                        return factory(num_joints=13)
                    except TypeError:
                        pass

    # Then try likely model class names.
    class_names = [
        "SpikingResNet18Heatmap",
        "SpikingResNet18HeatmapModel",
        "SpikingResNet18PoseModel",
        "SpikingResNetHeatmap",
        "SpikingResNetPoseModel",
        "SNNResNet18Heatmap",
        "SNNHeatmapPoseModel",
        "PoseModel",
        "HeatmapModel",
    ]

    constructor_argument_sets = [
        {},
        {"num_keypoints": 13},
        {"num_joints": 13},
        {"num_classes": 13},
    ]

    for class_name in class_names:
        model_class = getattr(training_module, class_name, None)

        if model_class is None:
            continue

        for arguments in constructor_argument_sets:
            try:
                model = model_class(**arguments)

                print(
                    f"Creating model with {class_name}"
                    f"({arguments})"
                )

                return model

            except TypeError:
                continue

    # Last fallback: find torch Module classes declared in script.
    discovered_classes = []

    for name, value in vars(training_module).items():
        if (
            isinstance(value, type)
            and issubclass(value, torch.nn.Module)
            and value.__module__ == training_module.__name__
        ):
            discovered_classes.append((name, value))

    for class_name, model_class in discovered_classes:
        for arguments in constructor_argument_sets:
            try:
                model = model_class(**arguments)

                # Avoid selecting an individual residual block.
                parameter_count = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )

                if parameter_count > 1_000_000:
                    print(
                        f"Automatically selected model class: "
                        f"{class_name}"
                    )
                    return model

            except (TypeError, ValueError):
                continue

    available = [
        name
        for name, value in vars(training_module).items()
        if isinstance(value, type)
    ]

    raise RuntimeError(
        "Could not automatically create the model from the 08 script.\n"
        f"Classes found in script: {available}\n\n"
        "Add the exact model class name to class_names in create_model()."
    )


def load_model(device: torch.device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            "Checkpoint was not found:\n"
            f"{CKPT_PATH}\n\n"
            "Confirm that the downloaded checkpoint is inside:\n"
            "server_outputs/08_Spiking_ResNet18_Heatmap/"
        )

    training_module = load_training_module()
    model = create_model(training_module)

    checkpoint = torch.load(
        CKPT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)
    state_dict = remove_state_dict_prefix(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)

    except RuntimeError as error:
        print("\nStrict checkpoint loading failed.")
        print("Trying non-strict loading for diagnosis...\n")

        result = model.load_state_dict(
            state_dict,
            strict=False,
        )

        print(f"Missing keys    : {result.missing_keys}")
        print(f"Unexpected keys : {result.unexpected_keys}")

        raise RuntimeError(
            "The checkpoint does not match the model created from 08.\n"
            "The 12 script must use exactly the same model class and "
            "configuration as training."
        ) from error

    model = model.to(device)
    model.eval()

    return model, checkpoint


# ============================================================
# 5. Image preprocessing
# ============================================================

def normalize_image(image_rgb: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    image = resized.astype(np.float32) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    tensor = torch.from_numpy(image).float()

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32,
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
    ).view(3, 1, 1)

    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)


# ============================================================
# 6. SNN state reset
# ============================================================

def reset_snn_state(model: torch.nn.Module) -> None:
    """
    Supports common SNN libraries and custom reset methods.
    """

    reset_method = getattr(model, "reset", None)

    if callable(reset_method):
        reset_method()
        return

    reset_state_method = getattr(model, "reset_state", None)

    if callable(reset_state_method):
        reset_state_method()
        return

    reset_hidden_method = getattr(model, "reset_hidden", None)

    if callable(reset_hidden_method):
        reset_hidden_method()
        return

    # snnTorch models often create membrane tensors locally, so no reset
    # is required. This fallback resets modules that expose reset methods.
    for module in model.modules():
        if module is model:
            continue

        for method_name in [
            "reset",
            "reset_state",
            "reset_hidden",
        ]:
            method = getattr(module, method_name, None)

            if callable(method):
                try:
                    method()
                except TypeError:
                    pass


# ============================================================
# 7. Model output and heatmap decoding
# ============================================================

def extract_heatmaps(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        heatmaps = model_output

    elif isinstance(model_output, (tuple, list)):
        heatmaps = None

        for value in model_output:
            if (
                torch.is_tensor(value)
                and value.ndim >= 4
            ):
                heatmaps = value
                break

        if heatmaps is None:
            raise RuntimeError(
                "Model returned a tuple/list, but no heatmap tensor "
                "was found."
            )

    elif isinstance(model_output, dict):
        heatmaps = None

        for key in [
            "heatmaps",
            "output",
            "outputs",
            "prediction",
            "predictions",
            "pred_heatmaps",
        ]:
            value = model_output.get(key)

            if torch.is_tensor(value):
                heatmaps = value
                break

        if heatmaps is None:
            raise RuntimeError(
                "Model returned a dictionary, but no heatmap tensor "
                "was found.\n"
                f"Available keys: {list(model_output.keys())}"
            )

    else:
        raise TypeError(
            "Unsupported model output type: "
            f"{type(model_output)}"
        )

    # Support temporal output [T, B, J, H, W].
    if heatmaps.ndim == 5:
        heatmaps = heatmaps.mean(dim=0)

    if heatmaps.ndim != 4:
        raise ValueError(
            "Expected heatmaps with shape [B, J, H, W], "
            f"but received {tuple(heatmaps.shape)}"
        )

    return heatmaps


def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert heatmaps [1, J, H, W] into coordinates in 224x224 space.
    """

    heatmaps = heatmaps.detach().float().cpu()

    batch_size, num_joints, height, width = heatmaps.shape

    if batch_size != 1:
        raise ValueError(
            "MP4 generation expects batch size 1, "
            f"but received {batch_size}."
        )

    flattened = heatmaps.reshape(
        batch_size,
        num_joints,
        -1,
    )

    confidence, flat_indices = flattened.max(dim=-1)

    x = flat_indices % width
    y = flat_indices // width

    x = x.float() * IMAGE_SIZE / float(width)
    y = y.float() * IMAGE_SIZE / float(height)

    coordinates = torch.stack(
        [x, y],
        dim=-1,
    )

    return (
        coordinates[0].numpy().astype(np.float32),
        confidence[0].numpy().astype(np.float32),
    )


# ============================================================
# 8. Ground-truth utilities
# ============================================================

def read_ground_truth(
    raw_keypoints: np.ndarray,
    raw_visibility: np.ndarray | None,
    original_width: int,
    original_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(
        raw_keypoints,
        dtype=np.float32,
    )[:, :2].copy()

    if raw_visibility is not None:
        visibility = np.asarray(
            raw_visibility,
            dtype=np.float32,
        ).reshape(-1)

    elif np.asarray(raw_keypoints).shape[-1] >= 3:
        visibility = np.asarray(
            raw_keypoints,
            dtype=np.float32,
        )[:, 2].copy()

    else:
        visibility = (
            np.isfinite(keypoints).all(axis=1)
            & (keypoints[:, 0] >= 0)
            & (keypoints[:, 1] >= 0)
        ).astype(np.float32)

    finite_values = keypoints[np.isfinite(keypoints)]

    coordinates_are_normalized = (
        finite_values.size > 0
        and np.max(finite_values) <= 1.5
        and np.min(finite_values) >= -0.5
    )

    if coordinates_are_normalized:
        keypoints[:, 0] *= original_width
        keypoints[:, 1] *= original_height

    invalid = ~np.isfinite(keypoints).all(axis=1)

    visibility = (visibility > 0).astype(np.float32)
    visibility[invalid] = 0.0
    keypoints[invalid] = 0.0

    return keypoints, visibility


def scale_predictions_to_original_image(
    predicted_xy_224: np.ndarray,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    predicted_xy = predicted_xy_224.copy()

    predicted_xy[:, 0] *= original_width / float(IMAGE_SIZE)
    predicted_xy[:, 1] *= original_height / float(IMAGE_SIZE)

    return predicted_xy


# ============================================================
# 9. Visualization
# ============================================================

def valid_point(
    point: np.ndarray,
    width: int,
    height: int,
) -> bool:
    return (
        np.isfinite(point).all()
        and 0 <= point[0] < width
        and 0 <= point[1] < height
    )


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    visibility: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> None:
    height, width = image.shape[:2]

    drawable = np.ones(
        len(keypoints),
        dtype=bool,
    )

    if visibility is not None:
        drawable &= visibility > 0

    if confidence is not None:
        drawable &= confidence >= CONFIDENCE_THRESHOLD

    for joint_index, point in enumerate(keypoints):
        drawable[joint_index] &= valid_point(
            point,
            width,
            height,
        )

    # Draw bones first.
    for joint_a, joint_b in SKELETON_CONNECTIONS:
        if not drawable[joint_a] or not drawable[joint_b]:
            continue

        point_a = tuple(
            np.round(keypoints[joint_a]).astype(int)
        )

        point_b = tuple(
            np.round(keypoints[joint_b]).astype(int)
        )

        cv2.line(
            image,
            point_a,
            point_b,
            color,
            BONE_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    # Draw joints over bones.
    for joint_index, point in enumerate(keypoints):
        if not drawable[joint_index]:
            continue

        center = tuple(
            np.round(point).astype(int)
        )

        cv2.circle(
            image,
            center,
            JOINT_RADIUS,
            color,
            JOINT_THICKNESS,
            lineType=cv2.LINE_AA,
        )


def draw_legend(image: np.ndarray) -> None:
    """
    Keep the same simple legend style used by the earlier scripts.
    """

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (225, 78),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        image,
        0.45,
        0,
        image,
    )

    cv2.circle(
        image,
        (28, 31),
        5,
        GROUND_TRUTH_COLOR,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "Ground Truth",
        (43, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.circle(
        image,
        (28, 59),
        5,
        PREDICTION_COLOR,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "Prediction",
        (43, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_frame_information(
    image: np.ndarray,
    video_id: str,
    frame_position: int,
    frame_count: int,
) -> None:
    text = (
        f"Video: {video_id}  "
        f"Frame: {frame_position + 1}/{frame_count}"
    )

    height = image.shape[0]

    cv2.putText(
        image,
        text,
        (12, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        (12, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# 10. Dataset loading
# ============================================================

def load_video_samples():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found:\n{NPZ_PATH}"
        )

    data = np.load(
        NPZ_PATH,
        allow_pickle=True,
    )

    image_path_key = find_npz_key(
        data,
        [
            "image_paths",
            "img_paths",
            "frame_paths",
            "paths",
            "images_path",
            "image_path",
            "img_path",
        ],
        required=True,
    )

    keypoint_key = find_npz_key(
        data,
        [
            "keypoints",
            "joints",
            "poses",
            "pose",
            "labels",
            "coordinates",
        ],
        required=True,
    )

    visibility_key = find_npz_key(
        data,
        [
            "visibility",
            "visibilities",
            "visible",
            "is_visible",
            "joint_visibility",
        ],
        required=False,
    )

    video_id_key = find_npz_key(
        data,
        [
            "video_ids",
            "video_id",
            "sequence_ids",
            "sequence_id",
            "seq_ids",
            "seq_id",
        ],
        required=False,
    )

    image_paths = np.asarray(data[image_path_key])
    keypoints = np.asarray(data[keypoint_key])

    if visibility_key is not None:
        visibility = np.asarray(data[visibility_key])
    else:
        visibility = None

    if video_id_key is not None:
        video_ids = np.asarray(
            [
                decode_string(value)
                for value in data[video_id_key]
            ],
            dtype=object,
        )
    else:
        video_ids = np.asarray(
            [
                infer_video_id_from_path(value)
                for value in image_paths
            ],
            dtype=object,
        )

    target_id = str(TARGET_VIDEO_ID)

    matching_indices = np.asarray(
        [
            index
            for index, video_id in enumerate(video_ids)
            if str(video_id).zfill(4) == target_id.zfill(4)
        ],
        dtype=np.int64,
    )

    if len(matching_indices) == 0:
        examples = sorted(
            set(str(value) for value in video_ids)
        )[:20]

        data.close()

        raise ValueError(
            f"Video ID {TARGET_VIDEO_ID} was not found.\n"
            f"Example available video IDs: {examples}"
        )

    matching_indices = sorted(
        matching_indices.tolist(),
        key=lambda index: frame_number_from_path(
            image_paths[index]
        ),
    )

    if MAX_FRAMES is not None:
        matching_indices = matching_indices[:MAX_FRAMES]

    print(f"NPZ keys       : {data.files}")
    print(f"Image path key : {image_path_key}")
    print(f"Keypoint key   : {keypoint_key}")
    print(f"Visibility key : {visibility_key}")
    print(f"Video ID key   : {video_id_key}")
    print(f"Selected video : {TARGET_VIDEO_ID}")
    print(f"Selected frames: {len(matching_indices)}")

    return (
        data,
        image_paths,
        keypoints,
        visibility,
        matching_indices,
    )


# ============================================================
# 11. Main
# ============================================================

def main() -> None:
    device = get_device()

    print("=" * 72)
    print("Spiking ResNet18 Heatmap MP4 Generation")
    print("=" * 72)
    print(f"Device       : {device}")
    print(f"Dataset      : {NPZ_PATH}")
    print(f"Training code: {TRAIN_SCRIPT_PATH}")
    print(f"Checkpoint   : {CKPT_PATH}")
    print(f"Target video : {TARGET_VIDEO_ID}")
    print(f"Output video : {OUTPUT_VIDEO_PATH}")
    print(f"Prediction   : {PREDICTION_NPZ_PATH}")
    print("=" * 72)

    model, checkpoint = load_model(device)

    (
        npz_data,
        image_paths,
        all_keypoints,
        all_visibility,
        selected_indices,
    ) = load_video_samples()

    first_image_path = resolve_image_path(
        image_paths[selected_indices[0]]
    )

    first_frame = cv2.imread(
        str(first_image_path),
        cv2.IMREAD_COLOR,
    )

    if first_frame is None:
        npz_data.close()
        raise RuntimeError(
            f"OpenCV could not read:\n{first_image_path}"
        )

    video_height, video_width = first_frame.shape[:2]

    video_writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (video_width, video_height),
    )

    if not video_writer.isOpened():
        npz_data.close()
        raise RuntimeError(
            f"Could not create output video:\n{OUTPUT_VIDEO_PATH}"
        )

    saved_predicted_xy = []
    saved_predicted_xy_224 = []
    saved_confidence = []
    saved_ground_truth_xy = []
    saved_visibility = []
    saved_sample_indices = []
    saved_image_paths = []

    model.eval()

    try:
        with torch.inference_mode():
            for frame_position, sample_index in enumerate(selected_indices):
                image_path = resolve_image_path(
                    image_paths[sample_index]
                )

                frame_bgr = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_COLOR,
                )

                if frame_bgr is None:
                    raise RuntimeError(
                        f"OpenCV could not read:\n{image_path}"
                    )

                original_height, original_width = frame_bgr.shape[:2]

                if (
                    original_width != video_width
                    or original_height != video_height
                ):
                    frame_bgr = cv2.resize(
                        frame_bgr,
                        (video_width, video_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    original_width = video_width
                    original_height = video_height

                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                input_tensor = normalize_image(
                    frame_rgb
                ).to(
                    device,
                    non_blocking=True,
                )

                # Each Penn Action frame is treated as an independent
                # pose sample, matching frame-wise training.
                reset_snn_state(model)

                raw_output = model(input_tensor)
                heatmaps = extract_heatmaps(raw_output)

                predicted_xy_224, confidence = (
                    heatmaps_to_keypoints(heatmaps)
                )

                predicted_xy = scale_predictions_to_original_image(
                    predicted_xy_224,
                    original_width=original_width,
                    original_height=original_height,
                )

                raw_visibility = (
                    all_visibility[sample_index]
                    if all_visibility is not None
                    else None
                )

                ground_truth_xy, visibility = read_ground_truth(
                    raw_keypoints=all_keypoints[sample_index],
                    raw_visibility=raw_visibility,
                    original_width=original_width,
                    original_height=original_height,
                )

                visualization = frame_bgr.copy()

                # Draw ground truth first and prediction on top,
                # matching the previous comparison-video style.
                draw_skeleton(
                    image=visualization,
                    keypoints=ground_truth_xy,
                    color=GROUND_TRUTH_COLOR,
                    visibility=visibility,
                )

                draw_skeleton(
                    image=visualization,
                    keypoints=predicted_xy,
                    color=PREDICTION_COLOR,
                    confidence=confidence,
                )

                draw_legend(visualization)

                draw_frame_information(
                    image=visualization,
                    video_id=str(TARGET_VIDEO_ID),
                    frame_position=frame_position,
                    frame_count=len(selected_indices),
                )

                video_writer.write(visualization)

                saved_predicted_xy.append(predicted_xy)
                saved_predicted_xy_224.append(predicted_xy_224)
                saved_confidence.append(confidence)
                saved_ground_truth_xy.append(ground_truth_xy)
                saved_visibility.append(visibility)
                saved_sample_indices.append(sample_index)
                saved_image_paths.append(str(image_path))

                if (
                    frame_position == 0
                    or (frame_position + 1) % 50 == 0
                    or frame_position + 1 == len(selected_indices)
                ):
                    print(
                        f"Processed "
                        f"{frame_position + 1:04d}/"
                        f"{len(selected_indices):04d} frames",
                        flush=True,
                    )

    finally:
        video_writer.release()
        npz_data.close()

    checkpoint_epoch = -1
    checkpoint_best_val_loss = np.nan

    if isinstance(checkpoint, dict):
        checkpoint_epoch = int(
            checkpoint.get("epoch", -1)
        )

        checkpoint_best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                checkpoint.get("val_loss", np.nan),
            )
        )

    np.savez_compressed(
        PREDICTION_NPZ_PATH,
        video_id=np.asarray(
            str(TARGET_VIDEO_ID)
        ),
        sample_indices=np.asarray(
            saved_sample_indices,
            dtype=np.int64,
        ),
        image_paths=np.asarray(
            saved_image_paths,
            dtype=object,
        ),
        predicted_xy=np.asarray(
            saved_predicted_xy,
            dtype=np.float32,
        ),
        predicted_xy_224=np.asarray(
            saved_predicted_xy_224,
            dtype=np.float32,
        ),
        confidence=np.asarray(
            saved_confidence,
            dtype=np.float32,
        ),
        ground_truth_xy=np.asarray(
            saved_ground_truth_xy,
            dtype=np.float32,
        ),
        visibility=np.asarray(
            saved_visibility,
            dtype=np.float32,
        ),
        joint_names=np.asarray(
            JOINT_NAMES,
            dtype=object,
        ),
        image_size=np.asarray(
            IMAGE_SIZE,
            dtype=np.int64,
        ),
        heatmap_size=np.asarray(
            HEATMAP_SIZE,
            dtype=np.int64,
        ),
        output_fps=np.asarray(
            OUTPUT_FPS,
            dtype=np.float32,
        ),
        checkpoint_path=np.asarray(
            str(CKPT_PATH)
        ),
        checkpoint_epoch=np.asarray(
            checkpoint_epoch,
            dtype=np.int64,
        ),
        checkpoint_best_val_loss=np.asarray(
            checkpoint_best_val_loss,
            dtype=np.float32,
        ),
    )

    metadata = {
        "video_id": str(TARGET_VIDEO_ID),
        "number_of_frames": len(saved_sample_indices),
        "checkpoint": str(CKPT_PATH),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_val_loss": checkpoint_best_val_loss,
        "video_path": str(OUTPUT_VIDEO_PATH),
        "prediction_path": str(PREDICTION_NPZ_PATH),
        "fps": OUTPUT_FPS,
    }

    metadata_path = OUTPUT_DIR / (
        f"{TARGET_VIDEO_ID}_spiking_resnet18_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Generation finished.")
    print("=" * 72)
    print(f"MP4        : {OUTPUT_VIDEO_PATH}")
    print(f"Prediction : {PREDICTION_NPZ_PATH}")
    print(f"Metadata   : {metadata_path}")
    print(f"Frames     : {len(saved_sample_indices)}")
    print("=" * 72)


if __name__ == "__main__":
    main()