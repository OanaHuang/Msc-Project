# Scripts/SNN Model/01_Train_SNN_Heatmap_Baseline.py

from pathlib import Path
import csv
import json
import random
import time

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import snntorch as snn
from snntorch import surrogate


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "SNN_Model"
    / "01_Train_SNN_Heatmap_Baseline"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT_PATH = OUTPUT_DIR / "best_snn_heatmap_baseline.pth"
LAST_CKPT_PATH = OUTPUT_DIR / "last_snn_heatmap_baseline.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
SPLIT_PATH = OUTPUT_DIR / "split_video_ids.json"


# ----------------------------
# Data settings
# ----------------------------

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_KEYPOINTS = 13
HEATMAP_SIGMA = 1.5

VAL_RATIO = 0.20
SEED = 42


# ----------------------------
# Training settings
# ----------------------------

BATCH_SIZE = 12
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 8
GRAD_CLIP_NORM = 1.0


# ----------------------------
# SNN settings
# ----------------------------

NUM_STEPS = 4
BETA = 0.90
SPIKE_THRESHOLD = 1.0
SURROGATE_SLOPE = 25.0


# ----------------------------
# Metric settings
# ----------------------------

PCK_THRESHOLD = 0.10


# For a quick pipeline test:
# MAX_SAMPLES = 300
MAX_SAMPLES = None


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


# ============================================================
# 2. Random seed and device
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# 3. NPZ key utilities
# ============================================================

def find_key(npz_file, possible_keys, required=True):
    """
    Search for the first available key in an NPZ file.
    """

    available_keys = set(npz_file.files)

    for key in possible_keys:
        if key in available_keys:
            return key

    if required:
        raise KeyError(
            "\nCould not find a required NPZ key.\n"
            f"Tried: {possible_keys}\n"
            f"Available keys: {sorted(available_keys)}\n"
        )

    return None


def decode_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def resolve_image_path(raw_path):
    """
    Resolve an image path stored in the NPZ.
    """

    path = Path(decode_string(raw_path)).expanduser()

    possible_paths = [
        path,
        PROJECT_ROOT / path,
        NPZ_PATH.parent / path,
        PROJECT_ROOT / "Datasets" / "Penn_Action" / path,
    ]

    for candidate in possible_paths:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "\nCould not resolve image path.\n"
        f"Stored path: {path}\n"
        f"Tried: {[str(p) for p in possible_paths]}\n"
    )


# ============================================================
# 4. Heatmap generation
# ============================================================

def make_gaussian_heatmaps(
    keypoints,
    visibility,
    image_size=IMAGE_SIZE,
    heatmap_size=HEATMAP_SIZE,
    sigma=HEATMAP_SIGMA,
):
    """
    Args:
        keypoints:
            [J, 2], coordinates in resized image pixels.

        visibility:
            [J], 1 means valid joint.

    Returns:
        heatmaps:
            [J, heatmap_size, heatmap_size]
    """

    heatmaps = np.zeros(
        (NUM_KEYPOINTS, heatmap_size, heatmap_size),
        dtype=np.float32,
    )

    coordinate_scale = heatmap_size / float(image_size)
    radius = max(1, int(3 * sigma))

    for joint_index in range(NUM_KEYPOINTS):

        if visibility[joint_index] <= 0:
            continue

        x = keypoints[joint_index, 0] * coordinate_scale
        y = keypoints[joint_index, 1] * coordinate_scale

        if not (
            0 <= x < heatmap_size
            and 0 <= y < heatmap_size
        ):
            continue

        x_min = max(0, int(x) - radius)
        x_max = min(heatmap_size, int(x) + radius + 1)

        y_min = max(0, int(y) - radius)
        y_max = min(heatmap_size, int(y) + radius + 1)

        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

        gaussian = np.exp(
            -(
                (xx - x) ** 2
                + (yy - y) ** 2
            )
            / (2 * sigma ** 2)
        ).astype(np.float32)

        heatmaps[
            joint_index,
            y_min:y_max,
            x_min:x_max,
        ] = gaussian

    return heatmaps


# ============================================================
# 5. Penn Action Dataset
# ============================================================

class PennActionDataset(Dataset):

    IMAGE_KEYS = [
        "images",
        "frames",
        "imgs",
        "image_data",
        "X",
    ]

    IMAGE_PATH_KEYS = [
        "image_paths",
        "frame_paths",
        "img_paths",
        "paths",
        "filenames",
        "files",
    ]

    KEYPOINT_KEYS = [
        "keypoints",
        "joints",
        "poses",
        "pose",
        "coords",
        "coordinates",
        "labels",
        "Y",
        "y",
    ]

    VISIBILITY_KEYS = [
        "visibility",
        "visible",
        "vis",
        "joint_visibility",
        "masks",
        "mask",
    ]

    VIDEO_ID_KEYS = [
        "video_ids",
        "video_id",
        "sequence_ids",
        "seq_ids",
    ]

    def __init__(
        self,
        npz_path,
        indices,
        augment=False,
    ):
        super().__init__()

        self.npz_path = Path(npz_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

        self.data = np.load(
            self.npz_path,
            allow_pickle=True,
        )

        self.image_key = find_key(
            self.data,
            self.IMAGE_KEYS,
            required=False,
        )

        self.image_path_key = find_key(
            self.data,
            self.IMAGE_PATH_KEYS,
            required=False,
        )

        self.keypoint_key = find_key(
            self.data,
            self.KEYPOINT_KEYS,
            required=True,
        )

        self.visibility_key = find_key(
            self.data,
            self.VISIBILITY_KEYS,
            required=False,
        )

        if (
            self.image_key is None
            and self.image_path_key is None
        ):
            raise KeyError(
                "The NPZ contains neither images nor image paths."
            )

        keypoints = self.data[self.keypoint_key]

        if keypoints.ndim != 3:
            raise ValueError(
                "Keypoints must have shape [N, J, 2] "
                f"or [N, J, 3], but got {keypoints.shape}."
            )

        if keypoints.shape[1] != NUM_KEYPOINTS:
            raise ValueError(
                f"Expected {NUM_KEYPOINTS} keypoints, "
                f"but found {keypoints.shape[1]}."
            )

    def __len__(self):
        return len(self.indices)

    def read_image(self, sample_index):

        if self.image_key is not None:
            image = np.asarray(
                self.data[self.image_key][sample_index]
            )

            # Convert CHW to HWC if needed.
            if (
                image.ndim == 3
                and image.shape[0] in [1, 3, 4]
                and image.shape[-1] not in [1, 3, 4]
            ):
                image = np.transpose(image, (1, 2, 0))

            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)

            if image.shape[-1] == 4:
                image = image[:, :, :3]

            return np.ascontiguousarray(image)

        image_path = resolve_image_path(
            self.data[self.image_path_key][sample_index]
        )

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise RuntimeError(
                f"OpenCV could not read image: {image_path}"
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        return image_rgb

    def read_pose(self, sample_index):

        raw_pose = np.asarray(
            self.data[self.keypoint_key][sample_index],
            dtype=np.float32,
        )

        keypoints = raw_pose[:, :2].copy()

        if self.visibility_key is not None:
            visibility = np.asarray(
                self.data[self.visibility_key][sample_index],
                dtype=np.float32,
            ).reshape(-1)

        elif raw_pose.shape[-1] >= 3:
            visibility = raw_pose[:, 2].copy()

        else:
            visibility = (
                np.isfinite(keypoints).all(axis=1)
                & (keypoints[:, 0] >= 0)
                & (keypoints[:, 1] >= 0)
            ).astype(np.float32)

        visibility = (visibility > 0).astype(np.float32)

        invalid = ~np.isfinite(keypoints).all(axis=1)

        visibility[invalid] = 0
        keypoints[invalid] = 0

        return keypoints, visibility

    def resize_image_and_pose(
        self,
        image,
        keypoints,
    ):
        old_height, old_width = image.shape[:2]

        # Detect normalized [0,1] coordinates.
        finite_values = keypoints[np.isfinite(keypoints)]

        coordinates_are_normalized = (
            finite_values.size > 0
            and np.max(finite_values) <= 1.5
            and np.min(finite_values) >= -0.5
        )

        if coordinates_are_normalized:
            keypoints[:, 0] *= old_width
            keypoints[:, 1] *= old_height

        resized_image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        keypoints[:, 0] *= IMAGE_SIZE / float(old_width)
        keypoints[:, 1] *= IMAGE_SIZE / float(old_height)

        return resized_image, keypoints

    def horizontal_flip(
        self,
        image,
        keypoints,
        visibility,
    ):
        if not self.augment:
            return image, keypoints, visibility

        if random.random() >= 0.5:
            return image, keypoints, visibility

        image = np.ascontiguousarray(image[:, ::-1])

        keypoints = keypoints.copy()
        visibility = visibility.copy()

        keypoints[:, 0] = (
            IMAGE_SIZE - 1 - keypoints[:, 0]
        )

        left_right_pairs = [
            (1, 2),    # shoulders
            (3, 4),    # elbows
            (5, 6),    # wrists
            (7, 8),    # hips
            (9, 10),   # knees
            (11, 12),  # ankles
        ]

        for left_index, right_index in left_right_pairs:

            keypoints[
                [left_index, right_index]
            ] = keypoints[
                [right_index, left_index]
            ]

            visibility[
                [left_index, right_index]
            ] = visibility[
                [right_index, left_index]
            ]

        return image, keypoints, visibility

    @staticmethod
    def normalize_image(image):

        image = image.astype(np.float32)

        if image.max() > 1.5:
            image /= 255.0

        image = np.transpose(image, (2, 0, 1))

        image = torch.from_numpy(image).float()

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        image = (image - mean) / std

        return image

    def __getitem__(self, dataset_index):

        sample_index = int(
            self.indices[dataset_index]
        )

        image = self.read_image(sample_index)

        keypoints, visibility = self.read_pose(
            sample_index
        )

        image, keypoints = self.resize_image_and_pose(
            image,
            keypoints,
        )

        image, keypoints, visibility = (
            self.horizontal_flip(
                image,
                keypoints,
                visibility,
            )
        )

        inside_image = (
            (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] < IMAGE_SIZE)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] < IMAGE_SIZE)
        )

        visibility = (
            visibility
            * inside_image.astype(np.float32)
        )

        heatmaps = make_gaussian_heatmaps(
            keypoints=keypoints,
            visibility=visibility,
        )

        return {
            "image": self.normalize_image(image),

            "heatmaps": torch.from_numpy(
                heatmaps
            ).float(),

            "keypoints": torch.from_numpy(
                keypoints
            ).float(),

            "visibility": torch.from_numpy(
                visibility
            ).float(),

            "sample_index": torch.tensor(
                sample_index,
                dtype=torch.long,
            ),
        }


# ============================================================
# 6. Train/validation split
# ============================================================

def load_video_ids(npz_path):

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:

        keypoint_key = find_key(
            data,
            PennActionDataset.KEYPOINT_KEYS,
            required=True,
        )

        sample_count = len(data[keypoint_key])

        video_id_key = find_key(
            data,
            PennActionDataset.VIDEO_ID_KEYS,
            required=False,
        )

        if video_id_key is not None:

            video_ids = np.asarray(
                [
                    decode_string(value)
                    for value in data[video_id_key]
                ],
                dtype=object,
            )

            return video_ids, sample_count

        image_path_key = find_key(
            data,
            PennActionDataset.IMAGE_PATH_KEYS,
            required=False,
        )

        if image_path_key is not None:

            video_ids = []

            for path_value in data[image_path_key]:

                frame_path = Path(
                    decode_string(path_value)
                )

                video_ids.append(
                    frame_path.parent.name
                )

            return np.asarray(
                video_ids,
                dtype=object,
            ), sample_count

    print(
        "\nWARNING:\n"
        "No video IDs were found in the NPZ.\n"
        "A sample-level split will be used.\n"
        "This can cause frames from the same video to appear "
        "in both training and validation sets.\n"
    )

    video_ids = np.asarray(
        [
            f"sample_{index:08d}"
            for index in range(sample_count)
        ],
        dtype=object,
    )

    return video_ids, sample_count


def make_video_level_split(
    video_ids,
    val_ratio,
    seed,
    max_samples=None,
):

    all_indices = np.arange(
        len(video_ids),
        dtype=np.int64,
    )

    if max_samples is not None:

        max_samples = min(
            max_samples,
            len(all_indices),
        )

        rng = np.random.default_rng(seed)

        all_indices = np.sort(
            rng.choice(
                all_indices,
                size=max_samples,
                replace=False,
            )
        )

        selected_video_ids = video_ids[all_indices]

    else:
        selected_video_ids = video_ids

    unique_video_ids = sorted(
        set(selected_video_ids.tolist())
    )

    if len(unique_video_ids) < 2:
        raise ValueError(
            "At least two videos are required."
        )

    random_generator = random.Random(seed)
    random_generator.shuffle(unique_video_ids)

    number_of_val_videos = max(
        1,
        round(len(unique_video_ids) * val_ratio),
    )

    number_of_val_videos = min(
        number_of_val_videos,
        len(unique_video_ids) - 1,
    )

    val_video_ids = set(
        unique_video_ids[:number_of_val_videos]
    )

    train_video_ids = set(
        unique_video_ids[number_of_val_videos:]
    )

    train_mask = np.asarray(
        [
            video_id in train_video_ids
            for video_id in selected_video_ids
        ]
    )

    val_mask = np.asarray(
        [
            video_id in val_video_ids
            for video_id in selected_video_ids
        ]
    )

    train_indices = all_indices[train_mask]
    val_indices = all_indices[val_mask]

    return (
        train_indices,
        val_indices,
        sorted(train_video_ids),
        sorted(val_video_ids),
    )


# ============================================================
# 7. SNN backbone
# ============================================================

class SpikingConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        beta,
        threshold,
        spike_gradient,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(
            out_channels
        )

        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_gradient,
            reset_mechanism="subtract",
        )

    def forward(self, x, membrane):

        current = self.batch_norm(
            self.conv(x)
        )

        spikes, membrane = self.lif(
            current,
            membrane,
        )

        return spikes, membrane


# ============================================================
# 8. Full SNN pose model
# ============================================================

class SNNHeatmapPoseModel(nn.Module):

    def __init__(self):
        super().__init__()

        spike_gradient = surrogate.fast_sigmoid(
            slope=SURROGATE_SLOPE
        )

        self.block1 = SpikingConvBlock(
            3,
            32,
            BETA,
            SPIKE_THRESHOLD,
            spike_gradient,
        )

        self.block2 = SpikingConvBlock(
            32,
            64,
            BETA,
            SPIKE_THRESHOLD,
            spike_gradient,
        )

        self.block3 = SpikingConvBlock(
            64,
            128,
            BETA,
            SPIKE_THRESHOLD,
            spike_gradient,
        )

        self.block4 = SpikingConvBlock(
            128,
            256,
            BETA,
            SPIKE_THRESHOLD,
            spike_gradient,
        )

        # Input:
        # 256 x 14 x 14
        #
        # Output:
        # 13 x 56 x 56

        self.heatmap_decoder = nn.Sequential(

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
                NUM_KEYPOINTS,
                kernel_size=1,
            ),
        )

        self.initialize_weights()

    def initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                (
                    nn.Conv2d,
                    nn.ConvTranspose2d,
                ),
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        images,
        return_spike_rate=False,
    ):
        """
        The same RGB image is directly presented to the SNN
        for NUM_STEPS simulation steps.
        """

        membrane1 = self.block1.lif.init_leaky()
        membrane2 = self.block2.lif.init_leaky()
        membrane3 = self.block3.lif.init_leaky()
        membrane4 = self.block4.lif.init_leaky()

        heatmap_sum = None

        total_spikes = images.new_tensor(0.0)
        total_spike_elements = 0

        for _ in range(NUM_STEPS):

            spikes1, membrane1 = self.block1(
                images,
                membrane1,
            )

            spikes2, membrane2 = self.block2(
                spikes1,
                membrane2,
            )

            spikes3, membrane3 = self.block3(
                spikes2,
                membrane3,
            )

            spikes4, membrane4 = self.block4(
                spikes3,
                membrane4,
            )

            heatmaps = self.heatmap_decoder(
                spikes4
            )

            if heatmap_sum is None:
                heatmap_sum = heatmaps
            else:
                heatmap_sum = (
                    heatmap_sum + heatmaps
                )

            if return_spike_rate:

                total_spikes += (
                    spikes1.detach().sum()
                    + spikes2.detach().sum()
                    + spikes3.detach().sum()
                    + spikes4.detach().sum()
                )

                total_spike_elements += (
                    spikes1.numel()
                    + spikes2.numel()
                    + spikes3.numel()
                    + spikes4.numel()
                )

        final_heatmaps = (
            heatmap_sum / float(NUM_STEPS)
        )

        if return_spike_rate:

            spike_rate = (
                total_spikes
                / max(total_spike_elements, 1)
            )

            return final_heatmaps, spike_rate

        return final_heatmaps


# ============================================================
# 9. Loss
# ============================================================

def masked_heatmap_mse(
    predictions,
    targets,
    visibility,
):
    """
    predictions:
        [B, J, H, W]

    targets:
        [B, J, H, W]

    visibility:
        [B, J]
    """

    joint_mask = visibility[
        :,
        :,
        None,
        None,
    ].to(predictions.dtype)

    squared_error = (
        predictions - targets
    ) ** 2

    masked_error = (
        squared_error * joint_mask
    )

    denominator = (
        joint_mask.sum()
        * predictions.shape[-2]
        * predictions.shape[-1]
    ).clamp_min(1.0)

    return masked_error.sum() / denominator


# ============================================================
# 10. Heatmap decoding
# ============================================================

def heatmaps_to_coordinates(heatmaps):
    """
    Convert heatmap maxima to coordinates in 224x224 space.

    Args:
        heatmaps:
            [B, J, H, W]

    Returns:
        coordinates:
            [B, J, 2]
    """

    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    heatmap_height = heatmaps.shape[2]
    heatmap_width = heatmaps.shape[3]

    flattened = heatmaps.reshape(
        batch_size,
        num_joints,
        -1,
    )

    maximum_indices = flattened.argmax(
        dim=-1
    )

    y = torch.div(
        maximum_indices,
        heatmap_width,
        rounding_mode="floor",
    )

    x = (
        maximum_indices
        % heatmap_width
    )

    scale_x = (
        IMAGE_SIZE / float(heatmap_width)
    )

    scale_y = (
        IMAGE_SIZE / float(heatmap_height)
    )

    coordinates = torch.stack(
        [
            x.float() * scale_x,
            y.float() * scale_y,
        ],
        dim=-1,
    )

    return coordinates


# ============================================================
# 11. PCK calculation
# ============================================================

def calculate_pck_counts(
    predictions,
    targets,
    visibility,
    threshold=PCK_THRESHOLD,
):
    """
    PCK normalization:
    diagonal of GT visible-joint bounding box.
    """

    total_correct = 0
    total_visible = 0

    for batch_index in range(
        targets.shape[0]
    ):

        visible_mask = (
            visibility[batch_index] > 0
        )

        if visible_mask.sum() < 2:
            continue

        visible_targets = targets[
            batch_index
        ][visible_mask]

        minimum_xy = visible_targets.min(
            dim=0
        ).values

        maximum_xy = visible_targets.max(
            dim=0
        ).values

        body_scale = torch.linalg.vector_norm(
            maximum_xy - minimum_xy
        ).clamp_min(1.0)

        distances = torch.linalg.vector_norm(
            predictions[batch_index]
            - targets[batch_index],
            dim=-1,
        )

        normalized_distances = (
            distances / body_scale
        )

        correct_mask = (
            normalized_distances
            <= threshold
        ) & visible_mask

        total_correct += int(
            correct_mask.sum().item()
        )

        total_visible += int(
            visible_mask.sum().item()
        )

    return total_correct, total_visible


# ============================================================
# 12. Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    total_spike_rate = 0.0
    total_batches = 0

    for batch in loader:

        images = batch["image"].to(device)
        target_heatmaps = batch["heatmaps"].to(device)
        visibility = batch["visibility"].to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions, spike_rate = model(
            images,
            return_spike_rate=True,
        )

        loss = masked_heatmap_mse(
            predictions,
            target_heatmaps,
            visibility,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss: {loss.item()}"
            )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
        )

        optimizer.step()

        batch_size = images.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        total_spike_rate += (
            spike_rate.item()
        )

        total_batches += 1

    average_loss = (
        total_loss / max(total_samples, 1)
    )

    average_spike_rate = (
        total_spike_rate
        / max(total_batches, 1)
    )

    return average_loss, average_spike_rate


# ============================================================
# 13. Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_correct = 0
    total_visible = 0

    total_spike_rate = 0.0
    total_batches = 0

    for batch in loader:

        images = batch["image"].to(device)

        target_heatmaps = batch[
            "heatmaps"
        ].to(device)

        target_keypoints = batch[
            "keypoints"
        ].to(device)

        visibility = batch[
            "visibility"
        ].to(device)

        predictions, spike_rate = model(
            images,
            return_spike_rate=True,
        )

        loss = masked_heatmap_mse(
            predictions,
            target_heatmaps,
            visibility,
        )

        predicted_keypoints = (
            heatmaps_to_coordinates(
                predictions
            )
        )

        correct, visible = (
            calculate_pck_counts(
                predicted_keypoints,
                target_keypoints,
                visibility,
                threshold=PCK_THRESHOLD,
            )
        )

        batch_size = images.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        total_correct += correct
        total_visible += visible

        total_spike_rate += (
            spike_rate.item()
        )

        total_batches += 1

    average_loss = (
        total_loss / max(total_samples, 1)
    )

    pck = (
        total_correct
        / max(total_visible, 1)
    )

    average_spike_rate = (
        total_spike_rate
        / max(total_batches, 1)
    )

    return (
        average_loss,
        pck,
        average_spike_rate,
    )


# ============================================================
# 14. Save utilities
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
):

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_val_loss":
                best_val_loss,

            "joint_names":
                JOINT_NAMES,

            "config": {
                "image_size": IMAGE_SIZE,
                "heatmap_size": HEATMAP_SIZE,
                "num_keypoints": NUM_KEYPOINTS,
                "heatmap_sigma": HEATMAP_SIGMA,
                "num_steps": NUM_STEPS,
                "beta": BETA,
                "spike_threshold":
                    SPIKE_THRESHOLD,
            },
        },
        path,
    )


def write_history_row(path, row):

    file_exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# 15. Main
# ============================================================

def main():

    set_seed(SEED)

    device = get_device()

    print("=" * 70)
    print("Penn Action SNN Heatmap Baseline")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {NPZ_PATH}")
    print(f"Output       : {OUTPUT_DIR}")
    print(f"Device       : {device}")
    print(f"SNN steps    : {NUM_STEPS}")
    print(f"Beta         : {BETA}")
    print("=" * 70)

    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{NPZ_PATH}"
        )

    # Print NPZ structure first.
    with np.load(
        NPZ_PATH,
        allow_pickle=True,
    ) as data:

        print("\nNPZ keys:")

        for key in data.files:

            array = data[key]

            print(
                f"  {key}: "
                f"shape={array.shape}, "
                f"dtype={array.dtype}"
            )

    video_ids, sample_count = load_video_ids(
        NPZ_PATH
    )

    (
        train_indices,
        val_indices,
        train_video_ids,
        val_video_ids,
    ) = make_video_level_split(
        video_ids=video_ids,
        val_ratio=VAL_RATIO,
        seed=SEED,
        max_samples=MAX_SAMPLES,
    )

    split_info = {
        "seed": SEED,

        "train_video_ids":
            train_video_ids,

        "val_video_ids":
            val_video_ids,

        "num_train_samples":
            len(train_indices),

        "num_val_samples":
            len(val_indices),
    }

    SPLIT_PATH.write_text(
        json.dumps(
            split_info,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDataset split:")
    print(f"Total samples : {sample_count}")
    print(f"Train samples : {len(train_indices)}")
    print(f"Val samples   : {len(val_indices)}")
    print(f"Train videos  : {len(train_video_ids)}")
    print(f"Val videos    : {len(val_video_ids)}")

    train_dataset = PennActionDataset(
        npz_path=NPZ_PATH,
        indices=train_indices,
        augment=True,
    )

    val_dataset = PennActionDataset(
        npz_path=NPZ_PATH,
        indices=val_indices,
        augment=False,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = SNNHeatmapPoseModel().to(
        device
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel:")
    print(f"Total parameters     : {total_parameters:,}")
    print(f"Trainable parameters : {trainable_parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
    )

    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()

    best_val_loss = float("inf")
    no_improvement_epochs = 0

    print("\nTraining starts...\n")

    for epoch in range(1, EPOCHS + 1):

        start_time = time.perf_counter()

        (
            train_loss,
            train_spike_rate,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        (
            val_loss,
            val_pck,
            val_spike_rate,
        ) = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        scheduler.step(val_loss)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        is_best = (
            val_loss < best_val_loss
        )

        if is_best:

            best_val_loss = val_loss
            no_improvement_epochs = 0

            save_checkpoint(
                path=BEST_CKPT_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )

        else:
            no_improvement_epochs += 1

        save_checkpoint(
            path=LAST_CKPT_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,

            f"val_pck@{PCK_THRESHOLD}":
                val_pck,

            "train_spike_rate":
                train_spike_rate,

            "val_spike_rate":
                val_spike_rate,

            "learning_rate":
                current_lr,

            "epoch_seconds":
                elapsed_seconds,

            "is_best":
                int(is_best),
        }

        write_history_row(
            HISTORY_PATH,
            history_row,
        )

        best_text = (
            " <-- best"
            if is_best
            else ""
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS:03d} | "
            f"train loss={train_loss:.6f} | "
            f"val loss={val_loss:.6f} | "
            f"PCK@{PCK_THRESHOLD:.2f}="
            f"{val_pck * 100:.2f}% | "
            f"spike rate={val_spike_rate:.4f} | "
            f"lr={current_lr:.2e} | "
            f"time={elapsed_seconds:.1f}s"
            f"{best_text}"
        )

        if (
            no_improvement_epochs
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping: validation loss "
                "has not improved for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    print("\nTraining finished.")
    print(f"Best checkpoint : {BEST_CKPT_PATH}")
    print(f"Last checkpoint : {LAST_CKPT_PATH}")
    print(f"History CSV     : {HISTORY_PATH}")
    print(f"Split JSON      : {SPLIT_PATH}")


if __name__ == "__main__":
    main()