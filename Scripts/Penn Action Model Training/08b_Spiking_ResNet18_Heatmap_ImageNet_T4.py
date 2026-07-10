# Scripts/Penn Action Model Training/
# 08b_Spiking_ResNet18_Heatmap_ImageNet_T4.py

from pathlib import Path
import random
import csv
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models

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
    / "PennAction_Model_Training"
    / "08b_Spiking_ResNet18_Heatmap_ImageNet_T4"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_Spiking_ResNet18_Heatmap_ImageNet_T4.pth"
)

LAST_MODEL_PATH = (
    OUTPUT_DIR
    / "last_Spiking_ResNet18_Heatmap_ImageNet_T4.pth"
)

SPLIT_JSON_PATH = (
    OUTPUT_DIR
    / "video_level_split.json"
)

SPLIT_CSV_PATH = (
    OUTPUT_DIR
    / "video_level_split_summary.csv"
)

# Reuse the exact video split produced by code 04.
BASELINE_SPLIT_JSON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "04_ResNet_Heatmap_Baseline"
    / "video_level_split.json"
)

IMAGE_SIZE = 224
HEATMAP_SIZE = 56
NUM_KEYPOINTS = 13

BATCH_SIZE = 32
EPOCHS = 20
NUM_WORKERS = 4

# Smaller LR for pretrained spiking backbone.
BACKBONE_LR = 1e-5

# Larger LR for randomly initialized heatmap decoder.
DECODER_LR = 1e-4

SIGMA = 2.0
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAX_VIDEOS = None


# ============================================================
# 2. SNN Config
# ============================================================

# Number of simulation steps for each static image.
NUM_STEPS = 4

# Membrane decay.
BETA = 0.90

# Firing threshold.
THRESHOLD = 1.0

# Surrogate-gradient slope.
SURROGATE_SLOPE = 25.0

# ImageNet initialization for Conv and BatchNorm layers.
USE_IMAGENET_PRETRAINED = True


# ============================================================
# 3. Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# 4. Device
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
    original_path = str(path_value).strip()
    processed_path = original_path

    if "Penn_Action" in processed_path:
        processed_path = (
            processed_path
            .split("Penn_Action")[-1]
            .lstrip("/")
        )

    candidates = [
        Path(processed_path),

        PROJECT_ROOT
        / processed_path,

        PROJECT_ROOT
        / "Datasets"
        / "Penn_Action"
        / processed_path,

        PROJECT_ROOT
        / "Datasets"
        / "Penn_Action"
        / "frames"
        / processed_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Image not found.\n"
        f"Original path: {original_path}\n"
        f"Processed path: {processed_path}"
    )


def make_gaussian_heatmap(
    x,
    y,
    heatmap_size,
    sigma,
):
    grid_y, grid_x = np.meshgrid(
        np.arange(heatmap_size),
        np.arange(heatmap_size),
        indexing="ij",
    )

    heatmap = np.exp(
        -(
            (grid_x - x) ** 2
            + (grid_y - y) ** 2
        )
        / (2 * sigma ** 2)
    )

    return heatmap.astype(np.float32)


# ============================================================
# 6. Video-Level Split
# ============================================================

def create_video_level_split(video_ids):
    unique_videos = sorted(
        list(set(video_ids))
    )

    random_generator = random.Random(
        RANDOM_SEED
    )
    random_generator.shuffle(
        unique_videos
    )

    if MAX_VIDEOS is not None:
        unique_videos = unique_videos[
            :MAX_VIDEOS
        ]

    num_videos = len(unique_videos)

    train_end = int(
        num_videos * TRAIN_RATIO
    )

    val_end = (
        train_end
        + int(num_videos * VAL_RATIO)
    )

    train_videos = unique_videos[
        :train_end
    ]

    val_videos = unique_videos[
        train_end:val_end
    ]

    test_videos = unique_videos[
        val_end:
    ]

    return build_split_from_video_lists(
        video_ids=video_ids,
        train_videos=train_videos,
        val_videos=val_videos,
        test_videos=test_videos,
    )


def build_split_from_video_lists(
    video_ids,
    train_videos,
    val_videos,
    test_videos,
):
    train_videos = [
        safe_video_id_to_str(video_id)
        for video_id in train_videos
    ]

    val_videos = [
        safe_video_id_to_str(video_id)
        for video_id in val_videos
    ]

    test_videos = [
        safe_video_id_to_str(video_id)
        for video_id in test_videos
    ]

    train_set = set(train_videos)
    val_set = set(val_videos)
    test_set = set(test_videos)

    train_indices = [
        index
        for index, video_id in enumerate(
            video_ids
        )
        if video_id in train_set
    ]

    val_indices = [
        index
        for index, video_id in enumerate(
            video_ids
        )
        if video_id in val_set
    ]

    test_indices = [
        index
        for index, video_id in enumerate(
            video_ids
        )
        if video_id in test_set
    ]

    return {
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def load_or_create_video_split(
    video_ids,
):
    if BASELINE_SPLIT_JSON_PATH.exists():
        print(
            "Loading the video split from code 04:"
        )
        print(BASELINE_SPLIT_JSON_PATH)

        with open(
            BASELINE_SPLIT_JSON_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            baseline_split = json.load(file)

        required_keys = [
            "train_videos",
            "val_videos",
            "test_videos",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in baseline_split
        ]

        if missing_keys:
            raise KeyError(
                "The code 04 split JSON is missing keys: "
                f"{missing_keys}"
            )

        split = build_split_from_video_lists(
            video_ids=video_ids,
            train_videos=baseline_split[
                "train_videos"
            ],
            val_videos=baseline_split[
                "val_videos"
            ],
            test_videos=baseline_split[
                "test_videos"
            ],
        )

        split_source = str(
            BASELINE_SPLIT_JSON_PATH
        )

    else:
        print(
            "Code 04 split JSON was not found."
        )
        print(
            "Creating a new split using the same "
            "random seed and ratios."
        )

        split = create_video_level_split(
            video_ids
        )

        split_source = (
            "Created by this script"
        )

    return split, split_source


def save_split_files(
    split,
    split_source,
):
    split_for_json = {
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "split_source": split_source,
        "num_train_videos": len(
            split["train_videos"]
        ),
        "num_val_videos": len(
            split["val_videos"]
        ),
        "num_test_videos": len(
            split["test_videos"]
        ),
        "num_train_frames": len(
            split["train_indices"]
        ),
        "num_val_frames": len(
            split["val_indices"]
        ),
        "num_test_frames": len(
            split["test_indices"]
        ),
        "train_videos": split[
            "train_videos"
        ],
        "val_videos": split[
            "val_videos"
        ],
        "test_videos": split[
            "test_videos"
        ],
    }

    with open(
        SPLIT_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_for_json,
            file,
            indent=4,
        )

    rows = [
        {
            "split": "train",
            "num_videos": len(
                split["train_videos"]
            ),
            "num_frames": len(
                split["train_indices"]
            ),
            "video_ids": " ".join(
                split["train_videos"]
            ),
        },
        {
            "split": "val",
            "num_videos": len(
                split["val_videos"]
            ),
            "num_frames": len(
                split["val_indices"]
            ),
            "video_ids": " ".join(
                split["val_videos"]
            ),
        },
        {
            "split": "test",
            "num_videos": len(
                split["test_videos"]
            ),
            "num_frames": len(
                split["test_indices"]
            ),
            "video_ids": " ".join(
                split["test_videos"]
            ),
        },
    ]

    with open(
        SPLIT_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "num_videos",
                "num_frames",
                "video_ids",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 7. Dataset
# ============================================================

class PennActionHeatmapDataset(Dataset):

    def __init__(
        self,
        npz_path,
        indices,
        image_size=224,
        heatmap_size=56,
        sigma=2.0,
    ):
        self.data = np.load(
            npz_path,
            allow_pickle=True,
        )

        self.image_paths = self.data[
            "image_paths"
        ]

        self.keypoints = self.data[
            "keypoints"
        ].astype(np.float32)

        self.visibility = self.data[
            "visibility"
        ].astype(np.float32)

        self.indices = list(indices)

        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        self.transform = T.Compose([
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]

        image_path = resolve_image_path(
            self.image_paths[real_index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        original_width, original_height = (
            image.size
        )

        image_tensor = self.transform(
            image
        )

        keypoints = self.keypoints[
            real_index
        ].copy()

        visibility = self.visibility[
            real_index
        ].copy()

        heatmaps = np.zeros(
            (
                NUM_KEYPOINTS,
                self.heatmap_size,
                self.heatmap_size,
            ),
            dtype=np.float32,
        )

        target_weights = np.zeros(
            (
                NUM_KEYPOINTS,
                1,
                1,
            ),
            dtype=np.float32,
        )

        for joint_index in range(
            NUM_KEYPOINTS
        ):
            if visibility[joint_index] <= 0:
                continue

            x, y = keypoints[
                joint_index
            ]

            if (
                x < 0
                or y < 0
                or x >= original_width
                or y >= original_height
            ):
                continue

            heatmap_x = (
                x
                / original_width
                * self.heatmap_size
            )

            heatmap_y = (
                y
                / original_height
                * self.heatmap_size
            )

            heatmaps[joint_index] = (
                make_gaussian_heatmap(
                    heatmap_x,
                    heatmap_y,
                    self.heatmap_size,
                    self.sigma,
                )
            )

            target_weights[
                joint_index,
                0,
                0,
            ] = 1.0

        return {
            "image": image_tensor,
            "heatmaps": torch.from_numpy(
                heatmaps
            ),
            "target_weights": (
                torch.from_numpy(
                    target_weights
                )
            ),
        }


# ============================================================
# 8. SNN Building Blocks
# ============================================================

def conv3x3(
    in_channels,
    out_channels,
    stride=1,
):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(
    in_channels,
    out_channels,
    stride=1,
):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def create_lif_node():
    spike_gradient = (
        surrogate.fast_sigmoid(
            slope=SURROGATE_SLOPE
        )
    )

    return snn.Leaky(
        beta=BETA,
        threshold=THRESHOLD,
        spike_grad=spike_gradient,
        reset_mechanism="subtract",
    )


class SpikingBasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.conv1 = conv3x3(
            in_channels,
            out_channels,
            stride,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.lif1 = create_lif_node()

        self.conv2 = conv3x3(
            out_channels,
            out_channels,
            stride=1,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.lif2 = create_lif_node()

        if (
            stride != 1
            or in_channels != out_channels
        ):
            self.downsample = nn.Sequential(
                conv1x1(
                    in_channels,
                    out_channels,
                    stride,
                ),
                nn.BatchNorm2d(
                    out_channels
                ),
            )
        else:
            self.downsample = None

    def init_state(self):
        return {
            "mem1": self.lif1.init_leaky(),
            "mem2": self.lif2.init_leaky(),
        }

    def forward(
        self,
        x,
        state,
    ):
        identity = x

        current = self.conv1(x)
        current = self.bn1(current)

        spike1, state["mem1"] = self.lif1(
            current,
            state["mem1"],
        )

        current = self.conv2(spike1)
        current = self.bn2(current)

        if self.downsample is not None:
            identity = self.downsample(
                identity
            )

        current = current + identity

        spike2, state["mem2"] = self.lif2(
            current,
            state["mem2"],
        )

        return spike2, state


# ============================================================
# 9. Spiking ResNet18 Backbone
# ============================================================

class SpikingResNet18Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.stem_lif = create_lif_node()

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self.make_layer(
            in_channels=64,
            out_channels=64,
            blocks=2,
            stride=1,
        )

        self.layer2 = self.make_layer(
            in_channels=64,
            out_channels=128,
            blocks=2,
            stride=2,
        )

        self.layer3 = self.make_layer(
            in_channels=128,
            out_channels=256,
            blocks=2,
            stride=2,
        )

        self.layer4 = self.make_layer(
            in_channels=256,
            out_channels=512,
            blocks=2,
            stride=2,
        )

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        blocks,
        stride,
    ):
        layers = nn.ModuleList()

        layers.append(
            SpikingBasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        )

        for _ in range(1, blocks):
            layers.append(
                SpikingBasicBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return layers

    def init_states(self):
        return {
            "stem": (
                self.stem_lif.init_leaky()
            ),

            "layer1": [
                block.init_state()
                for block in self.layer1
            ],

            "layer2": [
                block.init_state()
                for block in self.layer2
            ],

            "layer3": [
                block.init_state()
                for block in self.layer3
            ],

            "layer4": [
                block.init_state()
                for block in self.layer4
            ],
        }

    @staticmethod
    def forward_layer(
        x,
        layer,
        states,
    ):
        for block_index, block in enumerate(
            layer
        ):
            x, states[block_index] = block(
                x,
                states[block_index],
            )

        return x, states

    def forward(
        self,
        images,
        states,
    ):
        current = self.conv1(images)
        current = self.bn1(current)

        x, states["stem"] = self.stem_lif(
            current,
            states["stem"],
        )

        x = self.maxpool(x)

        x, states["layer1"] = (
            self.forward_layer(
                x,
                self.layer1,
                states["layer1"],
            )
        )

        x, states["layer2"] = (
            self.forward_layer(
                x,
                self.layer2,
                states["layer2"],
            )
        )

        x, states["layer3"] = (
            self.forward_layer(
                x,
                self.layer3,
                states["layer3"],
            )
        )

        x, states["layer4"] = (
            self.forward_layer(
                x,
                self.layer4,
                states["layer4"],
            )
        )

        return x, states


# ============================================================
# 10. Spiking ResNet18 Heatmap Model
# ============================================================

class SpikingResNet18Heatmap(nn.Module):

    def __init__(
        self,
        num_keypoints=13,
    ):
        super().__init__()

        self.backbone = (
            SpikingResNet18Backbone()
        )

        # Keep the heatmap decoder analog.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                512,
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

    def forward(self, images):
        # Reset membrane states for each independent image batch.
        states = self.backbone.init_states()

        heatmap_sum = None

        for _ in range(NUM_STEPS):
            features, states = self.backbone(
                images,
                states,
            )

            heatmaps = self.decoder(
                features
            )

            if heatmap_sum is None:
                heatmap_sum = heatmaps
            else:
                heatmap_sum = (
                    heatmap_sum
                    + heatmaps
                )

        return (
            heatmap_sum
            / float(NUM_STEPS)
        )


# ============================================================
# 11. ImageNet Weight Transfer
# ============================================================

def copy_batch_norm_state(
    source_batch_norm,
    target_batch_norm,
):
    target_batch_norm.load_state_dict(
        source_batch_norm.state_dict()
    )


def load_imagenet_resnet18_weights(
    spiking_model,
):
    print(
        "\nLoading ImageNet ResNet18 weights "
        "into the spiking backbone..."
    )

    ann_resnet = models.resnet18(
        weights=(
            models
            .ResNet18_Weights
            .IMAGENET1K_V1
        )
    )

    spiking_backbone = (
        spiking_model.backbone
    )

    spiking_backbone.conv1.load_state_dict(
        ann_resnet.conv1.state_dict()
    )

    copy_batch_norm_state(
        source_batch_norm=ann_resnet.bn1,
        target_batch_norm=spiking_backbone.bn1,
    )

    ann_layers = [
        ann_resnet.layer1,
        ann_resnet.layer2,
        ann_resnet.layer3,
        ann_resnet.layer4,
    ]

    spiking_layers = [
        spiking_backbone.layer1,
        spiking_backbone.layer2,
        spiking_backbone.layer3,
        spiking_backbone.layer4,
    ]

    for layer_index, (
        ann_layer,
        spiking_layer,
    ) in enumerate(
        zip(
            ann_layers,
            spiking_layers,
        ),
        start=1,
    ):
        if len(ann_layer) != len(
            spiking_layer
        ):
            raise RuntimeError(
                "Layer block count mismatch at "
                f"layer{layer_index}."
            )

        for block_index, (
            ann_block,
            spiking_block,
        ) in enumerate(
            zip(
                ann_layer,
                spiking_layer,
            )
        ):
            spiking_block.conv1.load_state_dict(
                ann_block.conv1.state_dict()
            )

            copy_batch_norm_state(
                source_batch_norm=ann_block.bn1,
                target_batch_norm=spiking_block.bn1,
            )

            spiking_block.conv2.load_state_dict(
                ann_block.conv2.state_dict()
            )

            copy_batch_norm_state(
                source_batch_norm=ann_block.bn2,
                target_batch_norm=spiking_block.bn2,
            )

            if (
                ann_block.downsample is not None
                and spiking_block.downsample
                is not None
            ):
                spiking_block.downsample.load_state_dict(
                    ann_block
                    .downsample
                    .state_dict()
                )

            elif (
                ann_block.downsample is None
                and spiking_block.downsample
                is None
            ):
                pass

            else:
                raise RuntimeError(
                    "Downsample structure mismatch at "
                    f"layer{layer_index}, "
                    f"block{block_index}."
                )

    print(
        "ImageNet weights loaded successfully."
    )


# ============================================================
# 12. Loss
# ============================================================

def heatmap_mse_loss(
    prediction,
    target,
    target_weights,
):
    loss = (
        prediction - target
    ) ** 2

    loss = loss * target_weights

    visible_count = (
        target_weights.sum()
    )

    if visible_count <= 0:
        return loss.mean()

    return (
        loss.sum()
        / visible_count
    )


# ============================================================
# 13. Train / Validate
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch_index, batch in enumerate(
        dataloader
    ):
        images = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        heatmaps = batch["heatmaps"].to(
            DEVICE,
            non_blocking=True,
        )

        target_weights = batch[
            "target_weights"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predicted_heatmaps = model(
            images
        )

        loss = heatmap_mse_loss(
            prediction=predicted_heatmaps,
            target=heatmaps,
            target_weights=target_weights,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss detected: "
                f"{loss.item()}"
            )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (
            batch_index + 1
        ) % 50 == 0:
            print(
                f"  Batch "
                f"[{batch_index + 1}/"
                f"{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return (
        total_loss
        / max(total_batches, 1)
    )


def validate_one_epoch(
    model,
    dataloader,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(
                DEVICE,
                non_blocking=True,
            )

            heatmaps = batch[
                "heatmaps"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            target_weights = batch[
                "target_weights"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            predicted_heatmaps = model(
                images
            )

            loss = heatmap_mse_loss(
                prediction=predicted_heatmaps,
                target=heatmaps,
                target_weights=target_weights,
            )

            total_loss += loss.item()
            total_batches += 1

    return (
        total_loss
        / max(total_batches, 1)
    )


# ============================================================
# 14. Checkpoint
# ============================================================

def create_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_loss,
    split,
    split_source,
):
    return {
        "method": (
            "Spiking_ResNet18_Heatmap_"
            "ImageNet_T4_VideoSplit"
        ),
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "best_val_loss": best_val_loss,

        "image_size": IMAGE_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "num_keypoints": NUM_KEYPOINTS,
        "sigma": SIGMA,

        "backbone": (
            "SpikingResNet18"
        ),
        "spiking": True,
        "pretrained": (
            USE_IMAGENET_PRETRAINED
        ),
        "pretrained_source": (
            "torchvision_"
            "ResNet18_"
            "IMAGENET1K_V1"
        ),
        "num_steps": NUM_STEPS,
        "beta": BETA,
        "threshold": THRESHOLD,
        "surrogate_slope": (
            SURROGATE_SLOPE
        ),
        "reset_mechanism": "subtract",
        "decoder_type": (
            "analog_deconvolution"
        ),

        "backbone_lr": BACKBONE_LR,
        "decoder_lr": DECODER_LR,

        "split_type": "video_level",
        "split_source": split_source,
        "train_videos": split[
            "train_videos"
        ],
        "val_videos": split[
            "val_videos"
        ],
        "test_videos": split[
            "test_videos"
        ],
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
    }


# ============================================================
# 15. Main
# ============================================================

def main():
    print("=" * 76)
    print(
        "Spiking ResNet18 Heatmap "
        "with ImageNet Initialization"
    )
    print("=" * 76)

    print(f"Device          : {DEVICE}")
    print(f"Project root    : {PROJECT_ROOT}")
    print(f"NPZ path        : {NPZ_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Simulation steps: {NUM_STEPS}")
    print(f"Beta            : {BETA}")
    print(f"Threshold       : {THRESHOLD}")
    print(
        f"ImageNet init   : "
        f"{USE_IMAGENET_PRETRAINED}"
    )
    print(
        f"Backbone LR     : "
        f"{BACKBONE_LR}"
    )
    print(
        f"Decoder LR      : "
        f"{DECODER_LR}"
    )

    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"NPZ not found: {NPZ_PATH}"
        )

    data = np.load(
        NPZ_PATH,
        allow_pickle=True,
    )

    video_ids = np.asarray([
        safe_video_id_to_str(
            video_id
        )
        for video_id in data["video_ids"]
    ])

    data.close()

    split, split_source = (
        load_or_create_video_split(
            video_ids
        )
    )

    save_split_files(
        split=split,
        split_source=split_source,
    )

    train_videos = set(
        split["train_videos"]
    )

    val_videos = set(
        split["val_videos"]
    )

    test_videos = set(
        split["test_videos"]
    )

    assert len(
        train_videos & val_videos
    ) == 0

    assert len(
        train_videos & test_videos
    ) == 0

    assert len(
        val_videos & test_videos
    ) == 0

    print(
        "\n========== Video-Level Split =========="
    )
    print(
        f"Split source: {split_source}"
    )
    print(
        f"Train videos: "
        f"{len(split['train_videos'])}"
    )
    print(
        f"Val videos  : "
        f"{len(split['val_videos'])}"
    )
    print(
        f"Test videos : "
        f"{len(split['test_videos'])}"
    )
    print(
        f"Train frames: "
        f"{len(split['train_indices'])}"
    )
    print(
        f"Val frames  : "
        f"{len(split['val_indices'])}"
    )
    print(
        f"Test frames : "
        f"{len(split['test_indices'])}"
    )

    train_dataset = (
        PennActionHeatmapDataset(
            NPZ_PATH,
            indices=split[
                "train_indices"
            ],
            image_size=IMAGE_SIZE,
            heatmap_size=HEATMAP_SIZE,
            sigma=SIGMA,
        )
    )

    val_dataset = (
        PennActionHeatmapDataset(
            NPZ_PATH,
            indices=split[
                "val_indices"
            ],
            image_size=IMAGE_SIZE,
            heatmap_size=HEATMAP_SIZE,
            sigma=SIGMA,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            torch.cuda.is_available()
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            torch.cuda.is_available()
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    model = SpikingResNet18Heatmap(
        num_keypoints=NUM_KEYPOINTS
    )

    if USE_IMAGENET_PRETRAINED:
        load_imagenet_resnet18_weights(
            model
        )

    model = model.to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters    : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": (
                    model
                    .backbone
                    .parameters()
                ),
                "lr": BACKBONE_LR,
            },
            {
                "params": (
                    model
                    .decoder
                    .parameters()
                ),
                "lr": DECODER_LR,
            },
        ]
    )

    best_val_loss = float("inf")

    print(
        "\n========== Training =========="
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        print(
            f"\nEpoch [{epoch}/{EPOCHS}]"
        )

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
        )

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: "
            f"{train_loss:.6f} | "
            f"Val Loss: "
            f"{val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = create_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                split=split,
                split_source=split_source,
            )

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print(
                "  New best model saved to:"
            )
            print(
                f"  {BEST_MODEL_PATH}"
            )

    last_checkpoint = create_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=EPOCHS,
        best_val_loss=best_val_loss,
        split=split,
        split_source=split_source,
    )

    torch.save(
        last_checkpoint,
        LAST_MODEL_PATH,
    )

    print(
        "\nTraining finished."
    )
    print(
        f"Best val loss: "
        f"{best_val_loss:.6f}"
    )
    print(
        f"Best model saved to: "
        f"{BEST_MODEL_PATH}"
    )
    print(
        f"Last model saved to: "
        f"{LAST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()