# Scripts/Penn Action Model Training/
# 07_ResNet50_Heatmap_Baseline.py

from pathlib import Path

import torch.nn as nn
from torchvision import models

from pose_heatmap_common import (
    PROJECT_ROOT,
    NUM_KEYPOINTS,
    run_training,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "07_ResNet50_Heatmap_Baseline"
)


class ResNet50HeatmapBaseline(nn.Module):

    def __init__(
        self,
        num_keypoints=NUM_KEYPOINTS,
    ):
        super().__init__()

        resnet = models.resnet50(
            weights=(
                models
                .ResNet50_Weights
                .IMAGENET1K_V2
            )
        )

        # 删除 avgpool 和 fc。
        # 输入 224x224：
        # 输出 [B, 2048, 7, 7]
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

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

    def forward(self, images):
        features = self.backbone(
            images
        )

        heatmaps = self.decoder(
            features
        )

        return heatmaps


def main():
    model = ResNet50HeatmapBaseline()

    run_training(
        model=model,

        method_name=(
            "ResNet50_Heatmap_"
            "Baseline_VideoSplit"
        ),

        output_dir=OUTPUT_DIR,

        best_model_name=(
            "best_ResNet50_"
            "Heatmap_Baseline.pth"
        ),

        last_model_name=(
            "last_ResNet50_"
            "Heatmap_Baseline.pth"
        ),

        extra_config={
            "backbone": "ResNet50",
            "pretrained": "ImageNet1K_V2",
            "spiking": False,
        },
    )


if __name__ == "__main__":
    main()