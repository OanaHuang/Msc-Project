# Scripts/NTU_RGBD/models/resnet_heatmap.py

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNetHeatmapModel(nn.Module):
    """
    ResNet backbone + deconvolution decoder for 2D heatmap regression.

    Input:
        [B, 3, 224, 224]

    Output:
        [B, num_joints, 56, 56]
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_joints: int = 25,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        backbone_name = backbone_name.lower().strip()

        if backbone_name == "resnet18":
            weights = (
                models.ResNet18_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )

            backbone = models.resnet18(
                weights=weights
            )

            backbone_channels = 512

        elif backbone_name == "resnet50":
            weights = (
                models.ResNet50_Weights.IMAGENET1K_V2
                if pretrained
                else None
            )

            backbone = models.resnet50(
                weights=weights
            )

            backbone_channels = 2048

        else:
            raise ValueError(
                "backbone_name must be "
                "'resnet18' or 'resnet50'"
            )

        self.backbone_name = backbone_name
        self.num_joints = num_joints

        # Remove average pooling and classification head.
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # 224x224 input:
        # ResNet output = 7x7
        # 7 -> 14 -> 28 -> 56
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                backbone_channels,
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
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = nn.Conv2d(
            in_channels=256,
            out_channels=num_joints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        for module in self.decoder.modules():
            if isinstance(
                module,
                nn.ConvTranspose2d,
            ):
                nn.init.normal_(
                    module.weight,
                    std=0.001,
                )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.constant_(
                    module.weight,
                    1.0,
                )
                nn.init.constant_(
                    module.bias,
                    0.0,
                )

        nn.init.normal_(
            self.heatmap_head.weight,
            std=0.001,
        )

        if self.heatmap_head.bias is not None:
            nn.init.constant_(
                self.heatmap_head.bias,
                0.0,
            )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [B, C, H, W]"
            )

        features = self.backbone(
            images
        )

        features = self.decoder(
            features
        )

        heatmaps = self.heatmap_head(
            features
        )

        return heatmaps


def build_resnet18_heatmap(
    num_joints: int = 25,
    pretrained: bool = True,
) -> ResNetHeatmapModel:
    return ResNetHeatmapModel(
        backbone_name="resnet18",
        num_joints=num_joints,
        pretrained=pretrained,
    )


def build_resnet50_heatmap(
    num_joints: int = 25,
    pretrained: bool = True,
) -> ResNetHeatmapModel:
    return ResNetHeatmapModel(
        backbone_name="resnet50",
        num_joints=num_joints,
        pretrained=pretrained,
    )


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total_parameters,
        trainable_parameters,
    )


def main() -> None:
    model = build_resnet50_heatmap(
        num_joints=25,
        pretrained=False,
    )

    test_input = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        output = model(
            test_input
        )

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    print("=" * 70)
    print("ResNet50 heatmap model test")
    print("=" * 70)

    print(
        f"Input shape:       "
        f"{tuple(test_input.shape)}"
    )

    print(
        f"Output shape:      "
        f"{tuple(output.shape)}"
    )

    print(
        f"Total parameters:  "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable params:  "
        f"{trainable_parameters:,}"
    )

    expected_output_shape = (
        2,
        25,
        56,
        56,
    )

    assert tuple(output.shape) == expected_output_shape
    assert torch.isfinite(output).all()

    print()
    print("ResNet50 heatmap model test passed.")


if __name__ == "__main__":
    main()