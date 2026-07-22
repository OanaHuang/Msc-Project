# Scripts/NTU_RGBD/models/ms_spiking_resnet50_membrane_heatmap.py

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

import snntorch as snn
from snntorch import surrogate
from snntorch import utils as snn_utils

from .ms_spiking_resnet50_heatmap import MSResidualSpikingBottleneck


class MSSpikingResNet50MembraneHeatmap(nn.Module):
    """Model 18: MS-Spiking ResNet50 with mean membrane readout."""

    def __init__(
        self,
        num_joints: int = 25,
        num_steps: int = 2,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if num_joints <= 0:
            raise ValueError("num_joints must be positive.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be between 0 and 1.")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        if surrogate_slope <= 0.0:
            raise ValueError("surrogate_slope must be positive.")

        self.num_joints = num_joints
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope
        self.in_channels = 64

        spike_gradient = surrogate.fast_sigmoid(
            slope=surrogate_slope,
        )

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.stem_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_gradient,
            init_hidden=True,
            output=False,
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)

        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "deconv1",
                        nn.ConvTranspose2d(
                            2048,
                            256,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("bn1", nn.BatchNorm2d(256)),
                    ("relu1", nn.ReLU(inplace=True)),
                    (
                        "deconv2",
                        nn.ConvTranspose2d(
                            256,
                            256,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("bn2", nn.BatchNorm2d(256)),
                    ("relu2", nn.ReLU(inplace=True)),
                    (
                        "deconv3",
                        nn.ConvTranspose2d(
                            256,
                            256,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("bn3", nn.BatchNorm2d(256)),
                    ("relu3", nn.ReLU(inplace=True)),
                    (
                        "final_conv",
                        nn.Conv2d(
                            256,
                            num_joints,
                            kernel_size=1,
                            bias=True,
                        ),
                    ),
                ]
            )
        )

        self._initialize_weights()

        if pretrained:
            self._load_pretrained_resnet50()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        output_channels = (
            planes * MSResidualSpikingBottleneck.expansion
        )

        downsample: nn.Module | None = None

        if stride != 1 or self.in_channels != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        layers: list[nn.Module] = [
            MSResidualSpikingBottleneck(
                in_channels=self.in_channels,
                planes=planes,
                stride=stride,
                downsample=downsample,
                beta=self.beta,
                threshold=self.threshold,
                surrogate_slope=self.surrogate_slope,
            )
        ]

        self.in_channels = output_channels

        for _ in range(1, blocks):
            layers.append(
                MSResidualSpikingBottleneck(
                    in_channels=self.in_channels,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    beta=self.beta,
                    threshold=self.threshold,
                    surrogate_slope=self.surrogate_slope,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _load_pretrained_resnet50(self) -> None:
        source = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2,
        ).state_dict()
        current = self.state_dict()

        compatible = {
            key: value
            for key, value in source.items()
            if key in current and current[key].shape == value.shape
        }

        result = self.load_state_dict(
            compatible,
            strict=False,
        )

        print()
        print("Loaded compatible ImageNet ResNet50 weights")
        print("-" * 70)
        print(f"Loaded tensors:        {len(compatible)}")
        print(f"Missing model tensors: {len(result.missing_keys)}")
        print(f"Unexpected tensors:    {len(result.unexpected_keys)}")

    def reset_hidden_states(self) -> None:
        snn_utils.reset(self)

    def forward_backbone_step(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.stem_lif(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Continuous membrane-form feature from the final
        # residual stage. Unlike Model 16, no final LIF is
        # applied before the decoder readout.
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected B x 3 x H x W input, "
                f"received {tuple(x.shape)}."
            )
        if x.shape[1] != 3:
            raise ValueError("Expected three RGB channels.")

        self.reset_hidden_states()

        membrane_features: list[torch.Tensor] = []

        for _ in range(self.num_steps):
            membrane_features.append(
                self.forward_backbone_step(x)
            )

        # T x B x 2048 x 7 x 7 -> B x 2048 x 7 x 7
        readout = torch.stack(
            membrane_features,
            dim=0,
        ).mean(dim=0)

        return self.decoder(readout)


def build_ms_spiking_resnet50_membrane_heatmap(
    num_joints: int = 25,
    num_steps: int = 2,
    beta: float = 0.90,
    threshold: float = 1.0,
    surrogate_slope: float = 25.0,
    pretrained: bool = True,
) -> MSSpikingResNet50MembraneHeatmap:
    return MSSpikingResNet50MembraneHeatmap(
        num_joints=num_joints,
        num_steps=num_steps,
        beta=beta,
        threshold=threshold,
        surrogate_slope=surrogate_slope,
        pretrained=pretrained,
    )