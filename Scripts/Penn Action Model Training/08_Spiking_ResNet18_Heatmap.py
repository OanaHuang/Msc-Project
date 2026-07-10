# Scripts/Penn Action Model Training/
# 08_Spiking_ResNet18_Heatmap.py

import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate

from pose_heatmap_common import (
    PROJECT_ROOT,
    NUM_KEYPOINTS,
    run_training,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "PennAction_Model_Training"
    / "08_Spiking_ResNet18_Heatmap"
)

NUM_STEPS = 2
BETA = 0.90
THRESHOLD = 1.0
SURROGATE_SLOPE = 25.0


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


class SpikingBasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        spike_gradient = (
            surrogate.fast_sigmoid(
                slope=SURROGATE_SLOPE
            )
        )

        self.conv1 = conv3x3(
            in_channels,
            out_channels,
            stride,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.lif1 = snn.Leaky(
            beta=BETA,
            threshold=THRESHOLD,
            spike_grad=spike_gradient,
            reset_mechanism="subtract",
        )

        self.conv2 = conv3x3(
            out_channels,
            out_channels,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.lif2 = snn.Leaky(
            beta=BETA,
            threshold=THRESHOLD,
            spike_grad=spike_gradient,
            reset_mechanism="subtract",
        )

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

        current = self.bn1(
            self.conv1(x)
        )

        spike1, state["mem1"] = (
            self.lif1(
                current,
                state["mem1"],
            )
        )

        current = self.bn2(
            self.conv2(spike1)
        )

        if self.downsample is not None:
            identity = self.downsample(
                identity
            )

        current = current + identity

        spike2, state["mem2"] = (
            self.lif2(
                current,
                state["mem2"],
            )
        )

        return spike2, state


class SpikingResNet18Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        spike_gradient = (
            surrogate.fast_sigmoid(
                slope=SURROGATE_SLOPE
            )
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
            beta=BETA,
            threshold=THRESHOLD,
            spike_grad=spike_gradient,
            reset_mechanism="subtract",
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self.make_layer(
            64,
            64,
            blocks=2,
            stride=1,
        )

        self.layer2 = self.make_layer(
            64,
            128,
            blocks=2,
            stride=2,
        )

        self.layer3 = self.make_layer(
            128,
            256,
            blocks=2,
            stride=2,
        )

        self.layer4 = self.make_layer(
            256,
            512,
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
                in_channels,
                out_channels,
                stride,
            )
        )

        for _ in range(1, blocks):
            layers.append(
                SpikingBasicBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                )
            )

        return layers

    def init_states(self):
        return {
            "stem":
                self.stem_lif.init_leaky(),

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
        for index, block in enumerate(
            layer
        ):
            x, states[index] = block(
                x,
                states[index],
            )

        return x, states

    def forward(
        self,
        images,
        states,
    ):
        current = self.bn1(
            self.conv1(images)
        )

        x, states["stem"] = (
            self.stem_lif(
                current,
                states["stem"],
            )
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


class SpikingResNet18Heatmap(nn.Module):

    def __init__(
        self,
        num_keypoints=NUM_KEYPOINTS,
    ):
        super().__init__()

        self.backbone = (
            SpikingResNet18Backbone()
        )

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
        states = (
            self.backbone.init_states()
        )

        heatmap_sum = None

        for _ in range(NUM_STEPS):
            features, states = (
                self.backbone(
                    images,
                    states,
                )
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


def main():
    model = SpikingResNet18Heatmap()

    run_training(
        model=model,

        method_name=(
            "Spiking_ResNet18_"
            "Heatmap_VideoSplit"
        ),

        output_dir=OUTPUT_DIR,

        best_model_name=(
            "best_Spiking_ResNet18_"
            "Heatmap.pth"
        ),

        last_model_name=(
            "last_Spiking_ResNet18_"
            "Heatmap.pth"
        ),

        extra_config={
            "backbone":
                "SpikingResNet18",

            "pretrained":
                False,

            "spiking":
                True,

            "num_steps":
                NUM_STEPS,

            "beta":
                BETA,

            "threshold":
                THRESHOLD,
        },
    )


if __name__ == "__main__":
    main()