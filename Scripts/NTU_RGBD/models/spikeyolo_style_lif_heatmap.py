# Scripts/NTU_RGBD/models/20_SpikeYOLO_Style_LIF_Heatmap.py

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate
from snntorch import utils as snn_utils


# ============================================================
# 1. Standard LIF pre-activation
# ============================================================

class LIFActivation(nn.Module):
    """Stateful binary LIF activation implemented with snnTorch."""

    def __init__(
        self,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        spike_gradient = surrogate.fast_sigmoid(
            slope=surrogate_slope,
        )

        self.neuron = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_gradient,
            init_hidden=True,
            output=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neuron(x)


# ============================================================
# 2. SpikeYOLO-style building blocks using standard LIF
# ============================================================

class LIFStandardConv(nn.Module):
    """
    Pre-activation block:

        LIF -> Conv2d -> BatchNorm2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        self.lif = LIFActivation(
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lif(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class LIFDownSampling(nn.Module):
    """
    SpikeYOLO-style downsampling.

    The first RGB stem does not apply LIF before its convolution.
    Later downsampling layers use LIF pre-activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        first_layer: bool = False,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        if stride not in (2, 4):
            raise ValueError("stride must be 2 or 4.")

        self.first_layer = first_layer

        if not first_layer:
            self.lif = LIFActivation(
                beta=beta,
                threshold=threshold,
                surrogate_slope=surrogate_slope,
            )

        kernel_size = 7 if first_layer else 3
        padding = 2 if first_layer else 1

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.first_layer:
            x = self.lif(x)

        x = self.conv(x)
        x = self.bn(x)
        return x


class LIFSepConv(nn.Module):
    """
    Separable large-kernel residual block:

        LIF -> 1x1 pointwise expansion
        LIF -> 7x7 depthwise convolution
        LIF -> 1x1 pointwise projection
        residual addition
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        hidden_channels = channels * expansion

        self.pw1 = LIFStandardConv(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

        self.dw = LIFStandardConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=7,
            padding=3,
            groups=hidden_channels,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

        self.pw2 = LIFStandardConv(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        return x + identity


class LIFMSAllConvBlock(nn.Module):
    """
    Shallow/high-resolution block.

        SepConv
        -> 3x3 expansion convolution
        -> 3x3 projection convolution
        -> residual addition
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        hidden_channels = channels * expansion

        self.sep_conv = LIFSepConv(
            channels=channels,
            expansion=2,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

        self.conv1 = LIFStandardConv(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

        self.conv2 = LIFStandardConv(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sep_conv(x)
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity


class LIFMSConvBlock(nn.Module):
    """
    Deeper semantic block.

        SepConv
        -> 1x1 channel expansion
        -> 3x3 depthwise spatial mixing
        -> 1x1 channel projection
        -> residual addition
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 3,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        hidden_channels = channels * expansion

        self.sep_conv = LIFSepConv(
            channels=channels,
            expansion=2,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

        self.expand = LIFStandardConv(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

        self.depthwise = LIFStandardConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

        self.project = LIFStandardConv(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sep_conv(x)
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        return x + identity


# ============================================================
# 3. SpikeYOLO-style LIF heatmap model
# ============================================================

class SpikeYOLOStyleLIFHeatmap(nn.Module):
    """
    Lightweight multi-scale SNN-ANN pose model.

    Input:
        B x 3 x 224 x 224

    SNN features:
        Stage 2: B x 128 x 28 x 28
        Stage 3: B x 256 x 14 x 14

    Output:
        B x num_joints x 56 x 56

    The same RGB frame is presented for ``num_steps`` SNN
    simulation steps. LIF states persist inside one forward
    call and are reset before the next independent batch.
    """

    def __init__(
        self,
        num_joints: int = 25,
        num_steps: int = 2,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        if num_joints <= 0:
            raise ValueError("num_joints must be positive.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        self.num_joints = num_joints
        self.num_steps = num_steps

        common = {
            "beta": beta,
            "threshold": threshold,
            "surrogate_slope": surrogate_slope,
        }

        # 224 -> 56
        self.stem = LIFDownSampling(
            in_channels=3,
            out_channels=64,
            stride=4,
            first_layer=True,
            **common,
        )

        self.stage1 = LIFMSAllConvBlock(
            channels=64,
            **common,
        )

        # 56 -> 28
        self.down2 = LIFDownSampling(
            in_channels=64,
            out_channels=128,
            stride=2,
            **common,
        )

        self.stage2 = nn.Sequential(
            LIFMSAllConvBlock(channels=128, **common),
            LIFMSAllConvBlock(channels=128, **common),
        )

        # 28 -> 14
        self.down3 = LIFDownSampling(
            in_channels=128,
            out_channels=256,
            stride=2,
            **common,
        )

        self.stage3 = nn.Sequential(
            LIFMSConvBlock(channels=256, **common),
            LIFMSConvBlock(channels=256, **common),
        )

        # ANN fusion after temporal aggregation.
        self.pose_fusion = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1x1",
                        nn.Conv2d(
                            in_channels=128 + 256,
                            out_channels=128,
                            kernel_size=1,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(128)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "deconv",
                        nn.ConvTranspose2d(
                            in_channels=128,
                            out_channels=128,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("deconv_bn", nn.BatchNorm2d(128)),
                    ("deconv_relu", nn.ReLU(inplace=True)),
                    (
                        "refine",
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("refine_bn", nn.BatchNorm2d(128)),
                    ("refine_relu", nn.ReLU(inplace=True)),
                    (
                        "final_conv",
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=num_joints,
                            kernel_size=1,
                            bias=True,
                        ),
                    ),
                ]
            )
        )

        self._initialize_weights()

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

    def reset_hidden_states(self) -> None:
        snn_utils.reset(self)

    def forward_backbone_step(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)

        stage2 = self.down2(x)
        stage2 = self.stage2(stage2)

        stage3 = self.down3(stage2)
        stage3 = self.stage3(stage3)

        return stage2, stage3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected B x 3 x H x W, received {tuple(x.shape)}."
            )
        if x.shape[1] != 3:
            raise ValueError("Expected three RGB channels.")

        self.reset_hidden_states()

        stage2_steps: list[torch.Tensor] = []
        stage3_steps: list[torch.Tensor] = []

        for _ in range(self.num_steps):
            stage2_t, stage3_t = self.forward_backbone_step(x)
            stage2_steps.append(stage2_t)
            stage3_steps.append(stage3_t)

        stage2 = torch.stack(stage2_steps, dim=0).mean(dim=0)
        stage3 = torch.stack(stage3_steps, dim=0).mean(dim=0)

        stage3_up = F.interpolate(
            stage3,
            size=stage2.shape[-2:],
            mode="nearest",
        )

        fused = torch.cat([stage2, stage3_up], dim=1)
        fused = self.pose_fusion(fused)
        heatmaps = self.decoder(fused)

        return heatmaps


def build_spikeyolo_style_lif_heatmap(
    num_joints: int = 25,
    num_steps: int = 2,
    beta: float = 0.90,
    threshold: float = 1.0,
    surrogate_slope: float = 25.0,
) -> SpikeYOLOStyleLIFHeatmap:
    return SpikeYOLOStyleLIFHeatmap(
        num_joints=num_joints,
        num_steps=num_steps,
        beta=beta,
        threshold=threshold,
        surrogate_slope=surrogate_slope,
    )


def _test_model() -> None:
    model = build_spikeyolo_style_lif_heatmap(
        num_joints=25,
        num_steps=2,
    )
    model.eval()

    test_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        test_output = model(test_input)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    expected_shape = (1, 25, 56, 56)

    print("=" * 72)
    print("20 - SpikeYOLO-style standard-LIF heatmap model")
    print("=" * 72)
    print(f"Input shape:      {tuple(test_input.shape)}")
    print(f"Output shape:     {tuple(test_output.shape)}")
    print(f"Expected shape:   {expected_shape}")
    print(f"Total parameters: {total_parameters:,}")

    if tuple(test_output.shape) != expected_shape:
        raise RuntimeError("Incorrect output shape.")

    print("Model test passed.")


if __name__ == "__main__":
    _test_model()
