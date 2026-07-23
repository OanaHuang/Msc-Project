# Scripts/NTU_RGBD/models/21_SpikeYOLO_Style_ILIF_Heatmap.py

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Integer-valued spike function
# ============================================================

class IntegerSpikeSTE(torch.autograd.Function):
    """
    Integer-valued spike quantizer with a straight-through
    gradient estimator.

    Forward:
        round and clamp the activation to [0, D]

    Backward:
        pass gradients through values inside the active range.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        max_spikes: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.max_spikes = max_spikes

        return torch.round(
            torch.clamp(
                x,
                min=0.0,
                max=float(max_spikes),
            )
        )

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors

        active = (
            (x >= 0.0)
            & (x <= float(ctx.max_spikes))
        ).to(dtype=grad_output.dtype)

        grad_x = grad_output * active

        return grad_x, None


class ILIFActivation(nn.Module):
    """
    Stateful Integer Leaky Integrate-and-Fire activation.

    One call processes one simulation step:

        membrane = beta * membrane + input
        integer_spikes = clamp(round(membrane / threshold), 0, D)
        membrane -= integer_spikes * threshold

    The output is an integer-valued tensor in [0, D].

    This test implementation follows the integer-valued
    training idea and is self-contained. It does not require
    snnTorch state management.
    """

    def __init__(
        self,
        beta: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()

        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be between 0 and 1.")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        if max_spikes <= 0:
            raise ValueError("max_spikes must be positive.")

        self.beta = float(beta)
        self.threshold = float(threshold)
        self.max_spikes = int(max_spikes)
        self.detach_reset = detach_reset

        self.register_buffer(
            "_membrane",
            torch.empty(0),
            persistent=False,
        )

    def reset_state(self) -> None:
        self._membrane = torch.empty(
            0,
            device=self._membrane.device,
        )

    def _state_is_compatible(
        self,
        x: torch.Tensor,
    ) -> bool:
        return (
            self._membrane.numel() > 0
            and self._membrane.shape == x.shape
            and self._membrane.device == x.device
            and self._membrane.dtype == x.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._state_is_compatible(x):
            self._membrane = torch.zeros_like(x)

        self._membrane = (
            self.beta * self._membrane
            + x
        )

        normalized_membrane = (
            self._membrane / self.threshold
        )

        spikes = IntegerSpikeSTE.apply(
            normalized_membrane,
            self.max_spikes,
        )

        reset_value = spikes * self.threshold

        if self.detach_reset:
            reset_value = reset_value.detach()

        self._membrane = (
            self._membrane - reset_value
        )

        return spikes


# ============================================================
# 2. SpikeYOLO-style building blocks using I-LIF
# ============================================================

class ILIFStandardConv(nn.Module):
    """
    Pre-activation block:

        I-LIF -> Conv2d -> BatchNorm2d
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
        max_spikes: int = 4,
    ) -> None:
        super().__init__()

        self.ilif = ILIFActivation(
            beta=beta,
            threshold=threshold,
            max_spikes=max_spikes,
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
        x = self.ilif(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class ILIFDownSampling(nn.Module):
    """
    SpikeYOLO-style downsampling with I-LIF pre-activation.

    The first RGB stem directly applies convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        first_layer: bool = False,
        beta: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
    ) -> None:
        super().__init__()

        if stride not in (2, 4):
            raise ValueError("stride must be 2 or 4.")

        self.first_layer = first_layer

        if not first_layer:
            self.ilif = ILIFActivation(
                beta=beta,
                threshold=threshold,
                max_spikes=max_spikes,
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
            x = self.ilif(x)

        x = self.conv(x)
        x = self.bn(x)
        return x


class ILIFSepConv(nn.Module):
    """
    I-LIF separable large-kernel residual block.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        beta: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
    ) -> None:
        super().__init__()

        hidden_channels = channels * expansion

        common = {
            "beta": beta,
            "threshold": threshold,
            "max_spikes": max_spikes,
        }

        self.pw1 = ILIFStandardConv(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            **common,
        )

        self.dw = ILIFStandardConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=7,
            padding=3,
            groups=hidden_channels,
            **common,
        )

        self.pw2 = ILIFStandardConv(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            **common,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        return x + identity


class ILIFMSAllConvBlock(nn.Module):
    """Shallow/high-resolution integer-spiking block."""

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        beta: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
    ) -> None:
        super().__init__()

        hidden_channels = channels * expansion

        common = {
            "beta": beta,
            "threshold": threshold,
            "max_spikes": max_spikes,
        }

        self.sep_conv = ILIFSepConv(
            channels=channels,
            expansion=2,
            **common,
        )

        self.conv1 = ILIFStandardConv(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            **common,
        )

        self.conv2 = ILIFStandardConv(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            **common,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sep_conv(x)
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity


class ILIFMSConvBlock(nn.Module):
    """Deeper integer-spiking semantic block."""

    def __init__(
        self,
        channels: int,
        expansion: int = 3,
        beta: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
    ) -> None:
        super().__init__()

        hidden_channels = channels * expansion

        common = {
            "beta": beta,
            "threshold": threshold,
            "max_spikes": max_spikes,
        }

        self.sep_conv = ILIFSepConv(
            channels=channels,
            expansion=2,
            **common,
        )

        self.expand = ILIFStandardConv(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            **common,
        )

        self.depthwise = ILIFStandardConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            **common,
        )

        self.project = ILIFStandardConv(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            **common,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sep_conv(x)
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        return x + identity


# ============================================================
# 3. SpikeYOLO-style I-LIF heatmap model
# ============================================================

class SpikeYOLOStyleILIFHeatmap(nn.Module):
    """
    Lightweight multi-scale I-LIF/ANN pose model.

    Input:
        B x 3 x 224 x 224

    SNN features:
        Stage 2: B x 128 x 28 x 28
        Stage 3: B x 256 x 14 x 14

    Output:
        B x num_joints x 56 x 56
    """

    def __init__(
        self,
        num_joints: int = 25,
        num_steps: int = 2,
        beta: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
    ) -> None:
        super().__init__()

        if num_joints <= 0:
            raise ValueError("num_joints must be positive.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        self.num_joints = num_joints
        self.num_steps = num_steps
        self.max_spikes = max_spikes

        common = {
            "beta": beta,
            "threshold": threshold,
            "max_spikes": max_spikes,
        }

        # 224 -> 56
        self.stem = ILIFDownSampling(
            in_channels=3,
            out_channels=64,
            stride=4,
            first_layer=True,
            **common,
        )

        self.stage1 = ILIFMSAllConvBlock(
            channels=64,
            **common,
        )

        # 56 -> 28
        self.down2 = ILIFDownSampling(
            in_channels=64,
            out_channels=128,
            stride=2,
            **common,
        )

        self.stage2 = nn.Sequential(
            ILIFMSAllConvBlock(channels=128, **common),
            ILIFMSAllConvBlock(channels=128, **common),
        )

        # 28 -> 14
        self.down3 = ILIFDownSampling(
            in_channels=128,
            out_channels=256,
            stride=2,
            **common,
        )

        self.stage3 = nn.Sequential(
            ILIFMSConvBlock(channels=256, **common),
            ILIFMSConvBlock(channels=256, **common),
        )

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
        for module in self.modules():
            if isinstance(module, ILIFActivation):
                module.reset_state()

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


def build_spikeyolo_style_ilif_heatmap(
    num_joints: int = 25,
    num_steps: int = 2,
    beta: float = 0.90,
    threshold: float = 1.0,
    max_spikes: int = 4,
) -> SpikeYOLOStyleILIFHeatmap:
    return SpikeYOLOStyleILIFHeatmap(
        num_joints=num_joints,
        num_steps=num_steps,
        beta=beta,
        threshold=threshold,
        max_spikes=max_spikes,
    )


def _test_model() -> None:
    model = build_spikeyolo_style_ilif_heatmap(
        num_joints=25,
        num_steps=2,
        max_spikes=4,
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
    print("21 - SpikeYOLO-style I-LIF heatmap model")
    print("=" * 72)
    print(f"Input shape:      {tuple(test_input.shape)}")
    print(f"Output shape:     {tuple(test_output.shape)}")
    print(f"Expected shape:   {expected_shape}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Integer levels:   0..{model.max_spikes}")

    if tuple(test_output.shape) != expected_shape:
        raise RuntimeError("Incorrect output shape.")

    print("Model test passed.")


if __name__ == "__main__":
    _test_model()