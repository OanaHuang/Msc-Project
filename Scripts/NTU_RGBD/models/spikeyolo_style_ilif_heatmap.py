from __future__ import annotations

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def merge_time_batch(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if x.ndim != 5:
        raise ValueError(f"Expected T x B x C x H x W, received {tuple(x.shape)}.")
    t, b = x.shape[:2]
    return x.flatten(0, 1), t, b


def restore_time_batch(x: torch.Tensor, t: int, b: int) -> torch.Tensor:
    return x.reshape(t, b, *x.shape[1:])


class IntegerSpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, max_spikes: int) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.max_spikes = int(max_spikes)
        return torch.round(torch.clamp(x, 0.0, float(max_spikes)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        active = ((x >= 0.0) & (x <= float(ctx.max_spikes))).to(grad_output.dtype)
        return grad_output * active, None


class MultiStepILIF(nn.Module):
    """Process the complete T dimension in one forward call.

    Membrane and spike states are local variables, so they never persist
    across independent batches. This matches SpikeYOLO's state-management
    pattern more closely than storing membrane state on the module object.
    """

    def __init__(
        self,
        decay: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.max_spikes = int(max_spikes)
        self.detach_reset = bool(detach_reset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("MultiStepILIF expects T x B x C x H x W.")

        membrane = torch.zeros_like(x[0])
        previous_spike = torch.zeros_like(x[0])
        outputs = []

        for t in range(x.shape[0]):
            if t == 0:
                membrane = x[t]
            else:
                reset_spike = previous_spike.detach() if self.detach_reset else previous_spike
                membrane = (membrane - reset_spike * self.threshold) * self.decay + x[t]

            spike = IntegerSpikeSTE.apply(
                membrane / self.threshold,
                self.max_spikes,
            )
            outputs.append(spike)
            previous_spike = spike

        return torch.stack(outputs, dim=0)


class TimeDistributedConvBN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, t, b = merge_time_batch(x)
        x = self.bn(self.conv(x))
        return restore_time_batch(x, t, b)


class ILIFStandardConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        decay: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
    ) -> None:
        super().__init__()
        self.ilif = MultiStepILIF(decay, threshold, max_spikes)
        self.conv_bn = TimeDistributedConvBN(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_bn(self.ilif(x))


class ILIFDownSampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        first_layer: bool = False,
        decay: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
    ) -> None:
        super().__init__()
        self.first_layer = first_layer
        if not first_layer:
            self.ilif = MultiStepILIF(decay, threshold, max_spikes)
        kernel_size = 7 if first_layer else 3
        padding = 2 if first_layer else 1
        self.conv_bn = TimeDistributedConvBN(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.first_layer:
            x = self.ilif(x)
        return self.conv_bn(x)


class ILIFSepConv(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, **kwargs) -> None:
        super().__init__()
        hidden = channels * expansion
        self.pw1 = ILIFStandardConv(channels, hidden, 1, **kwargs)
        self.dw = ILIFStandardConv(hidden, hidden, 7, padding=3, groups=hidden, **kwargs)
        self.pw2 = ILIFStandardConv(hidden, channels, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.pw2(self.dw(self.pw1(x)))
        return x + identity


class ILIFMSAllConvBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, **kwargs) -> None:
        super().__init__()
        hidden = channels * expansion
        self.sep = ILIFSepConv(channels, expansion=2, **kwargs)
        self.conv1 = ILIFStandardConv(channels, hidden, 3, padding=1, **kwargs)
        self.conv2 = ILIFStandardConv(hidden, channels, 3, padding=1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sep(x)
        identity = x
        return self.conv2(self.conv1(x)) + identity


class ILIFMSConvBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 3, **kwargs) -> None:
        super().__init__()
        hidden = channels * expansion
        self.sep = ILIFSepConv(channels, expansion=2, **kwargs)
        self.expand = ILIFStandardConv(channels, hidden, 1, **kwargs)
        self.depthwise = ILIFStandardConv(hidden, hidden, 3, padding=1, groups=hidden, **kwargs)
        self.project = ILIFStandardConv(hidden, channels, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sep(x)
        identity = x
        return self.project(self.depthwise(self.expand(x))) + identity


class SpikeYOLOStyleILIFHeatmap(nn.Module):
    def __init__(
        self,
        num_joints: int = 25,
        num_steps: int = 2,
        decay: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.num_steps = num_steps
        self.max_spikes = max_spikes

        common = dict(decay=decay, threshold=threshold, max_spikes=max_spikes)

        self.stem = ILIFDownSampling(3, 64, stride=4, first_layer=True, **common)
        self.stage1 = ILIFMSAllConvBlock(64, **common)
        self.down2 = ILIFDownSampling(64, 128, stride=2, **common)
        self.stage2 = nn.Sequential(
            ILIFMSAllConvBlock(128, **common),
            ILIFMSAllConvBlock(128, **common),
        )
        self.down3 = ILIFDownSampling(128, 256, stride=2, **common)
        self.stage3 = nn.Sequential(
            ILIFMSConvBlock(256, **common),
            ILIFMSConvBlock(256, **common),
        )

        self.pose_fusion = nn.Sequential(
            OrderedDict([
                ("conv1x1", nn.Conv2d(384, 128, 1, bias=False)),
                ("bn", nn.BatchNorm2d(128)),
                ("relu", nn.ReLU(inplace=True)),
            ])
        )

        self.decoder = nn.Sequential(
            OrderedDict([
                ("deconv", nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False)),
                ("deconv_bn", nn.BatchNorm2d(128)),
                ("deconv_relu", nn.ReLU(inplace=True)),
                ("refine", nn.Conv2d(128, 128, 3, padding=1, bias=False)),
                ("refine_bn", nn.BatchNorm2d(128)),
                ("refine_relu", nn.ReLU(inplace=True)),
                ("final_conv", nn.Conv2d(128, num_joints, 1)),
            ])
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_backbone(self, x: torch.Tensor):
        x = self.stage1(self.stem(x))
        stage2 = self.stage2(self.down2(x))
        stage3 = self.stage3(self.down3(stage2))
        return stage2, stage3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected B x 3 x H x W, received {tuple(x.shape)}.")

        # SpikeYOLO-style expansion to a full time sequence.
        x = x.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)
        stage2, stage3 = self.forward_backbone(x)

        # Temporal readout.
        stage2 = stage2.mean(dim=0)
        stage3 = stage3.mean(dim=0)

        stage3 = F.interpolate(stage3, size=stage2.shape[-2:], mode="nearest")
        fused = self.pose_fusion(torch.cat([stage2, stage3], dim=1))
        return self.decoder(fused)


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
        decay=beta,
        threshold=threshold,
        max_spikes=max_spikes,
    )


def _test_model() -> None:
    model = build_spikeyolo_style_ilif_heatmap()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", tuple(y.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    if tuple(y.shape) != (1, 25, 56, 56):
        raise RuntimeError("Incorrect output shape.")

    # Verify backward works without hidden-state leakage.
    loss = y.square().mean()
    loss.backward()
    print("Forward and backward tests passed.")


if __name__ == "__main__":
    _test_model()