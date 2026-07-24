from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


FusionType = Literal[
    "concat",
    "stage2_only",
    "stage3_only",
]

ReadoutType = Literal[
    "mean",
    "sum",
    "last",
]

DecoderType = Literal[
    "default",
    "no_refine",
    "bilinear",
]

BackboneVariant = Literal[
    "default",
    "shallow",
]


# ============================================================
# 1. Tensor utilities
# ============================================================

def merge_time_batch(
    x: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    if x.ndim != 5:
        raise ValueError(
            "Expected T x B x C x H x W, "
            f"received {tuple(x.shape)}."
        )

    t, b = x.shape[:2]

    return x.flatten(0, 1), t, b


def restore_time_batch(
    x: torch.Tensor,
    t: int,
    b: int,
) -> torch.Tensor:
    return x.reshape(
        t,
        b,
        *x.shape[1:],
    )


# ============================================================
# 2. Integer spike function
# ============================================================

class IntegerSpikeSTE(
    torch.autograd.Function
):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        max_spikes: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.max_spikes = int(max_spikes)

        return torch.round(
            torch.clamp(
                x,
                0.0,
                float(max_spikes),
            )
        )

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        (x,) = ctx.saved_tensors

        active = (
            (x >= 0.0)
            & (
                x
                <= float(ctx.max_spikes)
            )
        ).to(
            grad_output.dtype
        )

        return (
            grad_output * active,
            None,
        )


# ============================================================
# 3. Multi-step I-LIF
# ============================================================

class MultiStepILIF(
    nn.Module
):
    """
    Process the complete T dimension in one forward call.

    Membrane and spike states are local variables and therefore
    do not persist across independent batches.

    max_spikes controls the integer spike range:
        max_spikes=1 -> binary-valued I-LIF
        max_spikes=2 -> integer range 0..2
        max_spikes=4 -> integer range 0..4
        max_spikes=8 -> integer range 0..8
    """

    def __init__(
        self,
        decay: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()

        if decay < 0.0:
            raise ValueError(
                "decay must be non-negative."
            )

        if threshold <= 0.0:
            raise ValueError(
                "threshold must be positive."
            )

        if max_spikes < 1:
            raise ValueError(
                "max_spikes must be at least 1."
            )

        self.decay = float(decay)
        self.threshold = float(threshold)
        self.max_spikes = int(max_spikes)
        self.detach_reset = bool(
            detach_reset
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "MultiStepILIF expects "
                "T x B x C x H x W."
            )

        membrane = torch.zeros_like(
            x[0]
        )

        previous_spike = torch.zeros_like(
            x[0]
        )

        outputs: list[
            torch.Tensor
        ] = []

        for step in range(
            x.shape[0]
        ):
            if step == 0:
                membrane = x[step]

            else:
                reset_spike = (
                    previous_spike.detach()
                    if self.detach_reset
                    else previous_spike
                )

                membrane = (
                    (
                        membrane
                        - reset_spike
                        * self.threshold
                    )
                    * self.decay
                    + x[step]
                )

            spike = IntegerSpikeSTE.apply(
                membrane
                / self.threshold,
                self.max_spikes,
            )

            outputs.append(
                spike
            )

            previous_spike = spike

        return torch.stack(
            outputs,
            dim=0,
        )


# ============================================================
# 4. Time-distributed convolution
# ============================================================

class TimeDistributedConvBN(
    nn.Module
):
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

        self.bn = nn.BatchNorm2d(
            out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x, t, b = merge_time_batch(
            x
        )

        x = self.bn(
            self.conv(x)
        )

        return restore_time_batch(
            x,
            t,
            b,
        )


# ============================================================
# 5. I-LIF convolution modules
# ============================================================

class ILIFStandardConv(
    nn.Module
):
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

        self.ilif = MultiStepILIF(
            decay=decay,
            threshold=threshold,
            max_spikes=max_spikes,
        )

        self.conv_bn = (
            TimeDistributedConvBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv_bn(
            self.ilif(x)
        )


class ILIFDownSampling(
    nn.Module
):
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

        self.first_layer = bool(
            first_layer
        )

        if not self.first_layer:
            self.ilif = MultiStepILIF(
                decay=decay,
                threshold=threshold,
                max_spikes=max_spikes,
            )

        kernel_size = (
            7
            if self.first_layer
            else 3
        )

        padding = (
            2
            if self.first_layer
            else 1
        )

        self.conv_bn = (
            TimeDistributedConvBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not self.first_layer:
            x = self.ilif(x)

        return self.conv_bn(x)


class ILIFSepConv(
    nn.Module
):
    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()

        hidden = (
            channels
            * expansion
        )

        self.pw1 = ILIFStandardConv(
            channels,
            hidden,
            1,
            **kwargs,
        )

        self.dw = ILIFStandardConv(
            hidden,
            hidden,
            7,
            padding=3,
            groups=hidden,
            **kwargs,
        )

        self.pw2 = ILIFStandardConv(
            hidden,
            channels,
            1,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        identity = x

        x = self.pw2(
            self.dw(
                self.pw1(x)
            )
        )

        return x + identity


class ILIFMSAllConvBlock(
    nn.Module
):
    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()

        hidden = (
            channels
            * expansion
        )

        self.sep = ILIFSepConv(
            channels,
            expansion=2,
            **kwargs,
        )

        self.conv1 = ILIFStandardConv(
            channels,
            hidden,
            3,
            padding=1,
            **kwargs,
        )

        self.conv2 = ILIFStandardConv(
            hidden,
            channels,
            3,
            padding=1,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.sep(x)

        identity = x

        return (
            self.conv2(
                self.conv1(x)
            )
            + identity
        )


class ILIFMSConvBlock(
    nn.Module
):
    def __init__(
        self,
        channels: int,
        expansion: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()

        hidden = (
            channels
            * expansion
        )

        self.sep = ILIFSepConv(
            channels,
            expansion=2,
            **kwargs,
        )

        self.expand = ILIFStandardConv(
            channels,
            hidden,
            1,
            **kwargs,
        )

        self.depthwise = ILIFStandardConv(
            hidden,
            hidden,
            3,
            padding=1,
            groups=hidden,
            **kwargs,
        )

        self.project = ILIFStandardConv(
            hidden,
            channels,
            1,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.sep(x)

        identity = x

        return (
            self.project(
                self.depthwise(
                    self.expand(x)
                )
            )
            + identity
        )


# ============================================================
# 6. Experimental SpikeYOLO-style I-LIF heatmap model
# ============================================================

class SpikeYOLOStyleILIFHeatmapExperiment(
    nn.Module
):
    """
    Parameterised Model 21 for controlled ablation experiments.

    Baseline configuration:
        fusion_type="concat"
        readout_type="mean"
        decoder_type="default"
        backbone_variant="default"
        max_spikes=4
        num_steps=2

    Supported experiments:
        B0:
            baseline configuration

        S1:
            fusion_type="stage2_only"

        S2:
            fusion_type="stage3_only"

        N1:
            max_spikes=1

        D2:
            max_spikes=2

        D8:
            max_spikes=8

        T1:
            num_steps=1

        T4:
            num_steps=4

        R1:
            readout_type="sum"

        R2:
            readout_type="last"

        C1:
            decoder_type="no_refine"

        C2:
            decoder_type="bilinear"

        F1:
            backbone_variant="shallow"
    """

    VALID_FUSION_TYPES = {
        "concat",
        "stage2_only",
        "stage3_only",
    }

    VALID_READOUT_TYPES = {
        "mean",
        "sum",
        "last",
    }

    VALID_DECODER_TYPES = {
        "default",
        "no_refine",
        "bilinear",
    }

    VALID_BACKBONE_VARIANTS = {
        "default",
        "shallow",
    }

    def __init__(
        self,
        num_joints: int = 25,
        num_steps: int = 2,
        decay: float = 0.90,
        threshold: float = 1.0,
        max_spikes: int = 4,
        fusion_type: FusionType = "concat",
        readout_type: ReadoutType = "mean",
        decoder_type: DecoderType = "default",
        backbone_variant: BackboneVariant = "default",
    ) -> None:
        super().__init__()

        self._validate_configuration(
            num_joints=num_joints,
            num_steps=num_steps,
            max_spikes=max_spikes,
            fusion_type=fusion_type,
            readout_type=readout_type,
            decoder_type=decoder_type,
            backbone_variant=backbone_variant,
        )

        self.num_joints = int(
            num_joints
        )

        self.num_steps = int(
            num_steps
        )

        self.max_spikes = int(
            max_spikes
        )

        self.fusion_type = str(
            fusion_type
        )

        self.readout_type = str(
            readout_type
        )

        self.decoder_type = str(
            decoder_type
        )

        self.backbone_variant = str(
            backbone_variant
        )

        common = dict(
            decay=decay,
            threshold=threshold,
            max_spikes=max_spikes,
        )

        # Keep all original module names and shapes unchanged.
        # This preserves compatibility with the original Model 21
        # checkpoint when the baseline configuration is used.
        self.stem = ILIFDownSampling(
            3,
            64,
            stride=4,
            first_layer=True,
            **common,
        )

        self.stage1 = ILIFMSAllConvBlock(
            64,
            **common,
        )

        self.down2 = ILIFDownSampling(
            64,
            128,
            stride=2,
            **common,
        )

        self.stage2 = nn.Sequential(
            ILIFMSAllConvBlock(
                128,
                **common,
            ),
            ILIFMSAllConvBlock(
                128,
                **common,
            ),
        )

        self.down3 = ILIFDownSampling(
            128,
            256,
            stride=2,
            **common,
        )

        self.stage3 = nn.Sequential(
            ILIFMSConvBlock(
                256,
                **common,
            ),
            ILIFMSConvBlock(
                256,
                **common,
            ),
        )

        self.pose_fusion = nn.Sequential(
            OrderedDict([
                (
                    "conv1x1",
                    nn.Conv2d(
                        384,
                        128,
                        1,
                        bias=False,
                    ),
                ),
                (
                    "bn",
                    nn.BatchNorm2d(
                        128
                    ),
                ),
                (
                    "relu",
                    nn.ReLU(
                        inplace=True
                    ),
                ),
            ])
        )

        self.decoder = nn.Sequential(
            OrderedDict([
                (
                    "deconv",
                    nn.ConvTranspose2d(
                        128,
                        128,
                        4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                ),
                (
                    "deconv_bn",
                    nn.BatchNorm2d(
                        128
                    ),
                ),
                (
                    "deconv_relu",
                    nn.ReLU(
                        inplace=True
                    ),
                ),
                (
                    "refine",
                    nn.Conv2d(
                        128,
                        128,
                        3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (
                    "refine_bn",
                    nn.BatchNorm2d(
                        128
                    ),
                ),
                (
                    "refine_relu",
                    nn.ReLU(
                        inplace=True
                    ),
                ),
                (
                    "final_conv",
                    nn.Conv2d(
                        128,
                        num_joints,
                        1,
                    ),
                ),
            ])
        )

        self._initialize_weights()

    @classmethod
    def _validate_configuration(
        cls,
        num_joints: int,
        num_steps: int,
        max_spikes: int,
        fusion_type: str,
        readout_type: str,
        decoder_type: str,
        backbone_variant: str,
    ) -> None:
        if num_joints < 1:
            raise ValueError(
                "num_joints must be at least 1."
            )

        if num_steps < 1:
            raise ValueError(
                "num_steps must be at least 1."
            )

        if max_spikes < 1:
            raise ValueError(
                "max_spikes must be at least 1."
            )

        if (
            fusion_type
            not in cls.VALID_FUSION_TYPES
        ):
            raise ValueError(
                "Unsupported fusion_type: "
                f"{fusion_type}. Supported values: "
                f"{sorted(cls.VALID_FUSION_TYPES)}"
            )

        if (
            readout_type
            not in cls.VALID_READOUT_TYPES
        ):
            raise ValueError(
                "Unsupported readout_type: "
                f"{readout_type}. Supported values: "
                f"{sorted(cls.VALID_READOUT_TYPES)}"
            )

        if (
            decoder_type
            not in cls.VALID_DECODER_TYPES
        ):
            raise ValueError(
                "Unsupported decoder_type: "
                f"{decoder_type}. Supported values: "
                f"{sorted(cls.VALID_DECODER_TYPES)}"
            )

        if (
            backbone_variant
            not in cls.VALID_BACKBONE_VARIANTS
        ):
            raise ValueError(
                "Unsupported backbone_variant: "
                f"{backbone_variant}. Supported values: "
                f"{sorted(cls.VALID_BACKBONE_VARIANTS)}"
            )

    def _initialize_weights(
        self,
    ) -> None:
        for module in self.modules():
            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if (
                    module.bias
                    is not None
                ):
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
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
                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def _forward_stage2(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.down2(x)

        if (
            self.backbone_variant
            == "shallow"
        ):
            return self.stage2[0](x)

        return self.stage2(x)

    def _forward_stage3(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.down3(x)

        if (
            self.backbone_variant
            == "shallow"
        ):
            return self.stage3[0](x)

        return self.stage3(x)

    def forward_backbone(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        x = self.stage1(
            self.stem(x)
        )

        stage2 = self._forward_stage2(
            x
        )

        stage3 = self._forward_stage3(
            stage2
        )

        return stage2, stage3

    def _temporal_readout(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.readout_type
            == "mean"
        ):
            return x.mean(
                dim=0
            )

        if (
            self.readout_type
            == "sum"
        ):
            return x.sum(
                dim=0
            )

        if (
            self.readout_type
            == "last"
        ):
            return x[-1]

        raise RuntimeError(
            "Unexpected readout_type: "
            f"{self.readout_type}"
        )

    def _fuse_features(
        self,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
    ) -> torch.Tensor:
        stage3 = F.interpolate(
            stage3,
            size=stage2.shape[-2:],
            mode="nearest",
        )

        # Preserve the original 384-channel fusion input.
        # For single-stage ablations, the removed branch is
        # replaced with zeros of the same shape.
        if (
            self.fusion_type
            == "stage2_only"
        ):
            stage3 = torch.zeros_like(
                stage3
            )

        elif (
            self.fusion_type
            == "stage3_only"
        ):
            stage2 = torch.zeros_like(
                stage2
            )

        concatenated = torch.cat(
            [
                stage2,
                stage3,
            ],
            dim=1,
        )

        return self.pose_fusion(
            concatenated
        )

    def _decode_default(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(x)

    def _decode_without_refinement(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.decoder.deconv(x)
        x = self.decoder.deconv_bn(x)
        x = self.decoder.deconv_relu(x)

        return self.decoder.final_conv(
            x
        )

    def _decode_bilinear(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )

        # Reuse the existing post-upsampling modules.
        x = self.decoder.deconv_bn(x)
        x = self.decoder.deconv_relu(x)
        x = self.decoder.refine(x)
        x = self.decoder.refine_bn(x)
        x = self.decoder.refine_relu(x)

        return self.decoder.final_conv(
            x
        )

    def _decode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.decoder_type
            == "default"
        ):
            return self._decode_default(
                x
            )

        if (
            self.decoder_type
            == "no_refine"
        ):
            return (
                self._decode_without_refinement(
                    x
                )
            )

        if (
            self.decoder_type
            == "bilinear"
        ):
            return self._decode_bilinear(
                x
            )

        raise RuntimeError(
            "Unexpected decoder_type: "
            f"{self.decoder_type}"
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            x.ndim != 4
            or x.shape[1] != 3
        ):
            raise ValueError(
                "Expected B x 3 x H x W, "
                f"received {tuple(x.shape)}."
            )

        # Expand one RGB frame into the internal SNN time axis.
        x = x.unsqueeze(0).repeat(
            self.num_steps,
            1,
            1,
            1,
            1,
        )

        stage2, stage3 = (
            self.forward_backbone(x)
        )

        stage2 = (
            self._temporal_readout(
                stage2
            )
        )

        stage3 = (
            self._temporal_readout(
                stage3
            )
        )

        fused = self._fuse_features(
            stage2,
            stage3,
        )

        return self._decode(
            fused
        )

    def get_experiment_configuration(
        self,
    ) -> dict[str, object]:
        return {
            "num_joints": self.num_joints,
            "num_steps": self.num_steps,
            "max_spikes": self.max_spikes,
            "fusion_type": self.fusion_type,
            "readout_type": self.readout_type,
            "decoder_type": self.decoder_type,
            "backbone_variant": (
                self.backbone_variant
            ),
        }


# ============================================================
# 7. Model builder
# ============================================================

def build_spikeyolo_style_ilif_heatmap_experiment(
    num_joints: int = 25,
    num_steps: int = 2,
    beta: float = 0.90,
    threshold: float = 1.0,
    max_spikes: int = 4,
    fusion_type: FusionType = "concat",
    readout_type: ReadoutType = "mean",
    decoder_type: DecoderType = "default",
    backbone_variant: BackboneVariant = "default",
) -> SpikeYOLOStyleILIFHeatmapExperiment:
    return SpikeYOLOStyleILIFHeatmapExperiment(
        num_joints=num_joints,
        num_steps=num_steps,
        decay=beta,
        threshold=threshold,
        max_spikes=max_spikes,
        fusion_type=fusion_type,
        readout_type=readout_type,
        decoder_type=decoder_type,
        backbone_variant=backbone_variant,
    )


# ============================================================
# 8. Tests
# ============================================================

def _run_single_test(
    name: str,
    **kwargs,
) -> None:
    model = (
        build_spikeyolo_style_ilif_heatmap_experiment(
            **kwargs,
        )
    )

    x = torch.randn(
        1,
        3,
        224,
        224,
    )

    y = model(x)

    expected_shape = (
        1,
        kwargs.get(
            "num_joints",
            25,
        ),
        56,
        56,
    )

    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"{name}: incorrect output shape "
            f"{tuple(y.shape)}; expected "
            f"{expected_shape}."
        )

    loss = y.square().mean()
    loss.backward()

    print(
        f"{name:<18} "
        f"output={tuple(y.shape)} "
        "forward/backward=passed"
    )


def _test_model() -> None:
    print(
        "Testing experimental Model 21 variants"
    )
    print("-" * 72)

    experiments = [
        (
            "M21-B0",
            {},
        ),
        (
            "M21-S1",
            {
                "fusion_type": (
                    "stage2_only"
                ),
            },
        ),
        (
            "M21-S2",
            {
                "fusion_type": (
                    "stage3_only"
                ),
            },
        ),
        (
            "M21-N1",
            {
                "max_spikes": 1,
            },
        ),
        (
            "M21-D2",
            {
                "max_spikes": 2,
            },
        ),
        (
            "M21-D8",
            {
                "max_spikes": 8,
            },
        ),
        (
            "M21-T1",
            {
                "num_steps": 1,
            },
        ),
        (
            "M21-T4",
            {
                "num_steps": 4,
            },
        ),
        (
            "M21-R1",
            {
                "readout_type": "sum",
            },
        ),
        (
            "M21-R2",
            {
                "readout_type": "last",
            },
        ),
        (
            "M21-C1",
            {
                "decoder_type": (
                    "no_refine"
                ),
            },
        ),
        (
            "M21-C2",
            {
                "decoder_type": (
                    "bilinear"
                ),
            },
        ),
        (
            "M21-F1",
            {
                "backbone_variant": (
                    "shallow"
                ),
            },
        ),
    ]

    for (
        name,
        configuration,
    ) in experiments:
        _run_single_test(
            name,
            **configuration,
        )

    baseline = (
        build_spikeyolo_style_ilif_heatmap_experiment()
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in baseline.parameters()
    )

    print("-" * 72)
    print(
        "Baseline parameters: "
        f"{total_parameters:,}"
    )

    print(
        "All experimental variants passed."
    )


if __name__ == "__main__":
    _test_model()
