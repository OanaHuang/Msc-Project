# Scripts/NTU_RGBD/models/ms_spiking_resnet50_heatmap.py

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)

import snntorch as snn
from snntorch import surrogate
from snntorch import utils as snn_utils


# ============================================================
# 1. MS-Residual Spiking Bottleneck
# ============================================================

class MSResidualSpikingBottleneck(nn.Module):
    """
    MS-Residual spiking bottleneck for ResNet50.

    Main branch:
        LIF
        -> Conv1x1 + BN
        -> LIF
        -> Conv3x3 + BN
        -> LIF
        -> Conv1x1 + BN
        -> Residual Add

    Shortcut:
        Identity, or
        Conv1x1 + BN when channel/spatial size changes.

    The residual output is treated as membrane-potential
    information. The next block begins with an LIF neuron,
    ensuring that each convolution receives spike-form input.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive."
            )

        if planes <= 0:
            raise ValueError(
                "planes must be positive."
            )

        if stride not in (1, 2):
            raise ValueError(
                "stride must be 1 or 2."
            )

        spike_gradient = surrogate.fast_sigmoid(
            slope=surrogate_slope,
        )

        # ----------------------------------------------------
        # Pre-activation spiking neurons
        # ----------------------------------------------------

        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_gradient,
            init_hidden=True,
            output=False,
        )

        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_gradient,
            init_hidden=True,
            output=False,
        )

        self.lif3 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_gradient,
            init_hidden=True,
            output=False,
        )

        # ----------------------------------------------------
        # ResNet50 bottleneck convolutions
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            num_features=planes,
        )

        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            num_features=planes,
        )

        self.conv3 = nn.Conv2d(
            in_channels=planes,
            out_channels=(
                planes * self.expansion
            ),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(
            num_features=(
                planes * self.expansion
            ),
        )

        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Membrane-form feature tensor.

        Returns:
            Residual output in membrane form.
        """

        identity = x

        # ----------------------------------------------------
        # First pre-activation
        # ----------------------------------------------------

        spike1 = self.lif1(
            x,
        )

        # The downsampling shortcut also receives spike input.
        if self.downsample is not None:
            identity = self.downsample(
                spike1,
            )

        out = self.conv1(
            spike1,
        )

        out = self.bn1(
            out,
        )

        # ----------------------------------------------------
        # Second pre-activation
        # ----------------------------------------------------

        spike2 = self.lif2(
            out,
        )

        out = self.conv2(
            spike2,
        )

        out = self.bn2(
            out,
        )

        # ----------------------------------------------------
        # Third pre-activation
        # ----------------------------------------------------

        spike3 = self.lif3(
            out,
        )

        out = self.conv3(
            spike3,
        )

        out = self.bn3(
            out,
        )

        # ----------------------------------------------------
        # MS residual addition
        # ----------------------------------------------------
        #
        # The output is not required to be binary.
        # It is treated as membrane potential.
        #
        # The next residual block starts with LIF, so the next
        # convolution still receives binary spike input.
        # ----------------------------------------------------

        out = out + identity

        return out


# ============================================================
# 2. MS-Spiking ResNet50 Heatmap Model
# ============================================================

class MSSpikingResNet50Heatmap(nn.Module):
    """
    MS-Residual Spiking ResNet50 for 2D pose heatmaps.

    Current dataset input:
        B x 3 x 224 x 224

    Current temporal implementation:
        The same frame is processed for multiple SNN time
        steps while the LIF membrane states are retained.

    Backbone output:
        B x 2048 x 7 x 7

    Heatmap output:
        B x num_joints x 56 x 56

    Notes:
        The backbone is spiking, while the heatmap decoder
        remains an ANN decoder. Therefore, this is a hybrid
        SNN-ANN pose-estimation model.
    """

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
            raise ValueError(
                "num_joints must be positive."
            )

        if num_steps <= 0:
            raise ValueError(
                "num_steps must be positive."
            )

        if not 0.0 <= beta <= 1.0:
            raise ValueError(
                "beta must be between 0 and 1."
            )

        if threshold <= 0.0:
            raise ValueError(
                "threshold must be positive."
            )

        if surrogate_slope <= 0.0:
            raise ValueError(
                "surrogate_slope must be positive."
            )

        self.num_joints = num_joints
        self.num_steps = num_steps

        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope

        self.in_channels = 64

        spike_gradient = surrogate.fast_sigmoid(
            slope=surrogate_slope,
        )

        # ====================================================
        # 2.1 RGB spike encoder / ResNet stem
        # ====================================================
        #
        # The first convolution processes ordinary RGB input.
        # The subsequent LIF neuron converts the feature map
        # into spike-form input for the residual backbone.
        # ====================================================

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            num_features=64,
        )

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

        # ====================================================
        # 2.2 MS-Residual Spiking ResNet50 backbone
        # ====================================================
        #
        # ResNet50 stage configuration:
        #
        # Stage 1: 3 bottlenecks, output 256 channels
        # Stage 2: 4 bottlenecks, output 512 channels
        # Stage 3: 6 bottlenecks, output 1024 channels
        # Stage 4: 3 bottlenecks, output 2048 channels
        # ====================================================

        self.layer1 = self._make_layer(
            planes=64,
            blocks=3,
            stride=1,
        )

        self.layer2 = self._make_layer(
            planes=128,
            blocks=4,
            stride=2,
        )

        self.layer3 = self._make_layer(
            planes=256,
            blocks=6,
            stride=2,
        )

        self.layer4 = self._make_layer(
            planes=512,
            blocks=3,
            stride=2,
        )

        # Final spike readout from the backbone.
        self.output_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_gradient,
            init_hidden=True,
            output=False,
        )

        # ====================================================
        # 2.3 Heatmap decoder
        # ====================================================
        #
        # Input:
        #     B x 2048 x 7 x 7
        #
        # Output:
        #     B x num_joints x 56 x 56
        #
        # Spatial progression:
        #     7 -> 14 -> 28 -> 56
        # ====================================================

        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "deconv1",
                        nn.ConvTranspose2d(
                            in_channels=2048,
                            out_channels=256,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            output_padding=0,
                            bias=False,
                        ),
                    ),
                    (
                        "bn1",
                        nn.BatchNorm2d(
                            num_features=256,
                        ),
                    ),
                    (
                        "relu1",
                        nn.ReLU(
                            inplace=True,
                        ),
                    ),
                    (
                        "deconv2",
                        nn.ConvTranspose2d(
                            in_channels=256,
                            out_channels=256,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            output_padding=0,
                            bias=False,
                        ),
                    ),
                    (
                        "bn2",
                        nn.BatchNorm2d(
                            num_features=256,
                        ),
                    ),
                    (
                        "relu2",
                        nn.ReLU(
                            inplace=True,
                        ),
                    ),
                    (
                        "deconv3",
                        nn.ConvTranspose2d(
                            in_channels=256,
                            out_channels=256,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            output_padding=0,
                            bias=False,
                        ),
                    ),
                    (
                        "bn3",
                        nn.BatchNorm2d(
                            num_features=256,
                        ),
                    ),
                    (
                        "relu3",
                        nn.ReLU(
                            inplace=True,
                        ),
                    ),
                    (
                        "final_conv",
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=num_joints,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                ]
            )
        )

        # Initialize the complete model before optionally
        # replacing compatible backbone parameters with
        # ImageNet-pretrained ResNet50 weights.
        self._initialize_weights()

        if pretrained:
            self._load_pretrained_resnet50()

    # ========================================================
    # 3. Backbone construction
    # ========================================================

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        if blocks <= 0:
            raise ValueError(
                "blocks must be positive."
            )

        output_channels = (
            planes
            * MSResidualSpikingBottleneck.expansion
        )

        downsample: nn.Module | None = None

        if (
            stride != 1
            or self.in_channels != output_channels
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=output_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_features=output_channels,
                ),
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

        return nn.Sequential(
            *layers,
        )

    # ========================================================
    # 4. Weight initialization
    # ========================================================

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

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias,
                    )

            elif isinstance(
                module,
                nn.ConvTranspose2d,
            ):
                nn.init.normal_(
                    module.weight,
                    std=0.001,
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias,
                    )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.ones_(
                    module.weight,
                )

                nn.init.zeros_(
                    module.bias,
                )

    # ========================================================
    # 5. ImageNet pretrained weights
    # ========================================================

    def _load_pretrained_resnet50(
        self,
    ) -> None:
        """
        Load compatible convolution and BatchNorm tensors from
        torchvision's ImageNet-pretrained ResNet50.

        Compatible backbone names include:
            conv1
            bn1
            layer1.*.conv*
            layer1.*.bn*
            ...
            layer4.*.conv*
            layer4.*.bn*
            layer*.downsample.*

        LIF states and the pose decoder remain initialized by
        this model.
        """

        pretrained_model = resnet50(
            weights=(
                ResNet50_Weights.IMAGENET1K_V2
            ),
        )

        pretrained_state = (
            pretrained_model.state_dict()
        )

        current_state = self.state_dict()

        compatible_state: dict[
            str,
            torch.Tensor,
        ] = {}

        skipped_shape_keys: list[str] = []

        for key, pretrained_value in (
            pretrained_state.items()
        ):
            if key not in current_state:
                continue

            current_value = current_state[key]

            if (
                current_value.shape
                != pretrained_value.shape
            ):
                skipped_shape_keys.append(
                    key,
                )
                continue

            compatible_state[key] = (
                pretrained_value
            )

        load_result = self.load_state_dict(
            compatible_state,
            strict=False,
        )

        print()
        print(
            "Loaded compatible ImageNet "
            "ResNet50 weights"
        )
        print("-" * 70)

        print(
            f"Loaded tensors:          "
            f"{len(compatible_state)}"
        )

        print(
            f"Shape-mismatch tensors:  "
            f"{len(skipped_shape_keys)}"
        )

        print(
            f"Missing model tensors:   "
            f"{len(load_result.missing_keys)}"
        )

        print(
            f"Unexpected tensors:      "
            f"{len(load_result.unexpected_keys)}"
        )

    # ========================================================
    # 6. Hidden-state control
    # ========================================================

    def reset_hidden_states(
        self,
    ) -> None:
        """
        Reset the hidden membrane states of all snnTorch
        neurons.

        This must be called before processing an independent
        image batch or independent sequence.
        """

        snn_utils.reset(
            self,
        )

    # ========================================================
    # 7. Backbone forward
    # ========================================================

    def forward_backbone_step(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process one SNN simulation step.

        The LIF states are not reset inside this method.
        They persist across repeated calls within one forward
        pass.

        Args:
            x:
                B x 3 x H x W RGB tensor.

        Returns:
            B x 2048 x 7 x 7 spike feature tensor when the
            input resolution is 224 x 224.
        """

        # First convolution operates on non-spike RGB data.
        x = self.conv1(
            x,
        )

        x = self.bn1(
            x,
        )

        # Convert the stem feature to spikes.
        x = self.stem_lif(
            x,
        )

        x = self.maxpool(
            x,
        )

        # MS-Residual spiking stages.
        x = self.layer1(
            x,
        )

        x = self.layer2(
            x,
        )

        x = self.layer3(
            x,
        )

        x = self.layer4(
            x,
        )

        # Convert the final membrane-form backbone feature
        # into spikes for rate-based temporal readout.
        x = self.output_lif(
            x,
        )

        return x

    # ========================================================
    # 8. Full forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                B x 3 x H x W input tensor.

        Returns:
            B x num_joints x 56 x 56 heatmaps for a
            224 x 224 input.
        """

        if x.ndim != 4:
            raise ValueError(
                "Expected input shape B x 3 x H x W, "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[1] != 3:
            raise ValueError(
                "Expected three RGB channels, "
                f"but received {x.shape[1]} channels."
            )

        # Each batch currently represents independent frames.
        # Therefore, hidden states must not leak from the
        # previous batch.
        self.reset_hidden_states()

        step_features: list[torch.Tensor] = []

        # Current frame-based SNN implementation:
        #
        # The same frame is presented over NUM_STEPS.
        # Membrane states persist across these steps.
        for _ in range(self.num_steps):
            features_t = self.forward_backbone_step(
                x,
            )

            step_features.append(
                features_t,
            )

        # Stack:
        #     T x B x 2048 x 7 x 7
        stacked_features = torch.stack(
            step_features,
            dim=0,
        )

        # Spike-rate readout:
        #     B x 2048 x 7 x 7
        readout_features = (
            stacked_features.mean(
                dim=0,
            )
        )

        heatmaps = self.decoder(
            readout_features,
        )

        return heatmaps


# ============================================================
# 9. Builder function
# ============================================================

def build_ms_spiking_resnet50_heatmap(
    num_joints: int = 25,
    num_steps: int = 2,
    beta: float = 0.90,
    threshold: float = 1.0,
    surrogate_slope: float = 25.0,
    pretrained: bool = True,
) -> MSSpikingResNet50Heatmap:
    """
    Build an MS-Residual Spiking ResNet50 heatmap model.
    """

    return MSSpikingResNet50Heatmap(
        num_joints=num_joints,
        num_steps=num_steps,
        beta=beta,
        threshold=threshold,
        surrogate_slope=surrogate_slope,
        pretrained=pretrained,
    )


# ============================================================
# 10. Standalone model test
# ============================================================

def _test_model() -> None:
    """
    Run this file directly to verify the model output shape:

        python -m Scripts.NTU_RGBD.models.ms_spiking_resnet50_heatmap
    """

    model = build_ms_spiking_resnet50_heatmap(
        num_joints=25,
        num_steps=2,
        beta=0.90,
        threshold=1.0,
        surrogate_slope=25.0,
        pretrained=False,
    )

    model.eval()

    test_input = torch.randn(
        1,
        3,
        224,
        224,
    )

    with torch.no_grad():
        test_output = model(
            test_input,
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

    expected_shape = (
        1,
        25,
        56,
        56,
    )

    print()
    print("=" * 70)
    print("MS-Spiking ResNet50 Heatmap model test")
    print("=" * 70)

    print(
        f"Input shape:          "
        f"{tuple(test_input.shape)}"
    )

    print(
        f"Output shape:         "
        f"{tuple(test_output.shape)}"
    )

    print(
        f"Expected shape:       "
        f"{expected_shape}"
    )

    print(
        f"Total parameters:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    if tuple(test_output.shape) != expected_shape:
        raise RuntimeError(
            "Model output shape is incorrect."
        )

    print()
    print("Model test passed.")


if __name__ == "__main__":
    _test_model()