# Scripts/NTU_RGBD/training/losses.py

from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapMSELoss(nn.Module):
    """
    Visibility-aware mean squared error for pose heatmaps.

    prediction:
        [B, J, H, W]

    target:
        [B, J, H, W]

    visibility:
        [B, J]
        1 means the joint contributes to the loss.
    """

    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {
            "mean",
            "sum",
        }:
            raise ValueError(
                "reduction must be 'mean' or 'sum'"
            )

        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visibility: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} "
                f"does not match target shape {target.shape}"
            )

        if prediction.ndim != 4:
            raise ValueError(
                "prediction and target must have shape [B, J, H, W]"
            )

        squared_error = (
            prediction - target
        ) ** 2

        if visibility is None:
            if self.reduction == "sum":
                return squared_error.sum()

            return squared_error.mean()

        if visibility.ndim != 2:
            raise ValueError(
                "visibility must have shape [B, J]"
            )

        if (
            visibility.shape[0]
            != prediction.shape[0]
            or visibility.shape[1]
            != prediction.shape[1]
        ):
            raise ValueError(
                f"Visibility shape {visibility.shape} "
                f"does not match batch/joint dimensions "
                f"{prediction.shape[:2]}"
            )

        mask = visibility[
            :,
            :,
            None,
            None,
        ].to(
            device=prediction.device,
            dtype=prediction.dtype,
        )

        masked_error = (
            squared_error * mask
        )

        if self.reduction == "sum":
            return masked_error.sum()

        heatmap_pixels = (
            prediction.shape[-2]
            * prediction.shape[-1]
        )

        denominator = (
            mask.sum()
            * heatmap_pixels
        ).clamp_min(1.0)

        return (
            masked_error.sum()
            / denominator
        )


def temporal_velocity_loss(
    pose_sequence: torch.Tensor,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Penalize large frame-to-frame coordinate changes.

    pose_sequence:
        [B, T, J, C]

    visibility:
        Optional [B, T, J]
    """
    if pose_sequence.ndim != 4:
        raise ValueError(
            "pose_sequence must have shape [B, T, J, C]"
        )

    if pose_sequence.shape[1] < 2:
        return pose_sequence.new_tensor(0.0)

    velocity = (
        pose_sequence[:, 1:]
        - pose_sequence[:, :-1]
    )

    squared_velocity = (
        velocity ** 2
    ).sum(dim=-1)

    if visibility is None:
        return squared_velocity.mean()

    if visibility.ndim != 3:
        raise ValueError(
            "visibility must have shape [B, T, J]"
        )

    velocity_mask = (
        visibility[:, 1:]
        * visibility[:, :-1]
    ).to(
        dtype=squared_velocity.dtype
    )

    denominator = (
        velocity_mask.sum()
        .clamp_min(1.0)
    )

    return (
        squared_velocity
        * velocity_mask
    ).sum() / denominator


def temporal_acceleration_loss(
    pose_sequence: torch.Tensor,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Penalize large second-order temporal changes.

    pose_sequence:
        [B, T, J, C]

    visibility:
        Optional [B, T, J]
    """
    if pose_sequence.ndim != 4:
        raise ValueError(
            "pose_sequence must have shape [B, T, J, C]"
        )

    if pose_sequence.shape[1] < 3:
        return pose_sequence.new_tensor(0.0)

    acceleration = (
        pose_sequence[:, 2:]
        - 2.0 * pose_sequence[:, 1:-1]
        + pose_sequence[:, :-2]
    )

    squared_acceleration = (
        acceleration ** 2
    ).sum(dim=-1)

    if visibility is None:
        return squared_acceleration.mean()

    if visibility.ndim != 3:
        raise ValueError(
            "visibility must have shape [B, T, J]"
        )

    acceleration_mask = (
        visibility[:, 2:]
        * visibility[:, 1:-1]
        * visibility[:, :-2]
    ).to(
        dtype=squared_acceleration.dtype
    )

    denominator = (
        acceleration_mask.sum()
        .clamp_min(1.0)
    )

    return (
        squared_acceleration
        * acceleration_mask
    ).sum() / denominator


if __name__ == "__main__":
    criterion = HeatmapMSELoss()

    prediction = torch.randn(
        2,
        25,
        56,
        56,
    )

    target = torch.randn(
        2,
        25,
        56,
        56,
    )

    visibility = torch.ones(
        2,
        25,
    )

    loss = criterion(
        prediction,
        target,
        visibility,
    )

    print(f"Test heatmap loss: {loss.item():.6f}")
    print("losses.py test passed.")