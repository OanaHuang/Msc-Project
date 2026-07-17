# Scripts/NTU_RGBD/training/evaluator.py

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


# ============================================================
# 1. Heatmap decoding
# ============================================================

def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    image_size: int | tuple[int, int] = 224,
) -> torch.Tensor:
    """
    Decode heatmaps using argmax.

    Parameters
    ----------
    heatmaps:
        [B, J, H, W]

    image_size:
        int:
            square image size

        tuple:
            (width, height)

    Returns
    -------
    torch.Tensor
        [B, J, 2] in image-pixel coordinates.
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            "heatmaps must have shape [B, J, H, W]"
        )

    batch_size, num_joints, heatmap_height, heatmap_width = (
        heatmaps.shape
    )

    flattened = heatmaps.reshape(
        batch_size,
        num_joints,
        -1,
    )

    indices = torch.argmax(
        flattened,
        dim=-1,
    )

    x = (
        indices % heatmap_width
    ).to(
        dtype=heatmaps.dtype
    )

    y = torch.div(
        indices,
        heatmap_width,
        rounding_mode="floor",
    ).to(
        dtype=heatmaps.dtype
    )

    if isinstance(image_size, int):
        image_width = image_size
        image_height = image_size
    else:
        image_width, image_height = image_size

    x *= (
        image_width
        / heatmap_width
    )

    y *= (
        image_height
        / heatmap_height
    )

    return torch.stack(
        [x, y],
        dim=-1,
    )


# ============================================================
# 2. Pixel error
# ============================================================

def compute_batch_pixel_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> float:
    """
    Mean Euclidean keypoint error in pixels.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes must match"
        )

    if prediction.ndim != 3:
        raise ValueError(
            "prediction must have shape [B, J, 2]"
        )

    if visibility.shape != prediction.shape[:2]:
        raise ValueError(
            "visibility must have shape [B, J]"
        )

    distances = torch.linalg.norm(
        prediction - target,
        dim=-1,
    )

    valid = visibility > 0

    valid &= torch.isfinite(
        distances
    )

    if not torch.any(valid):
        return float("nan")

    return float(
        distances[valid]
        .mean()
        .item()
    )


# ============================================================
# 3. PCK
# ============================================================

def compute_batch_pck(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    normalization_length: torch.Tensor | float,
    threshold: float = 0.1,
) -> float:
    """
    Compute Percentage of Correct Keypoints.

    A prediction is correct when:

        distance <= threshold * normalization_length
    """
    if threshold <= 0:
        raise ValueError(
            "threshold must be positive"
        )

    distances = torch.linalg.norm(
        prediction - target,
        dim=-1,
    )

    normalization_length = torch.as_tensor(
        normalization_length,
        dtype=distances.dtype,
        device=distances.device,
    )

    if normalization_length.ndim == 0:
        normalization_length = (
            normalization_length.expand(
                distances.shape[0]
            )
        )

    if normalization_length.ndim == 1:
        normalization_length = (
            normalization_length[:, None]
        )

    try:
        normalization_length = (
            normalization_length.expand_as(
                distances
            )
        )
    except RuntimeError as error:
        raise ValueError(
            "normalization_length cannot be broadcast "
            "to shape [B, J]"
        ) from error

    valid = visibility > 0

    valid &= torch.isfinite(
        distances
    )

    valid &= (
        normalization_length > 1e-6
    )

    if not torch.any(valid):
        return float("nan")

    normalized_distance = (
        distances
        / normalization_length.clamp_min(
            1e-6
        )
    )

    correct = (
        normalized_distance
        <= threshold
    )

    return float(
        correct[valid]
        .float()
        .mean()
        .item()
        * 100.0
    )


def compute_pck_per_joint(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    normalization_length: torch.Tensor | float,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Return PCK for each joint.

    Output:
        [J]
    """
    num_joints = prediction.shape[1]

    scores = torch.full(
        (num_joints,),
        float("nan"),
        dtype=torch.float32,
    )

    for joint_index in range(
        num_joints
    ):
        score = compute_batch_pck(
            prediction=prediction[
                :,
                joint_index:
                joint_index + 1,
            ],
            target=target[
                :,
                joint_index:
                joint_index + 1,
            ],
            visibility=visibility[
                :,
                joint_index:
                joint_index + 1,
            ],
            normalization_length=(
                normalization_length
            ),
            threshold=threshold,
        )

        scores[joint_index] = score

    return scores


# ============================================================
# 4. Normalization helpers
# ============================================================

def compute_torso_normalization(
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
    left_shoulder_index: int = 4,
    right_hip_index: int = 16,
    fallback_length: float = 224.0,
) -> torch.Tensor:
    """
    Estimate one normalization length per sample.

    Uses the distance between left shoulder and right hip.
    Falls back to fallback_length when either joint is unavailable.
    """
    if keypoints.ndim != 3:
        raise ValueError(
            "keypoints must have shape [B, J, 2]"
        )

    batch_size = keypoints.shape[0]

    normalization = torch.full(
        (batch_size,),
        fallback_length,
        dtype=keypoints.dtype,
        device=keypoints.device,
    )

    point_a = keypoints[
        :,
        left_shoulder_index,
    ]

    point_b = keypoints[
        :,
        right_hip_index,
    ]

    valid = (
        visibility[
            :,
            left_shoulder_index
        ] > 0
    )

    valid &= (
        visibility[
            :,
            right_hip_index
        ] > 0
    )

    valid &= torch.isfinite(
        point_a
    ).all(dim=-1)

    valid &= torch.isfinite(
        point_b
    ).all(dim=-1)

    distances = torch.linalg.norm(
        point_a - point_b,
        dim=-1,
    )

    valid &= distances > 1e-6

    normalization[valid] = (
        distances[valid]
    )

    return normalization


# ============================================================
# 5. Full model evaluation
# ============================================================

@torch.no_grad()
def evaluate_heatmap_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    image_size: int = 224,
    pck_thresholds: Sequence[float] = (
        0.05,
        0.10,
        0.20,
        0.50,
    ),
) -> dict[str, float]:
    """
    Evaluate a heatmap model over a DataLoader.
    """
    model.eval()

    total_pixel_error = 0.0
    total_visible_joints = 0

    pck_correct = {
        threshold: 0
        for threshold in pck_thresholds
    }

    pck_total = {
        threshold: 0
        for threshold in pck_thresholds
    }

    for batch in dataloader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        target_keypoints = batch[
            "keypoints"
        ].to(
            device,
            non_blocking=True,
        )

        visibility = batch[
            "visibility"
        ].to(
            device,
            non_blocking=True,
        )

        heatmaps = model(
            images
        )

        prediction_keypoints = (
            heatmaps_to_keypoints(
                heatmaps,
                image_size=image_size,
            )
        )

        distances = torch.linalg.norm(
            prediction_keypoints
            - target_keypoints,
            dim=-1,
        )

        valid = visibility > 0

        valid &= torch.isfinite(
            distances
        )

        total_pixel_error += float(
            distances[valid]
            .sum()
            .item()
        )

        visible_count = int(
            valid.sum().item()
        )

        total_visible_joints += (
            visible_count
        )

        normalization = (
            compute_torso_normalization(
                target_keypoints,
                visibility,
                fallback_length=float(
                    image_size
                ),
            )
        )[:, None]

        normalized_distance = (
            distances
            / normalization.clamp_min(
                1e-6
            )
        )

        for threshold in pck_thresholds:
            correct = (
                normalized_distance
                <= threshold
            )

            pck_correct[
                threshold
            ] += int(
                correct[valid]
                .sum()
                .item()
            )

            pck_total[
                threshold
            ] += visible_count

    results = {
        "mean_pixel_error": (
            total_pixel_error
            / total_visible_joints
            if total_visible_joints > 0
            else float("nan")
        )
    }

    for threshold in pck_thresholds:
        total = pck_total[
            threshold
        ]

        results[
            f"pck_{threshold:.2f}"
        ] = (
            100.0
            * pck_correct[threshold]
            / total
            if total > 0
            else float("nan")
        )

    return results


# ============================================================
# 6. Local test
# ============================================================

if __name__ == "__main__":
    test_heatmaps = torch.zeros(
        2,
        25,
        56,
        56,
    )

    test_heatmaps[
        :,
        :,
        10,
        20,
    ] = 1.0

    decoded = heatmaps_to_keypoints(
        test_heatmaps,
        image_size=224,
    )

    print(
        f"Decoded shape: "
        f"{tuple(decoded.shape)}"
    )

    print(
        f"First point: "
        f"{decoded[0, 0].tolist()}"
    )

    print("evaluator.py test passed.")