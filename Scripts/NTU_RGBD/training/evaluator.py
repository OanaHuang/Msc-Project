# Scripts/NTU_RGBD/training/evaluator.py

from __future__ import annotations

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
        Shape [B, J, H, W].

    image_size:
        int:
            Square image size.

        tuple:
            (width, height).

    Returns
    -------
    torch.Tensor
        Shape [B, J, 2] in image-pixel coordinates.
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            "heatmaps must have shape [B, J, H, W]"
        )

    (
        batch_size,
        num_joints,
        heatmap_height,
        heatmap_width,
    ) = heatmaps.shape

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
        image_width, image_height = (
            image_size
        )

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
# 2. Validation helpers
# ============================================================

def _validate_pose_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> None:
    """
    Validate pose coordinate and visibility tensors.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes must match, "
            f"but received {prediction.shape} "
            f"and {target.shape}"
        )

    if prediction.ndim != 3:
        raise ValueError(
            "prediction must have shape [B, J, 2]"
        )

    if prediction.shape[-1] != 2:
        raise ValueError(
            "prediction and target must contain 2D coordinates"
        )

    if visibility.shape != prediction.shape[:2]:
        raise ValueError(
            "visibility must have shape [B, J], "
            f"but received {visibility.shape}"
        )


def _broadcast_normalization_length(
    normalization_length: torch.Tensor | float,
    distances: torch.Tensor,
) -> torch.Tensor:
    """
    Convert and broadcast a normalization value to shape [B, J].

    Supported input shapes:
        scalar
        [B]
        [B, 1]
        [B, J]
    """
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
        if (
            normalization_length.shape[0]
            != distances.shape[0]
        ):
            raise ValueError(
                "One-dimensional normalization_length "
                "must contain one value per sample"
            )

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
            f"from shape {normalization_length.shape} "
            f"to distance shape {distances.shape}"
        ) from error

    return normalization_length


def _compute_valid_mask(
    distances: torch.Tensor,
    visibility: torch.Tensor,
    normalization_length: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build a mask for valid joints.
    """
    valid = visibility > 0

    valid &= torch.isfinite(
        distances
    )

    if normalization_length is not None:
        valid &= torch.isfinite(
            normalization_length
        )

        valid &= (
            normalization_length > 1e-6
        )

    return valid


# ============================================================
# 3. Pixel error
# ============================================================

def compute_batch_pixel_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> float:
    """
    Compute mean Euclidean keypoint error in pixels.
    """
    _validate_pose_inputs(
        prediction=prediction,
        target=target,
        visibility=visibility,
    )

    distances = torch.linalg.norm(
        prediction - target,
        dim=-1,
    )

    valid = _compute_valid_mask(
        distances=distances,
        visibility=visibility,
    )

    if not torch.any(valid):
        return float("nan")

    return float(
        distances[valid]
        .mean()
        .item()
    )


# ============================================================
# 4. Generic PCK
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

    This is a generic function. The meaning of PCK depends on the
    supplied normalization length.
    """
    if threshold <= 0:
        raise ValueError(
            "threshold must be positive"
        )

    _validate_pose_inputs(
        prediction=prediction,
        target=target,
        visibility=visibility,
    )

    distances = torch.linalg.norm(
        prediction - target,
        dim=-1,
    )

    normalization_length = (
        _broadcast_normalization_length(
            normalization_length=(
                normalization_length
            ),
            distances=distances,
        )
    )

    valid = _compute_valid_mask(
        distances=distances,
        visibility=visibility,
        normalization_length=(
            normalization_length
        ),
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
    Compute generic PCK separately for every joint.

    Returns
    -------
    torch.Tensor
        Shape [J], containing percentages.
    """
    _validate_pose_inputs(
        prediction=prediction,
        target=target,
        visibility=visibility,
    )

    num_joints = prediction.shape[1]

    scores = torch.full(
        (num_joints,),
        float("nan"),
        dtype=torch.float32,
        device=prediction.device,
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
# 5. PCKh
# ============================================================

def compute_batch_pckh(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    head_length: torch.Tensor | float,
    threshold: float = 0.5,
) -> float:
    """
    Compute Percentage of Correct Keypoints normalized by head size.

    A prediction is correct when:

        distance <= threshold * head_length

    For PCKh@0.5:
        threshold = 0.5

    Notes
    -----
    In the current NTU pipeline, head_length is the calibrated
    2D Head-to-Neck distance after person cropping and resizing.
    """
    return compute_batch_pck(
        prediction=prediction,
        target=target,
        visibility=visibility,
        normalization_length=head_length,
        threshold=threshold,
    )


def compute_pckh_per_joint(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    head_length: torch.Tensor | float,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute PCKh separately for every joint.

    Returns
    -------
    torch.Tensor
        Shape [J], containing PCKh percentages.
    """
    return compute_pck_per_joint(
        prediction=prediction,
        target=target,
        visibility=visibility,
        normalization_length=head_length,
        threshold=threshold,
    )


# ============================================================
# 6. Legacy torso normalization helper
# ============================================================

def compute_torso_normalization(
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
    left_shoulder_index: int = 4,
    right_hip_index: int = 16,
    fallback_length: float = 224.0,
) -> torch.Tensor:
    """
    Estimate one torso normalization length per sample.

    This function is retained for legacy PCK evaluation. It is not
    used by the PCKh evaluation function.
    """
    if keypoints.ndim != 3:
        raise ValueError(
            "keypoints must have shape [B, J, 2]"
        )

    if visibility.shape != keypoints.shape[:2]:
        raise ValueError(
            "visibility must have shape [B, J]"
        )

    num_joints = keypoints.shape[1]

    if not (
        0 <= left_shoulder_index < num_joints
    ):
        raise IndexError(
            "left_shoulder_index is outside "
            "the joint range"
        )

    if not (
        0 <= right_hip_index < num_joints
    ):
        raise IndexError(
            "right_hip_index is outside "
            "the joint range"
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

    valid &= torch.isfinite(
        distances
    )

    valid &= distances > 1e-6

    normalization[valid] = (
        distances[valid]
    )

    return normalization


# ============================================================
# 7. Full model evaluation with PCKh@0.5
# ============================================================

@torch.no_grad()
def evaluate_heatmap_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    image_size: int = 224,
    pckh_threshold: float = 0.5,
) -> dict[str, object]:
    """
    Evaluate a heatmap model over a DataLoader.

    The returned metrics include:

        mean_pixel_error
        pckh_0.50
        pckh_per_joint
        valid_head_samples
        valid_joints

    PCKh uses the calibrated head_length supplied by
    NTUFrameDataset.
    """
    if pckh_threshold <= 0:
        raise ValueError(
            "pckh_threshold must be positive"
        )

    model.eval()

    total_pixel_error = 0.0
    total_visible_joints = 0

    total_pckh_correct = 0
    total_pckh_joints = 0

    per_joint_correct = None
    per_joint_total = None

    total_samples = 0
    valid_head_samples = 0

    for batch in dataloader:
        images = batch[
            "image"
        ].to(
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

        if "head_length" not in batch:
            raise KeyError(
                "The batch does not contain 'head_length'. "
                "Update NTUFrameDataset before running PCKh "
                "evaluation."
            )

        head_length = batch[
            "head_length"
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

        _validate_pose_inputs(
            prediction=prediction_keypoints,
            target=target_keypoints,
            visibility=visibility,
        )

        distances = torch.linalg.norm(
            prediction_keypoints
            - target_keypoints,
            dim=-1,
        )

        head_length = (
            _broadcast_normalization_length(
                normalization_length=(
                    head_length
                ),
                distances=distances,
            )
        )

        visible_valid = (
            _compute_valid_mask(
                distances=distances,
                visibility=visibility,
            )
        )

        total_pixel_error += float(
            distances[visible_valid]
            .sum()
            .item()
        )

        visible_count = int(
            visible_valid.sum().item()
        )

        total_visible_joints += (
            visible_count
        )

        pckh_valid = (
            _compute_valid_mask(
                distances=distances,
                visibility=visibility,
                normalization_length=(
                    head_length
                ),
            )
        )

        normalized_distance = (
            distances
            / head_length.clamp_min(
                1e-6
            )
        )

        correct = (
            normalized_distance
            <= pckh_threshold
        )

        total_pckh_correct += int(
            correct[pckh_valid]
            .sum()
            .item()
        )

        batch_pckh_total = int(
            pckh_valid.sum().item()
        )

        total_pckh_joints += (
            batch_pckh_total
        )

        batch_size = images.shape[0]

        total_samples += batch_size

        sample_head_valid = (
            torch.isfinite(
                head_length[:, 0]
            )
            & (
                head_length[:, 0]
                > 1e-6
            )
        )

        valid_head_samples += int(
            sample_head_valid.sum().item()
        )

        joint_correct = (
            correct
            & pckh_valid
        ).sum(
            dim=0
        )

        joint_total = (
            pckh_valid.sum(
                dim=0
            )
        )

        if per_joint_correct is None:
            per_joint_correct = (
                joint_correct.to(
                    dtype=torch.long
                )
            )

            per_joint_total = (
                joint_total.to(
                    dtype=torch.long
                )
            )

        else:
            per_joint_correct += (
                joint_correct.to(
                    dtype=torch.long
                )
            )

            per_joint_total += (
                joint_total.to(
                    dtype=torch.long
                )
            )

    mean_pixel_error = (
        total_pixel_error
        / total_visible_joints
        if total_visible_joints > 0
        else float("nan")
    )

    overall_pckh = (
        100.0
        * total_pckh_correct
        / total_pckh_joints
        if total_pckh_joints > 0
        else float("nan")
    )

    if (
        per_joint_correct is None
        or per_joint_total is None
    ):
        per_joint_scores = torch.empty(
            0,
            dtype=torch.float32,
        )

    else:
        per_joint_scores = torch.full(
            per_joint_total.shape,
            float("nan"),
            dtype=torch.float32,
            device=per_joint_total.device,
        )

        valid_joint_counts = (
            per_joint_total > 0
        )

        per_joint_scores[
            valid_joint_counts
        ] = (
            100.0
            * per_joint_correct[
                valid_joint_counts
            ].float()
            / per_joint_total[
                valid_joint_counts
            ].float()
        )

        per_joint_scores = (
            per_joint_scores
            .detach()
            .cpu()
        )

    return {
        "mean_pixel_error": (
            mean_pixel_error
        ),

        f"pckh_{pckh_threshold:.2f}": (
            overall_pckh
        ),

        "pckh_per_joint": (
            per_joint_scores
        ),

        "valid_head_samples": (
            valid_head_samples
        ),

        "total_samples": (
            total_samples
        ),

        "valid_joints": (
            total_pckh_joints
        ),
    }


# ============================================================
# 8. Local test
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

    test_target = decoded.clone()

    test_target[
        0,
        0,
        0,
    ] += 5.0

    test_visibility = torch.ones(
        2,
        25,
    )

    test_head_length = torch.tensor(
        [20.0, 20.0],
        dtype=torch.float32,
    )

    test_pckh = compute_batch_pckh(
        prediction=decoded,
        target=test_target,
        visibility=test_visibility,
        head_length=test_head_length,
        threshold=0.5,
    )

    test_per_joint = (
        compute_pckh_per_joint(
            prediction=decoded,
            target=test_target,
            visibility=test_visibility,
            head_length=test_head_length,
            threshold=0.5,
        )
    )

    print(
        f"Test PCKh@0.5: "
        f"{test_pckh:.2f}%"
    )

    print(
        "Per-joint shape: "
        f"{tuple(test_per_joint.shape)}"
    )

    print(
        "evaluator.py test passed."
    )