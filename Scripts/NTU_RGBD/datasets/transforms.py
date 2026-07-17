from __future__ import annotations

import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


class PoseTransform:
    def __init__(
        self,
        image_size: int = 224,
        training: bool = False,
    ) -> None:
        self.image_size = image_size
        self.training = training

    def __call__(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        visibility: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        keypoints = np.asarray(
            keypoints,
            dtype=np.float32,
        ).copy()

        visibility = np.asarray(
            visibility,
            dtype=np.float32,
        ).copy()

        original_height, original_width = image.shape[:2]

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        keypoints[:, 0] *= (
            self.image_size / original_width
        )

        keypoints[:, 1] *= (
            self.image_size / original_height
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image = (
            image - IMAGENET_MEAN
        ) / IMAGENET_STD

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        return {
            "image": torch.from_numpy(
                image
            ).float(),

            "keypoints": torch.from_numpy(
                keypoints
            ).float(),

            "visibility": torch.from_numpy(
                visibility
            ).float(),
        }


def build_train_transform(
    image_size: int = 224,
) -> PoseTransform:
    return PoseTransform(
        image_size=image_size,
        training=True,
    )


def build_eval_transform(
    image_size: int = 224,
) -> PoseTransform:
    return PoseTransform(
        image_size=image_size,
        training=False,
    )