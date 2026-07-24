# Scripts/NTU_RGBD/evaluation/npz_loader.py

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metric_config import EvaluationConfig


# ============================================================
# 1. Scalar helper
# ============================================================

def read_scalar(
    data: np.lib.npyio.NpzFile,
    key: str,
    default: str,
) -> str:
    """
    Read a scalar-like value from an NPZ file.

    If the key does not exist, return the supplied default.
    """
    if key not in data.files:
        return default

    value = np.asarray(
        data[key]
    )

    if value.size == 1:
        return str(
            value.item()
        )

    return str(value)


# ============================================================
# 2. Shape validation
# ============================================================

def validate_prediction_shapes(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    visibility: np.ndarray,
    predicted_visibility: np.ndarray | None,
    confidences: np.ndarray | None,
    npz_path: Path,
    config: EvaluationConfig,
) -> None:
    """
    Validate prediction arrays loaded from one NPZ file.
    """
    if predictions.ndim != 3:
        raise ValueError(
            f"{npz_path.name}: predictions 应为 "
            f"[T, J, 2]，实际为 "
            f"{predictions.shape}"
        )

    if ground_truth.ndim != 3:
        raise ValueError(
            f"{npz_path.name}: ground_truth 应为 "
            f"[T, J, 2]，实际为 "
            f"{ground_truth.shape}"
        )

    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"{npz_path.name}: predictions 和 "
            f"ground_truth shape 不一致："
            f"{predictions.shape} vs "
            f"{ground_truth.shape}"
        )

    if predictions.shape[-1] != 2:
        raise ValueError(
            f"{npz_path.name}: predictions 最后一维 "
            "必须是 x、y 两个坐标"
        )

    num_frames = predictions.shape[0]
    num_joints = predictions.shape[1]

    if num_frames <= 0:
        raise ValueError(
            f"{npz_path.name}: 没有可评估的帧"
        )

    if num_joints <= 0:
        raise ValueError(
            f"{npz_path.name}: 没有可评估的关节点"
        )

    maximum_required_index = max(
        config.head_index,
        config.neck_index,
    )

    if num_joints <= maximum_required_index:
        raise ValueError(
            f"{npz_path.name}: 关节点数量不足，"
            f"当前为 {num_joints}，无法访问 "
            f"Head={config.head_index} 和 "
            f"Neck={config.neck_index}"
        )

    expected_visibility_shape = (
        num_frames,
        num_joints,
    )

    if visibility.shape != expected_visibility_shape:
        raise ValueError(
            f"{npz_path.name}: "
            f"ground_truth_visibility shape 为 "
            f"{visibility.shape}，预期为 "
            f"{expected_visibility_shape}"
        )

    if (
        predicted_visibility is not None
        and predicted_visibility.shape
        != expected_visibility_shape
    ):
        raise ValueError(
            f"{npz_path.name}: "
            f"predicted_visibility shape 为 "
            f"{predicted_visibility.shape}，预期为 "
            f"{expected_visibility_shape}"
        )

    if (
        confidences is not None
        and confidences.shape
        != expected_visibility_shape
    ):
        raise ValueError(
            f"{npz_path.name}: confidences shape 为 "
            f"{confidences.shape}，预期为 "
            f"{expected_visibility_shape}"
        )


# ============================================================
# 3. One-file loader
# ============================================================

def load_prediction_file(
    npz_path: Path,
    config: EvaluationConfig,
) -> dict[str, object]:
    """
    Load one prediction NPZ file.

    Required NPZ keys
    -----------------
    predictions:
        Shape [T, J, 2].

    ground_truth:
        Shape [T, J, 2].

    ground_truth_visibility:
        Shape [T, J].

    Optional NPZ keys
    -----------------
    predicted_visibility:
        Shape [T, J].

    confidences:
        Shape [T, J].

    sample_id:
        Scalar or scalar-like value.

    model_version:
        Scalar or scalar-like value.
    """
    npz_path = Path(
        npz_path
    )

    if not npz_path.exists():
        raise FileNotFoundError(
            f"NPZ 文件不存在：{npz_path}"
        )

    if not npz_path.is_file():
        raise ValueError(
            f"NPZ 路径不是文件：{npz_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:
        required_keys = (
            "predictions",
            "ground_truth",
            "ground_truth_visibility",
        )

        missing_keys = [
            key
            for key in required_keys
            if key not in data.files
        ]

        if missing_keys:
            raise KeyError(
                f"{npz_path.name} 缺少键："
                f"{missing_keys}\n"
                f"当前可用键：{data.files}"
            )

        predictions = np.asarray(
            data["predictions"],
            dtype=np.float32,
        )

        ground_truth = np.asarray(
            data["ground_truth"],
            dtype=np.float32,
        )

        visibility = np.asarray(
            data["ground_truth_visibility"],
            dtype=np.float32,
        )

        predicted_visibility: (
            np.ndarray | None
        ) = None

        if "predicted_visibility" in data.files:
            predicted_visibility = np.asarray(
                data["predicted_visibility"],
                dtype=np.float32,
            )

        confidences: (
            np.ndarray | None
        ) = None

        if "confidences" in data.files:
            confidences = np.asarray(
                data["confidences"],
                dtype=np.float32,
            )

        sample_id = read_scalar(
            data=data,
            key="sample_id",
            default=npz_path.stem,
        )

        model_version = read_scalar(
            data=data,
            key="model_version",
            default=config.model_version,
        )

    validate_prediction_shapes(
        predictions=predictions,
        ground_truth=ground_truth,
        visibility=visibility,
        predicted_visibility=(
            predicted_visibility
        ),
        confidences=confidences,
        npz_path=npz_path,
        config=config,
    )

    return {
        "sample_id": sample_id,
        "model_version": model_version,
        "npz_path": npz_path,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "visibility": visibility,
        "predicted_visibility": (
            predicted_visibility
        ),
        "confidences": confidences,
    }


# ============================================================
# 4. File discovery
# ============================================================

def find_prediction_files(
    config: EvaluationConfig,
) -> list[Path]:
    """
    Find prediction NPZ files using the configured filename pattern.

    If no file matches the preferred pattern, fall back to all
    NPZ files in the directory.
    """
    if not config.npz_dir.exists():
        raise FileNotFoundError(
            "NPZ 文件夹不存在：\n"
            f"{config.npz_dir}"
        )

    if not config.npz_dir.is_dir():
        raise ValueError(
            "NPZ 路径不是文件夹：\n"
            f"{config.npz_dir}"
        )

    npz_paths = sorted(
        config.npz_dir.glob(
            config.filename_pattern
        )
    )

    if not npz_paths:
        npz_paths = sorted(
            config.npz_dir.glob(
                "*.npz"
            )
        )

    if not npz_paths:
        raise FileNotFoundError(
            "文件夹中没有找到 NPZ 文件：\n"
            f"{config.npz_dir}"
        )

    return npz_paths