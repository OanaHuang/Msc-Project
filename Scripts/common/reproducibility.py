# Scripts/common/reproducibility.py

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


# ============================================================
# 1. Random seed
# ============================================================

def seed_everything(
    seed: int = 42,
    deterministic: bool = False,
) -> None:
    """
    Seed Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Random seed.
    deterministic:
        Enable deterministic PyTorch algorithms where possible.

        This may reduce performance and may raise errors for some
        operations without deterministic implementations.
    """
    if seed < 0:
        raise ValueError(
            "seed must be non-negative"
        )

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        try:
            torch.use_deterministic_algorithms(
                True
            )
        except RuntimeError as exc:
            print(
                "Warning: deterministic algorithms "
                f"could not be fully enabled: {exc}"
            )
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


# ============================================================
# 2. Device selection
# ============================================================

def get_device(
    preferred: Optional[str] = None,
    verbose: bool = True,
) -> torch.device:
    """
    Select CUDA, Apple MPS, or CPU.

    Parameters
    ----------
    preferred:
        Examples:
        - "cuda"
        - "cuda:0"
        - "cuda:1"
        - "mps"
        - "cpu"

        When omitted, the best available device is selected.
    """
    if preferred is not None:
        preferred = preferred.lower().strip()

        if preferred.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is not available"
                )

            device = torch.device(preferred)

            if device.index is not None:
                gpu_count = torch.cuda.device_count()

                if device.index >= gpu_count:
                    raise ValueError(
                        f"Requested CUDA device {device.index}, "
                        f"but only {gpu_count} CUDA device(s) "
                        "are available"
                    )

        elif preferred == "mps":
            if not (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                raise RuntimeError(
                    "MPS was requested but is not available"
                )

            device = torch.device("mps")

        elif preferred == "cpu":
            device = torch.device("cpu")

        else:
            raise ValueError(
                f"Unsupported preferred device: {preferred}"
            )

    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")

        else:
            device = torch.device("cpu")

    if verbose:
        print_device_information(device)

    return device


# ============================================================
# 3. Device information
# ============================================================

def print_device_information(
    device: torch.device,
) -> None:
    """
    Print information about the selected compute device.
    """
    print("=" * 70)
    print(f"Selected device: {device}")

    if device.type == "cuda":
        device_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )

        properties = torch.cuda.get_device_properties(
            device_index
        )

        total_memory_gb = (
            properties.total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU name: "
            f"{torch.cuda.get_device_name(device_index)}"
        )

        print(
            f"CUDA device index: {device_index}"
        )

        print(
            f"GPU memory: {total_memory_gb:.2f} GB"
        )

        print(
            f"CUDA version: {torch.version.cuda}"
        )

    elif device.type == "mps":
        print(
            "Apple Metal Performance Shaders is enabled."
        )

    else:
        print(
            "Training will run on CPU."
        )

    print("=" * 70)


# ============================================================
# 4. DataLoader worker seed
# ============================================================

def seed_worker(
    worker_id: int,
) -> None:
    """
    Seed a PyTorch DataLoader worker.

    Usage:

        DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            generator=create_torch_generator(42),
        )
    """
    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_torch_generator(
    seed: int = 42,
) -> torch.Generator:
    """
    Create a seeded PyTorch generator for DataLoader.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


# ============================================================
# 5. CUDA memory information
# ============================================================

def print_cuda_memory(
    device: Optional[torch.device] = None,
) -> None:
    """
    Print CUDA memory allocation information.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    if device is None:
        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
        )

    if device.type != "cuda":
        print(
            f"Device {device} is not a CUDA device."
        )
        return

    allocated = (
        torch.cuda.memory_allocated(device)
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved(device)
        / (1024 ** 3)
    )

    max_allocated = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 3)
    )

    print("=" * 70)
    print(f"CUDA device: {device}")
    print(f"Allocated:    {allocated:.3f} GB")
    print(f"Reserved:     {reserved:.3f} GB")
    print(f"Max allocated:{max_allocated:.3f} GB")
    print("=" * 70)


if __name__ == "__main__":
    seed_everything(
        seed=42,
        deterministic=False,
    )

    selected_device = get_device()