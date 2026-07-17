"""
Shared utilities for the MSc human pose estimation project.

This package contains dataset-independent functions for:

- project paths
- video input/output
- pose visualization
- pose evaluation metrics
- experiment reproducibility
"""

from .paths import (
    PROJECT_ROOT,
    DATASETS_DIR,
    OUTPUTS_DIR,
    SERVER_OUTPUTS_DIR,
    PENN_ACTION_ROOT,
    NTU_RGBD_ROOT,
)

from .reproducibility import (
    seed_everything,
    get_device,
)

__all__ = [
    "PROJECT_ROOT",
    "DATASETS_DIR",
    "OUTPUTS_DIR",
    "SERVER_OUTPUTS_DIR",
    "PENN_ACTION_ROOT",
    "NTU_RGBD_ROOT",
    "seed_everything",
    "get_device",
]