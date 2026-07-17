# Scripts/common/paths.py

from pathlib import Path


# ============================================================
# 1. Project root
# ============================================================

# Current file:
# MSc Project/Scripts/common/paths.py
#
# parents[0] -> common
# parents[1] -> Scripts
# parents[2] -> MSc Project
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 2. Main project directories
# ============================================================

DATASETS_DIR = PROJECT_ROOT / "Datasets"
SCRIPTS_DIR = PROJECT_ROOT / "Scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SERVER_OUTPUTS_DIR = PROJECT_ROOT / "server_outputs"
LITERATURE_DIR = PROJECT_ROOT / "Literature"


# ============================================================
# 3. Penn Action paths
# ============================================================

PENN_ACTION_ROOT = DATASETS_DIR / "Penn_Action"

PENN_ACTION_NPZ = (
    PENN_ACTION_ROOT
    / "penn_action_processed.npz"
)

PENN_ACTION_FRAMES_DIR = (
    PENN_ACTION_ROOT
    / "frames"
)

PENN_ACTION_METADATA_DIR = (
    PENN_ACTION_ROOT
    / "metadata"
)

PENN_ACTION_PROCESSED_DIR = (
    PENN_ACTION_ROOT
    / "processed"
)


# ============================================================
# 4. NTU RGB+D paths
# ============================================================

NTU_RGBD_ROOT = DATASETS_DIR / "NTU_RGBD"

NTU_RGB_DIR = (
    NTU_RGBD_ROOT
    / "rgb_videos"
)

NTU_SKELETON_DIR = (
    NTU_RGBD_ROOT
    / "skeletons"
)

NTU_METADATA_DIR = (
    NTU_RGBD_ROOT
    / "metadata"
)

NTU_PROCESSED_DIR = (
    NTU_RGBD_ROOT
    / "processed"
)

NTU_SELECTED_DIR = (
    NTU_RGBD_ROOT
    / "selected"
)


# ============================================================
# 5. Dataset-specific output directories
# ============================================================

PENN_ACTION_OUTPUT_DIR = (
    OUTPUTS_DIR
    / "PennAction_Model_Training"
)

NTU_RGBD_OUTPUT_DIR = (
    OUTPUTS_DIR
    / "NTU_RGBD"
)


# ============================================================
# 6. Directory creation
# ============================================================

def ensure_directory(path: Path) -> Path:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        The same directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_project_directories() -> None:
    """
    Create the main project directories used by the codebase.

    Raw dataset directories are not populated or modified.
    """
    directories = [
        DATASETS_DIR,
        OUTPUTS_DIR,
        SERVER_OUTPUTS_DIR,
        PENN_ACTION_METADATA_DIR,
        PENN_ACTION_PROCESSED_DIR,
        NTU_RGB_DIR,
        NTU_SKELETON_DIR,
        NTU_METADATA_DIR,
        NTU_PROCESSED_DIR,
        NTU_SELECTED_DIR,
        PENN_ACTION_OUTPUT_DIR,
        NTU_RGBD_OUTPUT_DIR,
    ]

    for directory in directories:
        ensure_directory(directory)


def print_project_paths() -> None:
    """
    Print important project paths for debugging.
    """
    paths = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATASETS_DIR": DATASETS_DIR,
        "OUTPUTS_DIR": OUTPUTS_DIR,
        "PENN_ACTION_ROOT": PENN_ACTION_ROOT,
        "NTU_RGBD_ROOT": NTU_RGBD_ROOT,
        "NTU_RGB_DIR": NTU_RGB_DIR,
        "NTU_SKELETON_DIR": NTU_SKELETON_DIR,
    }

    print("=" * 70)
    print("Project paths")
    print("=" * 70)

    for name, path in paths.items():
        print(f"{name:<24}: {path}")

    print("=" * 70)


if __name__ == "__main__":
    create_project_directories()
    print_project_paths()