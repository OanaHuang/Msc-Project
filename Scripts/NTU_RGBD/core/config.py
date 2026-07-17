# Scripts/NTU_RGBD/core/config.py

from __future__ import annotations


# ============================================================
# 1. Dataset constants
# ============================================================

NTU_NUM_JOINTS = 25

NTU_DEFAULT_RGB_WIDTH = 1920
NTU_DEFAULT_RGB_HEIGHT = 1080
NTU_DEFAULT_FPS = 30.0


# ============================================================
# 2. NTU 25 joint names
# ============================================================

NTU_JOINT_NAMES = (
    "spine_base",       # 0
    "spine_mid",        # 1
    "neck",             # 2
    "head",             # 3
    "shoulder_left",    # 4
    "elbow_left",       # 5
    "wrist_left",       # 6
    "hand_left",        # 7
    "shoulder_right",   # 8
    "elbow_right",      # 9
    "wrist_right",      # 10
    "hand_right",       # 11
    "hip_left",         # 12
    "knee_left",        # 13
    "ankle_left",       # 14
    "foot_left",        # 15
    "hip_right",        # 16
    "knee_right",       # 17
    "ankle_right",      # 18
    "foot_right",       # 19
    "spine_shoulder",   # 20
    "hand_tip_left",    # 21
    "thumb_left",       # 22
    "hand_tip_right",   # 23
    "thumb_right",      # 24
)

NTU_JOINT_INDEX = {
    name: index
    for index, name in enumerate(NTU_JOINT_NAMES)
}


# ============================================================
# 3. NTU skeleton connections
# ============================================================

NTU_SKELETON_EDGES = (
    # Torso and head
    (0, 1),
    (1, 20),
    (20, 2),
    (2, 3),

    # Left arm
    (20, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 21),
    (7, 22),

    # Right arm
    (20, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 23),
    (11, 24),

    # Left leg
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),

    # Right leg
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
)


# ============================================================
# 4. Tracking states
# ============================================================

TRACKING_STATE_NOT_TRACKED = 0
TRACKING_STATE_INFERRED = 1
TRACKING_STATE_TRACKED = 2


# ============================================================
# 5. NTU60 official split information
# ============================================================

NTU60_CROSS_SUBJECT_TRAIN_PERFORMERS = frozenset({
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
})

NTU60_CROSS_VIEW_TRAIN_CAMERAS = frozenset({
    2,
    3,
})


# ============================================================
# 6. Penn-compatible 13-joint subset
# ============================================================

PENN_13_JOINT_NAMES = (
    "head",
    "shoulder_left",
    "shoulder_right",
    "elbow_left",
    "elbow_right",
    "wrist_left",
    "wrist_right",
    "hip_left",
    "hip_right",
    "knee_left",
    "knee_right",
    "ankle_left",
    "ankle_right",
)

PENN_13_FROM_NTU_INDICES = (
    3,   # head
    4,   # left shoulder
    8,   # right shoulder
    5,   # left elbow
    9,   # right elbow
    6,   # left wrist
    10,  # right wrist
    12,  # left hip
    16,  # right hip
    13,  # left knee
    17,  # right knee
    14,  # left ankle
    18,  # right ankle
)

PENN_13_SKELETON_EDGES = (
    (0, 1),
    (0, 2),

    (1, 3),
    (3, 5),

    (2, 4),
    (4, 6),

    (1, 7),
    (2, 8),
    (7, 8),

    (7, 9),
    (9, 11),

    (8, 10),
    (10, 12),
)