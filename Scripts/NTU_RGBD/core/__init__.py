# Scripts/NTU_RGBD/core/__init__.py

"""
Core utilities for NTU RGB+D.
"""

from .config import (
    NTU_NUM_JOINTS,
    NTU_JOINT_NAMES,
    NTU_JOINT_INDEX,
    NTU_SKELETON_EDGES,
    PENN_13_JOINT_NAMES,
    PENN_13_FROM_NTU_INDICES,
    PENN_13_SKELETON_EDGES,
)

from .filename_parser import (
    NTUSampleID,
    parse_ntu_filename,
    get_sample_id,
    is_ntu_sample_filename,
    build_rgb_filename,
    build_skeleton_filename,
)

from .skeleton_reader import (
    NTUJoint,
    NTUBody,
    NTUFrame,
    NTUSkeletonSequence,
    read_skeleton_file,
    read_skeleton_summary,
    extract_body_track,
)

from .rgb_reader import (
    open_rgb_video,
    find_rgb_video,
    get_rgb_video_info,
    read_rgb_frame,
    iterate_rgb_frames,
    validate_rgb_video,
)

from .sample_matcher import (
    index_rgb_files,
    index_skeleton_files,
    match_rgb_and_skeleton,
    filter_samples,
    build_sample_metadata,
)

from .coordinate_projection import (
    tracking_state_to_visibility,
    coordinate_visibility,
    normalize_keypoints,
    denormalize_keypoints,
    scale_keypoints,
    clip_keypoints_to_image,
    keypoints_to_bbox,
    root_center_3d_pose,
)

from .person_selector import (
    get_frame_body_counts,
    is_single_person_sequence,
    max_body_count,
    count_body_occurrences,
    select_primary_body_id,
    select_primary_body_in_frame,
    extract_primary_pose_sequence,
)

from .joint_mapping import (
    select_joints,
    joint_names_to_indices,
    select_joints_by_name,
    convert_ntu25_to_penn13,
    convert_ntu_visibility_to_penn13,
    get_ntu_joint_name,
    get_penn13_joint_name,
)


__all__ = [
    "NTU_NUM_JOINTS",
    "NTU_JOINT_NAMES",
    "NTU_JOINT_INDEX",
    "NTU_SKELETON_EDGES",
    "PENN_13_JOINT_NAMES",
    "PENN_13_FROM_NTU_INDICES",
    "PENN_13_SKELETON_EDGES",

    "NTUSampleID",
    "parse_ntu_filename",
    "get_sample_id",
    "is_ntu_sample_filename",
    "build_rgb_filename",
    "build_skeleton_filename",

    "NTUJoint",
    "NTUBody",
    "NTUFrame",
    "NTUSkeletonSequence",
    "read_skeleton_file",
    "read_skeleton_summary",
    "extract_body_track",

    "open_rgb_video",
    "find_rgb_video",
    "get_rgb_video_info",
    "read_rgb_frame",
    "iterate_rgb_frames",
    "validate_rgb_video",

    "index_rgb_files",
    "index_skeleton_files",
    "match_rgb_and_skeleton",
    "filter_samples",
    "build_sample_metadata",

    "tracking_state_to_visibility",
    "coordinate_visibility",
    "normalize_keypoints",
    "denormalize_keypoints",
    "scale_keypoints",
    "clip_keypoints_to_image",
    "keypoints_to_bbox",
    "root_center_3d_pose",

    "get_frame_body_counts",
    "is_single_person_sequence",
    "max_body_count",
    "count_body_occurrences",
    "select_primary_body_id",
    "select_primary_body_in_frame",
    "extract_primary_pose_sequence",

    "select_joints",
    "joint_names_to_indices",
    "select_joints_by_name",
    "convert_ntu25_to_penn13",
    "convert_ntu_visibility_to_penn13",
    "get_ntu_joint_name",
    "get_penn13_joint_name",
]