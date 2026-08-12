from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.real_robot_adapter import (
    CameraCalibration,
    RGBDFrame,
    fuse_rgbd_frames,
    robot_depth_keep_mask,
)


def test_robot_depth_keep_mask_removes_only_depth_consistent_robot_pixels():
    measured = np.asarray([[1.000, 0.700, 1.030, 0.0]], dtype=np.float32)
    rendered = np.asarray([[1.000, 1.000, 1.000, 1.000]], dtype=np.float32)

    keep = robot_depth_keep_mask(
        measured,
        rendered,
        absolute_tolerance_m=0.01,
        relative_tolerance=0.0,
        dilation_pixels=0,
    )

    # Keep a foreground object, a mismatched/background depth, and invalid depth.
    assert keep.tolist() == [[False, True, True, True]]


def test_robot_depth_keep_mask_dilates_visible_robot_surface_conservatively():
    measured = np.full((3, 3), 2.0, dtype=np.float32)
    measured[0, 0] = 1.0
    rendered = np.zeros((3, 3), dtype=np.float32)
    rendered[1, 1] = 1.0

    keep = robot_depth_keep_mask(
        measured,
        rendered,
        absolute_tolerance_m=0.0,
        relative_tolerance=0.0,
        dilation_pixels=1,
    )

    assert not keep[0, 0]
    assert keep[2, 2]


def test_fusion_uses_rendered_robot_depth_before_projecting_points():
    rgb = np.asarray([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    frame = RGBDFrame("front", rgb=rgb, depth_m=np.asarray([[1.0, 0.7]], dtype=np.float32))
    calibration = CameraCalibration("front", np.eye(3), np.eye(4))

    fused = fuse_rgbd_frames(
        [frame],
        {"front": calibration},
        robot_link_points=np.zeros((0, 3), dtype=np.float32),
        camera_names=("front",),
        rendered_robot_depths={"front": np.asarray([[1.0, 1.0]], dtype=np.float32)},
        rendered_robot_absolute_tolerance_m=0.001,
        rendered_robot_relative_tolerance=0.0,
        rendered_robot_dilation_pixels=0,
    )

    assert fused.scene_points.shape == (1, 3)
    assert np.array_equal(fused.scene_colors, np.asarray([[0, 255, 0]], dtype=np.uint8))
