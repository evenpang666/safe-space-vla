from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.demo_record_ur7e_safety_overlay_video import (
    build_tabletop_obbs,
    project_world_points_to_pixels,
    render_overlay_frame,
    select_tabletop_obstacle_points,
)
from real_scripts.real_robot_adapter import CameraCalibration


def test_project_world_points_to_pixels_uses_front_camera_calibration():
    calibration = CameraCalibration(
        name="front",
        intrinsics=np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    points = np.asarray([[0.1, -0.2, 1.0], [10.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32)

    uv, depth, valid = project_world_points_to_pixels(points, calibration, width=100, height=80)

    np.testing.assert_allclose(uv[0], [60.0, 20.0])
    np.testing.assert_allclose(depth[0], 1.0)
    np.testing.assert_array_equal(valid, [True, False, False])


def test_select_tabletop_obstacle_points_keeps_points_above_table_only():
    points = np.asarray(
        [
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.06],
            [0.0, 0.0, 0.30],
            [0.0, 0.0, 0.60],
        ],
        dtype=np.float32,
    )
    colors = np.arange(12, dtype=np.uint8).reshape(4, 3)

    kept_points, kept_colors = select_tabletop_obstacle_points(
        points,
        colors,
        table_z=0.0,
        min_height_above_table=0.05,
        max_height_above_table=0.50,
    )

    np.testing.assert_allclose(kept_points, [[0.0, 0.0, 0.06], [0.0, 0.0, 0.30]])
    np.testing.assert_array_equal(kept_colors, colors[1:3])


def test_build_tabletop_obbs_returns_upright_boxes_for_xy_clusters():
    first = np.asarray([[0.0, 0.0, 0.1], [0.1, 0.0, 0.1], [0.0, 0.2, 0.2], [0.1, 0.2, 0.2]], dtype=np.float32)
    second = first + np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    points = np.concatenate([first, second], axis=0)

    obbs = build_tabletop_obbs(points, cluster_radius=0.25, min_cluster_points=4)

    assert len(obbs) == 2
    centers = np.asarray(sorted([obb.center.tolist() for obb in obbs], key=lambda item: item[0]))
    np.testing.assert_allclose(centers[:, :2], [[0.05, 0.1], [1.05, 0.1]], atol=1e-5)
    assert all(obb.corners.shape == (8, 3) for obb in obbs)
    assert all(obb.extents[2] > 0.0 for obb in obbs)


def test_render_overlay_frame_draws_robot_points_obstacle_points_and_obb_edges():
    calibration = CameraCalibration(
        name="front",
        intrinsics=np.asarray([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    robot_points = np.asarray([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]], dtype=np.float32)
    obstacle_points = np.asarray([[0.0, 0.1, 1.0], [0.1, 0.1, 1.0], [0.0, 0.2, 1.0], [0.1, 0.2, 1.0]], dtype=np.float32)
    obbs = build_tabletop_obbs(obstacle_points, cluster_radius=0.25, min_cluster_points=4)

    overlay = render_overlay_frame(
        rgb,
        front_calibration=calibration,
        robot_link_points=robot_points,
        obstacle_points=obstacle_points,
        obstacle_obbs=obbs,
        point_radius=0,
    )

    assert overlay.shape == rgb.shape
    assert np.count_nonzero(overlay) > 0
    assert np.any(np.all(overlay == np.asarray([0, 220, 255], dtype=np.uint8), axis=-1))
    assert np.any(np.all(overlay == np.asarray([255, 80, 20], dtype=np.uint8), axis=-1))
    assert np.any(np.all(overlay == np.asarray([40, 255, 90], dtype=np.uint8), axis=-1))
