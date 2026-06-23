from pathlib import Path
import sys

import numpy as np
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts import reconstruct_realsense_pointcloud as recon
from real_scripts.reconstruct_realsense_pointcloud import (
    depth_rgb_to_camera_points,
    depth_to_vis,
    save_pointcloud_outputs,
)


def test_depth_rgb_to_camera_points_uses_intrinsics_stride_and_depth_limit():
    rgb = np.asarray(
        [
            [[10, 0, 0], [20, 0, 0], [30, 0, 0]],
            [[40, 0, 0], [50, 0, 0], [60, 0, 0]],
        ],
        dtype=np.uint8,
    )
    depth_m = np.asarray([[1.0, 0.0, 3.5], [2.0, 4.0, 1.5]], dtype=np.float32)
    intrinsics = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)

    points, colors = depth_rgb_to_camera_points(rgb, depth_m, intrinsics, stride=1, max_depth=3.0)

    np.testing.assert_allclose(
        points,
        [
            [-0.5, -0.25, 1.0],
            [-1.0, 0.5, 2.0],
            [0.75, 0.375, 1.5],
        ],
        atol=1e-6,
    )
    np.testing.assert_array_equal(colors, [[10, 0, 0], [40, 0, 0], [60, 0, 0]])


def test_save_pointcloud_outputs_writes_npz_ply_and_preview_images(tmp_path: Path):
    rgb = np.full((2, 2, 3), 80, dtype=np.uint8)
    depth_m = np.asarray([[1.0, 2.0], [0.0, 1.5]], dtype=np.float32)
    points = np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.5]], dtype=np.float32)
    colors = np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    intrinsics = np.eye(3, dtype=np.float64)

    outputs = save_pointcloud_outputs(
        tmp_path,
        name="front_camera",
        rgb=rgb,
        depth_m=depth_m,
        points=points,
        colors=colors,
        intrinsics=intrinsics,
    )

    assert outputs["npz"].name == "front_camera_pointcloud.npz"
    assert outputs["ply"].name == "front_camera_pointcloud.ply"
    assert outputs["rgb"].exists()
    assert outputs["depth_vis"].exists()
    assert outputs["topdown"].exists()
    assert outputs["html"].name == "front_camera_pointcloud_viewer.html"
    assert outputs["html"].exists()
    data = np.load(outputs["npz"])
    np.testing.assert_allclose(data["points"], points)
    np.testing.assert_array_equal(data["colors"], colors)
    assert "element vertex 2" in outputs["ply"].read_text(encoding="utf-8")
    html = outputs["html"].read_text(encoding="utf-8")
    assert "<canvas" in html
    assert "webgl" in html.lower()
    assert "POINT_DATA" in html


def test_save_interactive_pointcloud_html_embeds_camera_frame_points(tmp_path: Path):
    points = np.asarray([[0.0, 0.0, 1.0], [0.25, -0.1, 1.5]], dtype=np.float32)
    colors = np.asarray([[255, 0, 0], [0, 128, 255]], dtype=np.uint8)
    output = tmp_path / "viewer.html"

    recon.save_interactive_pointcloud_html(output, points, colors, title="front point cloud")

    html = output.read_text(encoding="utf-8")
    assert "front point cloud" in html
    assert '"points":[[0.0,0.0,1.0],[0.25,-0.10000000149011612,1.5]]' in html
    assert '"colors":[[255,0,0],[0,128,255]]' in html
    assert "addEventListener('mousemove'" in html


def test_filter_camera_bounds_keeps_only_desktop_roi_points():
    points = np.asarray(
        [
            [0.0, 0.2, 0.8],
            [0.7, 0.2, 0.8],
            [0.0, -0.4, 0.8],
            [0.0, 0.2, 2.2],
        ],
        dtype=np.float32,
    )
    colors = np.asarray([[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]], dtype=np.uint8)

    roi_points, roi_colors = recon.filter_camera_bounds(points, colors, bounds=(-0.5, 0.5, -0.2, 0.5, 0.3, 1.5))

    np.testing.assert_allclose(roi_points, [[0.0, 0.2, 0.8]])
    np.testing.assert_array_equal(roi_colors, [[10, 0, 0]])


def test_select_off_plane_points_removes_table_plane_and_keeps_object_points():
    table = np.asarray([[x, 0.25, z] for x in np.linspace(-0.2, 0.2, 5) for z in np.linspace(0.7, 0.9, 5)], dtype=np.float32)
    obj = np.asarray([[0.03, 0.18, 0.78], [0.05, 0.16, 0.80], [0.01, 0.17, 0.82]], dtype=np.float32)
    points = np.concatenate([table, obj], axis=0)
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)

    selected_points, selected_colors, plane = recon.select_off_plane_points(
        points,
        colors,
        plane_threshold=0.015,
        min_plane_distance=0.03,
        max_plane_distance=0.20,
        ransac_iterations=80,
    )

    assert selected_points.shape[0] == obj.shape[0]
    assert selected_colors.shape[0] == obj.shape[0]
    assert plane["inlier_count"] >= table.shape[0]


def test_fit_oriented_obbs_clusters_points_and_returns_corners():
    cluster_a = np.asarray([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.04, 0.0], [0.03, 0.04, 0.05]], dtype=np.float32)
    cluster_b = cluster_a + np.asarray([0.5, 0.0, 0.0], dtype=np.float32)
    points = np.concatenate([cluster_a, cluster_b], axis=0)

    obbs = recon.fit_oriented_obbs(points, cluster_radius=0.12, min_cluster_points=4)

    assert len(obbs) == 2
    assert all(obb.corners.shape == (8, 3) for obb in obbs)
    assert all(np.all(obb.extents > 0.0) for obb in obbs)


def test_save_pointcloud_outputs_writes_tabletop_roi_and_obb_files(tmp_path: Path):
    table = np.asarray([[x, 0.25, z] for x in np.linspace(-0.2, 0.2, 5) for z in np.linspace(0.7, 0.9, 5)], dtype=np.float32)
    obj = np.asarray([[0.03, 0.18, 0.78], [0.05, 0.16, 0.80], [0.01, 0.17, 0.82], [0.04, 0.15, 0.79]], dtype=np.float32)
    far = np.asarray([[0.0, 0.2, 2.5]], dtype=np.float32)
    points = np.concatenate([table, obj, far], axis=0)
    colors = np.tile(np.asarray([[120, 160, 180]], dtype=np.uint8), (points.shape[0], 1))
    rgb = np.full((4, 4, 3), 80, dtype=np.uint8)
    depth_m = np.ones((4, 4), dtype=np.float32)

    outputs = save_pointcloud_outputs(
        tmp_path,
        name="front",
        rgb=rgb,
        depth_m=depth_m,
        points=points,
        colors=colors,
        intrinsics=np.eye(3),
        tabletop_bounds=(-0.4, 0.4, 0.0, 0.4, 0.5, 1.2),
        obb_min_cluster_points=3,
        obb_cluster_radius=0.12,
    )

    assert outputs["tabletop_npz"].exists()
    assert outputs["tabletop_obstacle_npz"].exists()
    assert outputs["tabletop_obbs_json"].exists()
    assert outputs["tabletop_obbs_html"].exists()
    roi = np.load(outputs["tabletop_npz"])
    assert roi["points"].shape[0] == table.shape[0] + obj.shape[0]
    html = outputs["tabletop_obbs_html"].read_text(encoding="utf-8")
    assert '"obbs":[' in html


def test_split_robot_points_by_model_distance_separates_visible_robot_points():
    scene_points = np.asarray(
        [
            [0.01, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    colors = np.asarray([[255, 0, 0], [200, 0, 0], [0, 0, 255]], dtype=np.uint8)
    robot_model_points = np.asarray([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)

    robot_points, robot_colors, non_robot_points, non_robot_colors, mask = recon.split_robot_points_by_model_distance(
        scene_points,
        colors,
        robot_model_points,
        radius=0.06,
    )

    np.testing.assert_array_equal(mask, [True, True, False])
    np.testing.assert_allclose(robot_points, scene_points[:2])
    np.testing.assert_array_equal(robot_colors, colors[:2])
    np.testing.assert_allclose(non_robot_points, scene_points[2:])
    np.testing.assert_array_equal(non_robot_colors, colors[2:])


def test_save_pointcloud_outputs_writes_robot_filter_debug_files(tmp_path: Path):
    rgb = np.full((2, 2, 3), 80, dtype=np.uint8)
    depth_m = np.ones((2, 2), dtype=np.float32)
    points = np.asarray([[0.0, 0.0, 0.1625], [-0.2, 0.0, 0.1625], [1.0, 1.0, 1.0]], dtype=np.float32)
    colors = np.asarray([[255, 255, 255], [220, 220, 220], [0, 0, 255]], dtype=np.uint8)

    outputs = save_pointcloud_outputs(
        tmp_path,
        name="front",
        rgb=rgb,
        depth_m=depth_m,
        points=points,
        colors=colors,
        intrinsics=np.eye(3),
        robot_qpos=np.zeros(6, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float64),
        robot_filter_radius=0.08,
        robot_points_per_link=8,
    )

    assert outputs["robot_observed_npz"].exists()
    assert outputs["robot_observed_html"].exists()
    assert outputs["non_robot_npz"].exists()
    assert outputs["fk_robot_model_html"].exists()
    robot_data = np.load(outputs["robot_observed_npz"])
    non_robot_data = np.load(outputs["non_robot_npz"])
    assert robot_data["points"].shape[0] >= 1
    assert non_robot_data["points"].shape[0] == 1


def test_resolve_robot_qpos_reads_current_joints_from_rtde_controller():
    class FakeController:
        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.connected = False
            self.closed = False

        def connect(self):
            self.connected = True

        def get_current_joints(self):
            return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        def close(self):
            self.closed = True

    created = []

    def factory(robot_ip):
        controller = FakeController(robot_ip)
        created.append(controller)
        return controller

    args = SimpleNamespace(use_rtde_qpos=True, robot_qpos=None, robot_ip="192.168.0.10")

    qpos = recon.resolve_robot_qpos(args, controller_factory=factory)

    np.testing.assert_allclose(qpos, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert created[0].robot_ip == "192.168.0.10"
    assert created[0].connected is True
    assert created[0].closed is True


def test_resolve_robot_qpos_rejects_manual_and_rtde_qpos_together():
    args = SimpleNamespace(use_rtde_qpos=True, robot_qpos=[0.0] * 6, robot_ip="192.168.0.10")

    try:
        recon.resolve_robot_qpos(args, controller_factory=lambda robot_ip: None)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected qpos source conflict to fail")

    assert "--use-rtde-qpos" in message
    assert "--robot-qpos" in message


def test_depth_to_vis_renders_turbo_colorbar_and_keeps_invalid_depth_black():
    depth_m = np.asarray([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)

    image = depth_to_vis(depth_m, vis_max=4.0, with_colorbar=True)

    assert image.shape[0] == 2
    assert image.shape[1] > 2
    np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
    valid_pixel = image[0, 1]
    assert len(set(int(channel) for channel in valid_pixel)) > 1
    assert np.count_nonzero(image[:, 2:]) > 0
