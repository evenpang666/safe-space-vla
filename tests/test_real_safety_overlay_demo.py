import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.demo_record_ur7e_safety_overlay_video import (
    DEFAULT_DEMO_CAMERA_NAMES,
    DEFAULT_IMAGE_OUTPUT,
    DEFAULT_OUTPUT,
    build_tabletop_obbs,
    parse_args,
    load_adapter,
    project_world_points_to_pixels,
    render_overlay_frame,
    resolve_output_path,
    resolve_demo_actions,
    run_demo,
    select_tabletop_obstacle_points,
)
from real_scripts.real_robot_adapter import CameraCalibration, RGBDFrame


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


def test_parse_args_accepts_image_output_mode_and_random_video_action_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_record_ur7e_safety_overlay_video.py",
            "--camera-calibration",
            "calibration.json",
            "--output-mode",
            "image",
            "--random-action-count",
            "4",
            "--random-xyz-mm",
            "6",
            "--random-rot",
            "0.02",
            "--random-seed",
            "12",
        ],
    )

    args = parse_args()

    assert args.output_mode == "image"
    assert args.random_action_count == 4
    assert args.random_xyz_mm == 6.0
    assert args.random_rot == 0.02
    assert args.random_seed == 12


def test_resolve_output_path_uses_png_default_for_image_mode():
    image_args = SimpleNamespace(output=DEFAULT_OUTPUT, output_mode="image")
    explicit_args = SimpleNamespace(output=Path("custom.png"), output_mode="image")
    video_args = SimpleNamespace(output=DEFAULT_OUTPUT, output_mode="video")

    assert resolve_output_path(image_args) == DEFAULT_IMAGE_OUTPUT
    assert resolve_output_path(explicit_args) == Path("custom.png")
    assert resolve_output_path(video_args) == DEFAULT_OUTPUT


def test_resolve_demo_actions_defaults_to_deterministic_small_random_actions_for_hardware_video():
    args = SimpleNamespace(
        replay_jsonl=None,
        output_mode="video",
        no_demo_actions=False,
        demo_action=None,
        random_action_count=5,
        random_xyz_mm=10.0,
        random_rot=0.03,
        random_seed=7,
    )

    actions = resolve_demo_actions(args)

    assert actions.shape == (5, 7)
    np.testing.assert_allclose(actions, resolve_demo_actions(args))
    assert np.all(np.abs(actions[:, :3]) <= 10.0)
    assert np.all(np.abs(actions[:, 3:6]) <= 0.03)
    np.testing.assert_allclose(actions[:, 6], 0.5)


def test_resolve_demo_actions_can_be_overridden_disabled_or_suppressed_for_image():
    override = [[5.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.25], [-5.0, -1.0, 0.0, 0.0, 0.0, -0.1, 0.25]]
    args = SimpleNamespace(replay_jsonl=None, output_mode="video", no_demo_actions=False, demo_action=override)
    disabled_args = SimpleNamespace(replay_jsonl=None, output_mode="video", no_demo_actions=True, demo_action=override)
    image_args = SimpleNamespace(replay_jsonl=None, output_mode="image", no_demo_actions=False, demo_action=override)
    replay_args = SimpleNamespace(replay_jsonl=Path("replay.jsonl"), output_mode="video", no_demo_actions=False, demo_action=override)

    np.testing.assert_allclose(resolve_demo_actions(args), override)
    assert resolve_demo_actions(disabled_args).shape == (0, 7)
    assert resolve_demo_actions(image_args).shape == (0, 7)
    assert resolve_demo_actions(replay_args).shape == (0, 7)


class _FakeDemoAdapter:
    def __init__(self):
        self.actions = []
        self.reset_called = False
        self.closed = False

    def reset(self):
        self.reset_called = True

    def get_observation(self):
        return {"qpos": np.zeros(6, dtype=np.float32)}

    def get_rgbd_frames(self):
        return [
            RGBDFrame(
                "front",
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((8, 8), dtype=np.float32),
            ),
            RGBDFrame(
                "wrist",
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((8, 8), dtype=np.float32),
            )
        ]

    def execute_action(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))

    def is_done(self):
        return False

    def close(self):
        self.closed = True


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(np.asarray(frame))

    def close(self):
        self.closed = True


def test_run_demo_video_mode_sends_random_hardware_demo_actions(monkeypatch, tmp_path: Path):
    adapter = _FakeDemoAdapter()
    writer = _FakeWriter()
    calibration = CameraCalibration("front", np.eye(3, dtype=np.float64), np.eye(4, dtype=np.float64))
    wrist_calibration = CameraCalibration("wrist", np.eye(3, dtype=np.float64), np.eye(4, dtype=np.float64))
    fused_camera_names = []
    args = SimpleNamespace(
        output=tmp_path / "demo.mp4",
        output_mode="video",
        camera_calibration=tmp_path / "calibration.json",
        adapter="unused:create_adapter",
        replay_jsonl=None,
        front_camera_name="front",
        max_frames=3,
        duration_sec=None,
        fps=20.0,
        points_per_link=2,
        gripper_width=0.085,
        pointcloud_stride=1,
        max_depth=3.0,
        workspace_bounds=None,
        robot_filter_radius=0.045,
        table_z=0.0,
        min_obstacle_height=0.03,
        max_obstacle_height=0.50,
        cluster_radius=0.08,
        min_cluster_points=32,
        point_radius=0,
        debug_npz=None,
        no_demo_actions=False,
        demo_action=None,
        demo_action_interval_sec=0.0,
        random_action_count=2,
        random_xyz_mm=10.0,
        random_rot=0.03,
        random_seed=5,
        camera_names=DEFAULT_DEMO_CAMERA_NAMES,
    )

    monkeypatch.setattr("real_scripts.demo_record_ur7e_safety_overlay_video.load_adapter", lambda _: adapter)
    monkeypatch.setattr(
        "real_scripts.demo_record_ur7e_safety_overlay_video.load_camera_calibrations",
        lambda _: {"front": calibration, "wrist": wrist_calibration},
    )
    monkeypatch.setattr("real_scripts.demo_record_ur7e_safety_overlay_video._open_video_writer", lambda *_args, **_kwargs: writer)

    def fake_fuse_rgbd_frames(frames, *_args, **_kwargs):
        fused_camera_names.extend(frame.camera_name for frame in frames)
        return SimpleNamespace(
            environment_points=np.zeros((0, 3), dtype=np.float32),
            environment_colors=np.zeros((0, 3), dtype=np.uint8),
        )

    monkeypatch.setattr("real_scripts.demo_record_ur7e_safety_overlay_video.fuse_rgbd_frames", fake_fuse_rgbd_frames)

    count = run_demo(args)

    assert count == 3
    assert adapter.reset_called is True
    assert adapter.closed is True
    assert writer.closed is True
    assert len(writer.frames) == 3
    assert len(adapter.actions) == 2
    np.testing.assert_allclose(adapter.actions, resolve_demo_actions(args))
    assert fused_camera_names == ["front", "wrist"] * 3


def test_run_demo_image_mode_saves_one_overlay_without_motion(monkeypatch, tmp_path: Path):
    adapter = _FakeDemoAdapter()
    saved_images = []
    calibration = CameraCalibration("front", np.eye(3, dtype=np.float64), np.eye(4, dtype=np.float64))
    wrist_calibration = CameraCalibration("wrist", np.eye(3, dtype=np.float64), np.eye(4, dtype=np.float64))
    args = SimpleNamespace(
        output=tmp_path / "front_overlay.png",
        output_mode="image",
        camera_calibration=tmp_path / "calibration.json",
        adapter="unused:create_adapter",
        replay_jsonl=None,
        front_camera_name="front",
        max_frames=10,
        duration_sec=None,
        fps=20.0,
        points_per_link=2,
        gripper_width=0.085,
        pointcloud_stride=1,
        max_depth=3.0,
        workspace_bounds=None,
        robot_filter_radius=0.045,
        table_z=0.0,
        min_obstacle_height=0.03,
        max_obstacle_height=0.50,
        cluster_radius=0.08,
        min_cluster_points=32,
        point_radius=0,
        debug_npz=None,
        no_demo_actions=False,
        demo_action=None,
        demo_action_interval_sec=0.0,
        random_action_count=2,
        random_xyz_mm=10.0,
        random_rot=0.03,
        random_seed=5,
        camera_names=DEFAULT_DEMO_CAMERA_NAMES,
    )

    monkeypatch.setattr("real_scripts.demo_record_ur7e_safety_overlay_video.load_adapter", lambda _: adapter)
    monkeypatch.setattr(
        "real_scripts.demo_record_ur7e_safety_overlay_video.load_camera_calibrations",
        lambda _: {"front": calibration, "wrist": wrist_calibration},
    )
    monkeypatch.setattr(
        "real_scripts.demo_record_ur7e_safety_overlay_video._save_rgb_image",
        lambda path, image: saved_images.append((path, np.asarray(image))),
    )
    monkeypatch.setattr(
        "real_scripts.demo_record_ur7e_safety_overlay_video.fuse_rgbd_frames",
        lambda *_args, **_kwargs: SimpleNamespace(
            environment_points=np.zeros((0, 3), dtype=np.float32),
            environment_colors=np.zeros((0, 3), dtype=np.uint8),
        ),
    )

    count = run_demo(args)

    assert count == 1
    assert adapter.actions == []
    assert len(saved_images) == 1
    assert saved_images[0][0] == args.output
    assert saved_images[0][1].shape == (8, 8, 3)


def test_run_demo_image_mode_writes_debug_evidence_images_and_summary(monkeypatch, tmp_path: Path):
    adapter = _FakeDemoAdapter()
    calibration = CameraCalibration("front", np.eye(3, dtype=np.float64), np.eye(4, dtype=np.float64))
    wrist_calibration = CameraCalibration("wrist", np.eye(3, dtype=np.float64), np.eye(4, dtype=np.float64))
    debug_dir = tmp_path / "debug_overlay"
    args = SimpleNamespace(
        output=tmp_path / "front_overlay.png",
        output_mode="image",
        camera_calibration=tmp_path / "calibration.json",
        adapter="unused:create_adapter",
        replay_jsonl=None,
        front_camera_name="front",
        max_frames=10,
        duration_sec=None,
        fps=20.0,
        points_per_link=2,
        gripper_width=0.085,
        pointcloud_stride=1,
        max_depth=3.0,
        workspace_bounds=None,
        robot_filter_radius=0.045,
        table_z=0.9,
        min_obstacle_height=0.03,
        max_obstacle_height=0.50,
        cluster_radius=0.08,
        min_cluster_points=4,
        point_radius=0,
        debug_npz=None,
        debug_image_dir=debug_dir,
        no_demo_actions=False,
        demo_action=None,
        demo_action_interval_sec=0.0,
        random_action_count=2,
        random_xyz_mm=10.0,
        random_rot=0.03,
        random_seed=5,
        camera_names=DEFAULT_DEMO_CAMERA_NAMES,
    )

    obstacle_points = np.asarray(
        [
            [0.00, 0.00, 1.00],
            [0.02, 0.00, 1.00],
            [0.00, 0.02, 1.10],
            [0.02, 0.02, 1.10],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr("real_scripts.demo_record_ur7e_safety_overlay_video.load_adapter", lambda _: adapter)
    monkeypatch.setattr(
        "real_scripts.demo_record_ur7e_safety_overlay_video.load_camera_calibrations",
        lambda _: {"front": calibration, "wrist": wrist_calibration},
    )
    monkeypatch.setattr(
        "real_scripts.demo_record_ur7e_safety_overlay_video.fuse_rgbd_frames",
        lambda *_args, **_kwargs: SimpleNamespace(
            scene_points=obstacle_points,
            scene_colors=np.full((4, 3), 128, dtype=np.uint8),
            environment_points=obstacle_points,
            environment_colors=np.full((4, 3), 128, dtype=np.uint8),
        ),
    )

    count = run_demo(args)

    assert count == 1
    expected_files = {
        "front_rgb.png",
        "overlay_full.png",
        "overlay_robot_only.png",
        "overlay_obstacles_only.png",
        "overlay_obbs_only.png",
        "topdown_scene_points.png",
        "topdown_robot_points.png",
        "topdown_obstacle_points.png",
        "debug_summary.json",
    }
    assert expected_files <= {path.name for path in debug_dir.iterdir()}
    summary = json.loads((debug_dir / "debug_summary.json").read_text(encoding="utf-8"))
    assert summary["scene_point_count"] == 4
    assert summary["obstacle_point_count"] == 4
    assert summary["obb_count"] == 1
    assert summary["front_rgb_shape"] == [8, 8, 3]


def test_load_adapter_passes_demo_camera_names_to_compatible_factory(monkeypatch):
    received = []

    class FakeModule:
        @staticmethod
        def create_adapter(*, camera_names=None):
            received.append(tuple(camera_names))
            return _FakeDemoAdapter()

    monkeypatch.setattr(
        "real_scripts.demo_record_ur7e_safety_overlay_video.importlib.import_module",
        lambda name: FakeModule,
    )
    args = SimpleNamespace(replay_jsonl=None, adapter="fake_module:create_adapter", camera_names=DEFAULT_DEMO_CAMERA_NAMES)

    adapter = load_adapter(args)

    assert isinstance(adapter, _FakeDemoAdapter)
    assert received == [("front", "wrist")]
