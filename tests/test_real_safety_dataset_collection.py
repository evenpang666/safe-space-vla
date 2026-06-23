from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.collect_pi05_real_safety_dataset import (
    RealReplanSample,
    append_real_trajectory_samples,
    build_ur7_policy_input,
)
from real_scripts.real_robot_adapter import (
    CameraCalibration,
    RGBDFrame,
    ReplayJsonlAdapter,
    UR7E_DH_PARAMETERS,
    UR7ELinkPointSampler,
    crop_workspace,
    depth_to_world_points,
    filter_robot_points,
    fuse_rgbd_frames,
)


def test_ur7e_link_sampler_returns_fixed_topology_points():
    sampler = UR7ELinkPointSampler(points_per_link=5)

    points = sampler.link_points(np.zeros(6, dtype=np.float64))

    assert points.shape == (7, 5, 3)
    assert sampler.link_names == (
        "base_shoulder",
        "shoulder_upper",
        "upper_forearm",
        "forearm_wrist1",
        "wrist1_wrist2",
        "wrist2_wrist3",
        "gripper_width",
    )
    np.testing.assert_allclose(points[:, 0], sampler.link_segments(np.zeros(6))[:, 0])
    np.testing.assert_allclose(points[:, -1], sampler.link_segments(np.zeros(6))[:, 1])


def test_ur7e_dh_parameters_match_official_values():
    expected = (
        (0.0, np.pi / 2.0, 0.1625),
        (-0.425, 0.0, 0.0),
        (-0.3922, 0.0, 0.0),
        (0.0, np.pi / 2.0, 0.1333),
        (0.0, -np.pi / 2.0, 0.0997),
        (0.0, 0.0, 0.0996),
    )

    np.testing.assert_allclose(np.asarray(UR7E_DH_PARAMETERS), np.asarray(expected))


def test_depth_to_world_points_uses_camera_intrinsics_and_extrinsics():
    calibration = CameraCalibration(
        name="front",
        intrinsics=np.asarray([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64),
        camera_to_world=np.asarray(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    frame = RGBDFrame(
        camera_name="front",
        rgb=np.asarray([[[10, 20, 30]]], dtype=np.uint8),
        depth_m=np.asarray([[2.0]], dtype=np.float32),
    )

    points, colors = depth_to_world_points(frame, calibration)

    np.testing.assert_allclose(points, [[0.5, 1.5, 5.0]], atol=1e-6)
    np.testing.assert_array_equal(colors, [[10, 20, 30]])


def test_filter_robot_points_removes_points_near_ur7e_links():
    scene_points = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    robot_points = np.asarray([[[0.0, 0.0, 0.0]]], dtype=np.float32)

    keep = filter_robot_points(scene_points, robot_points, radius=0.05)

    np.testing.assert_array_equal(keep, [False, False, True])


def test_fuse_rgbd_frames_filters_robot_and_workspace():
    calibration = CameraCalibration(
        name="front",
        intrinsics=np.eye(3, dtype=np.float64),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    rgb = np.zeros((1, 2, 3), dtype=np.uint8)
    depth = np.asarray([[1.0, 2.0]], dtype=np.float32)
    frame = RGBDFrame(camera_name="front", rgb=rgb, depth_m=depth)
    robot_points = np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32)

    cloud = fuse_rgbd_frames(
        [frame],
        {"front": calibration},
        robot_link_points=robot_points,
        robot_filter_radius=0.05,
        workspace_bounds=(-0.5, 2.5, -0.5, 0.5, 0.0, 3.0),
    )

    assert cloud.scene_points.shape == (2, 3)
    assert cloud.environment_points.shape == (1, 3)
    np.testing.assert_allclose(cloud.environment_points[0], [2.0, 0.0, 2.0])


def test_crop_workspace_keeps_points_inside_bounds():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    colors = np.arange(9, dtype=np.uint8).reshape(3, 3)

    kept_points, kept_colors = crop_workspace(points, colors, (0.5, 1.5, -1, 1, -1, 1))

    np.testing.assert_allclose(kept_points, [[1, 0, 0]])
    np.testing.assert_array_equal(kept_colors, [[3, 4, 5]])


def test_build_ur7_policy_input_uses_three_d435i_camera_defaults():
    observation = {
        "front_rgb": np.ones((2, 2, 3), dtype=np.uint8),
        "side_rgb": np.full((2, 2, 3), 2, dtype=np.uint8),
        "wrist_rgb": np.full((2, 2, 3), 3, dtype=np.uint8),
        "qpos": np.arange(6, dtype=np.float32),
        "gripper": np.asarray([0.25], dtype=np.float32),
    }

    payload = build_ur7_policy_input(observation, prompt="pick up the block")

    np.testing.assert_array_equal(payload["base_rgb"], observation["front_rgb"])
    np.testing.assert_array_equal(payload["side_rgb"], observation["side_rgb"])
    np.testing.assert_array_equal(payload["wrist_rgb"], observation["wrist_rgb"])
    np.testing.assert_allclose(payload["joints"], np.arange(6, dtype=np.float32))
    np.testing.assert_allclose(payload["gripper"], [0.25])
    assert payload["prompt"] == "pick up the block"


def test_replay_jsonl_adapter_returns_three_d435i_rgbd_frames(tmp_path: Path):
    record = {
        "front_rgb": np.zeros((1, 1, 3), dtype=np.uint8).tolist(),
        "side_rgb": np.ones((1, 1, 3), dtype=np.uint8).tolist(),
        "wrist_rgb": np.full((1, 1, 3), 2, dtype=np.uint8).tolist(),
        "front_depth_m": np.asarray([[1.0]], dtype=np.float32).tolist(),
        "side_depth_m": np.asarray([[2.0]], dtype=np.float32).tolist(),
        "wrist_depth_m": np.asarray([[3.0]], dtype=np.float32).tolist(),
        "qpos": np.zeros(6, dtype=np.float32).tolist(),
    }
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(__import__("json").dumps(record) + "\n", encoding="utf-8")
    adapter = ReplayJsonlAdapter(replay_path)

    frames = adapter.get_rgbd_frames()

    assert [frame.camera_name for frame in frames] == ["front", "side", "wrist"]
    np.testing.assert_allclose([frame.depth_m[0, 0] for frame in frames], [1.0, 2.0, 3.0])


def test_append_real_trajectory_samples_saves_existing_dataset_schema(tmp_path: Path):
    sampler = UR7ELinkPointSampler(points_per_link=3)
    q0 = np.zeros(6, dtype=np.float32)
    q1 = np.asarray([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    surface_frames = np.stack([sampler.link_points(q0), sampler.link_points(q1)]).astype(np.float32)
    sample = RealReplanSample(
        prefix_tokens=np.ones((4, 8), dtype=np.float32),
        action_chunk=np.ones((1, 7), dtype=np.float32),
        start_joint_vector=q0,
        rollout_id=2,
        step_id=0,
    )

    count = append_real_trajectory_samples(
        [sample],
        surface_frames=surface_frames,
        link_names=np.asarray(sampler.link_names),
        output=tmp_path / "real_safety.npz",
        max_samples=4,
        policy_config="pi05_ur7",
        checkpoint_dir="checkpoint",
        points_per_link=3,
    )

    assert count == 1
    with np.load(tmp_path / "real_safety.npz", allow_pickle=False) as data:
        assert data["prefix_tokens"].shape == (1, 4, 8)
        assert data["action_chunks"].shape == (1, 1, 7)
        assert data["target_link_points"].shape == (1, 2, 7, 3, 3)
        assert data["arm_points"].shape == (1, 21, 3)
        assert str(data["task_suite"]) == "real_ur"
        assert str(data["skeleton_source"]) == "ur7e_fk_surface"
        assert str(data["target_source"]) == "real_rollout_surface"
