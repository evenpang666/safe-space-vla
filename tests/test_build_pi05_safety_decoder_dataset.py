import importlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from safety_module.safety_flow_point_model import SafetyFlowPointModel, sample_flow_matching_batch
from scripts.build_pi05_safety_decoder_dataset import (
    DatasetConfig,
    anchor_skeleton_segments_from_path,
    derive_flow_point_targets,
    load_seed_samples,
    save_decoder_dataset,
    sample_box_surface_points,
    robot_link_geom_ids,
    transform_link_surface_template,
    validate_seed_arrays,
)


def test_import_does_not_set_mujoco_gl(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    sys.modules.pop("scripts.build_pi05_safety_decoder_dataset", None)
    sys.modules.pop("libero_joint_swept_pointcloud", None)
    sys.modules.pop("scripts.libero_joint_swept_pointcloud", None)

    importlib.import_module("scripts.build_pi05_safety_decoder_dataset")

    assert "MUJOCO_GL" not in os.environ


def test_validate_seed_arrays_accepts_matching_prefix_actions_and_joints():
    prefix = np.zeros((3, 5, 7), dtype=np.float32)
    actions = np.zeros((3, 4, 6), dtype=np.float32)
    start_joints = np.zeros((3, 6), dtype=np.float32)

    validate_seed_arrays(prefix, actions, start_joints)


def test_validate_seed_arrays_rejects_mismatched_sample_count():
    prefix = np.zeros((3, 5, 7), dtype=np.float32)
    actions = np.zeros((2, 4, 6), dtype=np.float32)
    start_joints = np.zeros((3, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="same first dimension"):
        validate_seed_arrays(prefix, actions, start_joints)


def test_parse_args_defaults_to_surface_points(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_pi05_safety_decoder_dataset.py",
            "--seed-samples",
            "seed.npz",
        ],
    )
    module = importlib.import_module("scripts.build_pi05_safety_decoder_dataset")

    args = module.parse_args()

    assert args.skeleton_source == "surface"


def test_sample_box_surface_points_returns_points_on_box_boundary():
    rng = np.random.default_rng(0)

    points = sample_box_surface_points(np.asarray([0.1, 0.2, 0.3]), 64, rng)

    assert points.shape == (64, 3)
    assert np.all(np.abs(points) <= np.asarray([0.1, 0.2, 0.3]) + 1e-6)
    on_boundary = np.isclose(np.abs(points), np.asarray([0.1, 0.2, 0.3]), atol=1e-6).any(axis=1)
    assert np.all(on_boundary)


def test_transform_link_surface_template_preserves_link_point_topology():
    class _Data:
        geom_xpos = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        geom_xmat = np.asarray([np.eye(3), np.eye(3)], dtype=np.float64).reshape(2, 9)

    class _Sim:
        data = _Data()

    local_points = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
        ],
        dtype=np.float32,
    )
    geom_ids = np.asarray([[0, 0], [1, 1]], dtype=np.int64)

    world = transform_link_surface_template(_Sim(), local_points, geom_ids)

    assert world.shape == (2, 2, 3)
    np.testing.assert_allclose(world[0], [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    np.testing.assert_allclose(world[1], [[0.0, 2.0, 0.0], [0.0, 2.2, 0.0]])


def test_robot_link_geom_ids_accepts_link_child_body_names():
    class _Model:
        geom_bodyid = np.asarray([0, 1, 2, 3], dtype=np.int64)
        body_names = ["robot0_link1", "robot0_link1_visual", "robot0_link2_collision", "mount0_pedestal"]

    grouped = robot_link_geom_ids(
        _Model(),
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        link_body_names=("robot0_link1", "robot0_link2"),
    )

    assert grouped == [[0, 1], [2]]


def test_anchor_skeleton_segments_from_path_builds_seven_clean_arm_links():
    anchor_path = np.zeros((2, 8, 3), dtype=np.float64)
    anchor_path[:, :, 0] = np.arange(8, dtype=np.float64)
    anchor_path[1, :, 2] = 1.0

    segments, link_names = anchor_skeleton_segments_from_path(anchor_path)

    assert segments.shape == (2, 7, 2, 3)
    np.testing.assert_allclose(segments[0, 0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(segments[1, -1], [[6.0, 0.0, 1.0], [7.0, 0.0, 1.0]])
    assert link_names.tolist() == [
        "link0_link1",
        "link1_link2",
        "link2_link3",
        "link3_link4",
        "link4_link5",
        "link5_link6",
        "link6_link7",
    ]


def test_load_seed_samples_reads_required_arrays(tmp_path: Path):
    path = tmp_path / "seed_samples.npz"
    np.savez_compressed(
        path,
        prefix_tokens=np.ones((2, 3, 4), dtype=np.float32),
        action_chunks=np.ones((2, 5, 7), dtype=np.float32),
        start_joint_vectors=np.ones((2, 7), dtype=np.float32),
    )

    prefix, actions, start_joints = load_seed_samples(path)

    assert prefix.shape == (2, 3, 4)
    assert actions.shape == (2, 5, 7)
    assert start_joints.shape == (2, 7)


def test_normalize_fk_inputs_truncates_extra_start_and_action_dims():
    module = importlib.import_module("scripts.build_pi05_safety_decoder_dataset")

    start, actions = module.normalize_fk_inputs(
        start_joint_vector=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        action_chunk=np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        action_dim=3,
    )

    assert start.dtype == np.float64
    assert actions.dtype == np.float64
    np.testing.assert_allclose(start, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(actions, [[0.1, 0.2, 0.3]])


def test_derive_flow_point_targets_uses_current_points_as_input_and_future_offsets_as_target():
    target_link_points = np.asarray(
        [
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            [[[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]],
            [[[0.5, 1.0, 0.0], [1.5, 1.0, 0.0]]],
        ],
        dtype=np.float32,
    )

    derived = derive_flow_point_targets(target_link_points)

    assert derived["current_link_points"].shape == (1, 2, 3)
    assert derived["future_link_offsets"].shape == (2, 1, 2, 3)
    assert derived["arm_points"].shape == (2, 3)
    assert derived["target_point_offsets"].shape == (2, 2, 3)
    np.testing.assert_allclose(derived["current_link_points"], target_link_points[0])
    np.testing.assert_allclose(derived["arm_points"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        derived["target_point_offsets"],
        [
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.5, 1.0, 0.0], [0.5, 1.0, 0.0]],
        ],
    )


def test_derived_flow_targets_feed_safety_flow_point_model():
    target_link_points = np.random.randn(4, 2, 3, 3).astype(np.float32)
    derived = derive_flow_point_targets(target_link_points)
    prefix_tokens = np.random.randn(1, 5, 8).astype(np.float32)
    arm_points = derived["arm_points"][None, ...]
    x_1 = derived["target_point_offsets"][None, ...]
    x_s, s, _x_0, _v_target = sample_flow_matching_batch(torch_from_numpy(x_1))
    model = SafetyFlowPointModel(
        arm_point_dim=3,
        prefix_dim=8,
        hidden_dim=16,
        n_future=x_1.shape[1],
        max_points=arm_points.shape[1],
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        ffn_dim=32,
    )

    v_pred = model(
        arm_points=torch_from_numpy(arm_points),
        prefix_tokens=torch_from_numpy(prefix_tokens),
        x_s=x_s,
        s=s,
    )

    assert tuple(v_pred.shape) == x_1.shape


def test_save_decoder_dataset_writes_expected_schema(tmp_path: Path):
    output = tmp_path / "decoder_dataset.npz"
    config = DatasetConfig(task_suite="libero_spatial", task_id=0, init_state_id=0, points_per_link=4)
    save_decoder_dataset(
        output,
        prefix_tokens=np.zeros((2, 3, 4), dtype=np.float32),
        action_chunks=np.zeros((2, 5, 7), dtype=np.float32),
        start_joint_vectors=np.zeros((2, 7), dtype=np.float32),
        target_link_points=np.zeros((2, 6, 8, 4, 3), dtype=np.float32),
        link_names=np.asarray(["link0", "link1"]),
        config=config,
    )

    with np.load(output) as data:
        assert data["prefix_tokens"].shape == (2, 3, 4)
        assert data["action_chunks"].shape == (2, 5, 7)
        assert data["start_joint_vectors"].shape == (2, 7)
        assert data["target_link_points"].shape == (2, 6, 8, 4, 3)
        assert data["current_link_points"].shape == (2, 8, 4, 3)
        assert data["future_link_offsets"].shape == (2, 5, 8, 4, 3)
        assert data["arm_points"].shape == (2, 32, 3)
        assert data["target_point_offsets"].shape == (2, 5, 32, 3)
        assert data["link_names"].tolist() == ["link0", "link1"]
        assert str(data["task_suite"]) == "libero_spatial"
        assert int(data["task_id"]) == 0
        assert int(data["init_state_id"]) == 0
        assert int(data["points_per_link"]) == 4
        assert int(data["samples_per_action"]) == 1
        assert str(data["skeleton_source"]) == "surface"
        assert str(data["coordinate_frame"]) == "mujoco_world"
        assert str(data["target_link_points_frame"]) == "mujoco_world"
        assert str(data["arm_points_frame"]) == "mujoco_world"
        assert str(data["target_point_offsets_frame"]) == "mujoco_world_delta"


def torch_from_numpy(array: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(array, dtype=np.float32))
