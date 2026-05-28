import importlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.build_pi05_safety_decoder_dataset import (
    DatasetConfig,
    load_seed_samples,
    save_decoder_dataset,
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
        assert data["link_names"].tolist() == ["link0", "link1"]
        assert str(data["task_suite"]) == "libero_spatial"
        assert int(data["task_id"]) == 0
        assert int(data["init_state_id"]) == 0
        assert int(data["points_per_link"]) == 4
        assert int(data["samples_per_action"]) == 1
