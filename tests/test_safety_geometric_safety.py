import importlib
import os
import sys

import numpy as np
import pytest

from safety_module.geometric_safety import predicted_link_points_collision


def test_import_safety_module_does_not_set_mujoco_gl(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    for module_name in [
        "safety_module",
        "safety_module.geometric_safety",
        "scripts.libero_joint_swept_pointcloud",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    importlib.import_module("safety_module")

    assert "MUJOCO_GL" not in os.environ


def test_predicted_link_points_collision_reports_obb_hit():
    pred = np.asarray([[[[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]]]], dtype=np.float32)
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.2, 0.0, 0.0]], dtype=np.float64),
        "obstacle_box_axes": np.asarray([np.eye(3)], dtype=np.float64),
        "obstacle_box_half_sizes": np.asarray([[0.05, 0.05, 0.05]], dtype=np.float64),
    }

    result = predicted_link_points_collision(pred, safe_space)

    assert result.collides is True
    assert result.method == "oriented_boxes"
    assert result.collision_point_count == 1


def test_predicted_link_points_collision_reports_safe_when_no_point_overlaps():
    pred = np.asarray([[[[[1.0, 1.0, 1.0], [1.2, 1.0, 1.0]]]]], dtype=np.float32)
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.2, 0.0, 0.0]], dtype=np.float64),
        "obstacle_box_axes": np.asarray([np.eye(3)], dtype=np.float64),
        "obstacle_box_half_sizes": np.asarray([[0.05, 0.05, 0.05]], dtype=np.float64),
    }

    result = predicted_link_points_collision(pred, safe_space)

    assert result.collides is False
    assert result.collision_point_count == 0


def test_predicted_link_points_collision_rejects_multi_sample_batch():
    pred = np.zeros((2, 1, 1, 2, 3), dtype=np.float32)
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.2, 0.0, 0.0]], dtype=np.float64),
        "obstacle_box_axes": np.asarray([np.eye(3)], dtype=np.float64),
        "obstacle_box_half_sizes": np.asarray([[0.05, 0.05, 0.05]], dtype=np.float64),
    }

    with pytest.raises(ValueError, match="single sample"):
        predicted_link_points_collision(pred, safe_space)
