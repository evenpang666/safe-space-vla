from __future__ import annotations

import numpy as np

from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler
from real_scripts.real_cbf_qp import (
    OrientedBox,
    point_jacobian_fd,
    project_joint_delta_qp,
    select_inter_arm_constraints,
    select_point_flow_constraints,
)


def test_pi05_surface_validator_is_an_instance_method() -> None:
    import inspect

    from openpi.models_pytorch.pi0_pytorch import PI05SafetyPytorch

    parameters = list(inspect.signature(PI05SafetyPytorch._validate_surface_inputs).parameters)
    assert parameters == ["self", "robot_points", "joint_positions"]


def test_hdf5_task_description_has_precedence_over_project_default(tmp_path) -> None:
    import h5py

    from scripts.preprocess_quest3_hdf5 import _resolve_task_text

    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as file:
        file.attrs["task_description"] = "place the vial in the rack "
        text, source = _resolve_task_text(None, file.attrs, "fallback")
        assert text == "place the vial in the rack"
        assert source == "hdf5_root_attribute:task_description"
        text, source = _resolve_task_text("explicit instruction", file.attrs, "fallback")
        assert text == "explicit instruction"
        assert source == "cli_override"


def test_surface_sampler_does_not_invent_right_arm() -> None:
    sampler = DualUR7eSurfacePointSampler(points_per_link=4)
    left = sampler.link_points(np.zeros(6), np.zeros(1))
    dual = sampler.link_points(np.zeros(12), np.zeros(2))
    assert left.shape == (len(sampler.left_link_names), 4, 3)
    assert dual.shape == (len(sampler.left_link_names) + len(sampler.right_link_names), 4, 3)
    assert np.isfinite(dual).all()


def test_dual_finite_difference_jacobian() -> None:
    sampler = DualUR7eSurfacePointSampler(points_per_link=3)
    indices = np.asarray((10, len(sampler.left_link_names) * 3 + 10))
    jacobian = point_jacobian_fd(sampler, np.zeros(12), gripper_position=np.zeros(2), point_indices=indices)
    assert jacobian.shape == (2, 3, 12)
    assert np.isfinite(jacobian).all()


def test_obb_and_inter_arm_constraints_project_joint_delta() -> None:
    current = np.asarray(((0.10, 0, 0), (-0.10, 0, 0)), dtype=np.float32)
    predicted = np.asarray((((0.01, 0, 0), (-0.01, 0, 0)),), dtype=np.float32)
    boxes = [OrientedBox(np.zeros(3), np.eye(3), np.full(3, 0.05))]
    constraints = select_point_flow_constraints(current, predicted, boxes, trigger_margin_m=0.01)
    pairs = select_inter_arm_constraints(current, predicted, np.asarray((True, False)), minimum_distance_m=0.04, trigger_margin_m=0.01)
    assert constraints
    assert pairs
    jacobian = np.zeros((2, 3, 12), dtype=np.float32)
    jacobian[0, 0, 0] = 1.0
    jacobian[1, 0, 6] = 1.0
    safe, info = project_joint_delta_qp(
        np.zeros(12), jacobian, [*constraints, *pairs],
        lower_delta=np.full(12, -0.2), upper_delta=np.full(12, 0.2), iterations=64,
    )
    assert safe.shape == (12,)
    assert info["triggered"]
