from pathlib import Path
import types
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch

import scripts.collect_pi05_libero_safety_decoder_dataset as collector
import scripts.evaluate_pi05_safety_decoder_on_libero as evaluator
from scripts.collect_pi05_libero_safety_decoder_dataset import (
    EVAL_SCENE_WINE_BOTTLE_CATEGORY,
    SceneObstacleSpec,
    adapt_init_state_for_scene_obstacle,
    materialize_eval_scene_wine_bottle_xml,
    patch_bddl_with_scene_obstacle,
    register_eval_scene_obstacle_objects,
    reset_scene_obstacle_pose,
)
from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig
from safety_module.safety_flow_point_model import SafetyFlowPointModel
from scripts.evaluate_pi05_safety_decoder_on_libero import (
    VideoFrameBuffer,
    absolute_link_points_from_offsets,
    annotate_video_frame,
    append_prediction_video_frame,
    build_realtime_safe_space_from_env,
    compute_point_error_metrics,
    current_point_obb_cbf_constraints,
    draw_projected_obbs,
    infer_flow_points_per_link,
    load_safe_space_for_video,
    load_decoder_checkpoint,
    load_safety_model_checkpoint,
    load_repo_script_module,
    point_flow_obb_collision,
    point_flow_obb_cbf_constraints,
    predict_link_points,
    predict_safety_flow_link_points,
    query_remote_safety_prediction,
    resolve_device_name,
    save_evaluation,
    select_safety_prediction_source,
    solve_cbf_qp_projection,
)


def _bddl_section_text(text: str, section_name: str) -> str:
    start = text.index(f"(:{section_name}")
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise AssertionError(f"BDDL :{section_name} section is not balanced")


def test_evaluate_script_help_runs_when_invoked_by_path():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/evaluate_pi05_safety_decoder_on_libero.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--policy-server-host" in result.stdout
    assert "--video-output" in result.stdout
    assert "--prediction-steps" in result.stdout
    assert "--safe-space" in result.stdout
    assert "--realtime-obbs" in result.stdout
    assert "--no-realtime-obbs" in result.stdout
    assert "--obb-component-connectivity" in result.stdout
    assert "--obb-target-geom-name-patterns" in result.stdout
    assert "--skeleton-source" in result.stdout
    assert "--scene-obstacle" in result.stdout
    assert "--scene-obstacle-xy" in result.stdout
    assert "--cbf-action-space" in result.stdout
    assert "cartesian_action" in result.stdout
    assert "--cbf-cartesian-damping" in result.stdout
    assert "--cbf-correction-target" in result.stdout
    assert "projected" in result.stdout


def test_patch_bddl_with_scene_obstacle_inserts_default_center_wine_bottle():
    bddl = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Pick the bowl)
  (:regions
    (table_center
      (:target main_table)
      (:ranges (
        (-0.1 -0.01 -0.05 0.01)
      ))
    )
  )
  (:fixtures
    main_table - table
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
  )
  (:obj_of_interest
    akita_black_bowl_1
  )
  (:init
    (On akita_black_bowl_1 main_table_table_center)
  )
  (:goal
    (And (On akita_black_bowl_1 main_table_table_center))
  )
)"""

    patched = patch_bddl_with_scene_obstacle(bddl, SceneObstacleSpec(kind="wine_bottle"))

    fixtures = _bddl_section_text(patched, "fixtures")
    objects = _bddl_section_text(patched, "objects")

    assert "eval_scene_obstacle_region" in patched
    assert "eval_scene_obstacle_1 - eval_scene_wine_bottle_obstacle" not in fixtures
    assert "eval_scene_obstacle_1 - eval_scene_wine_bottle_obstacle" in objects
    assert "(On eval_scene_obstacle_1 main_table_eval_scene_obstacle_region)" in patched
    assert "(-0.01 -0.01 0.01 0.01)" in patched


def test_materialize_eval_scene_wine_bottle_xml_scales_mesh_geoms_and_aligns_sites(tmp_path: Path):
    source_dir = tmp_path / "wine_bottle"
    source_dir.mkdir()
    source = source_dir / "wine_bottle.xml"
    source.write_text(
        """<mujoco model="wine_bottle">
  <asset>
    <texture file="label.png" name="label" type="2d"/>
    <mesh file="visual/body.msh" name="body" scale="0.5 0.5 0.5"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom type="mesh" mesh="body" group="1"/>
        <geom type="box" pos="0 0 0.1" size="0.01 0.02 0.03" group="0"/>
      </body>
      <site name="bottom_site" pos="0 0 -0.05"/>
      <site name="top_site" pos="0 0 0.05"/>
    </body>
  </worldbody>
</mujoco>"""
    )

    output = materialize_eval_scene_wine_bottle_xml(source, output_dir=tmp_path / "out", scale=2.0)

    text = output.read_text()
    assert f'file="{source_dir / "label.png"}"' in text
    assert f'file="{source_dir / "visual/body.msh"}"' in text
    assert 'scale="1 1 1"' in text
    assert 'pos="0 0 0.2"' in text
    assert 'size="0.02 0.04 0.06"' in text
    assert 'pos="0 0 0.14"' in text
    assert 'pos="0 0 0.26"' in text


def test_registered_eval_scene_obstacle_defaults_to_damped_free_joint(monkeypatch, tmp_path: Path):
    fake_objects_dict = {}
    registered = {}

    def fake_register_object(cls):
        registered["cls"] = cls
        fake_objects_dict[EVAL_SCENE_WINE_BOTTLE_CATEGORY] = cls

    class FakeMujocoXMLObject:
        def __init__(self, xml_path, *, name, joints, obj_type, duplicate_collision_geoms):
            self.xml_path = xml_path
            self.name = name
            self.joints = joints
            self.obj_type = obj_type
            self.duplicate_collision_geoms = duplicate_collision_geoms

    fake_base_object = types.ModuleType("libero.libero.envs.base_object")
    fake_base_object.OBJECTS_DICT = fake_objects_dict
    fake_base_object.register_object = fake_register_object
    fake_objects = types.ModuleType("robosuite.models.objects")
    fake_objects.MujocoXMLObject = FakeMujocoXMLObject

    monkeypatch.setattr(collector, "ensure_third_party_paths", lambda: None)
    monkeypatch.setattr(
        collector,
        "materialize_eval_scene_wine_bottle_xml",
        lambda _source_xml, *, output_dir, scale: tmp_path / "eval_scene_wine_bottle_obstacle.xml",
    )
    monkeypatch.setitem(sys.modules, "libero.libero.envs.base_object", fake_base_object)
    monkeypatch.setitem(sys.modules, "robosuite.models.objects", fake_objects)

    register_eval_scene_obstacle_objects()
    obstacle = registered["cls"](name="eval_scene_obstacle_1")

    assert obstacle.joints == [dict(type="free", damping="0.0005")]


def test_patch_bddl_with_scene_obstacle_uses_explicit_xy():
    bddl = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Pick the bowl)
  (:regions
    (table_center
      (:target main_table)
      (:ranges (
        (-0.1 -0.01 -0.05 0.01)
      ))
    )
  )
  (:fixtures
    main_table - table
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
  )
  (:obj_of_interest
    akita_black_bowl_1
  )
  (:init
    (On akita_black_bowl_1 main_table_table_center)
  )
  (:goal
    (And (On akita_black_bowl_1 main_table_table_center))
  )
)"""

    patched = patch_bddl_with_scene_obstacle(
        bddl,
        SceneObstacleSpec(kind="wine_bottle", xy=(0.12, -0.08)),
    )

    assert "(0.11 -0.09 0.13 -0.07)" in patched


def test_adapt_init_state_for_scene_obstacle_pads_upright_zero_velocity_free_joint_state():
    class _FakeSim:
        class _Model:
            nq = 10
            nv = 9
            njnt = 1
            jnt_qposadr = np.asarray([2], dtype=np.int64)
            jnt_dofadr = np.asarray([1], dtype=np.int64)
            jnt_type = np.asarray([0], dtype=np.int64)

            def joint_id2name(self, _joint_id):
                return "eval_scene_obstacle_1_joint0"

        model = _Model()

    class _FakeEnv:
        sim = _FakeSim()

        def get_sim_state(self):
            # time + 10 qpos + 9 qvel
            return np.asarray([9.0, *range(10), *range(100, 109)], dtype=np.float64)

    original = np.asarray([1.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0], dtype=np.float64)

    adapted = adapt_init_state_for_scene_obstacle(
        original,
        _FakeEnv(),
        SceneObstacleSpec(kind="wine_bottle"),
    )

    np.testing.assert_allclose(
        adapted,
        [
            1.0,
            10.0,
            11.0,
            0.0,
            0.0,
            4.0,
            1.0,
            0.0,
            0.0,
            0.0,
            12.0,
            20.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            21.0,
            22.0,
        ],
    )


def test_reset_scene_obstacle_pose_sets_upright_pose_and_refreshes_sim():
    class _FakeSim:
        class _Model:
            nq = 7
            nv = 6
            njnt = 1
            jnt_qposadr = np.asarray([0], dtype=np.int64)
            jnt_dofadr = np.asarray([0], dtype=np.int64)
            jnt_type = np.asarray([0], dtype=np.int64)

            def joint_id2name(self, _joint_id):
                return "eval_scene_obstacle_1_joint0"

        class _Data:
            qpos = np.asarray([0.4, -0.2, 0.91, 0.707, 0.0, 0.707, 0.0], dtype=np.float64)
            qvel = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)

        model = _Model()
        data = _Data()
        forwarded = False

        def forward(self):
            self.forwarded = True

    class _FakeEnv:
        sim = _FakeSim()

    reset_scene_obstacle_pose(
        _FakeEnv(),
        SceneObstacleSpec(kind="wine_bottle", xy=(0.12, -0.08)),
    )

    np.testing.assert_allclose(_FakeEnv.sim.data.qpos, [0.12, -0.08, 0.91, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(_FakeEnv.sim.data.qvel, np.zeros(6))
    assert _FakeEnv.sim.forwarded


def test_load_repo_script_module_ignores_openpi_scripts_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "scripts", type(sys)("scripts"))

    module = load_repo_script_module("collect_pi05_libero_safety_decoder_dataset")

    assert Path(module.__file__).resolve() == (
        Path(__file__).resolve().parents[1] / "scripts" / "collect_pi05_libero_safety_decoder_dataset.py"
    ).resolve()
    assert hasattr(module, "query_policy_action_and_prefix")


def test_resolve_device_name_accepts_gpu_alias():
    assert resolve_device_name("gpu") == "cuda"


def test_select_safety_prediction_source_auto_prefers_remote_metadata():
    metadata = {"returns_safety_predictions": True}

    assert select_safety_prediction_source("auto", metadata) == "remote"


def test_select_safety_prediction_source_auto_falls_back_to_local_without_remote_metadata():
    assert select_safety_prediction_source("auto", {}) == "local"


def test_query_remote_safety_prediction_returns_predicted_link_points():
    class _FakePolicy:
        def infer(self, obs):
            np.testing.assert_allclose(obs["prefix_tokens"], [[1.0]])
            np.testing.assert_allclose(obs["current_link_points"], np.zeros((1, 2, 3), dtype=np.float32))
            assert obs["safety_only"] is True
            return {"pred_link_points": np.ones((2, 1, 2, 3), dtype=np.float32)}

    pred = query_remote_safety_prediction(
        _FakePolicy(),
        prefix_tokens=np.asarray([[1.0]], dtype=np.float32),
        current_link_points=np.zeros((1, 2, 3), dtype=np.float32),
    )

    assert pred.shape == (2, 1, 2, 3)
    assert pred.dtype == np.float32


def test_point_flow_obb_cbf_constraints_selects_future_obb_intrusion():
    current = np.asarray([[[1.2, 0.0, 0.0]]], dtype=np.float32)
    pred = np.asarray([[[[0.2, 0.0, 0.0]]]], dtype=np.float32)
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "obstacle_box_half_sizes": np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
    }

    constraints = point_flow_obb_cbf_constraints(
        pred,
        current,
        safe_space,
        collision_margin=0.0,
        trigger_margin=0.0,
        max_constraints=8,
    )

    assert len(constraints) == 1
    constraint = constraints[0]
    assert constraint.link_id == 0
    assert constraint.point_id == 0
    assert constraint.obb_id == 0
    np.testing.assert_allclose(constraint.normal, [1.0, 0.0, 0.0])
    assert constraint.h == pytest.approx(-0.3)


def test_point_flow_obb_cbf_constraints_can_filter_to_specific_future_times():
    current = np.asarray([[[0.3, 0.0, 0.0]]], dtype=np.float32)
    pred = np.zeros((3, 1, 1, 3), dtype=np.float32)
    pred[0, 0, 0] = [0.0, 0.0, 0.0]
    pred[1, 0, 0] = [0.4, 0.0, 0.0]
    pred[2, 0, 0] = [0.0, 0.0, 0.0]
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "obstacle_box_half_sizes": np.asarray([[0.1, 0.1, 0.1]], dtype=np.float32),
    }

    constraints = point_flow_obb_cbf_constraints(
        pred,
        current,
        safe_space,
        collision_margin=0.0,
        trigger_margin=0.0,
        max_constraints=8,
        allowed_time_indices={2},
    )

    assert [item.time_index for item in constraints] == [2]


def test_predicted_frame_action_cbf_constraints_default_excludes_current_points_at_time_zero():
    current = np.asarray(
        [
            [
                [0.12, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    pred = np.asarray(
        [
            [
                [
                    [0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            [
                [
                    [0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                ]
            ],
        ],
        dtype=np.float32,
    )
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "obstacle_box_half_sizes": np.asarray([[0.1, 0.1, 0.1]], dtype=np.float32),
    }

    constraints = evaluator.predicted_frame_action_cbf_constraints(
        pred,
        current,
        safe_space,
        collision_margin=0.0,
        trigger_margin=0.03,
        max_constraints=8,
    )

    assert [item.time_index for item in constraints] == [0]
    assert evaluator.cbf_action_indices_from_constraints(
        constraints,
        current_action_offset=3,
        action_count=5,
    ) == [3]
    predicted_points = [tuple(np.round(item.predicted_point, 4)) for item in constraints]
    assert (0.12, 0.0, 0.0) not in predicted_points
    assert (0.0, 0.0, 0.0) in predicted_points


def test_predicted_frame_action_cbf_constraints_can_include_current_points_explicitly():
    current = np.asarray([[[0.12, 0.0, 0.0]]], dtype=np.float32)
    pred = np.asarray([[[[0.5, 0.0, 0.0]]]], dtype=np.float32)
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "obstacle_box_half_sizes": np.asarray([[0.1, 0.1, 0.1]], dtype=np.float32),
    }

    constraints = evaluator.predicted_frame_action_cbf_constraints(
        pred,
        current,
        safe_space,
        collision_margin=0.0,
        trigger_margin=0.03,
        max_constraints=8,
        include_current_points=True,
    )

    assert [item.time_index for item in constraints] == [0]
    np.testing.assert_allclose(constraints[0].predicted_point, [0.12, 0.0, 0.0])


def test_predicted_frame_action_cbf_constraints_respect_allowed_time_indices_for_current_points():
    current = np.asarray([[[0.12, 0.0, 0.0]]], dtype=np.float32)
    pred = np.asarray(
        [
            [[[0.5, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0]]],
        ],
        dtype=np.float32,
    )
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "obstacle_box_half_sizes": np.asarray([[0.1, 0.1, 0.1]], dtype=np.float32),
    }

    constraints = evaluator.predicted_frame_action_cbf_constraints(
        pred,
        current,
        safe_space,
        collision_margin=0.0,
        trigger_margin=0.03,
        max_constraints=8,
        allowed_time_indices={1},
    )

    assert [item.time_index for item in constraints] == [1]
    np.testing.assert_allclose(constraints[0].predicted_point, [0.0, 0.0, 0.0])


def test_cbf_action_indices_from_constraints_maps_future_times_to_active_chunk_indices():
    constraints = [
        evaluator.PointFlowCbfConstraint(
            time_index=0,
            link_id=0,
            point_id=0,
            obb_id=0,
            face_axis=0,
            normal=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            h=-0.1,
            current_point=np.zeros((3,), dtype=np.float32),
            predicted_point=np.zeros((3,), dtype=np.float32),
        ),
        evaluator.PointFlowCbfConstraint(
            time_index=2,
            link_id=0,
            point_id=1,
            obb_id=0,
            face_axis=0,
            normal=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            h=-0.2,
            current_point=np.zeros((3,), dtype=np.float32),
            predicted_point=np.zeros((3,), dtype=np.float32),
        ),
    ]

    indices = evaluator.cbf_action_indices_from_constraints(
        constraints,
        current_action_offset=1,
        action_count=4,
    )

    assert indices == [1, 3]


def test_apply_frame_indexed_cbf_corrections_only_changes_matching_future_actions():
    constraints = [
        evaluator.PointFlowCbfConstraint(
            time_index=2,
            link_id=0,
            point_id=0,
            obb_id=0,
            face_axis=0,
            normal=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            h=-0.1,
            current_point=np.zeros((3,), dtype=np.float32),
            predicted_point=np.zeros((3,), dtype=np.float32),
        )
    ]
    chunk = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )

    corrected, info = evaluator.apply_frame_indexed_cbf_corrections(
        chunk,
        constraints,
        current_action_offset=0,
        correct_action_fn=lambda index, action, selected: (
            action + np.asarray([10.0 + index, 0.0], dtype=np.float64),
            {"triggered": bool(selected), "success": True, "max_violation": 0.0},
        ),
    )

    np.testing.assert_allclose(corrected[0], [1.0, 0.0])
    np.testing.assert_allclose(corrected[1], [2.0, 0.0])
    np.testing.assert_allclose(corrected[2], [15.0, 0.0])
    assert info["corrected_action_indices"] == [2]
    assert info["constraint_count"] == 1


def test_current_point_obb_cbf_constraints_selects_current_near_obstacle_points():
    current = np.asarray(
        [
            [[0.53, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "obstacle_box_half_sizes": np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
    }

    constraints = current_point_obb_cbf_constraints(
        current,
        safe_space,
        collision_margin=0.0,
        trigger_margin=0.05,
        max_constraints=8,
    )

    assert len(constraints) == 1
    constraint = constraints[0]
    assert constraint.link_id == 0
    assert constraint.point_id == 0
    assert constraint.time_index == 0
    np.testing.assert_allclose(constraint.normal, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(constraint.current_point, [0.53, 0.0, 0.0])
    np.testing.assert_allclose(constraint.predicted_point, constraint.current_point)
    assert constraint.h == pytest.approx(0.03)


def test_current_point_obb_cbf_constraints_does_not_double_count_collision_margin():
    current = np.asarray([[[0.535, 0.0, 0.0]]], dtype=np.float32)
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "obstacle_box_half_sizes": np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
    }

    constraints = current_point_obb_cbf_constraints(
        current,
        safe_space,
        collision_margin=0.01,
        trigger_margin=0.02,
        max_constraints=8,
    )

    assert constraints == []


def test_solve_cbf_qp_projection_removes_inward_component_and_keeps_tangent():
    nominal = np.asarray([-0.4, 0.25], dtype=np.float64)
    a = np.asarray([[1.0, 0.0]], dtype=np.float64)
    b = np.asarray([0.0], dtype=np.float64)

    result = solve_cbf_qp_projection(
        nominal,
        a,
        b,
        lower=np.asarray([-1.0, -1.0], dtype=np.float64),
        upper=np.asarray([1.0, 1.0], dtype=np.float64),
        iterations=4,
    )

    assert result.success is True
    np.testing.assert_allclose(result.action, [0.0, 0.25], atol=1e-8)


def test_cbf_qp_projected_fallback_keeps_best_effort_projection_tangent():
    projection = evaluator.CbfQpProjectionResult(
        action=np.asarray([0.0, 0.25], dtype=np.float64),
        success=False,
        max_violation=1e-3,
        iterations=1,
    )

    safe = evaluator.cbf_qp_action_from_projection(
        projection,
        qp_to_action=lambda action: np.asarray(action, dtype=np.float64),
        qp_nominal=np.asarray([-0.4, 0.25], dtype=np.float64),
        nominal=np.asarray([-0.4, 0.25], dtype=np.float64),
        fallback="projected",
    )

    np.testing.assert_allclose(safe, [0.0, 0.25], atol=1e-8)


def test_resolve_cbf_action_space_auto_uses_cartesian_for_libero_osc_position_env():
    assert evaluator.resolve_cbf_action_space("auto", action_dim=4, arm_dim=7) == "cartesian_action"
    assert evaluator.resolve_cbf_action_space("auto", action_dim=7, arm_dim=7) == "cartesian_action"
    assert evaluator.resolve_cbf_action_space("auto", action_dim=8, arm_dim=7) == "joint_delta"
    assert evaluator.resolve_cbf_action_space("joint_delta", action_dim=4, arm_dim=7) == "joint_delta"


def test_cartesian_action_adapter_optimizes_executable_xyz_directly():
    nominal = np.asarray([0.2, -0.1, 0.0, 0.75], dtype=np.float64)

    action_xyz, to_action = evaluator.cartesian_action_to_qp_action(nominal, action_dim=4)
    safe_action = to_action(np.asarray([0.0, -0.2, 0.3], dtype=np.float64))

    np.testing.assert_allclose(action_xyz, [0.2, -0.1, 0.0])
    np.testing.assert_allclose(safe_action, [0.0, -0.2, 0.3, 0.75])


def test_cartesian_delta_action_adapter_round_trips_through_joint_delta_space():
    nominal = np.asarray([0.2, -0.1, 0.0, 0.75], dtype=np.float64)
    eef_jacobian = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 4.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    joint_delta, to_action = evaluator.cartesian_delta_action_to_joint_delta(nominal, eef_jacobian, arm_dim=2)
    safe_action = to_action(np.asarray([0.0, -0.05], dtype=np.float64))

    np.testing.assert_allclose(joint_delta, [0.1, -0.025], atol=1e-8)
    np.testing.assert_allclose(safe_action, [0.0, -0.2, 0.0, 0.75], atol=1e-8)


def test_executable_libero_action_keeps_xyz_and_last_gripper_for_cartesian_env():
    action = np.asarray([0.1, 0.2, 0.3, 9.0, 8.0, 7.0, -1.0], dtype=np.float64)

    executable = evaluator.executable_libero_action(action, action_dim=4)

    np.testing.assert_allclose(executable, [0.1, 0.2, 0.3, -1.0])


def test_finite_difference_action_point_jacobians_differentiates_executable_action_space():
    def action_position_fn(action_xyz):
        action_xyz = np.asarray(action_xyz, dtype=np.float64)
        return np.asarray([[[action_xyz[0], 2.0 * action_xyz[1], -action_xyz[2]]]], dtype=np.float64)

    jacobians = evaluator.finite_difference_action_point_jacobians(
        action_position_fn,
        np.asarray([0.1, -0.2, 0.3], dtype=np.float64),
        [(0, 0)],
        eps=1e-5,
    )

    np.testing.assert_allclose(jacobians[(0, 0)], np.diag([1.0, 2.0, -1.0]), atol=1e-8)


def test_env_runtime_state_snapshot_restores_common_step_counters():
    env = types.SimpleNamespace(timestep=3, cur_time=0.25, _elapsed_steps=4)

    snapshot = evaluator.snapshot_env_runtime_state(env)
    env.timestep = 30
    env.cur_time = 2.5
    env._elapsed_steps = 40
    evaluator.restore_env_runtime_state(env, snapshot)

    assert env.timestep == 3
    assert env.cur_time == 0.25
    assert env._elapsed_steps == 4


def test_env_runtime_state_snapshot_restores_nested_done_flag():
    inner = types.SimpleNamespace(timestep=9, cur_time=0.45, done=False)
    env = types.SimpleNamespace(env=inner)

    snapshot = evaluator.snapshot_env_runtime_state(env)
    inner.timestep = 10
    inner.cur_time = 0.5
    inner.done = True
    evaluator.restore_env_runtime_state(env, snapshot)

    assert inner.timestep == 9
    assert inner.cur_time == 0.45
    assert inner.done is False


def test_cbf_action_delta_norms_measure_executed_change():
    nominal = np.asarray([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    executed = np.asarray([[1.0, 0.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float32)

    norms = evaluator.cbf_action_delta_norms(nominal, executed)

    np.testing.assert_allclose(norms, [2.0, 0.0])


def test_save_evaluation_writes_cbf_action_diagnostics(tmp_path: Path):
    output = tmp_path / "eval.npz"

    evaluator.save_evaluation(
        output,
        pred_link_points=np.zeros((1, 1, 1, 1, 3), dtype=np.float32),
        target_link_points=np.zeros((1, 1, 1, 1, 3), dtype=np.float32),
        prefix_tokens_shape=np.asarray([[1, 2]], dtype=np.int64),
        action_chunks=np.zeros((1, 1, 3), dtype=np.float32),
        metrics={
            "sample_mse": np.zeros((1,), dtype=np.float32),
            "sample_mean_l2": np.zeros((1,), dtype=np.float32),
            "sample_max_l2": np.zeros((1,), dtype=np.float32),
            "mean_mse": np.asarray(0.0, dtype=np.float32),
            "mean_l2": np.asarray(0.0, dtype=np.float32),
            "max_l2": np.asarray(0.0, dtype=np.float32),
        },
        rollout_ids=np.zeros((1,), dtype=np.int64),
        step_ids=np.zeros((1,), dtype=np.int64),
        link_names=np.asarray(["link"]),
        coordinate_frame="mujoco_world",
        real_collision_flags=np.asarray([True], dtype=bool),
        real_collision_contact_counts=np.asarray([2], dtype=np.int64),
        nominal_actions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        executed_actions=np.asarray([[1.0, 0.0, 3.0]], dtype=np.float32),
        cbf_action_delta_norms=np.asarray([2.0], dtype=np.float32),
        cbf_corrected_action_masks=np.asarray([[False, True, False]], dtype=bool),
    )

    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_allclose(data["nominal_actions"], [[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(data["executed_actions"], [[1.0, 0.0, 3.0]])
        np.testing.assert_allclose(data["cbf_action_delta_norms"], [2.0])
        np.testing.assert_array_equal(data["cbf_corrected_action_masks"], [[False, True, False]])
        np.testing.assert_array_equal(data["real_collision_flags"], [True])
        np.testing.assert_array_equal(data["real_collision_contact_counts"], [2])


def test_infer_flow_points_per_link_uses_surface_checkpoint_topology():
    assert infer_flow_points_per_link(max_points=896, skeleton_source="surface", requested_points_per_link=24) == 128


def test_infer_flow_points_per_link_rejects_non_divisible_surface_topology():
    with pytest.raises(ValueError, match="not divisible"):
        infer_flow_points_per_link(max_points=895, skeleton_source="surface", requested_points_per_link=24)


def test_load_decoder_checkpoint_and_predict_link_points(tmp_path: Path):
    checkpoint = tmp_path / "decoder.pt"
    config = SafetyPointDecoderConfig(
        token_dim=4,
        hidden_dim=8,
        num_layers=1,
        horizon=2,
        num_links=3,
        points_per_link=2,
    )
    model = SafetyPointDecoder(config)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict(), "epoch": 3, "loss": 0.1},
        checkpoint,
    )

    loaded = load_decoder_checkpoint(checkpoint, torch.device("cpu"))
    pred = predict_link_points(loaded, np.zeros((5, 4), dtype=np.float32), torch.device("cpu"))

    assert pred.shape == (2, 3, 2, 3)
    assert pred.dtype == np.float32


def test_load_safety_model_checkpoint_detects_flow_model(tmp_path: Path):
    checkpoint = tmp_path / "flow.pt"
    kwargs = {
        "arm_point_dim": 3,
        "prefix_dim": 4,
        "hidden_dim": 8,
        "n_future": 2,
        "max_points": 6,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 2,
        "ffn_dim": 16,
        "dropout": 0.0,
        "max_prefix_tokens": 16,
    }
    model = SafetyFlowPointModel(**kwargs)
    torch.save(
        {
            "model_type": "SafetyFlowPointModel",
            "model_state_dict": model.state_dict(),
            "model_kwargs": kwargs,
        },
        checkpoint,
    )

    loaded = load_safety_model_checkpoint(checkpoint, torch.device("cpu"))

    assert loaded.model_type == "flow"
    assert isinstance(loaded.model, SafetyFlowPointModel)


def test_absolute_link_points_from_offsets_reconstructs_future_link_points():
    current = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
    offsets = np.asarray(
        [
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    points = absolute_link_points_from_offsets(offsets, current)

    assert points.shape == (2, 1, 2, 3)
    np.testing.assert_allclose(points[0], current + offsets[0].reshape(1, 2, 3))


def test_predict_safety_flow_link_points_returns_absolute_future_points():
    torch.manual_seed(0)
    model = SafetyFlowPointModel(
        arm_point_dim=3,
        prefix_dim=4,
        hidden_dim=8,
        n_future=2,
        max_points=6,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        ffn_dim=16,
    )
    current = np.zeros((3, 2, 3), dtype=np.float32)
    prefix = np.zeros((5, 4), dtype=np.float32)

    pred = predict_safety_flow_link_points(
        model,
        prefix,
        current,
        device=torch.device("cpu"),
        prediction_steps=2,
    )

    assert pred.shape == (2, 3, 2, 3)
    assert pred.dtype == np.float32


def test_predict_safety_flow_link_points_rejects_wrong_point_count():
    model = SafetyFlowPointModel(
        arm_point_dim=3,
        prefix_dim=4,
        hidden_dim=8,
        n_future=2,
        max_points=6,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        ffn_dim=16,
    )

    with pytest.raises(ValueError, match="max_points"):
        predict_safety_flow_link_points(
            model,
            np.zeros((5, 4), dtype=np.float32),
            np.zeros((2, 2, 3), dtype=np.float32),
            device=torch.device("cpu"),
            prediction_steps=2,
        )


def test_predict_safety_flow_link_points_accepts_readonly_prefix_without_warning():
    model = SafetyFlowPointModel(
        arm_point_dim=3,
        prefix_dim=4,
        hidden_dim=8,
        n_future=1,
        max_points=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        ffn_dim=16,
    )
    prefix = np.zeros((3, 4), dtype=np.float32)
    prefix.setflags(write=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = predict_safety_flow_link_points(
            model,
            prefix,
            np.zeros((1, 2, 3), dtype=np.float32),
            device=torch.device("cpu"),
            prediction_steps=1,
        )

    assert pred.shape == (1, 1, 2, 3)
    assert not any("NumPy array is not writable" in str(item.message) for item in caught)


def test_compute_point_error_metrics_reports_l2_errors():
    pred = np.asarray([[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]], dtype=np.float32)
    target = np.asarray([[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32)

    metrics = compute_point_error_metrics(pred, target)

    assert metrics["mse"] == pytest.approx(2.5 / 3.0)
    assert metrics["mean_l2"] == 1.5
    assert metrics["max_l2"] == 2.0


def test_load_safe_space_for_video_reads_obb_fields(tmp_path: Path):
    safe_space = tmp_path / "safe_space.npz"
    np.savez_compressed(
        safe_space,
        obstacle_box_centers=np.zeros((1, 3), dtype=np.float32),
        obstacle_box_axes=np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        obstacle_box_half_sizes=np.ones((1, 3), dtype=np.float32),
        obstacle_box_corners=np.zeros((1, 8, 3), dtype=np.float32),
    )

    loaded = load_safe_space_for_video(safe_space)

    assert loaded["obstacle_box_centers"].shape == (1, 3)
    assert loaded["obstacle_box_axes"].shape == (1, 3, 3)
    assert loaded["obstacle_box_half_sizes"].shape == (1, 3)
    assert loaded["obstacle_box_corners"].shape == (1, 8, 3)


def test_build_realtime_safe_space_from_env_reconstructs_and_fits_obbs():
    class _FakeLiberoPc:
        def __init__(self):
            self.rendered_cameras = []

        def render_rgbd(self, _sim, camera_name, width, height):
            self.rendered_cameras.append((camera_name, width, height))
            return np.zeros((height, width, 3), dtype=np.uint8), np.ones((height, width), dtype=np.float32)

        def robot_pixel_mask(self, **kwargs):
            return np.zeros((kwargs["height"], kwargs["width"]), dtype=bool)

        def depth_to_world_points(self, **_kwargs):
            return (
                np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.02],
                        [0.2, 0.0, 0.12],
                        [0.2, 0.1, 0.12],
                    ],
                    dtype=np.float32,
                ),
                np.zeros((4, 3), dtype=np.uint8),
            )

        def crop_workspace(self, points, colors, bounds):
            return points, colors

    class _FakeSafeBuilder:
        def estimate_table_z(self, points, voxel_size):
            assert points.shape[1] == 3
            assert voxel_size == pytest.approx(0.04)
            return 0.0

        def estimate_table_workspace_bounds(self, **_kwargs):
            return np.asarray([-0.5, 0.5, -0.5, 0.5, 0.0, 0.5], dtype=np.float32), 12

        def tabletop_obstacle_points(self, points, bounds, table_z, min_height, max_height):
            assert table_z == 0.0
            keep = (points[:, 2] >= table_z + min_height) & (points[:, 2] <= table_z + max_height)
            return points[keep]

        def component_boxes_from_tabletop_points(self, **_kwargs):
            return (
                np.asarray([[0.15, -0.05, 0.0]], dtype=np.float32),
                np.asarray([[0.25, 0.05, 0.2]], dtype=np.float32),
                np.asarray([[0.2, 0.0, 0.1]], dtype=np.float32),
                np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                np.asarray([[0.05, 0.05, 0.1]], dtype=np.float32),
                np.zeros((1, 8, 3), dtype=np.float32),
                np.asarray([4], dtype=np.int64),
            )

    fake_pc = _FakeLiberoPc()

    safe_space = build_realtime_safe_space_from_env(
        env=types.SimpleNamespace(sim=object()),
        libero_pc=fake_pc,
        safe_space_builder=_FakeSafeBuilder(),
        camera_names=("frontview", "sideview"),
        width=16,
        height=12,
        stride=2,
        max_depth=4.0,
        robot_geom_ids=np.asarray([1, 2], dtype=np.int64),
        robot_mask_dilation=2,
        workspace_bounds=None,
        workspace_mode="table",
        workspace_margin=0.02,
        table_z=None,
        table_slab_height=0.02,
        table_obstacle_min_height=0.03,
        table_obstacle_max_height=0.3,
        component_voxel_size=0.02,
        component_connectivity=6,
        min_component_points=2,
        box_margin=0.01,
        box_shape="cuboid",
        box_orientation="xy_oriented",
        voxel_size=0.04,
    )

    assert fake_pc.rendered_cameras == [("frontview", 16, 12), ("sideview", 16, 12)]
    assert safe_space["obstacle_box_centers"].shape == (1, 3)
    assert safe_space["obstacle_box_axes"].shape == (1, 3, 3)
    assert safe_space["obstacle_box_half_sizes"].shape == (1, 3)
    assert safe_space["obstacle_box_corners"].shape == (1, 8, 3)
    assert safe_space["obstacle_box_point_counts"].tolist() == [4]


def test_build_realtime_safe_space_from_env_can_keep_only_named_obstacle_geoms():
    class _FakeModel:
        ngeom = 2
        geom_bodyid = np.asarray([0, 1], dtype=np.int32)

        def geom_id2name(self, geom_id):
            return ["eval_scene_obstacle_1_collision", "akita_black_bowl_1_collision"][int(geom_id)]

        def body_id2name(self, body_id):
            return ["eval_scene_obstacle_1", "akita_black_bowl_1"][int(body_id)]

    class _FakeLiberoPc:
        def __init__(self):
            self.keep_mask = None
            self.keep_masks = []

        def render_rgbd(self, _sim, _camera_name, width, height):
            return np.zeros((height, width, 3), dtype=np.uint8), np.ones((height, width), dtype=np.float32)

        def render_segmentation(self, _sim, _camera_name, width, height):
            segmentation = np.zeros((height, width, 2), dtype=np.int32)
            segmentation[..., 0] = 7
            segmentation[:, : width // 2, 1] = 0
            segmentation[:, width // 2 :, 1] = 1
            return segmentation

        def mujoco_geom_objtype(self):
            return 7

        def robot_pixel_mask(self, **kwargs):
            return np.zeros((kwargs["height"], kwargs["width"]), dtype=bool)

        def depth_to_world_points(self, **kwargs):
            self.keep_mask = np.asarray(kwargs["keep_mask"], dtype=bool)
            self.keep_masks.append(self.keep_mask.copy())
            pixel_points = np.asarray(
                [
                    [0.2, 0.0, 0.12],
                    [0.22, 0.0, 0.13],
                    [0.7, 0.0, 0.12],
                    [0.72, 0.0, 0.13],
                    [0.21, 0.02, 0.12],
                    [0.23, 0.02, 0.13],
                    [0.71, 0.02, 0.12],
                    [0.73, 0.02, 0.13],
                ],
                dtype=np.float32,
            )
            selected = self.keep_mask.reshape(-1)
            return (
                pixel_points[selected],
                np.zeros((int(selected.sum()), 3), dtype=np.uint8),
            )

        def crop_workspace(self, points, colors, bounds):
            return points, colors

    class _FakeSafeBuilder:
        def __init__(self):
            self.tabletop_points = None

        def estimate_table_z(self, _points, _voxel_size):
            return 0.0

        def estimate_table_workspace_bounds(self, **_kwargs):
            return np.asarray([-0.5, 0.5, -0.5, 0.5, 0.0, 0.5], dtype=np.float32), 3

        def tabletop_obstacle_points(self, points, **_kwargs):
            self.tabletop_points = np.asarray(points, dtype=np.float32)
            return self.tabletop_points

        def component_boxes_from_tabletop_points(self, points, **_kwargs):
            return (
                np.min(points, axis=0, keepdims=True),
                np.max(points, axis=0, keepdims=True),
                np.mean(points, axis=0, keepdims=True),
                np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                np.ptp(points, axis=0, keepdims=True) / 2.0,
                np.zeros((1, 8, 3), dtype=np.float32),
                np.asarray([len(points)], dtype=np.int64),
            )

    fake_pc = _FakeLiberoPc()
    fake_builder = _FakeSafeBuilder()

    safe_space = build_realtime_safe_space_from_env(
        env=types.SimpleNamespace(sim=types.SimpleNamespace(model=_FakeModel())),
        libero_pc=fake_pc,
        safe_space_builder=fake_builder,
        camera_names=("frontview",),
        width=4,
        height=2,
        stride=1,
        max_depth=4.0,
        robot_geom_ids=np.asarray([], dtype=np.int64),
        robot_mask_dilation=2,
        workspace_bounds=None,
        workspace_mode="table",
        workspace_margin=0.02,
        table_z=None,
        table_slab_height=0.02,
        table_obstacle_min_height=0.03,
        table_obstacle_max_height=0.3,
        component_voxel_size=0.02,
        component_connectivity=6,
        min_component_points=1,
        box_margin=0.01,
        box_shape="cuboid",
        box_orientation="xy_oriented",
        voxel_size=0.04,
        target_geom_name_patterns=("eval_scene_obstacle", "wine_bottle", "winebottle"),
    )

    assert len(fake_pc.keep_masks) == 2
    assert fake_pc.keep_masks[0].all()
    assert fake_pc.keep_masks[1].tolist() == [[True, True, False, False], [True, True, False, False]]
    np.testing.assert_allclose(fake_builder.tabletop_points[:, 0], [0.2, 0.22, 0.21, 0.23])
    assert safe_space["obstacle_box_point_counts"].tolist() == [4]


def test_build_realtime_safe_space_from_env_estimates_table_from_scene_when_target_filtered():
    class _FakeModel:
        ngeom = 2
        geom_bodyid = np.asarray([0, 1], dtype=np.int32)

        def geom_id2name(self, geom_id):
            return ["eval_scene_obstacle_1_collision", "main_table_collision"][int(geom_id)]

        def body_id2name(self, body_id):
            return ["eval_scene_obstacle_1", "main_table"][int(body_id)]

    class _FakeLiberoPc:
        def __init__(self):
            self.keep_masks = []

        def render_rgbd(self, _sim, _camera_name, width, height):
            return np.zeros((height, width, 3), dtype=np.uint8), np.ones((height, width), dtype=np.float32)

        def render_segmentation(self, _sim, _camera_name, width, height):
            segmentation = np.zeros((height, width, 2), dtype=np.int32)
            segmentation[..., 0] = 7
            segmentation[:, : width // 2, 1] = 0
            segmentation[:, width // 2 :, 1] = 1
            return segmentation

        def mujoco_geom_objtype(self):
            return 7

        def robot_pixel_mask(self, **kwargs):
            return np.zeros((kwargs["height"], kwargs["width"]), dtype=bool)

        def depth_to_world_points(self, **kwargs):
            keep_mask = np.asarray(kwargs["keep_mask"], dtype=bool)
            self.keep_masks.append(keep_mask.copy())
            pixel_points = np.asarray(
                [
                    [0.20, 0.00, 0.04],
                    [0.20, 0.02, 0.22],
                    [0.00, 0.00, 0.00],
                    [0.10, 0.00, 0.00],
                    [0.22, 0.00, 0.05],
                    [0.22, 0.02, 0.20],
                    [0.00, 0.10, 0.00],
                    [0.10, 0.10, 0.00],
                ],
                dtype=np.float32,
            )
            selected = keep_mask.reshape(-1)
            return pixel_points[selected], np.zeros((int(selected.sum()), 3), dtype=np.uint8)

        def crop_workspace(self, points, colors, bounds):
            return points, colors

    class _FakeSafeBuilder:
        def __init__(self):
            self.tabletop_points = None

        def estimate_table_z(self, points, _voxel_size):
            assert float(points[:, 2].min()) == pytest.approx(0.0)
            return 0.0

        def estimate_table_workspace_bounds(self, **kwargs):
            assert float(kwargs["points"][:, 2].min()) == pytest.approx(0.0)
            return np.asarray([-0.5, 0.5, -0.5, 0.5, 0.0, 0.5], dtype=np.float32), 4

        def tabletop_obstacle_points(self, points, **kwargs):
            self.tabletop_points = np.asarray(points, dtype=np.float32)
            table_z = float(kwargs["table_z"])
            min_height = float(kwargs["min_height"])
            max_height = float(kwargs["max_height"])
            keep = (points[:, 2] >= table_z + min_height) & (points[:, 2] <= table_z + max_height)
            return points[keep]

        def component_boxes_from_tabletop_points(self, points, **_kwargs):
            return (
                np.min(points, axis=0, keepdims=True),
                np.max(points, axis=0, keepdims=True),
                np.mean(points, axis=0, keepdims=True),
                np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                np.ptp(points, axis=0, keepdims=True) / 2.0,
                np.zeros((1, 8, 3), dtype=np.float32),
                np.asarray([len(points)], dtype=np.int64),
            )

    fake_pc = _FakeLiberoPc()
    fake_builder = _FakeSafeBuilder()

    safe_space = build_realtime_safe_space_from_env(
        env=types.SimpleNamespace(sim=types.SimpleNamespace(model=_FakeModel())),
        libero_pc=fake_pc,
        safe_space_builder=fake_builder,
        camera_names=("frontview",),
        width=4,
        height=2,
        stride=1,
        max_depth=4.0,
        robot_geom_ids=np.asarray([], dtype=np.int64),
        robot_mask_dilation=2,
        workspace_bounds=None,
        workspace_mode="table",
        workspace_margin=0.02,
        table_z=None,
        table_slab_height=0.02,
        table_obstacle_min_height=0.02,
        table_obstacle_max_height=0.3,
        component_voxel_size=0.02,
        component_connectivity=6,
        min_component_points=1,
        box_margin=0.01,
        box_shape="cuboid",
        box_orientation="xy_oriented",
        voxel_size=0.04,
        target_geom_name_patterns=("eval_scene_obstacle",),
    )

    assert len(fake_pc.keep_masks) == 2
    assert fake_pc.keep_masks[0].all()
    assert fake_pc.keep_masks[1].tolist() == [[True, True, False, False], [True, True, False, False]]
    assert fake_builder.tabletop_points.shape == (4, 3)
    assert safe_space["obstacle_box_point_counts"].tolist() == [4]


def test_real_robot_obstacle_collision_detects_robot_target_contact_only():
    class _FakeModel:
        ngeom = 3
        geom_bodyid = np.asarray([0, 1, 2], dtype=np.int32)

        def geom_id2name(self, geom_id):
            return ["robot0_link0_collision", "eval_scene_obstacle_1_collision", "akita_black_bowl_1_collision"][
                int(geom_id)
            ]

        def body_id2name(self, body_id):
            return ["robot0_link0", "eval_scene_obstacle_1", "akita_black_bowl_1"][int(body_id)]

    class _Contact:
        def __init__(self, geom1, geom2):
            self.geom1 = geom1
            self.geom2 = geom2

    model = _FakeModel()
    env = types.SimpleNamespace(
        sim=types.SimpleNamespace(
            model=model,
            data=types.SimpleNamespace(ncon=2, contact=[_Contact(0, 2), _Contact(1, 0)]),
        )
    )

    result = evaluator.real_robot_obstacle_collision(
        env,
        robot_geom_ids=np.asarray([0], dtype=np.int64),
        target_geom_name_patterns=("eval_scene_obstacle", "wine_bottle", "winebottle"),
    )

    assert result["collision"] is True
    assert result["contact_count"] == 1
    assert result["contact_geom_pairs"].tolist() == [[1, 0]]


def test_build_realtime_safe_space_from_env_skips_missing_cameras(capsys):
    class _FakeLiberoPc:
        def __init__(self):
            self.rendered_cameras = []

        def render_rgbd(self, _sim, camera_name, width, height):
            if camera_name == "missingview":
                raise ValueError('No "camera" with name missingview exists')
            self.rendered_cameras.append((camera_name, width, height))
            return np.zeros((height, width, 3), dtype=np.uint8), np.ones((height, width), dtype=np.float32)

        def robot_pixel_mask(self, **kwargs):
            return np.zeros((kwargs["height"], kwargs["width"]), dtype=bool)

        def depth_to_world_points(self, **_kwargs):
            return (
                np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.12], [0.2, 0.1, 0.12]], dtype=np.float32),
                np.zeros((3, 3), dtype=np.uint8),
            )

        def crop_workspace(self, points, colors, bounds):
            return points, colors

    class _FakeSafeBuilder:
        def estimate_table_z(self, _points, _voxel_size):
            return 0.0

        def estimate_table_workspace_bounds(self, **_kwargs):
            return np.asarray([-0.5, 0.5, -0.5, 0.5, 0.0, 0.5], dtype=np.float32), 3

        def tabletop_obstacle_points(self, points, **_kwargs):
            return points[points[:, 2] > 0.03]

        def component_boxes_from_tabletop_points(self, **_kwargs):
            return (
                np.zeros((1, 3), dtype=np.float32),
                np.ones((1, 3), dtype=np.float32),
                np.asarray([[0.2, 0.05, 0.1]], dtype=np.float32),
                np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                np.asarray([[0.1, 0.1, 0.1]], dtype=np.float32),
                np.zeros((1, 8, 3), dtype=np.float32),
                np.asarray([2], dtype=np.int64),
            )

    fake_pc = _FakeLiberoPc()

    safe_space = build_realtime_safe_space_from_env(
        env=types.SimpleNamespace(sim=object()),
        libero_pc=fake_pc,
        safe_space_builder=_FakeSafeBuilder(),
        camera_names=("frontview", "missingview"),
        width=16,
        height=12,
        stride=2,
        max_depth=4.0,
        robot_geom_ids=np.asarray([], dtype=np.int64),
        robot_mask_dilation=2,
        workspace_bounds=None,
        workspace_mode="table",
        workspace_margin=0.02,
        table_z=None,
        table_slab_height=0.02,
        table_obstacle_min_height=0.03,
        table_obstacle_max_height=0.3,
        component_voxel_size=0.02,
        component_connectivity=6,
        min_component_points=2,
        box_margin=0.01,
        box_shape="cuboid",
        box_orientation="xy_oriented",
        voxel_size=0.04,
    )

    captured = capsys.readouterr()
    assert fake_pc.rendered_cameras == [("frontview", 16, 12)]
    assert "skipped realtime OBB camera 'missingview'" in captured.out
    assert safe_space["obstacle_box_centers"].shape == (1, 3)


def test_point_flow_obb_collision_flags_future_point_inside_obb():
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "obstacle_box_half_sizes": np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        "obstacle_box_corners": np.zeros((1, 8, 3), dtype=np.float32),
    }
    pred = np.asarray(
        [
            [[[2.0, 0.0, 0.0]]],
            [[[0.25, 0.0, 0.0]]],
        ],
        dtype=np.float32,
    )

    result = point_flow_obb_collision(pred, safe_space)

    assert result["collision"] is True
    assert result["collision_point_count"] == 1
    np.testing.assert_array_equal(result["collision_point_indices"], [1])


def test_point_flow_obb_collision_checks_every_obb():
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.stack([np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]),
        "obstacle_box_half_sizes": np.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32),
        "obstacle_box_corners": np.zeros((2, 8, 3), dtype=np.float32),
    }
    pred = np.asarray([[[[3.25, 0.0, 0.0]]]], dtype=np.float32)

    result = point_flow_obb_collision(pred, safe_space)

    assert result["collision"] is True
    assert result["collision_point_count"] == 1
    np.testing.assert_array_equal(result["collision_point_indices"], [0])


def test_point_flow_obb_collision_rejects_mismatched_obb_arrays():
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        "obstacle_box_axes": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        "obstacle_box_half_sizes": np.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32),
        "obstacle_box_corners": np.zeros((2, 8, 3), dtype=np.float32),
    }
    pred = np.asarray([[[[3.25, 0.0, 0.0]]]], dtype=np.float32)

    with pytest.raises(ValueError, match="same number of OBBs"):
        point_flow_obb_collision(pred, safe_space)


def test_draw_projected_obbs_draws_box_edges():
    class _FakeSwept:
        def project_world_points_to_camera_pixels(self, _sim, _camera_name, _width, _height, points):
            uv = np.asarray(points[:, :2], dtype=np.float64)
            valid = np.ones((points.shape[0],), dtype=bool)
            return uv, valid

    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    corners = np.asarray(
        [
            [
                [2.0, 2.0, 0.0],
                [8.0, 2.0, 0.0],
                [8.0, 8.0, 0.0],
                [2.0, 8.0, 0.0],
                [2.0, 2.0, 1.0],
                [8.0, 2.0, 1.0],
                [8.0, 8.0, 1.0],
                [2.0, 8.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )

    drawn = draw_projected_obbs(
        frame,
        sim=object(),
        swept=_FakeSwept(),
        camera_name="agentview",
        width=12,
        height=12,
        obb_corners=corners,
    )

    assert int(drawn[:, :, 0].max()) > 0


def test_annotate_video_frame_draws_collision_status_bar():
    frame = np.zeros((48, 160, 3), dtype=np.uint8)

    annotated = annotate_video_frame(
        frame,
        rollout_id=0,
        step_id=3,
        sample_id=1,
        collision_result={
            "collision": True,
            "collision_point_count": 4,
            "collision_point_indices": np.asarray([0, 1, 2, 3], dtype=np.int64),
        },
    )

    bottom_band = annotated[-24:, :, :]
    red_status_pixels = (bottom_band[:, :, 0] >= 180) & (bottom_band[:, :, 1] < 80)
    assert int(red_status_pixels.sum()) > 0


def test_annotate_video_frame_draws_real_collision_status_bar():
    frame = np.zeros((64, 180, 3), dtype=np.uint8)

    annotated = annotate_video_frame(
        frame,
        rollout_id=0,
        step_id=3,
        sample_id=1,
        collision_result={
            "collision": True,
            "collision_point_count": 4,
            "collision_point_indices": np.asarray([0, 1, 2, 3], dtype=np.int64),
        },
        real_collision_result={
            "collision": True,
            "contact_count": 2,
            "contact_geom_pairs": np.asarray([[0, 1], [2, 1]], dtype=np.int64),
        },
    )

    real_collision_band = annotated[-44:-20, :, :]
    orange_pixels = (
        (real_collision_band[:, :, 0] >= 200)
        & (real_collision_band[:, :, 1] >= 80)
        & (real_collision_band[:, :, 1] <= 150)
        & (real_collision_band[:, :, 2] < 40)
    )
    assert int(orange_pixels.sum()) > 0


def test_save_evaluation_writes_predictions_targets_and_metrics(tmp_path: Path):
    output = tmp_path / "eval.npz"
    pred = np.zeros((2, 1, 3, 2, 3), dtype=np.float32)
    target = np.ones((2, 1, 3, 2, 3), dtype=np.float32)
    metrics = {
        "sample_mse": np.asarray([1.0, 2.0], dtype=np.float32),
        "sample_mean_l2": np.asarray([3.0, 4.0], dtype=np.float32),
        "sample_max_l2": np.asarray([5.0, 6.0], dtype=np.float32),
        "mean_mse": np.asarray(1.5, dtype=np.float32),
        "mean_l2": np.asarray(3.5, dtype=np.float32),
        "max_l2": np.asarray(6.0, dtype=np.float32),
    }

    save_evaluation(
        output,
        pred_link_points=pred,
        target_link_points=target,
        prefix_tokens_shape=np.asarray([2, 5, 4], dtype=np.int64),
        action_chunks=np.zeros((2, 10, 7), dtype=np.float32),
        metrics=metrics,
        rollout_ids=np.asarray([0, 0], dtype=np.int64),
        step_ids=np.asarray([0, 5], dtype=np.int64),
        link_names=np.asarray(["a", "b", "c"]),
        coordinate_frame="mujoco_world",
    )

    with np.load(output, allow_pickle=False) as data:
        assert data["pred_link_points"].shape == pred.shape
        assert data["target_link_points"].shape == target.shape
        assert str(data["coordinate_frame"]) == "mujoco_world"
        assert float(data["mean_l2"]) == 3.5


def test_append_prediction_video_frame_renders_overlay(monkeypatch):
    class _FakeSwept:
        def render_camera_rgb(self, _sim, _camera_name, width, height):
            return np.zeros((height, width, 3), dtype=np.uint8)

        def projected_point_image(self, _sim, _camera_name, width, height, _points, _colors, _point_radius, background):
            image = np.asarray(background, dtype=np.uint8).copy()
            image[:, :, 0] = 255
            return image

        def point_colors(self, link_ids):
            return np.zeros((len(link_ids), 3), dtype=np.uint8)

    class _FakeEnv:
        sim = object()

    buffer = VideoFrameBuffer(enabled=True)
    append_prediction_video_frame(
        buffer,
        env=_FakeEnv(),
        swept=_FakeSwept(),
        pred_link_points=np.zeros((2, 3, 4, 3), dtype=np.float32),
        camera_name="agentview",
        width=8,
        height=6,
        point_radius=2,
        rollout_id=0,
        step_id=3,
        sample_id=1,
    )

    assert len(buffer.frames) == 1
    assert buffer.frames[0].shape == (6, 8, 3)
    assert int(buffer.frames[0][:, :, 0].max()) == 255


def test_evaluate_online_video_recomputes_future_pointflow_each_env_step(monkeypatch, tmp_path: Path):
    class _FakePolicy:
        pass

    class _FakeTaskSuite:
        def get_task(self, _task_id):
            return types.SimpleNamespace()

        def get_task_init_states(self, _task_id):
            return [np.zeros((1,), dtype=np.float32)]

    class _FakeData:
        qpos = np.zeros((1,), dtype=np.float32)

    class _FakeSim:
        data = _FakeData()

    class _FakeEnv:
        action_dim = 7

        def __init__(self):
            self.sim = _FakeSim()
            self.step_count = 0

        def reset(self):
            self.step_count = 0

        def set_init_state(self, _state):
            return {"step": self.step_count}

        def step(self, _action):
            self.step_count += 1
            return {"step": self.step_count}, 0.0, self.step_count >= 3, {}

        def close(self):
            pass

    class _FakeSwept:
        def load_runtime_dependencies(self):
            pass

        def get_arm_qpos_indices(self, _env):
            return np.asarray([0], dtype=np.int64)

        def joint_limits(self, _sim, _indices):
            return np.asarray([-1.0]), np.asarray([1.0])

        def render_camera_rgb(self, _sim, _camera_name, width, height):
            return np.zeros((height, width, 3), dtype=np.uint8)

        def projected_point_image(self, _sim, _camera_name, width, height, points, _colors, _point_radius, background):
            frame = np.asarray(background, dtype=np.uint8).copy()
            frame[:, :, 0] = int(points[0, 0])
            return frame

        def point_colors(self, link_ids):
            return np.zeros((len(link_ids), 3), dtype=np.uint8)

    class _FakeLiberoPc:
        def find_robot_geoms(self, _env):
            return []

    class _FakeDatasetBuilder:
        def __init__(self):
            self.swept = _FakeSwept()
            self.libero_pc = _FakeLiberoPc()

        def import_script_module(self, name):
            if name == "libero_joint_swept_pointcloud":
                return self.swept
            if name == "libero_reconstruct_pointcloud":
                return self.libero_pc
            raise AssertionError(name)

        def fk_target_link_points(self, *_args, **_kwargs):
            return np.zeros((2, 1, 2, 3), dtype=np.float32), np.asarray(["link"])

    prefix_counter = {"value": 0}

    def fake_query_policy_action_and_prefix(_policy, _element, *, remote_prefix_tokens):
        prefix_counter["value"] += 1
        value = float(prefix_counter["value"])
        return np.ones((2, 7), dtype=np.float32), np.asarray([[value]], dtype=np.float32)

    def fake_predict_link_points(_model, prefix_tokens, _device):
        value = float(np.asarray(prefix_tokens)[0, 0])
        pred = np.zeros((2, 1, 2, 3), dtype=np.float32)
        pred[..., 0] = value
        return pred

    monkeypatch.setattr(evaluator.collector, "ensure_third_party_paths", lambda: None)
    monkeypatch.setattr(evaluator.collector, "load_repo_script_module", lambda _name: _FakeDatasetBuilder())
    monkeypatch.setattr(evaluator.collector, "load_remote_policy", lambda **_kwargs: _FakePolicy())
    monkeypatch.setattr(evaluator.collector, "create_libero_task_suite", lambda _name: _FakeTaskSuite())
    monkeypatch.setattr(evaluator.collector, "create_libero_env", lambda *_args, **_kwargs: (_FakeEnv(), "task"))
    monkeypatch.setattr(evaluator.collector, "make_dummy_action", lambda _env: np.zeros((7,), dtype=np.float32))
    monkeypatch.setattr(evaluator.collector, "robot_geom_ids_array", lambda ids: np.asarray(ids, dtype=np.int64))
    monkeypatch.setattr(
        evaluator.collector,
        "build_libero_policy_input",
        lambda obs, **_kwargs: {"step": obs["step"]},
    )
    monkeypatch.setattr(evaluator.collector, "query_policy_action_and_prefix", fake_query_policy_action_and_prefix)
    monkeypatch.setattr(evaluator.collector, "compute_fk_target_preserving_sim_state", lambda _env, builder: builder())
    monkeypatch.setattr(
        evaluator,
        "load_safety_model_checkpoint",
            lambda _path, _device: evaluator.LoadedSafetyModel(
                model_type="decoder",
                model=object(),
                config=types.SimpleNamespace(points_per_link=2),
            ),
        )
    monkeypatch.setattr(evaluator, "predict_link_points", fake_predict_link_points)

    args = types.SimpleNamespace(
        max_samples=3,
        num_rollouts=1,
        replan_steps=2,
        samples_per_action=1,
        points_per_link=2,
        prediction_steps=1,
        video_fps=12,
        mujoco_gl=None,
        seed=7,
        device="cpu",
        checkpoint=tmp_path / "model.pt",
        policy_server_host="127.0.0.1",
        policy_server_port=8000,
        task_suite="libero_spatial",
        task_id=0,
        max_steps=3,
        num_steps_wait=0,
        env_resolution=16,
        resize_size=16,
        skeleton_source="surface",
        no_video=False,
        video_output=tmp_path / "video.mp4",
        video_camera="agentview",
        video_width=4,
        video_height=3,
        video_point_radius=1,
        safe_space=None,
        realtime_obbs=False,
        collision_margin=0.0,
    )

    result = evaluator.evaluate_online(args)
    red_values = [int(frame[:, :, 0].max()) for frame in result["video_frames"]]

    assert red_values == [1, 2, 3]
