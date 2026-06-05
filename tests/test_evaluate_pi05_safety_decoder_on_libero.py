from pathlib import Path
import types
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch

import scripts.evaluate_pi05_safety_decoder_on_libero as evaluator
from scripts.collect_pi05_libero_safety_decoder_dataset import (
    SceneObstacleSpec,
    adapt_init_state_for_scene_obstacle,
    materialize_eval_scene_wine_bottle_xml,
    patch_bddl_with_scene_obstacle,
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
    assert "--skeleton-source" in result.stdout
    assert "--scene-obstacle" in result.stdout
    assert "--scene-obstacle-xy" in result.stdout


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

    assert "eval_scene_obstacle_region" in patched
    assert "eval_scene_obstacle_1 - eval_scene_wine_bottle_obstacle" in patched
    assert "(On eval_scene_obstacle_1 main_table_eval_scene_obstacle_region)" in patched
    assert "(-0.01 -0.01 0.01 0.01)" in patched


def test_materialize_eval_scene_wine_bottle_xml_scales_mesh_geoms_and_sites(tmp_path: Path):
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
    assert 'pos="0 0 -0.1"' in text


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


def test_adapt_init_state_for_scene_obstacle_pads_added_free_joint_state():
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
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            12.0,
            20.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            21.0,
            22.0,
        ],
    )


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
    assert constraint.h == pytest.approx(0.7)


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
