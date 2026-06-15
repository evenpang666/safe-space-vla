from pathlib import Path
import types

import numpy as np

from scripts import collect_pi05_libero_safety_decoder_dataset as collector
from scripts.collect_pi05_libero_safety_decoder_dataset import (
    CollectedSampleBuffer,
    ReplanSampleRecord,
    append_surface_trajectory_samples,
    build_task_worker_command,
    build_libero_policy_input,
    collectable_task_ids,
    compute_fk_target_preserving_sim_state,
    compute_rollout_surface_target_preserving_sim_state,
    collect_rollout_surface_target,
    is_valid_dataset_shard,
    load_openpi_policy,
    merge_dataset_shards,
    resolve_max_samples_per_task,
    resolve_task_shard_dir,
    resolve_task_ids,
    robot_geom_ids_array,
    run_task_worker_subprocesses,
    save_collected_dataset,
    should_isolate_task_processes,
    task_shard_output_path,
    surface_trajectory_target,
)


def test_parse_args_defaults_to_dense_link_points(monkeypatch):
    monkeypatch.setattr(collector.sys, "argv", ["collect_pi05_libero_safety_decoder_dataset.py"])

    args = collector.parse_args()

    assert args.points_per_link == 128
    assert args.task_ids is None
    assert args.max_samples_per_task is None
    assert args.isolate_task_processes
    assert not args.merge_after_collection
    assert not args.task_worker
    assert not args.skip_final_merge
    assert not hasattr(args, "skeleton_source")
    assert not hasattr(args, "target_source")


def test_parse_args_accepts_all_task_ids_and_per_task_limit(monkeypatch):
    monkeypatch.setattr(
        collector.sys,
        "argv",
        [
            "collect_pi05_libero_safety_decoder_dataset.py",
            "--task-ids",
            "all",
            "--max-samples-per-task",
            "32",
        ],
    )

    args = collector.parse_args()

    assert args.task_ids == ["all"]
    assert args.max_samples_per_task == 32


def test_parse_args_accepts_resume_and_merge_shard_options(monkeypatch, tmp_path: Path):
    shard_dir = tmp_path / "task_shards"
    monkeypatch.setattr(
        collector.sys,
        "argv",
        [
            "collect_pi05_libero_safety_decoder_dataset.py",
            "--per-task-output-dir",
            str(shard_dir),
            "--overwrite-task-shards",
            "--merge-task-shards-only",
            "--merge-after-collection",
        ],
    )

    args = collector.parse_args()

    assert args.per_task_output_dir == shard_dir
    assert args.overwrite_task_shards
    assert args.merge_task_shards_only
    assert args.merge_after_collection


def test_parse_args_accepts_disabling_task_process_isolation(monkeypatch):
    monkeypatch.setattr(
        collector.sys,
        "argv",
        [
            "collect_pi05_libero_safety_decoder_dataset.py",
            "--no-isolate-task-processes",
        ],
    )

    args = collector.parse_args()

    assert not args.isolate_task_processes


def test_should_isolate_task_processes_only_for_parent_multi_task_collection(monkeypatch):
    monkeypatch.setattr(collector.sys, "argv", ["collect_pi05_libero_safety_decoder_dataset.py"])
    args = collector.parse_args()

    assert should_isolate_task_processes(args, [0, 1])
    assert not should_isolate_task_processes(args, [0])

    args.merge_task_shards_only = True
    assert not should_isolate_task_processes(args, [0, 1])

    args.merge_task_shards_only = False
    args.task_worker = True
    assert not should_isolate_task_processes(args, [0, 1])

    args.task_worker = False
    args.isolate_task_processes = False
    assert not should_isolate_task_processes(args, [0, 1])


def test_build_task_worker_command_collects_one_task_without_recursive_isolation(monkeypatch, tmp_path: Path):
    output = tmp_path / "merged.npz"
    shard_dir = tmp_path / "shards"
    monkeypatch.setattr(
        collector.sys,
        "argv",
        [
            "collect_pi05_libero_safety_decoder_dataset.py",
            "--policy-config",
            "pi05_libero",
            "--checkpoint-dir",
            "checkpoint",
            "--task-suite",
            "libero_spatial",
            "--task-ids",
            "all",
            "--num-rollouts",
            "2",
            "--max-samples",
            "99",
            "--max-samples-per-task",
            "32",
            "--max-steps",
            "12",
            "--num-steps-wait",
            "3",
            "--replan-steps",
            "4",
            "--resize-size",
            "128",
            "--env-resolution",
            "64",
            "--points-per-link",
            "16",
            "--seed",
            "11",
            "--output",
            str(output),
            "--per-task-output-dir",
            str(shard_dir),
            "--mujoco-gl",
            "osmesa",
            "--policy-server-host",
            "127.0.0.1",
            "--policy-server-port",
            "9000",
            "--scene-obstacle",
            "wine_bottle",
            "--scene-obstacle-xy",
            "0.1",
            "0.2",
        ],
    )
    args = collector.parse_args()

    command = build_task_worker_command(args, task_id=3)

    assert command[0] == collector.sys.executable
    assert command[1] == str(collector.Path(collector.__file__).resolve())
    assert "--task-id" in command
    assert command[command.index("--task-id") + 1] == "3"
    assert "--task-ids" not in command
    assert "--task-worker" in command
    assert "--skip-final-merge" in command
    assert "--no-isolate-task-processes" in command
    assert "--policy-server-host" in command
    assert command[command.index("--policy-server-host") + 1] == "127.0.0.1"
    assert "--scene-obstacle-xy" in command


def test_run_task_worker_subprocesses_runs_each_worker_with_check(monkeypatch):
    monkeypatch.setattr(collector.sys, "argv", ["collect_pi05_libero_safety_decoder_dataset.py"])
    args = collector.parse_args()
    calls = []

    def fake_run(command, *, check):
        calls.append((command, check))

    monkeypatch.setattr(collector.subprocess, "run", fake_run)

    run_task_worker_subprocesses(args, [2, 4])

    assert [call[0][call[0].index("--task-id") + 1] for call in calls] == ["2", "4"]
    assert [call[1] for call in calls] == [True, True]


def test_resolve_task_ids_expands_all_and_validates_bounds():
    assert resolve_task_ids(task_id=0, task_ids=["all"], n_tasks=3) == [0, 1, 2]
    assert resolve_task_ids(task_id=0, task_ids=["2", "0"], n_tasks=3) == [2, 0]
    assert resolve_task_ids(task_id=1, task_ids=None, n_tasks=3) == [1]

    try:
        resolve_task_ids(task_id=0, task_ids=["all", "1"], n_tasks=3)
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("combined all and explicit task ids were accepted")

    try:
        resolve_task_ids(task_id=3, task_ids=None, n_tasks=3)
    except ValueError as exc:
        assert "must be in [0, 2]" in str(exc)
    else:
        raise AssertionError("out-of-range task id was accepted")


def test_resolve_max_samples_per_task_prefers_explicit_per_task_limit():
    assert resolve_max_samples_per_task(max_samples=512, max_samples_per_task=None) == 512
    assert resolve_max_samples_per_task(max_samples=512, max_samples_per_task=64) == 64

    try:
        resolve_max_samples_per_task(max_samples=512, max_samples_per_task=0)
    except ValueError as exc:
        assert "--max-samples-per-task must be > 0" in str(exc)
    else:
        raise AssertionError("non-positive per-task sample limit was accepted")


def test_task_shard_output_path_uses_stable_task_file_names(tmp_path: Path):
    output = tmp_path / "pi05_libero_decoder_dataset.npz"
    shard_dir = resolve_task_shard_dir(output=output, per_task_output_dir=None)

    assert shard_dir == tmp_path / "pi05_libero_decoder_dataset_tasks"
    assert task_shard_output_path(shard_dir=shard_dir, task_suite="libero_spatial", task_id=3) == (
        shard_dir / "libero_spatial_task003.npz"
    )


def test_collectable_task_ids_skips_existing_task_shards_unless_overwriting(tmp_path: Path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    buffer = CollectedSampleBuffer()
    buffer.append(
        prefix_tokens=np.ones((5, 8), dtype=np.float32),
        action_chunk=np.ones((10, 7), dtype=np.float32),
        start_joint_vector=np.zeros((7,), dtype=np.float32),
        target_link_points=np.zeros((11, 3, 4, 3), dtype=np.float32),
        task_id=1,
        rollout_id=0,
        step_id=0,
    )
    task_1_shard = task_shard_output_path(shard_dir=shard_dir, task_suite="libero_spatial", task_id=1)
    save_collected_dataset(
        task_1_shard,
        buffer=buffer,
        link_names=np.asarray(["link0", "link1", "link2"]),
        task_suite="libero_spatial",
        points_per_link=4,
        samples_per_action=1,
        policy_config="pi05_libero",
        checkpoint_dir="checkpoint",
    )

    assert collectable_task_ids(
        [0, 1, 2],
        shard_dir=shard_dir,
        task_suite="libero_spatial",
        overwrite_task_shards=False,
    ) == [0, 2]
    assert collectable_task_ids(
        [0, 1, 2],
        shard_dir=shard_dir,
        task_suite="libero_spatial",
        overwrite_task_shards=True,
    ) == [0, 1, 2]


def test_collectable_task_ids_recollects_corrupt_task_shards(tmp_path: Path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    corrupt_shard = task_shard_output_path(shard_dir=shard_dir, task_suite="libero_spatial", task_id=1)
    corrupt_shard.write_bytes(b"partial")

    assert not is_valid_dataset_shard(corrupt_shard)
    assert collectable_task_ids(
        [1],
        shard_dir=shard_dir,
        task_suite="libero_spatial",
        overwrite_task_shards=False,
    ) == [1]


def test_parse_args_rejects_removed_collection_modes(monkeypatch):
    monkeypatch.setattr(
        collector.sys,
        "argv",
        ["collect_pi05_libero_safety_decoder_dataset.py", "--target-source", "action_fk"],
    )

    try:
        collector.parse_args()
    except SystemExit:
        pass
    else:
        raise AssertionError("removed --target-source argument was accepted")


def test_parse_args_rejects_removed_sampling_modes(monkeypatch):
    monkeypatch.setattr(
        collector.sys,
        "argv",
        ["collect_pi05_libero_safety_decoder_dataset.py", "--skeleton-source", "surface"],
    )
    try:
        collector.parse_args()
    except SystemExit:
        pass
    else:
        raise AssertionError("removed --skeleton-source argument was accepted")

    monkeypatch.setattr(
        collector.sys,
        "argv",
        ["collect_pi05_libero_safety_decoder_dataset.py", "--samples-per-action", "2"],
    )
    try:
        collector.parse_args()
    except SystemExit:
        pass
    else:
        raise AssertionError("removed --samples-per-action argument was accepted")


def test_build_libero_policy_input_matches_pi05_libero_observation_schema():
    base = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    wrist = np.full((2, 3, 3), 7, dtype=np.uint8)
    obs = {
        "agentview_image": base,
        "robot0_eye_in_hand_image": wrist,
        "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.04, 0.05], dtype=np.float32),
    }

    element = build_libero_policy_input(
        obs,
        prompt="put the bowl on the plate",
        resize_size=224,
        image_resizer=lambda image, _size: image,
    )

    np.testing.assert_array_equal(element["observation/image"], base[::-1, ::-1])
    np.testing.assert_array_equal(element["observation/wrist_image"], wrist[::-1, ::-1])
    np.testing.assert_allclose(element["observation/state"], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.04, 0.05])
    assert element["prompt"] == "put the bowl on the plate"


def test_ensure_third_party_paths_prefers_openpi_repo_layout(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    openpi_src = repo_root / "openpi" / "src"
    openpi_client_src = repo_root / "openpi" / "packages" / "openpi-client" / "src"
    libero_root = repo_root / "openpi" / "third_party" / "libero"
    inner_libero_parent = libero_root / "libero"
    monkeypatch.setattr(collector.sys, "path", [str(inner_libero_parent), "/tmp/other"])

    collector.ensure_third_party_paths()

    assert str(inner_libero_parent) not in collector.sys.path
    assert str(openpi_src) in collector.sys.path
    assert str(openpi_client_src) in collector.sys.path
    assert str(libero_root) in collector.sys.path


def test_load_repo_script_module_ignores_openpi_scripts_package(monkeypatch):
    monkeypatch.setitem(collector.sys.modules, "scripts", types.ModuleType("scripts"))

    module = collector.load_repo_script_module("build_pi05_safety_decoder_dataset")

    assert Path(module.__file__).resolve() == (
        Path(__file__).resolve().parents[1] / "scripts" / "build_pi05_safety_decoder_dataset.py"
    ).resolve()
    assert hasattr(module, "fk_target_link_points")


def test_load_openpi_policy_rewrites_missing_jaxlib_error(monkeypatch):
    def fake_import(name, *args, **kwargs):
        if name == "openpi.policies":
            cause = ModuleNotFoundError("No module named 'jaxlib'")
            raise ModuleNotFoundError("jax requires jaxlib to be installed") from cause
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr(collector, "ensure_third_party_paths", lambda: None)
    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        load_openpi_policy(
            policy_config="pi05_libero",
            checkpoint_dir="gs://openpi-assets/checkpoints/pi05_libero",
            default_prompt=None,
            pytorch_device=None,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "jaxlib is missing" in message
        assert "--policy-server-host 127.0.0.1" in message
        assert "Python >=3.11" in message
    else:
        raise AssertionError("missing jaxlib error was not rewritten")


class _FakeRemotePolicy:
    def infer(self, _element):
        return {
            "actions": np.ones((3, 7), dtype=np.float32),
            "prefix_tokens": np.ones((5, 8), dtype=np.float32),
        }


class _FakeActionsOnlyPolicy:
    def infer(self, _element):
        return {"actions": np.ones((3, 7), dtype=np.float32)}


def test_query_policy_action_and_prefix_uses_remote_prefix_tokens_without_local_extractor():
    action_chunk, prefix_tokens = collector.query_policy_action_and_prefix(
        _FakeRemotePolicy(),
        {"prompt": "task"},
        remote_prefix_tokens=True,
        local_prefix_extractor=lambda _policy, _element: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    assert action_chunk.shape == (3, 7)
    assert prefix_tokens.shape == (5, 8)


def test_query_policy_action_and_prefix_requires_remote_prefix_tokens():
    try:
        collector.query_policy_action_and_prefix(
            _FakeActionsOnlyPolicy(),
            {"prompt": "task"},
            remote_prefix_tokens=True,
        )
    except KeyError as exc:
        assert "prefix_tokens" in str(exc)
    else:
        raise AssertionError("missing remote prefix_tokens was accepted")


def test_robot_geom_ids_array_expands_set_to_sorted_int64_array():
    geom_ids = robot_geom_ids_array({8, 3, 5})

    np.testing.assert_array_equal(geom_ids, np.asarray([3, 5, 8], dtype=np.int64))
    assert geom_ids.dtype == np.int64


def test_collected_sample_buffer_stacks_consistent_decoder_dataset_arrays(tmp_path: Path):
    buffer = CollectedSampleBuffer()
    prefix = np.zeros((5, 8), dtype=np.float32)
    action_chunk = np.zeros((10, 7), dtype=np.float32)
    start_joints = np.zeros((7,), dtype=np.float32)
    target = np.zeros((11, 3, 4, 3), dtype=np.float32)

    buffer.append(
        prefix_tokens=prefix,
        action_chunk=action_chunk,
        start_joint_vector=start_joints,
        target_link_points=target,
        task_id=2,
        rollout_id=0,
        step_id=12,
    )
    buffer.append(
        prefix_tokens=prefix + 1.0,
        action_chunk=action_chunk + 0.1,
        start_joint_vector=start_joints + 0.2,
        target_link_points=target + 0.3,
        task_id=2,
        rollout_id=0,
        step_id=17,
    )

    output = tmp_path / "dataset.npz"
    save_collected_dataset(
        output,
        buffer=buffer,
        link_names=np.asarray(["link0", "link1", "link2"]),
        task_suite="libero_spatial",
        points_per_link=4,
        samples_per_action=1,
        policy_config="pi05_libero",
        checkpoint_dir="checkpoint",
    )

    with np.load(output, allow_pickle=False) as data:
        assert data["prefix_tokens"].shape == (2, 5, 8)
        assert data["action_chunks"].shape == (2, 10, 7)
        assert data["start_joint_vectors"].shape == (2, 7)
        assert data["target_link_points"].shape == (2, 11, 3, 4, 3)
        assert data["current_link_points"].shape == (2, 3, 4, 3)
        assert data["future_link_offsets"].shape == (2, 10, 3, 4, 3)
        assert data["arm_points"].shape == (2, 12, 3)
        assert data["target_point_offsets"].shape == (2, 10, 12, 3)
        assert data["task_ids"].tolist() == [2, 2]
        assert data["rollout_ids"].tolist() == [0, 0]
        assert data["step_ids"].tolist() == [12, 17]
        assert data["link_names"].tolist() == ["link0", "link1", "link2"]
        assert str(data["skeleton_source"]) == "surface"
        assert str(data["target_source"]) == "rollout_surface"
        assert str(data["coordinate_frame"]) == "mujoco_world"
        assert str(data["target_link_points_frame"]) == "mujoco_world"
        assert str(data["arm_points_frame"]) == "mujoco_world"
        assert str(data["target_point_offsets_frame"]) == "mujoco_world_delta"
        assert str(data["policy_config"]) == "pi05_libero"


def test_merge_dataset_shards_concatenates_samples_and_preserves_metadata(tmp_path: Path):
    shard_paths = []
    for task_id in [0, 1]:
        buffer = CollectedSampleBuffer()
        prefix = np.full((5, 8), task_id, dtype=np.float32)
        action_chunk = np.full((10, 7), task_id + 0.1, dtype=np.float32)
        start_joints = np.full((7,), task_id + 0.2, dtype=np.float32)
        target = np.full((11, 3, 4, 3), task_id + 0.3, dtype=np.float32)
        buffer.append(
            prefix_tokens=prefix,
            action_chunk=action_chunk,
            start_joint_vector=start_joints,
            target_link_points=target,
            task_id=task_id,
            rollout_id=0,
            step_id=task_id * 10,
        )
        shard_path = tmp_path / f"task{task_id}.npz"
        save_collected_dataset(
            shard_path,
            buffer=buffer,
            link_names=np.asarray(["link0", "link1", "link2"]),
            task_suite="libero_spatial",
            points_per_link=4,
            samples_per_action=1,
            policy_config="pi05_libero",
            checkpoint_dir="checkpoint",
        )
        shard_paths.append(shard_path)

    output = tmp_path / "merged.npz"
    merged_count = merge_dataset_shards(output, shard_paths)

    assert merged_count == 2
    with np.load(output, allow_pickle=False) as data:
        assert data["prefix_tokens"].shape == (2, 5, 8)
        assert data["task_ids"].tolist() == [0, 1]
        assert data["step_ids"].tolist() == [0, 10]
        assert data["link_names"].tolist() == ["link0", "link1", "link2"]
        assert str(data["task_suite"]) == "libero_spatial"


def test_collected_sample_buffer_rejects_shape_changes():
    buffer = CollectedSampleBuffer()
    buffer.append(
        prefix_tokens=np.zeros((5, 8), dtype=np.float32),
        action_chunk=np.zeros((10, 7), dtype=np.float32),
        start_joint_vector=np.zeros((7,), dtype=np.float32),
        target_link_points=np.zeros((11, 3, 4, 3), dtype=np.float32),
        task_id=0,
        rollout_id=0,
        step_id=0,
    )

    try:
        buffer.append(
            prefix_tokens=np.zeros((6, 8), dtype=np.float32),
            action_chunk=np.zeros((10, 7), dtype=np.float32),
            start_joint_vector=np.zeros((7,), dtype=np.float32),
            target_link_points=np.zeros((11, 3, 4, 3), dtype=np.float32),
            task_id=0,
            rollout_id=0,
            step_id=1,
        )
    except ValueError as exc:
        assert "prefix_tokens shape changed" in str(exc)
    else:
        raise AssertionError("shape change was accepted")


class _FakeData:
    def __init__(self):
        self.qpos = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        self.qvel = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)


class _FakeSim:
    def __init__(self):
        self.data = _FakeData()
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class _FakeEnv:
    def __init__(self):
        self.sim = _FakeSim()


def test_compute_fk_target_preserving_sim_state_restores_qpos_qvel():
    env = _FakeEnv()

    def mutating_target_builder():
        env.sim.data.qpos[:] = 9.0
        env.sim.data.qvel[:] = 8.0
        return np.zeros((2, 1, 3, 3), dtype=np.float32), np.asarray(["link0"])

    target, link_names = compute_fk_target_preserving_sim_state(env, mutating_target_builder)

    assert target.shape == (2, 1, 3, 3)
    assert link_names.tolist() == ["link0"]
    np.testing.assert_allclose(env.sim.data.qpos, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(env.sim.data.qvel, [0.1, 0.2, 0.3])
    assert env.sim.forward_calls == 1


class _RolloutFakeEnv(_FakeEnv):
    action_dim = 2

    def __init__(self):
        super().__init__()
        self.actions = []
        self.step_count = 0

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.actions.append(action.copy())
        self.sim.data.qpos[:2] += action[:2]
        self.step_count += 1
        return {"step": self.step_count}, 0.0, False, {}


def test_collect_rollout_surface_target_uses_complete_action_chunk():
    env = _RolloutFakeEnv()
    action_chunk = np.asarray([[1.0, 0.0, 9.0], [0.0, 2.0, 9.0], [3.0, 4.0, 9.0]], dtype=np.float32)

    target, link_names = collect_rollout_surface_target(
        env,
        action_chunk,
        surface_snapshot=lambda: env.sim.data.qpos[:2].reshape(1, 2, 1),
        link_names=np.asarray(["surface"]),
    )

    assert target.shape == (4, 1, 2, 1)
    np.testing.assert_allclose(target[:, 0, :, 0], [[1.0, 2.0], [2.0, 2.0], [2.0, 4.0], [5.0, 8.0]])
    assert link_names.tolist() == ["surface"]
    assert len(env.actions) == 3
    np.testing.assert_allclose(env.actions[0], [1.0, 0.0])


def test_compute_rollout_surface_target_preserving_sim_state_restores_after_lookahead():
    env = _RolloutFakeEnv()
    original_qpos = env.sim.data.qpos.copy()
    action_chunk = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    target, _link_names = compute_rollout_surface_target_preserving_sim_state(
        env,
        action_chunk,
        surface_snapshot=lambda: env.sim.data.qpos[:2].reshape(1, 2, 1),
        link_names=np.asarray(["surface"]),
    )

    assert target.shape[0] == 3
    np.testing.assert_allclose(env.sim.data.qpos, original_qpos)
    assert env.sim.forward_calls == 1


def test_surface_trajectory_target_uses_recorded_future_frames():
    surface_frames = np.arange(6, dtype=np.float32).reshape(6, 1, 1, 1) + np.zeros((6, 1, 1, 3), dtype=np.float32)

    target = surface_trajectory_target(surface_frames, start_step=1, horizon=3)

    assert target.shape == (4, 1, 1, 3)
    np.testing.assert_allclose(target[:, 0, 0, 0], [1.0, 2.0, 3.0, 4.0])


def test_surface_trajectory_target_rejects_incomplete_future_horizon():
    surface_frames = np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1) + np.zeros((3, 1, 1, 3), dtype=np.float32)

    try:
        surface_trajectory_target(surface_frames, start_step=1, horizon=4)
    except ValueError as exc:
        assert "complete future horizon" in str(exc)
    else:
        raise AssertionError("incomplete future horizon was accepted")


def test_append_surface_trajectory_samples_drops_tail_records_without_complete_future():
    buffer = CollectedSampleBuffer()
    surface_frames = np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1) + np.zeros((5, 1, 1, 3), dtype=np.float32)
    records = [
        ReplanSampleRecord(
            prefix_tokens=np.full((5, 8), step, dtype=np.float32),
            action_chunk=np.ones((2, 7), dtype=np.float32),
            start_joint_vector=np.zeros((7,), dtype=np.float32),
            task_id=2,
            rollout_id=0,
            step_id=step,
        )
        for step in range(5)
    ]

    append_surface_trajectory_samples(
        buffer,
        records=records,
        surface_frames=surface_frames,
        link_names=np.asarray(["surface"]),
        max_samples=10,
    )

    assert len(buffer) == 3
    assert buffer.step_ids == [0, 1, 2]
    np.testing.assert_allclose(buffer.target_link_points[-1][:, 0, 0, 0], [2.0, 3.0, 4.0])


def test_append_surface_trajectory_samples_builds_targets_after_complete_rollout():
    buffer = CollectedSampleBuffer()
    surface_frames = np.arange(6, dtype=np.float32).reshape(6, 1, 1, 1) + np.zeros((6, 1, 1, 3), dtype=np.float32)
    record = ReplanSampleRecord(
        prefix_tokens=np.ones((5, 8), dtype=np.float32),
        action_chunk=np.ones((3, 7), dtype=np.float32),
        start_joint_vector=np.zeros((7,), dtype=np.float32),
        task_id=2,
        rollout_id=0,
        step_id=1,
    )

    append_surface_trajectory_samples(
        buffer,
        records=[record],
        surface_frames=surface_frames,
        link_names=np.asarray(["surface"]),
        max_samples=10,
    )

    assert len(buffer) == 1
    np.testing.assert_allclose(buffer.target_link_points[0][:, 0, 0, 0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(buffer.target_point_offsets[0][:, 0, 0], [1.0, 2.0, 3.0])
