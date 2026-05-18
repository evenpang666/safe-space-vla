#!/usr/bin/env python3
"""Collect deterministic robot point-flow data in LIBERO.

This dataset is intended as an FK/simulator geometry source or as supervision
for a small residual model. It is not the preferred place to learn scene
dynamics; use the saved ``robot_point_flow`` together with scene point clouds as
input to a point-world model.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from libero_reconstruct_pointcloud import (  # noqa: E402
    REPO_ROOT,
    create_env,
    load_runtime_dependencies,
    resolve_task,
    settle_scene,
)
from libero_robot_swept_pointcloud import (  # noqa: E402
    disable_nonrobot_collisions,
    find_robot_geoms,
    get_env_action_dim,
    normalize_action_chunk,
    sample_robot_pointcloud,
    voxel_downsample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect (state, action_chunk, robot_point_flow) samples from LIBERO."
    )
    parser.add_argument(
        "--task-suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--bddl-file", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-scale", type=float, default=0.35)
    parser.add_argument("--gripper-action", type=float, default=-1.0)
    parser.add_argument(
        "--random-prefix-steps",
        type=int,
        default=0,
        help="Before each recorded chunk, execute this many random actions to diversify start states.",
    )
    parser.add_argument(
        "--reset-every",
        type=int,
        default=1,
        help="Reset to the LIBERO initial state every N samples. Use 1 for independent samples.",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--points-per-geom", type=int, default=80)
    parser.add_argument("--target-points", type=int, default=1024)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument(
        "--state-mode",
        choices=["openpi", "qpos", "openpi_qpos"],
        default="openpi",
        help="Input state stored with the robot point flow. openpi matches PI05 LIBERO's 8D state.",
    )
    parser.add_argument("--disable-nonrobot-collisions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "robot_point_flow" / "libero_robot_point_flow_dataset.npz",
    )
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default=None)
    return parser.parse_args()


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - float(quat[3] * quat[3]), 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def openpi_state_from_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        )
    ).astype(np.float32)


def qpos_state_from_obs(obs: dict) -> np.ndarray:
    if "robot0_joint_pos" not in obs:
        raise KeyError("LIBERO observation does not contain robot0_joint_pos")
    gripper = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float32).reshape(-1)
    return np.concatenate((np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1), gripper)).astype(
        np.float32
    )


def state_from_obs(obs: dict, mode: str) -> np.ndarray:
    openpi_state = openpi_state_from_obs(obs)
    if mode == "openpi":
        return openpi_state
    qpos_state = qpos_state_from_obs(obs)
    if mode == "qpos":
        return qpos_state
    return np.concatenate((openpi_state, qpos_state)).astype(np.float32)


def make_random_action_chunk(
    rng: np.random.Generator,
    horizon: int,
    action_dim: int,
    action_scale: float,
    gripper_action: float | None,
) -> np.ndarray:
    actions = rng.uniform(-action_scale, action_scale, size=(horizon, action_dim)).astype(np.float32)
    if action_dim > 0 and gripper_action is not None:
        actions[:, -1] = float(gripper_action)
    return actions


def step_env(env, action: np.ndarray) -> dict:
    result = env.step(action)
    return result[0] if isinstance(result, tuple) else result


def fixed_size_points(points: np.ndarray, target_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        raise RuntimeError("empty swept point cloud")
    replace = len(points) < target_points
    indices = rng.choice(len(points), size=target_points, replace=replace)
    return points[indices].astype(np.float32, copy=False)


def collect_one_sample(
    env,
    robot_geom_ids: list[int],
    args: argparse.Namespace,
    rng: np.random.Generator,
    env_action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = env.env._get_observations()

    for _ in range(args.random_prefix_steps):
        prefix_action = make_random_action_chunk(
            rng,
            1,
            args.action_dim,
            args.action_scale,
            args.gripper_action,
        )
        prefix_action = normalize_action_chunk(prefix_action, env_action_dim)[0]
        obs = step_env(env, prefix_action)

    state = state_from_obs(obs, args.state_mode)
    actions = make_random_action_chunk(
        rng,
        args.horizon,
        args.action_dim,
        args.action_scale,
        args.gripper_action,
    )
    actions = normalize_action_chunk(actions, env_action_dim)

    robot_point_flow = []
    all_points = []
    for action in actions:
        step_env(env, action)
        points = sample_robot_pointcloud(env.sim, robot_geom_ids, args.points_per_geom, rng)
        if len(points) > 0:
            all_points.append(points)
            robot_point_flow.append(fixed_size_points(points, args.target_points, rng))

    if not all_points:
        raise RuntimeError("no robot points sampled")
    points = np.concatenate(all_points, axis=0).astype(np.float32, copy=False)
    if args.voxel_size > 0.0:
        dummy_colors = np.zeros((len(points), 3), dtype=np.uint8)
        points, _ = voxel_downsample(points, dummy_colors, args.voxel_size)
    return (
        state,
        actions,
        np.stack(robot_point_flow).astype(np.float32),
        fixed_size_points(points, args.target_points, rng),
    )


def main() -> None:
    args = parse_args()
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.reset_every <= 0:
        raise ValueError("--reset-every must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    load_runtime_dependencies()
    bddl_file, task_name, init_state = resolve_task(args)
    env = create_env(bddl_file, width=64, height=64, camera_names=["agentview"])
    rng = np.random.default_rng(args.seed)

    try:
        settle_scene(env, init_state, args.num_steps_wait)
        env_action_dim = get_env_action_dim(env)
        robot_geom_ids = find_robot_geoms(env, None)
        if not robot_geom_ids:
            raise RuntimeError("No robot geoms were found in the LIBERO simulator.")
        if args.disable_nonrobot_collisions:
            disable_nonrobot_collisions(env, robot_geom_ids)

        states = []
        actions = []
        robot_point_flows = []
        points = []
        failures = 0
        for sample_idx in range(args.num_samples):
            if sample_idx % args.reset_every == 0:
                settle_scene(env, init_state, args.num_steps_wait)

            try:
                state_i, actions_i, robot_point_flow_i, points_i = collect_one_sample(
                    env, robot_geom_ids, args, rng, env_action_dim
                )
            except Exception as exc:  # keep long collection runs from dying on one unstable rollout
                failures += 1
                print(f"[warn] sample {sample_idx} failed: {exc}")
                settle_scene(env, init_state, args.num_steps_wait)
                continue

            states.append(state_i)
            actions.append(actions_i)
            robot_point_flows.append(robot_point_flow_i)
            points.append(points_i)
            if len(states) == 1 or len(states) % 100 == 0:
                print(f"[info] collected {len(states)}/{args.num_samples} samples")

        if not states:
            raise RuntimeError("No samples were collected.")

        np.savez_compressed(
            args.output,
            states=np.stack(states).astype(np.float32),
            actions=np.stack(actions).astype(np.float32),
            robot_point_flow=np.stack(robot_point_flows).astype(np.float32),
            robot_point_flow_note=np.asarray("shape (N, H, P, 3): deterministic robot geometry per future step"),
            points=np.stack(points).astype(np.float32),
            points_note=np.asarray("legacy union swept cloud, shape (N, P, 3)"),
            task_name=np.asarray(task_name),
            task_suite=np.asarray(args.task_suite),
            task_id=np.asarray(args.task_id, dtype=np.int64),
            init_state_id=np.asarray(args.init_state_id, dtype=np.int64),
            state_mode=np.asarray(args.state_mode),
            failed_samples=np.asarray(failures, dtype=np.int64),
        )
        print(f"[done] saved dataset: {args.output}")
        print(
            "[done] states/actions/robot_point_flow/points: "
            f"{states[0].shape}, {actions[0].shape}, {robot_point_flows[0].shape}, {points[0].shape}"
        )
        if failures:
            print(f"[done] skipped failed samples: {failures}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
