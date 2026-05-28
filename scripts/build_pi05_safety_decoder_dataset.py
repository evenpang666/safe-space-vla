#!/usr/bin/env python3
"""Build PI05 latent safety-decoder datasets from prefix tokens and real action chunks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "libero_joint_swept_pointcloud" / "pi05_safety_decoder_dataset.npz"


@dataclass(frozen=True)
class DatasetConfig:
    task_suite: str
    task_id: int
    init_state_id: int
    points_per_link: int
    samples_per_action: int = 1
    mujoco_gl: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-samples", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--bddl-file", type=Path, default=None)
    parser.add_argument("--points-per-link", type=int, default=8)
    parser.add_argument("--samples-per-action", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default=None)
    return parser.parse_args()


def import_script_module(module_name: str):
    try:
        return importlib.import_module(f"scripts.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name not in {"scripts", f"scripts.{module_name}"}:
            raise
        return importlib.import_module(module_name)


def validate_seed_arrays(prefix_tokens: np.ndarray, action_chunks: np.ndarray, start_joint_vectors: np.ndarray) -> None:
    if prefix_tokens.ndim != 3:
        raise ValueError(f"prefix_tokens must have shape (S, N, D), got {prefix_tokens.shape}")
    if action_chunks.ndim != 3:
        raise ValueError(f"action_chunks must have shape (S, T, A), got {action_chunks.shape}")
    if start_joint_vectors.ndim != 2:
        raise ValueError(f"start_joint_vectors must have shape (S, J), got {start_joint_vectors.shape}")
    if not (prefix_tokens.shape[0] == action_chunks.shape[0] == start_joint_vectors.shape[0]):
        raise ValueError("prefix_tokens, action_chunks, and start_joint_vectors must have the same first dimension")


def load_seed_samples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        prefix_tokens = np.asarray(data["prefix_tokens"], dtype=np.float32)
        action_chunks = np.asarray(data["action_chunks"], dtype=np.float32)
        start_joint_vectors = np.asarray(data["start_joint_vectors"], dtype=np.float32)
    validate_seed_arrays(prefix_tokens, action_chunks, start_joint_vectors)
    return prefix_tokens, action_chunks, start_joint_vectors


def save_decoder_dataset(
    output: Path,
    *,
    prefix_tokens: np.ndarray,
    action_chunks: np.ndarray,
    start_joint_vectors: np.ndarray,
    target_link_points: np.ndarray,
    link_names: np.ndarray,
    config: DatasetConfig,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prefix_tokens=np.asarray(prefix_tokens, dtype=np.float32),
        action_chunks=np.asarray(action_chunks, dtype=np.float32),
        start_joint_vectors=np.asarray(start_joint_vectors, dtype=np.float32),
        target_link_points=np.asarray(target_link_points, dtype=np.float32),
        link_names=np.asarray(link_names),
        task_suite=np.asarray(config.task_suite),
        task_id=np.asarray(config.task_id),
        init_state_id=np.asarray(config.init_state_id),
        points_per_link=np.asarray(config.points_per_link),
        samples_per_action=np.asarray(config.samples_per_action),
    )


def normalize_fk_inputs(
    start_joint_vector: np.ndarray,
    action_chunk: np.ndarray,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start_joint_vector, dtype=np.float64).reshape(-1)
    if start.size < action_dim:
        raise ValueError(f"Joint vector needs at least {action_dim} values, got {start.size}")
    if start.size > action_dim:
        start = start[:action_dim]

    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2:
        raise ValueError(f"Action chunk must have shape (T, D), got {actions.shape}")
    if actions.shape[1] < action_dim:
        raise ValueError(f"Action chunk needs at least {action_dim} joint deltas, got {actions.shape[1]}")
    if actions.shape[1] > action_dim:
        actions = actions[:, :action_dim]
    return start, actions


def fk_target_link_points(
    env,
    qpos_indices: np.ndarray,
    geom_ids: np.ndarray,
    start_joint_vector: np.ndarray,
    action_chunk: np.ndarray,
    points_per_link: int,
    samples_per_action: int,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    swept = import_script_module("libero_joint_swept_pointcloud")
    link_targets = import_script_module("libero_link_point_targets")

    start, actions = normalize_fk_inputs(start_joint_vector, action_chunk, len(qpos_indices))
    joint_path = swept.integrate_joint_path(start, actions, low, high, samples_per_action)
    segment_path, _geom_kinds, _geom_names, _color_ids, link_names = swept.geom_skeleton_path(
        env,
        qpos_indices,
        np.asarray(geom_ids, dtype=np.int64),
        joint_path,
    )
    return link_targets.sample_link_points_from_segments(segment_path, points_per_link), link_names


def main() -> None:
    args = parse_args()
    if args.points_per_link < 2:
        raise ValueError("--points-per-link must be >= 2")
    if args.samples_per_action < 1:
        raise ValueError("--samples-per-action must be >= 1")
    if args.mujoco_gl is not None:
        import os

        os.environ["MUJOCO_GL"] = args.mujoco_gl

    prefix_tokens, action_chunks, start_joint_vectors = load_seed_samples(args.seed_samples)
    swept = import_script_module("libero_joint_swept_pointcloud")
    libero_pc = import_script_module("libero_reconstruct_pointcloud")

    swept.load_runtime_dependencies()
    bddl_file, _task_name, init_state = libero_pc.resolve_task(args)
    env = libero_pc.create_env(bddl_file, width=64, height=64, camera_names=["agentview"])
    try:
        libero_pc.settle_scene(env, init_state, num_steps=10)
        qpos_indices = swept.get_arm_qpos_indices(env)
        low, high = swept.joint_limits(env.sim, qpos_indices)
        geom_ids = libero_pc.find_robot_geoms(env)
        targets = []
        link_names = np.asarray([])
        for sample_idx in range(prefix_tokens.shape[0]):
            target, link_names = fk_target_link_points(
                env,
                qpos_indices,
                np.asarray(geom_ids, dtype=np.int64),
                start_joint_vectors[sample_idx],
                action_chunks[sample_idx],
                args.points_per_link,
                args.samples_per_action,
                low,
                high,
            )
            targets.append(target)
        save_decoder_dataset(
            args.output,
            prefix_tokens=prefix_tokens,
            action_chunks=action_chunks,
            start_joint_vectors=start_joint_vectors,
            target_link_points=np.stack(targets).astype(np.float32),
            link_names=link_names,
            config=DatasetConfig(
                task_suite=args.task_suite,
                task_id=args.task_id,
                init_state_id=args.init_state_id,
                points_per_link=args.points_per_link,
                samples_per_action=args.samples_per_action,
                mujoco_gl=args.mujoco_gl,
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
