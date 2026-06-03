#!/usr/bin/env python3
"""Build PI05 safety datasets from prefix tokens and real action chunks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "libero_joint_swept_pointcloud" / "pi05_safety_decoder_dataset.npz"
COORDINATE_FRAME = "mujoco_world"
OFFSET_FRAME = "mujoco_world_delta"
ANCHOR_SKELETON_LINK_NAMES = np.asarray([f"link{i}_link{i + 1}" for i in range(7)])
SURFACE_LINK_BODY_NAMES = tuple(f"robot0_link{i}" for i in range(1, 8))
GEOM_TYPE_NAMES = {
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}


@dataclass(frozen=True)
class DatasetConfig:
    task_suite: str
    task_id: int
    init_state_id: int
    points_per_link: int
    samples_per_action: int = 1
    mujoco_gl: str | None = None
    skeleton_source: str = "surface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-samples", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--bddl-file", type=Path, default=None)
    parser.add_argument("--points-per-link", type=int, default=24)
    parser.add_argument("--samples-per-action", type=int, default=1)
    parser.add_argument(
        "--skeleton-source",
        choices=["surface", "anchors", "geom"],
        default="surface",
        help=(
            "'surface' samples fixed surface points on robot0_link1..link7; "
            "'anchors' samples clean robot0_link0..link7 arm skeleton; "
            "'geom' samples all robot geom axes."
        ),
    )
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


def derive_flow_point_targets(target_link_points: np.ndarray) -> dict[str, np.ndarray]:
    """Derive current arm points and future offsets for SafetyFlowPointModel.

    Args:
        target_link_points: Link points with shape ``(T_path, L, P, 3)``.
            ``T_path`` must include the current step at index 0 followed by
            at least one future step.

    Returns:
        ``current_link_points`` with shape ``(L, P, 3)``.
        ``future_link_offsets`` with shape ``(T_path - 1, L, P, 3)``.
        ``arm_points`` with shape ``(L * P, 3)``.
        ``target_point_offsets`` with shape ``(T_path - 1, L * P, 3)``.
    """
    target_link_points = np.asarray(target_link_points, dtype=np.float32)
    if target_link_points.ndim != 4 or target_link_points.shape[-1] != 3:
        raise ValueError(f"target_link_points must have shape (T, L, P, 3), got {target_link_points.shape}")
    if target_link_points.shape[0] < 2:
        raise ValueError("target_link_points must include the current step and at least one future step")

    current_link_points = target_link_points[0].astype(np.float32)  # [L, P, 3]
    future_link_points = target_link_points[1:].astype(np.float32)  # [T_future, L, P, 3]
    future_link_offsets = future_link_points - current_link_points[None, :, :, :]  # [T_future, L, P, 3]
    arm_points = current_link_points.reshape(-1, 3).astype(np.float32)  # [K, 3], K = L * P
    target_point_offsets = future_link_offsets.reshape(future_link_offsets.shape[0], -1, 3).astype(np.float32)
    return {
        "current_link_points": current_link_points,
        "future_link_offsets": future_link_offsets.astype(np.float32),
        "arm_points": arm_points,
        "target_point_offsets": target_point_offsets,
    }


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
    target_link_points = np.asarray(target_link_points, dtype=np.float32)
    derived = [derive_flow_point_targets(sample) for sample in target_link_points]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prefix_tokens=np.asarray(prefix_tokens, dtype=np.float32),
        action_chunks=np.asarray(action_chunks, dtype=np.float32),
        start_joint_vectors=np.asarray(start_joint_vectors, dtype=np.float32),
        target_link_points=target_link_points,
        current_link_points=np.stack([item["current_link_points"] for item in derived]).astype(np.float32),
        future_link_offsets=np.stack([item["future_link_offsets"] for item in derived]).astype(np.float32),
        arm_points=np.stack([item["arm_points"] for item in derived]).astype(np.float32),
        target_point_offsets=np.stack([item["target_point_offsets"] for item in derived]).astype(np.float32),
        link_names=np.asarray(link_names),
        coordinate_frame=np.asarray(COORDINATE_FRAME),
        target_link_points_frame=np.asarray(COORDINATE_FRAME),
        current_link_points_frame=np.asarray(COORDINATE_FRAME),
        future_link_offsets_frame=np.asarray(OFFSET_FRAME),
        arm_points_frame=np.asarray(COORDINATE_FRAME),
        target_point_offsets_frame=np.asarray(OFFSET_FRAME),
        task_suite=np.asarray(config.task_suite),
        task_id=np.asarray(config.task_id),
        init_state_id=np.asarray(config.init_state_id),
        points_per_link=np.asarray(config.points_per_link),
        samples_per_action=np.asarray(config.samples_per_action),
        skeleton_source=np.asarray(config.skeleton_source),
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


def anchor_skeleton_segments_from_path(anchor_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Panda link anchor positions into seven adjacent arm-link segments.

    ``anchor_path`` has shape ``[T, 8, 3]`` for bodies robot0_link0..robot0_link7.
    The returned segments have shape ``[T, 7, 2, 3]``.
    """
    anchor_path = np.asarray(anchor_path, dtype=np.float64)
    if anchor_path.ndim != 3 or anchor_path.shape[1:] != (8, 3):
        raise ValueError(f"anchor_path must have shape (T, 8, 3), got {anchor_path.shape}")
    starts = anchor_path[:, :-1, :]  # [T, 7, 3]
    ends = anchor_path[:, 1:, :]  # [T, 7, 3]
    return np.stack([starts, ends], axis=2), ANCHOR_SKELETON_LINK_NAMES.copy()


def sample_box_surface_points(size: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    size = np.asarray(size, dtype=np.float64).reshape(-1)[:3]
    if size.size != 3:
        raise ValueError(f"box size must have 3 values, got {size.shape}")
    count = int(count)
    if count <= 0:
        return np.empty((0, 3), dtype=np.float32)

    points = rng.uniform(-size, size, size=(count, 3))
    face_axes = rng.integers(0, 3, size=count)
    face_signs = rng.choice(np.asarray([-1.0, 1.0]), size=count)
    points[np.arange(count), face_axes] = face_signs * size[face_axes]
    return points.astype(np.float32)


def sample_sphere_surface_points(radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    count = int(count)
    if count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    directions = rng.normal(size=(count, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    return (directions * float(radius)).astype(np.float32)


def sample_cylinder_surface_points(size: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    size = np.asarray(size, dtype=np.float64).reshape(-1)
    radius = float(size[0]) if size.size > 0 else 0.01
    half_height = float(size[1]) if size.size > 1 else radius
    count = int(count)
    if count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=count)
    z = rng.uniform(-half_height, half_height, size=count)
    points = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)
    return points.astype(np.float32)


def sample_capsule_surface_points(size: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    size = np.asarray(size, dtype=np.float64).reshape(-1)
    radius = float(size[0]) if size.size > 0 else 0.01
    half_height = float(size[1]) if size.size > 1 else radius
    points = sample_sphere_surface_points(radius, count, rng).astype(np.float64)
    cylinder_mask = rng.random(int(count)) < 0.6
    if np.any(cylinder_mask):
        points[cylinder_mask] = sample_cylinder_surface_points(size, int(np.count_nonzero(cylinder_mask)), rng)
    cap_mask = ~cylinder_mask
    if np.any(cap_mask):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=int(np.count_nonzero(cap_mask)))
        points[cap_mask, 2] += signs * half_height
    return points.astype(np.float32)


def sample_mesh_vertices_local(model, geom_id: int, count: int, rng: np.random.Generator) -> np.ndarray:
    try:
        mesh_id = int(model.geom_dataid[int(geom_id)])
        start = int(model.mesh_vertadr[mesh_id])
        num = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + num], dtype=np.float64)
    except Exception:
        vertices = np.empty((0, 3), dtype=np.float64)
    if len(vertices) == 0:
        return sample_box_surface_points(np.asarray(model.geom_size[int(geom_id)], dtype=np.float64)[:3], count, rng)
    indices = rng.choice(len(vertices), size=int(count), replace=len(vertices) < int(count))
    return vertices[indices].astype(np.float32)


def geom_type_name(model, geom_id: int) -> str:
    try:
        geom_type = int(model.geom_type[int(geom_id)])
    except Exception:
        return "box"
    return GEOM_TYPE_NAMES.get(geom_type, "box")


def sample_geom_surface_local(model, geom_id: int, count: int, rng: np.random.Generator) -> np.ndarray:
    kind = geom_type_name(model, geom_id)
    size = np.asarray(model.geom_size[int(geom_id)], dtype=np.float64)
    if kind == "sphere":
        return sample_sphere_surface_points(float(size[0]), count, rng)
    if kind in {"cylinder", "ellipsoid"}:
        return sample_cylinder_surface_points(size, count, rng)
    if kind == "capsule":
        return sample_capsule_surface_points(size, count, rng)
    if kind == "mesh":
        return sample_mesh_vertices_local(model, geom_id, count, rng)
    return sample_box_surface_points(size[:3], count, rng)


def model_body_name(model, body_id: int) -> str:
    try:
        return str(model.body_id2name(int(body_id)) or "")
    except Exception:
        pass
    try:
        names = getattr(model, "body_names")
        return str(names[int(body_id)])
    except Exception:
        return ""


def robot_link_geom_ids(model, geom_ids: np.ndarray, link_body_names: tuple[str, ...] = SURFACE_LINK_BODY_NAMES):
    grouped: list[list[int]] = [[] for _ in link_body_names]
    for geom_id in np.asarray(geom_ids, dtype=np.int64):
        body_id = int(model.geom_bodyid[int(geom_id)])
        body_name = model_body_name(model, body_id)
        for link_idx, link_name in enumerate(link_body_names):
            if body_name == link_name or body_name.startswith(f"{link_name}_"):
                grouped[link_idx].append(int(geom_id))
                break
    return grouped


def distribute_counts(total: int, bucket_count: int) -> np.ndarray:
    if bucket_count <= 0:
        return np.zeros((0,), dtype=np.int64)
    base = int(total) // int(bucket_count)
    remainder = int(total) % int(bucket_count)
    counts = np.full(bucket_count, base, dtype=np.int64)
    counts[:remainder] += 1
    return counts


def build_link_surface_template(
    model,
    geom_ids: np.ndarray,
    points_per_link: int,
    rng: np.random.Generator,
    link_body_names: tuple[str, ...] = SURFACE_LINK_BODY_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_per_link < 1:
        raise ValueError("points_per_link must be >= 1")
    grouped = robot_link_geom_ids(model, geom_ids, link_body_names)
    local_by_link = []
    geom_by_link = []
    for link_name, link_geom_ids in zip(link_body_names, grouped):
        if not link_geom_ids:
            raise RuntimeError(f"No MuJoCo geoms found for arm link body {link_name!r}")
        counts = distribute_counts(points_per_link, len(link_geom_ids))
        local_chunks = []
        geom_chunks = []
        for geom_id, count in zip(link_geom_ids, counts):
            if count <= 0:
                continue
            local_chunks.append(sample_geom_surface_local(model, geom_id, int(count), rng))
            geom_chunks.append(np.full(int(count), int(geom_id), dtype=np.int64))
        local_points = np.concatenate(local_chunks, axis=0)
        point_geom_ids = np.concatenate(geom_chunks, axis=0)
        if local_points.shape[0] != points_per_link:
            raise RuntimeError(f"Expected {points_per_link} points for {link_name}, got {local_points.shape[0]}")
        local_by_link.append(local_points.astype(np.float32))
        geom_by_link.append(point_geom_ids.astype(np.int64))
    return (
        np.stack(local_by_link).astype(np.float32),
        np.stack(geom_by_link).astype(np.int64),
        np.asarray(link_body_names),
    )


def transform_link_surface_template(sim, local_points: np.ndarray, geom_ids: np.ndarray) -> np.ndarray:
    local_points = np.asarray(local_points, dtype=np.float32)
    geom_ids = np.asarray(geom_ids, dtype=np.int64)
    if local_points.ndim != 3 or local_points.shape[-1] != 3:
        raise ValueError(f"local_points must have shape (L, P, 3), got {local_points.shape}")
    if geom_ids.shape != local_points.shape[:2]:
        raise ValueError(f"geom_ids must have shape {local_points.shape[:2]}, got {geom_ids.shape}")
    world = np.empty_like(local_points, dtype=np.float32)
    for geom_id in np.unique(geom_ids):
        mask = geom_ids == int(geom_id)
        rotation = np.asarray(sim.data.geom_xmat[int(geom_id)], dtype=np.float32).reshape(3, 3)
        position = np.asarray(sim.data.geom_xpos[int(geom_id)], dtype=np.float32)
        world[mask] = local_points[mask] @ rotation.T + position
    return world


def fk_surface_link_points(
    env,
    qpos_indices: np.ndarray,
    geom_ids: np.ndarray,
    joint_path: np.ndarray,
    points_per_link: int,
) -> tuple[np.ndarray, np.ndarray]:
    swept = import_script_module("libero_joint_swept_pointcloud")
    rng = np.random.default_rng(0)
    local_points, template_geom_ids, link_names = build_link_surface_template(
        env.sim.model,
        np.asarray(geom_ids, dtype=np.int64),
        points_per_link,
        rng,
    )
    path_points = []
    for q in joint_path:
        swept.set_arm_joint_vector(env.sim, qpos_indices, q)
        path_points.append(transform_link_surface_template(env.sim, local_points, template_geom_ids))
    return np.stack(path_points).astype(np.float32), link_names


def fk_anchor_link_points(
    env,
    qpos_indices: np.ndarray,
    joint_path: np.ndarray,
    points_per_link: int,
) -> tuple[np.ndarray, np.ndarray]:
    swept = import_script_module("libero_joint_swept_pointcloud")
    link_targets = import_script_module("libero_link_point_targets")

    anchor_body_ids = swept.resolve_body_ids(env.sim, swept.DEFAULT_PANDA_ANCHOR_BODIES)
    anchor_path = []
    for q in joint_path:
        swept.set_arm_joint_vector(env.sim, qpos_indices, q)
        anchor_path.append(np.asarray(env.sim.data.body_xpos[anchor_body_ids], dtype=np.float64).copy())
    segment_path, link_names = anchor_skeleton_segments_from_path(np.asarray(anchor_path, dtype=np.float64))
    return link_targets.sample_link_points_from_segments(segment_path, points_per_link), link_names


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
    skeleton_source: str = "surface",
) -> tuple[np.ndarray, np.ndarray]:
    swept = import_script_module("libero_joint_swept_pointcloud")
    link_targets = import_script_module("libero_link_point_targets")

    start, actions = normalize_fk_inputs(start_joint_vector, action_chunk, len(qpos_indices))
    joint_path = swept.integrate_joint_path(start, actions, low, high, samples_per_action)
    if skeleton_source == "surface":
        return fk_surface_link_points(env, qpos_indices, np.asarray(geom_ids, dtype=np.int64), joint_path, points_per_link)
    if skeleton_source == "anchors":
        return fk_anchor_link_points(env, qpos_indices, joint_path, points_per_link)
    if skeleton_source != "geom":
        raise ValueError(f"Unsupported skeleton_source: {skeleton_source}")
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
                args.skeleton_source,
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
                skeleton_source=args.skeleton_source,
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
