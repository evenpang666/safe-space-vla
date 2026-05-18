#!/usr/bin/env python3
"""Generate a swept robot point cloud for a LIBERO action chunk.

This script executes an action chunk in a LIBERO simulator and samples points
from the robot's MuJoCo geoms at every step. The output point cloud contains
only robot geometry and approximates the volume swept by the arm.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable

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
    save_preview_png,
    show_open3d,
    settle_scene,
    write_ascii_ply,
)


GEOM_SPHERE = 2
GEOM_CAPSULE = 3
GEOM_ELLIPSOID = 4
GEOM_CYLINDER = 5
GEOM_BOX = 6
GEOM_MESH = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the robot-only swept point cloud induced by a LIBERO action chunk."
    )
    parser.add_argument(
        "--task-suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO benchmark suite to instantiate.",
    )
    parser.add_argument("--task-id", type=int, default=0, help="Task index in the suite.")
    parser.add_argument(
        "--init-state-id",
        type=int,
        default=0,
        help="Initial-state index from the LIBERO task suite.",
    )
    parser.add_argument(
        "--bddl-file",
        type=Path,
        default=None,
        help="Optional direct path to a .bddl file. Overrides --task-suite/--task-id.",
    )
    parser.add_argument(
        "--action-chunk-file",
        type=Path,
        default=None,
        help="Optional .npy/.npz/.json/.csv action chunk. If omitted, random actions are generated.",
    )
    parser.add_argument("--horizon", type=int, default=16, help="Random action chunk length.")
    parser.add_argument("--action-dim", type=int, default=7, help="Random action dimension.")
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.35,
        help="Uniform random action range [-scale, scale] for generated action chunks.",
    )
    parser.add_argument(
        "--gripper-action",
        type=float,
        default=-1.0,
        help="Value for the final gripper action dimension in generated action chunks.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=10,
        help="No-op simulator steps after reset so objects settle.",
    )
    parser.add_argument(
        "--points-per-geom",
        type=int,
        default=500,
        help="Surface samples per robot geom per simulator step.",
    )
    parser.add_argument(
        "--geom-groups",
        type=int,
        nargs="+",
        default=None,
        help="Optional MuJoCo geom groups to include. By default all robot geoms are used.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.004,
        help="Voxel downsample size in meters. Use 0 to disable downsampling.",
    )
    parser.add_argument(
        "--include-initial",
        action="store_true",
        help="Also sample the robot before the first action is applied.",
    )
    parser.add_argument(
        "--disable-nonrobot-collisions",
        action="store_true",
        help="Set non-robot geoms to non-colliding before rollout. The point cloud is robot-only either way.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_robot_swept_pointcloud",
        help="Directory for point cloud and visualization outputs.",
    )
    parser.add_argument(
        "--preview-points",
        type=int,
        default=80000,
        help="Maximum number of points drawn in the saved preview image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive Open3D viewer if open3d is installed.",
    )
    parser.add_argument(
        "--mujoco-gl",
        choices=["egl", "osmesa", "glfw"],
        default=None,
        help="MuJoCo OpenGL backend. Must be set before robosuite is imported.",
    )
    return parser.parse_args()


def load_action_chunk(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        actions = np.load(path)
    elif suffix == ".npz":
        loaded = np.load(path)
        key = "actions" if "actions" in loaded.files else loaded.files[0]
        actions = loaded[key]
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        actions = payload["actions"] if isinstance(payload, dict) and "actions" in payload else payload
    elif suffix == ".csv":
        actions = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported action chunk file suffix: {suffix}")

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2:
        raise ValueError(f"Action chunk must have shape (T, D), got {actions.shape}")
    return actions


def make_random_action_chunk(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(
        low=-args.action_scale,
        high=args.action_scale,
        size=(args.horizon, args.action_dim),
    ).astype(np.float32)
    if args.action_dim > 0 and args.gripper_action is not None:
        actions[:, -1] = args.gripper_action
    return actions


def normalize_action_chunk(actions: np.ndarray, env_action_dim: int) -> np.ndarray:
    if actions.shape[1] < env_action_dim:
        raise ValueError(
            f"Action chunk has dim {actions.shape[1]}, but LIBERO env expects {env_action_dim}."
        )
    if actions.shape[1] > env_action_dim:
        print(
            f"[warn] action dim {actions.shape[1]} is larger than env action_dim "
            f"{env_action_dim}; using the first {env_action_dim} dims."
        )
        actions = actions[:, :env_action_dim]
    return actions.astype(np.float32, copy=False)


def model_name(model, kind: str, idx: int) -> str:
    try:
        return getattr(model, f"{kind}_id2name")(idx) or ""
    except Exception:
        names = getattr(model, f"{kind}_names", None)
        if names is not None and idx < len(names):
            return names[idx] or ""
    return ""


def model_name2id(model, kind: str, name: str) -> int | None:
    try:
        return int(getattr(model, f"{kind}_name2id")(name))
    except Exception:
        return None


def body_descendants(model, root_body_id: int) -> set[int]:
    parent_ids = np.asarray(model.body_parentid)
    descendants = {root_body_id}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parent_ids):
            if body_id not in descendants and int(parent_id) in descendants:
                descendants.add(body_id)
                changed = True
    return descendants


def robot_body_ids(env) -> set[int]:
    ids: set[int] = set()
    for robot in env.robots:
        root_name = getattr(robot.robot_model, "root_body", None)
        if root_name is None:
            continue
        root_id = model_name2id(env.sim.model, "body", root_name)
        if root_id is not None:
            ids.update(body_descendants(env.sim.model, root_id))
    return ids


def find_robot_geoms(env, geom_groups: Iterable[int] | None) -> list[int]:
    model = env.sim.model
    robot_bodies = robot_body_ids(env)
    allowed_groups = None if geom_groups is None else set(int(g) for g in geom_groups)

    geom_ids = []
    for geom_id in range(int(model.ngeom)):
        if allowed_groups is not None and int(model.geom_group[geom_id]) not in allowed_groups:
            continue

        body_id = int(model.geom_bodyid[geom_id])
        if robot_bodies and body_id in robot_bodies:
            geom_ids.append(geom_id)
            continue

        name = model_name(model, "geom", geom_id).lower()
        if any(key in name for key in ("robot", "panda", "ur5", "gripper", "finger", "hand", "arm")):
            geom_ids.append(geom_id)

    return sorted(set(geom_ids))


def disable_nonrobot_collisions(env, robot_geom_ids: Iterable[int]) -> None:
    robot_geom_ids = set(robot_geom_ids)
    model = env.sim.model
    for geom_id in range(int(model.ngeom)):
        if geom_id in robot_geom_ids:
            continue
        if hasattr(model, "geom_contype"):
            model.geom_contype[geom_id] = 0
        if hasattr(model, "geom_conaffinity"):
            model.geom_conaffinity[geom_id] = 0


def sample_box(size: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    sx, sy, sz = size[:3]
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy], dtype=np.float64)
    faces = rng.choice(6, size=n, p=areas / areas.sum())
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32) * size[:3]
    points[faces == 0, 0] = sx
    points[faces == 1, 0] = -sx
    points[faces == 2, 1] = sy
    points[faces == 3, 1] = -sy
    points[faces == 4, 2] = sz
    points[faces == 5, 2] = -sz
    return points


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    vectors = rng.normal(size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def sample_sphere(radius: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return random_unit_vectors(n, rng) * float(radius)


def sample_ellipsoid(size: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return random_unit_vectors(n, rng) * size[:3]


def sample_cylinder(size: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    radius = float(size[0])
    half_height = float(size[1])
    side_area = 2.0 * np.pi * radius * (2.0 * half_height)
    cap_area = np.pi * radius * radius
    probs = np.array([side_area, cap_area, cap_area], dtype=np.float64)
    parts = rng.choice(3, size=n, p=probs / probs.sum())
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    points = np.zeros((n, 3), dtype=np.float32)

    side = parts == 0
    points[side, 0] = radius * np.cos(theta[side])
    points[side, 1] = radius * np.sin(theta[side])
    points[side, 2] = rng.uniform(-half_height, half_height, size=np.count_nonzero(side))

    for part, z in ((1, half_height), (2, -half_height)):
        cap = parts == part
        radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=np.count_nonzero(cap)))
        points[cap, 0] = radial * np.cos(theta[cap])
        points[cap, 1] = radial * np.sin(theta[cap])
        points[cap, 2] = z
    return points


def sample_capsule(size: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    radius = float(size[0])
    half_length = float(size[1])
    cylinder_area = 2.0 * np.pi * radius * (2.0 * half_length)
    sphere_area = 4.0 * np.pi * radius * radius
    use_cylinder = rng.uniform(size=n) < cylinder_area / max(cylinder_area + sphere_area, 1e-8)
    points = np.zeros((n, 3), dtype=np.float32)

    cylinder_count = np.count_nonzero(use_cylinder)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=cylinder_count)
    points[use_cylinder, 0] = radius * np.cos(theta)
    points[use_cylinder, 1] = radius * np.sin(theta)
    points[use_cylinder, 2] = rng.uniform(-half_length, half_length, size=cylinder_count)

    sphere_count = n - cylinder_count
    dirs = random_unit_vectors(sphere_count, rng)
    sphere_points = dirs * radius
    sphere_points[:, 2] += np.where(sphere_points[:, 2] >= 0.0, half_length, -half_length)
    points[~use_cylinder] = sphere_points
    return points


def sample_mesh(model, geom_id: int, n: int, rng: np.random.Generator) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])

    vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float32)
    faces = np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return np.empty((0, 3), dtype=np.float32)

    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = areas > 1e-12
    if not np.any(valid):
        return vertices[rng.choice(len(vertices), size=n, replace=True)]

    tri = tri[valid]
    areas = areas[valid]
    chosen = rng.choice(len(tri), size=n, p=areas / areas.sum())
    selected = tri[chosen]
    u = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    v = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return selected[:, 0] + u * (selected[:, 1] - selected[:, 0]) + v * (selected[:, 2] - selected[:, 0])


def sample_geom_local(model, geom_id: int, n: int, rng: np.random.Generator) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float32)
    if geom_type == GEOM_BOX:
        return sample_box(size, n, rng)
    if geom_type == GEOM_SPHERE:
        return sample_sphere(size[0], n, rng)
    if geom_type == GEOM_ELLIPSOID:
        return sample_ellipsoid(size, n, rng)
    if geom_type == GEOM_CYLINDER:
        return sample_cylinder(size, n, rng)
    if geom_type == GEOM_CAPSULE:
        return sample_capsule(size, n, rng)
    if geom_type == GEOM_MESH:
        return sample_mesh(model, geom_id, n, rng)
    return np.empty((0, 3), dtype=np.float32)


def geom_points_world(sim, geom_id: int, points_per_geom: int, rng: np.random.Generator) -> np.ndarray:
    local_points = sample_geom_local(sim.model, geom_id, points_per_geom, rng)
    if len(local_points) == 0:
        return local_points
    rotation = np.asarray(sim.data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
    position = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float32)
    return local_points @ rotation.T + position


def step_color(step_idx: int, num_steps: int, n: int) -> np.ndarray:
    t = 0.0 if num_steps <= 1 else step_idx / float(num_steps - 1)
    color = np.array(
        [
            int(255 * t),
            int(70 + 120 * (1.0 - abs(2.0 * t - 1.0))),
            int(255 * (1.0 - t)),
        ],
        dtype=np.uint8,
    )
    return np.repeat(color[None, :], n, axis=0)


def sample_robot_pointcloud(
    sim,
    robot_geom_ids: Iterable[int],
    points_per_geom: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chunks = []
    for geom_id in robot_geom_ids:
        points = geom_points_world(sim, geom_id, points_per_geom, rng)
        if len(points) > 0:
            chunks.append(points)
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return points[unique_idx], colors[unique_idx]


def get_env_action_dim(env) -> int:
    return int(getattr(env.env, "action_dim", 7))


def main() -> None:
    args = parse_args()
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_runtime_dependencies()
    bddl_file, task_name, init_state = resolve_task(args)
    env = create_env(bddl_file, width=64, height=64, camera_names=["agentview"])

    rng = np.random.default_rng(args.seed)
    try:
        settle_scene(env, init_state, args.num_steps_wait)
        action_chunk = (
            load_action_chunk(args.action_chunk_file)
            if args.action_chunk_file is not None
            else make_random_action_chunk(args)
        )
        action_chunk = normalize_action_chunk(action_chunk, get_env_action_dim(env))

        robot_geom_ids = find_robot_geoms(env, args.geom_groups)
        if not robot_geom_ids:
            raise RuntimeError("No robot geoms were found in the LIBERO simulator.")
        if args.disable_nonrobot_collisions:
            disable_nonrobot_collisions(env, robot_geom_ids)

        print(f"[info] task: {task_name}")
        print(f"[info] action_chunk shape: {action_chunk.shape}")
        print(f"[info] robot geoms: {len(robot_geom_ids)}")

        all_points = []
        all_colors = []
        sample_steps = len(action_chunk) + (1 if args.include_initial else 0)
        color_step = 0

        if args.include_initial:
            points = sample_robot_pointcloud(env.sim, robot_geom_ids, args.points_per_geom, rng)
            all_points.append(points)
            all_colors.append(step_color(color_step, sample_steps, len(points)))
            print(f"[info] sampled initial pose: {len(points)} points")
            color_step += 1

        for action_idx, action in enumerate(action_chunk):
            env.step(action)
            points = sample_robot_pointcloud(env.sim, robot_geom_ids, args.points_per_geom, rng)
            all_points.append(points)
            all_colors.append(step_color(color_step, sample_steps, len(points)))
            print(f"[info] sampled action step {action_idx}: {len(points)} points")
            color_step += 1

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        raw_count = len(points)
        points, colors = voxel_downsample(points, colors, args.voxel_size)

        if len(points) == 0:
            raise RuntimeError("Generated an empty swept robot point cloud.")

        safe_task_name = task_name.replace("/", "_")
        prefix = f"{safe_task_name}_robot_swept"
        npz_path = args.output_dir / f"{prefix}.npz"
        ply_path = args.output_dir / f"{prefix}.ply"
        preview_path = args.output_dir / f"{prefix}_preview.png"
        action_path = args.output_dir / f"{prefix}_actions.npy"

        np.savez_compressed(
            npz_path,
            points=points.astype(np.float32),
            colors=colors.astype(np.uint8),
            raw_point_count=np.array(raw_count, dtype=np.int64),
            voxel_size=np.array(args.voxel_size, dtype=np.float32),
        )
        np.save(action_path, action_chunk)
        write_ascii_ply(ply_path, points, colors)
        save_preview_png(preview_path, points, colors, args.preview_points)

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        print(f"[done] raw swept points: {raw_count}")
        print(f"[done] saved swept points: {len(points)}")
        print(f"[done] bounds min xyz: {mins}")
        print(f"[done] bounds max xyz: {maxs}")
        print(f"[done] saved npz: {npz_path}")
        print(f"[done] saved ply: {ply_path}")
        print(f"[done] saved preview: {preview_path}")
        print(f"[done] saved actions: {action_path}")

        if args.show:
            show_open3d(points, colors)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
