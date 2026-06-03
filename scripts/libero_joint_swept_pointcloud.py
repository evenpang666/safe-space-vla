#!/usr/bin/env python3
"""Generate LIBERO robot skeleton swept point clouds from FK samples.

This is the LIBERO counterpart of visualize_robosuite_joint_swept_surfaces.py:
it integrates a joint-delta chunk, builds sparse skeleton segments at every FK
sample, and samples the swept ruled surfaces made by those segments over time.
By default, skeleton segments come from each robot geom's envelope center axis.
The legacy joint-anchor line mode is still available with --skeleton-source
anchors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import libero_reconstruct_pointcloud as libero_pc  # noqa: E402
from libero_reconstruct_pointcloud import (  # noqa: E402
    create_env,
    load_runtime_dependencies,
    resolve_task,
    settle_scene,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "libero_joint_swept_pointcloud"
DEFAULT_PANDA_ANCHOR_BODIES = tuple(f"robot0_link{i}" for i in range(8))
DEFAULT_GRIPPER_BODY = "gripper0_eef"
DEFAULT_LINK_NAMES = tuple(f"link{i}_link{i + 1}" for i in range(7)) + ("gripper_width",)
LINK_COLORS = np.array(
    [
        [0.20, 0.47, 0.85, 1.0],
        [0.12, 0.62, 0.52, 1.0],
        [0.95, 0.62, 0.16, 1.0],
        [0.86, 0.25, 0.25, 1.0],
        [0.55, 0.35, 0.80, 1.0],
        [0.35, 0.35, 0.35, 1.0],
        [0.05, 0.68, 0.90, 1.0],
        [0.90, 0.20, 0.65, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CollisionResult:
    collides: bool
    method: str
    collision_point_count: int
    colliding_point_indices: np.ndarray
    collision_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO benchmark suite to instantiate.",
    )
    parser.add_argument("--task-id", type=int, default=0, help="Task index in the suite.")
    parser.add_argument("--init-state-id", type=int, default=0, help="Initial-state index from the suite.")
    parser.add_argument("--bddl-file", type=Path, default=None, help="Optional direct .bddl path.")
    parser.add_argument(
        "--action-chunk-file",
        type=Path,
        default=None,
        help="Optional .npy/.npz/.json/.csv joint-delta chunk. Uses the first arm-joint dims.",
    )
    parser.add_argument("--horizon", type=int, default=100, help="Random joint-delta chunk length.")
    parser.add_argument("--action-scale", type=float, default=0.06, help="Random joint delta range in radians.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for generated joint deltas.")
    parser.add_argument("--samples-per-action", type=int, default=8, help="Interpolated FK intervals per action.")
    parser.add_argument("--joint-vector-file", type=Path, default=None, help="Optional start joint vector file.")
    parser.add_argument("--gripper-width", type=float, default=0.08, help="Virtual gripper segment length in meters.")
    parser.add_argument(
        "--skeleton-source",
        choices=["geom", "anchors"],
        default="geom",
        help=(
            "Source for swept skeleton segments. 'geom' uses robot geom envelope center axes; "
            "'anchors' keeps the legacy robot0_link0..link7 joint-anchor lines."
        ),
    )
    parser.add_argument(
        "--swept-point-link-samples",
        type=int,
        default=8,
        help="Samples along each joint-link segment on every swept panel.",
    )
    parser.add_argument(
        "--swept-point-time-samples",
        type=int,
        default=2,
        help="Samples along each swept panel's time direction.",
    )
    parser.add_argument("--frontview-width", type=int, default=768, help="Frontview output width.")
    parser.add_argument("--frontview-height", type=int, default=768, help="Frontview output height.")
    parser.add_argument("--frontview-point-size", type=float, default=3.0, help="Projected point marker size.")
    parser.add_argument("--plot-elev", type=float, default=18.0, help="3D front-view plot elevation.")
    parser.add_argument("--plot-azim", type=float, default=0.0, help="3D front-view plot azimuth.")
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save an MP4 where swept skeleton points accumulate over time.",
    )
    parser.add_argument("--video-fps", type=int, default=12, help="Frames per second for --save-video.")
    parser.add_argument("--video-dpi", type=int, default=160, help="Matplotlib DPI for --save-video frames.")
    parser.add_argument(
        "--safe-space",
        type=Path,
        default=None,
        help="Optional *_safe_space.npz used to check swept-point collision against obstacles.",
    )
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.0,
        help="Meters by which obstacle occupancy / OBBs are inflated for swept-point collision checks.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default=None, help="Output basename. Defaults to LIBERO task name.")
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default=None)
    return parser.parse_args()


def load_array(path: Path, key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        loaded = np.load(path)
        selected_key = key if key in loaded.files else loaded.files[0]
        arr = loaded[selected_key]
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if key is not None and isinstance(payload, dict) and key in payload:
            payload = payload[key]
        arr = np.asarray(payload)
    elif suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported array file suffix: {suffix}")
    return np.asarray(arr, dtype=np.float64)


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_random_action_chunk(horizon: int, action_dim: int, action_scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-action_scale, action_scale, size=(horizon, action_dim)).astype(np.float64)


def normalize_action_chunk(actions: np.ndarray, action_dim: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2:
        raise ValueError(f"Action chunk must have shape (T, D), got {actions.shape}")
    if actions.shape[1] < action_dim:
        raise ValueError(f"Action chunk needs at least {action_dim} joint deltas, got {actions.shape[1]}")
    if actions.shape[1] > action_dim:
        print(f"[warn] action dim {actions.shape[1]} > {action_dim}; using first {action_dim} dims.")
        actions = actions[:, :action_dim]
    return actions


def normalize_joint_vector(joint_vector: np.ndarray, action_dim: int) -> np.ndarray:
    joint_vector = np.asarray(joint_vector, dtype=np.float64).reshape(-1)
    if joint_vector.size < action_dim:
        raise ValueError(f"Joint vector needs at least {action_dim} values, got {joint_vector.size}")
    if joint_vector.size > action_dim:
        print(f"[warn] joint vector dim {joint_vector.size} > {action_dim}; using first {action_dim} values.")
        joint_vector = joint_vector[:action_dim]
    return joint_vector


def get_arm_qpos_indices(env) -> np.ndarray:
    indexes = getattr(env.robots[0], "_ref_joint_pos_indexes", None)
    if indexes is None:
        indexes = getattr(env.robots[0], "joint_indexes", None)
    if indexes is None:
        raise RuntimeError("Could not find LIBERO robot joint qpos indexes.")
    indexes = np.asarray(indexes, dtype=np.int64)
    if indexes.size < 1:
        raise RuntimeError("Robot has no arm joint qpos indexes.")
    return indexes


def joint_limits(sim, qpos_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lows = np.full(len(qpos_indices), -np.inf, dtype=np.float64)
    highs = np.full(len(qpos_indices), np.inf, dtype=np.float64)
    for joint_id in range(int(sim.model.njnt)):
        qpos_adr = int(sim.model.jnt_qposadr[joint_id])
        matches = np.where(qpos_indices == qpos_adr)[0]
        if len(matches) == 0:
            continue
        out_idx = int(matches[0])
        if bool(sim.model.jnt_limited[joint_id]):
            lows[out_idx], highs[out_idx] = np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64)
    return lows, highs


def integrate_joint_path(
    start_joint_vector: np.ndarray,
    action_chunk: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    samples_per_action: int,
) -> np.ndarray:
    if samples_per_action < 1:
        raise ValueError("--samples-per-action must be >= 1")
    q = np.asarray(start_joint_vector, dtype=np.float64).copy()
    path = [q.copy()]
    for action in np.asarray(action_chunk, dtype=np.float64):
        q_next = np.clip(q + action, low, high)
        for sample_idx in range(1, samples_per_action + 1):
            alpha = sample_idx / float(samples_per_action)
            path.append((1.0 - alpha) * q + alpha * q_next)
        q = q_next
    return np.asarray(path, dtype=np.float64)


def set_arm_joint_vector(sim, qpos_indices: np.ndarray, joint_vector: np.ndarray) -> None:
    sim.data.qpos[qpos_indices] = joint_vector
    sim.data.qvel[qpos_indices] = 0.0
    sim.forward()


def fk_path(
    env,
    qpos_indices: np.ndarray,
    anchor_body_ids: np.ndarray,
    gripper_body_id: int,
    joint_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_points = []
    gripper_rotations = []
    for q in joint_path:
        set_arm_joint_vector(env.sim, qpos_indices, q)
        anchor_points.append(np.asarray(env.sim.data.body_xpos[anchor_body_ids], dtype=np.float64).copy())
        gripper_rotations.append(
            np.asarray(env.sim.data.body_xmat[gripper_body_id], dtype=np.float64).reshape(3, 3).copy()
        )
    return np.asarray(anchor_points), np.asarray(gripper_rotations)


def gripper_x_axis_segment(anchor_points: np.ndarray, gripper_rotation: np.ndarray, gripper_width: float) -> np.ndarray:
    eef = anchor_points[-1]
    direction = np.asarray(gripper_rotation[:, 0], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    half = 0.5 * float(gripper_width)
    return np.asarray([eef - half * direction, eef + half * direction], dtype=np.float64)


def build_link_segments(
    anchor_path: np.ndarray,
    gripper_rotation_path: np.ndarray,
    gripper_width: float,
) -> np.ndarray:
    if gripper_width <= 0.0:
        raise ValueError("--gripper-width must be positive")
    anchor_path = np.asarray(anchor_path, dtype=np.float64)
    if anchor_path.ndim != 3 or anchor_path.shape[-1] != 3 or anchor_path.shape[1] < 2:
        raise ValueError(f"anchor_path must have shape (T, A>=2, 3), got {anchor_path.shape}")
    segment_count = anchor_path.shape[1]
    segments = np.empty((anchor_path.shape[0], segment_count, 2, 3), dtype=np.float64)
    for step_idx in range(anchor_path.shape[0]):
        for link_idx in range(anchor_path.shape[1] - 1):
            segments[step_idx, link_idx, 0] = anchor_path[step_idx, link_idx]
            segments[step_idx, link_idx, 1] = anchor_path[step_idx, link_idx + 1]
        segments[step_idx, -1] = gripper_x_axis_segment(
            anchor_path[step_idx],
            gripper_rotation_path[step_idx],
            gripper_width,
        )
    return segments


def mujoco_geom_type_names() -> dict[int, str]:
    try:
        import mujoco
    except ImportError:
        return {
            0: "plane",
            1: "hfield",
            2: "sphere",
            3: "capsule",
            4: "ellipsoid",
            5: "cylinder",
            6: "box",
            7: "mesh",
        }
    return {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    }


def model_name(model, kind: str, idx: int) -> str:
    try:
        return getattr(model, f"{kind}_id2name")(idx) or ""
    except Exception:
        names = getattr(model, f"{kind}_names", None)
        if names is not None and idx < len(names):
            return names[idx] or ""
    return ""


def geom_kind(model, geom_id: int) -> str:
    geom_types = np.asarray(model.geom_type)
    type_id = int(geom_types[int(geom_id)])
    return mujoco_geom_type_names().get(type_id, f"geom_type_{type_id}")


def central_axis_segment_for_geom(
    position: np.ndarray,
    rotation: np.ndarray,
    size: np.ndarray,
    geom_kind: str,
    local_segment: np.ndarray | None = None,
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if local_segment is not None:
        local_segment = np.asarray(local_segment, dtype=np.float64).reshape(2, 3)
        return local_segment @ rotation.T + position

    size = np.asarray(size, dtype=np.float64).reshape(-1)
    padded_size = np.zeros(3, dtype=np.float64)
    padded_size[: min(3, size.size)] = np.abs(size[: min(3, size.size)])
    kind = str(geom_kind).lower()

    if kind in {"capsule", "cylinder"}:
        axis_idx = 2
        half_length = float(padded_size[1])
    elif kind in {"box", "ellipsoid"}:
        axis_idx = int(np.argmax(padded_size))
        half_length = float(padded_size[axis_idx])
    else:
        axis_idx = int(np.argmax(padded_size))
        half_length = 0.0

    if half_length <= 0.0:
        return np.stack([position, position], axis=0)
    direction = rotation[:, axis_idx]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return np.stack([position, position], axis=0)
    direction = direction / norm
    return np.stack([position - half_length * direction, position + half_length * direction], axis=0)


def local_axis_segment_from_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(vertices) == 0:
        return np.zeros((2, 3), dtype=np.float64)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_sizes = 0.5 * (maxs - mins)
    axis_idx = int(np.argmax(half_sizes))
    half_length = float(half_sizes[axis_idx])
    if half_length <= 0.0:
        return np.stack([center, center], axis=0)
    segment = np.stack([center, center], axis=0)
    segment[0, axis_idx] -= half_length
    segment[1, axis_idx] += half_length
    return segment


def local_axis_segment_for_model_geom(model, geom_id: int, geom_kind_name: str) -> np.ndarray | None:
    if str(geom_kind_name).lower() != "mesh":
        return None
    try:
        mesh_id = int(model.geom_dataid[int(geom_id)])
        if mesh_id < 0:
            return None
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
    except Exception:
        return None
    return local_axis_segment_from_vertices(vertices)


def link_color_ids_from_body_names(body_names: list[str] | tuple[str, ...] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    link_to_color: dict[str, int] = {}
    color_ids = []
    ordered_names = []
    for raw_name in body_names:
        name = str(raw_name) if str(raw_name) else "unknown_link"
        if name not in link_to_color:
            link_to_color[name] = len(link_to_color)
            ordered_names.append(name)
        color_ids.append(link_to_color[name])
    return np.asarray(color_ids, dtype=np.int64), np.asarray(ordered_names)


def build_geom_skeleton_segments(
    positions: np.ndarray,
    rotations: np.ndarray,
    sizes: np.ndarray,
    geom_kinds: list[str] | tuple[str, ...] | np.ndarray,
    local_segments: np.ndarray | None = None,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape (T, G, 3), got {positions.shape}")
    if rotations.shape != positions.shape[:2] + (3, 3):
        raise ValueError(f"rotations must have shape (T, G, 3, 3), got {rotations.shape}")
    if sizes.shape[0] != positions.shape[1]:
        raise ValueError(f"sizes must describe {positions.shape[1]} geoms, got {sizes.shape}")
    if len(geom_kinds) != positions.shape[1]:
        raise ValueError(f"geom_kinds must describe {positions.shape[1]} geoms, got {len(geom_kinds)}")
    if local_segments is not None:
        local_segments = np.asarray(local_segments, dtype=np.float64)
        if local_segments.shape != positions.shape[1:2] + (2, 3):
            raise ValueError(f"local_segments must have shape (G, 2, 3), got {local_segments.shape}")

    segments = np.empty(positions.shape[:2] + (2, 3), dtype=np.float64)
    for step_idx in range(positions.shape[0]):
        for geom_idx in range(positions.shape[1]):
            local_segment = None if local_segments is None else local_segments[geom_idx]
            segments[step_idx, geom_idx] = central_axis_segment_for_geom(
                positions[step_idx, geom_idx],
                rotations[step_idx, geom_idx],
                sizes[geom_idx],
                str(geom_kinds[geom_idx]),
                local_segment=local_segment,
            )
    return segments


def geom_skeleton_path(
    env,
    qpos_indices: np.ndarray,
    geom_ids: np.ndarray,
    joint_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geom_ids = np.asarray(geom_ids, dtype=np.int64)
    if geom_ids.size == 0:
        raise RuntimeError("No robot geoms were found for geom skeleton generation.")

    positions = []
    rotations = []
    for q in joint_path:
        set_arm_joint_vector(env.sim, qpos_indices, q)
        positions.append(np.asarray(env.sim.data.geom_xpos[geom_ids], dtype=np.float64).copy())
        rotations.append(np.asarray(env.sim.data.geom_xmat[geom_ids], dtype=np.float64).reshape(-1, 3, 3).copy())

    sizes = np.asarray(env.sim.model.geom_size[geom_ids], dtype=np.float64)
    kinds = np.asarray([geom_kind(env.sim.model, int(geom_id)) for geom_id in geom_ids])
    names = np.asarray(
        [
            model_name(env.sim.model, "geom", int(geom_id)) or f"robot_geom_{int(geom_id)}"
            for geom_id in geom_ids
        ]
    )
    body_names = np.asarray(
        [
            model_name(env.sim.model, "body", int(env.sim.model.geom_bodyid[int(geom_id)]))
            or f"robot_body_{int(env.sim.model.geom_bodyid[int(geom_id)])}"
            for geom_id in geom_ids
        ]
    )
    segment_color_ids, link_names = link_color_ids_from_body_names(body_names)
    local_segments = np.asarray(
        [
            (
                local_axis_segment_for_model_geom(env.sim.model, int(geom_id), str(kind))
                if str(kind).lower() == "mesh"
                else np.full((2, 3), np.nan, dtype=np.float64)
            )
            for geom_id, kind in zip(geom_ids, kinds)
        ],
        dtype=np.float64,
    )
    local_segments_arg = None
    if np.isfinite(local_segments).any():
        for idx, segment in enumerate(local_segments):
            if not np.isfinite(segment).all():
                local_segments[idx] = central_axis_segment_for_geom(
                    np.zeros(3, dtype=np.float64),
                    np.eye(3, dtype=np.float64),
                    sizes[idx],
                    str(kinds[idx]),
                )
        local_segments_arg = local_segments
    return (
        build_geom_skeleton_segments(
            np.asarray(positions),
            np.asarray(rotations),
            sizes,
            kinds,
            local_segments=local_segments_arg,
        ),
        kinds,
        names,
        segment_color_ids,
        link_names,
    )


def points_inside_oriented_boxes(
    points: np.ndarray,
    centers: np.ndarray,
    axes: np.ndarray,
    half_sizes: np.ndarray,
    margin: float = 0.0,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    axes = np.asarray(axes, dtype=np.float64).reshape(-1, 3, 3)
    half_sizes = np.asarray(half_sizes, dtype=np.float64).reshape(-1, 3)
    if len(centers) == 0:
        return np.zeros(len(points), dtype=bool)
    if not (len(centers) == len(axes) == len(half_sizes)):
        raise ValueError("box centers, axes, and half_sizes must describe the same number of boxes")

    inflated_half_sizes = half_sizes + max(float(margin), 0.0)
    inside_any = np.zeros(len(points), dtype=bool)
    for center, box_axes, box_half_sizes in zip(centers, axes, inflated_half_sizes):
        local = (points - center) @ box_axes
        inside_any |= np.all(np.abs(local) <= (box_half_sizes + 1e-9), axis=1)
    return inside_any


def occupied_grid_collision_mask(
    points: np.ndarray,
    workspace_bounds: np.ndarray,
    voxel_size: float,
    occupied_grid: np.ndarray,
    margin: float = 0.0,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    bounds = np.asarray(workspace_bounds, dtype=np.float64).reshape(6)
    occupied = np.asarray(occupied_grid, dtype=bool)
    voxel = float(voxel_size)
    if voxel <= 0.0:
        raise ValueError("voxel_size must be positive")

    origin = np.asarray([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
    dims = np.asarray(occupied.shape, dtype=np.int64)
    base_indices = np.floor((points - origin) / voxel).astype(np.int64)
    colliding = np.zeros(len(points), dtype=bool)

    radius = int(np.ceil(max(float(margin), 0.0) / voxel))
    offsets = [
        np.asarray([dx, dy, dz], dtype=np.int64)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]
    for offset in offsets:
        candidate = base_indices + offset
        valid = np.all((candidate >= 0) & (candidate < dims), axis=1)
        if not np.any(valid):
            continue
        valid_indices = candidate[valid]
        occupied_hit = occupied[tuple(valid_indices.T)]
        if not np.any(occupied_hit):
            continue
        valid_point_indices = np.flatnonzero(valid)[occupied_hit]
        if margin <= 0.0:
            colliding[valid_point_indices] = True
            continue

        hit_cells = candidate[valid_point_indices]
        cell_min = origin + hit_cells.astype(np.float64) * voxel
        cell_max = cell_min + voxel
        hit_points = points[valid_point_indices]
        delta = np.maximum(np.maximum(cell_min - hit_points, hit_points - cell_max), 0.0)
        colliding[valid_point_indices] |= np.linalg.norm(delta, axis=1) <= margin
    return colliding


def nearest_obstacle_point_collision_mask(
    points: np.ndarray,
    obstacle_points: np.ndarray,
    threshold: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    obstacle_points = np.asarray(obstacle_points, dtype=np.float64).reshape(-1, 3)
    if len(obstacle_points) == 0:
        return np.zeros(len(points), dtype=bool)
    threshold_sq = float(threshold) ** 2
    colliding = np.zeros(len(points), dtype=bool)
    chunk = 4096
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        diff = points[start:stop, None, :] - obstacle_points[None, :, :]
        min_dist_sq = np.min(np.sum(diff * diff, axis=2), axis=1)
        colliding[start:stop] = min_dist_sq <= threshold_sq
    return colliding


def detect_swept_obstacle_collision(
    swept_points: np.ndarray,
    safe_space: dict[str, np.ndarray],
    collision_margin: float = 0.0,
) -> CollisionResult:
    margin = max(float(collision_margin), 0.0)
    if {"workspace_bounds", "voxel_size", "occupied_grid"}.issubset(safe_space):
        collision_mask = occupied_grid_collision_mask(
            swept_points,
            workspace_bounds=safe_space["workspace_bounds"],
            voxel_size=float(np.asarray(safe_space["voxel_size"])),
            occupied_grid=safe_space["occupied_grid"],
            margin=margin,
        )
        method = "occupied_grid"
    elif {"obstacle_box_centers", "obstacle_box_axes", "obstacle_box_half_sizes"}.issubset(safe_space):
        collision_mask = points_inside_oriented_boxes(
            swept_points,
            centers=safe_space["obstacle_box_centers"],
            axes=safe_space["obstacle_box_axes"],
            half_sizes=safe_space["obstacle_box_half_sizes"],
            margin=margin,
        )
        method = "oriented_boxes"
    elif "obstacle_centers" in safe_space:
        voxel_size = float(np.asarray(safe_space.get("voxel_size", np.asarray(0.0))))
        threshold = margin if margin > 0.0 else 0.5 * np.sqrt(3.0) * max(voxel_size, 0.0)
        collision_mask = nearest_obstacle_point_collision_mask(
            swept_points,
            obstacle_points=safe_space["obstacle_centers"],
            threshold=threshold,
        )
        method = "obstacle_centers"
    else:
        raise ValueError("safe-space data must contain occupied_grid, obstacle boxes, or obstacle_centers")

    indices = np.flatnonzero(collision_mask).astype(np.int64)
    return CollisionResult(
        collides=bool(len(indices) > 0),
        method=method,
        collision_point_count=int(len(indices)),
        colliding_point_indices=indices,
        collision_margin=margin,
    )


def build_swept_panels(segment_path: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panels = []
    panel_link_ids = []
    panel_step_ids = []
    for step_idx in range(segment_path.shape[0] - 1):
        s0 = segment_path[step_idx]
        s1 = segment_path[step_idx + 1]
        for link_idx in range(segment_path.shape[1]):
            panels.append([s0[link_idx, 0], s0[link_idx, 1], s1[link_idx, 1], s1[link_idx, 0]])
            panel_link_ids.append(link_idx)
            panel_step_ids.append(step_idx)
    return (
        np.asarray(panels, dtype=np.float64),
        np.asarray(panel_link_ids, dtype=np.int64),
        np.asarray(panel_step_ids, dtype=np.int64),
    )


def sample_swept_surface_points(
    segment_path: np.ndarray,
    link_samples: int,
    time_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if link_samples < 2:
        raise ValueError("--swept-point-link-samples must be >= 2")
    if time_samples < 2:
        raise ValueError("--swept-point-time-samples must be >= 2")
    segment_path = np.asarray(segment_path, dtype=np.float64)
    if segment_path.ndim != 4 or segment_path.shape[-2:] != (2, 3):
        raise ValueError(f"segment_path must have shape (T, L, 2, 3), got {segment_path.shape}")

    u_values = np.linspace(0.0, 1.0, link_samples, dtype=np.float64)
    v_values = np.linspace(0.0, 1.0, time_samples, dtype=np.float64)
    points = []
    link_ids = []
    step_ids = []
    for step_idx in range(segment_path.shape[0] - 1):
        s0 = segment_path[step_idx]
        s1 = segment_path[step_idx + 1]
        for link_idx in range(segment_path.shape[1]):
            p00 = s0[link_idx, 0]
            p01 = s0[link_idx, 1]
            p10 = s1[link_idx, 0]
            p11 = s1[link_idx, 1]
            for v in v_values:
                start = (1.0 - v) * p00 + v * p10
                end = (1.0 - v) * p01 + v * p11
                for u in u_values:
                    points.append((1.0 - u) * start + u * end)
                    link_ids.append(link_idx)
                    step_ids.append(step_idx)
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(link_ids, dtype=np.int64),
        np.asarray(step_ids, dtype=np.int64),
    )


def cumulative_swept_point_frame_indices(step_ids: np.ndarray) -> list[np.ndarray]:
    step_ids = np.asarray(step_ids, dtype=np.int64).reshape(-1)
    if len(step_ids) == 0:
        return []
    if np.any(step_ids < 0):
        raise ValueError("step_ids must be non-negative")
    return [np.flatnonzero(step_ids <= step_idx).astype(np.int64) for step_idx in range(int(step_ids.max()) + 1)]


def cumulative_projected_point_frames(
    uv: np.ndarray,
    valid: np.ndarray,
    colors: np.ndarray,
    step_ids: np.ndarray,
    width: int,
    height: int,
    point_radius: int,
    background: np.ndarray | None = None,
):
    from PIL import Image, ImageDraw

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    step_ids = np.asarray(step_ids, dtype=np.int64).reshape(-1)
    if not (len(uv) == len(valid) == len(colors) == len(step_ids)):
        raise ValueError("uv, valid, colors, and step_ids must have matching lengths")
    if len(step_ids) == 0:
        return
    if background is None:
        canvas = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    else:
        canvas = Image.fromarray(np.asarray(background, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    radius = int(max(point_radius, 0))
    for step_idx in range(int(step_ids.max()) + 1):
        point_indices = np.flatnonzero((step_ids == step_idx) & valid)
        for point_idx in point_indices:
            x, y = uv[point_idx]
            if x < -radius or x >= width + radius or y < -radius or y >= height + radius:
                continue
            color = tuple(int(c) for c in colors[point_idx])
            if radius <= 0:
                draw.point((float(x), float(y)), fill=color)
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        yield np.asarray(canvas, dtype=np.uint8).copy()


def resolve_body_ids(sim, body_names: tuple[str, ...]) -> np.ndarray:
    ids = []
    missing = []
    for name in body_names:
        try:
            ids.append(int(sim.model.body_name2id(name)))
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing robot anchor bodies in LIBERO model: {missing}")
    return np.asarray(ids, dtype=np.int64)


def render_camera_rgb(sim, camera_name: str, width: int, height: int) -> np.ndarray:
    rgb = sim.render(camera_name=camera_name, width=width, height=height, depth=False)
    if isinstance(rgb, (tuple, list)):
        rgb = rgb[0]
    return np.asarray(rgb, dtype=np.uint8)[::-1]


def project_world_points_to_camera_pixels(
    sim,
    camera_name: str,
    width: int,
    height: int,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if libero_pc.camera_utils is None:
        raise RuntimeError("LIBERO camera utilities were not loaded.")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    intrinsic = libero_pc.camera_utils.get_camera_intrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    camera_to_world = libero_pc.camera_utils.get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    world_to_camera = np.linalg.inv(camera_to_world)
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera_points = (world_to_camera @ hom.T).T[:, :3]
    z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (z > 1e-6)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = intrinsic[0, 0] * camera_points[valid, 0] / z[valid] + intrinsic[0, 2]
    uv[valid, 1] = intrinsic[1, 1] * camera_points[valid, 1] / z[valid] + intrinsic[1, 2]
    return uv, valid


def point_colors(link_ids: np.ndarray) -> np.ndarray:
    link_ids = np.asarray(link_ids, dtype=np.int64)
    return (LINK_COLORS[link_ids % len(LINK_COLORS), :3] * 255.0).astype(np.uint8)


def projected_point_image(
    sim,
    camera_name: str,
    width: int,
    height: int,
    points: np.ndarray,
    colors: np.ndarray,
    point_radius: int = 2,
    background: np.ndarray | None = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    if background is None:
        canvas = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    else:
        canvas = Image.fromarray(np.asarray(background, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    uv, valid = project_world_points_to_camera_pixels(sim, camera_name, width, height, points)
    order = np.argsort(points[:, 0])[::-1]
    radius = int(max(point_radius, 1))
    for idx in order:
        if not valid[idx]:
            continue
        x, y = uv[idx]
        if x < -radius or x >= width + radius or y < -radius or y >= height + radius:
            continue
        color = tuple(int(c) for c in colors[idx])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return np.asarray(canvas, dtype=np.uint8)


def save_frontview_projected_points(
    path: Path,
    overlay_path: Path,
    env,
    points: np.ndarray,
    link_ids: np.ndarray,
    width: int,
    height: int,
    point_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    colors = point_colors(link_ids)
    rgb = render_camera_rgb(env.sim, "frontview", width, height)
    radius = max(1, int(round(point_size / 2.0)))
    point_image = projected_point_image(env.sim, "frontview", width, height, points, colors, radius)
    point_overlay = projected_point_image(env.sim, "frontview", width, height, points, colors, radius, background=rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(point_image).save(path)
    Image.fromarray(point_overlay).save(overlay_path)
    return point_image, point_overlay


def write_rgb_frames_mp4(path: Path, frames, fps: int) -> None:
    iterator = iter(frames)
    try:
        first_frame = np.asarray(next(iterator), dtype=np.uint8)
    except StopIteration as exc:
        raise ValueError("cannot save MP4 with no frames") from exc

    height, width = first_frame.shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(int(fps), 1)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "mpeg4",
        "-q:v",
        "4",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("Saving MP4 video requires the ffmpeg executable to be installed.") from exc

    assert process.stdin is not None
    try:
        process.stdin.write(np.ascontiguousarray(first_frame[..., :3]).tobytes())
        for frame in iterator:
            frame = np.asarray(frame, dtype=np.uint8)
            process.stdin.write(np.ascontiguousarray(frame[..., :3]).tobytes())
    except Exception as exc:
        process.kill()
        raise RuntimeError("Failed while writing MP4 video frames.") from exc
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass

    stderr = process.stderr.read() if process.stderr is not None else b""
    if process.stdout is not None:
        process.stdout.read()
    returncode = process.wait()
    if returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-1200:]
        raise RuntimeError(f"ffmpeg failed while encoding swept point video: {detail}")


def save_cumulative_frontview_projected_points_video(
    path: Path,
    env,
    points: np.ndarray,
    link_ids: np.ndarray,
    step_ids: np.ndarray,
    width: int,
    height: int,
    point_size: float,
    fps: int,
) -> None:
    colors = point_colors(link_ids)
    rgb = render_camera_rgb(env.sim, "frontview", width, height)
    uv, valid = project_world_points_to_camera_pixels(env.sim, "frontview", width, height, points)
    radius = max(1, int(round(point_size / 2.0)))
    frames = cumulative_projected_point_frames(
        uv=uv,
        valid=valid,
        colors=colors,
        step_ids=step_ids,
        width=width,
        height=height,
        point_radius=radius,
        background=rgb,
    )
    write_rgb_frames_mp4(path, frames, fps=fps)


def set_equal_axes(ax, points: np.ndarray, margin: float = 0.06) -> None:
    mins = points.min(axis=0) - margin
    maxs = points.max(axis=0) + margin
    centers = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def save_frontview_3d_plot(
    path: Path,
    points: np.ndarray,
    link_ids: np.ndarray,
    elev: float,
    azim: float,
    title: str = "LIBERO front-view robot skeleton swept points",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = LINK_COLORS[np.asarray(link_ids, dtype=np.int64) % len(LINK_COLORS), :3]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1.0, alpha=0.82, linewidths=0)
    set_equal_axes(ax, points)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def matplotlib_figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def save_cumulative_swept_points_video(
    path: Path,
    points: np.ndarray,
    link_ids: np.ndarray,
    step_ids: np.ndarray,
    fps: int,
    elev: float,
    azim: float,
    dpi: int,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = max(int(fps), 1)
    frame_indices = cumulative_swept_point_frame_indices(step_ids)
    if not frame_indices:
        raise ValueError("cannot save video for an empty swept point cloud")

    all_points = np.asarray(points, dtype=np.float64)
    all_link_ids = np.asarray(link_ids, dtype=np.int64)
    mins = all_points.min(axis=0) - 0.06
    maxs = all_points.max(axis=0) + 0.06
    centers = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-3)
    axis_limits = (
        (centers[0] - radius, centers[0] + radius),
        (centers[1] - radius, centers[1] + radius),
        (centers[2] - radius, centers[2] + radius),
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    def render_frame(frame_idx: int, indices: np.ndarray, total_frames: int) -> np.ndarray:
        fig = None
        try:
            frame_points = all_points[indices]
            frame_colors = LINK_COLORS[all_link_ids[indices] % len(LINK_COLORS), :3]
            fig = plt.figure(figsize=(8, 7), dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                frame_points[:, 0],
                frame_points[:, 1],
                frame_points[:, 2],
                c=frame_colors,
                s=1.0,
                alpha=0.82,
                linewidths=0,
            )
            ax.set_xlim(*axis_limits[0])
            ax.set_ylim(*axis_limits[1])
            ax.set_zlim(*axis_limits[2])
            try:
                ax.set_box_aspect((1, 1, 1))
            except AttributeError:
                pass
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("world x")
            ax.set_ylabel("world y")
            ax.set_zlabel("world z")
            ax.set_title(f"{title} | frame {frame_idx + 1}/{total_frames}")
            fig.tight_layout()
            return matplotlib_figure_to_rgb(fig)
        finally:
            if fig is not None:
                plt.close(fig)

    total_frames = len(frame_indices)
    frames = (
        render_frame(frame_idx, indices, total_frames)
        for frame_idx, indices in enumerate(frame_indices)
    )
    write_rgb_frames_mp4(path, frames, fps=fps)


def main() -> None:
    args = parse_args()
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_runtime_dependencies()
    bddl_file, task_name, init_state = resolve_task(args)
    env = create_env(bddl_file, args.frontview_width, args.frontview_height, ["frontview"])
    try:
        settle_scene(env, init_state, num_steps_wait=10)
        qpos_indices = get_arm_qpos_indices(env)
        low, high = joint_limits(env.sim, qpos_indices)
        action_dim = len(qpos_indices)
        start_joint_vector = (
            normalize_joint_vector(load_array(args.joint_vector_file, key="joint_vector"), action_dim)
            if args.joint_vector_file is not None
            else np.asarray(env.sim.data.qpos[qpos_indices], dtype=np.float64).copy()
        )
        start_joint_vector = np.clip(start_joint_vector, low, high)
        action_chunk = (
            normalize_action_chunk(load_array(args.action_chunk_file, key="actions"), action_dim)
            if args.action_chunk_file is not None
            else make_random_action_chunk(args.horizon, action_dim, args.action_scale, args.seed)
        )

        anchor_body_ids = resolve_body_ids(env.sim, DEFAULT_PANDA_ANCHOR_BODIES)
        gripper_body_id = int(env.sim.model.body_name2id(DEFAULT_GRIPPER_BODY))
        joint_path = integrate_joint_path(
            start_joint_vector,
            action_chunk,
            low,
            high,
            samples_per_action=args.samples_per_action,
        )
        anchor_path, gripper_rotation_path = fk_path(
            env,
            qpos_indices,
            anchor_body_ids,
            gripper_body_id,
            joint_path,
        )
        skeleton_geom_ids = np.zeros((0,), dtype=np.int64)
        skeleton_geom_kinds = np.asarray([], dtype="<U1")
        skeleton_geom_names = np.asarray([], dtype="<U1")
        skeleton_segment_color_ids = np.arange(len(DEFAULT_LINK_NAMES), dtype=np.int64)
        if args.skeleton_source == "geom":
            skeleton_geom_ids = np.asarray(sorted(libero_pc.find_robot_geoms(env)), dtype=np.int64)
            (
                segment_path,
                skeleton_geom_kinds,
                skeleton_geom_names,
                skeleton_segment_color_ids,
                link_names,
            ) = geom_skeleton_path(
                env,
                qpos_indices,
                skeleton_geom_ids,
                joint_path,
            )
        else:
            segment_path = build_link_segments(anchor_path, gripper_rotation_path, args.gripper_width)
            link_names = np.asarray(DEFAULT_LINK_NAMES)
            skeleton_segment_color_ids = np.arange(len(link_names), dtype=np.int64)
        panels, panel_link_ids, panel_step_ids = build_swept_panels(segment_path)
        swept_points, swept_segment_ids, swept_step_ids = sample_swept_surface_points(
            segment_path,
            link_samples=args.swept_point_link_samples,
            time_samples=args.swept_point_time_samples,
        )
        swept_link_ids = skeleton_segment_color_ids[swept_segment_ids]
        panel_color_ids = skeleton_segment_color_ids[panel_link_ids]
        collision_result = None
        collision_title = "collision: not checked"
        if args.safe_space is not None:
            safe_space = load_npz_dict(args.safe_space)
            collision_result = detect_swept_obstacle_collision(
                swept_points,
                safe_space,
                collision_margin=args.collision_margin,
            )
            collision_title = (
                f"collision: {'YES' if collision_result.collides else 'NO'} "
                f"({collision_result.collision_point_count} swept points, {collision_result.method})"
            )

        safe_task_name = (args.name or task_name).replace("/", "_")
        prefix = args.output_dir / f"{safe_task_name}_joint_link_swept"
        point_png = prefix.with_name(f"{prefix.name}_frontview_swept_points.png")
        overlay_png = prefix.with_name(f"{prefix.name}_frontview_swept_points_overlay.png")
        plot_png = prefix.with_name(f"{prefix.name}_frontview_swept_points_3d.png")
        video_path = prefix.with_name(f"{prefix.name}_frontview_swept_points.mp4")
        npz_path = prefix.with_suffix(".npz")
        action_path = prefix.with_name(f"{prefix.name}_actions.npy")
        joint_path_path = prefix.with_name(f"{prefix.name}_joint_path.npy")

        point_image, point_overlay = save_frontview_projected_points(
            point_png,
            overlay_png,
            env,
            swept_points,
            swept_link_ids,
            args.frontview_width,
            args.frontview_height,
            args.frontview_point_size,
        )
        save_frontview_3d_plot(
            plot_png,
            swept_points,
            swept_link_ids,
            args.plot_elev,
            args.plot_azim,
            title=f"LIBERO robot skeleton swept points - {collision_title}",
        )
        if args.save_video:
            save_cumulative_frontview_projected_points_video(
                video_path,
                env,
                swept_points,
                swept_link_ids,
                swept_step_ids,
                width=args.frontview_width,
                height=args.frontview_height,
                point_size=args.frontview_point_size,
                fps=args.video_fps,
            )
        np.save(action_path, action_chunk.astype(np.float32))
        np.save(joint_path_path, joint_path.astype(np.float32))
        np.savez_compressed(
            npz_path,
            start_joint_vector=start_joint_vector.astype(np.float32),
            action_chunk=action_chunk.astype(np.float32),
            joint_path=joint_path.astype(np.float32),
            anchor_path=anchor_path.astype(np.float32),
            gripper_rotation_path=gripper_rotation_path.astype(np.float32),
            segment_path=segment_path.astype(np.float32),
            panels=panels.astype(np.float32),
            panel_link_ids=panel_link_ids.astype(np.int16),
            panel_color_ids=panel_color_ids.astype(np.int16),
            panel_step_ids=panel_step_ids.astype(np.int16),
            swept_surface_points=swept_points.astype(np.float32),
            swept_surface_point_link_ids=swept_link_ids.astype(np.int16),
            swept_surface_point_segment_ids=swept_segment_ids.astype(np.int16),
            swept_surface_point_step_ids=swept_step_ids.astype(np.int16),
            frontview_swept_points=point_image.astype(np.uint8),
            frontview_swept_points_overlay=point_overlay.astype(np.uint8),
            frontview_swept_points_video_path=np.asarray(str(video_path) if args.save_video else ""),
            video_fps=np.array(args.video_fps, dtype=np.int32),
            collision_checked=np.array(collision_result is not None),
            collision=np.array(False if collision_result is None else collision_result.collides),
            collision_method=np.asarray("" if collision_result is None else collision_result.method),
            collision_margin=np.array(args.collision_margin, dtype=np.float32),
            collision_point_count=np.array(
                0 if collision_result is None else collision_result.collision_point_count,
                dtype=np.int64,
            ),
            collision_swept_point_indices=(
                np.zeros((0,), dtype=np.int64)
                if collision_result is None
                else collision_result.colliding_point_indices.astype(np.int64)
            ),
            skeleton_source=np.asarray(args.skeleton_source),
            skeleton_geom_ids=skeleton_geom_ids.astype(np.int64),
            skeleton_geom_kinds=skeleton_geom_kinds,
            skeleton_geom_names=skeleton_geom_names,
            skeleton_segment_color_ids=skeleton_segment_color_ids.astype(np.int16),
            link_names=link_names,
            link_anchor_bodies=np.asarray(DEFAULT_PANDA_ANCHOR_BODIES),
            gripper_width=np.array(args.gripper_width, dtype=np.float32),
        )

        print(f"[info] task: {task_name}")
        print(f"[info] arm joints: {action_dim}")
        print(f"[info] start_joint_vector: {np.array2string(start_joint_vector, precision=4)}")
        print(f"[info] action_chunk shape: {action_chunk.shape}")
        print(f"[info] joint FK samples: {joint_path.shape[0]}")
        print(f"[info] skeleton source: {args.skeleton_source}")
        if args.skeleton_source == "geom":
            print(f"[info] robot geom skeleton segments per FK sample: {len(skeleton_geom_ids)}")
        print(f"[info] swept panels: {panels.shape[0]}")
        print(f"[info] skeleton swept surface points: {swept_points.shape[0]}")
        if collision_result is not None:
            print(
                "[info] collision: "
                f"{'YES' if collision_result.collides else 'NO'} "
                f"({collision_result.collision_point_count} swept points, method={collision_result.method})"
            )
        print(f"[done] saved frontview projected points: {point_png}")
        print(f"[done] saved frontview projected overlay: {overlay_png}")
        print(f"[done] saved frontview 3D point plot: {plot_png}")
        if args.save_video:
            print(f"[done] saved cumulative swept point video: {video_path}")
        print(f"[done] saved swept surface data: {npz_path}")
        print(f"[done] saved actions: {action_path}")
        print(f"[done] saved joint path: {joint_path_path}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
