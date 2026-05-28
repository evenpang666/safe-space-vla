#!/usr/bin/env python3
"""Visualize joint-space swept link surfaces in the upright-blocks scene.

The swept region here is not sampled from robot mesh geometry. It is built from
UR5e forward kinematics anchors: each action is interpreted as a 6-DoF joint
delta, integrated from the current joint vector, and each consecutive pair of
FK samples forms ruled quadrilateral link-sweep panels. The first six panels
come from arm links. The seventh is a virtual gripper-width segment centered at
the EEF and aligned with the gripper frame's local x axis.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_robosuite_upright_blocks_scene import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    UprightBlocksLift,
)
from robosuite.controllers import load_controller_config  # noqa: E402
from robosuite.utils import camera_utils  # noqa: E402


LINK_ANCHOR_BODIES = (
    "robot0_shoulder_link",
    "robot0_upper_arm_link",
    "robot0_forearm_link",
    "robot0_wrist_1_link",
    "robot0_wrist_2_link",
    "robot0_wrist_3_link",
    "gripper0_eef",
)

LINK_NAMES = (
    "shoulder_upper",
    "upper_forearm",
    "forearm_wrist1",
    "wrist1_wrist2",
    "wrist2_wrist3",
    "wrist3_eef",
    "gripper_width",
)

LINK_COLORS = np.array(
    [
        [0.20, 0.47, 0.85, 1.0],
        [0.12, 0.62, 0.52, 1.0],
        [0.95, 0.62, 0.16, 1.0],
        [0.86, 0.25, 0.25, 1.0],
        [0.55, 0.35, 0.80, 1.0],
        [0.35, 0.35, 0.35, 1.0],
        [0.05, 0.68, 0.90, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-chunk-file",
        type=Path,
        default=None,
        help="Optional .npy/.npz/.json/.csv action chunk. Uses the first 6 dims as joint deltas.",
    )
    parser.add_argument("--horizon", type=int, default=8, help="Random action chunk length.")
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.12,
        help="Uniform random joint delta range [-scale, scale] radians.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the action chunk.")
    parser.add_argument(
        "--samples-per-action",
        type=int,
        default=8,
        help="Number of interpolated FK intervals for each action.",
    )
    parser.add_argument(
        "--joint-vector-file",
        type=Path,
        default=None,
        help="Optional .npy/.npz/.json/.csv 6-DoF start joint vector. Defaults to scene reset qpos.",
    )
    parser.add_argument(
        "--gripper-width",
        type=float,
        default=0.085,
        help="Length of the virtual Robotiq gripper-width segment in meters.",
    )
    parser.add_argument("--width", type=float, default=9.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=7.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=170, help="Output figure DPI.")
    parser.add_argument("--elev", type=float, default=24.0, help="Matplotlib 3D elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="Matplotlib 3D azimuth.")
    parser.add_argument("--frontview-width", type=int, default=384, help="Front camera sweep image width in pixels.")
    parser.add_argument("--frontview-height", type=int, default=384, help="Front camera sweep image height in pixels.")
    parser.add_argument(
        "--frontview-overlay-alpha",
        type=float,
        default=0.45,
        help="Opacity of the projected sweep mask in the front camera overlay.",
    )
    parser.add_argument(
        "--swept-point-link-samples",
        type=int,
        default=8,
        help="Number of sparse points sampled along each joint-link segment on every swept panel.",
    )
    parser.add_argument(
        "--swept-point-time-samples",
        type=int,
        default=2,
        help="Number of sparse points sampled along each swept panel's time direction.",
    )
    parser.add_argument(
        "--frontview-point-size",
        type=float,
        default=5.0,
        help="Scatter marker size for front-view swept point previews.",
    )
    parser.add_argument(
        "--frontview-point-elev",
        type=float,
        default=18.0,
        help="Matplotlib elevation for the front-view 3D swept point preview.",
    )
    parser.add_argument(
        "--frontview-point-azim",
        type=float,
        default=0.0,
        help="Matplotlib azimuth for the front-view 3D swept point preview.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="ur5e_upright_blocks_joint_swept_surfaces")
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


def make_random_action_chunk(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    return rng.uniform(
        low=-args.action_scale,
        high=args.action_scale,
        size=(args.horizon, 6),
    ).astype(np.float64)


def normalize_action_chunk(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2:
        raise ValueError(f"Action chunk must have shape (T, D), got {actions.shape}")
    if actions.shape[1] < 6:
        raise ValueError(f"Action chunk must contain at least 6 joint deltas, got {actions.shape[1]}")
    if actions.shape[1] > 6:
        print(f"[warn] action dim {actions.shape[1]} > 6; using first 6 dims as joint deltas.")
        actions = actions[:, :6]
    return actions


def normalize_joint_vector(joint_vector: np.ndarray) -> np.ndarray:
    joint_vector = np.asarray(joint_vector, dtype=np.float64).reshape(-1)
    if joint_vector.size < 6:
        raise ValueError(f"Joint vector must contain at least 6 values, got {joint_vector.size}")
    if joint_vector.size > 6:
        print(f"[warn] joint vector dim {joint_vector.size} > 6; using first 6 values.")
        joint_vector = joint_vector[:6]
    return joint_vector


def get_joint_qpos_indices(env) -> np.ndarray:
    robot = env.robots[0]
    indexes = getattr(robot, "_ref_joint_pos_indexes", None)
    if indexes is None:
        indexes = getattr(robot, "joint_indexes", None)
    if indexes is None:
        raise RuntimeError("Could not find UR5e joint qpos indexes from robosuite robot.")
    indexes = np.asarray(indexes, dtype=np.int64)
    if indexes.size < 6:
        raise RuntimeError(f"Expected at least 6 arm joint indexes, got {indexes.size}")
    return indexes[:6]


def joint_limits(env, qpos_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = env.sim.model
    lows = np.full(6, -np.inf, dtype=np.float64)
    highs = np.full(6, np.inf, dtype=np.float64)
    for joint_id in range(int(model.njnt)):
        qpos_adr = int(model.jnt_qposadr[joint_id])
        matches = np.where(qpos_indices == qpos_adr)[0]
        if len(matches) == 0:
            continue
        out_idx = int(matches[0])
        if bool(model.jnt_limited[joint_id]):
            lows[out_idx], highs[out_idx] = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
    return lows, highs


def set_arm_joint_vector(env, qpos_indices: np.ndarray, joint_vector: np.ndarray) -> None:
    env.sim.data.qpos[qpos_indices] = joint_vector
    env.sim.data.qvel[:6] = 0.0
    env.sim.forward()


def fk_anchor_points(env, anchor_body_ids: np.ndarray) -> np.ndarray:
    return np.asarray(env.sim.data.body_xpos[anchor_body_ids], dtype=np.float64).copy()


def body_rotation(env, body_id: int) -> np.ndarray:
    return np.asarray(env.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3).copy()


def integrate_joint_path(
    start_joint_vector: np.ndarray,
    action_chunk: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    samples_per_action: int,
) -> np.ndarray:
    if samples_per_action < 1:
        raise ValueError("--samples-per-action must be >= 1")
    q = start_joint_vector.astype(np.float64, copy=True)
    path = [q.copy()]
    for action in action_chunk:
        q_next = np.clip(q + action, low, high)
        for sample_idx in range(1, samples_per_action + 1):
            alpha = sample_idx / float(samples_per_action)
            path.append((1.0 - alpha) * q + alpha * q_next)
        q = q_next
    return np.asarray(path, dtype=np.float64)


def fk_path(
    env,
    qpos_indices: np.ndarray,
    anchor_body_ids: np.ndarray,
    gripper_body_id: int,
    joint_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    gripper_rotations = []
    for q in joint_path:
        set_arm_joint_vector(env, qpos_indices, q)
        points.append(fk_anchor_points(env, anchor_body_ids))
        gripper_rotations.append(body_rotation(env, gripper_body_id))
    return np.asarray(points, dtype=np.float64), np.asarray(gripper_rotations, dtype=np.float64)


def gripper_x_axis_segment(
    anchor_points: np.ndarray,
    gripper_rotation: np.ndarray,
    gripper_width: float,
) -> np.ndarray:
    eef = anchor_points[-1]
    direction = gripper_rotation[:, 0]
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        raise RuntimeError("Gripper x axis has near-zero norm.")
    direction = direction / direction_norm

    half = 0.5 * float(gripper_width)
    return np.asarray([eef - half * direction, eef + half * direction], dtype=np.float64)


def build_link_segments(
    anchor_path: np.ndarray,
    gripper_rotation_path: np.ndarray,
    gripper_width: float,
) -> np.ndarray:
    if gripper_width <= 0.0:
        raise ValueError("--gripper-width must be positive")
    segments = np.empty((anchor_path.shape[0], 7, 2, 3), dtype=np.float64)
    for step_idx in range(anchor_path.shape[0]):
        for link_idx in range(6):
            segments[step_idx, link_idx, 0] = anchor_path[step_idx, link_idx]
            segments[step_idx, link_idx, 1] = anchor_path[step_idx, link_idx + 1]
        segments[step_idx, 6] = gripper_x_axis_segment(
            anchor_path[step_idx],
            gripper_rotation_path[step_idx],
            gripper_width,
        )
    return segments


def build_swept_panels(segment_path: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panels = []
    panel_link_ids = []
    panel_step_ids = []
    num_links = segment_path.shape[1]
    for step_idx in range(segment_path.shape[0] - 1):
        s0 = segment_path[step_idx]
        s1 = segment_path[step_idx + 1]
        for link_idx in range(num_links):
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


def cuboid_faces(center: np.ndarray, half_size: np.ndarray, rotation: np.ndarray | None = None) -> list[np.ndarray]:
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    corners = signs * half_size[None, :]
    if rotation is not None:
        corners = corners @ rotation.T
    corners = corners + center[None, :]
    face_ids = ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])
    return [corners[np.asarray(ids)] for ids in face_ids]


def cylinder_faces(center: np.ndarray, radius: float, half_height: float, segments: int = 48) -> list[np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circle = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    top = np.column_stack([circle, np.full(segments, half_height)]) + center[None, :]
    bottom = np.column_stack([circle, np.full(segments, -half_height)]) + center[None, :]
    faces = [top, bottom[::-1]]
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append(np.asarray([bottom[idx], bottom[nxt], top[nxt], top[idx]], dtype=np.float64))
    return faces


def draw_scene(ax, env) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    table_center = np.array([0.0, 0.0, env.table_offset[2] - env.table_full_size[2] / 2.0])
    table_faces = cuboid_faces(table_center, env.table_full_size / 2.0)
    ax.add_collection3d(
        Poly3DCollection(table_faces, facecolor=(0.72, 0.72, 0.67, 0.25), edgecolor=(0.55, 0.55, 0.50, 0.45), linewidths=0.4)
    )

    object_specs = [
        ("red_cube", env.red_cube_body_id, env.red_cube_size, (1.0, 0.02, 0.02, 0.75)),
        ("left_yellow_slab", env.left_slab_body_id, env.yellow_slab_size, (1.0, 0.82, 0.02, 0.55)),
        ("right_yellow_slab", env.right_slab_body_id, env.yellow_slab_size, (1.0, 0.82, 0.02, 0.55)),
    ]
    for _, body_id, half_size, color in object_specs:
        center = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64)
        rotation = np.asarray(env.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
        faces = cuboid_faces(center, half_size, rotation)
        ax.add_collection3d(
            Poly3DCollection(faces, facecolor=color, edgecolor=(0.08, 0.08, 0.08, 0.35), linewidths=0.3)
        )

    plate_center = np.asarray(env.sim.data.body_xpos[env.target_plate_body_id], dtype=np.float64)
    plate_faces = cylinder_faces(plate_center, float(env.plate_size[0]), float(env.plate_size[1]))
    ax.add_collection3d(
        Poly3DCollection(plate_faces, facecolor=(0.86, 0.94, 1.0, 0.45), edgecolor=(0.30, 0.45, 0.60, 0.25), linewidths=0.25)
    )


def set_axes_equal(ax, points: np.ndarray, margin: float = 0.08) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins)) + 2.0 * margin
    half = max(span / 2.0, 0.05)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(max(0.72, center[2] - half), center[2] + half)


def save_visualization(
    path: Path,
    env,
    anchor_path: np.ndarray,
    segment_path: np.ndarray,
    panels: np.ndarray,
    panel_link_ids: np.ndarray,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")

    draw_scene(ax, env)

    for link_idx, link_name in enumerate(LINK_NAMES):
        selected = panels[panel_link_ids == link_idx]
        if len(selected) == 0:
            continue
        color = LINK_COLORS[link_idx].copy()
        color[3] = 0.22
        edge = LINK_COLORS[link_idx].copy()
        edge[3] = 0.45
        ax.add_collection3d(
            Poly3DCollection(selected, facecolor=color, edgecolor=edge, linewidths=0.25)
        )
        mid = selected.reshape(-1, 3).mean(axis=0)
        ax.text(mid[0], mid[1], mid[2], str(link_idx + 1), color=LINK_COLORS[link_idx], fontsize=8)

    initial = anchor_path[0]
    final = anchor_path[-1]
    ax.plot(initial[:, 0], initial[:, 1], initial[:, 2], color="black", linewidth=2.0, marker="o", markersize=3)
    ax.plot(final[:, 0], final[:, 1], final[:, 2], color="white", linewidth=1.8, marker="o", markersize=3)
    ax.plot(final[:, 0], final[:, 1], final[:, 2], color="black", linewidth=0.8, marker="o", markersize=2)
    for line, color, width in (
        (segment_path[0, 6], "black", 2.0),
        (segment_path[-1, 6], "white", 2.0),
        (segment_path[-1, 6], "black", 0.8),
    ):
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=width, marker="o", markersize=3)

    table_top = env.table_offset[2]
    table_x = env.table_full_size[0] / 2.0
    table_y = env.table_full_size[1] / 2.0
    ax.plot(
        [-table_x, table_x, table_x, -table_x, -table_x],
        [-table_y, -table_y, table_y, table_y, -table_y],
        [table_top] * 5,
        color=(0.1, 0.1, 0.1, 0.55),
        linestyle="--",
        linewidth=1.0,
    )

    all_points = np.concatenate(
        [
            anchor_path.reshape(-1, 3),
            segment_path.reshape(-1, 3),
            panels.reshape(-1, 3),
            np.array(
                [
                    [-table_x, -table_y, table_top],
                    [table_x, table_y, table_top + 0.42],
                ],
                dtype=np.float64,
            ),
        ],
        axis=0,
    )
    set_axes_equal(ax, all_points)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("UR5e joint-delta swept link surfaces")

    handles = [
        Line2D([0], [0], color=LINK_COLORS[i], lw=5, alpha=0.75, label=f"{i + 1}: {name}")
        for i, name in enumerate(LINK_NAMES)
    ]
    handles.extend(
        [
            Line2D([0], [0], color="black", lw=2.0, label="initial skeleton"),
            Line2D([0], [0], color="white", markeredgecolor="black", lw=2.0, label="final skeleton"),
        ]
    )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), framealpha=0.88, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    intrinsic = camera_utils.get_camera_intrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    camera_to_world = camera_utils.get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    world_to_camera = np.linalg.inv(camera_to_world)

    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera_points = (world_to_camera @ hom.T).T[:, :3]
    z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (z > 1e-6)

    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = intrinsic[0, 0] * camera_points[valid, 0] / z[valid] + intrinsic[0, 2]
    uv[valid, 1] = intrinsic[1, 1] * camera_points[valid, 1] / z[valid] + intrinsic[1, 2]
    return uv, valid


def rasterize_polygons(polygons: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    mask_image = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in np.asarray(polygons, dtype=np.float64):
        finite = np.isfinite(polygon).all(axis=1)
        if np.count_nonzero(finite) < 3:
            continue
        coords = [tuple(point) for point in polygon[finite]]
        draw.polygon(coords, fill=255)
    return np.asarray(mask_image, dtype=np.uint8)


def projected_swept_mask(
    sim,
    camera_name: str,
    width: int,
    height: int,
    panels: np.ndarray,
) -> np.ndarray:
    panel_points = np.asarray(panels, dtype=np.float64).reshape(-1, 3)
    uv, valid = project_world_points_to_camera_pixels(sim, camera_name, width, height, panel_points)
    uv_panels = uv.reshape(-1, 4, 2)
    valid_panels = valid.reshape(-1, 4)
    return rasterize_polygons(uv_panels[valid_panels.all(axis=1)], height=height, width=width)


def overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 80, 20),
    alpha: float = 0.45,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask_bool = np.asarray(mask) > 0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = rgb.astype(np.float32, copy=True)
    color_arr = np.asarray(color, dtype=np.float32)
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


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


def save_frontview_swept_point_outputs(
    env,
    output_prefix: Path,
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
    Image.fromarray(point_image).save(output_prefix.with_name(f"{output_prefix.name}_frontview_swept_points.png"))
    Image.fromarray(point_overlay).save(output_prefix.with_name(f"{output_prefix.name}_frontview_swept_points_overlay.png"))
    np.save(output_prefix.with_name(f"{output_prefix.name}_swept_surface_points.npy"), points.astype(np.float32))
    return point_image, point_overlay


def save_frontview_3d_point_plot(
    path: Path,
    env,
    points: np.ndarray,
    link_ids: np.ndarray,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    draw_scene(ax, env)
    colors = LINK_COLORS[np.asarray(link_ids, dtype=np.int64) % len(LINK_COLORS), :3]
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=float(args.frontview_point_size),
        alpha=0.85,
        depthshade=False,
    )
    set_axes_equal(ax, points, margin=0.10)
    ax.view_init(elev=args.frontview_point_elev, azim=args.frontview_point_azim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Sparse swept-surface points, front view")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_frontview_swept_outputs(
    env,
    panels: np.ndarray,
    output_prefix: Path,
    width: int,
    height: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from PIL import Image

    rgb = render_camera_rgb(env.sim, "frontview", width, height)
    mask = projected_swept_mask(env.sim, "frontview", width, height, panels)
    overlay = overlay_mask(rgb, mask, alpha=alpha)

    Image.fromarray(rgb).save(output_prefix.with_name(f"{output_prefix.name}_frontview_rgb.png"))
    Image.fromarray(mask).save(output_prefix.with_name(f"{output_prefix.name}_frontview_swept_mask.png"))
    Image.fromarray(overlay).save(output_prefix.with_name(f"{output_prefix.name}_frontview_swept_overlay.png"))
    np.save(output_prefix.with_name(f"{output_prefix.name}_frontview_swept_mask.npy"), mask)
    return rgb, mask, overlay


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = UprightBlocksLift(
        controller_configs=controller_config,
        camera_names="frontview",
        camera_widths=64,
        camera_heights=64,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        render_camera="frontview",
    )

    try:
        env.reset()
        for _ in range(20):
            env.step(np.zeros(env.action_dim, dtype=np.float64))

        qpos_indices = get_joint_qpos_indices(env)
        low, high = joint_limits(env, qpos_indices)
        start_joint_vector = (
            normalize_joint_vector(load_array(args.joint_vector_file, key="joint_vector"))
            if args.joint_vector_file is not None
            else np.asarray(env.sim.data.qpos[qpos_indices], dtype=np.float64).copy()
        )
        start_joint_vector = np.clip(start_joint_vector, low, high)

        action_chunk = (
            normalize_action_chunk(load_array(args.action_chunk_file, key="actions"))
            if args.action_chunk_file is not None
            else make_random_action_chunk(args)
        )

        anchor_body_ids = np.asarray([env.sim.model.body_name2id(name) for name in LINK_ANCHOR_BODIES], dtype=np.int64)
        gripper_body_id = env.sim.model.body_name2id("gripper0_eef")
        joint_path = integrate_joint_path(
            start_joint_vector=start_joint_vector,
            action_chunk=action_chunk,
            low=low,
            high=high,
            samples_per_action=args.samples_per_action,
        )
        anchor_path, gripper_rotation_path = fk_path(
            env,
            qpos_indices,
            anchor_body_ids,
            gripper_body_id,
            joint_path,
        )
        segment_path = build_link_segments(anchor_path, gripper_rotation_path, args.gripper_width)
        panels, panel_link_ids, panel_step_ids = build_swept_panels(segment_path)
        swept_points, swept_point_link_ids, swept_point_step_ids = sample_swept_surface_points(
            segment_path,
            link_samples=args.swept_point_link_samples,
            time_samples=args.swept_point_time_samples,
        )

        prefix = args.name
        png_path = args.output_dir / f"{prefix}.png"
        npz_path = args.output_dir / f"{prefix}.npz"
        action_path = args.output_dir / f"{prefix}_actions.npy"
        joint_path_path = args.output_dir / f"{prefix}_joint_path.npy"
        output_prefix = args.output_dir / prefix
        frontview_point_plot_path = args.output_dir / f"{prefix}_frontview_swept_points_3d.png"

        save_visualization(png_path, env, anchor_path, segment_path, panels, panel_link_ids, args)
        save_frontview_3d_point_plot(
            frontview_point_plot_path,
            env,
            swept_points,
            swept_point_link_ids,
            args,
        )
        frontview_swept_points, frontview_swept_points_overlay = save_frontview_swept_point_outputs(
            env=env,
            output_prefix=output_prefix,
            points=swept_points,
            link_ids=swept_point_link_ids,
            width=args.frontview_width,
            height=args.frontview_height,
            point_size=args.frontview_point_size,
        )
        frontview_rgb, frontview_swept_mask, frontview_swept_overlay = save_frontview_swept_outputs(
            env=env,
            panels=panels,
            output_prefix=output_prefix,
            width=args.frontview_width,
            height=args.frontview_height,
            alpha=args.frontview_overlay_alpha,
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
            panel_step_ids=panel_step_ids.astype(np.int16),
            swept_surface_points=swept_points.astype(np.float32),
            swept_surface_point_link_ids=swept_point_link_ids.astype(np.int16),
            swept_surface_point_step_ids=swept_point_step_ids.astype(np.int16),
            frontview_rgb=frontview_rgb.astype(np.uint8),
            frontview_swept_mask=frontview_swept_mask.astype(np.uint8),
            frontview_swept_overlay=frontview_swept_overlay.astype(np.uint8),
            frontview_swept_points=frontview_swept_points.astype(np.uint8),
            frontview_swept_points_overlay=frontview_swept_points_overlay.astype(np.uint8),
            link_names=np.asarray(LINK_NAMES),
            link_anchor_bodies=np.asarray(LINK_ANCHOR_BODIES),
            gripper_width=np.array(args.gripper_width, dtype=np.float32),
            frontview_width=np.array(args.frontview_width, dtype=np.int32),
            frontview_height=np.array(args.frontview_height, dtype=np.int32),
            swept_point_link_samples=np.array(args.swept_point_link_samples, dtype=np.int32),
            swept_point_time_samples=np.array(args.swept_point_time_samples, dtype=np.int32),
        )

        print(f"[info] start_joint_vector: {np.array2string(start_joint_vector, precision=4)}")
        print(f"[info] action_chunk shape: {action_chunk.shape}")
        print(f"[info] joint FK samples: {joint_path.shape[0]}")
        print(f"[info] gripper virtual segment width: {args.gripper_width:.4f} m")
        print(f"[info] swept panels: {panels.shape[0]} ({len(LINK_NAMES)} links x {joint_path.shape[0] - 1} intervals)")
        print(f"[info] sparse swept surface points: {swept_points.shape[0]}")
        print(f"[done] saved visualization: {png_path}")
        print(f"[done] saved frontview 3D point plot: {frontview_point_plot_path}")
        print(f"[done] saved frontview projected points: {output_prefix}_frontview_swept_points.png")
        print(f"[done] saved frontview projected point overlay: {output_prefix}_frontview_swept_points_overlay.png")
        print(f"[done] saved frontview sweep mask: {output_prefix}_frontview_swept_mask.png")
        print(f"[done] saved frontview sweep overlay: {output_prefix}_frontview_swept_overlay.png")
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
