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

        prefix = args.name
        png_path = args.output_dir / f"{prefix}.png"
        npz_path = args.output_dir / f"{prefix}.npz"
        action_path = args.output_dir / f"{prefix}_actions.npy"
        joint_path_path = args.output_dir / f"{prefix}_joint_path.npy"

        save_visualization(png_path, env, anchor_path, segment_path, panels, panel_link_ids, args)
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
            link_names=np.asarray(LINK_NAMES),
            link_anchor_bodies=np.asarray(LINK_ANCHOR_BODIES),
            gripper_width=np.array(args.gripper_width, dtype=np.float32),
        )

        print(f"[info] start_joint_vector: {np.array2string(start_joint_vector, precision=4)}")
        print(f"[info] action_chunk shape: {action_chunk.shape}")
        print(f"[info] joint FK samples: {joint_path.shape[0]}")
        print(f"[info] gripper virtual segment width: {args.gripper_width:.4f} m")
        print(f"[info] swept panels: {panels.shape[0]} ({len(LINK_NAMES)} links x {joint_path.shape[0] - 1} intervals)")
        print(f"[done] saved visualization: {png_path}")
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
