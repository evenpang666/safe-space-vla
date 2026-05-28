#!/usr/bin/env python3
"""Generate LIBERO joint-link swept point clouds from FK anchor segments.

This is the LIBERO counterpart of visualize_robosuite_joint_swept_surfaces.py:
it does not sample robot mesh geometry. Instead, it integrates a joint-delta
chunk, builds line segments between consecutive robot link anchors, and samples
the swept ruled surfaces made by those segments over time.
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
    parser.add_argument("--horizon", type=int, default=50, help="Random joint-delta chunk length.")
    parser.add_argument("--action-scale", type=float, default=0.06, help="Random joint delta range in radians.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for generated joint deltas.")
    parser.add_argument("--samples-per-action", type=int, default=8, help="Interpolated FK intervals per action.")
    parser.add_argument("--joint-vector-file", type=Path, default=None, help="Optional start joint vector file.")
    parser.add_argument("--gripper-width", type=float, default=0.08, help="Virtual gripper segment length in meters.")
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
    ax.set_title("LIBERO front-view joint-link swept points")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


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
        segment_path = build_link_segments(anchor_path, gripper_rotation_path, args.gripper_width)
        panels, panel_link_ids, panel_step_ids = build_swept_panels(segment_path)
        swept_points, swept_link_ids, swept_step_ids = sample_swept_surface_points(
            segment_path,
            link_samples=args.swept_point_link_samples,
            time_samples=args.swept_point_time_samples,
        )

        safe_task_name = (args.name or task_name).replace("/", "_")
        prefix = args.output_dir / f"{safe_task_name}_joint_link_swept"
        point_png = prefix.with_name(f"{prefix.name}_frontview_swept_points.png")
        overlay_png = prefix.with_name(f"{prefix.name}_frontview_swept_points_overlay.png")
        plot_png = prefix.with_name(f"{prefix.name}_frontview_swept_points_3d.png")
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
        save_frontview_3d_plot(plot_png, swept_points, swept_link_ids, args.plot_elev, args.plot_azim)
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
            swept_surface_point_link_ids=swept_link_ids.astype(np.int16),
            swept_surface_point_step_ids=swept_step_ids.astype(np.int16),
            frontview_swept_points=point_image.astype(np.uint8),
            frontview_swept_points_overlay=point_overlay.astype(np.uint8),
            link_names=np.asarray(DEFAULT_LINK_NAMES),
            link_anchor_bodies=np.asarray(DEFAULT_PANDA_ANCHOR_BODIES),
            gripper_width=np.array(args.gripper_width, dtype=np.float32),
        )

        print(f"[info] task: {task_name}")
        print(f"[info] arm joints: {action_dim}")
        print(f"[info] start_joint_vector: {np.array2string(start_joint_vector, precision=4)}")
        print(f"[info] action_chunk shape: {action_chunk.shape}")
        print(f"[info] joint FK samples: {joint_path.shape[0]}")
        print(f"[info] swept panels: {panels.shape[0]}")
        print(f"[info] joint-link swept surface points: {swept_points.shape[0]}")
        print(f"[done] saved frontview projected points: {point_png}")
        print(f"[done] saved frontview projected overlay: {overlay_png}")
        print(f"[done] saved frontview 3D point plot: {plot_png}")
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
