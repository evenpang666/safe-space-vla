#!/usr/bin/env python3
"""Visualize a LIBERO robot point-cloud sweep process from an action chunk.

The robot point cloud is sampled once in each robot geom's local frame and then
transformed through the simulator over a rollout. This keeps point identities
stable, so yellow temporal lines connect corresponding surface points between
consecutive action steps.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from libero_reconstruct_pointcloud import (  # noqa: E402
    create_env,
    load_runtime_dependencies,
    resolve_task,
    save_preview_png,
    settle_scene,
    write_ascii_ply,
)
from libero_robot_swept_pointcloud import (  # noqa: E402
    disable_nonrobot_collisions,
    find_robot_geoms,
    get_env_action_dim,
    load_action_chunk,
    make_random_action_chunk,
    normalize_action_chunk,
    sample_geom_local,
    step_color,
)


YELLOW_LINE_RGBA = np.array([1.0, 0.86, 0.05, 0.52], dtype=np.float32)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "libero_robot_pointcloud_sweep_process"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO benchmark suite to instantiate.",
    )
    parser.add_argument("--task-id", type=int, default=0, help="Task index in the suite.")
    parser.add_argument("--init-state-id", type=int, default=0, help="Initial-state index.")
    parser.add_argument("--bddl-file", type=Path, default=None, help="Optional direct .bddl path.")
    parser.add_argument(
        "--action-chunk-file",
        type=Path,
        default=None,
        help="Optional .npy/.npz/.json/.csv action chunk. Random actions are generated if omitted.",
    )
    parser.add_argument("--horizon", type=int, default=10, help="Random action chunk length.")
    parser.add_argument("--action-dim", type=int, default=7, help="Random action dimension.")
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.18,
        help="Uniform random action range [-scale, scale] for generated action chunks.",
    )
    parser.add_argument("--gripper-action", type=float, default=-1.0, help="Generated gripper action value.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="No-op steps after reset.")
    parser.add_argument("--points-per-geom", type=int, default=35, help="Template surface samples per robot geom.")
    parser.add_argument(
        "--geom-groups",
        type=int,
        nargs="+",
        default=None,
        help="Optional MuJoCo geom groups to include. By default all robot geoms are used.",
    )
    parser.add_argument(
        "--disable-nonrobot-collisions",
        action="store_true",
        help="Set non-robot geoms to non-colliding before rollout.",
    )
    parser.add_argument("--plot-elev", type=float, default=18.0, help="3D plot elevation.")
    parser.add_argument("--plot-azim", type=float, default=0.0, help="3D plot azimuth.")
    parser.add_argument("--plot-point-size", type=float, default=0.55, help="Scatter point size.")
    parser.add_argument(
        "--max-line-points",
        type=int,
        default=1800,
        help="Maximum template points whose temporal lines are drawn in the PNG. NPZ still saves all lines.",
    )
    parser.add_argument("--preview-points", type=int, default=90000, help="Maximum points in the generic preview.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--name", default=None, help="Output basename. Defaults to the task name.")
    parser.add_argument(
        "--mujoco-gl",
        choices=["egl", "osmesa", "glfw"],
        default=None,
        help="MuJoCo OpenGL backend. Must be set before robosuite import.",
    )
    return parser.parse_args()


def build_robot_surface_template(
    model,
    robot_geom_ids: Iterable[int],
    points_per_geom: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    local_chunks = []
    geom_chunks = []
    for geom_id in robot_geom_ids:
        local_points = sample_geom_local(model, int(geom_id), int(points_per_geom), rng)
        if len(local_points) == 0:
            continue
        local_chunks.append(local_points.astype(np.float32, copy=False))
        geom_chunks.append(np.full(len(local_points), int(geom_id), dtype=np.int32))
    if not local_chunks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.concatenate(local_chunks, axis=0), np.concatenate(geom_chunks, axis=0)


def transform_template_points(sim, local_points: np.ndarray, geom_ids: np.ndarray) -> np.ndarray:
    points = np.empty_like(local_points, dtype=np.float32)
    for geom_id in np.unique(geom_ids):
        mask = geom_ids == geom_id
        rotation = np.asarray(sim.data.geom_xmat[int(geom_id)], dtype=np.float32).reshape(3, 3)
        position = np.asarray(sim.data.geom_xpos[int(geom_id)], dtype=np.float32)
        points[mask] = local_points[mask] @ rotation.T + position
    return points


def temporal_line_segments(points_by_step: np.ndarray) -> np.ndarray:
    points_by_step = np.asarray(points_by_step, dtype=np.float32)
    if points_by_step.ndim != 3 or points_by_step.shape[-1] != 3:
        raise ValueError(f"points_by_step must have shape (T, N, 3), got {points_by_step.shape}")
    if points_by_step.shape[0] < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    starts = points_by_step[:-1]
    ends = points_by_step[1:]
    return np.stack([starts, ends], axis=2).reshape(-1, 2, 3)


def temporal_line_colors(num_segments: int) -> np.ndarray:
    return np.repeat(YELLOW_LINE_RGBA[None, :], int(num_segments), axis=0)


def flatten_step_points(points_by_step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    color_chunks = []
    num_steps = int(points_by_step.shape[0])
    for step_idx, points in enumerate(points_by_step):
        chunks.append(points)
        color_chunks.append(step_color(step_idx, num_steps, len(points)))
    return np.concatenate(chunks, axis=0), np.concatenate(color_chunks, axis=0)


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


def save_frontview_process_plot(
    path: Path,
    points_by_step: np.ndarray,
    colors_by_point: np.ndarray,
    line_segments: np.ndarray,
    max_line_points: int,
    point_size: float,
    elev: float,
    azim: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    flat_points = points_by_step.reshape(-1, 3)
    line_segments_to_draw = line_segments
    if max_line_points > 0 and points_by_step.shape[1] > max_line_points:
        rng = np.random.default_rng(0)
        point_idx = np.sort(rng.choice(points_by_step.shape[1], size=max_line_points, replace=False))
        line_segments_to_draw = temporal_line_segments(points_by_step[:, point_idx])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        flat_points[:, 0],
        flat_points[:, 1],
        flat_points[:, 2],
        c=colors_by_point.astype(np.float32) / 255.0,
        s=float(point_size),
        linewidths=0,
        alpha=0.72,
        depthshade=False,
    )
    if len(line_segments_to_draw) > 0:
        ax.add_collection3d(
            Line3DCollection(
                line_segments_to_draw,
                colors=temporal_line_colors(len(line_segments_to_draw)),
                linewidths=0.22,
            )
        )
    set_equal_axes(ax, flat_points)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    ax.set_title("LIBERO front-view robot point-cloud sweep process")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def rollout_pointcloud_process(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    load_runtime_dependencies()
    bddl_file, task_name, init_state = resolve_task(args)
    env = create_env(bddl_file, width=64, height=64, camera_names=["frontview"])
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

        local_points, template_geom_ids = build_robot_surface_template(
            env.sim.model,
            robot_geom_ids,
            args.points_per_geom,
            rng,
        )
        if len(local_points) == 0:
            raise RuntimeError("Generated an empty robot surface template.")

        step_points = [transform_template_points(env.sim, local_points, template_geom_ids)]
        for action in action_chunk:
            env.step(action)
            step_points.append(transform_template_points(env.sim, local_points, template_geom_ids))
    finally:
        env.close()

    points_by_step = np.stack(step_points, axis=0).astype(np.float32)
    all_points, all_colors = flatten_step_points(points_by_step)
    line_segments = temporal_line_segments(points_by_step)
    safe_name = (args.name or task_name).replace("/", "_")
    prefix = args.output_dir / f"{safe_name}_random10_robot_pointcloud_process"
    npz_path = prefix.with_suffix(".npz")
    ply_path = prefix.with_suffix(".ply")
    preview_path = prefix.with_name(f"{prefix.name}_preview.png")
    frontview_plot_path = prefix.with_name(f"{prefix.name}_frontview_yellow_lines.png")
    action_path = prefix.with_name(f"{prefix.name}_actions.npy")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(action_path, action_chunk.astype(np.float32))
    np.savez_compressed(
        npz_path,
        points_by_step=points_by_step,
        points=all_points.astype(np.float32),
        colors=all_colors.astype(np.uint8),
        temporal_line_segments=line_segments.astype(np.float32),
        temporal_line_colors=temporal_line_colors(len(line_segments)).astype(np.float32),
        action_chunk=action_chunk.astype(np.float32),
        local_template_points=local_points.astype(np.float32),
        template_geom_ids=template_geom_ids.astype(np.int32),
        robot_geom_ids=np.asarray(robot_geom_ids, dtype=np.int32),
    )
    write_ascii_ply(ply_path, all_points, all_colors)
    save_preview_png(preview_path, all_points, all_colors, args.preview_points)
    save_frontview_process_plot(
        frontview_plot_path,
        points_by_step,
        all_colors,
        line_segments,
        args.max_line_points,
        args.plot_point_size,
        args.plot_elev,
        args.plot_azim,
    )

    print(f"[info] task: {task_name}")
    print(f"[info] action_chunk shape: {action_chunk.shape}")
    print(f"[info] robot geoms: {len(robot_geom_ids)}")
    print(f"[info] template points per step: {points_by_step.shape[1]}")
    print(f"[info] point-cloud states saved: {points_by_step.shape[0]} including initial pose")
    print(f"[info] temporal yellow line segments saved: {line_segments.shape[0]}")
    print(f"[done] saved process npz: {npz_path}")
    print(f"[done] saved process ply: {ply_path}")
    print(f"[done] saved generic preview: {preview_path}")
    print(f"[done] saved frontview yellow-line plot: {frontview_plot_path}")
    print(f"[done] saved actions: {action_path}")
    return npz_path, ply_path, preview_path, frontview_plot_path, action_path


def main() -> None:
    args = parse_args()
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    rollout_pointcloud_process(args)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
