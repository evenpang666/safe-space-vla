#!/usr/bin/env python3
"""Render front-view LIBERO point-cloud figures from generated project outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render front-view swept-pointcloud and obstacle-OBB figures for LIBERO outputs."
    )
    parser.add_argument(
        "--swept-pointcloud",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "libero_robot_swept_pointcloud"
        / f"{DEFAULT_TASK}_robot_swept.npz",
        help="Input robot swept pointcloud .npz from scripts/libero_robot_swept_pointcloud.py.",
    )
    parser.add_argument(
        "--scene-pointcloud",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_pointcloud" / f"{DEFAULT_TASK}_pointcloud.npz",
        help="Input scene pointcloud .npz from scripts/libero_reconstruct_pointcloud.py.",
    )
    parser.add_argument(
        "--obb-safe-space",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "safe_space"
        / f"{DEFAULT_TASK}_tabletop_xy_oriented_obstacle_obb_safe_space.npz",
        help="Input OBB safe-space .npz from scripts/build_safe_space_from_pointcloud.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_frontview_figures",
        help="Directory for front-view PNG outputs.",
    )
    parser.add_argument("--name", default=DEFAULT_TASK, help="Output basename.")
    parser.add_argument("--max-points", type=int, default=80000, help="Maximum points drawn per figure.")
    parser.add_argument("--elev", type=float, default=18.0, help="Matplotlib elevation for front view.")
    parser.add_argument("--azim", type=float, default=0.0, help="Matplotlib azimuth for front view.")
    return parser.parse_args()


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    if "points" not in data:
        raise ValueError(f"{path} does not contain a 'points' array")
    points = np.asarray(data["points"], dtype=np.float32)
    colors = np.asarray(data["colors"], dtype=np.uint8) if "colors" in data else None
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is not None:
        colors = colors[valid]
    if len(points) == 0:
        raise ValueError(f"{path} contains no valid points")
    return points, colors


def sample_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], colors[idx] if colors is not None else None


def set_equal_axes_from_bounds(ax, bounds: np.ndarray) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
    center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
    radius = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
    radius = max(float(radius), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def points_bounds(points: np.ndarray, margin: float = 0.03) -> np.ndarray:
    mins = points.min(axis=0) - margin
    maxs = points.max(axis=0) + margin
    return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=np.float32)


def select_display_workspace_bounds(
    workspace_bounds: np.ndarray,
    display_workspace_bounds: np.ndarray | None = None,
    table_z: np.ndarray | float | None = None,
) -> np.ndarray:
    if display_workspace_bounds is not None:
        display = np.asarray(display_workspace_bounds, dtype=np.float32)
        if display.shape == (6,):
            return display

    bounds = np.asarray(workspace_bounds, dtype=np.float32).copy()
    if table_z is not None:
        bounds[4] = np.float32(np.asarray(table_z).item())
        if bounds[5] <= bounds[4]:
            bounds[5] = bounds[4] + np.float32(1e-3)
    return bounds


def box_faces_from_corners(corners: np.ndarray) -> list[np.ndarray]:
    face_indices = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    return [corners[np.asarray(face, dtype=np.int64)] for face in face_indices]


def draw_frontview_axes(ax, bounds: np.ndarray, elev: float, azim: float) -> None:
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    ax.view_init(elev=elev, azim=azim)
    set_equal_axes_from_bounds(ax, bounds)


def save_swept_frontview(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None,
    max_points: int,
    elev: float,
    azim: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    draw_points, draw_colors = sample_points(points, colors, max_points, seed=0)
    if draw_colors is None:
        color_values = draw_points[:, 2]
        cmap = "turbo"
    else:
        color_values = draw_colors.astype(np.float32) / 255.0
        cmap = None

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        draw_points[:, 0],
        draw_points[:, 1],
        draw_points[:, 2],
        c=color_values,
        cmap=cmap,
        s=0.35,
        linewidths=0,
        alpha=0.82,
    )
    draw_frontview_axes(ax, points_bounds(draw_points), elev=elev, azim=azim)
    ax.set_title("LIBERO front-view robot swept point cloud")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_obstacle_obb_frontview(
    path: Path,
    scene_points: np.ndarray,
    scene_colors: np.ndarray | None,
    box_corners: np.ndarray,
    workspace_bounds: np.ndarray,
    point_counts: np.ndarray,
    max_points: int,
    elev: float,
    azim: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    draw_points, draw_colors = sample_points(scene_points, scene_colors, max_points, seed=1)
    if draw_colors is None:
        point_colors = "0.68"
    else:
        point_colors = draw_colors.astype(np.float32) / 255.0

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        draw_points[:, 0],
        draw_points[:, 1],
        draw_points[:, 2],
        c=point_colors,
        s=0.45,
        linewidths=0,
        alpha=0.32,
    )

    palette = (
        (0.9, 0.05, 0.05, 0.20),
        (0.05, 0.45, 0.95, 0.20),
        (0.05, 0.70, 0.25, 0.20),
        (0.9, 0.55, 0.05, 0.20),
        (0.55, 0.15, 0.85, 0.20),
        (0.0, 0.65, 0.75, 0.20),
    )
    for i, corners in enumerate(box_corners):
        color = palette[i % len(palette)]
        collection = Poly3DCollection(
            box_faces_from_corners(corners),
            facecolors=color,
            edgecolors=color[:3] + (0.9,),
            linewidths=1.5,
        )
        ax.add_collection3d(collection)
        center = corners.mean(axis=0)
        label = str(i + 1)
        if len(point_counts) == len(box_corners):
            label = f"{i + 1}: {int(point_counts[i])}"
        ax.text(center[0], center[1], center[2], label, color=color[:3], fontsize=8)

    draw_frontview_axes(ax, workspace_bounds, elev=elev, azim=azim)
    ax.set_title(f"LIBERO front-view obstacle point cloud + OBBs ({len(box_corners)} boxes)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def render_frontview_figures(
    swept_pointcloud: Path,
    scene_pointcloud: Path,
    obb_safe_space: Path,
    output_dir: Path,
    name: str,
    max_points: int = 80000,
    elev: float = 18.0,
    azim: float = 0.0,
) -> tuple[Path, Path]:
    swept_points, swept_colors = load_points(swept_pointcloud)
    scene_points, scene_colors = load_points(scene_pointcloud)
    obb_data = np.load(obb_safe_space)
    box_corners = np.asarray(obb_data["obstacle_box_corners"], dtype=np.float32)
    workspace_bounds = select_display_workspace_bounds(
        workspace_bounds=np.asarray(obb_data["workspace_bounds"], dtype=np.float32),
        display_workspace_bounds=(
            np.asarray(obb_data["display_workspace_bounds"], dtype=np.float32)
            if "display_workspace_bounds" in obb_data
            else None
        ),
        table_z=obb_data["table_z"] if "table_z" in obb_data else None,
    )
    point_counts = (
        np.asarray(obb_data["obstacle_box_point_counts"], dtype=np.int64)
        if "obstacle_box_point_counts" in obb_data
        else np.zeros((0,), dtype=np.int64)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    swept_png = output_dir / f"{name}_frontview_swept_pointcloud.png"
    obb_png = output_dir / f"{name}_frontview_obstacle_pointcloud_obb.png"
    save_swept_frontview(
        swept_png,
        swept_points,
        swept_colors,
        max_points=max_points,
        elev=elev,
        azim=azim,
    )
    save_obstacle_obb_frontview(
        obb_png,
        scene_points,
        scene_colors,
        box_corners,
        workspace_bounds,
        point_counts,
        max_points=max_points,
        elev=elev,
        azim=azim,
    )
    return swept_png, obb_png


def main() -> None:
    args = parse_args()
    swept_png, obb_png = render_frontview_figures(
        swept_pointcloud=args.swept_pointcloud,
        scene_pointcloud=args.scene_pointcloud,
        obb_safe_space=args.obb_safe_space,
        output_dir=args.output_dir,
        name=args.name,
        max_points=args.max_points,
        elev=args.elev,
        azim=args.azim,
    )
    print(f"[done] saved front-view swept point cloud: {swept_png}")
    print(f"[done] saved front-view obstacle point cloud OBB: {obb_png}")


if __name__ == "__main__":
    main()
