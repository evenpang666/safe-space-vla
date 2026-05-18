#!/usr/bin/env python3
"""Build workspace, obstacle cubes, and safe-space voxels from a point cloud."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voxelize a reconstructed scene point cloud into obstacle cubes and safe-space cells."
    )
    parser.add_argument(
        "--pointcloud",
        type=Path,
        required=True,
        help="Input .npz file containing arrays named points and optionally colors.",
    )
    parser.add_argument(
        "--workspace-bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="World-coordinate manipulation-space bounds. If omitted, they are estimated from the cloud.",
    )
    parser.add_argument(
        "--workspace-margin",
        type=float,
        default=0.05,
        help="Margin added around the point cloud when --workspace-bounds is omitted.",
    )
    parser.add_argument(
        "--workspace-mode",
        choices=["table", "pointcloud"],
        default="table",
        help=(
            "Automatic workspace estimation mode. 'table' uses tabletop x/y bounds "
            "and full scene z bounds; 'pointcloud' uses the whole point cloud."
        ),
    )
    parser.add_argument(
        "--table-z",
        type=float,
        default=None,
        help="World z coordinate of the tabletop. If omitted, it is estimated from the point cloud.",
    )
    parser.add_argument(
        "--table-slab-height",
        type=float,
        default=0.08,
        help="Full height of the horizontal slab used to estimate tabletop x/y bounds.",
    )
    cube_group = parser.add_mutually_exclusive_group()
    cube_group.add_argument(
        "--make-cube",
        dest="make_cube",
        action="store_true",
        help="Expand auto-estimated workspace bounds to an axis-aligned cube.",
    )
    cube_group.add_argument(
        "--no-make-cube",
        dest="make_cube",
        action="store_false",
        help="Use the auto-estimated workspace bounds without forcing a cube.",
    )
    parser.set_defaults(make_cube=True)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.04,
        help="Obstacle cube edge length in meters.",
    )
    parser.add_argument(
        "--min-points-per-obstacle",
        type=int,
        default=1,
        help="Minimum number of cloud points required for a voxel to become an obstacle cube.",
    )
    parser.add_argument(
        "--max-obstacle-cubes",
        type=int,
        default=2500,
        help="Maximum number of obstacle cubes drawn in the preview. All cubes are saved to .npz.",
    )
    parser.add_argument(
        "--draw-pointcloud",
        action="store_true",
        help="Also draw a light gray sampled point cloud in the preview.",
    )
    parser.add_argument(
        "--preview-points",
        type=int,
        default=30000,
        help="Maximum point cloud samples drawn when --draw-pointcloud is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "safe_space",
        help="Directory for generated .npz and visualization output.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output basename. Defaults to the input point cloud stem.",
    )
    return parser.parse_args()


def load_pointcloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    if "points" not in data:
        raise ValueError(f"{path} does not contain a 'points' array")
    points = np.asarray(data["points"], dtype=np.float32)
    colors = np.asarray(data["colors"], dtype=np.uint8) if "colors" in data else None
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is not None:
        colors = colors[valid]
    if len(points) == 0:
        raise ValueError("input point cloud is empty after removing invalid points")
    return points, colors


def estimate_table_z(points: np.ndarray, voxel_size: float) -> float:
    """Estimate tabletop height as the densest horizontal layer above low outliers."""

    z = points[:, 2]
    low = float(np.percentile(z, 20.0))
    high = float(np.percentile(z, 80.0))
    if high <= low:
        return float(np.median(z))

    candidates = z[(z >= low) & (z <= high)]
    if len(candidates) == 0:
        return float(np.median(z))

    bin_width = max(voxel_size / 2.0, 0.01)
    bins = max(int(np.ceil((high - low) / bin_width)), 10)
    hist, edges = np.histogram(candidates, bins=bins, range=(low, high))
    peak = int(np.argmax(hist))
    return float((edges[peak] + edges[peak + 1]) / 2.0)


def largest_connected_component_2d(occupied: np.ndarray) -> np.ndarray:
    visited = np.zeros_like(occupied, dtype=bool)
    best = []
    height, width = occupied.shape
    for start_y, start_x in np.argwhere(occupied):
        start_y = int(start_y)
        start_x = int(start_x)
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and occupied[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    return np.asarray(best, dtype=np.int64)


def estimate_table_xy_bounds(
    points: np.ndarray,
    table_z: float,
    slab_height: float,
    voxel_size: float,
    margin: float,
) -> tuple[np.ndarray, int]:
    half_height = max(float(slab_height) / 2.0, voxel_size / 2.0, 0.01)
    slab_points = points[np.abs(points[:, 2] - table_z) <= half_height]
    if len(slab_points) == 0:
        raise ValueError(
            "could not estimate tabletop x/y bounds: no points near table_z. "
            "Try increasing --table-slab-height or passing --workspace-bounds."
        )

    xy = slab_points[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    resolution = max(float(voxel_size), 0.01)
    dims = np.ceil((xy_max - xy_min) / resolution).astype(np.int64) + 1
    dims = np.maximum(dims, 1)
    xy_indices = np.floor((xy - xy_min) / resolution).astype(np.int64)
    xy_indices = np.clip(xy_indices, 0, dims - 1)

    occupied = np.zeros((int(dims[1]), int(dims[0])), dtype=bool)
    occupied[xy_indices[:, 1], xy_indices[:, 0]] = True
    component = largest_connected_component_2d(occupied)
    if len(component) == 0:
        raise ValueError("could not estimate tabletop x/y bounds from the point cloud")

    x_cells = component[:, 1]
    y_cells = component[:, 0]
    xmin = xy_min[0] + x_cells.min() * resolution - margin
    xmax = xy_min[0] + (x_cells.max() + 1) * resolution + margin
    ymin = xy_min[1] + y_cells.min() * resolution - margin
    ymax = xy_min[1] + (y_cells.max() + 1) * resolution + margin
    return np.array([xmin, xmax, ymin, ymax], dtype=np.float32), len(slab_points)


def estimate_pointcloud_workspace_bounds(
    points: np.ndarray,
    margin: float,
    make_cube: bool,
) -> np.ndarray:
    mins = points.min(axis=0) - margin
    maxs = points.max(axis=0) + margin
    if make_cube:
        center = (mins + maxs) / 2.0
        side = float(np.max(maxs - mins))
        mins = center - side / 2.0
        maxs = center + side / 2.0
    return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=np.float32)


def estimate_table_workspace_bounds(
    points: np.ndarray,
    margin: float,
    table_z: float,
    slab_height: float,
    voxel_size: float,
) -> tuple[np.ndarray, int]:
    xy_bounds, slab_count = estimate_table_xy_bounds(
        points=points,
        table_z=table_z,
        slab_height=slab_height,
        voxel_size=voxel_size,
        margin=margin,
    )
    zmin = float(points[:, 2].min() - margin)
    zmax = float(points[:, 2].max() + margin)
    bounds = np.array(
        [xy_bounds[0], xy_bounds[1], xy_bounds[2], xy_bounds[3], zmin, zmax],
        dtype=np.float32,
    )
    return bounds, slab_count


def bounds_to_origin_shape(bounds: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    maxs = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
    dims = np.ceil((maxs - mins) / voxel_size).astype(np.int64)
    dims = np.maximum(dims, 1)
    snapped_maxs = mins + dims.astype(np.float32) * voxel_size
    snapped_bounds = np.array(
        [mins[0], snapped_maxs[0], mins[1], snapped_maxs[1], mins[2], snapped_maxs[2]],
        dtype=np.float32,
    )
    return mins, dims, snapped_bounds


def voxelize_obstacles(
    points: np.ndarray,
    bounds: np.ndarray,
    voxel_size: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origin, dims, snapped_bounds = bounds_to_origin_shape(bounds, voxel_size)
    maxs = origin + dims.astype(np.float32) * voxel_size
    keep = np.all((points >= origin) & (points < maxs), axis=1)
    points = points[keep]
    if len(points) == 0:
        raise ValueError("no point cloud points are inside the workspace bounds")

    voxel_indices = np.floor((points - origin) / voxel_size).astype(np.int64)
    flat_indices = np.ravel_multi_index(voxel_indices.T, dims)
    unique_flat, counts = np.unique(flat_indices, return_counts=True)
    obstacle_flat = unique_flat[counts >= min_points]
    obstacle_indices = np.column_stack(np.unravel_index(obstacle_flat, dims)).astype(np.int64)

    occupied_grid = np.zeros(tuple(dims.tolist()), dtype=bool)
    if len(obstacle_indices) > 0:
        occupied_grid[tuple(obstacle_indices.T)] = True
    safe_grid = ~occupied_grid
    return obstacle_indices, occupied_grid, safe_grid, snapped_bounds


def voxel_centers(indices: np.ndarray, bounds: np.ndarray, voxel_size: float) -> np.ndarray:
    origin = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    return origin + (indices.astype(np.float32) + 0.5) * voxel_size


def draw_workspace_edges(ax, bounds: np.ndarray) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float32,
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color="black",
            linestyle="--",
            linewidth=1.2,
        )


def cube_faces(center: np.ndarray, half: float) -> list[np.ndarray]:
    x, y, z = center
    corners = np.array(
        [
            [x - half, y - half, z - half],
            [x + half, y - half, z - half],
            [x + half, y + half, z - half],
            [x - half, y + half, z - half],
            [x - half, y - half, z + half],
            [x + half, y - half, z + half],
            [x + half, y + half, z + half],
            [x - half, y + half, z + half],
        ],
        dtype=np.float32,
    )
    return [
        corners[[0, 1, 2, 3]],
        corners[[4, 5, 6, 7]],
        corners[[0, 1, 5, 4]],
        corners[[2, 3, 7, 6]],
        corners[[1, 2, 6, 5]],
        corners[[3, 0, 4, 7]],
    ]


def set_equal_axes(ax, bounds: np.ndarray) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
    radius = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def save_visualization(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None,
    bounds: np.ndarray,
    obstacle_centers: np.ndarray,
    voxel_size: float,
    max_obstacle_cubes: int,
    draw_pointcloud: bool,
    preview_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if draw_pointcloud:
        draw_points = points
        draw_colors = colors
        if len(draw_points) > preview_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(draw_points), size=preview_points, replace=False)
            draw_points = draw_points[idx]
            draw_colors = draw_colors[idx] if draw_colors is not None else None
        if draw_colors is None:
            c = "0.72"
        else:
            c = draw_colors.astype(np.float32) / 255.0
        ax.scatter(
            draw_points[:, 0],
            draw_points[:, 1],
            draw_points[:, 2],
            c=c,
            s=0.45,
            linewidths=0,
            alpha=0.35,
        )

    draw_centers = obstacle_centers
    if len(draw_centers) > max_obstacle_cubes:
        rng = np.random.default_rng(1)
        idx = rng.choice(len(draw_centers), size=max_obstacle_cubes, replace=False)
        draw_centers = draw_centers[idx]

    faces = []
    half = voxel_size / 2.0
    for center in draw_centers:
        faces.extend(cube_faces(center, half))
    if faces:
        cubes = Poly3DCollection(
            faces,
            facecolors=(1.0, 0.0, 0.0, 0.23),
            edgecolors=(0.72, 0.0, 0.0, 0.45),
            linewidths=0.18,
        )
        ax.add_collection3d(cubes)

    draw_workspace_edges(ax, bounds)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    ax.view_init(elev=25, azim=-55)
    set_equal_axes(ax, bounds)
    title = f"workspace + obstacle cubes ({len(obstacle_centers)} total"
    if len(draw_centers) != len(obstacle_centers):
        title += f", {len(draw_centers)} drawn"
    title += ")"
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.voxel_size <= 0.0:
        raise ValueError("--voxel-size must be positive")

    points, colors = load_pointcloud(args.pointcloud)
    table_z = float(args.table_z) if args.table_z is not None else estimate_table_z(points, args.voxel_size)
    table_slab_points = 0
    if args.workspace_bounds is not None:
        bounds = np.asarray(args.workspace_bounds, dtype=np.float32)
    elif args.workspace_mode == "table":
        bounds, table_slab_points = estimate_table_workspace_bounds(
            points=points,
            margin=args.workspace_margin,
            table_z=table_z,
            slab_height=args.table_slab_height,
            voxel_size=args.voxel_size,
        )
    else:
        bounds = estimate_pointcloud_workspace_bounds(
            points=points,
            margin=args.workspace_margin,
            make_cube=args.make_cube,
        )

    obstacle_indices, occupied_grid, safe_grid, bounds = voxelize_obstacles(
        points=points,
        bounds=bounds,
        voxel_size=args.voxel_size,
        min_points=max(args.min_points_per_obstacle, 1),
    )
    obstacle_centers = voxel_centers(obstacle_indices, bounds, args.voxel_size)
    safe_indices = np.column_stack(np.nonzero(safe_grid)).astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.pointcloud.stem
    npz_path = args.output_dir / f"{name}_safe_space.npz"
    preview_path = args.output_dir / f"{name}_safe_space_preview.png"

    np.savez_compressed(
        npz_path,
        workspace_bounds=bounds.astype(np.float32),
        workspace_mode=np.array(args.workspace_mode),
        table_z=np.array(table_z, dtype=np.float32),
        table_slab_height=np.array(args.table_slab_height, dtype=np.float32),
        table_slab_points=np.array(table_slab_points, dtype=np.int64),
        voxel_size=np.array(args.voxel_size, dtype=np.float32),
        grid_shape=np.array(occupied_grid.shape, dtype=np.int64),
        obstacle_indices=obstacle_indices,
        obstacle_centers=obstacle_centers.astype(np.float32),
        safe_indices=safe_indices,
        occupied_grid=occupied_grid,
        safe_grid=safe_grid,
    )
    save_visualization(
        path=preview_path,
        points=points,
        colors=colors,
        bounds=bounds,
        obstacle_centers=obstacle_centers,
        voxel_size=args.voxel_size,
        max_obstacle_cubes=args.max_obstacle_cubes,
        draw_pointcloud=args.draw_pointcloud,
        preview_points=args.preview_points,
    )

    print(f"[done] workspace bounds xyz: {bounds}")
    print(f"[done] workspace mode: {args.workspace_mode}")
    print(f"[done] table z: {table_z}")
    if args.workspace_mode == "table" and args.workspace_bounds is None:
        print(f"[done] tabletop slab points: {table_slab_points}")
    print(f"[done] voxel size: {args.voxel_size}")
    print(f"[done] grid shape: {occupied_grid.shape}")
    print(f"[done] obstacle cubes: {len(obstacle_indices)}")
    print(f"[done] safe cells: {len(safe_indices)}")
    print(f"[done] saved safe-space npz: {npz_path}")
    print(f"[done] saved preview: {preview_path}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)
