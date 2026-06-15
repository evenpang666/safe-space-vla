#!/usr/bin/env python3
"""Build workspace, obstacle boxes, and safe-space voxels from a point cloud."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

COORDINATE_FRAME = "mujoco_world"


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build obstacle boxes or voxel obstacles from a reconstructed scene point cloud."
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
        "--obstacle-mode",
        choices=["voxel", "tabletop_boxes"],
        default="voxel",
        help=(
            "Obstacle construction mode. 'voxel' keeps the legacy per-voxel cubes. "
            "'tabletop_boxes' segments points above the table and wraps each connected obstacle "
            "component in a bounding box."
        ),
    )
    parser.add_argument(
        "--table-obstacle-min-height",
        type=float,
        default=0.015,
        help="Minimum height above table_z for points to be considered tabletop obstacle points.",
    )
    parser.add_argument(
        "--table-obstacle-max-height",
        type=float,
        default=0.45,
        help="Maximum height above table_z for tabletop obstacle points; filters upper robot/background remnants.",
    )
    parser.add_argument(
        "--component-voxel-size",
        type=float,
        default=0.02,
        help="Voxel size used only for connected-component grouping in tabletop_boxes mode.",
    )
    parser.add_argument(
        "--component-connectivity",
        type=int,
        choices=[6, 18, 26],
        default=6,
        help=(
            "3D voxel connectivity used for tabletop component grouping. "
            "Use 6 for tighter obstacle separation; 26 preserves legacy corner-connected grouping."
        ),
    )
    parser.add_argument(
        "--min-component-points",
        type=int,
        default=40,
        help="Minimum number of obstacle points required to keep a tabletop component box.",
    )
    parser.add_argument(
        "--box-margin",
        type=float,
        default=0.01,
        help="Margin added to each tabletop obstacle box in meters.",
    )
    parser.add_argument(
        "--box-shape",
        choices=["cuboid", "cube"],
        default="cuboid",
        help=(
            "Shape used for tabletop component boxes. 'cuboid' keeps independent x/y/z extents; "
            "'cube' expands each box to equal side lengths."
        ),
    )
    parser.add_argument(
        "--box-orientation",
        choices=["axis_aligned", "xy_oriented", "pca_3d"],
        default="axis_aligned",
        help=(
            "Orientation used for tabletop component boxes. 'xy_oriented' fits a rotated "
            "minimum-area tabletop footprint with vertical z; 'pca_3d' allows a fully "
            "tilted PCA-oriented 3D box."
        ),
    )
    parser.add_argument(
        "--min-points-per-obstacle",
        type=int,
        default=1,
        help="Minimum number of cloud points required for a voxel to become occupied in voxel mode.",
    )
    parser.add_argument(
        "--max-obstacle-cubes",
        type=int,
        default=2500,
        help="Maximum number of voxel obstacle cubes drawn in voxel-mode previews. All cells are saved to .npz.",
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


def table_aligned_display_bounds(
    workspace_bounds: np.ndarray,
    table_z: float,
    table_xy_bounds: np.ndarray | None = None,
) -> np.ndarray:
    """Bounds used only for dashed visualization, with its bottom on the table."""

    bounds = np.asarray(workspace_bounds, dtype=np.float32).copy()
    if table_xy_bounds is not None:
        xy = np.asarray(table_xy_bounds, dtype=np.float32)
        if xy.shape != (4,):
            raise ValueError(f"table_xy_bounds must have shape (4,), got {xy.shape}")
        bounds[:4] = xy
    bounds[4] = np.float32(table_z)
    if bounds[5] <= bounds[4]:
        bounds[5] = bounds[4] + np.float32(1e-3)
    return bounds


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


def neighbor_offsets_3d(connectivity: int = 26) -> list[tuple[int, int, int]]:
    if int(connectivity) not in (6, 18, 26):
        raise ValueError("connectivity must be one of 6, 18, or 26")
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan != 1:
                    continue
                if connectivity == 18 and manhattan > 2:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def connected_components_from_indices(indices: np.ndarray, *, connectivity: int = 26) -> list[np.ndarray]:
    if len(indices) == 0:
        return []

    active = {tuple(int(v) for v in index) for index in indices}
    visited: set[tuple[int, int, int]] = set()
    offsets = neighbor_offsets_3d(connectivity)
    components = []

    for start in active:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            cx, cy, cz = current
            for dx, dy, dz in offsets:
                neighbor = (cx + dx, cy + dy, cz + dz)
                if neighbor in active and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))

    return components


def tabletop_obstacle_points(
    points: np.ndarray,
    bounds: np.ndarray,
    table_z: float,
    min_height: float,
    max_height: float,
) -> np.ndarray:
    xmin, xmax, ymin, ymax, _, _ = bounds
    zmin = table_z + min_height
    zmax = table_z + max_height
    keep = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )
    return points[keep]


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()

    sorted_points = sorted((float(x), float(y)) for x, y in np.unique(points, axis=0))
    if len(sorted_points) <= 1:
        return np.asarray(sorted_points, dtype=np.float32)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def minimum_area_rect_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise ValueError("cannot fit an oriented rectangle to an empty component")
    if len(points) == 1:
        center = points[0].astype(np.float32)
        axes = np.eye(2, dtype=np.float32)
        half_sizes = np.zeros(2, dtype=np.float32)
        return center, axes, half_sizes

    hull = convex_hull_2d(points)
    if len(hull) <= 1:
        center = hull[0].astype(np.float32)
        axes = np.eye(2, dtype=np.float32)
        half_sizes = np.zeros(2, dtype=np.float32)
        return center, axes, half_sizes

    best_area = np.inf
    best_center = None
    best_axes = None
    best_half_sizes = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        length = float(np.linalg.norm(edge))
        if length <= 1e-8:
            continue
        axis_u = edge / length
        axis_v = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)
        axes = np.column_stack((axis_u, axis_v)).astype(np.float32)
        local = points @ axes
        local_min = local.min(axis=0)
        local_max = local.max(axis=0)
        sizes = local_max - local_min
        area = float(sizes[0] * sizes[1])
        if area < best_area:
            best_area = area
            local_center = (local_min + local_max) / 2.0
            best_center = axes @ local_center
            best_axes = axes
            best_half_sizes = sizes / 2.0

    if best_center is None or best_axes is None or best_half_sizes is None:
        center = points.mean(axis=0).astype(np.float32)
        axes = np.eye(2, dtype=np.float32)
        half_sizes = (points.max(axis=0) - points.min(axis=0)) / 2.0
        return center, axes, half_sizes.astype(np.float32)

    return (
        np.asarray(best_center, dtype=np.float32),
        np.asarray(best_axes, dtype=np.float32),
        np.asarray(best_half_sizes, dtype=np.float32),
    )


def pca_axes_3d(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    if len(points) < 3 or np.allclose(centered, 0.0):
        return np.eye(3, dtype=np.float32)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].astype(np.float32)
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes


def box_corners_from_center_axes(center: np.ndarray, axes: np.ndarray, half_sizes: np.ndarray) -> np.ndarray:
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    return center + (signs * half_sizes) @ axes.T


def oriented_box_from_component(
    component_points: np.ndarray,
    table_z: float,
    box_margin: float,
    box_shape: str,
    box_orientation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if box_orientation == "xy_oriented":
        center_xy, axes_xy, half_xy = minimum_area_rect_2d(component_points[:, :2])
        z_min = min(float(component_points[:, 2].min() - box_margin), table_z)
        z_max = float(component_points[:, 2].max() + box_margin)
        center = np.array([center_xy[0], center_xy[1], (z_min + z_max) / 2.0], dtype=np.float32)
        axes = np.eye(3, dtype=np.float32)
        axes[:2, :2] = axes_xy
        half_sizes = np.array(
            [half_xy[0] + box_margin, half_xy[1] + box_margin, (z_max - z_min) / 2.0],
            dtype=np.float32,
        )
    elif box_orientation == "pca_3d":
        axes = pca_axes_3d(component_points)
        local = component_points @ axes
        local_min = local.min(axis=0) - box_margin
        local_max = local.max(axis=0) + box_margin
        center = axes @ ((local_min + local_max) / 2.0)
        half_sizes = ((local_max - local_min) / 2.0).astype(np.float32)
    else:
        box_min = component_points.min(axis=0) - box_margin
        box_max = component_points.max(axis=0) + box_margin
        box_min[2] = min(box_min[2], table_z)
        center = ((box_min + box_max) / 2.0).astype(np.float32)
        axes = np.eye(3, dtype=np.float32)
        half_sizes = ((box_max - box_min) / 2.0).astype(np.float32)

    if box_shape == "cube":
        half_sizes[:] = float(np.max(half_sizes))

    corners = box_corners_from_center_axes(center, axes, half_sizes)
    return center.astype(np.float32), axes.astype(np.float32), half_sizes.astype(np.float32), corners.astype(np.float32)


def component_boxes_from_tabletop_points(
    points: np.ndarray,
    bounds: np.ndarray,
    table_z: float,
    component_voxel_size: float,
    min_component_points: int,
    box_margin: float,
    box_shape: str,
    box_orientation: str,
    component_connectivity: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if component_voxel_size <= 0.0:
        raise ValueError("--component-voxel-size must be positive")
    if int(component_connectivity) not in (6, 18, 26):
        raise ValueError("--component-connectivity must be one of 6, 18, or 26")

    mins = np.array([bounds[0], bounds[2], table_z], dtype=np.float32)
    maxs = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
    dims = np.ceil((maxs - mins) / component_voxel_size).astype(np.int64)
    dims = np.maximum(dims, 1)

    keep = np.all((points >= mins) & (points < maxs), axis=1)
    points = points[keep]
    if len(points) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 8, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    voxel_indices = np.floor((points - mins) / component_voxel_size).astype(np.int64)
    voxel_indices = np.clip(voxel_indices, 0, dims - 1)
    unique_indices = np.unique(voxel_indices, axis=0)
    components = connected_components_from_indices(unique_indices, connectivity=int(component_connectivity))

    voxel_to_component = {}
    for component_id, component in enumerate(components):
        for index in component:
            voxel_to_component[tuple(int(v) for v in index)] = component_id

    point_component_ids = np.array(
        [voxel_to_component[tuple(int(v) for v in index)] for index in voxel_indices],
        dtype=np.int64,
    )

    box_mins = []
    box_maxs = []
    box_centers = []
    box_axes = []
    box_half_sizes = []
    box_corners = []
    point_counts = []
    workspace_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    workspace_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)

    for component_id in range(len(components)):
        component_points = points[point_component_ids == component_id]
        if len(component_points) < min_component_points:
            continue
        center, axes, half_sizes, corners = oriented_box_from_component(
            component_points=component_points,
            table_z=table_z,
            box_margin=box_margin,
            box_shape=box_shape,
            box_orientation=box_orientation,
        )
        box_min = corners.min(axis=0)
        box_max = corners.max(axis=0)
        box_min = np.maximum(box_min, workspace_min)
        box_max = np.minimum(box_max, workspace_max)
        if np.all(box_max > box_min):
            box_mins.append(box_min.astype(np.float32))
            box_maxs.append(box_max.astype(np.float32))
            box_centers.append(center.astype(np.float32))
            box_axes.append(axes.astype(np.float32))
            box_half_sizes.append(half_sizes.astype(np.float32))
            box_corners.append(corners.astype(np.float32))
            point_counts.append(len(component_points))

    if not box_mins:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 8, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    order = np.argsort(np.asarray(point_counts))[::-1]
    return (
        np.asarray(box_mins, dtype=np.float32)[order],
        np.asarray(box_maxs, dtype=np.float32)[order],
        np.asarray(box_centers, dtype=np.float32)[order],
        np.asarray(box_axes, dtype=np.float32)[order],
        np.asarray(box_half_sizes, dtype=np.float32)[order],
        np.asarray(box_corners, dtype=np.float32)[order],
        np.asarray(point_counts, dtype=np.int64)[order],
    )


def boxes_to_occupied_grid(
    box_centers: np.ndarray,
    box_axes: np.ndarray,
    box_half_sizes: np.ndarray,
    box_corners: np.ndarray,
    bounds: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origin, dims, snapped_bounds = bounds_to_origin_shape(bounds, voxel_size)
    occupied_grid = np.zeros(tuple(dims.tolist()), dtype=bool)

    for center, axes, half_sizes, corners in zip(box_centers, box_axes, box_half_sizes, box_corners):
        box_min = corners.min(axis=0)
        box_max = corners.max(axis=0)
        start = np.floor((box_min - origin) / voxel_size).astype(np.int64)
        stop = np.ceil((box_max - origin) / voxel_size).astype(np.int64)
        start = np.clip(start, 0, dims)
        stop = np.clip(stop, 0, dims)
        if np.any(stop <= start):
            continue
        xs = origin[0] + (np.arange(start[0], stop[0], dtype=np.float32) + 0.5) * voxel_size
        ys = origin[1] + (np.arange(start[1], stop[1], dtype=np.float32) + 0.5) * voxel_size
        zs = origin[2] + (np.arange(start[2], stop[2], dtype=np.float32) + 0.5) * voxel_size
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
        local = (grid - center) @ axes
        inside = np.all(np.abs(local) <= (half_sizes + 1e-6), axis=-1)
        occupied_grid[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] |= inside

    obstacle_indices = np.column_stack(np.nonzero(occupied_grid)).astype(np.int64)
    safe_grid = ~occupied_grid
    return obstacle_indices, occupied_grid, safe_grid, snapped_bounds


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


def box_faces(box_min: np.ndarray, box_max: np.ndarray) -> list[np.ndarray]:
    xmin, ymin, zmin = box_min
    xmax, ymax, zmax = box_max
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
    return [
        corners[[0, 1, 2, 3]],
        corners[[4, 5, 6, 7]],
        corners[[0, 1, 5, 4]],
        corners[[2, 3, 7, 6]],
        corners[[1, 2, 6, 5]],
        corners[[3, 0, 4, 7]],
    ]


def box_faces_from_corners(corners: np.ndarray) -> list[np.ndarray]:
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
    display_bounds: np.ndarray,
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
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        keep = (
            (draw_points[:, 0] >= xmin)
            & (draw_points[:, 0] <= xmax)
            & (draw_points[:, 1] >= ymin)
            & (draw_points[:, 1] <= ymax)
            & (draw_points[:, 2] >= zmin)
            & (draw_points[:, 2] <= zmax)
        )
        draw_points = draw_points[keep]
        draw_colors = draw_colors[keep] if draw_colors is not None else None
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

    draw_workspace_edges(ax, display_bounds)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    ax.view_init(elev=25, azim=-55)
    set_equal_axes(ax, display_bounds)
    title = f"workspace + obstacle cubes ({len(obstacle_centers)} total"
    if len(draw_centers) != len(obstacle_centers):
        title += f", {len(draw_centers)} drawn"
    title += ")"
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_box_visualization(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None,
    bounds: np.ndarray,
    display_bounds: np.ndarray,
    box_mins: np.ndarray,
    box_maxs: np.ndarray,
    box_corners: np.ndarray,
    component_point_counts: np.ndarray,
    box_orientation: str,
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
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        keep = (
            (draw_points[:, 0] >= xmin)
            & (draw_points[:, 0] <= xmax)
            & (draw_points[:, 1] >= ymin)
            & (draw_points[:, 1] <= ymax)
            & (draw_points[:, 2] >= zmin)
            & (draw_points[:, 2] <= zmax)
        )
        draw_points = draw_points[keep]
        draw_colors = draw_colors[keep] if draw_colors is not None else None
        if len(draw_points) > preview_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(draw_points), size=preview_points, replace=False)
            draw_points = draw_points[idx]
            draw_colors = draw_colors[idx] if draw_colors is not None else None
        c = "0.72" if draw_colors is None else draw_colors.astype(np.float32) / 255.0
        ax.scatter(
            draw_points[:, 0],
            draw_points[:, 1],
            draw_points[:, 2],
            c=c,
            s=0.55,
            linewidths=0,
            alpha=0.34,
        )

    palette = (
        (0.9, 0.05, 0.05, 0.20),
        (0.05, 0.45, 0.95, 0.20),
        (0.05, 0.70, 0.25, 0.20),
        (0.9, 0.55, 0.05, 0.20),
        (0.55, 0.15, 0.85, 0.20),
        (0.0, 0.65, 0.75, 0.20),
    )
    for i, (box_min, box_max, corners) in enumerate(zip(box_mins, box_maxs, box_corners)):
        color = palette[i % len(palette)]
        boxes = Poly3DCollection(
            box_faces_from_corners(corners),
            facecolors=color,
            edgecolors=color[:3] + (0.85,),
            linewidths=1.4,
        )
        ax.add_collection3d(boxes)
        center = corners.mean(axis=0)
        label = str(i + 1)
        if len(component_point_counts) == len(box_mins):
            label = f"{i + 1}: {int(component_point_counts[i])}"
        ax.text(center[0], center[1], box_max[2] + 0.015, label, color=color[:3], fontsize=8)

    draw_workspace_edges(ax, display_bounds)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    ax.view_init(elev=28, azim=-58)
    set_equal_axes(ax, display_bounds)
    ax.set_title(f"tabletop obstacle {box_orientation} bounding boxes ({len(box_mins)} boxes)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.voxel_size <= 0.0:
        raise ValueError("--voxel-size must be positive")
    if args.table_obstacle_min_height < 0.0:
        raise ValueError("--table-obstacle-min-height must be non-negative")
    if args.table_obstacle_max_height <= args.table_obstacle_min_height:
        raise ValueError("--table-obstacle-max-height must be greater than --table-obstacle-min-height")
    if args.min_component_points < 1:
        raise ValueError("--min-component-points must be positive")
    if args.box_margin < 0.0:
        raise ValueError("--box-margin must be non-negative")

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

    if args.obstacle_mode == "tabletop_boxes":
        bounds = bounds.copy()
        bounds[4] = max(float(bounds[4]), table_z)
        if bounds[5] <= bounds[4]:
            raise ValueError(
                "workspace z max must be above table_z for tabletop_boxes mode. "
                "Increase --workspace-bounds ZMAX or pass a lower --table-z."
            )

    display_bounds = table_aligned_display_bounds(bounds, table_z, bounds[:4])

    box_mins = np.zeros((0, 3), dtype=np.float32)
    box_maxs = np.zeros((0, 3), dtype=np.float32)
    box_centers = np.zeros((0, 3), dtype=np.float32)
    box_axes = np.zeros((0, 3, 3), dtype=np.float32)
    box_half_sizes = np.zeros((0, 3), dtype=np.float32)
    box_corners = np.zeros((0, 8, 3), dtype=np.float32)
    component_point_counts = np.zeros((0,), dtype=np.int64)
    tabletop_points = np.zeros((0, 3), dtype=np.float32)

    if args.obstacle_mode == "voxel":
        obstacle_indices, occupied_grid, safe_grid, bounds = voxelize_obstacles(
            points=points,
            bounds=bounds,
            voxel_size=args.voxel_size,
            min_points=max(args.min_points_per_obstacle, 1),
        )
    else:
        tabletop_points = tabletop_obstacle_points(
            points=points,
            bounds=bounds,
            table_z=table_z,
            min_height=args.table_obstacle_min_height,
            max_height=args.table_obstacle_max_height,
        )
        (
            box_mins,
            box_maxs,
            box_centers,
            box_axes,
            box_half_sizes,
            box_corners,
            component_point_counts,
        ) = component_boxes_from_tabletop_points(
            points=tabletop_points,
            bounds=bounds,
            table_z=table_z,
            component_voxel_size=args.component_voxel_size,
            component_connectivity=args.component_connectivity,
            min_component_points=args.min_component_points,
            box_margin=args.box_margin,
            box_shape=args.box_shape,
            box_orientation=args.box_orientation,
        )
        obstacle_indices, occupied_grid, safe_grid, bounds = boxes_to_occupied_grid(
            box_centers=box_centers,
            box_axes=box_axes,
            box_half_sizes=box_half_sizes,
            box_corners=box_corners,
            bounds=bounds,
            voxel_size=args.voxel_size,
        )
    obstacle_centers = voxel_centers(obstacle_indices, bounds, args.voxel_size)
    safe_indices = np.column_stack(np.nonzero(safe_grid)).astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.pointcloud.stem
    npz_path = args.output_dir / f"{name}_safe_space.npz"
    preview_path = args.output_dir / f"{name}_safe_space_preview.png"

    np.savez_compressed(
        npz_path,
        coordinate_frame=np.asarray(COORDINATE_FRAME),
        workspace_bounds=bounds.astype(np.float32),
        display_workspace_bounds=display_bounds.astype(np.float32),
        workspace_mode=np.array(args.workspace_mode),
        obstacle_mode=np.array(args.obstacle_mode),
        table_z=np.array(table_z, dtype=np.float32),
        table_slab_height=np.array(args.table_slab_height, dtype=np.float32),
        table_slab_points=np.array(table_slab_points, dtype=np.int64),
        table_obstacle_min_height=np.array(args.table_obstacle_min_height, dtype=np.float32),
        table_obstacle_max_height=np.array(args.table_obstacle_max_height, dtype=np.float32),
        component_voxel_size=np.array(args.component_voxel_size, dtype=np.float32),
        component_connectivity=np.array(args.component_connectivity, dtype=np.int64),
        min_component_points=np.array(args.min_component_points, dtype=np.int64),
        box_margin=np.array(args.box_margin, dtype=np.float32),
        box_shape=np.array(args.box_shape),
        box_orientation=np.array(args.box_orientation),
        voxel_size=np.array(args.voxel_size, dtype=np.float32),
        grid_shape=np.array(occupied_grid.shape, dtype=np.int64),
        obstacle_indices=obstacle_indices,
        obstacle_centers=obstacle_centers.astype(np.float32),
        obstacle_box_mins=box_mins.astype(np.float32),
        obstacle_box_maxs=box_maxs.astype(np.float32),
        obstacle_box_centers=box_centers.astype(np.float32),
        obstacle_box_axes=box_axes.astype(np.float32),
        obstacle_box_half_sizes=box_half_sizes.astype(np.float32),
        obstacle_box_corners=box_corners.astype(np.float32),
        obstacle_box_sizes=(box_half_sizes * 2.0).astype(np.float32),
        obstacle_box_point_counts=component_point_counts,
        safe_indices=safe_indices,
        occupied_grid=occupied_grid,
        safe_grid=safe_grid,
    )
    if args.obstacle_mode == "voxel":
        save_visualization(
            path=preview_path,
            points=points,
            colors=colors,
            bounds=bounds,
            display_bounds=display_bounds,
            obstacle_centers=obstacle_centers,
            voxel_size=args.voxel_size,
            max_obstacle_cubes=args.max_obstacle_cubes,
            draw_pointcloud=args.draw_pointcloud,
            preview_points=args.preview_points,
        )
    else:
        save_box_visualization(
            path=preview_path,
            points=points,
            colors=colors,
            bounds=bounds,
            display_bounds=display_bounds,
            box_mins=box_mins,
            box_maxs=box_maxs,
            box_corners=box_corners,
            component_point_counts=component_point_counts,
            box_orientation=args.box_orientation,
            draw_pointcloud=True if not args.draw_pointcloud else args.draw_pointcloud,
            preview_points=args.preview_points,
        )

    print(f"[done] workspace bounds xyz: {bounds}")
    print(f"[done] display workspace bounds xyz: {display_bounds}")
    print(f"[done] workspace mode: {args.workspace_mode}")
    print(f"[done] obstacle mode: {args.obstacle_mode}")
    print(f"[done] table z: {table_z}")
    if args.workspace_mode == "table" and args.workspace_bounds is None:
        print(f"[done] tabletop slab points: {table_slab_points}")
    print(f"[done] voxel size: {args.voxel_size}")
    print(f"[done] grid shape: {occupied_grid.shape}")
    print(f"[done] occupied grid cells: {len(obstacle_indices)}")
    if args.obstacle_mode == "tabletop_boxes":
        print(f"[done] tabletop obstacle points: {len(tabletop_points)}")
        print(f"[done] obstacle box shape: {args.box_shape}")
        print(f"[done] obstacle box orientation: {args.box_orientation}")
        print(f"[done] obstacle boxes: {len(box_mins)}")
        for i, (center, half_sizes, count) in enumerate(zip(box_centers, box_half_sizes, component_point_counts), start=1):
            size = half_sizes * 2.0
            print(f"[done] box {i}: center={center}, size={size}, points={int(count)}")
    print(f"[done] safe cells: {len(safe_indices)}")
    print(f"[done] saved safe-space npz: {npz_path}")
    print(f"[done] saved preview: {preview_path}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)
