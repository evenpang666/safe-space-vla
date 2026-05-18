#!/usr/bin/env python3
"""Convert a voxelized safe-space file into a signed distance field."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a safe-space SDF from scripts/build_safe_space_from_pointcloud.py output. "
            "SDF is positive in safe cells and negative in obstacles or outside the workspace."
        )
    )
    parser.add_argument(
        "--safe-space",
        type=Path,
        required=True,
        help="Input safe-space .npz containing occupied_grid, safe_grid, workspace_bounds, and voxel_size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <input stem>_sdf.npz in the same directory.",
    )
    parser.add_argument(
        "--obstacle-inflation",
        type=float,
        default=0.0,
        help="Meters by which obstacle cells are conservatively inflated before SDF construction.",
    )
    parser.add_argument(
        "--truncate-distance",
        type=float,
        default=0.30,
        help="Clip SDF values to +/- this distance in meters. Use <=0 to disable clipping.",
    )
    return parser.parse_args()


def load_safe_space(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = ("occupied_grid", "safe_grid", "workspace_bounds", "voxel_size")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")

    occupied = np.asarray(data["occupied_grid"], dtype=bool)
    safe = np.asarray(data["safe_grid"], dtype=bool)
    if occupied.shape != safe.shape:
        raise ValueError(f"occupied_grid and safe_grid shapes differ: {occupied.shape} vs {safe.shape}")
    if occupied.ndim != 3:
        raise ValueError(f"occupied_grid must be 3D, got {occupied.shape}")

    bounds = np.asarray(data["workspace_bounds"], dtype=np.float32)
    if bounds.shape != (6,):
        raise ValueError(f"workspace_bounds must have shape (6,), got {bounds.shape}")

    voxel_size = float(np.asarray(data["voxel_size"]).item())
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    return {
        "occupied_grid": occupied,
        "safe_grid": safe,
        "workspace_bounds": bounds,
        "voxel_size": np.array(voxel_size, dtype=np.float32),
    }


def inflate_obstacles(occupied: np.ndarray, voxel_size: float, inflation: float) -> np.ndarray:
    if inflation <= 0.0 or not occupied.any():
        return occupied.copy()
    distance_to_obstacle = ndimage.distance_transform_edt(~occupied, sampling=voxel_size)
    return occupied | (distance_to_obstacle <= inflation)


def workspace_boundary_distance(bounds: np.ndarray, shape: tuple[int, int, int], voxel_size: float) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
    xs = xmin + (np.arange(shape[0], dtype=np.float32) + 0.5) * voxel_size
    ys = ymin + (np.arange(shape[1], dtype=np.float32) + 0.5) * voxel_size
    zs = zmin + (np.arange(shape[2], dtype=np.float32) + 0.5) * voxel_size
    dx = np.minimum(xs - xmin, xmax - xs)[:, None, None]
    dy = np.minimum(ys - ymin, ymax - ys)[None, :, None]
    dz = np.minimum(zs - zmin, zmax - zs)[None, None, :]
    return np.minimum(np.minimum(dx, dy), dz).astype(np.float32)


def build_sdf(
    occupied: np.ndarray,
    bounds: np.ndarray,
    voxel_size: float,
    obstacle_inflation: float,
    truncate_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inflated_occupied = inflate_obstacles(occupied, voxel_size, obstacle_inflation)
    inflated_safe = ~inflated_occupied
    half_voxel = 0.5 * voxel_size

    if inflated_occupied.any():
        distance_to_obstacle = ndimage.distance_transform_edt(~inflated_occupied, sampling=voxel_size)
        positive_sdf = distance_to_obstacle - half_voxel
    else:
        positive_sdf = np.full(inflated_occupied.shape, np.inf, dtype=np.float32)

    if inflated_safe.any():
        distance_to_safe = ndimage.distance_transform_edt(inflated_occupied, sampling=voxel_size)
        negative_sdf = -(distance_to_safe - half_voxel)
    else:
        negative_sdf = np.full(inflated_occupied.shape, -np.inf, dtype=np.float32)

    sdf = np.where(inflated_occupied, negative_sdf, positive_sdf).astype(np.float32)
    sdf = np.minimum(sdf, workspace_boundary_distance(bounds, inflated_occupied.shape, voxel_size))

    if truncate_distance > 0.0:
        sdf = np.clip(sdf, -truncate_distance, truncate_distance)
    return sdf.astype(np.float32), inflated_occupied, inflated_safe


def main() -> None:
    args = parse_args()
    if args.obstacle_inflation < 0.0:
        raise ValueError("--obstacle-inflation must be non-negative")

    safe_space = load_safe_space(args.safe_space)
    occupied = safe_space["occupied_grid"]
    bounds = safe_space["workspace_bounds"]
    voxel_size = float(np.asarray(safe_space["voxel_size"]).item())

    sdf_grid, inflated_occupied, inflated_safe = build_sdf(
        occupied=occupied,
        bounds=bounds,
        voxel_size=voxel_size,
        obstacle_inflation=args.obstacle_inflation,
        truncate_distance=args.truncate_distance,
    )

    output = args.output
    if output is None:
        output = args.safe_space.with_name(f"{args.safe_space.stem}_sdf.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    origin = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    np.savez_compressed(
        output,
        sdf_grid=sdf_grid,
        workspace_bounds=bounds.astype(np.float32),
        voxel_size=np.array(voxel_size, dtype=np.float32),
        grid_shape=np.array(sdf_grid.shape, dtype=np.int64),
        sdf_origin=origin,
        sdf_axis_order=np.array("xyz"),
        obstacle_inflation=np.array(args.obstacle_inflation, dtype=np.float32),
        truncate_distance=np.array(args.truncate_distance, dtype=np.float32),
        inflated_occupied_grid=inflated_occupied,
        inflated_safe_grid=inflated_safe,
    )

    finite = np.isfinite(sdf_grid)
    print(f"[done] saved safe-space sdf: {output}")
    print(f"[done] grid shape: {sdf_grid.shape}")
    print(f"[done] voxel size: {voxel_size}")
    print(f"[done] workspace bounds xyz: {bounds}")
    print(f"[done] inflated obstacle cells: {int(inflated_occupied.sum())}")
    print(f"[done] safe cells after inflation: {int(inflated_safe.sum())}")
    print(f"[done] sdf min/max: {float(sdf_grid[finite].min())} / {float(sdf_grid[finite].max())}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)
