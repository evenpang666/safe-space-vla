#!/usr/bin/env python3
"""Reconstruct a camera-frame point cloud directly from a RealSense RGB-D frame."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html as html_lib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "realsense_pointcloud"


@dataclass(frozen=True)
class OrientedOBB:
    center: np.ndarray
    rotation: np.ndarray
    extents: np.ndarray
    corners: np.ndarray
    point_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--camera-name", default="front", help="Name used for output files.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--wait-timeout-ms", type=int, default=10000)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--depth-vis-max", type=float, default=4.0, help="Upper meter value for the colorized depth preview.")
    parser.add_argument("--viewer-max-points", type=int, default=60000, help="Maximum points embedded in the interactive HTML viewer.")
    parser.add_argument(
        "--tabletop-bounds",
        nargs=6,
        type=float,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="Optional camera-frame ROI in meters for desktop points. Use this to remove far background points.",
    )
    parser.add_argument("--table-plane-threshold", type=float, default=0.015, help="RANSAC table plane inlier threshold in meters.")
    parser.add_argument("--min-plane-distance", type=float, default=0.03, help="Minimum distance from the table plane for OBB obstacle points.")
    parser.add_argument("--max-plane-distance", type=float, default=0.30, help="Maximum distance from the table plane for OBB obstacle points.")
    parser.add_argument("--obb-cluster-radius", type=float, default=0.08)
    parser.add_argument("--obb-min-cluster-points", type=int, default=32)
    parser.add_argument(
        "--robot-qpos",
        nargs=6,
        type=float,
        default=None,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
        help="Optional UR7e joint vector in radians. Enables FK-based robot point filtering.",
    )
    parser.add_argument(
        "--use-rtde-qpos",
        action="store_true",
        help="Read the current UR7e joint vector from RTDE instead of passing --robot-qpos manually.",
    )
    parser.add_argument(
        "--robot-ip",
        default=None,
        help="UR robot IP for --use-rtde-qpos. Defaults to UR_ROBOT_IP or the project controller default.",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=None,
        help="Calibration JSON containing camera_to_world for --camera-name. Required with --robot-qpos.",
    )
    parser.add_argument("--robot-filter-radius", type=float, default=0.05, help="Distance threshold in meters for observed robot points.")
    parser.add_argument("--robot-points-per-link", type=int, default=128)
    parser.add_argument("--gripper-width", type=float, default=0.085)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def depth_rgb_to_camera_points(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    stride: int = 1,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"depth shape {depth.shape} must match rgb height/width {rgb.shape[:2]}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}")
    stride = max(1, int(stride))

    mask = np.isfinite(depth) & (depth > 0.0)
    if max_depth is not None:
        mask &= depth <= float(max_depth)
    mask[::stride, ::stride] &= True
    if stride > 1:
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[::stride, ::stride] = True
        mask &= stride_mask

    v, u = np.nonzero(mask)
    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    z = depth[v, u]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        raise ValueError("camera intrinsics fx/fy must be non-zero")
    points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    colors = rgb[v, u]
    return points.astype(np.float32), colors.astype(np.uint8)


def _turbo_like_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    stops = np.asarray(
        [
            [0.00, 0, 35, 255],
            [0.20, 0, 220, 255],
            [0.42, 80, 255, 100],
            [0.62, 255, 245, 0],
            [0.78, 255, 120, 0],
            [1.00, 120, 0, 0],
        ],
        dtype=np.float32,
    )
    rgb = np.empty((*values.shape, 3), dtype=np.float32)
    for channel in range(3):
        rgb[..., channel] = np.interp(values, stops[:, 0], stops[:, channel + 1])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _add_colorbar(image: np.ndarray, *, vis_max: float, bar_width: int = 44) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    canvas = Image.new("RGB", (width + int(bar_width), height), (0, 0, 0))
    canvas.paste(Image.fromarray(image), (0, 0))
    draw = ImageDraw.Draw(canvas)
    gradient = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    bar = _turbo_like_colormap(np.repeat(gradient, max(8, bar_width // 3), axis=1))
    bar_x0 = width + 4
    canvas.paste(Image.fromarray(bar), (bar_x0, 0))
    font = ImageFont.load_default()
    for value in range(0, int(np.floor(float(vis_max))) + 1):
        y = int(round(height - 1 - (float(value) / max(float(vis_max), 1e-6)) * (height - 1)))
        draw.line((bar_x0, y, bar_x0 + bar.shape[1] + 3, y), fill=(230, 230, 230))
        draw.text((bar_x0 + bar.shape[1] + 5, max(0, y - 6)), str(value), fill=(230, 230, 230), font=font)
    return np.asarray(canvas, dtype=np.uint8)


def depth_to_vis(
    depth_m: np.ndarray,
    *,
    max_depth: float | None = None,
    vis_max: float | None = None,
    with_colorbar: bool = True,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        image = np.zeros((*depth.shape, 3), dtype=np.uint8)
        return _add_colorbar(image, vis_max=float(vis_max or max_depth or 4.0)) if with_colorbar else image
    high = float(vis_max) if vis_max is not None else float(max_depth) if max_depth is not None else float(np.percentile(depth[valid], 95))
    high = max(high, 1e-6)
    scaled = np.clip(depth / high, 0.0, 1.0)
    image = _turbo_like_colormap(scaled)
    image[~valid] = 0
    return _add_colorbar(image, vis_max=high) if with_colorbar else image


def render_topdown_camera_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    image_size: int = 640,
    margin_m: float = 0.05,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    image = Image.new("RGB", (image_size, image_size), (8, 10, 12))
    if points.shape[0] == 0:
        return np.asarray(image, dtype=np.uint8)
    draw = ImageDraw.Draw(image)
    x_min, z_min = points[:, [0, 2]].min(axis=0) - float(margin_m)
    x_max, z_max = points[:, [0, 2]].max(axis=0) + float(margin_m)
    if abs(x_max - x_min) <= 1e-9:
        x_min -= 0.5
        x_max += 0.5
    if abs(z_max - z_min) <= 1e-9:
        z_min -= 0.5
        z_max += 0.5
    sx = (image_size - 1) / (x_max - x_min)
    sz = (image_size - 1) / (z_max - z_min)
    for point, color in zip(points, colors):
        x = int(round((float(point[0]) - x_min) * sx))
        y = int(round((z_max - float(point[2])) * sz))
        if 0 <= x < image_size and 0 <= y < image_size:
            draw.point((x, y), fill=tuple(int(value) for value in color))
    return np.asarray(image, dtype=np.uint8)


def save_ply_ascii(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def filter_camera_bounds(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    bounds: tuple[float, float, float, float, float, float] | list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")
    if bounds is None:
        return points, colors
    if len(bounds) != 6:
        raise ValueError("bounds must contain x_min x_max y_min y_max z_min z_max")
    x_min, x_max, y_min, y_max, z_min, z_max = [float(value) for value in bounds]
    keep = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[keep].astype(np.float32), colors[keep].astype(np.uint8)


def _plane_from_points(sample: np.ndarray) -> tuple[np.ndarray, float] | None:
    sample = np.asarray(sample, dtype=np.float64).reshape(3, 3)
    normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None
    normal = normal / norm
    offset = -float(normal @ sample[0])
    return normal, offset


def estimate_dominant_plane(
    points: np.ndarray,
    *,
    threshold: float = 0.015,
    ransac_iterations: int = 160,
) -> tuple[np.ndarray, float, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float64), 0.0, np.zeros((points.shape[0],), dtype=bool)

    rng = np.random.default_rng(0)
    best_inliers = np.zeros((points.shape[0],), dtype=bool)
    best_count = -1
    iterations = max(1, int(ransac_iterations))
    for _ in range(iterations):
        indices = rng.choice(points.shape[0], size=3, replace=False)
        plane = _plane_from_points(points[indices])
        if plane is None:
            continue
        normal, offset = plane
        distances = np.abs(points @ normal + offset)
        inliers = distances <= float(threshold)
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if int(best_inliers.sum()) >= 3:
        inlier_points = points[best_inliers]
        center = inlier_points.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
        normal = vh[-1]
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        offset = -float(normal @ center)
        distances = np.abs(points @ normal + offset)
        best_inliers = distances <= float(threshold)
        return normal.astype(np.float64), offset, best_inliers

    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    offset = -float(normal @ center)
    distances = np.abs(points @ normal + offset)
    return normal.astype(np.float64), offset, distances <= float(threshold)


def select_off_plane_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    plane_threshold: float = 0.015,
    min_plane_distance: float = 0.03,
    max_plane_distance: float = 0.30,
    ransac_iterations: int = 160,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")
    if points.shape[0] == 0:
        return points, colors, {"normal": [0.0, 1.0, 0.0], "offset": 0.0, "inlier_count": 0}

    normal, offset, inliers = estimate_dominant_plane(points, threshold=plane_threshold, ransac_iterations=ransac_iterations)
    distances = np.abs(points.astype(np.float64) @ normal + offset)
    keep = (~inliers) & (distances >= float(min_plane_distance)) & (distances <= float(max_plane_distance))
    plane = {
        "normal": normal.astype(float).tolist(),
        "offset": float(offset),
        "inlier_count": int(inliers.sum()),
    }
    return points[keep].astype(np.float32), colors[keep].astype(np.uint8), plane


def _cluster_points_3d(points: np.ndarray, *, cluster_radius: float, min_cluster_points: int) -> list[np.ndarray]:
    from collections import defaultdict, deque

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return []
    cell_size = max(float(cluster_radius), 1e-6)
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    point_cells = np.floor(points / cell_size).astype(np.int64)
    for idx, cell in enumerate(point_cells):
        cells[(int(cell[0]), int(cell[1]), int(cell[2]))].append(idx)

    visited: set[tuple[int, int, int]] = set()
    clusters: list[np.ndarray] = []
    for seed in cells:
        if seed in visited:
            continue
        queue: deque[tuple[int, int, int]] = deque([seed])
        visited.add(seed)
        cluster_indices: list[int] = []
        while queue:
            cell = queue.popleft()
            cluster_indices.extend(cells[cell])
            cx, cy, cz = cell
            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    for nz in range(cz - 1, cz + 2):
                        neighbor = (nx, ny, nz)
                        if neighbor in cells and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        if len(cluster_indices) >= int(min_cluster_points):
            clusters.append(np.asarray(cluster_indices, dtype=np.int64))
    return clusters


def _fit_pca_obb(points: np.ndarray) -> OrientedOBB | None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] < 3:
        return None
    center0 = points.astype(np.float64).mean(axis=0)
    centered = points.astype(np.float64) - center0
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotation = vh.T
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    local = points.astype(np.float64) @ rotation
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    extents = np.maximum(local_max - local_min, 1e-4)
    local_center = 0.5 * (local_min + local_max)
    center = local_center @ rotation.T
    signs = np.asarray(
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
    local_corners = local_center[None, :] + 0.5 * signs * extents[None, :]
    corners = local_corners @ rotation.T
    return OrientedOBB(
        center=center.astype(np.float32),
        rotation=rotation.astype(np.float32),
        extents=extents.astype(np.float32),
        corners=corners.astype(np.float32),
        point_count=int(points.shape[0]),
    )


def fit_oriented_obbs(points: np.ndarray, *, cluster_radius: float = 0.08, min_cluster_points: int = 32) -> list[OrientedOBB]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    obbs: list[OrientedOBB] = []
    for indices in _cluster_points_3d(points, cluster_radius=cluster_radius, min_cluster_points=min_cluster_points):
        obb = _fit_pca_obb(points[indices])
        if obb is not None:
            obbs.append(obb)
    return obbs


def save_obbs_json(path: Path, obbs: list[OrientedOBB], *, plane: dict[str, object] | None = None) -> None:
    payload = {
        "coordinate_frame": "camera",
        "plane": plane or {},
        "obbs": [
            {
                "center": obb.center.astype(float).tolist(),
                "rotation": obb.rotation.astype(float).tolist(),
                "extents": obb.extents.astype(float).tolist(),
                "corners": obb.corners.astype(float).tolist(),
                "point_count": int(obb.point_count),
            }
            for obb in obbs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    if points.shape[0] == 0:
        return points.astype(np.float32)
    homogeneous = np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (transform @ homogeneous.T).T[:, :3].astype(np.float32)


def robot_model_points_in_camera(
    qpos: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    points_per_link: int = 128,
    gripper_width: float = 0.085,
) -> np.ndarray:
    from real_scripts.real_robot_adapter import UR7ELinkPointSampler

    sampler = UR7ELinkPointSampler(points_per_link=int(points_per_link), gripper_width=float(gripper_width))
    world_robot_points = sampler.link_points(np.asarray(qpos, dtype=np.float32).reshape(-1))
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    return transform_points(world_robot_points.reshape(-1, 3), world_to_camera)


def split_robot_points_by_model_distance(
    points: np.ndarray,
    colors: np.ndarray,
    robot_model_points: np.ndarray,
    *,
    radius: float,
    chunk_size: int = 16384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    robot_model_points = np.asarray(robot_model_points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")
    if points.shape[0] == 0:
        mask = np.zeros((0,), dtype=bool)
        return points, colors, points, colors, mask
    if robot_model_points.shape[0] == 0 or float(radius) <= 0.0:
        mask = np.zeros((points.shape[0],), dtype=bool)
        return points[mask], colors[mask], points[~mask], colors[~mask], mask

    mask = np.zeros((points.shape[0],), dtype=bool)
    radius_sq = float(radius) ** 2
    for start in range(0, points.shape[0], max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), points.shape[0])
        diff = points[start:stop, None, :] - robot_model_points[None, :, :]
        min_dist_sq = np.min(np.sum(diff * diff, axis=-1), axis=1)
        mask[start:stop] = min_dist_sq <= radius_sq
    return (
        points[mask].astype(np.float32),
        colors[mask].astype(np.uint8),
        points[~mask].astype(np.float32),
        colors[~mask].astype(np.uint8),
        mask,
    )


def resolve_robot_qpos(args: argparse.Namespace, *, controller_factory=None) -> np.ndarray | None:
    use_rtde_qpos = bool(getattr(args, "use_rtde_qpos", False))
    manual_qpos = getattr(args, "robot_qpos", None)
    if use_rtde_qpos and manual_qpos is not None:
        raise ValueError("Use either --use-rtde-qpos or --robot-qpos, not both.")
    if manual_qpos is not None:
        qpos = np.asarray(manual_qpos, dtype=np.float32).reshape(-1)
        if qpos.size != 6:
            raise ValueError(f"--robot-qpos must contain 6 values, got {qpos.size}")
        return qpos
    if not use_rtde_qpos:
        return None

    if controller_factory is None:
        from real_scripts.ur7e_controller import ROBOT_IP, UR7eVectorController

        robot_ip = getattr(args, "robot_ip", None) or os.environ.get("UR_ROBOT_IP", ROBOT_IP)
        controller_factory = lambda ip: UR7eVectorController(robot_ip=ip, strict_gripper_connection=False)
    else:
        robot_ip = getattr(args, "robot_ip", None) or os.environ.get("UR_ROBOT_IP")
    if not robot_ip:
        from real_scripts.ur7e_controller import ROBOT_IP

        robot_ip = ROBOT_IP

    controller = controller_factory(robot_ip)
    try:
        controller.connect()
        qpos = np.asarray(controller.get_current_joints(), dtype=np.float32).reshape(-1)
        if qpos.size != 6:
            raise RuntimeError(f"RTDE returned {qpos.size} joints; expected 6")
        return qpos
    finally:
        if hasattr(controller, "close"):
            controller.close()


def _viewer_payload(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int | None = None,
    obbs: list[OrientedOBB] | None = None,
) -> dict[str, object]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")
    if max_points is not None and max_points > 0 and points.shape[0] > int(max_points):
        indices = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
        points = points[indices]
        colors = colors[indices]
    return {
        "points": points.astype(float).tolist(),
        "colors": colors.astype(int).tolist(),
        "obbs": [
            {
                "corners": obb.corners.astype(float).tolist(),
                "center": obb.center.astype(float).tolist(),
                "extents": obb.extents.astype(float).tolist(),
                "point_count": int(obb.point_count),
            }
            for obb in (obbs or [])
        ],
    }


def save_interactive_pointcloud_html(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    title: str = "RealSense point cloud",
    max_points: int | None = None,
    obbs: list[OrientedOBB] | None = None,
) -> None:
    payload = _viewer_payload(points, colors, max_points=max_points, obbs=obbs)
    payload_json = json.dumps(payload, separators=(",", ":"))
    safe_title = html_lib.escape(str(title))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #080a0d; color: #e6edf3; font-family: Arial, sans-serif; }}
    #hud {{ position: fixed; left: 12px; top: 10px; z-index: 2; background: rgba(0,0,0,.48); padding: 8px 10px; border-radius: 6px; font-size: 13px; line-height: 1.45; }}
    canvas {{ width: 100vw; height: 100vh; display: block; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
  </style>
</head>
<body>
  <div id="hud">
    <strong>{safe_title}</strong><br>
    drag: rotate | wheel: zoom | points: <span id="count"></span>
  </div>
  <canvas id="viewer"></canvas>
  <script>
const POINT_DATA = {payload_json};
const canvas = document.getElementById('viewer');
const countEl = document.getElementById('count');
countEl.textContent = POINT_DATA.points.length.toString();
const gl = canvas.getContext('webgl', {{ antialias: true }});
if (!gl) {{
  document.body.innerHTML = '<pre style="padding:16px">WebGL is not available in this browser.</pre>';
  throw new Error('webgl unavailable');
}}

const vertexShaderSource = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform mat4 uMvp;
uniform float uPointSize;
varying vec3 vColor;
void main() {{
  gl_Position = uMvp * vec4(aPosition, 1.0);
  gl_PointSize = uPointSize;
  vColor = aColor / 255.0;
}}`;
const fragmentShaderSource = `
precision mediump float;
varying vec3 vColor;
void main() {{
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  if (dot(uv, uv) > 1.0) discard;
  gl_FragColor = vec4(vColor, 1.0);
}}`;

function compileShader(type, source) {{
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
    throw new Error(gl.getShaderInfoLog(shader));
  }}
  return shader;
}}
const program = gl.createProgram();
gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexShaderSource));
gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
  throw new Error(gl.getProgramInfoLog(program));
}}
gl.useProgram(program);

const n = POINT_DATA.points.length;
const positions = new Float32Array(n * 3);
const colors = new Float32Array(n * 3);
let cx = 0, cy = 0, cz = 0;
for (let i = 0; i < n; i++) {{
  cx += POINT_DATA.points[i][0];
  cy += POINT_DATA.points[i][1];
  cz += POINT_DATA.points[i][2];
}}
if (n > 0) {{ cx /= n; cy /= n; cz /= n; }}
let radius = 0.1;
for (let i = 0; i < n; i++) {{
  const p = POINT_DATA.points[i];
  const x = p[0] - cx, y = p[1] - cy, z = p[2] - cz;
  positions[i * 3] = x;
  positions[i * 3 + 1] = -y;
  positions[i * 3 + 2] = -z;
  const d = Math.hypot(x, y, z);
  if (d > radius) radius = d;
  const c = POINT_DATA.colors[i];
  colors[i * 3] = c[0];
  colors[i * 3 + 1] = c[1];
  colors[i * 3 + 2] = c[2];
}}

function bindBuffer(name, data) {{
  const location = gl.getAttribLocation(program, name);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 0, 0);
}}
bindBuffer('aPosition', positions);
bindBuffer('aColor', colors);

const edgeIndices = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7];
const linePositions = [];
const lineColors = [];
for (const obb of POINT_DATA.obbs || []) {{
  for (let i = 0; i < edgeIndices.length; i++) {{
    const p = obb.corners[edgeIndices[i]];
    linePositions.push(p[0] - cx, -(p[1] - cy), -(p[2] - cz));
    lineColors.push(40, 255, 90);
  }}
}}
const linePositionData = new Float32Array(linePositions);
const lineColorData = new Float32Array(lineColors);
function makeBuffer(data) {{
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return buffer;
}}
const linePositionBuffer = makeBuffer(linePositionData);
const lineColorBuffer = makeBuffer(lineColorData);

const uMvp = gl.getUniformLocation(program, 'uMvp');
const uPointSize = gl.getUniformLocation(program, 'uPointSize');
let yaw = 0.35;
let pitch = -0.25;
let distance = Math.max(radius * 2.8, 0.5);
let dragging = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener('mousedown', event => {{
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
}});
window.addEventListener('mouseup', () => {{ dragging = false; }});
canvas.addEventListener('mousemove', event => {{
  if (!dragging) return;
  yaw += (event.clientX - lastX) * 0.008;
  pitch += (event.clientY - lastY) * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
  lastX = event.clientX;
  lastY = event.clientY;
  draw();
}});
canvas.addEventListener('wheel', event => {{
  event.preventDefault();
  distance *= Math.exp(event.deltaY * 0.001);
  distance = Math.max(radius * 0.3, Math.min(radius * 20.0, distance));
  draw();
}}, {{ passive: false }});

function multiply(a, b) {{
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) {{
    for (let c = 0; c < 4; c++) {{
      out[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] + a[1 * 4 + r] * b[c * 4 + 1] + a[2 * 4 + r] * b[c * 4 + 2] + a[3 * 4 + r] * b[c * 4 + 3];
    }}
  }}
  return out;
}}
function perspective(fovy, aspect, near, far) {{
  const f = 1.0 / Math.tan(fovy / 2.0);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}}
function viewMatrix() {{
  const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const eye = [distance * syaw * cp, distance * sp, distance * cyaw * cp];
  const z = normalize(eye);
  const x = normalize([z[2], 0, -z[0]]);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}}
function normalize(v) {{
  const d = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / d, v[1] / d, v[2] / d];
}}
function dot(a, b) {{ return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }}

function resize() {{
  const scale = window.devicePixelRatio || 1;
  const width = Math.floor(canvas.clientWidth * scale);
  const height = Math.floor(canvas.clientHeight * scale);
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width;
    canvas.height = height;
  }}
  gl.viewport(0, 0, canvas.width, canvas.height);
}}
function draw() {{
  resize();
  gl.clearColor(0.03, 0.04, 0.05, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  const aspect = canvas.width / Math.max(1, canvas.height);
  const projection = perspective(Math.PI / 4, aspect, Math.max(0.001, radius * 0.01), Math.max(10.0, radius * 30.0));
  const mvp = multiply(projection, viewMatrix());
  gl.uniformMatrix4fv(uMvp, false, mvp);
  gl.uniform1f(uPointSize, Math.max(1.5, Math.min(4.0, window.devicePixelRatio || 1)));
  bindBuffer('aPosition', positions);
  bindBuffer('aColor', colors);
  gl.drawArrays(gl.POINTS, 0, n);
  if (linePositionData.length > 0) {{
    const posLocation = gl.getAttribLocation(program, 'aPosition');
    const colorLocation = gl.getAttribLocation(program, 'aColor');
    gl.bindBuffer(gl.ARRAY_BUFFER, linePositionBuffer);
    gl.enableVertexAttribArray(posLocation);
    gl.vertexAttribPointer(posLocation, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineColorBuffer);
    gl.enableVertexAttribArray(colorLocation);
    gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
    gl.lineWidth(2);
    gl.drawArrays(gl.LINES, 0, linePositionData.length / 3);
  }}
}}
window.addEventListener('resize', draw);
draw();
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _save_rgb(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _save_point_npz(path: Path, points: np.ndarray, colors: np.ndarray, **extra_arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points": np.asarray(points, dtype=np.float32),
        "colors": np.asarray(colors, dtype=np.uint8),
        "coordinate_frame": np.asarray("camera"),
    }
    payload.update(extra_arrays)
    np.savez_compressed(path, **payload)


def _save_tabletop_outputs(
    output_dir: Path,
    *,
    prefix: str,
    points: np.ndarray,
    colors: np.ndarray,
    tabletop_bounds: tuple[float, float, float, float, float, float] | list[float],
    plane_threshold: float,
    min_plane_distance: float,
    max_plane_distance: float,
    obb_cluster_radius: float,
    obb_min_cluster_points: int,
    viewer_max_points: int,
) -> dict[str, Path]:
    tabletop_points, tabletop_colors = filter_camera_bounds(points, colors, bounds=tabletop_bounds)
    obstacle_points, obstacle_colors, plane = select_off_plane_points(
        tabletop_points,
        tabletop_colors,
        plane_threshold=plane_threshold,
        min_plane_distance=min_plane_distance,
        max_plane_distance=max_plane_distance,
    )
    obbs = fit_oriented_obbs(
        obstacle_points,
        cluster_radius=obb_cluster_radius,
        min_cluster_points=obb_min_cluster_points,
    )
    outputs = {
        "tabletop_npz": output_dir / f"{prefix}_tabletop_pointcloud.npz",
        "tabletop_ply": output_dir / f"{prefix}_tabletop_pointcloud.ply",
        "tabletop_html": output_dir / f"{prefix}_tabletop_pointcloud_viewer.html",
        "tabletop_obstacle_npz": output_dir / f"{prefix}_tabletop_obstacle_points.npz",
        "tabletop_obstacle_ply": output_dir / f"{prefix}_tabletop_obstacle_points.ply",
        "tabletop_obbs_json": output_dir / f"{prefix}_tabletop_obbs.json",
        "tabletop_obbs_html": output_dir / f"{prefix}_tabletop_obbs_viewer.html",
    }
    _save_point_npz(outputs["tabletop_npz"], tabletop_points, tabletop_colors, bounds=np.asarray(tabletop_bounds, dtype=np.float32))
    save_ply_ascii(outputs["tabletop_ply"], tabletop_points, tabletop_colors)
    save_interactive_pointcloud_html(
        outputs["tabletop_html"],
        tabletop_points,
        tabletop_colors,
        title=f"{prefix} tabletop ROI point cloud",
        max_points=viewer_max_points,
    )
    _save_point_npz(
        outputs["tabletop_obstacle_npz"],
        obstacle_points,
        obstacle_colors,
        plane_normal=np.asarray(plane["normal"], dtype=np.float32),
        plane_offset=np.asarray(plane["offset"], dtype=np.float32),
    )
    save_ply_ascii(outputs["tabletop_obstacle_ply"], obstacle_points, obstacle_colors)
    save_obbs_json(outputs["tabletop_obbs_json"], obbs, plane=plane)
    save_interactive_pointcloud_html(
        outputs["tabletop_obbs_html"],
        obstacle_points,
        obstacle_colors,
        title=f"{prefix} tabletop obstacle OBBs",
        max_points=viewer_max_points,
        obbs=obbs,
    )
    return outputs


def _save_robot_filter_outputs(
    output_dir: Path,
    *,
    prefix: str,
    points: np.ndarray,
    colors: np.ndarray,
    robot_qpos: np.ndarray,
    camera_to_world: np.ndarray,
    robot_filter_radius: float,
    robot_points_per_link: int,
    gripper_width: float,
    viewer_max_points: int,
) -> dict[str, Path]:
    robot_model_points = robot_model_points_in_camera(
        robot_qpos,
        camera_to_world,
        points_per_link=robot_points_per_link,
        gripper_width=gripper_width,
    )
    robot_points, robot_colors, non_robot_points, non_robot_colors, mask = split_robot_points_by_model_distance(
        points,
        colors,
        robot_model_points,
        radius=robot_filter_radius,
    )
    outputs = {
        "robot_observed_npz": output_dir / f"{prefix}_robot_observed_points.npz",
        "robot_observed_ply": output_dir / f"{prefix}_robot_observed_points.ply",
        "robot_observed_html": output_dir / f"{prefix}_robot_observed_points_viewer.html",
        "non_robot_npz": output_dir / f"{prefix}_non_robot_points.npz",
        "non_robot_ply": output_dir / f"{prefix}_non_robot_points.ply",
        "non_robot_html": output_dir / f"{prefix}_non_robot_points_viewer.html",
        "fk_robot_model_npz": output_dir / f"{prefix}_fk_robot_model_points.npz",
        "fk_robot_model_ply": output_dir / f"{prefix}_fk_robot_model_points.ply",
        "fk_robot_model_html": output_dir / f"{prefix}_fk_robot_model_points_viewer.html",
    }
    _save_point_npz(
        outputs["robot_observed_npz"],
        robot_points,
        robot_colors,
        robot_mask=mask.astype(np.uint8),
        robot_filter_radius=np.asarray(float(robot_filter_radius), dtype=np.float32),
    )
    save_ply_ascii(outputs["robot_observed_ply"], robot_points, robot_colors)
    save_interactive_pointcloud_html(
        outputs["robot_observed_html"],
        robot_points,
        robot_colors,
        title=f"{prefix} observed robot points",
        max_points=viewer_max_points,
    )
    _save_point_npz(outputs["non_robot_npz"], non_robot_points, non_robot_colors)
    save_ply_ascii(outputs["non_robot_ply"], non_robot_points, non_robot_colors)
    save_interactive_pointcloud_html(
        outputs["non_robot_html"],
        non_robot_points,
        non_robot_colors,
        title=f"{prefix} non-robot points",
        max_points=viewer_max_points,
    )
    model_colors = np.tile(np.asarray([[0, 220, 255]], dtype=np.uint8), (robot_model_points.shape[0], 1))
    _save_point_npz(
        outputs["fk_robot_model_npz"],
        robot_model_points,
        model_colors,
        qpos=np.asarray(robot_qpos, dtype=np.float32),
    )
    save_ply_ascii(outputs["fk_robot_model_ply"], robot_model_points, model_colors)
    save_interactive_pointcloud_html(
        outputs["fk_robot_model_html"],
        robot_model_points,
        model_colors,
        title=f"{prefix} FK robot model points",
        max_points=viewer_max_points,
    )
    return outputs


def save_pointcloud_outputs(
    output_dir: Path,
    *,
    name: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    depth_vis_max: float = 4.0,
    viewer_max_points: int = 60000,
    tabletop_bounds: tuple[float, float, float, float, float, float] | list[float] | None = None,
    table_plane_threshold: float = 0.015,
    min_plane_distance: float = 0.03,
    max_plane_distance: float = 0.30,
    obb_cluster_radius: float = 0.08,
    obb_min_cluster_points: int = 32,
    robot_qpos: np.ndarray | None = None,
    camera_to_world: np.ndarray | None = None,
    robot_filter_radius: float = 0.05,
    robot_points_per_link: int = 128,
    gripper_width: float = 0.085,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(name)
    outputs = {
        "npz": output_dir / f"{prefix}_pointcloud.npz",
        "ply": output_dir / f"{prefix}_pointcloud.ply",
        "rgb": output_dir / f"{prefix}_rgb.png",
        "depth_vis": output_dir / f"{prefix}_depth_vis.png",
        "topdown": output_dir / f"{prefix}_topdown_camera_points.png",
        "html": output_dir / f"{prefix}_pointcloud_viewer.html",
    }
    np.savez_compressed(
        outputs["npz"],
        points=np.asarray(points, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.uint8),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        depth_m=np.asarray(depth_m, dtype=np.float32),
        coordinate_frame=np.asarray("camera"),
    )
    save_ply_ascii(outputs["ply"], points, colors)
    _save_rgb(outputs["rgb"], rgb)
    _save_rgb(outputs["depth_vis"], depth_to_vis(depth_m, vis_max=depth_vis_max, with_colorbar=True))
    _save_rgb(outputs["topdown"], render_topdown_camera_points(points, colors))
    save_interactive_pointcloud_html(
        outputs["html"],
        points,
        colors,
        title=f"{prefix} RealSense point cloud",
        max_points=viewer_max_points,
        )
    if tabletop_bounds is not None:
        outputs.update(
            _save_tabletop_outputs(
                output_dir,
                prefix=prefix,
                points=points,
                colors=colors,
                tabletop_bounds=tabletop_bounds,
                plane_threshold=table_plane_threshold,
                min_plane_distance=min_plane_distance,
                max_plane_distance=max_plane_distance,
                obb_cluster_radius=obb_cluster_radius,
                obb_min_cluster_points=obb_min_cluster_points,
                viewer_max_points=viewer_max_points,
            )
        )
    if robot_qpos is not None:
        if camera_to_world is None:
            raise ValueError("camera_to_world is required when robot_qpos is provided")
        outputs.update(
            _save_robot_filter_outputs(
                output_dir,
                prefix=prefix,
                points=points,
                colors=colors,
                robot_qpos=np.asarray(robot_qpos, dtype=np.float32),
                camera_to_world=np.asarray(camera_to_world, dtype=np.float64),
                robot_filter_radius=robot_filter_radius,
                robot_points_per_link=robot_points_per_link,
                gripper_width=gripper_width,
                viewer_max_points=viewer_max_points,
            )
        )
    return outputs


def capture_realsense_rgbd(
    *,
    serial: str | None,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    wait_timeout_ms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("This script requires pyrealsense2.") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(str(serial))
    config.enable_stream(rs.stream.color, int(width), int(height), rs.format.rgb8, int(fps))
    config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        frames = None
        for _ in range(max(1, int(warmup_frames))):
            frames = align.process(pipeline.wait_for_frames(int(wait_timeout_ms)))
        if frames is None:
            raise RuntimeError("No RealSense frames were captured.")
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to read synchronized RGB-D frames.")
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsics = np.asarray([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
        rgb = np.ascontiguousarray(np.asarray(color_frame.get_data(), dtype=np.uint8))
        depth_m = np.asarray(depth_frame.get_data(), dtype=np.float32) * float(depth_frame.get_units())
        return rgb, depth_m.astype(np.float32), intrinsics
    finally:
        pipeline.stop()


def main() -> None:
    args = parse_args()
    robot_qpos = resolve_robot_qpos(args)
    camera_to_world = None
    if robot_qpos is not None:
        if args.camera_calibration is None:
            raise ValueError("--camera-calibration is required when robot filtering is enabled")
        from real_scripts.real_robot_adapter import load_camera_calibrations

        calibrations = load_camera_calibrations(args.camera_calibration)
        if args.camera_name not in calibrations:
            raise KeyError(f"Missing camera calibration {args.camera_name!r}")
        camera_to_world = calibrations[args.camera_name].camera_to_world

    rgb, depth_m, intrinsics = capture_realsense_rgbd(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        wait_timeout_ms=args.wait_timeout_ms,
    )
    points, colors = depth_rgb_to_camera_points(
        rgb,
        depth_m,
        intrinsics,
        stride=args.stride,
        max_depth=args.max_depth,
    )
    outputs = save_pointcloud_outputs(
        args.output_dir,
        name=args.camera_name,
        rgb=rgb,
        depth_m=depth_m,
        points=points,
        colors=colors,
        intrinsics=intrinsics,
        depth_vis_max=args.depth_vis_max,
        viewer_max_points=args.viewer_max_points,
        tabletop_bounds=args.tabletop_bounds,
        table_plane_threshold=args.table_plane_threshold,
        min_plane_distance=args.min_plane_distance,
        max_plane_distance=args.max_plane_distance,
        obb_cluster_radius=args.obb_cluster_radius,
        obb_min_cluster_points=args.obb_min_cluster_points,
        robot_qpos=robot_qpos,
        camera_to_world=camera_to_world,
        robot_filter_radius=args.robot_filter_radius,
        robot_points_per_link=args.robot_points_per_link,
        gripper_width=args.gripper_width,
    )
    print(f"[done] reconstructed {points.shape[0]} camera-frame points")
    for label, path in outputs.items():
        print(f"[done] {label}: {path}")


if __name__ == "__main__":
    main()
