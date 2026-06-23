#!/usr/bin/env python3
"""Record a passive UR SafetyModule overlay demo video.

Run this while another process controls the UR arm. This script only reads the
robot state and RGB-D cameras, then writes the front camera view overlaid with
UR link surface points, tabletop obstacle points, and obstacle OBBs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.real_robot_adapter import (  # noqa: E402
    CameraCalibration,
    ReplayJsonlAdapter,
    UR7ELinkPointSampler,
    crop_workspace,
    fuse_rgbd_frames,
    load_camera_calibrations,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "real_ur_safety_overlay_demo.mp4"
ROBOT_COLOR = np.asarray([0, 220, 255], dtype=np.uint8)
OBSTACLE_COLOR = np.asarray([255, 80, 20], dtype=np.uint8)
OBB_COLOR = np.asarray([40, 255, 90], dtype=np.uint8)


@dataclass(frozen=True)
class UprightOBB:
    center: np.ndarray
    rotation: np.ndarray
    extents: np.ndarray
    corners: np.ndarray
    point_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-calibration", type=Path, required=True)
    parser.add_argument("--adapter", default=None, help="Import path 'module:factory' returning a RealRobotAdapter.")
    parser.add_argument("--replay-jsonl", type=Path, default=None, help="Offline replay source for smoke tests.")
    parser.add_argument("--front-camera-name", default="front")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--points-per-link", type=int, default=128)
    parser.add_argument("--gripper-width", type=float, default=0.085)
    parser.add_argument("--pointcloud-stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--workspace-bounds", nargs=6, type=float, default=None)
    parser.add_argument("--robot-filter-radius", type=float, default=0.045)
    parser.add_argument("--table-z", type=float, default=0.0)
    parser.add_argument("--min-obstacle-height", type=float, default=0.03)
    parser.add_argument("--max-obstacle-height", type=float, default=0.50)
    parser.add_argument("--cluster-radius", type=float, default=0.08)
    parser.add_argument("--min-cluster-points", type=int, default=32)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--debug-npz", type=Path, default=None)
    return parser.parse_args()


def load_adapter(args: argparse.Namespace):
    if args.replay_jsonl is not None:
        return ReplayJsonlAdapter(args.replay_jsonl)
    if args.adapter is None:
        raise ValueError("Provide --adapter module:factory for hardware, or --replay-jsonl for offline replay.")
    module_name, sep, factory_name = str(args.adapter).partition(":")
    if not sep:
        raise ValueError("--adapter must have form module:factory")
    module = importlib.import_module(module_name)
    return getattr(module, factory_name)()


def project_world_points_to_pixels(
    points: np.ndarray,
    calibration: CameraCalibration,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    depth = np.full((points.shape[0],), np.nan, dtype=np.float64)
    if points.shape[0] == 0:
        return uv, depth, np.zeros((0,), dtype=bool)

    world_to_camera = np.linalg.inv(calibration.camera_to_world)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (depth > 1e-6)
    fx = float(calibration.intrinsics[0, 0])
    fy = float(calibration.intrinsics[1, 1])
    cx = float(calibration.intrinsics[0, 2])
    cy = float(calibration.intrinsics[1, 2])
    if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        raise ValueError("camera intrinsics fx/fy must be non-zero")

    uv[valid, 0] = fx * camera_points[valid, 0] / depth[valid] + cx
    uv[valid, 1] = fy * camera_points[valid, 1] / depth[valid] + cy
    valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] < float(width)) & (uv[:, 1] >= 0.0) & (uv[:, 1] < float(height))
    return uv, depth, valid


def select_tabletop_obstacle_points(
    environment_points: np.ndarray,
    environment_colors: np.ndarray,
    *,
    table_z: float,
    min_height_above_table: float,
    max_height_above_table: float,
    workspace_bounds: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(environment_points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(environment_colors, dtype=np.uint8).reshape(-1, 3)
    points, colors = crop_workspace(points, colors, workspace_bounds)
    if points.shape[0] == 0:
        return points, colors
    height = points[:, 2] - float(table_z)
    keep = (height >= float(min_height_above_table)) & (height <= float(max_height_above_table))
    return points[keep].astype(np.float32), colors[keep].astype(np.uint8)


def _cluster_points_by_xy_grid(points: np.ndarray, *, cluster_radius: float, min_cluster_points: int) -> list[np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return []
    cell_size = max(float(cluster_radius), 1e-6)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    xy_cells = np.floor(points[:, :2] / cell_size).astype(np.int64)
    for idx, cell in enumerate(xy_cells):
        cells[(int(cell[0]), int(cell[1]))].append(idx)

    visited: set[tuple[int, int]] = set()
    clusters: list[np.ndarray] = []
    for seed in cells:
        if seed in visited:
            continue
        queue: deque[tuple[int, int]] = deque([seed])
        visited.add(seed)
        cluster_indices: list[int] = []
        while queue:
            cell = queue.popleft()
            cluster_indices.extend(cells[cell])
            cx, cy = cell
            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    neighbor = (nx, ny)
                    if neighbor in cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        if len(cluster_indices) >= int(min_cluster_points):
            clusters.append(np.asarray(cluster_indices, dtype=np.int64))
    return clusters


def _estimate_upright_obb(points: np.ndarray) -> UprightOBB | None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] < 3:
        return None

    xy = points[:, :2].astype(np.float64)
    xy_center = xy.mean(axis=0)
    centered_xy = xy - xy_center
    if np.max(np.linalg.norm(centered_xy, axis=1)) <= 1e-8:
        xy_axes = np.eye(2, dtype=np.float64)
    else:
        cov = centered_xy.T @ centered_xy / max(points.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        xy_axes = eigvecs[:, order]
        if np.linalg.det(xy_axes) < 0.0:
            xy_axes[:, 1] *= -1.0

    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    rotation[:2, 0] = xy_axes[:, 0]
    rotation[:2, 1] = xy_axes[:, 1]
    rotation[:, 2] = z_axis

    local = points.astype(np.float64) @ rotation
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    extents = np.maximum(local_max - local_min, 1e-4)
    local_center = 0.5 * (local_min + local_max)
    center = rotation @ local_center

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
    return UprightOBB(
        center=center.astype(np.float32),
        rotation=rotation.astype(np.float32),
        extents=extents.astype(np.float32),
        corners=corners.astype(np.float32),
        point_count=int(points.shape[0]),
    )


def build_tabletop_obbs(
    obstacle_points: np.ndarray,
    *,
    cluster_radius: float,
    min_cluster_points: int,
) -> list[UprightOBB]:
    points = np.asarray(obstacle_points, dtype=np.float32).reshape(-1, 3)
    obbs: list[UprightOBB] = []
    for indices in _cluster_points_by_xy_grid(points, cluster_radius=cluster_radius, min_cluster_points=min_cluster_points):
        obb = _estimate_upright_obb(points[indices])
        if obb is not None:
            obbs.append(obb)
    return obbs


def _draw_projected_points(
    draw,
    *,
    uv: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    color: np.ndarray,
    radius: int,
) -> None:
    valid_indices = np.flatnonzero(valid)
    for idx in valid_indices[np.argsort(depth[valid_indices])[::-1]]:
        x, y = uv[idx]
        fill = tuple(int(c) for c in color)
        if int(radius) <= 0:
            draw.point((float(x), float(y)), fill=fill)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def render_overlay_frame(
    front_rgb: np.ndarray,
    *,
    front_calibration: CameraCalibration,
    robot_link_points: np.ndarray,
    obstacle_points: np.ndarray,
    obstacle_obbs: list[UprightOBB],
    point_radius: int,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(front_rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    height, width = np.asarray(front_rgb).shape[:2]

    edge_indices = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for obb in obstacle_obbs:
        corner_uv, _, corner_valid = project_world_points_to_pixels(obb.corners, front_calibration, width=width, height=height)
        for start, stop in edge_indices:
            if bool(corner_valid[start]) and bool(corner_valid[stop]):
                draw.line(
                    (tuple(corner_uv[start].tolist()), tuple(corner_uv[stop].tolist())),
                    fill=tuple(int(c) for c in OBB_COLOR),
                    width=2,
                )

    obstacle_uv, obstacle_depth, obstacle_valid = project_world_points_to_pixels(
        np.asarray(obstacle_points, dtype=np.float32).reshape(-1, 3),
        front_calibration,
        width=width,
        height=height,
    )
    _draw_projected_points(
        draw,
        uv=obstacle_uv,
        depth=obstacle_depth,
        valid=obstacle_valid,
        color=OBSTACLE_COLOR,
        radius=max(int(point_radius) - 1, 0),
    )

    robot_points = np.asarray(robot_link_points, dtype=np.float32).reshape(-1, 3)
    robot_uv, robot_depth, robot_valid = project_world_points_to_pixels(
        robot_points,
        front_calibration,
        width=width,
        height=height,
    )
    _draw_projected_points(
        draw,
        uv=robot_uv,
        depth=robot_depth,
        valid=robot_valid,
        color=ROBOT_COLOR,
        radius=int(point_radius),
    )
    return np.asarray(image, dtype=np.uint8)


def _open_video_writer(path: Path, *, fps: float):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Writing video requires imageio or imageio-ffmpeg in this environment.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=float(fps), codec="libx264", quality=8)


def _save_debug_npz(
    path: Path | None,
    *,
    robot_frames: list[np.ndarray],
    obstacle_frames: list[np.ndarray],
    obb_frames: list[list[UprightOBB]],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        robot_link_points=np.asarray(robot_frames, dtype=np.float32),
        obstacle_points=np.asarray(obstacle_frames, dtype=object),
        obb_centers=np.asarray([[obb.center for obb in frame] for frame in obb_frames], dtype=object),
        obb_extents=np.asarray([[obb.extents for obb in frame] for frame in obb_frames], dtype=object),
        obb_corners=np.asarray([[obb.corners for obb in frame] for frame in obb_frames], dtype=object),
    )


def run_demo(args: argparse.Namespace) -> int:
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")
    if args.fps <= 0.0:
        raise ValueError("--fps must be > 0")

    calibrations = load_camera_calibrations(args.camera_calibration)
    if args.front_camera_name not in calibrations:
        raise KeyError(f"Missing front camera calibration {args.front_camera_name!r}")
    front_calibration = calibrations[args.front_camera_name]
    sampler = UR7ELinkPointSampler(points_per_link=args.points_per_link, gripper_width=args.gripper_width)
    adapter = load_adapter(args)

    writer = _open_video_writer(args.output, fps=args.fps)
    robot_frames: list[np.ndarray] = []
    obstacle_frames: list[np.ndarray] = []
    obb_frames: list[list[UprightOBB]] = []
    start_time = time.monotonic()
    frame_count = 0

    adapter.reset()
    try:
        while frame_count < int(args.max_frames):
            if args.duration_sec is not None and time.monotonic() - start_time >= float(args.duration_sec):
                break

            observation = adapter.get_observation()
            frames = adapter.get_rgbd_frames()
            front_frames = [frame for frame in frames if frame.camera_name == args.front_camera_name]
            if not front_frames:
                raise KeyError(f"Adapter did not return RGB-D frame {args.front_camera_name!r}")

            qpos = np.asarray(observation["qpos"], dtype=np.float32).reshape(-1)[:6]
            robot_link_points = sampler.link_points(qpos)
            cloud = fuse_rgbd_frames(
                frames,
                calibrations,
                robot_link_points=robot_link_points,
                stride=args.pointcloud_stride,
                max_depth=args.max_depth,
                robot_filter_radius=args.robot_filter_radius,
                workspace_bounds=args.workspace_bounds,
            )
            obstacle_points, obstacle_colors = select_tabletop_obstacle_points(
                cloud.environment_points,
                cloud.environment_colors,
                table_z=args.table_z,
                min_height_above_table=args.min_obstacle_height,
                max_height_above_table=args.max_obstacle_height,
            )
            obbs = build_tabletop_obbs(
                obstacle_points,
                cluster_radius=args.cluster_radius,
                min_cluster_points=args.min_cluster_points,
            )
            overlay = render_overlay_frame(
                front_frames[0].rgb,
                front_calibration=front_calibration,
                robot_link_points=robot_link_points,
                obstacle_points=obstacle_points,
                obstacle_obbs=obbs,
                point_radius=args.point_radius,
            )
            writer.append_data(overlay)

            robot_frames.append(robot_link_points)
            obstacle_frames.append(obstacle_points)
            obb_frames.append(obbs)
            frame_count += 1

            if args.replay_jsonl is not None:
                adapter.execute_action(np.zeros((0,), dtype=np.float32))
            elif adapter.is_done():
                break
            else:
                elapsed_target = frame_count / float(args.fps)
                sleep_sec = start_time + elapsed_target - time.monotonic()
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
    finally:
        writer.close()
        adapter.close()

    _save_debug_npz(args.debug_npz, robot_frames=robot_frames, obstacle_frames=obstacle_frames, obb_frames=obb_frames)
    return frame_count


def main() -> None:
    args = parse_args()
    count = run_demo(args)
    print(f"[done] wrote {count} frames to {args.output}")
    if args.debug_npz is not None:
        print(f"[done] wrote debug point clouds to {args.debug_npz}")


if __name__ == "__main__":
    main()
