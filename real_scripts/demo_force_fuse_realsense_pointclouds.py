#!/usr/bin/env python3
"""Force-fuse two RealSense point clouds without external calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.generate_realsense_tabletop_scene import DEFAULT_FRONT_SERIAL, DEFAULT_SIDE_SERIAL  # noqa: E402
from real_scripts.reconstruct_realsense_pointcloud import (  # noqa: E402
    capture_realsense_rgbd,
    depth_rgb_to_camera_points,
    depth_to_vis,
    filter_camera_bounds,
    fit_oriented_obbs,
    save_interactive_pointcloud_html,
    save_ply_ascii,
    select_off_plane_points,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "realsense_force_fused_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-serial", default=os.environ.get("REAL_SENSE_FRONT_SERIAL", DEFAULT_FRONT_SERIAL))
    parser.add_argument("--side-serial", default=os.environ.get("REAL_SENSE_SIDE_SERIAL", DEFAULT_SIDE_SERIAL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="front_side_force_fused")
    parser.add_argument("--width", type=int, default=int(os.environ.get("REAL_SENSE_WIDTH", "640")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("REAL_SENSE_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("REAL_SENSE_FPS", "30")))
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--wait-timeout-ms", type=int, default=int(os.environ.get("REAL_SENSE_WAIT_TIMEOUT_MS", "10000")))
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--depth-vis-max", type=float, default=4.0)
    parser.add_argument("--viewer-max-points", type=int, default=80000)
    parser.add_argument(
        "--side-translation",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Rough side-camera translation in the front-camera demo frame, meters.",
    )
    parser.add_argument(
        "--side-rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Rough side-camera rotation in degrees applied before translation.",
    )
    parser.add_argument(
        "--auto-align-by-clusters",
        action="store_true",
        help=(
            "Estimate an extra side-to-front correction from per-camera tabletop obstacle clusters. "
            "One matched cluster gives translation only; two or more clusters also estimate rotation."
        ),
    )
    parser.add_argument(
        "--alignment-bounds",
        nargs=6,
        type=float,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="Optional native camera-frame ROI used only before per-camera cluster alignment.",
    )
    parser.add_argument("--alignment-max-clusters", type=int, default=6)
    parser.add_argument(
        "--tabletop-bounds",
        nargs=6,
        type=float,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="Optional ROI in the forced-fused demo frame.",
    )
    parser.add_argument("--table-plane-threshold", type=float, default=0.015)
    parser.add_argument("--min-plane-distance", type=float, default=0.03)
    parser.add_argument("--max-plane-distance", type=float, default=0.30)
    parser.add_argument("--obb-cluster-radius", type=float, default=0.08)
    parser.add_argument("--obb-min-cluster-points", type=int, default=32)
    return parser.parse_args()


def _rotation_from_rpy_deg(rpy_deg: tuple[float, float, float] | list[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def side_to_front_demo_transform(
    *,
    translation: tuple[float, float, float] | list[float],
    rpy_deg: tuple[float, float, float] | list[float],
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_rpy_deg(rpy_deg)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return points.copy()
    homogeneous = np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3].astype(np.float32)


def _extract_alignment_clusters(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    bounds: tuple[float, float, float, float, float, float] | list[float] | None,
    plane_threshold: float,
    min_plane_distance: float,
    max_plane_distance: float,
    cluster_radius: float,
    min_cluster_points: int,
    max_clusters: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    roi_points, roi_colors = filter_camera_bounds(points, colors, bounds=bounds)
    obstacle_points, _obstacle_colors, plane = select_off_plane_points(
        roi_points,
        roi_colors,
        plane_threshold=plane_threshold,
        min_plane_distance=min_plane_distance,
        max_plane_distance=max_plane_distance,
    )
    obbs = fit_oriented_obbs(
        obstacle_points,
        cluster_radius=cluster_radius,
        min_cluster_points=min_cluster_points,
    )
    obbs = sorted(obbs, key=lambda obb: int(obb.point_count), reverse=True)[: max(1, int(max_clusters))]
    centers = np.asarray([obb.center for obb in obbs], dtype=np.float32).reshape(-1, 3)
    counts = np.asarray([obb.point_count for obb in obbs], dtype=np.int64)
    summary = {
        "roi_point_count": int(roi_points.shape[0]),
        "obstacle_point_count": int(obstacle_points.shape[0]),
        "cluster_count": int(centers.shape[0]),
        "cluster_centers": centers.astype(float).tolist(),
        "cluster_point_counts": counts.astype(int).tolist(),
        "plane": plane,
    }
    return centers, counts, summary


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-9 or target_norm <= 1e-9:
        return np.eye(3, dtype=np.float64)
    a = source / source_norm
    b = target / target_norm
    cross = np.cross(a, b)
    dot = float(np.clip(a @ b, -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm <= 1e-9:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.cross(a, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) <= 1e-9:
            axis = np.cross(a, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        skew = np.asarray(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * (skew @ skew)
    axis = cross / cross_norm
    angle = float(np.arctan2(cross_norm, dot))
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _estimate_matched_cluster_transform(
    source_centers: np.ndarray,
    target_centers: np.ndarray,
) -> tuple[np.ndarray, str]:
    source = np.asarray(source_centers, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_centers, dtype=np.float64).reshape(-1, 3)
    match_count = min(source.shape[0], target.shape[0])
    transform = np.eye(4, dtype=np.float64)
    if match_count <= 0:
        return transform, "none"
    source = source[:match_count]
    target = target[:match_count]
    if match_count == 1:
        transform[:3, 3] = target[0] - source[0]
        return transform, "translation_only"
    if match_count == 2:
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        rotation = _rotation_between_vectors(source[1] - source[0], target[1] - target[0])
        transform[:3, :3] = rotation
        transform[:3, 3] = target_center - rotation @ source_center
        return transform, "two_cluster_vector"

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ target_centered
    u, _s, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform, "kabsch"


def estimate_cluster_side_to_front_transform(
    *,
    front_points: np.ndarray,
    front_colors: np.ndarray,
    side_points: np.ndarray,
    side_colors: np.ndarray,
    seed_side_to_front: np.ndarray,
    alignment_bounds: tuple[float, float, float, float, float, float] | list[float] | None,
    plane_threshold: float,
    min_plane_distance: float,
    max_plane_distance: float,
    cluster_radius: float,
    min_cluster_points: int,
    max_clusters: int,
) -> tuple[np.ndarray, dict[str, object]]:
    front_centers, front_counts, front_summary = _extract_alignment_clusters(
        front_points,
        front_colors,
        bounds=alignment_bounds,
        plane_threshold=plane_threshold,
        min_plane_distance=min_plane_distance,
        max_plane_distance=max_plane_distance,
        cluster_radius=cluster_radius,
        min_cluster_points=min_cluster_points,
        max_clusters=max_clusters,
    )
    side_centers_raw, side_counts, side_summary = _extract_alignment_clusters(
        side_points,
        side_colors,
        bounds=alignment_bounds,
        plane_threshold=plane_threshold,
        min_plane_distance=min_plane_distance,
        max_plane_distance=max_plane_distance,
        cluster_radius=cluster_radius,
        min_cluster_points=min_cluster_points,
        max_clusters=max_clusters,
    )
    seed = np.asarray(seed_side_to_front, dtype=np.float64)
    side_centers_seeded = _transform_points(side_centers_raw, seed)
    correction, method = _estimate_matched_cluster_transform(side_centers_seeded, front_centers)
    final_transform = correction @ seed
    match_count = int(min(front_centers.shape[0], side_centers_seeded.shape[0]))
    summary = {
        "enabled": True,
        "method": method,
        "match_count": match_count,
        "front": front_summary,
        "side": side_summary,
        "front_matched_centers": front_centers[:match_count].astype(float).tolist(),
        "side_seeded_matched_centers": side_centers_seeded[:match_count].astype(float).tolist(),
        "front_cluster_point_counts": front_counts[:match_count].astype(int).tolist(),
        "side_cluster_point_counts": side_counts[:match_count].astype(int).tolist(),
        "seed_side_to_front_demo_transform": seed.astype(float).tolist(),
        "cluster_correction_transform": correction.astype(float).tolist(),
        "side_to_front_demo_transform": final_transform.astype(float).tolist(),
    }
    return final_transform, summary


def _save_rgb(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _capture_camera_cloud(
    *,
    serial: str,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    wait_timeout_ms: int,
    stride: int,
    max_depth: float,
    capture_fn=capture_realsense_rgbd,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb, depth_m, intrinsics = capture_fn(
        serial=str(serial),
        width=int(width),
        height=int(height),
        fps=int(fps),
        warmup_frames=int(warmup_frames),
        wait_timeout_ms=int(wait_timeout_ms),
    )
    points, colors = depth_rgb_to_camera_points(rgb, depth_m, intrinsics, stride=stride, max_depth=max_depth)
    return rgb, depth_m, intrinsics, points, colors


def _obb_arrays(obbs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not obbs:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 8, 3), dtype=np.float32),
        )
    centers = np.asarray([obb.center for obb in obbs], dtype=np.float32)
    rotations = np.asarray([obb.rotation for obb in obbs], dtype=np.float32)
    extents = np.asarray([obb.extents for obb in obbs], dtype=np.float32)
    corners = np.asarray([obb.corners for obb in obbs], dtype=np.float32)
    return centers, rotations, extents, corners


def _write_obbs_json(
    path: Path,
    obbs: list,
    *,
    plane: dict[str, object],
    side_transform: np.ndarray,
    cluster_alignment: dict[str, object],
) -> None:
    payload = {
        "coordinate_frame": "front_camera_force_fused_demo",
        "plane": plane,
        "side_to_front_demo_transform": np.asarray(side_transform, dtype=float).tolist(),
        "cluster_alignment": cluster_alignment,
        "obbs": [
            {
                "center": np.asarray(obb.center, dtype=float).tolist(),
                "rotation": np.asarray(obb.rotation, dtype=float).tolist(),
                "extents": np.asarray(obb.extents, dtype=float).tolist(),
                "half_sizes": (np.asarray(obb.extents, dtype=float) / 2.0).tolist(),
                "corners": np.asarray(obb.corners, dtype=float).tolist(),
                "point_count": int(obb.point_count),
            }
            for obb in obbs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_force_fuse_demo(args: argparse.Namespace, *, capture_fn=capture_realsense_rgbd) -> dict[str, Path | int]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = str(args.name)

    front_rgb, front_depth, front_intr, front_points, front_colors = _capture_camera_cloud(
        serial=args.front_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        wait_timeout_ms=args.wait_timeout_ms,
        stride=args.stride,
        max_depth=args.max_depth,
        capture_fn=capture_fn,
    )
    side_rgb, side_depth, side_intr, side_points_raw, side_colors = _capture_camera_cloud(
        serial=args.side_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        wait_timeout_ms=args.wait_timeout_ms,
        stride=args.stride,
        max_depth=args.max_depth,
        capture_fn=capture_fn,
    )
    seed_side_transform = side_to_front_demo_transform(translation=args.side_translation, rpy_deg=args.side_rpy_deg)
    cluster_alignment: dict[str, object] = {
        "enabled": False,
        "side_to_front_demo_transform": seed_side_transform.astype(float).tolist(),
    }
    side_transform = seed_side_transform
    if bool(getattr(args, "auto_align_by_clusters", False)):
        side_transform, cluster_alignment = estimate_cluster_side_to_front_transform(
            front_points=front_points,
            front_colors=front_colors,
            side_points=side_points_raw,
            side_colors=side_colors,
            seed_side_to_front=seed_side_transform,
            alignment_bounds=args.alignment_bounds,
            plane_threshold=args.table_plane_threshold,
            min_plane_distance=args.min_plane_distance,
            max_plane_distance=args.max_plane_distance,
            cluster_radius=args.obb_cluster_radius,
            min_cluster_points=args.obb_min_cluster_points,
            max_clusters=args.alignment_max_clusters,
        )
    side_points = _transform_points(side_points_raw, side_transform)

    points = np.concatenate([front_points, side_points], axis=0).astype(np.float32)
    colors = np.concatenate([front_colors, side_colors], axis=0).astype(np.uint8)
    source_camera = np.concatenate(
        [
            np.zeros((front_points.shape[0],), dtype=np.uint8),
            np.ones((side_points.shape[0],), dtype=np.uint8),
        ],
        axis=0,
    )

    tabletop_points, tabletop_colors = filter_camera_bounds(points, colors, bounds=args.tabletop_bounds)
    obstacle_points, obstacle_colors, plane = select_off_plane_points(
        tabletop_points,
        tabletop_colors,
        plane_threshold=args.table_plane_threshold,
        min_plane_distance=args.min_plane_distance,
        max_plane_distance=args.max_plane_distance,
    )
    obbs = fit_oriented_obbs(
        obstacle_points,
        cluster_radius=args.obb_cluster_radius,
        min_cluster_points=args.obb_min_cluster_points,
    )
    centers, rotations, extents, corners = _obb_arrays(obbs)

    scene_npz = output_dir / f"{name}_pointcloud.npz"
    scene_ply = output_dir / f"{name}_pointcloud.ply"
    scene_html = output_dir / f"{name}_pointcloud_viewer.html"
    tabletop_npz = output_dir / f"{name}_tabletop_pointcloud.npz"
    obstacle_npz = output_dir / f"{name}_tabletop_obstacle_points.npz"
    obbs_npz = output_dir / f"{name}_tabletop_obbs.npz"
    obbs_json = output_dir / f"{name}_tabletop_obbs.json"
    obbs_html = output_dir / f"{name}_tabletop_obbs_viewer.html"

    np.savez_compressed(
        scene_npz,
        points=points,
        colors=colors,
        source_camera=source_camera,
        front_intrinsics=np.asarray(front_intr, dtype=np.float64),
        side_intrinsics=np.asarray(side_intr, dtype=np.float64),
        side_to_front_demo_transform=side_transform.astype(np.float32),
        cluster_alignment_json=np.asarray(json.dumps(cluster_alignment)),
        coordinate_frame=np.asarray("front_camera_force_fused_demo"),
    )
    save_ply_ascii(scene_ply, points, colors)
    save_interactive_pointcloud_html(
        scene_html,
        points,
        colors,
        title=f"{name} force-fused RealSense point cloud",
        max_points=args.viewer_max_points,
    )
    np.savez_compressed(
        tabletop_npz,
        points=tabletop_points,
        colors=tabletop_colors,
        tabletop_bounds=np.asarray(args.tabletop_bounds if args.tabletop_bounds is not None else [], dtype=np.float32),
        coordinate_frame=np.asarray("front_camera_force_fused_demo"),
    )
    np.savez_compressed(
        obstacle_npz,
        points=obstacle_points,
        colors=obstacle_colors,
        plane_normal=np.asarray(plane["normal"], dtype=np.float32),
        plane_offset=np.asarray(plane["offset"], dtype=np.float32),
        coordinate_frame=np.asarray("front_camera_force_fused_demo"),
    )
    np.savez_compressed(
        obbs_npz,
        obstacle_box_centers=centers,
        obstacle_box_axes=rotations,
        obstacle_box_sizes=extents,
        obstacle_box_half_sizes=(extents / 2.0).astype(np.float32),
        obstacle_box_corners=corners,
        coordinate_frame=np.asarray("front_camera_force_fused_demo"),
    )
    _write_obbs_json(obbs_json, obbs, plane=plane, side_transform=side_transform, cluster_alignment=cluster_alignment)
    save_interactive_pointcloud_html(
        obbs_html,
        tabletop_points,
        tabletop_colors,
        title=f"{name} tabletop point cloud with force-fused OBBs",
        max_points=args.viewer_max_points,
        obbs=obbs,
    )

    _save_rgb(output_dir / f"{name}_front_rgb.png", front_rgb)
    _save_rgb(output_dir / f"{name}_side_rgb.png", side_rgb)
    _save_rgb(output_dir / f"{name}_front_depth_vis.png", depth_to_vis(front_depth, vis_max=args.depth_vis_max))
    _save_rgb(output_dir / f"{name}_side_depth_vis.png", depth_to_vis(side_depth, vis_max=args.depth_vis_max))

    return {
        "scene_npz": scene_npz,
        "scene_ply": scene_ply,
        "scene_html": scene_html,
        "tabletop_npz": tabletop_npz,
        "obstacle_npz": obstacle_npz,
        "obbs_npz": obbs_npz,
        "obbs_json": obbs_json,
        "obbs_html": obbs_html,
        "scene_point_count": int(points.shape[0]),
        "tabletop_point_count": int(tabletop_points.shape[0]),
        "obstacle_point_count": int(obstacle_points.shape[0]),
        "obb_count": int(len(obbs)),
        "cluster_alignment_match_count": int(cluster_alignment.get("match_count", 0)),
    }


def main() -> None:
    args = parse_args()
    outputs = run_force_fuse_demo(args)
    print(f"[done] force-fused scene points: {outputs['scene_point_count']}")
    print(f"[done] tabletop ROI points: {outputs['tabletop_point_count']}")
    print(f"[done] obstacle points: {outputs['obstacle_point_count']}")
    print(f"[done] OBB count: {outputs['obb_count']}")
    if int(outputs.get("cluster_alignment_match_count", 0)) > 0:
        print(f"[done] cluster alignment matches: {outputs['cluster_alignment_match_count']}")
    for key, value in outputs.items():
        if isinstance(value, Path):
            print(f"[done] {key}: {value}")


if __name__ == "__main__":
    main()
