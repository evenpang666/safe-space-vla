#!/usr/bin/env python3
"""Generate a fused tabletop point cloud and obstacle OBBs from two D435i cameras."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.demo_record_ur7e_safety_overlay_video import (  # noqa: E402
    build_tabletop_obbs,
    select_tabletop_obstacle_points,
)
from real_scripts.real_robot_adapter import (  # noqa: E402
    RGBDFrame,
    UR7ELinkPointSampler,
    crop_workspace,
    fuse_rgbd_frames,
    load_camera_calibrations,
)
from real_scripts.reconstruct_realsense_pointcloud import (  # noqa: E402
    depth_to_vis,
    save_interactive_pointcloud_html,
    save_ply_ascii,
)
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource  # noqa: E402
from scripts.build_safe_space_from_pointcloud import (  # noqa: E402
    boxes_to_occupied_grid,
    table_aligned_display_bounds,
    voxel_centers,
)


DEFAULT_FRONT_SERIAL = "405622074939"
DEFAULT_SIDE_SERIAL = "348522070576"
DEFAULT_CAMERA_CALIBRATION = REPO_ROOT / "real_scripts" / "ur7e_d435i_camera_calibration.generated.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "realsense_tabletop_scene"
DEFAULT_WORKSPACE_BOUNDS = (-0.8, 0.8, -0.8, 0.8, -0.05, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-serial", default=os.environ.get("REAL_SENSE_FRONT_SERIAL", DEFAULT_FRONT_SERIAL))
    parser.add_argument("--side-serial", default=os.environ.get("REAL_SENSE_SIDE_SERIAL", DEFAULT_SIDE_SERIAL))
    parser.add_argument("--camera-calibration", type=Path, default=DEFAULT_CAMERA_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="front_side_tabletop")
    parser.add_argument("--width", type=int, default=int(os.environ.get("REAL_SENSE_WIDTH", "640")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("REAL_SENSE_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("REAL_SENSE_FPS", "30")))
    parser.add_argument("--wait-timeout-ms", type=int, default=int(os.environ.get("REAL_SENSE_WAIT_TIMEOUT_MS", "10000")))
    parser.add_argument("--read-retries", type=int, default=int(os.environ.get("REAL_SENSE_READ_RETRIES", "5")))
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--pointcloud-stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--depth-vis-max", type=float, default=4.0)
    parser.add_argument("--viewer-max-points", type=int, default=80000)
    parser.add_argument(
        "--workspace-bounds",
        nargs=6,
        type=float,
        default=list(DEFAULT_WORKSPACE_BOUNDS),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--table-z", type=float, default=0.0)
    parser.add_argument("--min-obstacle-height", type=float, default=0.03)
    parser.add_argument("--max-obstacle-height", type=float, default=0.50)
    parser.add_argument("--cluster-radius", type=float, default=0.08)
    parser.add_argument("--min-cluster-points", type=int, default=32)
    parser.add_argument("--robot-filter-radius", type=float, default=0.045)
    parser.add_argument("--use-rtde-qpos", action="store_true", help="Read UR7e joints and filter visible robot points.")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--points-per-link", type=int, default=128)
    parser.add_argument("--gripper-width", type=float, default=0.085)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    return parser.parse_args()


def _save_rgb(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _save_pointcloud_bundle(
    output_dir: Path,
    *,
    stem: str,
    points: np.ndarray,
    colors: np.ndarray,
    title: str,
    viewer_max_points: int,
    extra: dict[str, np.ndarray] | None = None,
) -> dict[str, Path]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz = output_dir / f"{stem}.npz"
    ply = output_dir / f"{stem}.ply"
    html = output_dir / f"{stem}_viewer.html"
    payload = {
        "points": points,
        "colors": colors,
        "coordinate_frame": np.asarray("world"),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(npz, **payload)
    save_ply_ascii(ply, points, colors)
    save_interactive_pointcloud_html(html, points, colors, title=title, max_points=viewer_max_points)
    return {"npz": npz, "ply": ply, "html": html}


def _capture_rgbd_frames(
    *,
    front_serial: str,
    side_serial: str,
    width: int,
    height: int,
    fps: int,
    wait_timeout_ms: int,
    read_retries: int,
    warmup_frames: int,
    source_factory=RealSenseD435iSource,
) -> list[RGBDFrame]:
    source = source_factory(
        cameras=(
            D435iCameraConfig(name="front", serial=str(front_serial)),
            D435iCameraConfig(name="side", serial=str(side_serial)),
        ),
        width=width,
        height=height,
        fps=fps,
        wait_timeout_ms=wait_timeout_ms,
        read_retries=read_retries,
    )
    source.start()
    try:
        frames_by_name = None
        for _ in range(max(1, int(warmup_frames))):
            frames_by_name = source.read()
        if frames_by_name is None:
            raise RuntimeError("No RGB-D frames were captured.")
        return [
            RGBDFrame("front", frames_by_name["front"][0], frames_by_name["front"][1]),
            RGBDFrame("side", frames_by_name["side"][0], frames_by_name["side"][1]),
        ]
    finally:
        source.stop()


def _read_rtde_qpos(args: argparse.Namespace) -> np.ndarray:
    from real_scripts.reconstruct_realsense_pointcloud import resolve_robot_qpos

    qpos_args = argparse.Namespace(use_rtde_qpos=True, robot_qpos=None, robot_ip=args.robot_ip)
    qpos = resolve_robot_qpos(qpos_args)
    if qpos is None:
        raise RuntimeError("Failed to read UR7e qpos from RTDE.")
    return np.asarray(qpos, dtype=np.float32).reshape(-1)[:6]


def _robot_points_for_filter(args: argparse.Namespace) -> np.ndarray:
    if not bool(args.use_rtde_qpos):
        return np.zeros((0, 3), dtype=np.float32)
    qpos = _read_rtde_qpos(args)
    sampler = UR7ELinkPointSampler(points_per_link=args.points_per_link, gripper_width=args.gripper_width)
    return sampler.link_points(qpos)


def _obb_arrays(
    obbs: Sequence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray([obb.center for obb in obbs], dtype=np.float32).reshape(-1, 3)
    axes = np.asarray([obb.rotation for obb in obbs], dtype=np.float32).reshape(-1, 3, 3)
    sizes = np.asarray([obb.extents for obb in obbs], dtype=np.float32).reshape(-1, 3)
    half_sizes = sizes / 2.0
    corners = np.asarray([obb.corners for obb in obbs], dtype=np.float32).reshape(-1, 8, 3)
    point_counts = np.asarray([obb.point_count for obb in obbs], dtype=np.int64)
    if len(obbs) == 0:
        centers = np.zeros((0, 3), dtype=np.float32)
        axes = np.zeros((0, 3, 3), dtype=np.float32)
        sizes = np.zeros((0, 3), dtype=np.float32)
        half_sizes = np.zeros((0, 3), dtype=np.float32)
        corners = np.zeros((0, 8, 3), dtype=np.float32)
        point_counts = np.zeros((0,), dtype=np.int64)
    mins = corners.min(axis=1) if len(obbs) else np.zeros((0, 3), dtype=np.float32)
    maxs = corners.max(axis=1) if len(obbs) else np.zeros((0, 3), dtype=np.float32)
    return mins, maxs, centers, axes, half_sizes, corners, sizes, point_counts


def _save_obbs_json(path: Path, obbs: Sequence, *, table_z: float, workspace_bounds: Sequence[float]) -> None:
    payload = {
        "coordinate_frame": "world",
        "table_z": float(table_z),
        "workspace_bounds": [float(value) for value in workspace_bounds],
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


def _save_safe_space_npz(
    path: Path,
    *,
    obbs: Sequence,
    workspace_bounds: Sequence[float],
    table_z: float,
    voxel_size: float,
    min_obstacle_height: float,
    max_obstacle_height: float,
    cluster_radius: float,
    min_cluster_points: int,
) -> Path:
    bounds = np.asarray(workspace_bounds, dtype=np.float32)
    box_mins, box_maxs, centers, axes, half_sizes, corners, sizes, point_counts = _obb_arrays(obbs)
    obstacle_indices, occupied_grid, safe_grid, snapped_bounds = boxes_to_occupied_grid(
        box_centers=centers,
        box_axes=axes,
        box_half_sizes=half_sizes,
        box_corners=corners,
        bounds=bounds,
        voxel_size=float(voxel_size),
    )
    obstacle_centers = voxel_centers(obstacle_indices, snapped_bounds, float(voxel_size))
    safe_indices = np.column_stack(np.nonzero(safe_grid)).astype(np.int64)
    display_bounds = table_aligned_display_bounds(snapped_bounds, float(table_z), snapped_bounds[:4])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coordinate_frame=np.asarray("world"),
        workspace_bounds=snapped_bounds.astype(np.float32),
        display_workspace_bounds=display_bounds.astype(np.float32),
        obstacle_mode=np.asarray("tabletop_boxes"),
        table_z=np.asarray(float(table_z), dtype=np.float32),
        table_obstacle_min_height=np.asarray(float(min_obstacle_height), dtype=np.float32),
        table_obstacle_max_height=np.asarray(float(max_obstacle_height), dtype=np.float32),
        component_voxel_size=np.asarray(float(cluster_radius), dtype=np.float32),
        min_component_points=np.asarray(int(min_cluster_points), dtype=np.int64),
        voxel_size=np.asarray(float(voxel_size), dtype=np.float32),
        grid_shape=np.asarray(occupied_grid.shape, dtype=np.int64),
        obstacle_indices=obstacle_indices,
        obstacle_centers=obstacle_centers.astype(np.float32),
        obstacle_box_mins=box_mins.astype(np.float32),
        obstacle_box_maxs=box_maxs.astype(np.float32),
        obstacle_box_centers=centers.astype(np.float32),
        obstacle_box_axes=axes.astype(np.float32),
        obstacle_box_half_sizes=half_sizes.astype(np.float32),
        obstacle_box_corners=corners.astype(np.float32),
        obstacle_box_sizes=sizes.astype(np.float32),
        obstacle_box_point_counts=point_counts.astype(np.int64),
        safe_indices=safe_indices,
        occupied_grid=occupied_grid,
        safe_grid=safe_grid,
    )
    return path


def generate_tabletop_scene(
    args: argparse.Namespace,
    *,
    source_factory=RealSenseD435iSource,
) -> dict[str, Path | int]:
    calibrations = load_camera_calibrations(args.camera_calibration)
    missing = [name for name in ("front", "side") if name not in calibrations]
    if missing:
        raise KeyError(f"Missing camera calibrations for {missing}")

    frames = _capture_rgbd_frames(
        front_serial=args.front_serial,
        side_serial=args.side_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        wait_timeout_ms=args.wait_timeout_ms,
        read_retries=args.read_retries,
        warmup_frames=args.warmup_frames,
        source_factory=source_factory,
    )
    robot_points = _robot_points_for_filter(args)
    cloud = fuse_rgbd_frames(
        frames,
        calibrations,
        robot_link_points=robot_points,
        stride=args.pointcloud_stride,
        max_depth=args.max_depth,
        robot_filter_radius=args.robot_filter_radius,
        workspace_bounds=args.workspace_bounds,
    )
    tabletop_points, tabletop_colors = crop_workspace(cloud.environment_points, cloud.environment_colors, args.workspace_bounds)
    obstacle_points, obstacle_colors = select_tabletop_obstacle_points(
        tabletop_points,
        tabletop_colors,
        table_z=args.table_z,
        min_height_above_table=args.min_obstacle_height,
        max_height_above_table=args.max_obstacle_height,
    )
    obbs = build_tabletop_obbs(
        obstacle_points,
        cluster_radius=args.cluster_radius,
        min_cluster_points=args.min_cluster_points,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = str(args.name)
    outputs: dict[str, Path | int] = {}
    outputs.update(
        {f"scene_{key}": value for key, value in _save_pointcloud_bundle(
            output_dir,
            stem=f"{name}_scene_pointcloud",
            points=cloud.scene_points,
            colors=cloud.scene_colors,
            title=f"{name} fused scene point cloud",
            viewer_max_points=args.viewer_max_points,
            extra={"workspace_bounds": np.asarray(args.workspace_bounds, dtype=np.float32)},
        ).items()}
    )
    outputs.update(
        {f"tabletop_{key}": value for key, value in _save_pointcloud_bundle(
            output_dir,
            stem=f"{name}_tabletop_pointcloud",
            points=tabletop_points,
            colors=tabletop_colors,
            title=f"{name} fused tabletop point cloud",
            viewer_max_points=args.viewer_max_points,
            extra={"workspace_bounds": np.asarray(args.workspace_bounds, dtype=np.float32)},
        ).items()}
    )
    outputs.update(
        {f"obstacle_{key}": value for key, value in _save_pointcloud_bundle(
            output_dir,
            stem=f"{name}_tabletop_obstacle_points",
            points=obstacle_points,
            colors=obstacle_colors,
            title=f"{name} tabletop obstacle points",
            viewer_max_points=args.viewer_max_points,
            extra={
                "table_z": np.asarray(args.table_z, dtype=np.float32),
                "min_obstacle_height": np.asarray(args.min_obstacle_height, dtype=np.float32),
                "max_obstacle_height": np.asarray(args.max_obstacle_height, dtype=np.float32),
            },
        ).items()}
    )

    obbs_json = output_dir / f"{name}_tabletop_obbs.json"
    obbs_npz = output_dir / f"{name}_tabletop_obbs.npz"
    safe_space_npz = output_dir / f"{name}_safe_space.npz"
    _save_obbs_json(obbs_json, obbs, table_z=args.table_z, workspace_bounds=args.workspace_bounds)
    box_mins, box_maxs, centers, axes, half_sizes, corners, sizes, point_counts = _obb_arrays(obbs)
    np.savez_compressed(
        obbs_npz,
        coordinate_frame=np.asarray("world"),
        table_z=np.asarray(args.table_z, dtype=np.float32),
        workspace_bounds=np.asarray(args.workspace_bounds, dtype=np.float32),
        obstacle_box_mins=box_mins,
        obstacle_box_maxs=box_maxs,
        obstacle_box_centers=centers,
        obstacle_box_axes=axes,
        obstacle_box_half_sizes=half_sizes,
        obstacle_box_corners=corners,
        obstacle_box_sizes=sizes,
        obstacle_box_point_counts=point_counts,
    )
    save_interactive_pointcloud_html(
        output_dir / f"{name}_tabletop_obbs_viewer.html",
        tabletop_points,
        tabletop_colors,
        title=f"{name} tabletop point cloud with OBBs",
        max_points=args.viewer_max_points,
        obbs=obbs,
    )
    _save_safe_space_npz(
        safe_space_npz,
        obbs=obbs,
        workspace_bounds=args.workspace_bounds,
        table_z=args.table_z,
        voxel_size=args.voxel_size,
        min_obstacle_height=args.min_obstacle_height,
        max_obstacle_height=args.max_obstacle_height,
        cluster_radius=args.cluster_radius,
        min_cluster_points=args.min_cluster_points,
    )

    for frame in frames:
        _save_rgb(output_dir / f"{name}_{frame.camera_name}_rgb.png", frame.rgb)
        _save_rgb(
            output_dir / f"{name}_{frame.camera_name}_depth_vis.png",
            depth_to_vis(frame.depth_m, vis_max=args.depth_vis_max, with_colorbar=True),
        )

    outputs["obbs_json"] = obbs_json
    outputs["obbs_npz"] = obbs_npz
    outputs["obbs_html"] = output_dir / f"{name}_tabletop_obbs_viewer.html"
    outputs["safe_space_npz"] = safe_space_npz
    outputs["scene_point_count"] = int(cloud.scene_points.shape[0])
    outputs["tabletop_point_count"] = int(tabletop_points.shape[0])
    outputs["obstacle_point_count"] = int(obstacle_points.shape[0])
    outputs["obb_count"] = int(len(obbs))
    return outputs


def main() -> None:
    args = parse_args()
    outputs = generate_tabletop_scene(args)
    print(f"[done] fused scene points: {outputs['scene_point_count']}")
    print(f"[done] tabletop points: {outputs['tabletop_point_count']}")
    print(f"[done] obstacle points: {outputs['obstacle_point_count']}")
    print(f"[done] OBB count: {outputs['obb_count']}")
    for key, value in outputs.items():
        if isinstance(value, Path):
            print(f"[done] {key}: {value}")


if __name__ == "__main__":
    main()
