#!/usr/bin/env python3
"""Remove observed UR7e body points using RTDE qpos and official collision meshes.

The script only creates an RTDE receive connection. It never creates a motion
or control interface and sends no command to the robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.real_robot_adapter import RGBDFrame, depth_to_world_points, load_camera_calibrations, robot_depth_keep_mask, voxel_downsample_points
from real_scripts.reconstruct_realsense_pointcloud import depth_to_vis, render_topdown_camera_points, save_interactive_pointcloud_html, save_ply_ascii
from real_scripts.ur7e_collision_mesh import (
    collision_volume_keep_mask,
    flange_transform,
    occupied_collision_voxels,
    render_collision_depth,
    render_surface_points_depth,
    sample_mesh_surface_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "outputs" / "ur7e_d435i_lingbot_fused_current")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "ur7e_d435i_urdf_mesh_filtered")
    parser.add_argument("--calibration", type=Path, default=REPO_ROOT / "real_scripts" / "ur7e_d435i_camera_calibration.json")
    parser.add_argument("--robot-ip", default="169.254.175.10")
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.008)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--dilation-pixels", type=int, default=1)
    parser.add_argument("--volume-voxel-pitch-m", type=float, default=0.006, help="Filled collision-volume voxel edge length.")
    parser.add_argument("--volume-exterior-margin-m", type=float, default=0.015, help="Conservative exterior expansion of the filled collision volume.")
    parser.add_argument(
        "--pika-mount-transform-json",
        type=Path,
        default=None,
        help="Optional JSON containing a measured 4x4 flange_to_pika_step_frame transform. Enables PiKA full-mesh masking.",
    )
    parser.add_argument(
        "--pika-full-collision-mesh",
        type=Path,
        default=REPO_ROOT / "assets" / "robot_models" / "pika_gripper" / "collision" / "pika_gripper_full_collision.stl",
        help="PiKA STEP collision mesh expressed in millimetres.",
    )
    return parser.parse_args()


def _load_transform(path: Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    value = payload.get("flange_to_pika_step_frame", payload)
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("PiKA mount transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError("PiKA mount transform has invalid homogeneous final row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("PiKA mount transform rotation must be orthonormal with determinant +1")
    return transform


def main() -> None:
    args = parse_args()
    from rtde_receive import RTDEReceiveInterface

    receiver = RTDEReceiveInterface(str(args.robot_ip))
    try:
        qpos = np.asarray(receiver.getActualQ(), dtype=np.float32)
    finally:
        receiver.disconnect()
    if qpos.shape != (6,):
        raise RuntimeError(f"RTDE returned invalid qpos {qpos}")

    pika_mesh = None
    pika_to_base = None
    if args.pika_mount_transform_json is not None:
        import trimesh

        mount = _load_transform(args.pika_mount_transform_json)
        pika_mesh = trimesh.load_mesh(args.pika_full_collision_mesh, process=False)
        pika_mesh.vertices = np.asarray(pika_mesh.vertices, dtype=np.float64) * 0.001  # PiKA STEP/STL units are millimetres.
        pika_to_base = flange_transform(qpos) @ mount

    extra_meshes = () if pika_mesh is None or pika_to_base is None else ((pika_mesh, pika_to_base),)
    occupied_volume, volume_pitch = occupied_collision_voxels(
        qpos,
        voxel_pitch_m=args.volume_voxel_pitch_m,
        exterior_margin_m=args.volume_exterior_margin_m,
        extra_meshes=extra_meshes,
    )

    calibrations = load_camera_calibrations(args.calibration)
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_sets: list[tuple[np.ndarray, np.ndarray]] = []
    environment_sets: list[tuple[np.ndarray, np.ndarray]] = []
    camera_stats: dict[str, object] = {}
    for name in ("front", "side"):
        rgb = np.asarray(Image.open(input_dir / f"{name}_rgb.png").convert("RGB"), dtype=np.uint8)
        depth = np.load(input_dir / f"{name}_lingbot_depth_m.npy").astype(np.float32)
        rendered = render_collision_depth(qpos, calibrations[name].camera_to_world, calibrations[name].intrinsics, width=rgb.shape[1], height=rgb.shape[0])
        pika_rendered = None
        if pika_mesh is not None and pika_to_base is not None:
            pika_points = sample_mesh_surface_points(pika_mesh, pika_to_base, samples_per_face=4)
            pika_rendered = render_surface_points_depth(
                pika_points, calibrations[name].camera_to_world, calibrations[name].intrinsics,
                width=rgb.shape[1], height=rgb.shape[0], splat_radius_pixels=3,
            )
            rendered = np.where(
                (rendered > 0.0) & (pika_rendered > 0.0), np.minimum(rendered, pika_rendered), np.maximum(rendered, pika_rendered)
            )
        keep = robot_depth_keep_mask(depth, rendered, absolute_tolerance_m=args.absolute_tolerance_m, relative_tolerance=args.relative_tolerance, dilation_pixels=args.dilation_pixels)
        frame = RGBDFrame(name, rgb, depth)
        all_points, all_colors = depth_to_world_points(frame, calibrations[name], stride=2, max_depth=2.5)
        depth_robot_points, depth_robot_colors = depth_to_world_points(frame, calibrations[name], stride=2, max_depth=2.5, keep_mask=~keep)
        depth_environment_points, depth_environment_colors = depth_to_world_points(frame, calibrations[name], stride=2, max_depth=2.5, keep_mask=keep)
        volume_keep_all = collision_volume_keep_mask(all_points, occupied_volume, voxel_pitch_m=volume_pitch)
        volume_keep_environment = collision_volume_keep_mask(depth_environment_points, occupied_volume, voxel_pitch_m=volume_pitch)
        volume_robot_points, volume_robot_colors = all_points[~volume_keep_all], all_colors[~volume_keep_all]
        robot_sets.append((np.concatenate((depth_robot_points, volume_robot_points)), np.concatenate((depth_robot_colors, volume_robot_colors))))
        environment_sets.append((depth_environment_points[volume_keep_environment], depth_environment_colors[volume_keep_environment]))
        Image.fromarray(depth_to_vis(rendered, vis_max=2.0, with_colorbar=True)).save(output_dir / f"{name}_urdf_rendered_depth.png")
        if pika_rendered is not None:
            Image.fromarray(depth_to_vis(pika_rendered, vis_max=2.0, with_colorbar=True)).save(output_dir / f"{name}_pika_rendered_depth.png")
        overlay = rgb.copy()
        overlay[~keep] = (0.35 * overlay[~keep] + 0.65 * np.asarray([255, 0, 255])).astype(np.uint8)
        Image.fromarray(overlay).save(output_dir / f"{name}_urdf_removed_overlay.png")
        camera_stats[name] = {
            "rendered_robot_pixels": int((rendered > 0).sum()),
            "depth_matched_robot_pixels": int((~keep).sum()),
            "pika_rendered_pixels": 0 if pika_rendered is None else int((pika_rendered > 0).sum()),
            "volume_contained_points": int((~volume_keep_all).sum()),
            "volume_additional_environment_points_removed": int((~volume_keep_environment).sum()),
        }

    def fuse(sets: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        return voxel_downsample_points(np.concatenate([item[0] for item in sets]), np.concatenate([item[1] for item in sets]), voxel_size=0.005)

    robot_points, robot_colors = fuse(robot_sets)
    environment_points, environment_colors = fuse(environment_sets)
    for label, points, colors in (("ur7e_urdf_observed", robot_points, robot_colors), ("environment_without_ur7e", environment_points, environment_colors)):
        np.savez_compressed(output_dir / f"{label}.npz", points=points, colors=colors, coordinate_frame=np.asarray("ur_base"))
        save_ply_ascii(output_dir / f"{label}.ply", points, colors)
        save_interactive_pointcloud_html(output_dir / f"{label}_viewer.html", points, colors, title=f"{label} (UR base)", max_points=100000)
        Image.fromarray(render_topdown_camera_points(points, colors, image_size=800, margin_m=0.05)).save(output_dir / f"{label}_topdown.png")
    summary = {"coordinate_frame": "ur_base", "robot_ip": str(args.robot_ip), "qpos_rad": qpos.astype(float).tolist(), "method": "RTDE actual_q + official UR7e and optional PiKA collision mesh; rendered-depth agreement plus filled-and-expanded collision-volume removal", "volume_voxel_pitch_m": float(volume_pitch), "volume_exterior_margin_m": float(args.volume_exterior_margin_m), "occupied_volume_voxel_count": int(len(occupied_volume)), "pika_mount_transform_json": None if args.pika_mount_transform_json is None else str(args.pika_mount_transform_json), "excludes": None if pika_mesh is not None else "physical gripper/tool, which is not in the official UR7e URDF", "camera_stats": camera_stats, "ur7e_observed_point_count": int(len(robot_points)), "environment_point_count": int(len(environment_points))}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
