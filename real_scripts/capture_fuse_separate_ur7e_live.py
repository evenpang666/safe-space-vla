#!/usr/bin/env python3
"""Capture both fixed D435i views, refine depth, fuse, and remove UR7e points.

This utility is receive-only for the robot: it opens ``RTDEReceiveInterface``
to read ``actual_q`` before and after the RGB-D capture and never creates a
control interface or sends robot commands.  The robot should remain still for
the capture and the ensuing (CPU) LingBot-Depth refinement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.lingbot_depth import LingBotDepthRefiner
from real_scripts.real_robot_adapter import RGBDFrame, depth_to_world_points, load_camera_calibrations, robot_depth_keep_mask, voxel_downsample_points
from real_scripts.reconstruct_realsense_pointcloud import depth_to_vis, render_topdown_camera_points, save_interactive_pointcloud_html, save_ply_ascii
from real_scripts.ur7e_collision_mesh import collision_volume_keep_mask, occupied_collision_voxels, render_collision_depth
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "ur7e_d435i_live_lingbot_urdf")
    parser.add_argument("--calibration", type=Path, default=REPO_ROOT / "real_scripts" / "ur7e_d435i_camera_calibration.json")
    parser.add_argument("--robot-ip", default="169.254.175.10")
    parser.add_argument("--front-serial", default=os.environ.get("REAL_SENSE_FRONT_SERIAL", "405622074939"))
    parser.add_argument("--side-serial", default=os.environ.get("REAL_SENSE_SIDE_SERIAL", "348522070576"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=45)
    parser.add_argument("--samples-per-face", type=int, default=16)
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.012)
    parser.add_argument("--relative-tolerance", type=float, default=0.015)
    parser.add_argument("--dilation-pixels", type=int, default=2)
    parser.add_argument("--voxel-size-m", type=float, default=0.005)
    parser.add_argument("--volume-voxel-pitch-m", type=float, default=0.006, help="Filled UR collision-volume voxel edge length.")
    parser.add_argument("--volume-exterior-margin-m", type=float, default=0.015, help="Extra exterior mask margin for calibration/depth error.")
    return parser.parse_args()


def _read_q(receiver) -> np.ndarray:
    qpos = np.asarray(receiver.getActualQ(), dtype=np.float32)
    if qpos.shape != (6,) or not np.isfinite(qpos).all():
        raise RuntimeError(f"RTDE returned invalid actual_q: {qpos}")
    return qpos


def _fuse(sets: list[tuple[np.ndarray, np.ndarray]], voxel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    nonempty = [(points, colors) for points, colors in sets if len(points)]
    if not nonempty:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return voxel_downsample_points(
        np.concatenate([points for points, _ in nonempty]),
        np.concatenate([colors for _, colors in nonempty]),
        voxel_size=float(voxel_size_m),
    )


def _save_cloud_bundle(output_dir: Path, stem: str, points: np.ndarray, colors: np.ndarray) -> None:
    np.savez_compressed(output_dir / f"{stem}.npz", points=points, colors=colors, coordinate_frame=np.asarray("ur_base"))
    save_ply_ascii(output_dir / f"{stem}.ply", points, colors)
    save_interactive_pointcloud_html(output_dir / f"{stem}_viewer.html", points, colors, title=f"{stem} (UR base)", max_points=100000)
    Image.fromarray(render_topdown_camera_points(points, colors, image_size=800, margin_m=0.05)).save(output_dir / f"{stem}_topdown.png")


def main() -> None:
    args = parse_args()
    from rtde_receive import RTDEReceiveInterface

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrations = load_camera_calibrations(args.calibration)
    receiver = RTDEReceiveInterface(str(args.robot_ip))
    source = RealSenseD435iSource(
        cameras=(D435iCameraConfig("front", str(args.front_serial)), D435iCameraConfig("side", str(args.side_serial))),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    try:
        source.start()
        for _ in range(max(1, int(args.warmup_frames))):
            source.read()
        q_before = _read_q(receiver)
        captured = source.read()
        q_after = _read_q(receiver)
    finally:
        source.stop()
        receiver.disconnect()
    qpos = ((q_before.astype(np.float64) + q_after.astype(np.float64)) / 2.0).astype(np.float32)
    q_delta = float(np.max(np.abs(q_after - q_before)))
    frames = [RGBDFrame(name, captured[name][0], captured[name][1]) for name in ("front", "side")]
    for frame in frames:
        Image.fromarray(frame.rgb).save(output_dir / f"{frame.camera_name}_rgb.png")
        np.save(output_dir / f"{frame.camera_name}_raw_depth_m.npy", frame.depth_m)
        Image.fromarray(depth_to_vis(frame.depth_m, vis_max=2.5, with_colorbar=True)).save(output_dir / f"{frame.camera_name}_raw_depth.png")

    # The local safety environment is CPU-only, so this deliberately chooses
    # the tested PyTorch-SDPA compatibility path (no FP16 / no xFormers).
    refiner = LingBotDepthRefiner(device="cpu", use_fp16=False)
    refined = refiner.refine(frames, calibrations)
    occupied_volume, volume_pitch = occupied_collision_voxels(
        qpos,
        voxel_pitch_m=args.volume_voxel_pitch_m,
        exterior_margin_m=args.volume_exterior_margin_m,
    )

    robot_sets: list[tuple[np.ndarray, np.ndarray]] = []
    environment_sets: list[tuple[np.ndarray, np.ndarray]] = []
    fused_sets: list[tuple[np.ndarray, np.ndarray]] = []
    camera_stats: dict[str, object] = {}
    for frame in refined:
        name = frame.camera_name
        np.save(output_dir / f"{name}_lingbot_depth_m.npy", frame.depth_m)
        Image.fromarray(depth_to_vis(frame.depth_m, vis_max=2.5, with_colorbar=True)).save(output_dir / f"{name}_lingbot_depth.png")
        rendered = render_collision_depth(
            qpos, calibrations[name].camera_to_world, calibrations[name].intrinsics,
            width=frame.rgb.shape[1], height=frame.rgb.shape[0],
            samples_per_face=args.samples_per_face, splat_radius_pixels=2,
        )
        keep = robot_depth_keep_mask(
            frame.depth_m, rendered, absolute_tolerance_m=args.absolute_tolerance_m,
            relative_tolerance=args.relative_tolerance, dilation_pixels=args.dilation_pixels,
        )
        rgbd = RGBDFrame(name, frame.rgb, frame.depth_m)
        all_points, all_colors = depth_to_world_points(rgbd, calibrations[name], stride=2, max_depth=2.5)
        depth_robot_points, depth_robot_colors = depth_to_world_points(rgbd, calibrations[name], stride=2, max_depth=2.5, keep_mask=~keep)
        depth_environment_points, depth_environment_colors = depth_to_world_points(rgbd, calibrations[name], stride=2, max_depth=2.5, keep_mask=keep)
        volume_keep_all = collision_volume_keep_mask(all_points, occupied_volume, voxel_pitch_m=volume_pitch)
        volume_keep_environment = collision_volume_keep_mask(depth_environment_points, occupied_volume, voxel_pitch_m=volume_pitch)
        robot_sets.append((np.concatenate((depth_robot_points, all_points[~volume_keep_all])), np.concatenate((depth_robot_colors, all_colors[~volume_keep_all]))))
        environment_sets.append((depth_environment_points[volume_keep_environment], depth_environment_colors[volume_keep_environment]))
        fused_sets.append((all_points, all_colors))
        Image.fromarray(depth_to_vis(rendered, vis_max=2.5, with_colorbar=True)).save(output_dir / f"{name}_urdf_rendered_depth.png")
        overlay = frame.rgb.copy()
        overlay[~keep] = (0.35 * overlay[~keep] + 0.65 * np.asarray([255, 0, 255])).astype(np.uint8)
        Image.fromarray(overlay).save(output_dir / f"{name}_urdf_removed_overlay.png")
        camera_stats[name] = {
            "rendered_robot_pixels": int((rendered > 0).sum()),
            "depth_matched_robot_pixels": int((~keep).sum()),
            "raw_depth_valid_pixels": int(np.count_nonzero(frames[("front", "side").index(name)].depth_m > 0)),
            "lingbot_depth_valid_pixels": int(np.count_nonzero(frame.depth_m > 0)),
            "volume_contained_points": int((~volume_keep_all).sum()),
            "volume_additional_environment_points_removed": int((~volume_keep_environment).sum()),
        }

    all_points, all_colors = _fuse(fused_sets, args.voxel_size_m)
    robot_points, robot_colors = _fuse(robot_sets, args.voxel_size_m)
    environment_points, environment_colors = _fuse(environment_sets, args.voxel_size_m)
    _save_cloud_bundle(output_dir, "lingbot_fused_scene", all_points, all_colors)
    _save_cloud_bundle(output_dir, "ur7e_urdf_observed", robot_points, robot_colors)
    _save_cloud_bundle(output_dir, "environment_without_ur7e", environment_points, environment_colors)
    summary = {
        "coordinate_frame": "ur_base",
        "robot_ip": str(args.robot_ip),
        "qpos_rad": qpos.astype(float).tolist(),
        "max_q_delta_during_capture_rad": q_delta,
        "lingbot_inference_seconds": refiner.last_inference_seconds,
        "method": "synchronized RGB-D capture + RTDE actual_q + LingBot-Depth + official UR7e collision mesh z-buffer depth consistency",
        "volume_voxel_pitch_m": float(volume_pitch),
        "volume_exterior_margin_m": float(args.volume_exterior_margin_m),
        "occupied_volume_voxel_count": int(len(occupied_volume)),
        "excludes": "PiKA gripper is excluded unless a measured flange-to-PiKA mesh transform is supplied to the offline filter.",
        "camera_stats": camera_stats,
        "fused_point_count": int(len(all_points)),
        "ur7e_observed_point_count": int(len(robot_points)),
        "environment_point_count": int(len(environment_points)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
