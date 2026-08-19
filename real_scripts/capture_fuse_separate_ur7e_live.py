#!/usr/bin/env python3
"""Capture calibrated D435i view(s), refine depth, optionally fuse, and remove UR7e points.

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
from real_scripts.pika_gripper_live_reader import PikaOpeningReader
from real_scripts.real_robot_adapter import RGBDFrame, depth_to_world_points, load_camera_calibration_session, robot_depth_keep_mask, voxel_downsample_points
from real_scripts.reconstruct_realsense_pointcloud import depth_to_vis, render_topdown_camera_points, save_interactive_pointcloud_html, save_ply_ascii
from real_scripts.ur7e_collision_mesh import (
    collision_volume_keep_mask,
    flange_transform,
    occupied_collision_voxels,
    render_collision_depth,
    render_surface_points_depth,
    sample_mesh_surface_points,
)
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "ur7e_d435i_live_lingbot_urdf")
    parser.add_argument("--calibration", type=Path, default=REPO_ROOT / "real_scripts" / "ur7e_d435i_camera_calibration.json")
    parser.add_argument("--robot-ip", default="169.254.175.10")
    parser.add_argument("--front-serial", default=os.environ.get("REAL_SENSE_FRONT_SERIAL", "405622074939"))
    parser.add_argument("--side-serial", default=os.environ.get("REAL_SENSE_SIDE_SERIAL", "348522070576"))
    parser.add_argument(
        "--camera-serial",
        action="append",
        default=[],
        metavar="NAME=SERIAL",
        help="Serial override for a camera key in the calibration file. Not needed for integrated-calibration files.",
    )
    parser.add_argument("--width", type=int, default=None, help="Override calibrated stream width (legacy fallback: 1280).")
    parser.add_argument("--height", type=int, default=None, help="Override calibrated stream height (legacy fallback: 720).")
    parser.add_argument("--fps", type=int, default=None, help="Override calibrated stream FPS (legacy fallback: 30).")
    parser.add_argument("--warmup-frames", type=int, default=45)
    parser.add_argument("--lingbot-device", default=None, help="Torch device for LingBot; defaults to CUDA when available.")
    parser.add_argument("--samples-per-face", type=int, default=16)
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.012)
    parser.add_argument("--relative-tolerance", type=float, default=0.015)
    parser.add_argument("--dilation-pixels", type=int, default=2)
    parser.add_argument("--voxel-size-m", type=float, default=0.005)
    parser.add_argument("--volume-voxel-pitch-m", type=float, default=0.006, help="Filled UR collision-volume voxel edge length.")
    parser.add_argument("--volume-exterior-margin-m", type=float, default=0.015, help="Extra exterior mask margin for calibration/depth error.")
    parser.add_argument(
        "--pika-mount-transform-json",
        type=Path,
        default="outputs/calibration/pika_mount_from_tcp_provisional.json",
        help="Measured 4x4 flange_to_pika_step_frame JSON. Enables conservative PiKA full-mesh masking.",
    )
    parser.add_argument(
        "--pika-full-collision-mesh",
        type=Path,
        default=REPO_ROOT / "assets" / "robot_models" / "pika_gripper" / "collision" / "pika_gripper_full_collision.stl",
        help="PiKA full collision mesh in STEP millimetres.",
    )
    parser.add_argument(
        "--pika-gripper-port",
        default=os.environ.get("PIKA_GRIPPER_PORT", "/dev/ttyUSB1"),
        help="PiKA USB serial port used only to read the captured jaw opening (default: PIKA_GRIPPER_PORT or /dev/ttyUSB1).",
    )
    parser.add_argument(
        "--pika-mesh-reference-opening-mm",
        type=float,
        default=0.0,
        help="Finger opening represented by the exported PiKA STEP meshes (default: 0, closed).",
    )
    parser.add_argument(
        "--pika-max-opening-mm",
        type=float,
        default=95.0,
        help="Reject PiKA opening readbacks outside [0, this value] mm (default: 95).",
    )
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


def _camera_serial_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        name, separator, serial = str(value).partition("=")
        if not separator or not name or not serial:
            raise ValueError(f"--camera-serial must have NAME=SERIAL form, got {value!r}")
        if name in overrides:
            raise ValueError(f"Duplicate --camera-serial override for camera {name!r}")
        overrides[name] = serial
    return overrides


def _load_pika_mount_transform(path: Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    transform = np.asarray(payload.get("flange_to_pika_step_frame", payload), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("PiKA mount transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError("PiKA mount transform has an invalid final homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("PiKA mount transform rotation must be orthonormal with determinant +1")
    return transform


def _load_pika_articulated_meshes(full_mesh_path: Path):
    """Load the static PiKA body and its two movable fingers in metres."""
    import trimesh

    collision_dir = Path(full_mesh_path).parent

    def load(name: str):
        path = collision_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"PiKA collision mesh does not exist: {path}")
        mesh = trimesh.load_mesh(path, process=False)
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
        return mesh

    return load("pika_gripper_body_collision.stl"), load("finger_a_candidate.stl"), load("finger_b_candidate.stl")


def _pika_meshes_at_opening(parts, *, opening_mm: float, reference_opening_mm: float):
    """Move the negative/positive PiKA-X fingers symmetrically about the body."""
    body, finger_a, finger_b = parts
    offset_m = 0.5 * (float(opening_mm) - float(reference_opening_mm)) / 1000.0
    moved_a = finger_a.copy()
    moved_b = finger_b.copy()
    moved_a.apply_translation((-offset_m, 0.0, 0.0))
    moved_b.apply_translation((offset_m, 0.0, 0.0))
    return body, moved_a, moved_b


def _camera_configs_for_calibration(session, args: argparse.Namespace) -> tuple[D435iCameraConfig, ...]:
    """Map calibration camera keys to physical RealSense serials.

    Integrated calibration files include each serial.  Legacy front/side JSON
    files retain their former command-line serial flags for compatibility.
    """
    overrides = _camera_serial_overrides(args.camera_serial)
    legacy_serials = {"front": str(args.front_serial), "side": str(args.side_serial)}
    configs = []
    for name in session.camera_names:
        serial = overrides.get(name) or session.camera_serials[name]
        if serial is None and name.isdigit():
            serial = name
        if serial is None:
            serial = legacy_serials.get(name) or os.environ.get(f"REAL_SENSE_{name.upper()}_SERIAL")
        if not serial:
            raise ValueError(
                f"No physical serial is recorded for calibrated camera {name!r}. "
                f"Pass --camera-serial {name}=SERIAL."
            )
        configs.append(D435iCameraConfig(name=name, serial=str(serial)))
    unknown = sorted(set(overrides) - set(session.camera_names))
    if unknown:
        raise ValueError(f"--camera-serial specifies cameras absent from the calibration file: {unknown}")
    return tuple(configs)


def _stream_config_for_calibration(session, args: argparse.Namespace) -> tuple[int, int, int]:
    """Use the stream profile that produced the intrinsics unless overridden."""
    recorded = {profile for profile in session.camera_streams.values() if profile is not None}
    if len(recorded) > 1:
        raise ValueError("All calibrated cameras must use one common width, height, and FPS for this generator")
    profile = next(iter(recorded), (1280, 720, 30))
    width = profile[0] if args.width is None else int(args.width)
    height = profile[1] if args.height is None else int(args.height)
    fps = profile[2] if args.fps is None else int(args.fps)
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("--width, --height, and --fps must be positive")
    return width, height, fps


def _combine_clouds(
    sets: list[tuple[np.ndarray, np.ndarray]], *, fusion_enabled: bool, voxel_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve a single camera cloud; combine and voxel-fuse only multi-view clouds."""
    nonempty = [(points, colors) for points, colors in sets if len(points)]
    if not nonempty:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    if not fusion_enabled:
        if len(nonempty) != 1:
            raise RuntimeError("Single-camera calibration unexpectedly produced more than one point-cloud input")
        return nonempty[0]
    return _fuse(nonempty, voxel_size_m)


def _save_cloud_bundle(output_dir: Path, stem: str, points: np.ndarray, colors: np.ndarray) -> None:
    np.savez_compressed(output_dir / f"{stem}.npz", points=points, colors=colors, coordinate_frame=np.asarray("ur_base"))
    save_ply_ascii(output_dir / f"{stem}.ply", points, colors)
    save_interactive_pointcloud_html(output_dir / f"{stem}_viewer.html", points, colors, title=f"{stem} (UR base)", max_points=100000)
    Image.fromarray(render_topdown_camera_points(points, colors, image_size=800, margin_m=0.05)).save(output_dir / f"{stem}_topdown.png")


def main() -> None:
    args = parse_args()
    from rtde_receive import RTDEReceiveInterface

    if not 0.0 <= args.pika_mesh_reference_opening_mm <= args.pika_max_opening_mm:
        raise ValueError("--pika-mesh-reference-opening-mm must be within the configured PiKA opening range")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_session = load_camera_calibration_session(args.calibration)
    calibrations = calibration_session.calibrations
    camera_names = calibration_session.camera_names
    fusion_enabled = calibration_session.fusion_enabled
    width, height, fps = _stream_config_for_calibration(calibration_session, args)
    receiver = RTDEReceiveInterface(str(args.robot_ip))
    source = RealSenseD435iSource(
        cameras=_camera_configs_for_calibration(calibration_session, args),
        width=width,
        height=height,
        fps=fps,
    )
    opening_reader: PikaOpeningReader | None = None
    pika_opening_before_mm: float | None = None
    pika_opening_after_mm: float | None = None
    pika_opening_status = "PiKA opening unavailable; using mesh reference"
    if args.pika_mount_transform_json is not None and args.pika_gripper_port:
        try:
            opening_reader = PikaOpeningReader(args.pika_gripper_port, max_opening_mm=args.pika_max_opening_mm)
            opening_reader.connect()
            pika_opening_status = f"PiKA opening reader {args.pika_gripper_port}"
            print(f"[capture] {pika_opening_status} connected (read-only)")
        except Exception as exc:
            print(f"[WARN] cannot read PiKA opening on {args.pika_gripper_port}: {exc}")
            opening_reader = None
    try:
        source.start()
        for _ in range(max(1, int(args.warmup_frames))):
            source.read()
        q_before = _read_q(receiver)
        if opening_reader is not None:
            pika_opening_before_mm = opening_reader.opening_mm()
        captured = source.read()
        q_after = _read_q(receiver)
        if opening_reader is not None:
            pika_opening_after_mm = opening_reader.opening_mm()
    finally:
        source.stop()
        receiver.disconnect()
        if opening_reader is not None:
            opening_reader.close()
    qpos = ((q_before.astype(np.float64) + q_after.astype(np.float64)) / 2.0).astype(np.float32)
    q_delta = float(np.max(np.abs(q_after - q_before)))
    pika_opening_mm = float(args.pika_mesh_reference_opening_mm)
    if pika_opening_before_mm is not None and pika_opening_after_mm is not None:
        pika_opening_mm = 0.5 * (pika_opening_before_mm + pika_opening_after_mm)
        pika_opening_status = f"PiKA opening {pika_opening_mm:.1f} mm"
    frames = [
        RGBDFrame(name, captured[name].rgb, captured[name].depth_m)
        for name in camera_names
    ]
    raw_frames_by_name = {frame.camera_name: frame for frame in frames}
    for frame in frames:
        Image.fromarray(frame.rgb).save(output_dir / f"{frame.camera_name}_rgb.png")
        np.save(output_dir / f"{frame.camera_name}_raw_depth_m.npy", frame.depth_m)
        Image.fromarray(depth_to_vis(frame.depth_m, vis_max=2.5, with_colorbar=True)).save(output_dir / f"{frame.camera_name}_raw_depth.png")

    pika_meshes = ()
    pika_to_base = None
    if args.pika_mount_transform_json is not None:
        pika_meshes = _pika_meshes_at_opening(
            _load_pika_articulated_meshes(args.pika_full_collision_mesh),
            opening_mm=pika_opening_mm,
            reference_opening_mm=args.pika_mesh_reference_opening_mm,
        )
        pika_to_base = flange_transform(qpos) @ _load_pika_mount_transform(args.pika_mount_transform_json)
    else:
        print("[WARN] PiKA gripper is not masked: pass --pika-mount-transform-json with a measured flange-to-PiKA transform.")

    refiner = LingBotDepthRefiner(
        camera_names=camera_names,
        device=args.lingbot_device,
        use_fp16=(args.lingbot_device is None or str(args.lingbot_device).lower().startswith("cuda")),
    )
    refined = refiner.refine(frames, calibrations)
    extra_meshes = () if pika_to_base is None else tuple((mesh, pika_to_base) for mesh in pika_meshes)
    pika_surface_points = (
        None
        if pika_to_base is None
        else np.concatenate(
            [sample_mesh_surface_points(mesh, pika_to_base, samples_per_face=args.samples_per_face) for mesh in pika_meshes],
            axis=0,
        )
    )
    occupied_volume, volume_pitch = occupied_collision_voxels(
        qpos,
        voxel_pitch_m=args.volume_voxel_pitch_m,
        exterior_margin_m=args.volume_exterior_margin_m,
        extra_meshes=extra_meshes,
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
        pika_rendered = None
        if pika_surface_points is not None:
            pika_rendered = render_surface_points_depth(
                pika_surface_points,
                calibrations[name].camera_to_world,
                calibrations[name].intrinsics,
                width=frame.rgb.shape[1],
                height=frame.rgb.shape[0],
                splat_radius_pixels=3,
            )
            rendered = np.where(
                (rendered > 0.0) & (pika_rendered > 0.0),
                np.minimum(rendered, pika_rendered),
                np.maximum(rendered, pika_rendered),
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
        if pika_rendered is not None:
            Image.fromarray(depth_to_vis(pika_rendered, vis_max=2.5, with_colorbar=True)).save(output_dir / f"{name}_pika_rendered_depth.png")
        overlay = frame.rgb.copy()
        overlay[~keep] = (0.35 * overlay[~keep] + 0.65 * np.asarray([255, 0, 255])).astype(np.uint8)
        Image.fromarray(overlay).save(output_dir / f"{name}_urdf_removed_overlay.png")
        camera_stats[name] = {
            "rendered_robot_pixels": int((rendered > 0).sum()),
            "depth_matched_robot_pixels": int((~keep).sum()),
            "raw_depth_valid_pixels": int(np.count_nonzero(raw_frames_by_name[name].depth_m > 0)),
            "lingbot_depth_valid_pixels": int(np.count_nonzero(frame.depth_m > 0)),
            "pika_rendered_pixels": 0 if pika_rendered is None else int(np.count_nonzero(pika_rendered > 0)),
            "volume_contained_points": int((~volume_keep_all).sum()),
            "volume_additional_environment_points_removed": int((~volume_keep_environment).sum()),
        }

    all_points, all_colors = _combine_clouds(fused_sets, fusion_enabled=fusion_enabled, voxel_size_m=args.voxel_size_m)
    robot_points, robot_colors = _combine_clouds(robot_sets, fusion_enabled=fusion_enabled, voxel_size_m=args.voxel_size_m)
    environment_points, environment_colors = _combine_clouds(environment_sets, fusion_enabled=fusion_enabled, voxel_size_m=args.voxel_size_m)
    _save_cloud_bundle(output_dir, "lingbot_fused_scene", all_points, all_colors)
    _save_cloud_bundle(output_dir, "ur7e_urdf_observed", robot_points, robot_colors)
    _save_cloud_bundle(output_dir, "environment_without_ur7e", environment_points, environment_colors)
    summary = {
        "coordinate_frame": "ur_base",
        "calibration_file": str(args.calibration),
        "camera_names": list(camera_names),
        "point_cloud_mode": "multi_camera_fusion" if fusion_enabled else "single_camera",
        "fusion_enabled": fusion_enabled,
        "stream_profile": {"width": width, "height": height, "fps": fps},
        "robot_ip": str(args.robot_ip),
        "qpos_rad": qpos.astype(float).tolist(),
        "max_q_delta_during_capture_rad": q_delta,
        "lingbot_inference_seconds": refiner.last_inference_seconds,
        "method": "synchronized RGB-D capture + RTDE actual_q + PiKA jaw-opening readback + LingBot-Depth + UR7e/PiKA articulated collision meshes with z-buffer depth consistency",
        "volume_voxel_pitch_m": float(volume_pitch),
        "volume_exterior_margin_m": float(args.volume_exterior_margin_m),
        "occupied_volume_voxel_count": int(len(occupied_volume)),
        "pika_mount_transform_json": None if args.pika_mount_transform_json is None else str(args.pika_mount_transform_json),
        "pika_opening_mm": None if pika_to_base is None else float(pika_opening_mm),
        "pika_opening_status": pika_opening_status,
        "pika_mesh_reference_opening_mm": float(args.pika_mesh_reference_opening_mm),
        "pika_opening_before_mm": pika_opening_before_mm,
        "pika_opening_after_mm": pika_opening_after_mm,
        "excludes": None if pika_to_base is not None else "PiKA gripper/tool: supply --pika-mount-transform-json to enable full-mesh masking.",
        "camera_stats": camera_stats,
        "fused_point_count": int(len(all_points)),
        "ur7e_observed_point_count": int(len(robot_points)),
        "environment_point_count": int(len(environment_points)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
