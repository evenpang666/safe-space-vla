#!/usr/bin/env python3
"""Convert a raw PI05 RGB-D UR episode into RGB + fixed robot-surface points.

The raw input is an ``UR7eSafetyEpisodeRecorder`` directory.  Depth is used
offline to reconstruct and count the observed robot/gripper cloud for every
frame, but is deliberately not copied to the output.  The training target is
instead a deterministic collision-surface sample transformed by the measured
joint state.  Thus every output point has a stable (link_id, point_id) across
time, which is required for a future point-flow loss.  Optionally, externally
computed 2-D tracks (for example, CoTracker tracks) are depth-lifted and
FK-gated into a second, real observed robot-surface trajectory target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.real_robot_adapter import (  # noqa: E402
    RGBDFrame,
    depth_to_world_points,
    load_camera_calibrations,
    robot_depth_keep_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True, help="Completed raw directory written by UR7eSafetyEpisodeRecorder.")
    parser.add_argument("--output", type=Path, required=True, help="Output compressed .npz without depth arrays.")
    parser.add_argument("--calibration", type=Path, default=None, help="Defaults to EPISODE_DIR/calibration.json.")
    parser.add_argument("--scene-camera-names", nargs="+", default=None, help="Episode camera names used for the depth reconstruction audit (for example: front side).")
    parser.add_argument(
        "--scene-camera-map",
        action="append",
        default=[],
        metavar="EPISODE_CAMERA=CALIBRATION_CAMERA",
        help="Map a raw episode camera name to a calibration key; required when the calibration uses serial-number keys.",
    )
    parser.add_argument("--points-per-link", type=int, default=128)
    parser.add_argument("--pika-mount-transform-json", type=Path, required=True)
    parser.add_argument(
        "--pika-full-collision-mesh",
        type=Path,
        default=REPO_ROOT / "assets" / "robot_models" / "pika_gripper" / "collision" / "pika_gripper_full_collision.stl",
    )
    parser.add_argument(
        "--pika-mesh-reference-opening-mm",
        type=float,
        default=0.0,
        help="Finger opening represented by the exported PiKA STEP meshes (default: 0, closed).",
    )
    parser.add_argument(
        "--on-missing-pika-opening",
        choices=("error", "reference"),
        default="error",
        help="How to handle legacy ticks without gripper_opening_mm.npy (default: error).",
    )
    parser.add_argument("--future-horizon", type=int, default=8, help="Future ticks used to create point-flow training targets.")
    parser.add_argument(
        "--robot-tracks",
        type=Path,
        default=None,
        help="Optional .npz of stable-ID 2-D robot tracks. Requires tracks_xy[T,M,2], visibility[T,M], and tick_ids[T].",
    )
    parser.add_argument(
        "--robot-tracks-camera",
        default=None,
        help="Episode camera used by --robot-tracks; defaults to the first --scene-camera-names camera.",
    )
    parser.add_argument("--track-min-confidence", type=float, default=0.5, help="Minimum optional track confidence in [0, 1].")
    parser.add_argument("--depth-stride", type=int, default=2)
    parser.add_argument("--max-depth-m", type=float, default=2.5)
    parser.add_argument("--absolute-depth-tolerance-m", type=float, default=0.012)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.015)
    parser.add_argument("--depth-dilation-pixels", type=int, default=2)
    parser.add_argument("--max-camera-robot-skew-ms", type=float, default=100.0)
    parser.add_argument("--on-timestamp-skew", choices=("error", "skip"), default="error")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _tick_dirs(episode_dir: Path) -> list[Path]:
    frame_dir = Path(episode_dir) / "frames"
    ticks = sorted(path for path in frame_dir.iterdir() if path.is_dir()) if frame_dir.is_dir() else []
    if not ticks:
        raise FileNotFoundError(f"No frame directories found under {frame_dir}")
    return ticks


def _rgb_camera_names(first_tick: Path) -> tuple[str, ...]:
    suffix = "_rgb.npy"
    names = tuple(sorted(path.name[: -len(suffix)] for path in first_tick.glob(f"*{suffix}")))
    if not names:
        raise ValueError(f"No RGB arrays found in {first_tick}")
    return names


def _camera_name_map(values: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        episode_name, separator, calibration_name = str(value).partition("=")
        if not separator or not episode_name or not calibration_name:
            raise ValueError(f"--scene-camera-map must have EPISODE_CAMERA=CALIBRATION_CAMERA form, got {value!r}")
        if episode_name in mapping:
            raise ValueError(f"Duplicate --scene-camera-map for episode camera {episode_name!r}")
        mapping[episode_name] = calibration_name
    return mapping


def _timestamp_ok(metadata: dict[str, Any], *, camera_names: Sequence[str], max_skew_ns: int) -> tuple[bool, int | None]:
    robot_timestamp = metadata.get("robot_state_timestamp_ns")
    if robot_timestamp is None:
        return True, None
    deltas = []
    for name in camera_names:
        frame = metadata.get("frames", {}).get(name, {})
        timestamp = frame.get("host_timestamp_ns")
        if timestamp is not None:
            deltas.append(abs(int(timestamp) - int(robot_timestamp)))
    worst = None if not deltas else max(deltas)
    return worst is None or worst <= max_skew_ns, worst


def _combined_robot_depth(
    *,
    qpos: np.ndarray,
    fixed_surface_points: np.ndarray,
    calibration,
    height: int,
    width: int,
) -> np.ndarray:
    """Render UR7e and PiKA samples into one camera-depth mask."""
    from real_scripts.ur7e_collision_mesh import render_collision_depth, render_surface_points_depth

    arm_depth = render_collision_depth(
        qpos,
        calibration.camera_to_world,
        calibration.intrinsics,
        width=width,
        height=height,
        samples_per_face=4,
        splat_radius_pixels=2,
    )
    gripper_depth = render_surface_points_depth(
        fixed_surface_points[-1],
        calibration.camera_to_world,
        calibration.intrinsics,
        width=width,
        height=height,
        splat_radius_pixels=2,
    )
    return np.where(
        (arm_depth > 0.0) & (gripper_depth > 0.0),
        np.minimum(arm_depth, gripper_depth),
        np.maximum(arm_depth, gripper_depth),
    )


def _observed_robot_point_count(
    frame: RGBDFrame,
    *,
    calibration,
    qpos: np.ndarray,
    fixed_surface_points: np.ndarray,
    stride: int,
    max_depth_m: float,
    absolute_tolerance_m: float,
    relative_tolerance: float,
    dilation_pixels: int,
) -> int:
    rendered = _combined_robot_depth(
        qpos=qpos,
        fixed_surface_points=fixed_surface_points,
        calibration=calibration,
        height=frame.rgb.shape[0],
        width=frame.rgb.shape[1],
    )
    # robot_depth_keep_mask is True for non-robot/environment pixels.
    environment_mask = robot_depth_keep_mask(
        frame.depth_m,
        rendered,
        absolute_tolerance_m=absolute_tolerance_m,
        relative_tolerance=relative_tolerance,
        dilation_pixels=dilation_pixels,
    )
    points, _colors = depth_to_world_points(
        frame,
        calibration,
        stride=stride,
        max_depth=max_depth_m,
        keep_mask=~environment_mask,
    )
    return int(len(points))


def _load_tick(
    tick_dir: Path,
    *,
    rgb_camera_names: Sequence[str],
    scene_camera_names: Sequence[str],
    calibration_camera_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, RGBDFrame], np.ndarray, np.ndarray, float | None, np.ndarray, dict[str, Any]]:
    metadata = _load_json(tick_dir / "metadata.json")
    rgb = {}
    for name in rgb_camera_names:
        path = tick_dir / f"{name}_rgb.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        rgb[name] = np.asarray(np.load(path, allow_pickle=False), dtype=np.uint8)
    qpos = np.asarray(np.load(tick_dir / "qpos.npy", allow_pickle=False), dtype=np.float32).reshape(-1)
    if qpos.size < 6 or not np.isfinite(qpos[:6]).all():
        raise ValueError(f"Invalid qpos in {tick_dir}: {qpos.shape}")
    gripper_path = tick_dir / "gripper_state.npy"
    gripper = np.asarray(np.load(gripper_path, allow_pickle=False), dtype=np.float32).reshape(-1) if gripper_path.is_file() else np.zeros((1,), dtype=np.float32)
    opening_path = tick_dir / "gripper_opening_mm.npy"
    opening_mm = None
    if opening_path.is_file():
        opening_values = np.asarray(np.load(opening_path, allow_pickle=False), dtype=np.float32).reshape(-1)
        if opening_values.size != 1 or not np.isfinite(opening_values[0]) or not 0.0 <= float(opening_values[0]) <= 95.0:
            raise ValueError(f"Invalid PiKA opening in {opening_path}: {opening_values}")
        opening_mm = float(opening_values[0])
    action_path = tick_dir / "commanded_action.npy"
    action = np.asarray(np.load(action_path, allow_pickle=False), dtype=np.float32).reshape(-1) if action_path.is_file() else np.zeros((7,), dtype=np.float32)
    if action_path.is_file() and action.shape != (7,):
        raise ValueError(f"Expected 7-D commanded action in {action_path}, got {action.shape}")
    scene_frames = {}
    for name, calibration_name in zip(scene_camera_names, calibration_camera_names, strict=True):
        depth_path = tick_dir / f"{name}_depth_raw_m.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing depth for selected scene camera {name!r}: {depth_path}")
        scene_frames[name] = RGBDFrame(
            calibration_name,
            rgb[name],
            np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32),
        )
    return rgb, scene_frames, qpos[:6], gripper, opening_mm, action, metadata


def _load_pika_articulated_meshes(full_mesh_path: Path):
    """Load the PiKA body and the two moving fingers in STEP metres."""
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


def _articulated_pika_surface_points(local_parts, *, opening_mm: float, reference_opening_mm: float) -> np.ndarray:
    """Return fixed-order PiKA-local points for the measured jaw opening."""
    body, finger_a, finger_b = local_parts
    offset_m = 0.5 * (float(opening_mm) - float(reference_opening_mm)) / 1000.0
    return np.concatenate(
        (
            body,
            finger_a + np.array((-offset_m, 0.0, 0.0)),
            finger_b + np.array((offset_m, 0.0, 0.0)),
        ),
        axis=0,
    )


def _resample_fixed_order(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministically retain a stable subset without shuffling point IDs."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) < 1:
        raise ValueError("PiKA surface has no points")
    indices = np.linspace(0, len(points) - 1, int(count), dtype=np.int64)
    return points[indices]


def _sample_indices(
    action_valid: np.ndarray, tick_ids: np.ndarray, *, frame_count: int, future_horizon: int
) -> np.ndarray:
    indices = []
    for start in range(frame_count - future_horizon):
        consecutive = bool(np.all(np.diff(tick_ids[start : start + future_horizon + 1]) == 1))
        if consecutive and bool(np.all(action_valid[start : start + future_horizon])):
            indices.append(start)
    return np.asarray(indices, dtype=np.int64)


def _load_robot_tracks(path: Path) -> dict[str, np.ndarray]:
    """Load a tracker-neutral stable-ID 2-D track interchange file.

    ``tracks_xy[t, seed_id]`` uses colour-image pixel coordinates ``(u, v)``.
    IDs are the column indices and must never be reordered after an occlusion.
    ``visibility`` and optional ``confidence`` come from the tracker.  Tick IDs
    make the file safe to use when timestamp-skewed raw frames are skipped.
    """
    with np.load(path, allow_pickle=False) as data:
        required = {"tracks_xy", "visibility", "tick_ids"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Robot track file {path} is missing arrays: {missing}")
        tracks_xy = np.asarray(data["tracks_xy"], dtype=np.float32)
        visibility = np.asarray(data["visibility"], dtype=bool)
        tick_ids = np.asarray(data["tick_ids"], dtype=np.int64).reshape(-1)
        confidence = np.asarray(data["confidence"], dtype=np.float32) if "confidence" in data.files else np.ones(visibility.shape, dtype=np.float32)
        seed_xy = np.asarray(data["seed_xy"], dtype=np.float32) if "seed_xy" in data.files else tracks_xy[0].copy()
    if tracks_xy.ndim != 3 or tracks_xy.shape[2] != 2:
        raise ValueError(f"tracks_xy must have shape [T, M, 2], got {tracks_xy.shape}")
    if visibility.shape != tracks_xy.shape[:2] or confidence.shape != tracks_xy.shape[:2]:
        raise ValueError("visibility and confidence must both have shape [T, M]")
    if tick_ids.shape != (tracks_xy.shape[0],) or len(np.unique(tick_ids)) != len(tick_ids):
        raise ValueError("tick_ids must have one unique entry for each tracker frame")
    if seed_xy.shape != (tracks_xy.shape[1], 2):
        raise ValueError(f"seed_xy must have shape [M, 2], got {seed_xy.shape}")
    if not np.isfinite(tracks_xy).all() or not np.isfinite(confidence).all():
        raise ValueError("tracks_xy and confidence must be finite")
    return {"tracks_xy": tracks_xy, "visibility": visibility, "confidence": confidence, "tick_ids": tick_ids, "seed_xy": seed_xy}


def _lift_fk_gated_robot_tracks(
    frame: RGBDFrame,
    *,
    calibration,
    qpos: np.ndarray,
    fixed_surface_points: np.ndarray,
    tracks_xy: np.ndarray,
    tracker_visibility: np.ndarray,
    tracker_confidence: np.ndarray,
    min_confidence: float,
    absolute_tolerance_m: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project tracks only where depth agrees with the rendered robot.

    The result remains indexed by the original track/seed ID.  Invalid,
    occluded, or FK-inconsistent rows are represented by a false mask, not by
    deleting or reordering points.
    """
    xy = np.asarray(tracks_xy, dtype=np.float32)
    visible = np.asarray(tracker_visibility, dtype=bool).copy()
    confidence = np.asarray(tracker_confidence, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or visible.shape != (len(xy),) or confidence.shape != (len(xy),):
        raise ValueError("Track frame must contain xy[M,2], visibility[M], and confidence[M]")
    height, width = frame.depth_m.shape
    u = np.rint(xy[:, 0]).astype(np.int64)
    v = np.rint(xy[:, 1]).astype(np.int64)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    safe_u = np.clip(u, 0, width - 1)
    safe_v = np.clip(v, 0, height - 1)
    measured_depth = frame.depth_m[safe_v, safe_u]
    rendered_depth = _combined_robot_depth(
        qpos=qpos,
        fixed_surface_points=fixed_surface_points,
        calibration=calibration,
        height=height,
        width=width,
    )[safe_v, safe_u]
    tolerance = float(absolute_tolerance_m) + float(relative_tolerance) * rendered_depth
    visible &= in_bounds
    visible &= np.isfinite(measured_depth) & (measured_depth > 0.0)
    visible &= rendered_depth > 0.0
    visible &= np.abs(measured_depth - rendered_depth) <= tolerance
    visible &= confidence >= float(min_confidence)

    points = np.zeros((len(xy), 3), dtype=np.float32)
    if np.any(visible):
        z = measured_depth[visible].astype(np.float64)
        fx, fy = float(calibration.intrinsics[0, 0]), float(calibration.intrinsics[1, 1])
        cx, cy = float(calibration.intrinsics[0, 2]), float(calibration.intrinsics[1, 2])
        camera_points = np.stack(((u[visible] - cx) * z / fx, (v[visible] - cy) * z / fy, z), axis=1)
        homogeneous = np.concatenate((camera_points, np.ones((len(camera_points), 1))), axis=1)
        points[visible] = (calibration.camera_to_world @ homogeneous.T).T[:, :3].astype(np.float32)
    return points, visible


def _preprocess_raw_episode(args: argparse.Namespace) -> dict[str, Any]:
    if args.points_per_link < 1 or args.future_horizon < 1 or args.depth_stride < 1 or args.max_depth_m <= 0.0:
        raise ValueError("points-per-link, future-horizon, depth-stride, and max-depth-m must be positive")
    if args.max_camera_robot_skew_ms < 0.0:
        raise ValueError("--max-camera-robot-skew-ms must be >= 0")
    if not 0.0 <= args.track_min_confidence <= 1.0:
        raise ValueError("--track-min-confidence must be in [0, 1]")
    episode_dir = Path(args.episode_dir)
    manifest = _load_json(episode_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Episode manifest is not complete: {manifest.get('status')!r}")
    calibration_path = Path(args.calibration) if args.calibration is not None else episode_dir / "calibration.json"
    calibrations = load_camera_calibrations(calibration_path)
    ticks = _tick_dirs(episode_dir)
    rgb_camera_names = _rgb_camera_names(ticks[0])
    camera_map = _camera_name_map(args.scene_camera_map)
    manifest_camera_map = manifest.get("scene_camera_map", {})
    if not isinstance(manifest_camera_map, dict):
        raise ValueError("manifest scene_camera_map must be an object when present")
    # A live teleoperation episode stores its friendly raw stream name (for
    # example ``front``) plus the serial-number key used by calibration.json.
    # Explicit CLI mappings deliberately take precedence for legacy episodes.
    for episode_name, calibration_name in manifest_camera_map.items():
        if episode_name not in camera_map and isinstance(episode_name, str) and isinstance(calibration_name, str):
            camera_map[episode_name] = calibration_name
    scene_camera_names = tuple(args.scene_camera_names) if args.scene_camera_names is not None else tuple(camera_map or (name for name in calibrations if name in rgb_camera_names))
    if not scene_camera_names:
        raise ValueError("No calibrated scene cameras are present in the episode; pass --scene-camera-names")
    calibration_camera_names = tuple(camera_map.get(name, name) for name in scene_camera_names)
    missing = [
        name
        for name, calibration_name in zip(scene_camera_names, calibration_camera_names, strict=True)
        if name not in rgb_camera_names or calibration_name not in calibrations
    ]
    if missing:
        raise KeyError(f"Selected scene cameras are absent from the calibration or RGB episode data: {missing}")
    robot_tracks = _load_robot_tracks(args.robot_tracks) if args.robot_tracks is not None else None
    track_camera_name = None
    track_calibration_name = None
    track_row_by_tick: dict[int, int] = {}
    if robot_tracks is not None:
        track_camera_name = args.robot_tracks_camera or scene_camera_names[0]
        if track_camera_name not in scene_camera_names:
            raise ValueError("--robot-tracks-camera must be one of --scene-camera-names")
        track_calibration_name = calibration_camera_names[scene_camera_names.index(track_camera_name)]
        track_row_by_tick = {int(tick_id): index for index, tick_id in enumerate(robot_tracks["tick_ids"])}

    # Importing trimesh is deliberately deferred until a real preprocessing
    # invocation; inspecting --help or this module does not require it.
    from real_scripts.ur7e_collision_mesh import UR7ePikaCollisionSurfacePointSampler, flange_transform, mesh_surface_samples

    sampler = UR7ePikaCollisionSurfacePointSampler(
        points_per_link=args.points_per_link,
        pika_mount_transform_json=args.pika_mount_transform_json,
        pika_full_collision_mesh=args.pika_full_collision_mesh,
    )
    if args.pika_mesh_reference_opening_mm < 0.0 or args.pika_mesh_reference_opening_mm > 95.0:
        raise ValueError("--pika-mesh-reference-opening-mm must be in [0, 95]")
    pika_parts = _load_pika_articulated_meshes(args.pika_full_collision_mesh)
    pika_local_part_samples = tuple(mesh_surface_samples(mesh, samples_per_face=4) for mesh in pika_parts)
    rgb_per_camera = {name: [] for name in rgb_camera_names}
    qpos_frames: list[np.ndarray] = []
    gripper_frames: list[np.ndarray] = []
    pika_opening_frames: list[float] = []
    action_frames: list[np.ndarray] = []
    action_valid: list[bool] = []
    tick_ids: list[int] = []
    query_ids: list[int] = []
    action_indices: list[int] = []
    fixed_surface_frames: list[np.ndarray] = []
    observed_robot_counts: list[list[int]] = []
    timestamp_skews_ns: list[int] = []
    visual_robot_tracks: list[np.ndarray] = []
    visual_robot_visible: list[np.ndarray] = []
    visual_robot_confidence: list[np.ndarray] = []
    visual_robot_xy: list[np.ndarray] = []
    visual_robot_tracker_visible: list[np.ndarray] = []
    max_skew_ns = int(round(float(args.max_camera_robot_skew_ms) * 1_000_000.0))

    for tick_dir in ticks:
        rgb, scene_frames, qpos, gripper, pika_opening_mm, action, metadata = _load_tick(
            tick_dir,
            rgb_camera_names=rgb_camera_names,
            scene_camera_names=scene_camera_names,
            calibration_camera_names=calibration_camera_names,
        )
        timestamp_valid, skew_ns = _timestamp_ok(metadata, camera_names=scene_camera_names, max_skew_ns=max_skew_ns)
        if not timestamp_valid:
            message = f"Tick {tick_dir.name} camera/robot timestamp skew {skew_ns / 1_000_000.0:.1f} ms exceeds {args.max_camera_robot_skew_ms:.1f} ms"
            if args.on_timestamp_skew == "error":
                raise ValueError(message)
            print(f"[skip] {message}")
            continue
        if pika_opening_mm is None:
            if args.on_missing_pika_opening == "error":
                raise FileNotFoundError(
                    f"Tick {tick_dir.name} has no gripper_opening_mm.npy. "
                    "Re-record with the current collector, or pass --on-missing-pika-opening reference."
                )
            pika_opening_mm = float(args.pika_mesh_reference_opening_mm)
        pika_local_points = _articulated_pika_surface_points(
            pika_local_part_samples,
            opening_mm=pika_opening_mm,
            reference_opening_mm=args.pika_mesh_reference_opening_mm,
        )
        pika_to_base = flange_transform(qpos) @ sampler.pika_mount_transform
        pika_points = (pika_to_base[:3, :3] @ pika_local_points.T).T + pika_to_base[:3, 3]
        fixed_surface = sampler.link_points(qpos)
        fixed_surface[-1] = _resample_fixed_order(pika_points, fixed_surface.shape[1])
        counts = [
            _observed_robot_point_count(
                scene_frames[name],
                calibration=calibrations[calibration_name],
                qpos=qpos,
                fixed_surface_points=fixed_surface,
                stride=args.depth_stride,
                max_depth_m=args.max_depth_m,
                absolute_tolerance_m=args.absolute_depth_tolerance_m,
                relative_tolerance=args.relative_depth_tolerance,
                dilation_pixels=args.depth_dilation_pixels,
            )
            for name, calibration_name in zip(scene_camera_names, calibration_camera_names, strict=True)
        ]
        tick_id = int(metadata.get("tick_id", tick_dir.name))
        if robot_tracks is not None:
            if tick_id not in track_row_by_tick:
                raise ValueError(f"Robot track file does not contain retained episode tick {tick_id}")
            row = track_row_by_tick[tick_id]
            lifted, visible = _lift_fk_gated_robot_tracks(
                scene_frames[track_camera_name],
                calibration=calibrations[track_calibration_name],
                qpos=qpos,
                fixed_surface_points=fixed_surface,
                tracks_xy=robot_tracks["tracks_xy"][row],
                tracker_visibility=robot_tracks["visibility"][row],
                tracker_confidence=robot_tracks["confidence"][row],
                min_confidence=args.track_min_confidence,
                absolute_tolerance_m=args.absolute_depth_tolerance_m,
                relative_tolerance=args.relative_depth_tolerance,
            )
            visual_robot_tracks.append(lifted)
            visual_robot_visible.append(visible)
            visual_robot_confidence.append(robot_tracks["confidence"][row])
            visual_robot_xy.append(robot_tracks["tracks_xy"][row])
            visual_robot_tracker_visible.append(robot_tracks["visibility"][row])
        for name in rgb_camera_names:
            rgb_per_camera[name].append(rgb[name])
        qpos_frames.append(qpos)
        gripper_frames.append(gripper)
        pika_opening_frames.append(float(pika_opening_mm))
        action_frames.append(action)
        action_valid.append((tick_dir / "commanded_action.npy").is_file())
        tick_ids.append(tick_id)
        query_ids.append(-1 if metadata.get("policy_query_id") is None else int(metadata["policy_query_id"]))
        action_indices.append(-1 if metadata.get("action_index") is None else int(metadata["action_index"]))
        fixed_surface_frames.append(fixed_surface)
        observed_robot_counts.append(counts)
        timestamp_skews_ns.append(-1 if skew_ns is None else int(skew_ns))

    if len(qpos_frames) <= args.future_horizon:
        raise ValueError("Too few valid frames to form one future point-flow target")
    gripper_sizes = {array.size for array in gripper_frames}
    if len(gripper_sizes) != 1:
        raise ValueError(f"Inconsistent gripper-state dimensions across frames: {sorted(gripper_sizes)}")
    qpos_array = np.stack(qpos_frames).astype(np.float32)
    gripper_array = np.stack(gripper_frames).astype(np.float32)
    actions = np.stack(action_frames).astype(np.float32)
    fixed_surface = np.stack(fixed_surface_frames).astype(np.float32)  # [T, L, P, 3]
    arm_points = fixed_surface.reshape(fixed_surface.shape[0], -1, 3)
    valid_actions = np.asarray(action_valid, dtype=bool)
    sample_indices = _sample_indices(
        valid_actions,
        np.asarray(tick_ids, dtype=np.int64),
        frame_count=len(fixed_surface),
        future_horizon=args.future_horizon,
    )
    if len(sample_indices) == 0:
        raise ValueError("No point-flow windows have a complete action horizon")
    target_offsets = np.stack(
        [arm_points[index + 1 : index + args.future_horizon + 1] - arm_points[index] for index in sample_indices]
    ).astype(np.float32)
    action_chunks = np.stack([actions[index : index + args.future_horizon] for index in sample_indices]).astype(np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": np.asarray(2, dtype=np.int32),
        "task_text": np.asarray(str(manifest.get("prompt", ""))),
        "coordinate_frame": np.asarray("ur_base"),
        "depth_removed": np.asarray(True),
        "source_episode_dir": np.asarray(str(episode_dir)),
        "calibration_path": np.asarray(str(calibration_path)),
        "tick_ids": np.asarray(tick_ids, dtype=np.int64),
        "policy_query_ids": np.asarray(query_ids, dtype=np.int64),
        "action_indices": np.asarray(action_indices, dtype=np.int64),
        "qpos": qpos_array,
        "gripper_state": gripper_array,
        "pika_opening_mm": np.asarray(pika_opening_frames, dtype=np.float32),
        "actions": actions,
        "action_valid": valid_actions,
        "fixed_link_points": fixed_surface,
        "arm_points": arm_points,
        "link_names": np.asarray(sampler.link_names),
        "point_ids": np.asarray(sampler.point_ids, dtype=np.int32),
        "local_link_points": np.asarray(sampler.local_link_points, dtype=np.float32),
        "point_identity_version": np.asarray(sampler.point_identity_version),
        "surface_model_hash": np.asarray(sampler.mesh_model_hash),
        "gripper_geometry_mode": np.asarray("articulated_body_plus_symmetric_fingers"),
        "pika_mesh_reference_opening_mm": np.asarray(args.pika_mesh_reference_opening_mm, dtype=np.float32),
        "observed_robot_point_counts": np.asarray(observed_robot_counts, dtype=np.int32),
        "observed_robot_camera_names": np.asarray(scene_camera_names),
        "observed_robot_calibration_camera_names": np.asarray(calibration_camera_names),
        "camera_robot_timestamp_skew_ns": np.asarray(timestamp_skews_ns, dtype=np.int64),
        "sample_frame_indices": sample_indices,
        "action_chunks": action_chunks,
        "current_link_points": arm_points[sample_indices],
        "target_point_offsets": target_offsets,
        "future_link_offsets": target_offsets.reshape(len(sample_indices), args.future_horizon, fixed_surface.shape[1], fixed_surface.shape[2], 3),
    }
    if robot_tracks is not None:
        visual_tracks_array = np.stack(visual_robot_tracks).astype(np.float32)
        visual_visible_array = np.stack(visual_robot_visible).astype(bool)
        visual_confidence_array = np.stack(visual_robot_confidence).astype(np.float32)
        visual_xy_array = np.stack(visual_robot_xy).astype(np.float32)
        visual_tracker_visible_array = np.stack(visual_robot_tracker_visible).astype(bool)
        visual_target_offsets = np.stack(
            [visual_tracks_array[index + 1 : index + args.future_horizon + 1] - visual_tracks_array[index] for index in sample_indices]
        ).astype(np.float32)
        visual_future_visible = np.stack(
            [visual_visible_array[index + 1 : index + args.future_horizon + 1] for index in sample_indices]
        ).astype(bool)
        visual_flow_mask = visual_visible_array[sample_indices, None, :] & visual_future_visible
        payload.update(
            {
                "visual_robot_track_source": np.asarray(str(args.robot_tracks)),
                "visual_robot_track_camera": np.asarray(track_camera_name),
                "visual_robot_seed_ids": np.arange(visual_tracks_array.shape[1], dtype=np.int32),
                "visual_robot_seed_xy": robot_tracks["seed_xy"].astype(np.float32),
                "visual_robot_track_xy": visual_xy_array,
                "visual_robot_tracker_visible_mask": visual_tracker_visible_array,
                "visual_robot_tracks": visual_tracks_array,
                "visual_robot_visible_mask": visual_visible_array,
                "visual_robot_confidence": visual_confidence_array,
                "visual_robot_current_points": visual_tracks_array[sample_indices],
                "visual_robot_current_visible_mask": visual_visible_array[sample_indices],
                "visual_robot_future_offsets": visual_target_offsets,
                "visual_robot_future_visible_mask": visual_future_visible,
                "visual_robot_flow_supervision_mask": visual_flow_mask,
            }
        )
    for name, frames in rgb_per_camera.items():
        payload[f"rgb_{name}"] = np.stack(frames).astype(np.uint8)
    np.savez_compressed(output, **payload)
    return {
        "output": str(output),
        "frame_count": len(fixed_surface),
        "sample_count": len(sample_indices),
        "point_count": int(arm_points.shape[1]),
        "scene_camera_names": list(scene_camera_names),
        "calibration_camera_names": list(calibration_camera_names),
    }


def _materialize_legacy_demo_episode(
    source: Path,
    destination: Path,
    *,
    scene_camera_map_override: dict[str, str] | None = None,
) -> None:
    """Build a temporary raw-tick view of a legacy Gradio demo episode.

    The source layout remains untouched: the compatibility view exists only
    for this preprocessing invocation.
    """
    import cv2

    metadata = _load_json(source / "metadata.json")
    calibration_source = source / "calibration.json"
    if not calibration_source.is_file():
        raise FileNotFoundError(
            f"Legacy episode {source} has no calibration.json. Re-record it after enabling RGB-D capture."
        )
    camera_dirs = {
        "front": (
            source / str(metadata.get("front_rgb_dir", "front_rgb")),
            source / str(metadata.get("front_depth_m_dir", "front_depth_m")),
        )
    }
    side_rgb_dir = metadata.get("side_rgb_dir")
    side_depth_dir = metadata.get("side_depth_m_dir")
    if (side_rgb_dir is None) != (side_depth_dir is None):
        raise ValueError("Legacy episode must provide both side_rgb_dir and side_depth_m_dir")
    if side_rgb_dir is not None:
        camera_dirs["side"] = (source / str(side_rgb_dir), source / str(side_depth_dir))
    camera_paths = {name: sorted(rgb_dir.glob("frame_*.png")) for name, (rgb_dir, _depth_dir) in camera_dirs.items()}
    if not camera_paths["front"]:
        raise FileNotFoundError(f"No front PNG frames found in {camera_dirs['front'][0]}")
    for name, (rgb_dir, depth_dir) in camera_dirs.items():
        if not depth_dir.is_dir():
            raise FileNotFoundError(
                f"Legacy episode {source} has no {depth_dir.name}/ depth directory for {name}. "
                "Depth cannot be reconstructed from RGB-only recordings."
            )
        if len(camera_paths[name]) != len(camera_paths["front"]):
            raise ValueError(
                f"Legacy {name} RGB count is {len(camera_paths[name])}, expected {len(camera_paths['front'])} from front RGB"
            )
    joints = np.asarray(np.load(source / "joints.npy", allow_pickle=False), dtype=np.float32)
    gripper = np.asarray(np.load(source / "gripper.npy", allow_pickle=False), dtype=np.float32)
    opening_path = source / str(metadata.get("gripper_opening_mm_file", "gripper_opening_mm.npy"))
    if not opening_path.is_file():
        raise FileNotFoundError(f"Legacy episode is missing gripper opening values: {opening_path}")
    openings = np.asarray(np.load(opening_path, allow_pickle=False), dtype=np.float32)
    actions = np.asarray(np.load(source / "actions.npy", allow_pickle=False), dtype=np.float32)
    timestamps_path = source / str(metadata.get("timestamps_file", "timestamps_s.npy"))
    timestamps = np.asarray(np.load(timestamps_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    frame_count = len(camera_paths["front"])
    arrays = {"joints": joints, "gripper": gripper, "gripper_opening_mm": openings, "actions": actions, "timestamps_s": timestamps}
    invalid = {name: value.shape for name, value in arrays.items() if len(value) != frame_count}
    if invalid:
        raise ValueError(f"Legacy episode frame count is {frame_count}, but arrays disagree: {invalid}")
    if joints.ndim != 2 or joints.shape[1] < 6 or actions.shape != (frame_count, 7):
        raise ValueError(f"Invalid legacy joints/actions shapes: {joints.shape}, {actions.shape}")

    destination.mkdir(parents=True, exist_ok=False)
    scene_camera_map = metadata.get("scene_camera_map", {})
    if not isinstance(scene_camera_map, dict):
        raise ValueError("Legacy episode metadata scene_camera_map must be an object")
    scene_camera_map = {str(name): str(calibration_name) for name, calibration_name in scene_camera_map.items()}
    if scene_camera_map_override:
        scene_camera_map.update(scene_camera_map_override)
    missing_camera_maps = [name for name in camera_dirs if name not in scene_camera_map]
    if missing_camera_maps:
        missing = ", ".join(missing_camera_maps)
        raise ValueError(
            f"Legacy episode has RGB-D data for {missing}, but no calibration map. "
            f"Pass --scene-camera-map {missing}=CALIBRATION_CAMERA."
        )
    _atomic_manifest = {
        "status": "complete",
        "format_version": 2,
        "prompt": str(metadata.get("task", "")),
        "scene_camera_map": scene_camera_map,
        "source_layout": "ur7e_vla_legacy_png_npy",
    }
    (destination / "manifest.json").write_text(json.dumps(_atomic_manifest, indent=2), encoding="utf-8")

    # Some older episodes may have downsampled PNG/depth frames, while the
    # copied D435i calibration describes the original stream. Rescale
    # fx/fy/cx/cy in the temporary copy only when dimensions differ.
    calibration_payload = _load_json(calibration_source)
    camera_payloads = calibration_payload.get("cameras", calibration_payload)
    for name, rgb_paths in camera_paths.items():
        first_bgr = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
        if first_bgr is None:
            raise RuntimeError(f"Cannot read {rgb_paths[0]}")
        height, width = first_bgr.shape[:2]
        calibration_name = scene_camera_map[name]
        item = camera_payloads.get(str(calibration_name))
        if not isinstance(item, dict):
            raise KeyError(f"Calibration {calibration_name!r} referenced by legacy metadata is absent")
        old_width, old_height = int(item["width"]), int(item["height"])
        intrinsics = np.asarray(item["intrinsics"], dtype=np.float64)
        intrinsics[0, :] *= width / old_width
        intrinsics[1, :] *= height / old_height
        intrinsics[2, :] = (0.0, 0.0, 1.0)
        item["intrinsics"] = intrinsics.tolist()
        item["width"], item["height"] = width, height
    (destination / "calibration.json").write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    for tick_id in range(frame_count):
        tick_dir = destination / "frames" / f"{tick_id:06d}"
        tick_dir.mkdir(parents=True)
        for name, (rgb_dir, depth_dir) in camera_dirs.items():
            rgb_path = camera_paths[name][tick_id]
            depth_path = depth_dir / f"{rgb_path.stem}.npy"
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing {name} depth corresponding to {rgb_path.name}: {depth_path}")
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Cannot read {rgb_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
            if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all():
                raise ValueError(f"Invalid {name} depth {depth_path}: {depth.shape}, expected {rgb.shape[:2]}")
            np.save(tick_dir / f"{name}_rgb.npy", rgb, allow_pickle=False)
            np.save(tick_dir / f"{name}_depth_raw_m.npy", depth, allow_pickle=False)
        np.save(tick_dir / "qpos.npy", joints[tick_id, :6], allow_pickle=False)
        np.save(tick_dir / "gripper_state.npy", gripper[tick_id], allow_pickle=False)
        np.save(tick_dir / "gripper_opening_mm.npy", openings[tick_id], allow_pickle=False)
        np.save(tick_dir / "commanded_action.npy", actions[tick_id], allow_pickle=False)
        timestamp_ns = int(round(float(timestamps[tick_id]) * 1_000_000_000.0))
        (tick_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "tick_id": tick_id,
                    "robot_state_timestamp_ns": timestamp_ns,
                    "policy_query_id": None,
                    "action_index": tick_id,
                    "frames": {name: {"host_timestamp_ns": timestamp_ns} for name in camera_dirs},
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def preprocess_episode(args: argparse.Namespace) -> dict[str, Any]:
    episode_dir = Path(args.episode_dir)
    if (episode_dir / "manifest.json").is_file():
        return _preprocess_raw_episode(args)
    with tempfile.TemporaryDirectory(prefix="pi05_legacy_episode_") as temporary_root:
        compatibility_dir = Path(temporary_root) / "episode"
        _materialize_legacy_demo_episode(
            episode_dir,
            compatibility_dir,
            scene_camera_map_override=_camera_name_map(args.scene_camera_map),
        )
        compatibility_values = dict(vars(args))
        compatibility_values["episode_dir"] = compatibility_dir
        compatibility_args = argparse.Namespace(**compatibility_values)
        return _preprocess_raw_episode(compatibility_args)


def main() -> None:
    result = preprocess_episode(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
