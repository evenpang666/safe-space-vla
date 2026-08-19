#!/usr/bin/env python3
"""Create PointWorld-style stable robot-surface tracks with CoTracker3.

Initial seeds come from *measured* RGB-D pixels whose depth agrees with a
rendered robot mask.  CoTracker preserves those pixel IDs through the video;
the companion preprocessor then lifts every tracked pixel with measured depth
and applies the same FK depth gate again.  Thus the robot model is used only
to select/gate pixels, never as the output 3-D surface geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.preprocess_pi05_rgbd_surface_dataset import _combined_robot_depth
from real_scripts.real_robot_adapter import RGBDFrame, load_camera_calibrations, robot_depth_keep_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True, help="Raw PNG/NPY demo episode.")
    parser.add_argument("--surface-npz", type=Path, required=True, help="Dual/single-view safety preprocessing output used only for FK masks.")
    parser.add_argument("--camera", choices=("front", "side"), required=True)
    parser.add_argument("--calibration-camera", required=True, help="Calibration JSON key for --camera.")
    parser.add_argument("--output", type=Path, required=True, help="Tracker-interchange .npz output.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional local CoTracker3 scaled-offline checkpoint, avoiding a Hub download.")
    parser.add_argument("--max-seeds", type=int, default=128)
    parser.add_argument("--seed-stride", type=int, default=6, help="Pixel stride before uniform seed subsampling.")
    parser.add_argument("--tracking-scale", type=float, default=0.5, help="CoTracker input scale in (0, 1]. Coordinates are rescaled back before save.")
    parser.add_argument("--chunk-size", type=int, default=8, help="Frames per offline window; adjacent windows overlap by one frame to preserve seed IDs.")
    parser.add_argument("--tracker", choices=("cotracker", "lk"), default="cotracker", help="2-D tracker. lk is a dependency-free CPU fallback with the same stable-ID output contract.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
    return value


def _load_rgb_frames(episode_dir: Path, camera: str) -> np.ndarray:
    directory = episode_dir / f"{camera}_rgb"
    paths = sorted(directory.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No RGB PNGs found in {directory}")
    frames = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"{camera} RGB frame shapes are inconsistent: {sorted(shapes)}")
    return np.stack(frames).astype(np.uint8)


def _initial_surface_seeds(
    *,
    episode_dir: Path,
    camera: str,
    rgb: np.ndarray,
    qpos: np.ndarray,
    fixed_surface: np.ndarray,
    calibration,
    max_seeds: int,
    stride: int,
) -> np.ndarray:
    if max_seeds < 1 or stride < 1:
        raise ValueError("--max-seeds and --seed-stride must be positive")
    depth_path = episode_dir / f"{camera}_depth_m" / "frame_000000.npy"
    depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
    if depth.shape != rgb.shape[1:3]:
        raise ValueError(f"Initial depth {depth.shape} does not match RGB {rgb.shape[1:3]}")
    rendered_depth = _combined_robot_depth(
        qpos=qpos[0],
        fixed_surface_points=fixed_surface[0],
        calibration=calibration,
        height=depth.shape[0],
        width=depth.shape[1],
    )
    environment = robot_depth_keep_mask(
        depth,
        rendered_depth,
        absolute_tolerance_m=0.012,
        relative_tolerance=0.015,
        dilation_pixels=2,
    )
    mask = ~environment
    grid = np.zeros_like(mask, dtype=bool)
    grid[::stride, ::stride] = True
    mask &= grid
    v, u = np.nonzero(mask)
    if len(u) == 0:
        raise RuntimeError("No measured-depth robot pixels passed the initial FK gate")
    candidates = np.c_[u, v].astype(np.float32)
    if len(candidates) > max_seeds:
        candidates = candidates[np.linspace(0, len(candidates) - 1, max_seeds, dtype=np.int64)]
    return candidates


def _track_lk(video: np.ndarray, seed_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Track initial pixels with forward-backward pyramidal Lucas--Kanade.

    An invalid point is never reinitialized: its stable ID remains present,
    with a false visibility entry, exactly as required by the PointWorld
    interchange format.
    """
    frames = np.asarray(video, dtype=np.uint8)
    tracks = np.zeros((len(frames), len(seed_xy), 2), dtype=np.float32)
    visibility = np.zeros((len(frames), len(seed_xy)), dtype=bool)
    current = np.asarray(seed_xy, dtype=np.float32).copy()
    active = np.ones(len(current), dtype=bool)
    tracks[0] = current
    visibility[0] = active
    previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    height, width = previous_gray.shape
    params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for index in range(1, len(frames)):
        next_gray = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY)
        next_points, forward_ok, _error = cv2.calcOpticalFlowPyrLK(previous_gray, next_gray, current.reshape(-1, 1, 2), None, **params)
        back_points, backward_ok, _error = cv2.calcOpticalFlowPyrLK(next_gray, previous_gray, next_points, None, **params)
        next_points = next_points.reshape(-1, 2)
        back_points = back_points.reshape(-1, 2)
        forward_ok = forward_ok.reshape(-1).astype(bool)
        backward_ok = backward_ok.reshape(-1).astype(bool)
        round_trip_error = np.linalg.norm(back_points - current, axis=1)
        in_bounds = (next_points[:, 0] >= 0.0) & (next_points[:, 0] < width) & (next_points[:, 1] >= 0.0) & (next_points[:, 1] < height)
        active &= forward_ok & backward_ok & in_bounds & np.isfinite(next_points).all(axis=1) & (round_trip_error <= 1.5)
        current[active] = next_points[active]
        tracks[index] = current
        visibility[index] = active
        previous_gray = next_gray
    return tracks, visibility


def main() -> None:
    args = parse_args()
    if not 0.0 < args.tracking_scale <= 1.0:
        raise ValueError("--tracking-scale must be in (0, 1]")
    device = _device(args.device)
    rgb = _load_rgb_frames(args.episode_dir, args.camera)
    with np.load(args.surface_npz, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fixed_surface = np.asarray(data["fixed_link_points"], dtype=np.float32)
    if len(rgb) != len(qpos) or len(qpos) != len(fixed_surface):
        raise ValueError("RGB frame count and surface-NPZ frame count must agree")
    calibrations = load_camera_calibrations(args.episode_dir / "calibration.json")
    if args.calibration_camera not in calibrations:
        raise KeyError(f"Unknown calibration key {args.calibration_camera!r}; choices: {sorted(calibrations)}")
    seed_xy = _initial_surface_seeds(
        episode_dir=args.episode_dir,
        camera=args.camera,
        rgb=rgb,
        qpos=qpos,
        fixed_surface=fixed_surface,
        calibration=calibrations[args.calibration_camera],
        max_seeds=args.max_seeds,
        stride=args.seed_stride,
    )
    if args.tracking_scale != 1.0:
        scaled_width = int(round(rgb.shape[2] * args.tracking_scale))
        scaled_height = int(round(rgb.shape[1] * args.tracking_scale))
        video = np.stack([cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA) for frame in rgb])
        model_seed_xy = seed_xy * args.tracking_scale
    else:
        video = rgb
        model_seed_xy = seed_xy
    if args.chunk_size < 2:
        raise ValueError("--chunk-size must be at least 2")
    print(f"[info] camera={args.camera} frames={len(rgb)} seeds={len(seed_xy)} device={device} tracking_size={video.shape[2]}x{video.shape[1]} chunk_size={args.chunk_size}", flush=True)
    if args.tracker == "lk":
        tracks_xy, visibility = _track_lk(video, model_seed_xy)
    else:
        model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
            pretrained=args.checkpoint is None,
            trust_repo=True,
        )
        if args.checkpoint is not None:
            if not args.checkpoint.is_file():
                raise FileNotFoundError(f"CoTracker checkpoint does not exist: {args.checkpoint}")
            state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model.model.load_state_dict(state_dict)
        model = model.to(device).eval()
        tracked_xy_parts: list[np.ndarray] = []
        visibility_parts: list[np.ndarray] = []
        current_xy = model_seed_xy.copy()
        start = 0
        while start < len(video):
            end = min(start + args.chunk_size, len(video))
            queries = np.concatenate((np.zeros((len(current_xy), 1), dtype=np.float32), current_xy), axis=1)[None]
            video_tensor = torch.from_numpy(video[start:end]).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
            query_tensor = torch.from_numpy(queries).to(device)
            print(f"[track] {args.camera}: frames {start + 1}-{end}/{len(video)}", flush=True)
            with torch.inference_mode():
                tracks, visibility = model(video_tensor, queries=query_tensor)
            local_xy = tracks[0].detach().cpu().numpy().astype(np.float32)
            local_visibility = visibility[0].detach().cpu().numpy()
            if local_visibility.ndim == 3 and local_visibility.shape[-1] == 1:
                local_visibility = local_visibility[..., 0]
            local_visibility = np.asarray(local_visibility, dtype=bool)
            if start:
                local_xy = local_xy[1:]
                local_visibility = local_visibility[1:]
            tracked_xy_parts.append(local_xy)
            visibility_parts.append(local_visibility)
            current_xy = tracks[0, -1].detach().cpu().numpy().astype(np.float32)
            if end == len(video):
                break
            start = end - 1
        tracks_xy = np.concatenate(tracked_xy_parts, axis=0)
        visibility = np.concatenate(visibility_parts, axis=0)
    if args.tracking_scale != 1.0:
        tracks_xy /= args.tracking_scale
    if tracks_xy.shape != (len(rgb), len(seed_xy), 2) or visibility.shape != tracks_xy.shape[:2]:
        raise RuntimeError(f"Unexpected CoTracker output: tracks={tracks_xy.shape}, visibility={visibility.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        tracks_xy=tracks_xy,
        visibility=visibility,
        confidence=visibility.astype(np.float32),
        tick_ids=np.arange(len(rgb), dtype=np.int64),
        seed_xy=seed_xy,
        camera=np.asarray(args.camera),
        tracker=np.asarray("CoTracker3 offline; stable seed IDs are columns" if args.tracker == "cotracker" else "OpenCV forward-backward pyramidal LK; stable seed IDs are columns"),
        tracking_scale=np.asarray(args.tracking_scale, dtype=np.float32),
        chunk_size=np.asarray(args.chunk_size, dtype=np.int32),
    )
    print(f"[done] wrote {args.output}; CoTracker-visible observations={int(visibility.sum())}/{visibility.size}", flush=True)


if __name__ == "__main__":
    main()
