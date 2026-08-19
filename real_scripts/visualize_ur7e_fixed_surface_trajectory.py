#!/usr/bin/env python3
"""Render a UR7e/PiKA fixed-surface point-cloud trajectory as an MP4.

The input is the ``.npz`` produced by
``preprocess_pi05_rgbd_surface_dataset.py``.  The points are coloured by
stable link ID, rather than by image colour, so the same physical sample has
the same identity throughout the video.  The first pane overlays the rendered
points on the front image.  When the preprocessed episode includes
``rgb_side``, a second camera pane can be enabled with ``--side-camera-name``.
The final pane is a UR-base x/z view with the PiKA-centroid path accumulated
through the current frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.real_robot_adapter import load_camera_calibrations


# RGB colours, one deterministic colour for each of the seven UR7e collision
# links plus the PiKA gripper.  The final magenta entry intentionally makes
# the gripper easy to inspect during approach and grasp motions.
LINK_COLORS = np.asarray(
    (
        (143, 190, 255),  # base
        (59, 221, 182),  # shoulder
        (248, 193, 66),  # upper arm
        (255, 139, 77),  # forearm
        (220, 108, 255),  # wrist 1
        (103, 157, 255),  # wrist 2
        (192, 225, 94),  # wrist 3
        (255, 61, 173),  # PiKA gripper
    ),
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-npz", type=Path, required=True, help="Preprocessed fixed-surface .npz.")
    parser.add_argument("--calibration", type=Path, required=True, help="Source episode camera calibration JSON.")
    parser.add_argument("--camera-name", default="405622074939", help="Calibration key used for the recorded RGB frames.")
    parser.add_argument("--side-camera-name", default=None, help="Optional calibration key for rgb_side, producing a second camera pane.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--preview", type=Path, default=None, help="Optional first-frame PNG path.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--point-radius", type=int, default=2)
    return parser.parse_args()


def _project(points: np.ndarray, *, camera_to_world: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Project base-frame points and return pixels plus their valid mask."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    camera = (world_to_camera @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    z = camera[:, 2]
    u = np.rint(float(intrinsics[0, 0]) * camera[:, 0] / z + float(intrinsics[0, 2])).astype(np.int32)
    v = np.rint(float(intrinsics[1, 1]) * camera[:, 1] / z + float(intrinsics[1, 2])).astype(np.int32)
    valid = np.isfinite(camera).all(axis=1) & (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.c_[u, v], valid


def _draw_camera_overlay(rgb: np.ndarray, link_points: np.ndarray, *, camera_to_world: np.ndarray, intrinsics: np.ndarray, radius: int) -> np.ndarray:
    image = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    points = np.asarray(link_points, dtype=np.float32)
    pixels, valid = _project(points.reshape(-1, 3), camera_to_world=camera_to_world, intrinsics=intrinsics, width=image.shape[1], height=image.shape[0])
    colors = np.repeat(LINK_COLORS[: points.shape[0]], points.shape[1], axis=0)
    # Draw distant samples first, giving nearer points visual precedence.
    for (u, v), color in zip(pixels[valid], colors[valid], strict=True):
        cv2.circle(image, (int(u), int(v)), max(1, int(radius)), tuple(int(channel) for channel in color[::-1]), thickness=-1, lineType=cv2.LINE_AA)
    return image


def _add_pane_label(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (10, 68), (160, 94), (12, 14, 18), thickness=-1)
    cv2.putText(result, label, (18, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def _draw_base_view(link_points: np.ndarray, history: np.ndarray, *, x_limits: tuple[float, float], z_limits: tuple[float, float], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)
    x0, x1 = x_limits
    z0, z1 = z_limits

    def pixel(point: np.ndarray) -> tuple[int, int]:
        u = int(round((float(point[0]) - x0) / (x1 - x0) * (width - 1)))
        v = int(round((z1 - float(point[2])) / (z1 - z0) * (height - 1)))
        return u, v

    # Coordinate axes make the base frame orientation explicit.
    if x0 <= 0.0 <= x1:
        axis_x = pixel(np.asarray((0.0, 0.0, z0), dtype=np.float32))[0]
        cv2.line(image, (axis_x, 0), (axis_x, height - 1), (55, 59, 66), 1, cv2.LINE_AA)
    if z0 <= 0.0 <= z1:
        axis_z = pixel(np.asarray((x0, 0.0, 0.0), dtype=np.float32))[1]
        cv2.line(image, (0, axis_z), (width - 1, axis_z), (55, 59, 66), 1, cv2.LINE_AA)
    if len(history) > 1:
        cv2.polylines(image, [np.asarray([pixel(point) for point in history], dtype=np.int32)], False, (205, 205, 205), 1, cv2.LINE_AA)
    for link_index, points in enumerate(link_points):
        color = tuple(int(channel) for channel in LINK_COLORS[link_index, ::-1])
        for point in points:
            u, v = pixel(point)
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(image, (u, v), 2 if link_index == len(link_points) - 1 else 1, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(image, "UR base view: x-z", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(image, "white = PiKA centroid path", (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(image, "x", (width - 24, max(18, pixel(np.asarray((x1, 0.0, 0.0)))[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(image, "z", (max(4, pixel(np.asarray((0.0, 0.0, z1)))[0] + 6), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return image


def _add_header(frame: np.ndarray, *, index: int, count: int, opening_mm: float, observed_count: int, link_names: np.ndarray) -> np.ndarray:
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 58), (10, 12, 16), thickness=-1)
    cv2.putText(image, f"Fixed-identity UR7e + PiKA collision-surface point cloud  |  frame {index + 1}/{count}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(image, f"PiKA opening: {opening_mm:.1f} mm   |   FK-gated observed robot points: {observed_count}   |   1,024 stable samples", (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (205, 211, 217), 1, cv2.LINE_AA)
    legend_x = image.shape[1] - 368
    for index, name in enumerate(link_names):
        column, row = divmod(index, 4)
        x, y = legend_x + column * 180, 17 + row * 22
        color = tuple(int(channel) for channel in LINK_COLORS[index, ::-1])
        cv2.circle(image, (x, y), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(image, str(name).replace("pika_gripper_rigid", "PiKA gripper"), (x + 8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    return image


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    with np.load(args.surface_npz, allow_pickle=False) as data:
        link_points = np.asarray(data["fixed_link_points"], dtype=np.float32)
        rgb = np.asarray(data["rgb_front"], dtype=np.uint8)
        side_rgb = np.asarray(data["rgb_side"], dtype=np.uint8) if "rgb_side" in data.files else None
        link_names = np.asarray(data["link_names"])
        opening_mm = np.asarray(data["pika_opening_mm"], dtype=np.float32)
        observed = np.asarray(data["observed_robot_point_counts"], dtype=np.int32).reshape(len(link_points), -1).sum(axis=1)
    if link_points.ndim != 4 or link_points.shape[-1] != 3 or rgb.shape[0] != len(link_points):
        raise ValueError("Surface NPZ has inconsistent fixed_link_points/rgb_front arrays")
    if side_rgb is not None and (side_rgb.shape[0] != len(link_points) or side_rgb.ndim != 4 or side_rgb.shape[-1] != 3):
        raise ValueError("Surface NPZ has inconsistent rgb_side array")
    if link_points.shape[0] != len(opening_mm) or len(link_names) != link_points.shape[1]:
        raise ValueError("Surface NPZ has inconsistent point metadata")
    if len(link_names) > len(LINK_COLORS):
        raise ValueError(f"Only {len(LINK_COLORS)} link colours are defined, got {len(link_names)}")

    calibrations = load_camera_calibrations(args.calibration)
    if args.camera_name not in calibrations:
        raise KeyError(f"Camera {args.camera_name!r} is absent from {args.calibration}; choices: {sorted(calibrations)}")
    calibration = calibrations[args.camera_name]
    side_calibration = None
    if args.side_camera_name is not None:
        if side_rgb is None:
            raise ValueError("--side-camera-name was supplied, but the surface NPZ has no rgb_side frames")
        if args.side_camera_name not in calibrations:
            raise KeyError(f"Side camera {args.side_camera_name!r} is absent from {args.calibration}; choices: {sorted(calibrations)}")
        side_calibration = calibrations[args.side_camera_name]
    all_points = link_points.reshape(-1, 3)
    x_low, z_low = all_points[:, (0, 2)].min(axis=0)
    x_high, z_high = all_points[:, (0, 2)].max(axis=0)
    margin = 0.06
    x_limits = (float(x_low - margin), float(x_high + margin))
    z_limits = (float(min(z_low - margin, -0.04)), float(z_high + margin))
    height, width = rgb.shape[1:3]
    output_size = (width * (3 if side_calibration is not None else 2), height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
    gripper_centroids = link_points[:, -1].mean(axis=1)
    writer = imageio.get_writer(args.output, fps=float(args.fps), codec="libx264", macro_block_size=1, ffmpeg_log_level="error")
    try:
        for frame_index, frame_points in enumerate(link_points):
            camera_view = _add_pane_label(
                _draw_camera_overlay(rgb[frame_index], frame_points, camera_to_world=calibration.camera_to_world, intrinsics=calibration.intrinsics, radius=args.point_radius),
                "front RGB + FK",
            )
            base_view = _draw_base_view(frame_points, gripper_centroids[: frame_index + 1], x_limits=x_limits, z_limits=z_limits, size=(width, height))
            panes = [camera_view]
            if side_calibration is not None:
                side_view = _draw_camera_overlay(side_rgb[frame_index], frame_points, camera_to_world=side_calibration.camera_to_world, intrinsics=side_calibration.intrinsics, radius=args.point_radius)
                if side_view.shape[:2] != (height, width):
                    side_view = cv2.resize(side_view, (width, height), interpolation=cv2.INTER_AREA)
                panes.append(_add_pane_label(side_view, "side RGB + FK"))
            panes.append(base_view)
            combined = np.concatenate(panes, axis=1)
            combined = _add_header(combined, index=frame_index, count=len(link_points), opening_mm=float(opening_mm[frame_index]), observed_count=int(observed[frame_index]), link_names=link_names)
            rgb_frame = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
            if frame_index == 0 and args.preview is not None:
                cv2.imwrite(str(args.preview), combined)
    finally:
        writer.close()
    print(f"[done] wrote {len(link_points)} frames to {args.output} ({output_size[0]}x{output_size[1]} @ {args.fps:g} fps)")


if __name__ == "__main__":
    main()
