#!/usr/bin/env python3
"""Render measured, depth-lifted PointWorld robot tracks from one or two cameras.

Unlike the fixed collision-surface visualizer, every rendered 3-D point in
this video originates in RGB tracking plus measured depth.  The FK model has
already been used only as a gate by the preprocessing step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-npz", type=Path, required=True)
    parser.add_argument("--side-npz", type=Path, default=None, help="Optional PointWorld data containing side-camera tracks.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def _load(path: Path, camera: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        rgb = np.asarray(data[f"rgb_{camera}"], dtype=np.uint8)
        xy = np.asarray(data["visual_robot_track_xy"], dtype=np.float32)
        points = np.asarray(data["visual_robot_tracks"], dtype=np.float32)
        visible = np.asarray(data["visual_robot_visible_mask"], dtype=bool)
        saved_camera = str(data["visual_robot_track_camera"].item())
    if saved_camera != camera:
        raise ValueError(f"{path} contains PointWorld tracks for {saved_camera!r}, expected {camera!r}")
    if xy.shape[:2] != visible.shape or points.shape[:2] != visible.shape or rgb.shape[0] != len(visible):
        raise ValueError(f"Inconsistent PointWorld arrays in {path}")
    return rgb, xy, points, visible


def _seed_colors(count: int) -> np.ndarray:
    hsv = np.zeros((count, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (np.arange(count) * 179 // max(count, 1)).astype(np.uint8)
    hsv[:, 0, 1:] = (210, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]


def _overlay(rgb: np.ndarray, xy: np.ndarray, visible: np.ndarray, colors: np.ndarray, label: str) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for point, color in zip(xy[visible], colors[visible], strict=True):
        cv2.circle(image, tuple(np.rint(point).astype(int)), 2, tuple(int(x) for x in color), -1, cv2.LINE_AA)
    cv2.rectangle(image, (10, 68), (220, 94), (12, 14, 18), -1)
    cv2.putText(image, label, (18, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (245, 245, 245), 1, cv2.LINE_AA)
    return image


def _base_view(points: np.ndarray, visibility: np.ndarray, colors: np.ndarray, *, limits: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    x0, x1, z0, z1 = limits
    image = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)

    def pixel(point: np.ndarray) -> tuple[int, int]:
        return (int(round((point[0] - x0) / (x1 - x0) * (width - 1))), int(round((z1 - point[2]) / (z1 - z0) * (height - 1))))

    if x0 <= 0 <= x1:
        u = pixel(np.asarray((0.0, 0.0, z0), dtype=np.float32))[0]
        cv2.line(image, (u, 0), (u, height - 1), (55, 59, 66), 1, cv2.LINE_AA)
    if z0 <= 0 <= z1:
        v = pixel(np.asarray((x0, 0.0, 0.0), dtype=np.float32))[1]
        cv2.line(image, (0, v), (width - 1, v), (55, 59, 66), 1, cv2.LINE_AA)
    for point, color in zip(points[visibility], colors[visibility], strict=True):
        u, v = pixel(point)
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(image, (u, v), 2, tuple(int(x) for x in color), -1, cv2.LINE_AA)
    cv2.putText(image, "UR base x-z: measured points", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(image, "only depth-lifted, FK-gated tracks", (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (205, 205, 205), 1, cv2.LINE_AA)
    return image


def main() -> None:
    args = parse_args()
    front_rgb, front_xy, front_points, front_visible = _load(args.front_npz, "front")
    side = _load(args.side_npz, "side") if args.side_npz is not None else None
    if side is not None and (front_rgb.shape[0] != side[0].shape[0] or front_rgb.shape[1:3] != side[0].shape[1:3]):
        raise ValueError("Front and side RGB sequences must have equal frame count and dimensions")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    front_colors = _seed_colors(front_xy.shape[1])
    if side is not None:
        side_rgb, side_xy, side_points, side_visible = side
        side_colors = _seed_colors(side_xy.shape[1])
        valid_points = np.concatenate((front_points[front_visible], side_points[side_visible]), axis=0)
    else:
        side_rgb = side_xy = side_points = side_visible = side_colors = None
        valid_points = front_points[front_visible]
    if not len(valid_points):
        raise ValueError("No depth-lifted PointWorld points are visible")
    low = valid_points[:, (0, 2)].min(axis=0) - 0.06
    high = valid_points[:, (0, 2)].max(axis=0) + 0.06
    limits = (float(low[0]), float(high[0]), float(min(low[1], -0.04)), float(high[1]))
    height, width = front_rgb.shape[1:3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", macro_block_size=1, ffmpeg_log_level="error")
    try:
        for index in range(len(front_rgb)):
            front = _overlay(front_rgb[index], front_xy[index], front_visible[index], front_colors, "front: measured PointWorld")
            panes = [front]
            points, visible, colors = front_points[index], front_visible[index], front_colors
            if side_rgb is not None:
                side_view = _overlay(side_rgb[index], side_xy[index], side_visible[index], side_colors, "side: measured PointWorld")
                panes.append(side_view)
                points = np.concatenate((points, side_points[index]), axis=0)
                visible = np.concatenate((visible, side_visible[index]))
                colors = np.concatenate((colors, side_colors), axis=0)
            base = _base_view(points, visible, colors, limits=limits, size=(width, height))
            panes.append(base)
            combined = np.concatenate(panes, axis=1)
            cv2.rectangle(combined, (0, 0), (combined.shape[1], 58), (10, 12, 16), -1)
            cv2.putText(combined, f"Measured PointWorld robot surface tracks  |  frame {index + 1}/{len(front_rgb)}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (245, 245, 245), 1, cv2.LINE_AA)
            status = f"front valid: {int(front_visible[index].sum())}/{front_visible.shape[1]}"
            if side_visible is not None:
                status += f"   |   side valid: {int(side_visible[index].sum())}/{side_visible.shape[1]}"
            cv2.putText(combined, status + "   |   model points are not rendered", (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (205, 211, 217), 1, cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            if index == 0 and args.preview is not None:
                cv2.imwrite(str(args.preview), combined)
    finally:
        writer.close()
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
