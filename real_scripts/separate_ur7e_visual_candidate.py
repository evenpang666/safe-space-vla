#!/usr/bin/env python3
"""Separate a manually verified visible-UR7e point-cloud candidate.

This diagnostic tool uses image-space silhouettes for one captured front/side
pair, then back-projects the selected pixels using the calibrated depth maps.
It is intentionally not a replacement for the FK/URDF mesh-based mask used in
safety decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.real_robot_adapter import RGBDFrame, depth_to_world_points, load_camera_calibrations, voxel_downsample_points
from real_scripts.reconstruct_realsense_pointcloud import render_topdown_camera_points, save_interactive_pointcloud_html, save_ply_ascii


DEFAULT_INPUT = REPO_ROOT / "outputs" / "ur7e_d435i_lingbot_fused_current"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "ur7e_d435i_visual_robot_separation"

# Conservative visible-robot silhouettes verified against the source pair
# captured for this run. Image coordinates are x, y at 1280 x 720.
ROBOT_POLYGONS = {
    "front": (
        ((755, 0), (938, 0), (918, 112), (866, 129), (855, 219), (828, 276), (759, 287), (709, 271), (699, 230), (727, 191), (733, 133), (755, 113)),
        ((695, 220), (797, 210), (852, 238), (871, 321), (845, 355), (718, 357), (683, 327)),
    ),
    "side": (
        ((873, 0), (1280, 0), (1280, 341), (983, 337), (939, 319), (913, 280), (934, 226), (973, 184), (974, 87)),
        ((565, 0), (689, 0), (676, 119), (653, 162), (588, 174), (568, 131)),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration", type=Path, default=REPO_ROOT / "real_scripts" / "ur7e_d435i_camera_calibration.json")
    parser.add_argument("--mask-dilation-pixels", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrations = load_camera_calibrations(args.calibration)
    robot_sets: list[tuple[np.ndarray, np.ndarray]] = []
    environment_sets: list[tuple[np.ndarray, np.ndarray]] = []

    for name in ("front", "side"):
        rgb = np.asarray(Image.open(input_dir / f"{name}_rgb.png").convert("RGB"), dtype=np.uint8)
        depth = np.load(input_dir / f"{name}_lingbot_depth_m.npy").astype(np.float32)
        mask = np.zeros(depth.shape, dtype=np.uint8)
        for polygon in ROBOT_POLYGONS[name]:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        width = 2 * int(args.mask_dilation_pixels) + 1
        mask = cv2.dilate(mask, np.ones((width, width), dtype=np.uint8), iterations=1).astype(bool)
        frame = RGBDFrame(name, rgb, depth)
        robot_sets.append(depth_to_world_points(frame, calibrations[name], stride=2, max_depth=2.5, keep_mask=mask))
        environment_sets.append(depth_to_world_points(frame, calibrations[name], stride=2, max_depth=2.5, keep_mask=~mask))
        overlay = rgb.copy()
        overlay[mask] = (0.40 * overlay[mask] + 0.60 * np.asarray([255, 0, 0])).astype(np.uint8)
        Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / f"{name}_robot_candidate_mask.png")
        Image.fromarray(overlay).save(output_dir / f"{name}_robot_candidate_overlay.png")

    def fuse(sets: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        return voxel_downsample_points(np.concatenate([item[0] for item in sets]), np.concatenate([item[1] for item in sets]), voxel_size=0.005)

    robot_points, robot_colors = fuse(robot_sets)
    environment_points, environment_colors = fuse(environment_sets)
    for label, points, colors in (("ur7e_visual_candidate", robot_points, robot_colors), ("environment_visual_candidate", environment_points, environment_colors)):
        np.savez_compressed(output_dir / f"{label}.npz", points=points, colors=colors, coordinate_frame=np.asarray("ur_base"))
        save_ply_ascii(output_dir / f"{label}.ply", points, colors)
        save_interactive_pointcloud_html(output_dir / f"{label}_viewer.html", points, colors, title=f"{label} (UR base)", max_points=100000)
        Image.fromarray(render_topdown_camera_points(points, colors, image_size=800, margin_m=0.05)).save(output_dir / f"{label}_topdown.png")
    summary = {
        "coordinate_frame": "ur_base",
        "method": "manual visible-UR7e image silhouettes, calibrated LingBot-depth back-projection, then 5 mm two-view voxel fusion",
        "safety_status": "visual candidate only; use current joint angles plus FK/URDF mesh projection for a safety-grade mask",
        "robot_candidate_point_count": int(robot_points.shape[0]),
        "environment_candidate_point_count": int(environment_points.shape[0]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
