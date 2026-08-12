#!/usr/bin/env python3
"""Cluster tabletop objects and draw upright OBBs from a filtered UR-base cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.demo_record_ur7e_safety_overlay_video import _estimate_upright_obb, project_world_points_to_pixels
from real_scripts.real_robot_adapter import load_camera_calibrations
from real_scripts.reconstruct_realsense_pointcloud import (
    _cluster_points_3d,
    estimate_dominant_plane,
    render_topdown_camera_points,
    save_interactive_pointcloud_html,
    save_obbs_json,
    save_ply_ascii,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="environment_without_ur7e.npz in the UR base frame")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=REPO_ROOT / "real_scripts" / "ur7e_d435i_camera_calibration.json")
    parser.add_argument("--source-rgb-dir", type=Path, default=None, help="Directory containing front_rgb.png and side_rgb.png for OBB overlays")
    parser.add_argument("--plane-threshold-m", type=float, default=0.012)
    parser.add_argument("--min-height-m", type=float, default=0.020)
    parser.add_argument("--max-height-m", type=float, default=0.35)
    parser.add_argument(
        "--outlier-neighbor-distance-m",
        type=float,
        default=0.020,
        help="Drop a candidate point when its nearest distinct neighbour is farther than this value; set <= 0 to disable.",
    )
    parser.add_argument("--cluster-radius-m", type=float, default=0.060)
    parser.add_argument("--min-cluster-points", type=int, default=100)
    parser.add_argument(
        "--table-attachment-distance-m",
        type=float,
        default=0.050,
        help="A cluster is an obstacle only if its minimum signed height above the fitted tabletop is no greater than this value.",
    )
    parser.add_argument("--box-margin-m", type=float, default=0.008)
    return parser.parse_args()


def _expanded_obb(obb, margin_m: float):
    """Return the same upright OBB with a uniform collision/display margin."""
    from dataclasses import replace

    margin = max(0.0, float(margin_m))
    extents = np.asarray(obb.extents, dtype=np.float32) + 2.0 * margin
    signs = np.asarray(((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)), dtype=np.float32)
    corners = np.asarray(obb.center, dtype=np.float32)[None, :] + (0.5 * signs * extents[None, :]) @ np.asarray(obb.rotation, dtype=np.float32).T
    return replace(obb, extents=extents, corners=corners.astype(np.float32))


def _remove_isolated_points(points: np.ndarray, colors: np.ndarray, *, max_neighbour_distance_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove one-point depth speckles by nearest *distinct* point distance."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) < 2 or float(max_neighbour_distance_m) <= 0.0:
        return points, colors, np.ones(len(points), dtype=bool)
    distances, _ = cKDTree(points).query(points, k=2, workers=-1)
    keep = distances[:, 1] <= float(max_neighbour_distance_m)
    return points[keep], colors[keep], keep


def _overlay(rgb: np.ndarray, obbs, calibration) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    height, width = rgb.shape[:2]
    for index, obb in enumerate(obbs, start=1):
        uv, _, valid = project_world_points_to_pixels(obb.corners, calibration, width=width, height=height)
        for a, b in edges:
            if valid[a] and valid[b]:
                draw.line((tuple(uv[a]), tuple(uv[b])), fill=(50, 255, 90), width=2)
        center_uv, _, center_valid = project_world_points_to_pixels(np.asarray(obb.center)[None, :], calibration, width=width, height=height)
        if center_valid[0]:
            draw.text(tuple(center_uv[0]), f"#{index}", fill=(255, 255, 0), stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    payload = np.load(args.input)
    points = np.asarray(payload["points"], dtype=np.float32)
    colors = np.asarray(payload["colors"], dtype=np.uint8)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The environment cloud includes room/background surfaces.  The dominant
    # horizontal plane is estimated only near the tabletop's expected base
    # region, then height is evaluated against that fitted plane everywhere.
    candidate = points[(points[:, 0] > -0.2) & (points[:, 0] < 1.0) & (points[:, 1] > -0.8) & (points[:, 1] < 0.4) & (points[:, 2] > -0.10) & (points[:, 2] < 0.12)]
    normal, offset, inliers = estimate_dominant_plane(candidate, threshold=args.plane_threshold_m, ransac_iterations=300)
    if normal[2] < 0.0:
        normal, offset = -normal, -offset
    signed_height = points.astype(np.float64) @ normal + offset
    tabletop = (signed_height >= float(args.min_height_m)) & (signed_height <= float(args.max_height_m))
    # Work area limits reject room/background surfaces while retaining the full visible tabletop.
    tabletop &= (points[:, 0] > -0.2) & (points[:, 0] < 1.0) & (points[:, 1] > -0.8) & (points[:, 1] < 0.4)
    candidate_points, candidate_colors = points[tabletop], colors[tabletop]
    object_points, object_colors, outlier_keep = _remove_isolated_points(
        candidate_points, candidate_colors, max_neighbour_distance_m=args.outlier_neighbor_distance_m,
    )
    # A 3-D connected component is important here: XY-only grouping would
    # incorrectly join a suspended item to the tabletop object beneath it.
    object_heights = signed_height[tabletop][outlier_keep]
    obstacle_components: list[np.ndarray] = []
    floating_components: list[np.ndarray] = []
    for indices in _cluster_points_3d(object_points, cluster_radius=args.cluster_radius_m, min_cluster_points=args.min_cluster_points):
        if float(object_heights[indices].min()) <= float(args.table_attachment_distance_m):
            obstacle_components.append(indices)
        else:
            floating_components.append(indices)
    obstacle_points = np.concatenate([object_points[item] for item in obstacle_components]) if obstacle_components else np.zeros((0, 3), dtype=np.float32)
    obstacle_colors = np.concatenate([object_colors[item] for item in obstacle_components]) if obstacle_components else np.zeros((0, 3), dtype=np.uint8)
    obbs = []
    for indices in obstacle_components:
        fitted = _estimate_upright_obb(object_points[indices])
        if fitted is not None:
            obbs.append(_expanded_obb(fitted, args.box_margin_m))
    # Stable label ordering: left-to-right in the UR base X direction.
    obbs.sort(key=lambda item: (float(item.center[0]), float(item.center[1])))

    np.savez_compressed(output_dir / "tabletop_object_points.npz", points=obstacle_points, colors=obstacle_colors, coordinate_frame=np.asarray("ur_base"))
    save_ply_ascii(output_dir / "tabletop_object_points.ply", obstacle_points, obstacle_colors)
    save_interactive_pointcloud_html(output_dir / "tabletop_objects_with_obbs_viewer.html", obstacle_points, obstacle_colors, title="Table-connected tabletop obstacles and OBBs (UR base)", max_points=100000, obbs=obbs)
    Image.fromarray(render_topdown_camera_points(obstacle_points, obstacle_colors, image_size=800, margin_m=0.05)).save(output_dir / "tabletop_object_points_topdown.png")
    save_obbs_json(output_dir / "tabletop_object_obbs.json", obbs, plane={"normal": normal.astype(float).tolist(), "offset": float(offset), "inlier_count": int(inliers.sum()), "min_height_m": float(args.min_height_m), "max_height_m": float(args.max_height_m), "table_attachment_distance_m": float(args.table_attachment_distance_m), "box_margin_m": float(args.box_margin_m)})
    floating = [{"point_count": int(len(indices)), "minimum_height_m": float(object_heights[indices].min()), "maximum_height_m": float(object_heights[indices].max())} for indices in floating_components]
    summary = {"coordinate_frame": "ur_base", "input": str(args.input), "plane_normal": normal.astype(float).tolist(), "plane_offset": float(offset), "table_height_at_origin_m": float(-offset / normal[2]), "tabletop_candidate_point_count": int(len(candidate_points)), "isolated_points_removed": int((~outlier_keep).sum()), "outlier_neighbour_distance_m": float(args.outlier_neighbor_distance_m), "table_attachment_distance_m": float(args.table_attachment_distance_m), "table_connected_obstacle_point_count": int(len(obstacle_points)), "cluster_count": int(len(obbs)), "floating_cluster_count": int(len(floating)), "floating_clusters": floating, "clusters": [{"id": i, "center_m": obb.center.astype(float).tolist(), "extents_m": obb.extents.astype(float).tolist(), "point_count": int(obb.point_count)} for i, obb in enumerate(obbs, start=1)]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.source_rgb_dir is not None:
        calibrations = load_camera_calibrations(args.calibration)
        for name in ("front", "side"):
            rgb = np.asarray(Image.open(Path(args.source_rgb_dir) / f"{name}_rgb.png").convert("RGB"), dtype=np.uint8)
            Image.fromarray(_overlay(rgb, obbs, calibrations[name])).save(output_dir / f"{name}_object_obbs_overlay.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
