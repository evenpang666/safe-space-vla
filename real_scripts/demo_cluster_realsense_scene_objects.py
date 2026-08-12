#!/usr/bin/env python3
"""Cluster a full RealSense scene point cloud into per-object HTML viewers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.reconstruct_realsense_pointcloud import (  # noqa: E402
    _cluster_points_3d,
    filter_camera_bounds,
    save_interactive_pointcloud_html,
    save_ply_ascii,
)


DEFAULT_INPUT_NPZ = REPO_ROOT / "outputs" / "realsense_pointcloud" / "front_pointcloud.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "object_pointcloud_clusters"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT_NPZ, help="Full-scene point cloud .npz with points/colors arrays.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root output directory under which cluster files are written.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subfolder name. Defaults to '<input stem>_clusters'.",
    )
    parser.add_argument("--cluster-radius", type=float, default=0.04, help="3D grid connectivity radius in meters.")
    parser.add_argument("--min-cluster-points", type=int, default=64, help="Drop connected components with fewer points.")
    parser.add_argument("--viewer-max-points", type=int, default=60000, help="Maximum points embedded per cluster HTML viewer.")
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="Optional camera-frame crop before clustering.",
    )
    return parser.parse_args()


def load_pointcloud_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "points" not in data:
        raise KeyError(f"{path} does not contain a 'points' array")
    points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
    if "colors" in data:
        colors = np.asarray(data["colors"], dtype=np.uint8).reshape(-1, 3)
    else:
        colors = np.tile(np.asarray([[180, 180, 180]], dtype=np.uint8), (points.shape[0], 1))
    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"points/colors length mismatch: {points.shape[0]} vs {colors.shape[0]}")
    return points, colors


def _cluster_summary(index: int, points: np.ndarray, *, npz_path: Path, ply_path: Path, html_path: Path) -> dict[str, object]:
    return {
        "index": int(index),
        "point_count": int(points.shape[0]),
        "center": points.mean(axis=0).astype(float).tolist(),
        "min_bound": points.min(axis=0).astype(float).tolist(),
        "max_bound": points.max(axis=0).astype(float).tolist(),
        "npz": npz_path.as_posix(),
        "ply": ply_path.as_posix(),
        "html": html_path.as_posix(),
    }


def cluster_scene_objects(
    input_npz: Path,
    output_dir: Path,
    *,
    run_name: str | None = None,
    cluster_radius: float = 0.04,
    min_cluster_points: int = 64,
    viewer_max_points: int = 60000,
    bounds: tuple[float, float, float, float, float, float] | list[float] | None = None,
) -> dict[str, object]:
    if float(cluster_radius) <= 0.0:
        raise ValueError("cluster_radius must be > 0")
    if int(min_cluster_points) <= 0:
        raise ValueError("min_cluster_points must be > 0")

    input_npz = Path(input_npz)
    output_root = Path(output_dir)
    run_dir = output_root / (run_name or f"{input_npz.stem}_clusters")
    run_dir.mkdir(parents=True, exist_ok=True)

    points, colors = load_pointcloud_npz(input_npz)
    total_point_count = int(points.shape[0])
    points, colors = filter_camera_bounds(points, colors, bounds=bounds)
    filtered_point_count = int(points.shape[0])

    clusters = _cluster_points_3d(points, cluster_radius=cluster_radius, min_cluster_points=min_cluster_points)
    clusters = sorted(clusters, key=lambda indices: (-len(indices), int(indices[0]) if len(indices) else 0))

    cluster_summaries: list[dict[str, object]] = []
    clustered_point_count = 0
    for cluster_index, indices in enumerate(clusters):
        cluster_points = points[indices]
        cluster_colors = colors[indices]
        clustered_point_count += int(cluster_points.shape[0])
        stem = f"cluster_{cluster_index:03d}"
        npz_path = run_dir / f"{stem}_pointcloud.npz"
        ply_path = run_dir / f"{stem}_pointcloud.ply"
        html_path = run_dir / f"{stem}_viewer.html"
        np.savez_compressed(
            npz_path,
            points=cluster_points.astype(np.float32),
            colors=cluster_colors.astype(np.uint8),
            source_indices=indices.astype(np.int64),
            coordinate_frame=np.asarray("camera"),
        )
        save_ply_ascii(ply_path, cluster_points, cluster_colors)
        save_interactive_pointcloud_html(
            html_path,
            cluster_points,
            cluster_colors,
            title=f"{input_npz.stem} cluster {cluster_index:03d}",
            max_points=viewer_max_points,
        )
        cluster_summaries.append(_cluster_summary(cluster_index, cluster_points, npz_path=npz_path, ply_path=ply_path, html_path=html_path))

    summary = {
        "input_npz": input_npz.as_posix(),
        "output_dir": run_dir.as_posix(),
        "cluster_radius": float(cluster_radius),
        "min_cluster_points": int(min_cluster_points),
        "bounds": None if bounds is None else [float(value) for value in bounds],
        "total_point_count": total_point_count,
        "filtered_point_count": filtered_point_count,
        "clustered_point_count": int(clustered_point_count),
        "ignored_point_count": int(filtered_point_count - clustered_point_count),
        "cluster_count": int(len(cluster_summaries)),
        "clusters": cluster_summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = cluster_scene_objects(
        args.input_npz,
        args.output_dir,
        run_name=args.run_name,
        cluster_radius=args.cluster_radius,
        min_cluster_points=args.min_cluster_points,
        viewer_max_points=args.viewer_max_points,
        bounds=args.bounds,
    )
    print(f"[info] wrote {summary['cluster_count']} clusters to {summary['output_dir']}")
    for item in summary["clusters"]:
        print(f"[info] cluster_{item['index']:03d}: {item['point_count']} points -> {item['html']}")
    print(f"[info] ignored points below min cluster size: {summary['ignored_point_count']}")


if __name__ == "__main__":
    main()
