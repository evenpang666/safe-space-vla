#!/usr/bin/env python3
"""Visualize one collected PI05 safety-decoder dataset sample."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "outputs" / "pi05_safety_decoder" / "pi05_libero_task0_decoder_dataset.npz"


@dataclass(frozen=True)
class DatasetSample:
    index: int
    target_link_points: np.ndarray
    prefix_tokens_shape: tuple
    action_chunk_shape: tuple
    start_joint_vector: np.ndarray
    task_id: Optional[int]
    rollout_id: Optional[int]
    step_id: Optional[int]
    link_names: np.ndarray


def default_output_path(dataset: Path, sample_index: int) -> Path:
    return dataset.with_name(f"{dataset.stem}_sample{sample_index:04d}_link_points.png")


def _optional_scalar(data, key: str, index: int) -> Optional[int]:
    if key not in data:
        return None
    values = np.asarray(data[key])
    if values.ndim == 0:
        return int(values)
    return int(values[index])


def _validate_rollout_surface_dataset(data) -> None:
    if "target_source" not in data:
        return
    target_source = str(np.asarray(data["target_source"]))
    if target_source != "rollout_surface":
        raise ValueError(
            "This visualizer only supports rollout_surface point-flow datasets; "
            f"got target_source={target_source!r}"
        )


def load_dataset_sample(path: Path, sample_index: int) -> DatasetSample:
    with np.load(path, allow_pickle=False) as data:
        if "target_link_points" not in data:
            raise KeyError("dataset is missing required array 'target_link_points'")
        _validate_rollout_surface_dataset(data)
        targets = np.asarray(data["target_link_points"], dtype=np.float32)
        if targets.ndim != 5 or targets.shape[-1] != 3:
            raise ValueError(f"target_link_points must have shape (S, T, L, P, 3), got {targets.shape}")
        sample_count = int(targets.shape[0])
        if not -sample_count <= sample_index < sample_count:
            raise IndexError(f"sample index {sample_index} out of range for {sample_count} samples")
        resolved_index = sample_index % sample_count

        prefix_tokens_shape = tuple(np.asarray(data["prefix_tokens"][resolved_index]).shape) if "prefix_tokens" in data else ()
        action_chunk_shape = tuple(np.asarray(data["action_chunks"][resolved_index]).shape) if "action_chunks" in data else ()
        start_joint_vector = (
            np.asarray(data["start_joint_vectors"][resolved_index], dtype=np.float32)
            if "start_joint_vectors" in data
            else np.zeros((0,), dtype=np.float32)
        )
        link_names = np.asarray(data["link_names"]) if "link_names" in data else np.asarray([])

        return DatasetSample(
            index=resolved_index,
            target_link_points=targets[resolved_index],
            prefix_tokens_shape=prefix_tokens_shape,
            action_chunk_shape=action_chunk_shape,
            start_joint_vector=start_joint_vector,
            task_id=_optional_scalar(data, "task_ids", resolved_index),
            rollout_id=_optional_scalar(data, "rollout_ids", resolved_index),
            step_id=_optional_scalar(data, "step_ids", resolved_index),
            link_names=link_names,
        )


def flatten_link_points_for_plot(
    link_points: np.ndarray,
    time_index: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    link_points = np.asarray(link_points, dtype=np.float32)
    if link_points.ndim != 4 or link_points.shape[-1] != 3:
        raise ValueError(f"link_points must have shape (T, L, P, 3), got {link_points.shape}")
    time_count, link_count, points_per_link, _xyz = link_points.shape
    if time_index is not None:
        if not -time_count <= time_index < time_count:
            raise IndexError(f"time index {time_index} out of range for {time_count} steps")
        resolved_time = time_index % time_count
        points = link_points[resolved_time].reshape(-1, 3)
        link_ids = np.repeat(np.arange(link_count, dtype=np.int64), points_per_link)
        step_ids = np.full(points.shape[0], resolved_time, dtype=np.int64)
        return points, link_ids, step_ids

    points = link_points.reshape(-1, 3)
    link_ids = np.tile(np.repeat(np.arange(link_count, dtype=np.int64), points_per_link), time_count)
    step_ids = np.repeat(np.arange(time_count, dtype=np.int64), link_count * points_per_link)
    return points, link_ids, step_ids


def load_safe_space_obbs(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "obstacle_box_corners" not in data:
            raise KeyError("safe-space file is missing required array 'obstacle_box_corners'")
        corners = np.asarray(data["obstacle_box_corners"], dtype=np.float32)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError(f"obstacle_box_corners must have shape (N, 8, 3), got {corners.shape}")
    return corners


def box_faces_from_corners(corners: np.ndarray) -> list[np.ndarray]:
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (8, 3):
        raise ValueError(f"box corners must have shape (8, 3), got {corners.shape}")
    return [
        corners[[0, 1, 2, 3]],
        corners[[4, 5, 6, 7]],
        corners[[0, 1, 5, 4]],
        corners[[1, 2, 6, 5]],
        corners[[2, 3, 7, 6]],
        corners[[3, 0, 4, 7]],
    ]


def set_equal_axes(ax, points: np.ndarray, extra_points: Optional[np.ndarray] = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    if extra_points is not None and len(extra_points) > 0:
        points = np.concatenate([points.reshape(-1, 3), np.asarray(extra_points, dtype=np.float32).reshape(-1, 3)])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def sample_title(sample: DatasetSample, time_index: Optional[int]) -> str:
    time_count, link_count, points_per_link, _xyz = sample.target_link_points.shape
    scope = f"time={time_index % time_count}" if time_index is not None else f"all {time_count} future steps"
    parts = [
        f"sample {sample.index}",
        f"rollout {sample.rollout_id}" if sample.rollout_id is not None else None,
        f"step {sample.step_id}" if sample.step_id is not None else None,
        scope,
        f"L={link_count}, P={points_per_link}",
    ]
    return " | ".join(part for part in parts if part is not None)


def save_sample_plot(
    output: Path,
    sample: DatasetSample,
    *,
    time_index: Optional[int],
    elev: float,
    azim: float,
    point_size: float,
    obb_corners: Optional[np.ndarray] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    points, link_ids, step_ids = flatten_link_points_for_plot(sample.target_link_points, time_index=time_index)
    cmap = plt.get_cmap("tab20")
    colors = cmap((link_ids % 20).astype(np.float64) / 19.0)
    if time_index is None:
        time_count = max(int(step_ids.max()) + 1, 1)
        alphas = 0.25 + 0.65 * (step_ids.astype(np.float64) + 1.0) / time_count
        colors[:, 3] = alphas

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size, linewidths=0)
    if obb_corners is not None and len(obb_corners) > 0:
        for corners in np.asarray(obb_corners, dtype=np.float32):
            poly = Poly3DCollection(
                box_faces_from_corners(corners),
                facecolor=(1.0, 0.35, 0.10, 0.16),
                edgecolor=(0.75, 0.18, 0.05, 0.75),
                linewidths=0.8,
            )
            ax.add_collection3d(poly)
    set_equal_axes(ax, points, extra_points=obb_corners)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    ax.set_title(sample_title(sample, time_index), fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one PI05 safety-decoder dataset sample.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Input decoder dataset .npz.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index to visualize.")
    parser.add_argument("--time-index", type=int, default=None, help="Optional single future time index to plot.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--elev", type=float, default=18.0, help="Matplotlib 3D elevation.")
    parser.add_argument("--azim", type=float, default=0.0, help="Matplotlib 3D azimuth.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Scatter point size.")
    parser.add_argument("--safe-space", type=Path, default=None, help="Optional safe-space .npz containing OBBs.")
    parser.add_argument("--draw-obb", action="store_true", help="Draw obstacle OBBs from --safe-space.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = load_dataset_sample(args.dataset, args.sample_index)
    output = args.output or default_output_path(args.dataset, sample.index)
    obb_corners = load_safe_space_obbs(args.safe_space) if args.draw_obb and args.safe_space is not None else None
    save_sample_plot(
        output,
        sample,
        time_index=args.time_index,
        elev=args.elev,
        azim=args.azim,
        point_size=args.point_size,
        obb_corners=obb_corners,
    )
    print(f"[info] sample index: {sample.index}")
    print(f"[info] target_link_points: {sample.target_link_points.shape}")
    print(f"[info] prefix_tokens: {sample.prefix_tokens_shape}")
    print(f"[info] action_chunk: {sample.action_chunk_shape}")
    print(f"[info] rollout_id: {sample.rollout_id}, step_id: {sample.step_id}, task_id: {sample.task_id}")
    if obb_corners is not None:
        print(f"[info] obstacle OBBs: {len(obb_corners)}")
    print(f"[done] saved visualization: {output}")


if __name__ == "__main__":
    main()
