"""Real RGB-D scene reconstruction and tabletop OBB extraction for safety control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from real_scripts.real_cbf_qp import OrientedBox
from real_scripts.real_robot_adapter import (
    CameraCalibration,
    RGBDFrame,
    crop_workspace,
    depth_to_world_points,
    robot_depth_keep_mask,
    voxel_downsample_points,
)
from real_scripts.ur7e_collision_mesh import (
    collision_volume_keep_mask,
    flange_transform,
    mesh_surface_samples,
    occupied_collision_voxels,
    render_collision_depth,
    render_surface_points_depth,
)


@dataclass(frozen=True)
class RealSceneGeometry:
    environment_points: np.ndarray
    environment_colors: np.ndarray
    boxes: list[OrientedBox]


def _components_xy(points: np.ndarray, *, cell_size: float, min_points: int) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    cells: dict[tuple[int, int], list[int]] = {}
    coords = np.floor(np.asarray(points)[:, :2] / max(float(cell_size), 1e-5)).astype(np.int64)
    for index, (x, y) in enumerate(coords):
        cells.setdefault((int(x), int(y)), []).append(index)
    visited: set[tuple[int, int]] = set()
    groups: list[np.ndarray] = []
    for seed in cells:
        if seed in visited:
            continue
        pending = deque([seed])
        visited.add(seed)
        indices: list[int] = []
        while pending:
            x, y = pending.popleft()
            indices.extend(cells[(x, y)])
            for nx in range(x - 1, x + 2):
                for ny in range(y - 1, y + 2):
                    key = (nx, ny)
                    if key in cells and key not in visited:
                        visited.add(key)
                        pending.append(key)
        if len(indices) >= int(min_points):
            groups.append(np.asarray(indices, dtype=np.int64))
    return groups


def _upright_obb(points: np.ndarray, *, margin_m: float) -> OrientedBox:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    xy_center = points[:, :2].mean(axis=0)
    centered = points[:, :2] - xy_center
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    _eigenvalues, xy_axes = np.linalg.eigh(covariance)
    xy_axes = xy_axes[:, ::-1]
    if np.linalg.det(xy_axes) < 0.0:
        xy_axes[:, 1] *= -1.0
    axes = np.eye(3, dtype=np.float64)
    axes[:2, 0] = xy_axes[:, 0]
    axes[:2, 1] = xy_axes[:, 1]
    local = points @ axes
    low, high = local.min(axis=0), local.max(axis=0)
    half_sizes = 0.5 * (high - low) + float(margin_m)
    center = axes @ (0.5 * (low + high))
    return OrientedBox(center.astype(np.float32), axes.astype(np.float32), np.maximum(half_sizes, 1e-4).astype(np.float32))


class RealSceneOBBBuilder:
    def __init__(
        self,
        *,
        calibrations: dict[str, CameraCalibration],
        sampler,
        camera_names: Iterable[str] = ("front", "side"),
        table_z: float,
        workspace_bounds: Iterable[float],
        max_depth_m: float = 2.5,
        voxel_size_m: float = 0.005,
        min_height_m: float = 0.025,
        max_height_m: float = 0.35,
        cluster_radius_m: float = 0.04,
        min_cluster_points: int = 48,
        box_margin_m: float = 0.008,
    ) -> None:
        self.calibrations = calibrations
        self.sampler = sampler
        self.camera_names = tuple(camera_names)
        self.table_z = float(table_z)
        self.workspace_bounds = tuple(float(value) for value in workspace_bounds)
        self.max_depth_m = float(max_depth_m)
        self.voxel_size_m = float(voxel_size_m)
        self.min_height_m = float(min_height_m)
        self.max_height_m = float(max_height_m)
        self.cluster_radius_m = float(cluster_radius_m)
        self.min_cluster_points = int(min_cluster_points)
        self.box_margin_m = float(box_margin_m)
        self._pika_dense_samples = mesh_surface_samples(self.sampler.pika_mesh, samples_per_face=4)

    def _render_depth(self, qpos: np.ndarray, frame: RGBDFrame) -> np.ndarray:
        calibration = self.calibrations[frame.camera_name]
        ur_depth = render_collision_depth(
            qpos,
            calibration.camera_to_world,
            calibration.intrinsics,
            width=frame.rgb.shape[1],
            height=frame.rgb.shape[0],
            samples_per_face=4,
        )
        pika_to_base = flange_transform(qpos) @ self.sampler.pika_mount_transform
        pika_points = (pika_to_base[:3, :3] @ self._pika_dense_samples.T).T + pika_to_base[:3, 3]
        pika_depth = render_surface_points_depth(
            pika_points,
            calibration.camera_to_world,
            calibration.intrinsics,
            width=frame.rgb.shape[1],
            height=frame.rgb.shape[0],
            splat_radius_pixels=3,
        )
        return np.where((ur_depth > 0.0) & (pika_depth > 0.0), np.minimum(ur_depth, pika_depth), np.maximum(ur_depth, pika_depth))

    def build(self, frames: Iterable[RGBDFrame], qpos: np.ndarray) -> RealSceneGeometry:
        q = np.asarray(qpos, dtype=np.float32).reshape(6)
        pika_to_base = flange_transform(q) @ self.sampler.pika_mount_transform
        occupied, pitch = occupied_collision_voxels(q, extra_meshes=((self.sampler.pika_mesh, pika_to_base),))
        point_sets: list[np.ndarray] = []
        color_sets: list[np.ndarray] = []
        by_name = {frame.camera_name: frame for frame in frames}
        for name in self.camera_names:
            if name not in by_name:
                raise KeyError(f"Missing scene camera frame {name!r}")
            frame = by_name[name]
            robot_depth = self._render_depth(q, frame)
            keep = robot_depth_keep_mask(frame.depth_m, robot_depth, absolute_tolerance_m=0.012, relative_tolerance=0.015, dilation_pixels=2)
            points, colors = depth_to_world_points(frame, self.calibrations[name], stride=2, max_depth=self.max_depth_m, keep_mask=keep)
            if len(points):
                outside_robot = collision_volume_keep_mask(points, occupied, voxel_pitch_m=pitch)
                point_sets.append(points[outside_robot])
                color_sets.append(colors[outside_robot])
        if not point_sets:
            empty_points = np.zeros((0, 3), dtype=np.float32)
            return RealSceneGeometry(empty_points, np.zeros((0, 3), dtype=np.uint8), [])
        points, colors = voxel_downsample_points(np.concatenate(point_sets), np.concatenate(color_sets), voxel_size=self.voxel_size_m)
        points, colors = crop_workspace(points, colors, self.workspace_bounds)
        height = points[:, 2] - self.table_z
        candidates = points[(height >= self.min_height_m) & (height <= self.max_height_m)]
        boxes = [_upright_obb(candidates[index], margin_m=self.box_margin_m) for index in _components_xy(candidates, cell_size=self.cluster_radius_m, min_points=self.min_cluster_points)]
        return RealSceneGeometry(points, colors, boxes)
