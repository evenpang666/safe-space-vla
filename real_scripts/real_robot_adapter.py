#!/usr/bin/env python3
"""Real-robot adapters and UR7e geometry helpers for SafetyModule collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


UR7E_LINK_NAMES = (
    "base_shoulder",
    "shoulder_upper",
    "upper_forearm",
    "forearm_wrist1",
    "wrist1_wrist2",
    "wrist2_wrist3",
    "gripper_width",
)
UR7E_DH_PARAMETERS = (
    (0.0, np.pi / 2.0, 0.1625),
    (-0.425, 0.0, 0.0),
    (-0.3922, 0.0, 0.0),
    (0.0, np.pi / 2.0, 0.1333),
    (0.0, -np.pi / 2.0, 0.0997),
    (0.0, 0.0, 0.0996),
)
DEFAULT_RGBD_CAMERA_NAMES = ("front", "side", "wrist")


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    intrinsics: np.ndarray
    camera_to_world: np.ndarray

    def __post_init__(self) -> None:
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        camera_to_world = np.asarray(self.camera_to_world, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}")
        if camera_to_world.shape != (4, 4):
            raise ValueError(f"camera_to_world must have shape (4, 4), got {camera_to_world.shape}")
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", camera_to_world)


@dataclass(frozen=True)
class RGBDFrame:
    camera_name: str
    rgb: np.ndarray
    depth_m: np.ndarray

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        depth_m = np.asarray(self.depth_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
        if depth_m.shape != rgb.shape[:2]:
            raise ValueError(f"depth_m shape {depth_m.shape} must match rgb height/width {rgb.shape[:2]}")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth_m)


@dataclass(frozen=True)
class FusedPointCloud:
    scene_points: np.ndarray
    scene_colors: np.ndarray
    environment_points: np.ndarray
    environment_colors: np.ndarray


class RealRobotAdapter(Protocol):
    """Interface expected by the online collector.

    Projects should implement this protocol for their robot SDK. The collector
    intentionally keeps this boundary small: observations for PI05, qpos for
    UR FK, three D435i RGB-D frames by default, and action execution.
    """

    def reset(self) -> None: ...

    def get_observation(self) -> dict: ...

    def get_rgbd_frames(self) -> list[RGBDFrame]: ...

    def execute_action(self, action: np.ndarray) -> None: ...

    def is_done(self) -> bool: ...

    def close(self) -> None: ...


def transform_from_dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    return np.asarray(
        [
            [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
            [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
            [0.0, s_alpha, c_alpha, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class UR7ELinkPointSampler:
    """Fixed-topology UR7e link-point sampler based on official UR7e DH FK."""

    def __init__(
        self,
        *,
        points_per_link: int,
        base_to_world: np.ndarray | None = None,
        gripper_width: float = 0.085,
    ):
        if int(points_per_link) < 2:
            raise ValueError("points_per_link must be >= 2")
        if float(gripper_width) <= 0.0:
            raise ValueError("gripper_width must be > 0")
        self.points_per_link = int(points_per_link)
        self.base_to_world = np.eye(4, dtype=np.float64) if base_to_world is None else np.asarray(base_to_world, dtype=np.float64)
        if self.base_to_world.shape != (4, 4):
            raise ValueError(f"base_to_world must have shape (4, 4), got {self.base_to_world.shape}")
        self.gripper_width = float(gripper_width)
        self.link_names = UR7E_LINK_NAMES

    def forward_kinematics(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.size < 6:
            raise ValueError(f"UR7e qpos must contain at least 6 joints, got {q.size}")
        q = q[:6]
        dh_rows = tuple((*params, theta) for params, theta in zip(UR7E_DH_PARAMETERS, q))

        transforms = [np.asarray(self.base_to_world, dtype=np.float64)]
        current = np.asarray(self.base_to_world, dtype=np.float64).copy()
        for a, alpha, d, theta in dh_rows:
            current = current @ transform_from_dh(a, alpha, d, theta)
            transforms.append(current.copy())
        anchors = np.stack([transform[:3, 3] for transform in transforms], axis=0)
        return anchors.astype(np.float64), transforms[-1][:3, :3].astype(np.float64)

    def link_segments(self, qpos: np.ndarray) -> np.ndarray:
        anchors, eef_rotation = self.forward_kinematics(qpos)
        segments = np.empty((7, 2, 3), dtype=np.float64)
        for link_idx in range(6):
            segments[link_idx, 0] = anchors[link_idx]
            segments[link_idx, 1] = anchors[link_idx + 1]

        gripper_axis = eef_rotation[:, 0]
        norm = float(np.linalg.norm(gripper_axis))
        if norm <= 1e-8:
            raise RuntimeError("UR7e gripper x axis has near-zero norm")
        gripper_axis = gripper_axis / norm
        half_width = 0.5 * self.gripper_width
        eef = anchors[-1]
        segments[6, 0] = eef - half_width * gripper_axis
        segments[6, 1] = eef + half_width * gripper_axis
        return segments.astype(np.float32)

    def link_points(self, qpos: np.ndarray) -> np.ndarray:
        segments = self.link_segments(qpos).astype(np.float64)
        u = np.linspace(0.0, 1.0, self.points_per_link, dtype=np.float64)
        start = segments[:, 0, :]
        end = segments[:, 1, :]
        points = (1.0 - u[None, :, None]) * start[:, None, :]
        points += u[None, :, None] * end[:, None, :]
        return points.astype(np.float32)


def depth_to_world_points(
    frame: RGBDFrame,
    calibration: CameraCalibration,
    *,
    stride: int = 1,
    max_depth: float | None = None,
    keep_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if frame.camera_name != calibration.name:
        raise ValueError(f"frame camera {frame.camera_name!r} does not match calibration {calibration.name!r}")
    stride = max(int(stride), 1)
    depth = np.asarray(frame.depth_m, dtype=np.float64)
    rgb = np.asarray(frame.rgb, dtype=np.uint8)
    if keep_mask is None:
        mask = np.isfinite(depth) & (depth > 0.0)
    else:
        mask = np.asarray(keep_mask, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    if max_depth is not None:
        mask &= depth <= float(max_depth)
    mask[::stride, ::stride] &= True
    if stride > 1:
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[::stride, ::stride] = True
        mask &= stride_mask

    v, u = np.nonzero(mask)
    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    z = depth[v, u]
    fx = float(calibration.intrinsics[0, 0])
    fy = float(calibration.intrinsics[1, 1])
    cx = float(calibration.intrinsics[0, 2])
    cy = float(calibration.intrinsics[1, 2])
    if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        raise ValueError("camera intrinsics fx/fy must be non-zero")

    camera_points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    homogeneous = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1), dtype=np.float64)], axis=1)
    world_points = (calibration.camera_to_world @ homogeneous.T).T[:, :3]
    colors = rgb[v, u]
    return world_points.astype(np.float32), colors.astype(np.uint8)


def crop_workspace(
    points: np.ndarray,
    colors: np.ndarray,
    bounds: Iterable[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if bounds is None or points.size == 0:
        return points, colors
    xmin, xmax, ymin, ymax, zmin, zmax = [float(item) for item in bounds]
    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )
    return points[mask], colors[mask]


def filter_robot_points(
    scene_points: np.ndarray,
    robot_link_points: np.ndarray,
    *,
    radius: float,
    chunk_size: int = 16384,
) -> np.ndarray:
    scene_points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
    robot_points = np.asarray(robot_link_points, dtype=np.float32).reshape(-1, 3)
    if scene_points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if robot_points.shape[0] == 0 or float(radius) <= 0.0:
        return np.ones((scene_points.shape[0],), dtype=bool)

    keep = np.ones((scene_points.shape[0],), dtype=bool)
    radius_sq = float(radius) ** 2
    for start in range(0, scene_points.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), scene_points.shape[0])
        diff = scene_points[start:stop, None, :] - robot_points[None, :, :]
        min_dist_sq = np.min(np.sum(diff * diff, axis=-1), axis=1)
        keep[start:stop] = min_dist_sq > radius_sq
    return keep


def fuse_rgbd_frames(
    frames: list[RGBDFrame],
    calibrations: dict[str, CameraCalibration],
    *,
    robot_link_points: np.ndarray,
    stride: int = 1,
    max_depth: float | None = None,
    robot_filter_radius: float = 0.04,
    workspace_bounds: Iterable[float] | None = None,
) -> FusedPointCloud:
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    for frame in frames:
        if frame.camera_name not in calibrations:
            raise KeyError(f"Missing calibration for camera {frame.camera_name!r}")
        points, colors = depth_to_world_points(
            frame,
            calibrations[frame.camera_name],
            stride=stride,
            max_depth=max_depth,
        )
        if len(points) > 0:
            point_sets.append(points)
            color_sets.append(colors)

    if not point_sets:
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_colors = np.zeros((0, 3), dtype=np.uint8)
        return FusedPointCloud(empty_points, empty_colors, empty_points, empty_colors)

    scene_points = np.concatenate(point_sets, axis=0).astype(np.float32)
    scene_colors = np.concatenate(color_sets, axis=0).astype(np.uint8)
    scene_points, scene_colors = crop_workspace(scene_points, scene_colors, workspace_bounds)
    keep = filter_robot_points(scene_points, robot_link_points, radius=robot_filter_radius)
    return FusedPointCloud(
        scene_points=scene_points,
        scene_colors=scene_colors,
        environment_points=scene_points[keep].astype(np.float32),
        environment_colors=scene_colors[keep].astype(np.uint8),
    )


def load_camera_calibrations(path: Path) -> dict[str, CameraCalibration]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    camera_payloads = payload["cameras"] if isinstance(payload, dict) and "cameras" in payload else payload
    calibrations = {}
    for name, item in camera_payloads.items():
        calibrations[str(name)] = CameraCalibration(
            name=str(name),
            intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
            camera_to_world=np.asarray(item["camera_to_world"], dtype=np.float64),
        )
    return calibrations


class ReplayJsonlAdapter:
    """Offline adapter for testing collector wiring with recorded JSONL frames.

    Each line must contain qpos plus image paths or inline lists. This adapter
    is intentionally simple; production deployments should implement
    RealRobotAdapter against the robot and camera SDKs directly.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"Replay file has no records: {self.path}")
        self.index = 0
        self.last_action = None

    def reset(self) -> None:
        self.index = 0
        self.last_action = None

    def get_observation(self) -> dict:
        record = self.records[min(self.index, len(self.records) - 1)]
        wrist_rgb = record.get("wrist_rgb", record["front_rgb"])
        return {
            "front_rgb": np.asarray(record["front_rgb"], dtype=np.uint8),
            "side_rgb": np.asarray(record["side_rgb"], dtype=np.uint8),
            "wrist_rgb": np.asarray(wrist_rgb, dtype=np.uint8),
            "qpos": np.asarray(record["qpos"], dtype=np.float32),
            "gripper": np.asarray(record.get("gripper", [0.0]), dtype=np.float32),
        }

    def get_rgbd_frames(self) -> list[RGBDFrame]:
        record = self.records[min(self.index, len(self.records) - 1)]
        frames = [
            RGBDFrame("front", np.asarray(record["front_rgb"], dtype=np.uint8), np.asarray(record["front_depth_m"], dtype=np.float32)),
            RGBDFrame("side", np.asarray(record["side_rgb"], dtype=np.uint8), np.asarray(record["side_depth_m"], dtype=np.float32)),
        ]
        if "wrist_depth_m" in record:
            frames.append(
                RGBDFrame(
                    "wrist",
                    np.asarray(record.get("wrist_rgb", record["front_rgb"]), dtype=np.uint8),
                    np.asarray(record["wrist_depth_m"], dtype=np.float32),
                )
            )
        return frames

    def execute_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32)
        self.index = min(self.index + 1, len(self.records))

    def is_done(self) -> bool:
        return self.index >= len(self.records)

    def close(self) -> None:
        return None
