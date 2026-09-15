"""Fixed-identity collision-surface points for the current dual-UR7e cell."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from real_scripts.robotiq_vendor_visual_models import (
    TWO_F85_LINK_NAMES,
    robotiq_2f85_points_in_active_tcp,
    robotiq_epick_points_in_active_tcp,
)
from real_scripts.tool_urdf_collision import sample_collision_surface
from real_scripts.ur7e_collision_mesh import UR7eCollisionSurfacePointSampler, flange_transform
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path


UR_NAMES = ("base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3")


def _transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    flat = values.reshape(-1, 3)
    result = (transform[:3, :3] @ flat.T).T + transform[:3, 3]
    return result.reshape(values.shape).astype(np.float32)


def _matrix(value: object, *, name: str) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("matrix")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all() or not np.allclose(result[3], (0, 0, 0, 1)):
        raise ValueError(f"{name} must be a finite homogeneous 4x4 matrix")
    return result


def _right_base_to_left_base(hardware_path: Path) -> np.ndarray:
    with hardware_path.open(encoding="utf-8") as handle:
        hardware = yaml.safe_load(handle)
    left_path = (hardware_path.parent / hardware["arms"]["left_arm"]["front_calibration"]).resolve()
    right_path = (hardware_path.parent / hardware["arms"]["right_arm"]["front_calibration"]).resolve()
    with left_path.open(encoding="utf-8") as handle:
        left_to_camera = _matrix(yaml.safe_load(handle)["matrix"], name=str(left_path))
    with right_path.open(encoding="utf-8") as handle:
        right_to_camera = _matrix(yaml.safe_load(handle)["matrix"], name=str(right_path))
    return left_to_camera @ np.linalg.inv(right_to_camera)


class DualUR7eSurfacePointSampler:
    """Generate stable UR7e + 2F-85/EPick collision points in ``left_base``.

    A six-joint input emits the recorded left arm only.  A twelve-joint input
    emits both arms in configured order.  This prevents single-arm Quest3
    episodes from inventing a stationary right-arm target.
    """

    point_identity_version = "dual_ur7e_2f85_epick_collision_surface_v1"

    def __init__(self, *, points_per_link: int = 128, project_config: Path | str = DEFAULT_PROJECT_CONFIG) -> None:
        if int(points_per_link) < 2:
            raise ValueError("points_per_link must be at least two")
        self.points_per_link = int(points_per_link)
        self.config, self.config_path = load_project_config(project_config)
        robot = self.config["robot"]
        self.tcp_transform_verified = {
            arm: bool(robot["flange_to_active_tcp"][arm].get("verified", False))
            for arm in ("left_arm", "right_arm")
        }
        self.flange_to_tcp = {
            arm: _matrix(robot["flange_to_active_tcp"][arm], name=f"flange_to_active_tcp.{arm}")
            for arm in ("left_arm", "right_arm")
        }
        hardware_path = resolve_repo_path(self.config["quest3"]["hardware_config"])
        self.right_base_to_left_base = _right_base_to_left_base(hardware_path)
        self.ur = UR7eCollisionSurfacePointSampler(points_per_link=self.points_per_link)
        self.left_link_names = tuple(f"left_ur7e_{name}" for name in UR_NAMES) + tuple(
            f"left_{name}" for name in TWO_F85_LINK_NAMES
        )
        self.right_link_names = tuple(f"right_ur7e_{name}" for name in UR_NAMES) + ("right_robotiq_epick",)
        epick_urdf = Path(__file__).resolve().parents[1] / "assets" / "robot_models" / "robotiq_epick" / "urdf" / "robotiq_epick_active_tcp_collision.urdf"
        epick_cloud = np.concatenate(
            (
                robotiq_epick_points_in_active_tcp(samples=max(self.points_per_link * 4, 512)),
                sample_collision_surface(epick_urdf, margin_m=0.0, resolution_m=0.005),
            )
        )
        indices = np.linspace(0, len(epick_cloud) - 1, self.points_per_link, dtype=np.int64)
        self.epick_local = np.asarray(epick_cloud[indices], dtype=np.float32)
        left_hasher = hashlib.sha256()
        left_hasher.update(self.point_identity_version.encode())
        left_hasher.update(self.ur.mesh_model_hash.encode())
        left_hasher.update(self.flange_to_tcp["left_arm"].tobytes())
        left_hasher.update(
            robotiq_2f85_points_in_active_tcp(finger_angle_rad=0.0, samples_per_link=self.points_per_link).tobytes()
        )
        self.left_surface_model_hash = left_hasher.hexdigest()
        dual_hasher = hashlib.sha256()
        dual_hasher.update(self.left_surface_model_hash.encode())
        dual_hasher.update(self.ur.mesh_model_hash.encode())
        dual_hasher.update(self.flange_to_tcp["right_arm"].tobytes())
        dual_hasher.update(self.right_base_to_left_base.tobytes())
        dual_hasher.update(self.epick_local.tobytes())
        self.dual_surface_model_hash = dual_hasher.hexdigest()

    @staticmethod
    def arm_count(qpos: np.ndarray) -> int:
        size = np.asarray(qpos).reshape(-1).size
        if size not in (6, 12):
            raise ValueError(f"Expected 6 or 12 Quest3 joint values, got {size}")
        return size // 6

    def link_names(self, arm_count: int) -> tuple[str, ...]:
        if arm_count == 1:
            return self.left_link_names
        if arm_count == 2:
            return self.left_link_names + self.right_link_names
        raise ValueError("arm_count must be one or two")

    def model_hash(self, arm_count: int) -> str:
        if arm_count == 1:
            return self.left_surface_model_hash
        if arm_count == 2:
            return self.dual_surface_model_hash
        raise ValueError("arm_count must be one or two")

    def link_points(self, qpos: np.ndarray, gripper_position: np.ndarray | None = None) -> np.ndarray:
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        count = self.arm_count(q)
        grip = np.zeros((count,), dtype=np.float64) if gripper_position is None else np.asarray(gripper_position, dtype=np.float64).reshape(-1)
        if grip.size != count:
            raise ValueError(f"Expected {count} gripper positions, got {grip.size}")
        left_q = q[:6]
        left_ur = self.ur.link_points(left_q)
        left_tool_local = robotiq_2f85_points_in_active_tcp(
            finger_angle_rad=float(np.clip(grip[0], 0.0, 1.0) * 0.8),
            samples_per_link=self.points_per_link,
        ).reshape(len(TWO_F85_LINK_NAMES), self.points_per_link, 3)
        left_tool = _transform(left_tool_local, flange_transform(left_q) @ self.flange_to_tcp["left_arm"])
        parts = [left_ur, left_tool]
        if count == 2:
            right_q = q[6:12]
            right_ur = _transform(self.ur.link_points(right_q), self.right_base_to_left_base)
            right_tool_tf = self.right_base_to_left_base @ flange_transform(right_q) @ self.flange_to_tcp["right_arm"]
            right_tool = _transform(self.epick_local, right_tool_tf)[None, ...]
            parts.extend((right_ur, right_tool))
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def point_ids(self, arm_count: int) -> np.ndarray:
        names = self.link_names(arm_count)
        links = np.broadcast_to(np.arange(len(names), dtype=np.int32)[:, None], (len(names), self.points_per_link))
        points = np.broadcast_to(np.arange(self.points_per_link, dtype=np.int32)[None, :], links.shape)
        return np.stack((links, points), axis=-1)
