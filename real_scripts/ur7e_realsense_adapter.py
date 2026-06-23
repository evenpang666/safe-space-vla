#!/usr/bin/env python3
"""UR7e + three Intel RealSense D435i adapter for real SafetyModule scripts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol, Sequence

import numpy as np

from real_scripts.real_robot_adapter import DEFAULT_RGBD_CAMERA_NAMES, RGBDFrame
from real_scripts.ur7e_controller import ROBOT_IP, UR7eVectorController


DEFAULT_D435I_CAMERA_NAMES = DEFAULT_RGBD_CAMERA_NAMES


class UR7eControllerLike(Protocol):
    def connect(self) -> None: ...

    def close(self) -> None: ...

    def get_current_joints(self) -> Sequence[float]: ...

    def get_gripper_open_ratio(self) -> float: ...

    def send_ee_delta_vector(
        self,
        delta7,
        acceleration: float = 0.18,
        velocity: float = 0.04,
        wait_after_arm_s: float = 0.2,
    ): ...


class RGBDSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self) -> dict[str, tuple[np.ndarray, np.ndarray]]: ...


@dataclass(frozen=True)
class D435iCameraConfig:
    name: str
    serial: str | None = None


class RealSenseD435iSource:
    """Small pyrealsense2 wrapper returning RGB uint8 and depth in meters."""

    def __init__(
        self,
        *,
        cameras: Sequence[D435iCameraConfig],
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.cameras = tuple(cameras)
        if not self.cameras:
            raise ValueError("At least one RealSense camera config is required")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._rs = None
        self._pipelines = []
        self._aligns = []

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError("Missing dependency 'pyrealsense2' for Intel RealSense D435i cameras.") from exc

        self._rs = rs
        self._pipelines = []
        self._aligns = []
        for camera in self.cameras:
            pipeline = rs.pipeline()
            config = rs.config()
            if camera.serial:
                config.enable_device(str(camera.serial))
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            pipeline.start(config)
            self._pipelines.append((camera.name, pipeline))
            self._aligns.append(rs.align(rs.stream.color))

    def stop(self) -> None:
        for _, pipeline in self._pipelines:
            pipeline.stop()
        self._pipelines = []
        self._aligns = []

    def read(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self._rs is None or not self._pipelines:
            raise RuntimeError("RealSenseD435iSource is not started")
        frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for (name, pipeline), align in zip(self._pipelines, self._aligns):
            aligned = align.process(pipeline.wait_for_frames())
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError(f"Failed to read synchronized RGB-D frame from camera {name!r}")
            rgb = np.ascontiguousarray(np.asarray(color_frame.get_data(), dtype=np.uint8))
            depth_scale = float(depth_frame.get_units())
            depth_m = np.asarray(depth_frame.get_data(), dtype=np.float32) * depth_scale
            frames[name] = (rgb, depth_m.astype(np.float32))
        return frames


class UR7eRealSenseAdapter:
    """RealRobotAdapter implementation using UR7eVectorController and D435i RGB-D."""

    def __init__(
        self,
        *,
        controller: UR7eControllerLike,
        camera_source: RGBDSource,
        acceleration: float = 0.18,
        velocity: float = 0.04,
        wait_after_arm_s: float = 0.2,
    ) -> None:
        self.controller = controller
        self.camera_source = camera_source
        self.acceleration = float(acceleration)
        self.velocity = float(velocity)
        self.wait_after_arm_s = float(wait_after_arm_s)
        self._closed = False
        self._latest_frames: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def reset(self) -> None:
        self.controller.connect()
        self.camera_source.start()
        self._closed = False
        self._latest_frames = None

    def _read_frames(self, *, refresh: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if refresh or self._latest_frames is None:
            self._latest_frames = self.camera_source.read()
        return self._latest_frames

    def get_observation(self) -> dict:
        frames = self._read_frames(refresh=True)
        qpos = np.asarray(self.controller.get_current_joints(), dtype=np.float32)
        if qpos.size != 6:
            raise RuntimeError(f"UR controller returned {qpos.size} joints; expected 6")
        gripper = np.asarray([self.controller.get_gripper_open_ratio()], dtype=np.float32)
        observation = {
            "qpos": qpos,
            "gripper": gripper,
        }
        for name in DEFAULT_D435I_CAMERA_NAMES:
            if name not in frames:
                raise KeyError(f"Missing D435i camera frame {name!r}")
            observation[f"{name}_rgb"] = np.ascontiguousarray(frames[name][0].astype(np.uint8))
        return observation

    def get_rgbd_frames(self) -> list[RGBDFrame]:
        frames = self._read_frames()
        rgbd_frames = []
        for name in DEFAULT_D435I_CAMERA_NAMES:
            if name not in frames:
                raise KeyError(f"Missing D435i camera frame {name!r}")
            rgb, depth_m = frames[name]
            rgbd_frames.append(RGBDFrame(name, np.asarray(rgb, dtype=np.uint8), np.asarray(depth_m, dtype=np.float32)))
        return rgbd_frames

    def execute_action(self, action: np.ndarray) -> None:
        action7 = np.asarray(action, dtype=np.float32).reshape(-1)
        if action7.size != 7:
            raise ValueError(f"UR7e action must be 7D [dx_mm,dy_mm,dz_mm,droll,dpitch,dyaw,g], got {action7.size}")
        self.controller.send_ee_delta_vector(
            action7.tolist(),
            acceleration=self.acceleration,
            velocity=self.velocity,
            wait_after_arm_s=self.wait_after_arm_s,
        )
        self._latest_frames = None

    def is_done(self) -> bool:
        return False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.camera_source.stop()
        finally:
            self.controller.close()
            self._closed = True


def _serials_from_env() -> dict[str, str | None]:
    return {
        "front": os.environ.get("REAL_SENSE_FRONT_SERIAL"),
        "side": os.environ.get("REAL_SENSE_SIDE_SERIAL"),
        "wrist": os.environ.get("REAL_SENSE_WRIST_SERIAL"),
    }


def create_adapter() -> UR7eRealSenseAdapter:
    """Factory for --adapter real_scripts.ur7e_realsense_adapter:create_adapter."""
    serials = _serials_from_env()
    cameras = [D435iCameraConfig(name=name, serial=serials.get(name)) for name in DEFAULT_D435I_CAMERA_NAMES]
    controller = UR7eVectorController(
        robot_ip=os.environ.get("UR_ROBOT_IP", ROBOT_IP),
        strict_gripper_connection=os.environ.get("UR_STRICT_GRIPPER", "1") not in {"0", "false", "False"},
        robotiq_urscript_defs_path=os.environ.get("ROBOTIQ_URSCRIPT_DEFS") or None,
    )
    camera_source = RealSenseD435iSource(
        cameras=cameras,
        width=int(os.environ.get("REAL_SENSE_WIDTH", "640")),
        height=int(os.environ.get("REAL_SENSE_HEIGHT", "480")),
        fps=int(os.environ.get("REAL_SENSE_FPS", "30")),
    )
    return UR7eRealSenseAdapter(
        controller=controller,
        camera_source=camera_source,
        acceleration=float(os.environ.get("UR_ACTION_ACCELERATION", "0.18")),
        velocity=float(os.environ.get("UR_ACTION_VELOCITY", "0.04")),
        wait_after_arm_s=float(os.environ.get("UR_WAIT_AFTER_ARM_S", "0.2")),
    )
