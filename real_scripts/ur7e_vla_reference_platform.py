"""Hardware bridge using the action semantics of the sibling ``ur7e_inference`` project.

The referenced VLA emits seven values: six joint-position policy targets and a
PiKA gripper target.  This bridge never interprets them as Cartesian deltas.
It provides two fixed scene RGB-D cameras (front/side) plus a wrist RGB-D view
for the VLA input.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from real_scripts.real_robot_adapter import RGBDFrame
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource


DEFAULT_INFERENCE_ROOT = Path(r"C:\Users\15261\Documents\projects\ur7e_inference")


def _load_reference_modules(inference_root: Path):
    source_root = Path(inference_root) / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Expected ur7e_inference source directory: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from ur7e_vla.cameras import resize_with_pad
    from ur7e_vla.config import load_config
    from ur7e_vla.hardware import PikaGripper, UR7e

    return load_config, UR7e, PikaGripper, resize_with_pad


@dataclass(frozen=True)
class ReferencePlatformConfig:
    inference_root: Path
    inference_config: Path
    front_serial: str
    side_serial: str
    wrist_serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    execute: bool = False


class UR7eReferenceVLAPlatform:
    """Read state/images and execute rate-limited *joint-target* policy actions."""

    def __init__(self, config: ReferencePlatformConfig) -> None:
        self.config = config
        load_config, UR7e, PikaGripper, resize_with_pad = _load_reference_modules(config.inference_root)
        self.app_config = load_config(config.inference_config)
        self._resize_with_pad = resize_with_pad
        self.robot = UR7e(self.app_config.robot, execute=bool(config.execute))
        self.gripper = PikaGripper(self.app_config.gripper, execute=bool(config.execute))
        self.camera_source = RealSenseD435iSource(
            cameras=(
                D435iCameraConfig("front", config.front_serial),
                D435iCameraConfig("side", config.side_serial),
                D435iCameraConfig("wrist", config.wrist_serial),
            ),
            width=int(config.width),
            height=int(config.height),
            fps=int(config.fps),
        )
        self._latest_frames: dict[str, RGBDFrame] | None = None

    @property
    def control_hz(self) -> float:
        return float(self.app_config.robot.control_hz)

    @property
    def action_mode(self) -> str:
        return str(self.app_config.robot.action_mode)

    @property
    def policy_name(self) -> str:
        return "mani_real_pi05"

    def start(self) -> None:
        self.robot.connect()
        try:
            self.gripper.connect()
            self.camera_source.start()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        try:
            self.camera_source.stop()
        finally:
            try:
                stop = getattr(self.gripper, "stop", None)
                if callable(stop):
                    stop()
            finally:
                self.robot.stop()

    def observe(self, prompt: str) -> tuple[dict, list[RGBDFrame], np.ndarray, np.ndarray]:
        frames = self.camera_source.read()
        self._latest_frames = frames
        qpos = np.asarray(self.robot.joints(), dtype=np.float32)
        gripper = np.asarray([self.gripper.position()], dtype=np.float32)
        policy = self.app_config.policy
        observation = {
            policy.exterior_image_key: self._resize_with_pad(frames["front"].rgb, policy.image_size),
            policy.wrist_image_key: self._resize_with_pad(frames["wrist"].rgb, policy.image_size),
            policy.joint_state_key: qpos,
            policy.gripper_state_key: gripper,
            "prompt": str(prompt),
        }
        return observation, [frames["front"], frames["side"], frames["wrist"]], qpos, gripper

    def execute_policy_action(self, action: np.ndarray, *, current_qpos: np.ndarray | None = None) -> np.ndarray:
        """Execute one absolute/delta joint policy row after reference-project limits."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"Expected one finite 7-D policy action, got {action.shape}")
        current = np.asarray(self.robot.joints() if current_qpos is None else current_qpos, dtype=np.float64)
        target = np.asarray(self.robot.action_to_target(action, current), dtype=np.float64)
        self.robot.send_target(target)
        self.gripper.apply(action)
        return target.astype(np.float32)


def platform_config_from_args(args) -> ReferencePlatformConfig:
    def serial(name: str) -> str:
        value = getattr(args, f"{name}_serial") or os.environ.get(f"REAL_SENSE_{name.upper()}_SERIAL")
        if not value:
            raise ValueError(f"Set --{name}-serial or REAL_SENSE_{name.upper()}_SERIAL")
        return str(value)

    return ReferencePlatformConfig(
        inference_root=Path(args.inference_root),
        inference_config=Path(args.inference_config),
        front_serial=serial("front"),
        side_serial=serial("side"),
        wrist_serial=serial("wrist"),
        width=int(args.camera_width),
        height=int(args.camera_height),
        fps=int(args.camera_fps),
        execute=bool(args.execute),
    )
