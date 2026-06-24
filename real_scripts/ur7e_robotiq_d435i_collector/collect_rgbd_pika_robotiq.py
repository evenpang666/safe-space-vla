from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__:
    from .utils.camera_rgbd import MultiRGBDCamera
    from .utils.episode_writer import EpisodeWriter
    from .utils.gripper_adapters import GripperMapping, RobotiqGripperAdapter
    from .utils.keyboard_trigger import (
        CollectorMode,
        CollectorState,
        FunctionKeyListener,
    )
    from .utils.pika_interface import PikaSense, detect_pika_ports
    from .utils.robot_interface import UR7eInterface
    from .utils.teleop_controller import PikaTeleopController
else:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    from utils.camera_rgbd import MultiRGBDCamera
    from utils.episode_writer import EpisodeWriter
    from utils.gripper_adapters import GripperMapping, RobotiqGripperAdapter
    from utils.keyboard_trigger import CollectorMode, CollectorState, FunctionKeyListener
    from utils.pika_interface import PikaSense, detect_pika_ports
    from utils.robot_interface import UR7eInterface
    from utils.teleop_controller import PikaTeleopController


logger = logging.getLogger(__name__)

DEFAULT_UR_ROBOT_IP = "169.254.26.10"

STATE_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "gripper",
]
ACTION_NAMES = STATE_NAMES


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_task(cli_task: str | None, cfg: dict[str, Any]) -> str:
    if cli_task is not None:
        task = cli_task
    else:
        collection = cfg.get("collection", {})
        if not isinstance(collection, dict):
            raise ValueError(
                "Config collection must be a mapping with collection.task, "
                "or pass --task."
            )
        task = collection.get("task", "")

    task = str(task).strip()
    if not task:
        raise ValueError(
            "A non-empty task description is required. "
            "Pass --task or set collection.task."
        )
    return task


def resolve_robot_host(cli_robot_ip: str | None, cfg: dict[str, Any]) -> str:
    robot_cfg = cfg.get("robot") or {}
    if not isinstance(robot_cfg, dict):
        raise ValueError("Config robot must be a mapping with robot.host.")

    candidates = (
        cli_robot_ip,
        robot_cfg.get("host"),
        os.environ.get("UR_ROBOT_IP"),
        DEFAULT_UR_ROBOT_IP,
    )
    for candidate in candidates:
        host = str(candidate or "").strip()
        if host:
            return host

    raise ValueError(
        "UR robot IP is required. Pass --robot-ip, set robot.host, "
        "or set UR_ROBOT_IP."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UR7e + Robotiq + D435i RGB-D collector controlled by PikaSense"
    )
    parser.add_argument("--config", default="configs/ur7e_robotiq_d435i.yaml")
    parser.add_argument("--task", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--sense-port", default=None)
    return parser.parse_args()


def _resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path)
    if config_path.is_absolute() or config_path.exists():
        return config_path

    package_relative = Path(__file__).resolve().parent / config_path
    if package_relative.exists():
        return package_relative

    return config_path


class RGBDCollector:
    def __init__(
        self,
        cfg: dict[str, Any],
        task: str,
        dataset_name: str,
        output_dir: str | Path,
    ):
        self.cfg = cfg
        self.task = task
        self.dataset_name = dataset_name
        self.output_dir = output_dir

        collection_cfg = cfg.get("collection") or {}
        controls_cfg = cfg.get("controls") or {}
        robot_cfg = cfg["robot"]
        sense_cfg = cfg.get("pika_sense") or {}
        gripper_cfg = cfg.get("robotiq_gripper") or cfg.get("gripper") or {}
        teleop_cfg = cfg.get("teleoperation") or {}
        safety_cfg = teleop_cfg.get("safety") or {}

        self.fps = int(collection_cfg.get("fps", 30))
        self.state = CollectorState(
            save_on_teleop_stop=bool(controls_cfg.get("save_on_teleop_stop", True))
        )
        self._lock = threading.Lock()
        self._pending_events: list[str] = []
        self._active_writer_episode = False
        self._episode_start_s: float | None = None
        self._last_camera_warning_s = 0.0
        self._listener: FunctionKeyListener | None = None
        self._robot_connected = False
        self._sense_connected = False
        self._gripper_connected = False
        self._cameras_connected = False
        self._teleop_started = False

        self.robot = UR7eInterface(
            host=str(robot_cfg["host"]),
            frequency=float(robot_cfg.get("frequency", 500.0)),
        )
        self.sense = PikaSense(
            port=str(sense_cfg.get("port", "") or ""),
            tracker_device=str(sense_cfg.get("tracker_device", "T20")),
            tracker_config=sense_cfg.get("tracker_config"),
            tracker_lh_config=sense_cfg.get("tracker_lh_config"),
        )
        self.gripper = RobotiqGripperAdapter(
            host=self.robot.host,
            port=int(gripper_cfg.get("port", 63352)),
            mapping=GripperMapping.from_config(cfg.get("gripper_mapping")),
            force=int(gripper_cfg.get("force", 150)),
            speed_min=int(gripper_cfg.get("speed_min", 80)),
            speed_max=int(gripper_cfg.get("speed_max", 255)),
            max_norm_speed_per_s=float(gripper_cfg.get("max_norm_speed_per_s", 2.0)),
        )
        smoothing_cfg = teleop_cfg.get("smoothing") or {}
        max_tilt_deg = safety_cfg.get("max_tilt_from_down_deg")
        max_tilt_rad = (
            float(np.deg2rad(max_tilt_deg)) if max_tilt_deg is not None else None
        )
        self.teleop = PikaTeleopController(
            robot=self.robot,
            sense=self.sense,
            gripper=self.gripper,
            pika_to_arm=teleop_cfg.get(
                "pika_to_arm",
                [0.0, 0.0, 0.0, 1.703151, 1.539109, 1.728148],
            ),
            position_scale=float(teleop_cfg.get("position_scale", 1.0)),
            max_delta_m=float(teleop_cfg.get("max_delta_m", 1.0)),
            servo_hz=int(teleop_cfg.get("servo_hz", 50)),
            smoothing_alpha=float(smoothing_cfg.get("pose_alpha", 1.0)),
            gripper_smoothing_alpha=float(smoothing_cfg.get("gripper_alpha", 1.0)),
            workspace_bounds=safety_cfg.get("workspace"),
            joint_limits=safety_cfg.get("joint_limits"),
            max_tilt_from_down_rad=max_tilt_rad,
            ik_mode=str(teleop_cfg.get("ik_mode", "ur_native_servol")),
            base_bias_min_radius_m=float(
                teleop_cfg.get("base_bias_min_radius_m", 0.05)
            ),
            servo_lookahead_s=float(teleop_cfg.get("servo_lookahead_s", 0.2)),
            servo_gain=float(teleop_cfg.get("servo_gain", 100.0)),
            max_lin_vel_m_s=float(teleop_cfg.get("max_lin_vel_m_s", 0.30)),
            max_ang_vel_rad_s=float(teleop_cfg.get("max_ang_vel_rad_s", 1.50)),
            max_joint_vel_rad_s=float(teleop_cfg.get("max_joint_vel_rad_s", 1.50)),
            base_limit_rad=float(teleop_cfg.get("base_limit_rad", 2.6)),
            base_limit_damping_threshold=float(
                teleop_cfg.get("base_limit_damping_threshold", 0.8)
            ),
        )
        self.cameras = MultiRGBDCamera(cfg["cameras"])
        self.writer = EpisodeWriter(
            dataset_root=Path(output_dir) / dataset_name,
            dataset_name=dataset_name,
            task=task,
            fps=self.fps,
            robot_ip=self.robot.host,
            camera_config=cfg["cameras"],
            state_names=STATE_NAMES,
            action_names=ACTION_NAMES,
            pika_sense_config={
                "port": self.sense.port,
                "tracker_device": self.sense.tracker_device,
                "tracker_config": sense_cfg.get("tracker_config"),
                "tracker_lh_config": sense_cfg.get("tracker_lh_config"),
            },
            robotiq_config={
                "host": self.robot.host,
                "port": int(gripper_cfg.get("port", 63352)),
                "force": int(gripper_cfg.get("force", 150)),
                "speed_min": int(gripper_cfg.get("speed_min", 80)),
                "speed_max": int(gripper_cfg.get("speed_max", 255)),
            },
        )

    def _queue_event(self, event: str) -> None:
        with self._lock:
            self._pending_events.append(event)

    def on_f2(self) -> None:
        with self._lock:
            event = self.state.on_teleop_toggle()
            self._pending_events.append(event)

    def on_f3(self) -> None:
        with self._lock:
            event = self.state.on_record_toggle()
            self._pending_events.append(event)

    def _drain_events(self) -> list[str]:
        with self._lock:
            events = self._pending_events
            self._pending_events = []
        return events

    def _build_state_action(self) -> tuple[np.ndarray, np.ndarray]:
        robot_state = self.robot.get_state()
        gripper_actual = self.gripper.read_position()
        target_q = self.robot.get_target_q()
        gripper_cmd = self.teleop.get_command_snapshot()["gripper_cmd"]
        state = np.concatenate(
            [
                robot_state["joint_positions"],
                np.array([gripper_actual], dtype=np.float32),
            ]
        ).astype(np.float32)
        action = np.concatenate(
            [target_q, np.array([gripper_cmd], dtype=np.float32)]
        ).astype(np.float32)
        return state, action

    def _handle_event(self, event: str) -> bool:
        if event == "teleop_started":
            self.teleop.engage()
        elif event == "teleop_stopped":
            self.teleop.release()
        elif event == "record_started":
            self.writer.start_episode()
            self._episode_start_s = time.time()
            self._active_writer_episode = True
            print("[Collector] RECORDING started by F3")
        elif event == "record_stopped":
            if self._active_writer_episode:
                out = self.writer.end_episode(save=True)
                self._active_writer_episode = False
                self._episode_start_s = None
                print(f"[Collector] RECORDING saved: {out}")
        elif event == "record_saved_and_teleop_stopped":
            if self._active_writer_episode:
                out = self.writer.end_episode(save=True)
                self._active_writer_episode = False
                self._episode_start_s = None
                print(f"[Collector] RECORDING saved: {out}")
            self.teleop.release()
        elif event == "record_discarded_and_teleop_stopped":
            if self._active_writer_episode:
                self.writer.end_episode(save=False)
                self._active_writer_episode = False
                self._episode_start_s = None
            self.teleop.release()
        elif event == "record_ignored":
            print("[Collector] F3 ignored: press F2 to start teleop first")
        else:
            logger.warning("Ignoring unknown collector event: %s", event)
        return True

    def _record_tick(self, loop_start: float) -> bool:
        if (
            self.state.mode != CollectorMode.RECORDING
            or not self._active_writer_episode
        ):
            return False

        try:
            frames = self.cameras.get_latest()
        except Exception as exc:
            now = time.time()
            if now - self._last_camera_warning_s > 1.0:
                logger.warning("Skipping frame: RGB-D capture unavailable: %s", exc)
                self._last_camera_warning_s = now
            return False

        episode_start_s = self._episode_start_s or loop_start
        state, action = self._build_state_action()
        self.writer.add_frame(
            timestamp_s=time.time() - episode_start_s,
            state=state,
            action=action,
            frames=frames,
        )
        return True

    def run(self) -> None:
        listener = FunctionKeyListener(self.on_f2, self.on_f3)
        self._listener = listener

        try:
            listener.start()

            sense_port, _ = detect_pika_ports(self.sense.port or None, None)
            self.sense.port = sense_port

            self.robot.connect()
            self._robot_connected = True
            self.sense.connect()
            self._sense_connected = True
            self.gripper.connect()
            self._gripper_connected = True
            self.cameras.connect()
            self._cameras_connected = True
            self.teleop.start()
            self._teleop_started = True

            period_s = 1.0 / max(float(self.fps), 1.0)
            print("[Collector] Ready. F2 toggles teleop; F3 toggles recording.")
            while True:
                loop_start = time.time()
                for event in self._drain_events():
                    self._handle_event(event)

                if self.teleop.aborted:
                    if self._active_writer_episode:
                        self.writer.end_episode(save=False)
                        self._active_writer_episode = False
                        self._episode_start_s = None
                    reason = self.teleop.abort_reason or "teleoperation aborted"
                    print(f"[Collector] Exiting: {reason}")
                    break

                self._record_tick(loop_start)

                elapsed_s = time.time() - loop_start
                sleep_s = period_s - elapsed_s
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\n[Collector] Interrupted; shutting down.")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as exc:
                logger.warning("Failed to stop keyboard listener: %s", exc)

        if self._active_writer_episode:
            try:
                self.writer.end_episode(save=False)
            except Exception as exc:
                logger.warning("Failed to discard active episode: %s", exc)
            self._active_writer_episode = False
            self._episode_start_s = None

        if self._teleop_started:
            try:
                self.teleop.stop()
            except Exception as exc:
                logger.warning("Failed to stop teleop: %s", exc)
            self._teleop_started = False

        if self._cameras_connected:
            try:
                self.cameras.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect cameras: %s", exc)
            self._cameras_connected = False

        if self._gripper_connected:
            try:
                self.gripper.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect gripper: %s", exc)
            self._gripper_connected = False

        if self._sense_connected:
            try:
                self.sense.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect PikaSense: %s", exc)
            self._sense_connected = False

        if self._robot_connected:
            try:
                self.robot.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect robot: %s", exc)
            self._robot_connected = False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    cfg = load_config(_resolve_config_path(args.config))

    try:
        cfg.setdefault("robot", {})["host"] = resolve_robot_host(args.robot_ip, cfg)
    except ValueError as exc:
        print(exc)
        raise SystemExit(2) from exc
    if args.sense_port:
        cfg.setdefault("pika_sense", {})["port"] = args.sense_port

    try:
        task = resolve_task(args.task, cfg)
    except ValueError as exc:
        print(exc)
        raise SystemExit(2) from exc

    collection_cfg = cfg.get("collection") or {}
    dataset_name = args.dataset_name or collection_cfg.get(
        "dataset_name", "ur7e_robotiq_rgbd"
    )
    output_dir = args.output_dir or collection_cfg.get("output_dir", "datasets")

    collector = RGBDCollector(
        cfg=cfg,
        task=task,
        dataset_name=str(dataset_name),
        output_dir=output_dir,
    )
    collector.run()


if __name__ == "__main__":
    main()
