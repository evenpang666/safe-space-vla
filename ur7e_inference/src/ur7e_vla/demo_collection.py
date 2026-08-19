from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import uuid
from typing import Callable, Optional

import numpy as np

from .cameras import CameraPair
from .config import AppConfig, DemoConfig
from .device_discovery import autodetect_pika_hardware, discover_pika_serial_devices
from .hardware import PikaGripper, UR7e

LOG = logging.getLogger(__name__)


def _import_pika_sense():
    """Prefer the PiKA SDK vendored beside this workspace over agx-pypika."""
    try:
        from pika.sense import Sense
        return Sense
    except ImportError:
        workspace_root = Path(__file__).resolve().parents[3]
        sdk_root = workspace_root / "real_scripts" / "ur7e_robotiq_d435i_collector" / "pika_sdk"
        if sdk_root.is_dir() and str(sdk_root) not in sys.path:
            sys.path.insert(0, str(sdk_root))
        try:
            from pika.sense import Sense
            return Sense
        except ImportError as exc:
            raise RuntimeError(
                "PiKA SDK is unavailable. The workspace vendored SDK was not found; "
                "do not install agx-pypika solely for this project."
            ) from exc


def normalize_task(task: str) -> str:
    return "_".join(task.strip().lower().replace("-", " ").split())


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + _skew(vector)
    axis = vector / angle
    cross = _skew(axis)
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def matrix_to_rotation_vector(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    cos_angle = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.asarray(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
        ) / 2.0
    if abs(math.pi - angle) < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + np.eye(3)) / 2.0)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        return axis * angle
    axis = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected xyz+rotation-vector pose, got {pose.shape}")
    result = np.eye(4)
    result[:3, :3] = rotation_vector_to_matrix(pose[3:])
    result[:3, 3] = pose[:3]
    return result


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[:3, 3], matrix_to_rotation_vector(matrix[:3, :3])])


def tracker_pose_change(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return translation (m) and rotation (rad) between two tracker transforms."""
    delta = np.linalg.inv(np.asarray(previous, dtype=np.float64)) @ np.asarray(current, dtype=np.float64)
    return float(np.linalg.norm(delta[:3, 3])), float(np.linalg.norm(matrix_to_rotation_vector(delta[:3, :3])))


def clamp_joint_target_step(target: np.ndarray, previous: np.ndarray, max_step_rad: float) -> tuple[np.ndarray, float]:
    """Rate-limit an IK target without allowing a 2π representation jump.

    This matches RobotControl's servoJ behaviour: an unreachable-in-one-tick
    Cartesian target is approached over several safe servo ticks rather than
    aborting collection or commanding the full joint discontinuity.
    """
    target = np.asarray(target, dtype=np.float64).copy()
    previous = np.asarray(previous, dtype=np.float64)
    if target.shape != (6,) or previous.shape != (6,) or max_step_rad <= 0:
        raise ValueError("Joint targets must be six-dimensional and max_step_rad must be positive")
    delta = target - previous
    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    unclamped_step = float(np.max(np.abs(delta)))
    return previous + np.clip(delta, -max_step_rad, max_step_rad), unclamped_step


def clamp_scalar_step(target: float, previous: float, max_step: float) -> tuple[float, float]:
    """Limit a normalized gripper command to one safe control-tick step."""
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    requested_step = float(target - previous)
    return previous + float(np.clip(requested_step, -max_step, max_step)), abs(requested_step)


def normalize_pika_encoder_angle(angle_rad: float, cfg: DemoConfig) -> float:
    """Map raw Pika Sense encoder radians to the dataset gripper convention."""
    normalized = (float(angle_rad) - cfg.gripper_closed_rad) / (cfg.gripper_open_rad - cfg.gripper_closed_rad)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return 1.0 - normalized if cfg.gripper_invert else normalized


@dataclass(frozen=True)
class SensorSample:
    monotonic_s: float
    tracker_pose: np.ndarray
    gripper: float
    # Physical Pika motor target in radians. Kept separately from ``gripper``
    # so stored data can use VLA's 0=open/1=closed convention.
    gripper_motor_rad: Optional[float] = None


@dataclass(frozen=True)
class DemonstrationTrajectory:
    samples: tuple[SensorSample, ...]
    fps: int

    def __post_init__(self) -> None:
        if len(self.samples) < 2:
            raise ValueError("A demonstration trajectory needs at least two samples")


class PikaSenseSource:
    """Lazy Pika Sense wrapper so a missing Lighthouse backend has a useful error."""

    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        self._sense = None
        self._device: Optional[str] = cfg.tracker_device
        self._last_tracker_timestamp = None
        self._last_update_monotonic = 0.0

    @property
    def is_ready(self) -> bool:
        return bool(self._sense is not None and self._sense.is_connected and self._device is not None)

    @property
    def tracker_device(self) -> Optional[str]:
        return self._device

    @staticmethod
    def _is_lighthouse(name: str) -> bool:
        # libsurvive exposes base stations as LH0/LH1. They have a pose too,
        # but are fixed infrastructure rather than the hand-held Pika tracker.
        return re.fullmatch(r"lh\d+", name.strip(), flags=re.IGNORECASE) is not None

    def connect(self) -> None:
        # Tear down a previous failed context before constructing a clean
        # libsurvive instance for this Start Teleoperation request.
        self.close()
        if str(self.cfg.pika_sense_port).strip().lower() == "auto" or not Path(self.cfg.pika_sense_port).exists():
            devices = discover_pika_serial_devices((self.cfg.pika_sense_port,))
            if devices.sense is None:
                if devices.inaccessible:
                    raise RuntimeError(
                        "Cannot read PiKA serial port(s) "
                        f"{list(devices.inaccessible)}. Add the current user to the dialout group and start a new login session."
                    )
                raise RuntimeError("PiKA Sense was not identified from passive serial telemetry")
            self.cfg.pika_sense_port = devices.sense
        self._device = self.cfg.tracker_device
        self._last_tracker_timestamp = None
        self._last_update_monotonic = 0.0
        Sense = _import_pika_sense()
        try:
            import pysurvive  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Pika Lighthouse tracking requires pysurvive, but it is not available in this Python environment. "
                "Use a supported Linux environment with libsurvive/pysurvive and USB access to both base stations."
            ) from exc
        def wait_for_initial_pose(tracker) -> None:
            deadline = time.monotonic() + self.cfg.tracker_start_timeout_s
            last_pose_names: list[str] = []
            last_device_names: list[str] = []
            last_device_info: dict = {}
            while time.monotonic() < deadline:
                poses = tracker.get_pose()
                last_pose_names = sorted(poses)
                # get_pose() only exposes devices after their first valid pose.
                # Keep the raw device list as a diagnostic, but never select a
                # raw device until it has a valid pose.
                try:
                    last_device_names = sorted(tracker.get_devices())
                    last_device_info = tracker.get_device_info() or {}
                except (AttributeError, TypeError):
                    pass
                if self._device is not None:
                    if self._device in poses:
                        LOG.info("Using configured Pika tracker device %s", self._device)
                        return
                else:
                    candidates = [name for name in last_pose_names if not self._is_lighthouse(name)]
                    if len(candidates) == 1:
                        self._device = candidates[0]
                        LOG.info("Using detected Pika tracker device %s", self._device)
                        return
                    if len(candidates) > 1:
                        raise RuntimeError(
                            "Multiple non-Lighthouse tracked devices found: "
                            f"{candidates}. Set demo.tracker_device explicitly."
                        )
                time.sleep(0.1)
            raise TimeoutError(
                "No hand-held Pika tracker pose received before timeout. "
                f"Valid-pose devices: {last_pose_names or 'none'}; "
                f"detected devices: {last_device_names or 'none'}; "
                f"device stats: {last_device_info or 'none'}. "
                "A detected device with zero updates means Lighthouse geometry has not solved; "
                "LH0/LH1 are base stations, not the sensor."
            )

        attempts = self.cfg.tracker_connect_attempts
        for attempt in range(1, attempts + 1):
            try:
                self._sense = Sense(self.cfg.pika_sense_port)
                if not self._sense.connect():
                    raise RuntimeError(f"Cannot connect to Pika Sense at {self.cfg.pika_sense_port}")
                tracker = self._sense.get_vive_tracker()
                if tracker is None or not tracker.running:
                    raise RuntimeError("Pika Vive Tracker could not start")
                wait_for_initial_pose(tracker)
                return
            except TimeoutError as exc:
                self.close()
                if attempt == attempts:
                    raise RuntimeError(
                        f"Pika Tracker received no valid hand-held pose after {attempts} clean startup attempt(s). {exc}"
                    ) from exc
                LOG.warning(
                    "PiKA Tracker startup attempt %d/%d timed out; releasing libsurvive and retrying in %.1fs",
                    attempt,
                    attempts,
                    self.cfg.tracker_reconnect_backoff_s,
                )
                time.sleep(self.cfg.tracker_reconnect_backoff_s)
            except BaseException:
                self.close()
                raise

    def sample(self) -> SensorSample:
        if self._sense is None or self._device is None:
            raise RuntimeError("Pika Sense is not connected")
        pose = self._sense.get_pose(self._device)
        if pose is None:
            raise RuntimeError(f"Tracker {self._device!r} has no pose")
        now = time.monotonic()
        if pose.timestamp != self._last_tracker_timestamp:
            self._last_tracker_timestamp = pose.timestamp
            self._last_update_monotonic = now
        if now - self._last_update_monotonic > self.cfg.max_tracker_age_s:
            raise TimeoutError(f"Tracker pose is stale for {now - self._last_update_monotonic:.3f}s")
        # RobotControl forwards the AS5047 encoder angle straight to the Pika
        # motor.  Do not convert it through the SDK's estimated finger-tip
        # distance: that geometry is nonlinear and caused a closed Sensor to
        # map to a non-closed motor target.
        encoder = self._sense.get_encoder_data()
        try:
            encoder_rad = float(encoder["rad"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Pika Sense returned invalid encoder data: {encoder!r}") from exc
        gripper = normalize_pika_encoder_angle(encoder_rad, self.cfg)
        return SensorSample(
            monotonic_s=now,
            tracker_pose=np.asarray([*pose.position, *_quaternion_to_rotation_vector(pose.rotation)], dtype=np.float64),
            gripper=gripper,
            gripper_motor_rad=float(np.clip(encoder_rad, self.cfg.gripper_closed_rad, self.cfg.gripper_open_rad)),
        )

    def close(self) -> None:
        if self._sense is not None:
            self._sense.disconnect()
            self._sense = None
        self._device = self.cfg.tracker_device
        self._last_tracker_timestamp = None
        self._last_update_monotonic = 0.0


def _quaternion_to_rotation_vector(quaternion_xyzw) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or norm < 1e-12:
        raise ValueError("Invalid tracker quaternion")
    quaternion /= norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[:3]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[3]))
    return quaternion[:3] / vector_norm * angle


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the fixed Sensor-to-tool rotation used by live teleoperation."""
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def record_sensor_trajectory(
    source: PikaSenseSource,
    fps: int,
    stop_event: threading.Event,
    on_sample: Optional[Callable[[int], None]] = None,
) -> DemonstrationTrajectory:
    period = 1.0 / fps
    next_tick = time.monotonic()
    samples: list[SensorSample] = []
    while not stop_event.is_set():
        samples.append(source.sample())
        if on_sample is not None:
            on_sample(len(samples))
        next_tick += period
        stop_event.wait(max(0.0, next_tick - time.monotonic()))
    return DemonstrationTrajectory(tuple(samples), fps)


def map_tracker_trajectory(
    trajectory: DemonstrationTrajectory,
    robot_anchor_pose: np.ndarray,
    cfg: DemoConfig,
) -> list[np.ndarray]:
    tracker_anchor = pose_to_matrix(trajectory.samples[0].tracker_pose)
    robot_anchor = pose_to_matrix(robot_anchor_pose)
    targets = []
    for sample in trajectory.samples:
        delta = np.linalg.inv(tracker_anchor) @ pose_to_matrix(sample.tracker_pose)
        translation = delta[:3, 3] * cfg.translation_scale
        rotation_vector = matrix_to_rotation_vector(delta[:3, :3]) * cfg.rotation_scale
        if np.linalg.norm(translation) > cfg.max_translation_m:
            raise RuntimeError("Demonstration exceeds configured translation workspace limit")
        if np.linalg.norm(rotation_vector) > cfg.max_rotation_rad:
            raise RuntimeError("Demonstration exceeds configured rotation workspace limit")
        scaled_delta = np.eye(4)
        scaled_delta[:3, :3] = rotation_vector_to_matrix(rotation_vector)
        scaled_delta[:3, 3] = translation
        targets.append(matrix_to_pose(robot_anchor @ scaled_delta))
    return targets


def calibration_file(cfg: DemoConfig) -> Path:
    return Path(cfg.calibration_path).expanduser().resolve()


def _average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    mean = sum(rotations) / len(rotations)
    left, _, right = np.linalg.svd(mean)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _hand_eye_residual(
    sensor_poses: list[np.ndarray], tcp_poses: list[np.ndarray], sensor_to_tcp: np.ndarray
) -> np.ndarray:
    """Residual for inv(T_tcp0) T_tcpi = inv(B) inv(T_sensor0) T_sensori B."""
    sensor_zero = pose_to_matrix(sensor_poses[0])
    tcp_zero = pose_to_matrix(tcp_poses[0])
    residuals: list[np.ndarray] = []
    inverse_extrinsic = np.linalg.inv(sensor_to_tcp)
    for sensor_pose, tcp_pose in zip(sensor_poses[1:], tcp_poses[1:], strict=True):
        sensor_delta = np.linalg.inv(sensor_zero) @ pose_to_matrix(sensor_pose)
        tcp_delta = np.linalg.inv(tcp_zero) @ pose_to_matrix(tcp_pose)
        error = np.linalg.inv(tcp_delta) @ inverse_extrinsic @ sensor_delta @ sensor_to_tcp
        pose_error = matrix_to_pose(error)
        # Balance metres and radians while preserving an interpretable residual.
        residuals.append(np.concatenate([pose_error[:3] / 0.05, pose_error[3:] / 0.25]))
    return np.concatenate(residuals)


def solve_sensor_to_tcp_hand_eye(
    sensor_poses: list[np.ndarray], tcp_poses: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve ``T_tcp = A @ T_sensor @ B`` from at least three paired poses.

    ``A`` is Lighthouse-to-UR-base and ``B`` is the fixed Sensor-to-TCP
    extrinsic. A small NumPy Gauss-Newton solve avoids requiring SciPy in the
    real-time collector environment.
    """
    if len(sensor_poses) != len(tcp_poses) or len(sensor_poses) < 3:
        raise ValueError("Hand-eye calibration needs at least three Sensor/TCP pose pairs")
    if any(np.asarray(pose).shape != (6,) for pose in [*sensor_poses, *tcp_poses]):
        raise ValueError("Hand-eye calibration poses must all be six-dimensional")

    parameters = np.zeros(6, dtype=np.float64)  # pose representation of B
    epsilon = 1e-5
    for _ in range(40):
        extrinsic = pose_to_matrix(parameters)
        residual = _hand_eye_residual(sensor_poses, tcp_poses, extrinsic)
        jacobian = np.empty((residual.size, 6), dtype=np.float64)
        for axis in range(6):
            shifted = parameters.copy()
            shifted[axis] += epsilon
            jacobian[:, axis] = (
                _hand_eye_residual(sensor_poses, tcp_poses, pose_to_matrix(shifted)) - residual
            ) / epsilon
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        parameters += step
        if float(np.linalg.norm(step)) < 1e-8:
            break

    sensor_to_tcp = pose_to_matrix(parameters)
    inverse_extrinsic = np.linalg.inv(sensor_to_tcp)
    base_candidates = [
        pose_to_matrix(tcp_pose) @ inverse_extrinsic @ np.linalg.inv(pose_to_matrix(sensor_pose))
        for sensor_pose, tcp_pose in zip(sensor_poses, tcp_poses, strict=True)
    ]
    tracker_to_base = np.eye(4)
    tracker_to_base[:3, :3] = _average_rotations([candidate[:3, :3] for candidate in base_candidates])
    tracker_to_base[:3, 3] = np.mean([candidate[:3, 3] for candidate in base_candidates], axis=0)

    errors = [
        matrix_to_pose(
            np.linalg.inv(pose_to_matrix(tcp_pose))
            @ tracker_to_base
            @ pose_to_matrix(sensor_pose)
            @ sensor_to_tcp
        )
        for sensor_pose, tcp_pose in zip(sensor_poses, tcp_poses, strict=True)
    ]
    position_rmse = float(np.sqrt(np.mean([np.dot(error[:3], error[:3]) for error in errors])))
    orientation_rmse = float(np.sqrt(np.mean([np.dot(error[3:], error[3:]) for error in errors])))
    return tracker_to_base, sensor_to_tcp, position_rmse, orientation_rmse


def save_hand_eye_calibration(
    cfg: DemoConfig,
    tracker_poses: list[np.ndarray],
    tcp_poses: list[np.ndarray],
    tracker_device: Optional[str],
) -> Path:
    tracker_to_base, sensor_to_tcp, position_rmse, orientation_rmse = solve_sensor_to_tcp_hand_eye(
        tracker_poses, tcp_poses
    )
    if position_rmse > cfg.calibration_max_position_rmse_m:
        raise RuntimeError(
            f"Hand-eye calibration position RMSE is {position_rmse:.4f} m, exceeding "
            f"{cfg.calibration_max_position_rmse_m:.4f} m"
        )
    if orientation_rmse > cfg.calibration_max_orientation_rmse_rad:
        raise RuntimeError(
            f"Hand-eye calibration orientation RMSE is {orientation_rmse:.3f} rad, exceeding "
            f"{cfg.calibration_max_orientation_rmse_rad:.3f} rad"
        )
    path = calibration_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "tracker_device": tracker_device,
        "pose_pair_count": len(tracker_poses),
        "position_rmse_m": position_rmse,
        "orientation_rmse_rad": orientation_rmse,
        "tracker_to_base_transform": tracker_to_base.tolist(),
        "sensor_to_tcp_transform": sensor_to_tcp.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_rigid_transform(payload: dict, field: str, path: Path) -> np.ndarray:
    transform = np.asarray(payload[field], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError(f"Invalid {field} in calibration file: {path}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise RuntimeError(f"{field} is not a rigid transform: {path}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise RuntimeError(f"{field} rotation is invalid: {path}")
    return transform


def load_tracker_to_tcp_calibration(cfg: DemoConfig) -> tuple[np.ndarray, np.ndarray]:
    path = calibration_file(cfg)
    if not path.is_file():
        raise RuntimeError(
            f"Sensor-to-UR calibration is missing: {path}. "
            "Capture at least three paired poses with `ur7e-vla calibrate-demo --config config.yaml`."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid Sensor-to-UR calibration file: {path}") from exc
    if payload.get("version") != 2:
        raise RuntimeError(f"Calibration file {path} uses an obsolete single-pose format; recalibrate with 3+ poses")
    return (
        _load_rigid_transform(payload, "tracker_to_base_transform", path),
        _load_rigid_transform(payload, "sensor_to_tcp_transform", path),
    )


def map_calibrated_tracker_trajectory(
    trajectory: DemonstrationTrajectory,
    tracker_to_base: np.ndarray,
    cfg: DemoConfig,
    sensor_to_tcp: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Map an absolute Lighthouse trajectory into absolute UR TCP targets."""
    tracker_to_base = np.asarray(tracker_to_base, dtype=np.float64)
    sensor_to_tcp = np.eye(4) if sensor_to_tcp is None else np.asarray(sensor_to_tcp, dtype=np.float64)
    if tracker_to_base.shape != (4, 4) or sensor_to_tcp.shape != (4, 4):
        raise ValueError("calibrated trajectory transforms must both be 4x4")
    tracker_anchor = pose_to_matrix(trajectory.samples[0].tracker_pose)
    tcp_anchor = tracker_to_base @ tracker_anchor @ sensor_to_tcp
    inverse_sensor_to_tcp = np.linalg.inv(sensor_to_tcp)
    targets: list[np.ndarray] = []
    for sample in trajectory.samples:
        delta = np.linalg.inv(tracker_anchor) @ pose_to_matrix(sample.tracker_pose)
        translation = delta[:3, 3] * cfg.translation_scale
        rotation_vector = matrix_to_rotation_vector(delta[:3, :3]) * cfg.rotation_scale
        if np.linalg.norm(translation) > cfg.max_translation_m:
            raise RuntimeError("Demonstration exceeds configured translation workspace limit")
        if np.linalg.norm(rotation_vector) > cfg.max_rotation_rad:
            raise RuntimeError("Demonstration exceeds configured rotation workspace limit")
        scaled_delta = np.eye(4)
        scaled_delta[:3, :3] = rotation_vector_to_matrix(rotation_vector)
        scaled_delta[:3, 3] = translation
        targets.append(matrix_to_pose(tcp_anchor @ inverse_sensor_to_tcp @ scaled_delta @ sensor_to_tcp))
    return targets


def calibrate_demo_sensor_to_tcp(
    cfg: AppConfig, wait_for_operator: Callable[[str], str] = input
) -> Path:
    """Interactively capture multiple shared physical poses, without robot motion.

    The Sensor and TCP samples may be captured sequentially.  The Lighthouse
    base stations and the physical calibration reference must remain fixed
    between the two captures.
    """
    source = PikaSenseSource(cfg.demo)
    robot = UR7e(cfg.robot, execute=False)
    try:
        source.connect()
        robot.connect()
        sensor_poses: list[np.ndarray] = []
        tcp_poses: list[np.ndarray] = []
        count = cfg.demo.calibration_pose_count
        for index in range(count):
            wait_for_operator(
                f"UR TCP {index + 1}/{count}: move TCP to reference pose {index + 1} "
                "with a distinct position and orientation. This command will not move the robot; "
                "keep base stations and reference marks fixed, then press Enter. "
            )
            tcp_poses.append(robot.tcp_pose())
        wait_for_operator(
            "All UR TCP poses captured. Do not move base stations or reference marks. "
            "Prepare to capture Sensor poses in the same reference-pose order, then press Enter. "
        )
        for index in range(count):
            wait_for_operator(
                f"Sensor {index + 1}/{count}: put Pika Sensor at the same physical reference pose {index + 1}. "
                "Keep its position and orientation aligned, then press Enter. "
            )
            sensor_poses.append(source.sample().tracker_pose)
        return save_hand_eye_calibration(cfg.demo, sensor_poses, tcp_poses, source.tracker_device)
    finally:
        try:
            robot.stop()
        finally:
            source.close()


@dataclass(frozen=True)
class PendingFrame:
    image_path: Path
    wrist_image_path: Path
    depth_path: Optional[Path]
    side_image_path: Optional[Path]
    side_depth_path: Optional[Path]
    monotonic_s: float
    joints: np.ndarray
    gripper: np.ndarray
    gripper_opening_mm: Optional[np.ndarray]
    actions: np.ndarray


class PendingEpisode:
    def __init__(self, staging_root: str | Path, *, side_enabled: bool = False):
        base = Path(staging_root).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"episode_{uuid.uuid4().hex}"
        self.path.mkdir()
        self.front_rgb_dir = self.path / "front_rgb"
        self.wrist_rgb_dir = self.path / "wrist_rgb"
        self.front_depth_dir = self.path / "front_depth_m"
        self.side_rgb_dir = self.path / "side_rgb" if side_enabled else None
        self.side_depth_dir = self.path / "side_depth_m" if side_enabled else None
        self.front_rgb_dir.mkdir()
        self.wrist_rgb_dir.mkdir()
        self.front_depth_dir.mkdir()
        if self.side_rgb_dir is not None and self.side_depth_dir is not None:
            self.side_rgb_dir.mkdir()
            self.side_depth_dir.mkdir()
        self.frames: list[PendingFrame] = []

    def add(
        self,
        exterior_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        joints: np.ndarray,
        gripper: float,
        action: np.ndarray,
        width: int | None = None,
        height: int | None = None,
        *,
        exterior_depth_m: Optional[np.ndarray] = None,
        side_rgb: Optional[np.ndarray] = None,
        side_depth_m: Optional[np.ndarray] = None,
        gripper_opening_mm: Optional[float] = None,
    ) -> None:
        import cv2

        index = len(self.frames)
        image_path = self.front_rgb_dir / f"frame_{index:06d}.png"
        wrist_path = self.wrist_rgb_dir / f"frame_{index:06d}.png"
        exterior = np.asarray(exterior_rgb, dtype=np.uint8)
        wrist = np.asarray(wrist_rgb, dtype=np.uint8)
        if exterior.ndim != 3 or exterior.shape[2] != 3 or wrist.ndim != 3 or wrist.shape[2] != 3:
            raise ValueError("RGB frames must have shape (H, W, 3)")
        depth_path = None
        if exterior_depth_m is not None:
            depth = np.asarray(exterior_depth_m, dtype=np.float32)
            if depth.shape != exterior.shape[:2] or not np.isfinite(depth).all():
                raise ValueError(f"Invalid front depth frame {depth.shape}; expected {exterior.shape[:2]}")
            depth_path = self.front_depth_dir / f"frame_{index:06d}.npy"
            np.save(depth_path, depth, allow_pickle=False)
        side_image_path = None
        side_depth_path = None
        if self.side_rgb_dir is not None or self.side_depth_dir is not None:
            if side_rgb is None or side_depth_m is None:
                raise ValueError("side RGB and depth are required when the side camera is enabled")
            assert self.side_rgb_dir is not None and self.side_depth_dir is not None
            side_image = np.asarray(side_rgb, dtype=np.uint8)
            side_depth = np.asarray(side_depth_m, dtype=np.float32)
            if side_image.ndim != 3 or side_image.shape[2] != 3:
                raise ValueError("Side RGB frame must have shape (H, W, 3)")
            if side_depth.shape != side_image.shape[:2] or not np.isfinite(side_depth).all():
                raise ValueError(f"Invalid side depth frame {side_depth.shape}; expected {side_image.shape[:2]}")
            side_image_path = self.side_rgb_dir / f"frame_{index:06d}.png"
            side_depth_path = self.side_depth_dir / f"frame_{index:06d}.npy"
            np.save(side_depth_path, side_depth, allow_pickle=False)
        if not cv2.imwrite(str(image_path), cv2.cvtColor(exterior, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to stage {image_path}")
        if not cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to stage {wrist_path}")
        if side_image_path is not None and not cv2.imwrite(
            str(side_image_path), cv2.cvtColor(side_image, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"Failed to stage {side_image_path}")
        self.frames.append(
            PendingFrame(
                image_path=image_path,
                wrist_image_path=wrist_path,
                depth_path=depth_path,
                side_image_path=side_image_path,
                side_depth_path=side_depth_path,
                monotonic_s=time.monotonic(),
                joints=np.asarray(joints, dtype=np.float32),
                gripper=np.asarray([gripper], dtype=np.float32),
                gripper_opening_mm=(
                    None if gripper_opening_mm is None else np.asarray([gripper_opening_mm], dtype=np.float32)
                ),
                actions=np.asarray(action, dtype=np.float32),
            )
        )

    def discard(self) -> None:
        if self.path.exists() and self.path.parent != self.path:
            shutil.rmtree(self.path)
        self.frames.clear()


def _read_tasks(path: Path) -> list[str]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(str(json.loads(line)["task"]))
    return tasks


def find_dataset_for_task(root: Path, task: str) -> Optional[Path]:
    wanted = normalize_task(task)
    matches = []
    if root.exists():
        for tasks_file in root.rglob("tasks.jsonl"):
            if any(normalize_task(item) == wanted for item in _read_tasks(tasks_file)):
                matches.append(tasks_file.parent.parent)
    if len(matches) > 1:
        raise RuntimeError(f"Task {task!r} occurs in multiple demo datasets: {matches}")
    return matches[0] if matches else None


def _features(cfg: DemoConfig) -> dict:
    image = {"dtype": "video", "shape": (cfg.image_height, cfg.image_width, 3), "names": ["height", "width", "channel"]}
    return {
        "joints": {"dtype": "float32", "shape": (6,), "names": [f"joint_{i}" for i in range(6)]},
        "gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": [*[f"joint_{i}" for i in range(6)], "gripper"]},
        "image": image,
        "wrist_image": dict(image),
    }


def save_pending_episode(pending: PendingEpisode, task: str, cfg: DemoConfig) -> Path:
    """Persist one recording as portable files, without importing LeRobot."""
    if not pending.frames:
        raise ValueError("Cannot save an empty episode")
    front_shape = None
    for frame in pending.frames:
        if not frame.image_path.is_file() or not frame.wrist_image_path.is_file():
            raise RuntimeError("A staged RGB frame is missing")
        if frame.depth_path is not None and not frame.depth_path.is_file():
            raise RuntimeError("A staged front-depth frame is missing")
        if frame.side_image_path is not None and not frame.side_image_path.is_file():
            raise RuntimeError("A staged side RGB frame is missing")
        if frame.side_depth_path is not None and not frame.side_depth_path.is_file():
            raise RuntimeError("A staged side-depth frame is missing")
        if front_shape is None:
            import cv2

            image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read staged front frame {frame.image_path}")
            front_shape = [int(image.shape[1]), int(image.shape[0]), int(image.shape[2])]

    np.save(pending.path / "joints.npy", np.stack([frame.joints for frame in pending.frames]).astype(np.float32))
    np.save(pending.path / "gripper.npy", np.stack([frame.gripper for frame in pending.frames]).astype(np.float32))
    has_depth = all(frame.depth_path is not None for frame in pending.frames)
    has_side = all(
        frame.side_image_path is not None and frame.side_depth_path is not None
        for frame in pending.frames
    )
    has_opening = all(frame.gripper_opening_mm is not None for frame in pending.frames)
    if has_opening:
        np.save(
            pending.path / "gripper_opening_mm.npy",
            np.stack([frame.gripper_opening_mm for frame in pending.frames if frame.gripper_opening_mm is not None]).astype(np.float32),
        )
    np.save(pending.path / "actions.npy", np.stack([frame.actions for frame in pending.frames]).astype(np.float32))
    timestamps = np.asarray([frame.monotonic_s for frame in pending.frames], dtype=np.float64)
    np.save(pending.path / "timestamps_s.npy", timestamps - timestamps[0])
    calibration_path = Path(cfg.safety_calibration_path).expanduser()
    if cfg.safety_calibration_path:
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Configured safety calibration does not exist: {calibration_path}")
        shutil.copy2(calibration_path, pending.path / "calibration.json")
    (pending.path / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "task": task,
                "fps": cfg.fps,
                "frame_count": len(pending.frames),
                "front_rgb_dir": "front_rgb",
                "wrist_rgb_dir": "wrist_rgb",
                "front_depth_m_dir": "front_depth_m" if has_depth else None,
                "side_rgb_dir": "side_rgb" if has_side else None,
                "side_depth_m_dir": "side_depth_m" if has_side else None,
                "front_rgb_shape_whc": front_shape,
                "joints_file": "joints.npy",
                "gripper_file": "gripper.npy",
                "gripper_opening_mm_file": "gripper_opening_mm.npy" if has_opening else None,
                "actions_file": "actions.npy",
                "timestamps_file": "timestamps_s.npy",
                "calibration_file": "calibration.json" if cfg.safety_calibration_path else None,
                "scene_camera_map": {"front": cfg.safety_front_calibration_name}
                if cfg.safety_front_calibration_name
                else {},
                "conventions": {"gripper": "0=open, 1=closed"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_root = Path(cfg.output_dir).expanduser().resolve()
    task_name = normalize_task(task)
    if not task_name:
        raise ValueError("Cannot save an episode with an empty task name")
    task_root = output_root / task_name
    task_root.mkdir(parents=True, exist_ok=True)
    existing_numbers = [
        int(path.name.removeprefix("episode_"))
        for path in task_root.iterdir()
        if path.is_dir() and path.name.startswith("episode_") and path.name.removeprefix("episode_").isdigit()
    ]
    next_number = max(existing_numbers, default=0) + 1
    destination = task_root / f"episode_{next_number:03d}"
    while destination.exists():
        next_number += 1
        destination = task_root / f"episode_{next_number:03d}"
    shutil.move(str(pending.path), str(destination))
    return destination


class LiveDemoCollector:
    """Direct Pika Sense teleoperation with independently controlled staging.

    The first valid sensor pose and current TCP pose form a session anchor.
    Subsequent Pika motion is mapped relatively, as in RobotControl's teleop,
    so collection does not require an absolute Lighthouse-to-UR calibration.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.source = PikaSenseSource(cfg.demo)
        self.robot = UR7e(cfg.robot, execute=True)
        self.gripper = PikaGripper(cfg.gripper, execute=True)
        self.cameras = CameraPair(cfg.cameras)

    def run(
        self,
        stop_event: threading.Event,
        record_event: threading.Event,
        on_recording_stopped: Callable[[PendingEpisode], None],
        on_ready: Optional[Callable[[], None]] = None,
        on_recording_started: Optional[Callable[[], None]] = None,
        on_frame: Optional[Callable[[int], None]] = None,
        *,
        pause_event: Optional[threading.Event] = None,
        on_paused: Optional[Callable[[], None]] = None,
        on_resumed: Optional[Callable[[], None]] = None,
    ) -> None:
        pending: Optional[PendingEpisode] = None
        pause_event = pause_event or threading.Event()
        period = 1.0 / self.cfg.demo.fps
        try:
            # Resolve all physical endpoints before opening RTDE or enabling a
            # motor. This turns stale Windows COM settings into a clear, safe
            # discovery error rather than a failed live-teleoperation session.
            autodetect_pika_hardware(
                self.cfg,
                require_sense=True,
                require_gripper=self.cfg.gripper.enabled,
                require_wrist_camera=True,
            )
            self.source.connect()
            self.robot.connect()
            self.gripper.connect()
            # Keep both RealSense streams closed during free teleoperation.
            # They are needed only while an episode is being recorded, and
            # otherwise compete with the Pika USB serial devices unnecessarily.

            sensor_to_tool = np.eye(4)
            sensor_to_tool[:3, :3] = rpy_to_matrix(*self.cfg.demo.sensor_to_tool_rpy)

            def reset_teleop_anchor() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
                """Anchor resume at the current Sensor/TCP poses to prevent a jump."""
                sample = self.source.sample()
                sensor_anchor = pose_to_matrix(sample.tracker_pose) @ sensor_to_tool
                robot_anchor = pose_to_matrix(self.robot.tcp_pose())
                previous_target = self.robot.joints()
                previous_gripper = self.cfg.gripper.policy_min + sample.gripper * (
                    self.cfg.gripper.policy_max - self.cfg.gripper.policy_min
                )
                initial_motor_normalized = (previous_gripper - self.cfg.gripper.policy_min) / (
                    self.cfg.gripper.policy_max - self.cfg.gripper.policy_min
                )
                if self.cfg.gripper.invert:
                    initial_motor_normalized = 1.0 - initial_motor_normalized
                LOG.info(
                    "Pika teleoperation anchor reset: Sensor policy=%.3f -> motor target=%.3f rad",
                    sample.gripper,
                    self.cfg.gripper.min_angle_rad
                    + initial_motor_normalized * (self.cfg.gripper.max_angle_rad - self.cfg.gripper.min_angle_rad),
                )
                return sensor_anchor, sensor_anchor, robot_anchor, previous_target, previous_gripper

            sensor_anchor, previous_sensor_pose, robot_anchor, previous_target, previous_gripper = reset_teleop_anchor()
            lower = np.asarray(self.cfg.robot.joint_min_rad)
            upper = np.asarray(self.cfg.robot.joint_max_rad)
            # RobotControl rate-limits servoJ targets instead of treating an
            # ordinary tracker update as a fatal error.  Use the stricter of
            # the configured step ceiling and physical joint-speed limit.
            max_joint_step = min(
                self.cfg.demo.max_ik_joint_step_rad,
                self.cfg.demo.teleop_joint_speed_rad_s * period,
            )
            next_tick = time.monotonic()
            index = 0
            last_joint_rate_log = 0.0
            pause_notified = False

            def stop_pending_recording() -> None:
                nonlocal pending
                if pending is None:
                    return
                completed = pending
                pending = None
                if completed.frames:
                    LOG.info("Live demo recording stopped with %d frames", len(completed.frames))
                    on_recording_stopped(completed)
                else:
                    completed.discard()
                LOG.info("Stopping cameras after live demo recording")
                self.cameras.stop()

            if on_ready is not None:
                on_ready()

            while not stop_event.is_set():
                if pause_event.is_set():
                    # Soft pause: release the active servo trajectory but keep
                    # PiKA serial, RTDE and camera objects connected. A resume
                    # only needs a fresh relative-motion anchor, not a new
                    # hardware connection or Lighthouse startup.
                    stop_pending_recording()
                    self.robot.stop_servo()
                    if not pause_notified:
                        pause_notified = True
                        if on_paused is not None:
                            on_paused()
                    while pause_event.is_set() and not stop_event.is_set():
                        stop_event.wait(0.05)
                    if stop_event.is_set():
                        break
                    sensor_anchor, previous_sensor_pose, robot_anchor, previous_target, previous_gripper = (
                        reset_teleop_anchor()
                    )
                    next_tick = time.monotonic()
                    pause_notified = False
                    if on_resumed is not None:
                        on_resumed()
                    continue
                if record_event.is_set() and pending is None:
                    LOG.info("Starting cameras for live demo recording")
                    self.cameras.start()
                    pending = PendingEpisode(
                        self.cfg.demo.staging_dir,
                        side_enabled=self.cfg.cameras.side.enabled,
                    )
                    index = 0
                    LOG.info("Live demo recording started: %s", pending.path)
                    if on_recording_started is not None:
                        on_recording_started()
                elif not record_event.is_set() and pending is not None:
                    stop_pending_recording()
                sample = self.source.sample()

                # Keep the Pika serial path ahead of all RTDE/IK and camera
                # work.  RobotControl commands the gripper immediately after
                # reading the Sensor; doing it after getInverseKinematics()
                # can delay a hand close whenever the robot call stalls.
                gripper_target = self.cfg.gripper.policy_min + sample.gripper * (
                    self.cfg.gripper.policy_max - self.cfg.gripper.policy_min
                )
                motor_target = (
                    sample.gripper_motor_rad
                    if sample.gripper_motor_rad is not None
                    else self.cfg.gripper.min_angle_rad
                    + (1.0 - sample.gripper) * (
                        self.cfg.gripper.max_angle_rad - self.cfg.gripper.min_angle_rad
                    )
                )
                self.gripper.set_motor_angle(motor_target)
                if abs(gripper_target - previous_gripper) >= 0.01:
                    feedback = self.gripper.motor_feedback() or {}
                    LOG.info(
                        "Pika gripper command sent: Sensor policy=%.3f -> target=%.3f rad; "
                        "raw=%.3f rad actual=%.3f rad speed=%.3f rad/s current=%s mA voltage=%s V",
                        gripper_target,
                        motor_target,
                        float(sample.gripper_motor_rad if sample.gripper_motor_rad is not None else float("nan")),
                        float(feedback.get("Position", float("nan"))),
                        float(feedback.get("Speed", float("nan"))),
                        feedback.get("Current", "?"),
                        feedback.get("Voltage", "?"),
                    )
                sensor_pose = pose_to_matrix(sample.tracker_pose) @ sensor_to_tool
                tracker_translation, tracker_rotation = tracker_pose_change(previous_sensor_pose, sensor_pose)
                is_tracker_jitter = (
                    tracker_translation <= self.cfg.demo.tracker_position_deadband_m
                    and tracker_rotation <= self.cfg.demo.tracker_orientation_deadband_rad
                )
                joint_target = previous_target
                if not is_tracker_jitter:
                    delta = np.linalg.inv(sensor_anchor) @ sensor_pose
                    translation = delta[:3, 3] * self.cfg.demo.translation_scale
                    rotation_vector = matrix_to_rotation_vector(delta[:3, :3]) * self.cfg.demo.rotation_scale
                    if np.linalg.norm(translation) > self.cfg.demo.max_translation_m:
                        raise RuntimeError("Pika translation exceeds configured teleoperation workspace limit")
                    if np.linalg.norm(rotation_vector) > self.cfg.demo.max_rotation_rad:
                        raise RuntimeError("Pika rotation exceeds configured teleoperation workspace limit")

                    scaled_delta = np.eye(4)
                    scaled_delta[:3, :3] = rotation_vector_to_matrix(rotation_vector)
                    scaled_delta[:3, 3] = translation
                    tcp_target = matrix_to_pose(robot_anchor @ scaled_delta)
                    joint_target = self.robot.inverse_kinematics(
                        tcp_target,
                        previous_target,
                        self.cfg.demo.ik_position_tolerance_m,
                        self.cfg.demo.ik_orientation_tolerance_rad,
                    )
                    joint_target, joint_step = clamp_joint_target_step(
                        joint_target, previous_target, max_joint_step
                    )
                    if joint_step > max_joint_step and time.monotonic() - last_joint_rate_log > 1.0:
                        LOG.warning(
                            "Rate-limiting Pika IK target: %.4f rad requested, %.4f rad allowed "
                            "(tracker change %.2f mm, %.2f deg)",
                            joint_step,
                            max_joint_step,
                            tracker_translation * 1000,
                            math.degrees(tracker_rotation),
                        )
                        last_joint_rate_log = time.monotonic()
                    previous_sensor_pose = sensor_pose
                if np.any(joint_target < lower) or np.any(joint_target > upper):
                    raise RuntimeError("Pika teleoperation target violates configured UR joint limits")

                action = np.concatenate([joint_target, np.asarray([gripper_target])])
                if pending is not None:
                    joints = self.robot.joints()
                    gripper_state = self.gripper.position()
                    front_frame = self.cameras.front_rgbd_frame()
                    side_frame = (
                        self.cameras.side_rgbd_frame()
                        if self.cfg.cameras.side.enabled
                        else None
                    )
                    gripper_opening_mm = self.gripper.opening_mm()
                    if gripper_opening_mm is None:
                        normalized = (gripper_state - self.cfg.gripper.policy_min) / (
                            self.cfg.gripper.policy_max - self.cfg.gripper.policy_min
                        )
                        gripper_opening_mm = self.cfg.demo.gripper_distance_min_mm + (1.0 - normalized) * (
                            self.cfg.demo.gripper_distance_max_mm - self.cfg.demo.gripper_distance_min_mm
                        )
                    pending.add(
                        front_frame.rgb,
                        self.cameras.wrist_rgb_frame(),
                        joints,
                        gripper_state,
                        action,
                        exterior_depth_m=front_frame.depth_m,
                        side_rgb=None if side_frame is None else side_frame.rgb,
                        side_depth_m=None if side_frame is None else side_frame.depth_m,
                        gripper_opening_mm=gripper_opening_mm,
                    )
                self.robot.send_target(
                    joint_target,
                    period_s=period,
                    speed_rad_s=self.cfg.demo.teleop_joint_speed_rad_s,
                    acceleration_rad_s2=self.cfg.demo.teleop_joint_acceleration_rad_s2,
                    lookahead_s=self.cfg.demo.teleop_servo_lookahead_s,
                    gain=self.cfg.demo.teleop_servo_gain,
                )
                previous_target = joint_target
                previous_gripper = gripper_target
                if pending is not None:
                    index += 1
                    if on_frame is not None:
                        on_frame(index)
                next_tick += period
                stop_event.wait(max(0.0, next_tick - time.monotonic()))

        except BaseException:
            if pending is not None:
                pending.discard()
                pending = None
            raise
        finally:
            if pending is not None:
                if pending.frames:
                    on_recording_stopped(pending)
                else:
                    pending.discard()
            try:
                self.robot.stop()
            finally:
                try:
                    self.gripper.stop()
                finally:
                    try:
                        self.cameras.stop()
                    finally:
                        self.source.close()


def run_headless_demo(
    cfg: AppConfig, task: str, *, execute: bool, wait_for_record: bool = False
) -> None:
    """Run one demo worker, optionally waiting for an explicit ``record`` command.

    The parent Gradio process can stop this worker with SIGINT.  A normal stop
    leaves the live loop and synchronously saves the current standalone-file
    episode before this process exits. With ``wait_for_record=True``, the
    worker teleoperates without opening cameras until it receives ``record``
    on standard input.
    """
    if not execute:
        raise ValueError("The headless demo worker requires --execute")
    task = task.strip()
    if not task:
        raise ValueError("The headless demo worker requires a non-empty --task")
    stop_event = threading.Event()
    record_event = threading.Event()
    if not wait_for_record:
        record_event.set()
    collector = LiveDemoCollector(cfg)
    saved = False

    def request_stop(*_args) -> None:
        LOG.info("Headless demo worker received stop request")
        stop_event.set()

    def save_recording(pending: PendingEpisode) -> None:
        nonlocal saved
        if saved:
            return
        saved = True
        path = save_pending_episode(pending, task, cfg.demo)
        print(f"[demo-worker] SAVED {path}", flush=True)

    import signal

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(f"[demo-worker] START task={task!r}", flush=True)

    if wait_for_record:
        def listen_for_control() -> None:
            import sys

            for line in sys.stdin:
                command = line.strip().lower()
                if command == "record":
                    if not record_event.is_set():
                        record_event.set()
                        print("[demo-worker] RECORD_REQUESTED", flush=True)
                elif command:
                    print(f"[demo-worker] UNKNOWN_CONTROL {command!r}", flush=True)

        threading.Thread(target=listen_for_control, name="demo-worker-control", daemon=True).start()
    try:
        collector.run(
            stop_event,
            record_event,
            save_recording,
            on_ready=lambda: print(
                f"[demo-worker] READY recording={'on' if record_event.is_set() else 'off'}", flush=True
            ),
            on_recording_started=lambda: print("[demo-worker] RECORDING_STARTED", flush=True),
        )
    except BaseException as exc:
        print(f"[demo-worker] FAILED {exc}", flush=True)
        raise
    print("[demo-worker] STOPPED", flush=True)


class DemoReplay:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.robot = UR7e(cfg.robot, execute=True)
        self.gripper = PikaGripper(cfg.gripper, execute=True)
        self.cameras = CameraPair(cfg.cameras)

    def run(
        self,
        trajectory: DemonstrationTrajectory,
        stop_event: threading.Event,
        on_frame: Optional[Callable[[int], None]] = None,
    ) -> PendingEpisode:
        pending = PendingEpisode(
            self.cfg.demo.staging_dir,
            side_enabled=self.cfg.cameras.side.enabled,
        )
        period = 1.0 / self.cfg.demo.fps
        try:
            autodetect_pika_hardware(
                self.cfg,
                require_sense=False,
                require_gripper=self.cfg.gripper.enabled,
                require_wrist_camera=True,
            )
            self.robot.connect()
            self.gripper.connect()
            self.cameras.start()
            current_joints = self.robot.joints()
            tracker_to_base, sensor_to_tcp = load_tracker_to_tcp_calibration(self.cfg.demo)
            tcp_targets = map_calibrated_tracker_trajectory(
                trajectory, tracker_to_base, self.cfg.demo, sensor_to_tcp
            )
            joint_targets = []
            near = current_joints
            max_step = min(self.cfg.demo.max_ik_joint_step_rad, self.cfg.robot.max_joint_step_rad)
            lower = np.asarray(self.cfg.robot.joint_min_rad)
            upper = np.asarray(self.cfg.robot.joint_max_rad)
            for tcp_target in tcp_targets:
                target = self.robot.inverse_kinematics(
                    tcp_target,
                    near,
                    self.cfg.demo.ik_position_tolerance_m,
                    self.cfg.demo.ik_orientation_tolerance_rad,
                )
                # The first target is reached with a blocking, slow moveJ below;
                # every recorded replay step still has to satisfy servoJ limits.
                if joint_targets and float(np.max(np.abs(target - near))) > max_step:
                    raise RuntimeError(
                        "IK trajectory contains an unsafe joint discontinuity; slow down the demonstration "
                        "or increase demo FPS"
                    )
                if np.any(target < lower) or np.any(target > upper):
                    raise RuntimeError("IK trajectory violates configured UR joint limits")
                joint_targets.append(target)
                near = target

            previous_gripper = self.gripper.position()
            gripper_targets = [
                self.cfg.gripper.policy_min
                + sample.gripper * (self.cfg.gripper.policy_max - self.cfg.gripper.policy_min)
                for sample in trajectory.samples
            ]
            for index, gripper_target in enumerate(gripper_targets):
                if index and abs(gripper_target - previous_gripper) > self.cfg.demo.max_gripper_step:
                    raise RuntimeError(
                        "Demonstration contains an unsafe gripper step; close/open the Pika Sense more slowly"
                    )
                previous_gripper = gripper_target

            if stop_event.is_set():
                raise RuntimeError("Replay cancelled before moving to start pose")
            LOG.info("Moving safely to calibrated demonstration start pose before recording")
            self.robot.move_joints(
                joint_targets[0],
                self.cfg.demo.approach_joint_speed_rad_s,
                self.cfg.demo.approach_joint_acceleration_rad_s2,
            )
            reached = self.robot.joints()
            if float(np.max(np.abs(reached - joint_targets[0]))) > self.cfg.initial_state.joint_tolerance_rad:
                raise RuntimeError("UR did not reach the calibrated demonstration start pose")
            self.gripper.apply(np.asarray([*joint_targets[0], gripper_targets[0]]))
            if self.cfg.demo.approach_settle_s:
                stop_event.wait(self.cfg.demo.approach_settle_s)
            if stop_event.is_set():
                raise RuntimeError("Replay cancelled before trajectory execution")

            next_tick = time.monotonic()
            for index, (target, gripper_action) in enumerate(zip(joint_targets, gripper_targets, strict=True)):
                if stop_event.is_set():
                    raise RuntimeError("Replay cancelled; episode was not saved")
                joints = self.robot.joints()
                gripper_state = self.gripper.position()
                front_frame = self.cameras.front_rgbd_frame()
                side_frame = (
                    self.cameras.side_rgbd_frame()
                    if self.cfg.cameras.side.enabled
                    else None
                )
                action = np.concatenate([target, np.asarray([gripper_action])])
                pending.add(
                    front_frame.rgb,
                    self.cameras.wrist_rgb_frame(),
                    joints,
                    gripper_state,
                    action,
                    self.cfg.demo.image_width,
                    self.cfg.demo.image_height,
                    exterior_depth_m=front_frame.depth_m,
                    side_rgb=None if side_frame is None else side_frame.rgb,
                    side_depth_m=None if side_frame is None else side_frame.depth_m,
                )
                self.robot.send_target(target, period_s=period)
                self.gripper.apply(action)
                if on_frame is not None:
                    on_frame(index + 1)
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
            self.robot.stop_servo()
            return pending
        except BaseException:
            pending.discard()
            raise
        finally:
            try:
                self.robot.stop()
            finally:
                try:
                    self.gripper.stop()
                finally:
                    self.cameras.stop()
