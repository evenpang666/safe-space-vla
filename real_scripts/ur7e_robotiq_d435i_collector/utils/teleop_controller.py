from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

import numpy as np

from .math_tools import MathTools
from .pika_interface import PikaSense
from .robot_interface import UR7eInterface


logger = logging.getLogger(__name__)


class PikaTeleopController:
    """Background Pika Sense to UR7e teleoperation controller.

    The Sense reports a 6-DoF Vive tracker pose plus hand encoder value.
    Tracker deltas are mapped into the arm frame, safety-filtered, and sent
    through either servoL or base-biased servoJ depending on ``ik_mode``.
    Teleop engagement is controlled externally through ``engage()`` and
    ``release()``.
    """

    def __init__(
        self,
        robot: UR7eInterface,
        sense: PikaSense,
        gripper,
        pika_to_arm: list,
        position_scale: float = 1.0,
        max_delta_m: float = 1.0,
        servo_hz: int = 50,
        smoothing_alpha: float = 1.0,
        gripper_smoothing_alpha: float = 1.0,
        workspace_bounds: Optional[dict] = None,
        joint_limits: Optional[list] = None,
        max_tilt_from_down_rad: Optional[float] = None,
        ik_mode: str = "ur_native_servol",
        base_bias_min_radius_m: float = 0.05,
        servo_lookahead_s: float = 0.2,
        servo_gain: float = 100.0,
        max_lin_vel_m_s: float = 0.30,
        max_ang_vel_rad_s: float = 1.50,
        max_joint_vel_rad_s: float = 1.50,
        base_limit_rad: float = 2.6,
        base_limit_damping_threshold: float = 0.8,
    ):
        self.robot = robot
        self.sense = sense
        self.gripper = gripper
        self.pika_to_arm = list(pika_to_arm)
        self.position_scale = float(position_scale)
        self.max_delta_m = float(max_delta_m)
        self.servo_hz = int(servo_hz)
        self.dt = 1.0 / self.servo_hz

        # EMA low-pass on the tracker pose and gripper command to damp
        # operator hand tremor before it reaches either the arm or dataset.
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.gripper_smoothing_alpha = float(np.clip(gripper_smoothing_alpha,
                                                     0.0, 1.0))
        self._smoothed_pos: Optional[np.ndarray] = None
        self._smoothed_quat: Optional[np.ndarray] = None
        self._smoothed_gripper: Optional[float] = None

        self.servo_lookahead_s = float(servo_lookahead_s)
        self.servo_gain = float(servo_gain)

        self.max_lin_vel = float(max_lin_vel_m_s)
        self.max_ang_vel = float(max_ang_vel_rad_s)
        self.max_joint_vel = float(max_joint_vel_rad_s)
        self._last_sent_pos: Optional[np.ndarray] = None
        self._last_sent_quat: Optional[np.ndarray] = None
        self._last_sent_q: Optional[np.ndarray] = None

        self._dbg_ik_called = 0
        self._dbg_ik_failed = 0
        self._dbg_servoj = 0
        self._dbg_servol_fallback = 0
        self._dbg_last_print = 0.0

        self.tools = MathTools()

        self._initial_pose_rpy: Optional[list] = None
        self._base_pose: Optional[list] = None
        self._tracker_xyzrpy: Optional[list] = None
        self._teleop_active = False
        self._last_gripper_cmd = 0.0

        self._servo_fail_streak = 0
        self._servo_fail_limit = 30
        self.controller_lost = False
        self.aborted = False
        self.abort_reason: str = ""

        self.workspace_bounds = workspace_bounds or {}
        self.joint_limits = joint_limits
        self.max_tilt_from_down_rad = max_tilt_from_down_rad
        self._reject_log_t = 0.0
        self._reject_count = 0

        valid_modes = ("ur_native_servol", "base_biased_servoj", "base_biased_adaptive")
        self.ik_mode = ik_mode if ik_mode in valid_modes else "ur_native_servol"
        self.base_bias_min_radius_m = float(base_bias_min_radius_m)

        self.base_limit_rad = float(base_limit_rad)
        self.base_limit_damping_threshold = float(base_limit_damping_threshold)
        self._last_base_bias = 0.0

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # public API used by the recording loop
    # ------------------------------------------------------------------

    def start(self):
        # Snapshot the arm's current pose as the "zero" reference.
        self._initial_pose_rpy = self._tcp_actual_xyzrpy()
        self._base_pose = list(self._initial_pose_rpy)
        if not self.sense.wait_for_tracker(timeout=30.0):
            print("[Teleop] !! Tracker pose not received in 30 s. Continuing "
                  "anyway — teleop will activate as soon as the tracker locks on.")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.robot.servo_stop()
        except Exception:
            pass

    def engage(self) -> None:
        self._initial_pose_rpy = self._tcp_actual_xyzrpy()
        if self._tracker_xyzrpy is not None:
            self._base_pose = list(self._tracker_xyzrpy)
        else:
            self._base_pose = list(self._initial_pose_rpy)
        self._teleop_active = True
        print("[Teleop] >> ENGAGED by F2")

    def release(self) -> None:
        self._teleop_active = False
        self._initial_pose_rpy = self._tcp_actual_xyzrpy()
        self._base_pose = list(self._initial_pose_rpy)
        try:
            self.robot.servo_stop()
        except Exception:
            pass
        print("[Teleop] << RELEASED by F2")

    def get_command_snapshot(self) -> Dict[str, float]:
        """Return the latest commanded gripper rad — read from the recording loop."""
        with self._lock:
            return {"gripper_cmd": float(self._last_gripper_cmd)}

    @property
    def is_teleop_active(self) -> bool:
        return self._teleop_active

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _tcp_actual_xyzrpy(self) -> list:
        rotvec = self.robot.get_tcp_pose().tolist()
        roll, pitch, yaw = self.tools.rotvec_to_rpy(rotvec[3:6])
        return [rotvec[0], rotvec[1], rotvec[2], roll, pitch, yaw]

    def _adjust_pika_to_arm(self, x, y, z, rx, ry, rz):
        """Apply the rigid offset between the pika frame and arm tool frame."""
        T = self.tools.xyzrpy2Mat(x, y, z, rx, ry, rz)
        adj = self.tools.xyzrpy2Mat(*self.pika_to_arm)
        T = T @ adj
        return self.tools.mat2xyzrpy(T)

    def _refresh_tracker_pose(self):
        """Read the tracker pose, smooth it, then project into the arm frame."""
        pose = self.sense.get_tracker_pose()
        if pose is None:
            return
        position, rotation = pose
        raw_pos = np.asarray(position, dtype=float)
        raw_quat = self.tools.quat_normalize(np.asarray(rotation, dtype=float))

        if self.smoothing_alpha < 1.0 and self._smoothed_pos is not None:
            self._smoothed_pos = (
                self.smoothing_alpha * raw_pos
                + (1.0 - self.smoothing_alpha) * self._smoothed_pos
            )
        else:
            self._smoothed_pos = raw_pos

        if self.smoothing_alpha < 1.0 and self._smoothed_quat is not None:
            self._smoothed_quat = self.tools.slerp(
                self._smoothed_quat, raw_quat, self.smoothing_alpha
            )
        else:
            self._smoothed_quat = raw_quat

        roll, pitch, yaw = self.tools.quaternion_to_rpy(
            self._smoothed_quat[0], self._smoothed_quat[1],
            self._smoothed_quat[2], self._smoothed_quat[3],
        )
        sx, sy, sz = self._smoothed_pos
        self._tracker_xyzrpy = list(self._adjust_pika_to_arm(
            sx, sy, sz, roll, pitch, yaw,
        ))

    def _filter_target(self, target_xyzrpy: list) -> tuple[Optional[list], str]:
        """Apply workspace, tilt, and IK joint-limit safety filters."""
        out = list(target_xyzrpy)

        for axis_idx, axis_name in ((0, "x"), (1, "y"), (2, "z")):
            bounds = self.workspace_bounds.get(axis_name)
            if bounds and len(bounds) == 2 and bounds[0] is not None:
                out[axis_idx] = max(bounds[0], min(bounds[1], out[axis_idx]))

        if self.max_tilt_from_down_rad is not None:
            T = self.tools.xyzrpy2Mat(*out)
            tool_z = np.array([T[0, 2], T[1, 2], T[2, 2]])
            cos_angle = float(np.clip(-tool_z[2], -1.0, 1.0))
            tilt = float(np.arccos(cos_angle))
            if tilt > self.max_tilt_from_down_rad:
                return None, f"tilt={np.degrees(tilt):.0f}deg"

        if self.joint_limits is not None:
            try:
                rotvec = self.tools.rpy_to_rotvec(out[3], out[4], out[5])
                pose_for_ik = [out[0], out[1], out[2],
                               float(rotvec[0]), float(rotvec[1]),
                               float(rotvec[2])]
                rtde_c = self.robot._rtde_c
                q_near = self.robot.get_state()["joint_positions"].tolist()
                q_pred = rtde_c.getInverseKinematics(pose_for_ik, q_near)
                if q_pred and len(q_pred) == 6:
                    for i, qi in enumerate(q_pred):
                        bounds = self.joint_limits[i]
                        if not bounds:
                            continue
                        lo, hi = bounds
                        if lo is not None and qi < lo:
                            return None, f"q[{i}]={qi:.2f}<{lo}"
                        if hi is not None and qi > hi:
                            return None, f"q[{i}]={qi:.2f}>{hi}"
            except Exception:
                return None, "IK_failed"

        return out, ""

    def _clamp_tcp_velocity(self, target_xyzrpy: list) -> list:
        """Limit how fast the commanded TCP can move per tick."""
        target_pos = np.array(target_xyzrpy[:3], dtype=float)
        target_quat = self.tools.rpy_to_quat(target_xyzrpy[3],
                                             target_xyzrpy[4],
                                             target_xyzrpy[5])

        if self._last_sent_pos is not None:
            d = target_pos - self._last_sent_pos
            n = float(np.linalg.norm(d))
            cap = self.max_lin_vel * self.dt
            if n > cap and n > 1e-9:
                target_pos = self._last_sent_pos + d * (cap / n)

        if self._last_sent_quat is not None:
            target_quat = self.tools.axis_angle_step(
                self._last_sent_quat, target_quat,
                self.max_ang_vel * self.dt,
            )

        self._last_sent_pos = target_pos.copy()
        self._last_sent_quat = target_quat.copy()

        roll, pitch, yaw = self.tools.quaternion_to_rpy(
            target_quat[0], target_quat[1], target_quat[2], target_quat[3])
        return [float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
                float(roll), float(pitch), float(yaw)]

    def _clamp_joint_velocity(self, q_target: list) -> list:
        """Limit per-tick joint delta before servoJ dispatch."""
        q_target = np.asarray(q_target, dtype=float)
        if self._last_sent_q is None:
            self._last_sent_q = q_target.copy()
            return q_target.tolist()
        cap = self.max_joint_vel * self.dt
        d = q_target - self._last_sent_q
        d = np.clip(d, -cap, +cap)
        clamped = self._last_sent_q + d
        self._last_sent_q = clamped.copy()
        return clamped.tolist()

    def _calc_pose_increment(self) -> Optional[list]:
        """Compute the next arm TCP target from current tracker reading."""
        if (self._tracker_xyzrpy is None
                or self._base_pose is None
                or self._initial_pose_rpy is None):
            return None

        begin = self.tools.xyzrpy2Mat(*self._base_pose)
        zero = self.tools.xyzrpy2Mat(*self._initial_pose_rpy)
        end = self.tools.xyzrpy2Mat(*self._tracker_xyzrpy)

        delta = np.linalg.inv(begin) @ end
        delta = np.array(delta, dtype=float)
        delta[:3, 3] *= self.position_scale
        result = zero @ delta
        return self.tools.mat2xyzrpy(result)

    def _loop(self):
        last_log = 0.0
        last_health = 0.0
        while self._running:
            t0 = time.time()

            if t0 - last_health > 0.5:
                last_health = t0
                health = [
                    ("UR RTDE control script",
                     self.robot.is_control_alive(),
                     "Protective Stop / Local mode / E-stop on the pendant"),
                    ("Pika Sense USB serial",
                     self.sense.is_alive(),
                     "USB cable unplugged, hub power loss, or Sense rebooted"),
                    ("Gripper backend",
                     self.gripper.is_alive(),
                     "USB/socket disconnected, power tripped, or gripper rebooted"),
                ]
                for name, ok, hint in health:
                    if not ok and not self.aborted:
                        self.aborted = True
                        self.controller_lost = True
                        self.abort_reason = (
                            f"{name} dropped — likely cause: {hint}")
                        print(f"\n[Teleop] !!! {self.abort_reason}")
                        print("[Teleop] Auto-stopping. Fix the hardware, "
                              "then restart the script.")
                        self._running = False
                        break
                if not self._running:
                    break

            self._refresh_tracker_pose()

            if self._teleop_active:
                raw_gripper = self.sense.get_encoder_rad()
                if (self.gripper_smoothing_alpha < 1.0
                        and self._smoothed_gripper is not None):
                    gripper_cmd = (
                        self.gripper_smoothing_alpha * raw_gripper
                        + (1.0 - self.gripper_smoothing_alpha) * self._smoothed_gripper
                    )
                else:
                    gripper_cmd = raw_gripper
                self._smoothed_gripper = gripper_cmd
                gripper_cmd = self.gripper.command_from_pika_encoder(
                    gripper_cmd, self.dt)
                with self._lock:
                    self._last_gripper_cmd = gripper_cmd

            target = self._calc_pose_increment()
            if target is not None and self._teleop_active:
                init_xyz = np.asarray(self._initial_pose_rpy[:3])
                tgt_xyz = np.asarray(target[:3])
                delta_m = float(np.linalg.norm(tgt_xyz - init_xyz))

                if delta_m > self.max_delta_m:
                    if time.time() - last_log > 0.2:
                        logger.warning(
                            "Hold pose: |Δ|=%.0fcm > max %.0fcm",
                            delta_m * 100, self.max_delta_m * 100,
                        )
                        last_log = time.time()
                    try:
                        target = self._tcp_actual_xyzrpy()
                    except Exception:
                        target = None

            if target is not None and self._teleop_active:
                filtered, reject_reason = self._filter_target(target)
                if filtered is None:
                    self._reject_count += 1
                    if time.time() - self._reject_log_t > 0.4:
                        logger.warning("Rejected servoL: %s (cumulative %d)",
                                       reject_reason, self._reject_count)
                        self._reject_log_t = time.time()
                    try:
                        target = self._tcp_actual_xyzrpy()
                    except Exception:
                        target = None
                else:
                    target = filtered

            if target is not None and self._teleop_active:
                target = self._clamp_tcp_velocity(target)

                rotvec = self.tools.rpy_to_rotvec(target[3], target[4], target[5])
                pose_cmd = [target[0], target[1], target[2],
                            float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]

                sent = False
                if self.ik_mode in ("base_biased_servoj", "base_biased_adaptive"):
                    q_current = self.robot.get_state()["joint_positions"].tolist()
                    cur_tcp = self.robot.get_tcp_pose().tolist()
                    q_seed = list(q_current)
                    cur_radius = float(np.hypot(cur_tcp[0], cur_tcp[1]))
                    tgt_radius = float(np.hypot(target[0], target[1]))
                    if (cur_radius > self.base_bias_min_radius_m and
                            tgt_radius > self.base_bias_min_radius_m):
                        cur_az = float(np.arctan2(cur_tcp[1], cur_tcp[0]))
                        tgt_az = float(np.arctan2(target[1], target[0]))
                        d_raw = (tgt_az - cur_az + np.pi) % (2 * np.pi) - np.pi

                        if self.ik_mode == "base_biased_adaptive":
                            proximity = abs(q_current[0]) / self.base_limit_rad
                            if proximity > self.base_limit_damping_threshold:
                                damping_factor = max(0.2, 1.5 - proximity)
                                d_raw = d_raw * damping_factor

                            d_smooth = 0.3 * d_raw + 0.7 * self._last_base_bias
                            self._last_base_bias = d_smooth
                            d = d_smooth
                        else:
                            d = d_raw

                        q_seed[0] = q_current[0] + d

                    q_target = self.robot.get_inverse_kinematics(pose_cmd, q_seed)
                    self._dbg_ik_called += 1
                    if q_target is None:
                        self._dbg_ik_failed += 1
                        if self.ik_mode == "base_biased_adaptive":
                            q_target = self.robot.get_inverse_kinematics(
                                pose_cmd, q_current)

                    if q_target is not None:
                        for i in range(6):
                            while q_target[i] - q_current[i] > np.pi:
                                q_target[i] -= 2 * np.pi
                            while q_target[i] - q_current[i] < -np.pi:
                                q_target[i] += 2 * np.pi
                        q_target = self._clamp_joint_velocity(q_target)
                        try:
                            self.robot.servo_j(
                                q=q_target,
                                speed=0.5, acc=0.5, dt=self.dt,
                                lookahead=self.servo_lookahead_s,
                                gain=self.servo_gain,
                            )
                            sent = True
                            self._dbg_servoj += 1
                            self._servo_fail_streak = 0
                        except Exception as e:
                            self._servo_fail_streak += 1
                            if time.time() - last_log > 0.5:
                                logger.error("servoJ failed: %s", e)
                                last_log = time.time()

                if not sent:
                    try:
                        self.robot.servo_l(
                            pose=pose_cmd,
                            speed=0.5, acc=0.5, dt=self.dt,
                            lookahead=self.servo_lookahead_s,
                            gain=self.servo_gain,
                        )
                        if self.ik_mode == "base_biased_servoj":
                            self._dbg_servol_fallback += 1
                        self._servo_fail_streak = 0
                    except Exception as e:
                        self._servo_fail_streak += 1
                        if time.time() - last_log > 0.5:
                            logger.error("servoL failed: %s", e)
                            last_log = time.time()
                        if (self._servo_fail_streak >= self._servo_fail_limit
                                and not self.controller_lost):
                            self.controller_lost = True
                            print("\n[Teleop] !!! UR control dropped — "
                                  "RTDE control script no longer running.")

            if os.environ.get("DEBUG_TELEOP") == "1":
                if t0 - self._dbg_last_print > 1.0:
                    print(f"[dbg] ik_mode={self.ik_mode}  "
                          f"ik_called={self._dbg_ik_called}  "
                          f"ik_failed={self._dbg_ik_failed}  "
                          f"servoJ={self._dbg_servoj}  "
                          f"servoL_fallback={self._dbg_servol_fallback}  "
                          f"reject={self._reject_count}",
                          flush=True)
                    self._dbg_last_print = t0

            elapsed = time.time() - t0
            sleep = self.dt - elapsed
            if sleep > 0:
                time.sleep(sleep)
