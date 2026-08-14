#!/usr/bin/env python3
"""Run real UR7e VLA inference with RGB-D OBBs and FK-Jacobian CBF-QP safety."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for path in (REPO_ROOT, OPENPI_CLIENT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_scripts.real_cbf_qp import point_jacobian_fd, project_joint_delta_qp, select_point_flow_constraints
from real_scripts.real_robot_adapter import load_camera_calibrations
from real_scripts.real_scene_geometry import RealSceneOBBBuilder
from real_scripts.ur7e_collision_mesh import UR7ePikaCollisionSurfacePointSampler
from real_scripts.ur7e_vla_reference_platform import DEFAULT_INFERENCE_ROOT, UR7eReferenceVLAPlatform, platform_config_from_args


DEFAULT_PIKA_MESH = REPO_ROOT / "assets" / "robot_models" / "pika_gripper" / "collision" / "pika_gripper_full_collision.stl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--front-serial", default=None)
    parser.add_argument("--side-serial", default=None)
    parser.add_argument("--wrist-serial", default=None)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--policy-server-host", default="127.0.0.1")
    parser.add_argument("--policy-server-port", type=int, default=8000)
    parser.add_argument("--safety-server-host", default=None)
    parser.add_argument("--safety-server-port", type=int, default=None)
    parser.add_argument("--camera-calibration", type=Path, required=True)
    parser.add_argument("--pika-mount-transform-json", type=Path, required=True)
    parser.add_argument("--pika-full-collision-mesh", type=Path, default=DEFAULT_PIKA_MESH)
    parser.add_argument("--points-per-link", type=int, default=128)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--workspace-bounds", nargs=6, type=float, required=True)
    parser.add_argument("--table-z", type=float, required=True)
    parser.add_argument("--collision-margin-m", type=float, default=0.0)
    parser.add_argument("--trigger-margin-m", type=float, default=0.02)
    parser.add_argument("--cbf-alpha", type=float, default=1.0)
    parser.add_argument("--cbf-fd-epsilon-rad", type=float, default=1e-4)
    parser.add_argument("--cbf-iterations", type=int, default=32)
    parser.add_argument("--execute", action="store_true", help="Actually command the robot. Omit for a non-motion integration test.")
    return parser.parse_args()


def _client(host: str, port: int):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=host, port=port)


def _policy_query(policy, observation: dict) -> tuple[np.ndarray, np.ndarray]:
    result = policy.infer(observation)
    if "actions" not in result or "prefix_tokens" not in result:
        raise KeyError("Policy endpoint must return actions and prefix_tokens")
    actions = np.asarray(result["actions"], dtype=np.float32)
    prefix = np.asarray(result["prefix_tokens"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or prefix.ndim != 2:
        raise ValueError(f"Unexpected policy result shapes actions={actions.shape}, prefix={prefix.shape}")
    return actions, prefix


def _predict_safety(policy, prefix_tokens: np.ndarray, current_link_points: np.ndarray) -> np.ndarray:
    result = policy.infer({"safety_only": True, "prefix_tokens": prefix_tokens, "current_link_points": current_link_points})
    if "pred_link_points" not in result:
        raise KeyError("Safety endpoint did not return pred_link_points; start it with --safety-checkpoint")
    prediction = np.asarray(result["pred_link_points"], dtype=np.float32)
    if prediction.ndim != 4 or prediction.shape[1:] != current_link_points.shape:
        raise ValueError(f"Safety prediction shape {prediction.shape} does not match current links {current_link_points.shape}")
    return prediction


def _action_with_joint_target(platform: UR7eReferenceVLAPlatform, nominal_action: np.ndarray, qpos: np.ndarray, target: np.ndarray) -> np.ndarray:
    action = np.asarray(nominal_action, dtype=np.float32).copy()
    indices = np.asarray(platform.app_config.robot.action_joint_indices, dtype=np.int64)
    if platform.action_mode == "joint_position":
        action[indices] = target
    elif platform.action_mode == "joint_delta":
        scale = float(platform.app_config.robot.joint_delta_scale)
        if abs(scale) <= 1e-12:
            raise ValueError("reference robot joint_delta_scale must be non-zero")
        action[indices] = (target - qpos) / scale
    else:
        raise ValueError(f"Unsupported reference action mode {platform.action_mode!r}")
    return action


def run(args: argparse.Namespace) -> None:
    sampler = UR7ePikaCollisionSurfacePointSampler(
        points_per_link=args.points_per_link,
        pika_mount_transform_json=args.pika_mount_transform_json,
        pika_full_collision_mesh=args.pika_full_collision_mesh,
    )
    platform = UR7eReferenceVLAPlatform(platform_config_from_args(args))
    calibrations = load_camera_calibrations(args.camera_calibration)
    scene_builder = RealSceneOBBBuilder(
        calibrations=calibrations,
        sampler=sampler,
        table_z=args.table_z,
        workspace_bounds=args.workspace_bounds,
    )
    vla = _client(args.policy_server_host, args.policy_server_port)
    safety = _client(args.safety_server_host or args.policy_server_host, args.safety_server_port or args.policy_server_port)
    period_s = 1.0 / platform.control_hz
    action_chunk: np.ndarray | None = None
    prefix_tokens: np.ndarray | None = None
    action_offset = 0
    replan_offset = 0
    platform.start()
    try:
        for step in range(args.max_steps):
            started = time.monotonic()
            observation, frames, qpos, _gripper = platform.observe(args.prompt)
            if action_chunk is None or action_offset >= len(action_chunk) or replan_offset >= args.replan_steps:
                action_chunk, prefix_tokens = _policy_query(vla, observation)
                action_offset = 0
                replan_offset = 0
            if prefix_tokens is None:
                raise RuntimeError("Missing prefix tokens for safety prediction")
            current = sampler.link_points(qpos)
            scene = scene_builder.build(frames, qpos)
            prediction = _predict_safety(safety, prefix_tokens, current)
            nominal_action = action_chunk[action_offset]
            nominal_target = np.asarray(platform.robot.action_to_target(nominal_action, qpos), dtype=np.float32)
            nominal_delta = nominal_target - qpos
            robot_cfg = platform.app_config.robot
            lower = np.maximum(np.asarray(robot_cfg.joint_min_rad, dtype=np.float32) - qpos, -float(robot_cfg.max_joint_step_rad))
            upper = np.minimum(np.asarray(robot_cfg.joint_max_rad, dtype=np.float32) - qpos, float(robot_cfg.max_joint_step_rad))
            constraints = select_point_flow_constraints(
                current,
                prediction,
                scene.boxes,
                collision_margin_m=args.collision_margin_m,
                trigger_margin_m=args.trigger_margin_m,
            )
            jacobian = point_jacobian_fd(sampler, qpos, epsilon_rad=args.cbf_fd_epsilon_rad) if constraints else None
            if jacobian is None:
                safe_delta, info = nominal_delta, {"triggered": False, "success": True, "constraint_count": 0, "max_violation": 0.0}
            else:
                safe_delta, info = project_joint_delta_qp(
                    nominal_delta,
                    jacobian,
                    constraints,
                    lower_delta=lower,
                    upper_delta=upper,
                    alpha=args.cbf_alpha,
                    iterations=args.cbf_iterations,
                )
            safe_target = qpos + safe_delta
            if not bool(info["success"]):
                safe_target = qpos.copy()  # fail closed: hold arm; preserve gripper command below
            safe_action = _action_with_joint_target(platform, nominal_action, qpos, safe_target)
            platform.execute_policy_action(safe_action, current_qpos=qpos)
            print(
                f"[safety] step={step} obbs={len(scene.boxes)} constraints={info['constraint_count']} "
                f"triggered={info['triggered']} success={info['success']} violation={float(info['max_violation']):.3e}"
            )
            action_offset += 1
            replan_offset += 1
            remaining = period_s - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        platform.close()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
