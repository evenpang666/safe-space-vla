#!/usr/bin/env python3
"""Run the canonical dual-UR7e PI05 + obstacle OBB + CBF-QP loop.

The obstacle service is the sole owner of the front D455 and RTDE receive
streams. This executor consumes its timestamped snapshot, queries the jointly
trained PI05 action/point-flow service, and projects the two-arm joint delta
through one coupled 12-DOF CBF problem. It is dry-run by default.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT = REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for path in (REPO_ROOT, OPENPI_CLIENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler  # noqa: E402
from real_scripts.real_cbf_qp import (  # noqa: E402
    OrientedBox, point_jacobian_fd, project_joint_delta_qp,
    select_inter_arm_constraints, select_point_flow_constraints,
)
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path  # noqa: E402


EXECUTE_ACK = "I_UNDERSTAND_DUAL_UR7E_WILL_MOVE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--obstacle-url", default=None)
    parser.add_argument("--points-per-link", type=int, default=None)
    parser.add_argument("--poll-hz", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-unverified-right-tcp", action="store_true", help="Allow provisional EPick FK in dry-run only.")
    parser.add_argument("--execute-ack", default="", help=f"With --execute, must equal {EXECUTE_ACK!r}.")
    return parser.parse_args()


def _snapshot(url: str, timeout_s: float) -> dict:
    request = Request(url, headers={"Accept-Encoding": "identity", "Cache-Control": "no-cache"})
    with urlopen(request, timeout=timeout_s) as response:
        value = json.loads(response.read())
    if value.get("schema") != "dual_ur7e_obstacle_snapshot_v2" or value.get("coordinate_frame") != "left_base":
        raise ValueError("Obstacle endpoint does not expose the v2 left_base safety snapshot")
    return value


def _rgb(snapshot: dict) -> np.ndarray:
    payload = base64.b64decode(snapshot["rgb_front_jpeg_base64"], validate=True)
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Obstacle snapshot contains an invalid front RGB JPEG")
    return bgr[..., ::-1]


def _boxes(snapshot: dict) -> list[OrientedBox]:
    return [OrientedBox(np.asarray(item["center"], np.float32), np.asarray(item["axes"], np.float32), np.asarray(item["half_sizes"], np.float32)) for item in snapshot["obbs"]]


def _qpos(snapshot: dict) -> np.ndarray:
    q = np.asarray((*snapshot["qpos"]["left"], *snapshot["qpos"]["right"]), dtype=np.float32)
    if q.shape != (12,) or not np.isfinite(q).all():
        raise ValueError("Obstacle snapshot qpos is not a finite dual-arm vector")
    return q


def _gripper(snapshot: dict) -> np.ndarray:
    value = snapshot.get("gripper_position", {})
    return np.asarray((value.get("left", 0.0), value.get("right", 0.0)), dtype=np.float32)


def _joint_delta(action: np.ndarray) -> np.ndarray:
    if action.shape != (14,):
        raise ValueError(f"Dual-arm executor requires one 14-D action, got {action.shape}")
    return np.concatenate((action[:6], action[7:13])).astype(np.float32)


class DualRTDECommander:
    """Explicitly armed, rate-limited dual servoJ writer; no gripper guessing."""

    def __init__(self, left_ip: str, right_ip: str, period_s: float) -> None:
        from rtde_control import RTDEControlInterface

        self.controls = (RTDEControlInterface(left_ip), RTDEControlInterface(right_ip))
        self.period_s = float(period_s)

    def send(self, qpos: np.ndarray, safe_delta: np.ndarray) -> None:
        targets = (qpos[:6] + safe_delta[:6], qpos[6:] + safe_delta[6:])

        def command(pair) -> bool:
            control, target = pair
            return bool(control.servoJ(target.tolist(), 0.10, 0.30, self.period_s, 0.10, 300))

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(command, zip(self.controls, targets, strict=True)))
        if not all(results):
            raise RuntimeError("At least one UR controller rejected servoJ")

    def close(self) -> None:
        for control in self.controls:
            try:
                control.servoStop(1.0)
            finally:
                control.disconnect()


def run(args: argparse.Namespace) -> None:
    config, config_path = load_project_config(args.project_config)
    safety = config["safety"]
    points_per_link = int(args.points_per_link or config["preprocess"]["points_per_link"])
    obstacle_url = args.obstacle_url or str(safety["obstacle_url"])
    period_s = 1.0 / float(args.poll_hz)
    if args.execute and args.execute_ack != EXECUTE_ACK:
        raise ValueError(f"Physical execution requires --execute-ack {EXECUTE_ACK}")
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    sampler = DualUR7eSurfacePointSampler(points_per_link=points_per_link, project_config=config_path)
    if not sampler.tcp_transform_verified["right_arm"]:
        if args.execute or not args.allow_unverified_right_tcp:
            raise RuntimeError(
                "Right flange-to-active-TCP is not verified; physical execution is blocked. "
                "For geometry-only testing, pass --allow-unverified-right-tcp without --execute."
            )
    commander = None
    if args.execute:
        with resolve_repo_path(config["quest3"]["hardware_config"]).open(encoding="utf-8") as handle:
            import yaml
            hardware = yaml.safe_load(handle)
        commander = DualRTDECommander(str(hardware["arms"]["left_arm"]["robot_ip"]), str(hardware["arms"]["right_arm"]["robot_ip"]), period_s)
        print("[execute] arm servoJ enabled; gripper outputs remain disabled until an EPick command interface is configured")
    last_frame = -1
    try:
        for step in range(args.max_steps):
            began = time.monotonic()
            snap = _snapshot(obstacle_url, timeout_s=float(safety["stale_timeout_s"]))
            age_s = (time.monotonic_ns() - int(snap["host_timestamp_ns"])) / 1e9
            if age_s < -0.1 or age_s > float(safety["stale_timeout_s"]):
                raise RuntimeError(f"Obstacle snapshot is stale or from another clock domain: age={age_s:.3f}s")
            if int(snap["frame"]) == last_frame:
                time.sleep(max(0.0, period_s - (time.monotonic() - began)))
                continue
            last_frame = int(snap["frame"])
            qpos, grip = _qpos(snap), _gripper(snap)
            full_points = sampler.link_points(qpos, grip).reshape(-1, 3)
            result = policy.infer({"prompt": args.prompt, "rgb_front": _rgb(snap), "qpos": qpos, "robot_points": full_points})
            actions = np.asarray(result["actions"], dtype=np.float32)
            prediction = np.asarray(result["predicted_robot_points"], dtype=np.float32)
            selected = np.asarray(result["selected_point_indices"], dtype=np.int64)
            if str(result.get("surface_model_hash", "")) != sampler.model_hash(2):
                raise ValueError("Policy checkpoint surface mesh/hash differs from the live dual-arm sampler")
            if int(result.get("points_per_link", 0)) != points_per_link:
                raise ValueError("Policy checkpoint points_per_link differs from the executor")
            if actions.ndim != 2 or actions.shape[1] != 14 or prediction.ndim != 3 or prediction.shape[1:] != (len(selected), 3):
                raise ValueError(f"Policy contract mismatch: actions={actions.shape}, points={prediction.shape}")
            current = full_points[selected]
            boxes = _boxes(snap)
            obstacle_constraints = select_point_flow_constraints(current, prediction, boxes, collision_margin_m=float(safety["collision_margin_m"]), trigger_margin_m=float(safety["trigger_margin_m"]))
            left_full_count = len(sampler.left_link_names) * points_per_link
            arm_constraints = select_inter_arm_constraints(current, prediction, selected < left_full_count, minimum_distance_m=float(safety["inter_arm_margin_m"]), trigger_margin_m=float(safety["trigger_margin_m"]))
            constraints = [*obstacle_constraints, *arm_constraints]
            nominal = _joint_delta(actions[0])
            limit = float(safety["max_joint_step_rad"])
            if constraints:
                jacobian = point_jacobian_fd(sampler, qpos, gripper_position=grip, point_indices=selected)
                safe_delta, info = project_joint_delta_qp(nominal, jacobian, constraints, lower_delta=np.full(12, -limit), upper_delta=np.full(12, limit), alpha=float(safety["cbf_alpha"]), iterations=int(safety["cbf_iterations"]))
            else:
                safe_delta = np.clip(nominal, -limit, limit)
                info = {"triggered": False, "success": True, "constraint_count": 0, "max_violation": 0.0}
            if not bool(info["success"]):
                safe_delta = np.zeros(12, dtype=np.float32)
            if commander is not None:
                commander.send(qpos, safe_delta)
            print(json.dumps({
                "step": step, "source_frame": last_frame, "snapshot_age_s": round(age_s, 4), "obbs": len(boxes),
                "obstacle_constraints": len(obstacle_constraints), "inter_arm_constraints": len(arm_constraints), "cbf": info,
                "nominal_joint_delta": nominal.tolist(), "safe_joint_delta": safe_delta.tolist(),
                "gripper_targets_not_executed": [float(actions[0, 6]), float(actions[0, 13])],
                "mode": "execute_arms_only" if commander is not None else "dry_run",
            }, ensure_ascii=False))
            if args.once:
                break
            time.sleep(max(0.0, period_s - (time.monotonic() - began)))
    finally:
        if commander is not None:
            commander.close()


if __name__ == "__main__":
    run(parse_args())
