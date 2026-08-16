#!/usr/bin/env python3
"""Collect a real UR7e/PiKA VLA surface-point-flow dataset.

This is intentionally independent of LIBERO collection code.  The VLA action
contract follows ``ur7e_inference``: each action is a 7-D joint target/delta
according to its config, never an end-effector delta.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for path in (REPO_ROOT, OPENPI_CLIENT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_scripts.real_safety_dataset import RealSafetyDatasetBuffer, save_real_safety_dataset
from real_scripts.ur7e_collision_mesh import UR7ePikaCollisionSurfacePointSampler
from real_scripts.ur7e_safety_episode_recorder import UR7eSafetyEpisodeRecorder
from real_scripts.ur7e_vla_reference_platform import DEFAULT_INFERENCE_ROOT, UR7eReferenceVLAPlatform, platform_config_from_args


DEFAULT_PIKA_MESH = REPO_ROOT / "assets" / "robot_models" / "pika_gripper" / "collision" / "pika_gripper_full_collision.stl"


@dataclass(frozen=True)
class ReplanRecord:
    prefix_tokens: np.ndarray
    action_chunk: np.ndarray
    start_qpos: np.ndarray
    step_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True, help="New directory for raw real-hardware episode data.")
    parser.add_argument("--camera-calibration", type=Path, required=True, help="Measured fixed-camera calibration copied into the raw episode for offline RGB-D preprocessing.")
    parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT)
    parser.add_argument("--inference-config", type=Path, required=True, help="ur7e_inference YAML config defining action semantics and limits.")
    parser.add_argument("--front-serial", default=None)
    parser.add_argument("--side-serial", default=None)
    parser.add_argument("--wrist-serial", default=None)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--policy-server-host", default="127.0.0.1")
    parser.add_argument("--policy-server-port", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--points-per-link", type=int, default=128)
    parser.add_argument("--pika-mount-transform-json", type=Path, required=True)
    parser.add_argument("--pika-full-collision-mesh", type=Path, default=DEFAULT_PIKA_MESH)
    parser.add_argument("--execute", action="store_true", help="Actually command the robot. Omit for sensing/policy dry run.")
    parser.add_argument("--episode-queue-size", type=int, default=128)
    parser.add_argument("--drop-episode-records-on-backpressure", action="store_true")
    return parser.parse_args()


def _load_policy(host: str, port: int):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=host, port=port)


def _infer(policy, observation: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    request_ns = time.monotonic_ns()
    result = policy.infer(observation)
    response_ns = time.monotonic_ns()
    if "actions" not in result or "prefix_tokens" not in result:
        raise KeyError("Policy server must return both actions and prefix_tokens; use scripts/serve_pi05_prefix_policy.py")
    actions = np.asarray(result["actions"], dtype=np.float32)
    prefix = np.asarray(result["prefix_tokens"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 1:
        raise ValueError(f"Expected VLA actions shape (T,7), got {actions.shape}")
    if prefix.ndim != 2:
        raise ValueError(f"Expected VLA prefix_tokens shape (N,D), got {prefix.shape}")
    return actions, prefix, request_ns, response_ns


def run_collection(args: argparse.Namespace) -> int:
    if args.points_per_link < 2 or args.replan_steps < 1 or args.max_steps < 1:
        raise ValueError("points-per-link must be >=2 and replan/max-steps must be positive")
    sampler = UR7ePikaCollisionSurfacePointSampler(
        points_per_link=args.points_per_link,
        pika_mount_transform_json=args.pika_mount_transform_json,
        pika_full_collision_mesh=args.pika_full_collision_mesh,
    )
    platform = UR7eReferenceVLAPlatform(platform_config_from_args(args))
    policy = _load_policy(args.policy_server_host, args.policy_server_port)
    recorder = UR7eSafetyEpisodeRecorder(
        args.episode_dir,
        queue_size=args.episode_queue_size,
        drop_on_backpressure=args.drop_episode_records_on_backpressure,
        manifest={
            "prompt": args.prompt,
            "policy_config": platform.policy_name,
            "policy_server": f"{args.policy_server_host}:{args.policy_server_port}",
            "action_mode": platform.action_mode,
            "control_hz": platform.control_hz,
            "coordinate_frame": "ur_base",
            "surface_model_hash": sampler.mesh_model_hash,
            "point_identity_version": sampler.point_identity_version,
        },
        calibration_path=args.camera_calibration,
    )
    records: list[ReplanRecord] = []
    surface_frames: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    action_chunk: np.ndarray | None = None
    action_offset = 0
    replan_offset = 0
    active_query_id: int | None = None
    next_query_id = 0
    period_s = 1.0 / platform.control_hz
    platform.start()
    try:
        for step_id in range(args.max_steps):
            tick_started = time.monotonic()
            observation, frames, qpos, gripper = platform.observe(args.prompt)
            surface_frames.append(sampler.link_points(qpos))
            need_query = action_chunk is None or action_offset >= len(action_chunk) or replan_offset >= args.replan_steps
            if need_query:
                action_chunk, prefix_tokens, request_ns, response_ns = _infer(policy, observation)
                action_offset = 0
                replan_offset = 0
                active_query_id = next_query_id
                next_query_id += 1
                recorder.record_policy_query(
                    query_id=active_query_id,
                    request_timestamp_ns=request_ns,
                    response_timestamp_ns=response_ns,
                    prefix_tokens=prefix_tokens,
                    action_chunk=action_chunk,
                )
                records.append(ReplanRecord(prefix_tokens, action_chunk, qpos.copy(), step_id))
            action = np.asarray(action_chunk[action_offset], dtype=np.float32)
            recorder.record_tick(
                tick_id=step_id,
                frames=frames,
                qpos=qpos,
                gripper_state=gripper,
                robot_state_timestamp_ns=time.monotonic_ns(),
                policy_query_id=active_query_id,
                action_index=action_offset,
                commanded_action=action,
            )
            platform.execute_policy_action(action, current_qpos=qpos)
            executed_actions.append(action.copy())
            action_offset += 1
            replan_offset += 1
            remaining = period_s - (time.monotonic() - tick_started)
            if remaining > 0.0:
                time.sleep(remaining)
        # Measured terminal state closes the final available point-flow window.
        _obs, _frames, final_qpos, _gripper = platform.observe(args.prompt)
        surface_frames.append(sampler.link_points(final_qpos))
    finally:
        try:
            platform.close()
        finally:
            recorder.close()

    trajectory = np.stack(surface_frames).astype(np.float32)
    buffer = RealSafetyDatasetBuffer()
    for record in records:
        horizon = int(record.action_chunk.shape[0])
        if record.step_id + horizon >= trajectory.shape[0] or record.step_id + horizon > len(executed_actions):
            continue
        buffer.append(
            prefix_tokens=record.prefix_tokens,
            # The future surface trajectory follows the actually executed
            # re-planned commands, not necessarily every unexecuted row of the
            # original VLA chunk.
            action_chunk=np.stack(executed_actions[record.step_id : record.step_id + horizon]).astype(np.float32),
            start_joint_vector=record.start_qpos,
            link_trajectory=trajectory[record.step_id : record.step_id + horizon + 1],
            episode_id=0,
            step_id=record.step_id,
        )
    save_real_safety_dataset(
        args.output,
        buffer=buffer,
        link_names=np.asarray(sampler.link_names),
        point_ids=sampler.point_ids,
        local_link_points=sampler.local_link_points,
        surface_model_hash=sampler.mesh_model_hash,
        point_identity_version=sampler.point_identity_version,
        policy_config=platform.policy_name,
        action_mode=platform.action_mode,
        control_hz=platform.control_hz,
    )
    return len(buffer)


def main() -> None:
    args = parse_args()
    count = run_collection(args)
    print(f"[done] wrote {count} real UR7e/PiKA surface-flow samples to {args.output}")


if __name__ == "__main__":
    main()
