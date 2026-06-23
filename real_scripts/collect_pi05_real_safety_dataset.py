#!/usr/bin/env python3
"""Collect SafetyModule training data while running a PI05 policy on a UR arm."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for path in (REPO_ROOT, OPENPI_CLIENT_SRC):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from real_scripts.real_robot_adapter import (  # noqa: E402
    ReplayJsonlAdapter,
    UR7ELinkPointSampler,
    fuse_rgbd_frames,
    load_camera_calibrations,
)
from scripts.collect_pi05_libero_safety_decoder_dataset import (  # noqa: E402
    CollectedSampleBuffer,
    ReplanSampleRecord,
    append_surface_trajectory_samples,
    save_collected_dataset,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pi05_safety_decoder" / "pi05_real_ur_safety_dataset.npz"


@dataclass(frozen=True)
class RealReplanSample:
    prefix_tokens: np.ndarray
    action_chunk: np.ndarray
    start_joint_vector: np.ndarray
    rollout_id: int
    step_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", required=True, help="Language instruction sent to the PI05 policy.")
    parser.add_argument("--policy-server-host", default="127.0.0.1")
    parser.add_argument("--policy-server-port", type=int, default=8000)
    parser.add_argument("--policy-config", default="pi05_ur7")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--points-per-link", type=int, default=128)
    parser.add_argument("--gripper-width", type=float, default=0.085)
    parser.add_argument("--robot-filter-radius", type=float, default=0.045)
    parser.add_argument("--pointcloud-stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--workspace-bounds", nargs=6, type=float, default=None)
    parser.add_argument("--camera-calibration", type=Path, default=None)
    parser.add_argument("--debug-pointcloud-output", type=Path, default=None)
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Import path 'module:factory'. The factory is called with no arguments and must return a RealRobotAdapter. "
            "Use --replay-jsonl for offline smoke tests."
        ),
    )
    parser.add_argument("--replay-jsonl", type=Path, default=None)
    return parser.parse_args()


def build_ur7_policy_input(observation: dict, *, prompt: str) -> dict:
    """Build the raw UR payload expected by the OpenPI UR example transforms."""
    qpos = np.asarray(observation["qpos"], dtype=np.float32).reshape(-1)
    if qpos.size < 6:
        raise ValueError(f"observation['qpos'] must contain at least 6 values, got {qpos.size}")
    gripper = np.asarray(observation.get("gripper", [0.0]), dtype=np.float32).reshape(-1)
    return {
        "base_rgb": np.ascontiguousarray(np.asarray(observation["front_rgb"], dtype=np.uint8)),
        "side_rgb": np.ascontiguousarray(np.asarray(observation["side_rgb"], dtype=np.uint8)),
        "wrist_rgb": np.ascontiguousarray(np.asarray(observation.get("wrist_rgb", observation["front_rgb"]), dtype=np.uint8)),
        "joints": qpos[:6].astype(np.float32),
        "gripper": gripper.astype(np.float32),
        "prompt": str(prompt),
    }


def load_policy_client(host: str, port: int):
    from openpi_client import websocket_client_policy

    return websocket_client_policy.WebsocketClientPolicy(host=host, port=port)


def query_policy_action_and_prefix(policy, payload: dict) -> tuple[np.ndarray, np.ndarray]:
    result = policy.infer(payload)
    if "actions" not in result:
        raise KeyError("Policy response must contain 'actions'")
    if "prefix_tokens" not in result:
        raise KeyError(
            "Policy response must contain 'prefix_tokens'. Start scripts/serve_pi05_prefix_policy.py, "
            "not OpenPI's default serve_policy.py."
        )
    return np.asarray(result["actions"], dtype=np.float32), np.asarray(result["prefix_tokens"], dtype=np.float32)


def load_adapter(args: argparse.Namespace):
    if args.replay_jsonl is not None:
        return ReplayJsonlAdapter(args.replay_jsonl)
    if args.adapter is None:
        raise ValueError("Provide --adapter module:factory for real hardware, or --replay-jsonl for offline replay.")
    module_name, sep, factory_name = str(args.adapter).partition(":")
    if not sep:
        raise ValueError("--adapter must have form module:factory")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    return factory()


def append_real_trajectory_samples(
    records: list[RealReplanSample],
    *,
    surface_frames: np.ndarray,
    link_names: np.ndarray,
    output: Path,
    max_samples: int,
    policy_config: str,
    checkpoint_dir: str,
    points_per_link: int,
) -> int:
    buffer = CollectedSampleBuffer()
    libero_records = [
        ReplanSampleRecord(
            prefix_tokens=record.prefix_tokens,
            action_chunk=record.action_chunk,
            start_joint_vector=record.start_joint_vector,
            task_id=0,
            rollout_id=record.rollout_id,
            step_id=record.step_id,
        )
        for record in records
    ]
    appended = append_surface_trajectory_samples(
        buffer,
        records=libero_records,
        surface_frames=np.asarray(surface_frames, dtype=np.float32),
        link_names=np.asarray(link_names),
        max_samples=int(max_samples),
    )
    if appended > 0:
        save_collected_dataset(
            Path(output),
            buffer=buffer,
            link_names=np.asarray(link_names),
            task_suite="real_ur",
            points_per_link=int(points_per_link),
            samples_per_action=1,
            policy_config=policy_config,
            checkpoint_dir=checkpoint_dir,
            skeleton_source="ur7e_fk_surface",
            target_source="real_rollout_surface",
        )
    return appended


def maybe_save_debug_pointcloud(path: Path | None, pointclouds: list, *, link_points: list[np.ndarray]) -> None:
    if path is None or not pointclouds:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scene_points=np.asarray([cloud.scene_points for cloud in pointclouds], dtype=object),
        scene_colors=np.asarray([cloud.scene_colors for cloud in pointclouds], dtype=object),
        environment_points=np.asarray([cloud.environment_points for cloud in pointclouds], dtype=object),
        environment_colors=np.asarray([cloud.environment_colors for cloud in pointclouds], dtype=object),
        link_points=np.asarray(link_points, dtype=np.float32),
    )


def run_collection(args: argparse.Namespace) -> int:
    if args.points_per_link < 2:
        raise ValueError("--points-per-link must be >= 2")
    if args.replan_steps <= 0:
        raise ValueError("--replan-steps must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")

    sampler = UR7ELinkPointSampler(points_per_link=args.points_per_link, gripper_width=args.gripper_width)
    calibrations = load_camera_calibrations(args.camera_calibration) if args.camera_calibration is not None else None
    adapter = load_adapter(args)
    policy = load_policy_client(args.policy_server_host, args.policy_server_port)

    records: list[RealReplanSample] = []
    surface_frames: list[np.ndarray] = []
    debug_clouds = []
    action_chunk: np.ndarray | None = None
    action_offset = 0
    replan_offset = 0

    adapter.reset()
    try:
        for step_id in range(int(args.max_steps)):
            observation = adapter.get_observation()
            qpos = np.asarray(observation["qpos"], dtype=np.float32).reshape(-1)[:6]
            link_points = sampler.link_points(qpos)
            surface_frames.append(link_points)

            if calibrations is not None:
                debug_clouds.append(
                    fuse_rgbd_frames(
                        adapter.get_rgbd_frames(),
                        calibrations,
                        robot_link_points=link_points,
                        stride=args.pointcloud_stride,
                        max_depth=args.max_depth,
                        robot_filter_radius=args.robot_filter_radius,
                        workspace_bounds=args.workspace_bounds,
                    )
                )

            need_query = action_chunk is None or action_offset >= len(action_chunk) or replan_offset >= args.replan_steps
            if need_query:
                payload = build_ur7_policy_input(observation, prompt=args.prompt)
                action_chunk, prefix_tokens = query_policy_action_and_prefix(policy, payload)
                action_offset = 0
                replan_offset = 0
                if len(records) < int(args.max_samples):
                    records.append(
                        RealReplanSample(
                            prefix_tokens=prefix_tokens,
                            action_chunk=action_chunk,
                            start_joint_vector=qpos,
                            rollout_id=0,
                            step_id=step_id,
                        )
                    )

            action = np.asarray(action_chunk[action_offset], dtype=np.float32)
            adapter.execute_action(action)
            action_offset += 1
            replan_offset += 1
            if adapter.is_done():
                break

        final_observation = adapter.get_observation()
        final_qpos = np.asarray(final_observation["qpos"], dtype=np.float32).reshape(-1)[:6]
        surface_frames.append(sampler.link_points(final_qpos))
    finally:
        adapter.close()

    surface_array = np.stack(surface_frames).astype(np.float32)
    appended = append_real_trajectory_samples(
        records,
        surface_frames=surface_array,
        link_names=np.asarray(sampler.link_names),
        output=args.output,
        max_samples=args.max_samples,
        policy_config=args.policy_config,
        checkpoint_dir=args.checkpoint_dir,
        points_per_link=args.points_per_link,
    )
    maybe_save_debug_pointcloud(args.debug_pointcloud_output, debug_clouds, link_points=surface_frames)
    return appended


def main() -> None:
    args = parse_args()
    appended = run_collection(args)
    print(f"[done] wrote {appended} real UR SafetyModule samples to {args.output}")


if __name__ == "__main__":
    main()
