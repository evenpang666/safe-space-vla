#!/usr/bin/env python3
"""Collect upright-block demos and save joint-action episodes for LeRobot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np


DEFAULT_OUTPUT_DIR = Path("outputs/robosuite_collision_scene/lerobot_demos")
DEFAULT_REPO_ID = "local/ur5e_upright_blocks"
DEFAULT_TASK = "pick up the red cube and place it on the plate without touching the yellow blocks"
DEFAULT_CAMERAS = ("frontview", "sideview", "leftsideview")
ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
STATE_NAMES = (*ARM_JOINT_NAMES, "gripper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot repo id, e.g. local/ur5e_upright_blocks.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Fixed language task saved with every episode.")
    parser.add_argument("--num-demos", type=int, default=20, help="Number of successful episodes to save.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum steps per attempt. 0 means unlimited until success, yellow contact, or operator reset.",
    )
    parser.add_argument("--success-hold", type=int, default=10, help="Consecutive success steps before saving.")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--viewer-camera", default="frontview")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--primary-image-camera", default="frontview")
    parser.add_argument("--wrist-image-camera", default="leftsideview")
    parser.add_argument("--device", choices=["keyboard", "spacemouse"], default="keyboard")
    parser.add_argument("--controller", choices=["OSC_POSE", "IK_POSE"], default="OSC_POSE")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--raw-only", action="store_true", help="Only save raw npz episodes; skip LeRobot writing.")
    parser.add_argument(
        "--require-lerobot",
        action="store_true",
        help="Fail immediately if the LeRobot package is not importable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing LeRobot dataset with the same repo id before writing.",
    )
    parser.add_argument(
        "--convert-only",
        type=Path,
        default=None,
        help="Convert a raw episode directory from this script to LeRobot format without opening robosuite.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Headless environment and writer validation.")
    return parser.parse_args()


def import_lerobot():
    try:
        from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset, LEROBOT_HOME
    except Exception:
        try:
            from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

            return LeRobotDataset, HF_LEROBOT_HOME
        except Exception as exc:
            raise ImportError(
                "Could not import LeRobotDataset. Run conversion in the OpenPI environment or install lerobot."
            ) from exc


def create_lerobot_dataset(args: argparse.Namespace):
    if args.raw_only:
        return None, None
    try:
        LeRobotDataset, lerobot_home = import_lerobot()
    except ImportError as exc:
        if args.require_lerobot:
            raise
        print(f"[warn] {exc}")
        print("[warn] continuing with raw npz only; run --convert-only on the raw directory later.")
        return None, None

    output_path = Path(lerobot_home) / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite or use a new --repo-id")
        shutil.rmtree(output_path)

    features = {
        "image": {
            "dtype": "image",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": ["actions"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": [list(STATE_NAMES)],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": [list(STATE_NAMES)],
        },
    }
    for camera in args.cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="ur5e_robotiq85",
        fps=args.control_freq,
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )
    return dataset, output_path


def render_camera(env, camera_name: str, width: int, height: int) -> np.ndarray:
    rgb = env.sim.render(camera_name=camera_name, width=width, height=height)
    return np.asarray(rgb[::-1], dtype=np.uint8)


def get_arm_joint_positions(env) -> np.ndarray:
    robot = env.robots[0]
    return np.asarray(env.sim.data.qpos[robot._ref_joint_pos_indexes[:6]], dtype=np.float32)


def get_gripper_position(env) -> float:
    robot = env.robots[0]
    indexes = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    if not indexes:
        return 0.0
    return float(np.mean(np.asarray(env.sim.data.qpos[indexes], dtype=np.float32)))


def get_state_vector(env) -> np.ndarray:
    return np.concatenate([get_arm_joint_positions(env), np.asarray([get_gripper_position(env)], dtype=np.float32)])


def get_images(env, cameras: Iterable[str], width: int, height: int) -> dict[str, np.ndarray]:
    return {camera: render_camera(env, camera, width, height) for camera in cameras}


def complete_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    if action.size == action_dim:
        return action
    if action.size < action_dim:
        return np.concatenate([action, np.zeros(action_dim - action.size, dtype=action.dtype)])
    return action[:action_dim]


def yellow_contact(env) -> tuple[bool, str | None]:
    model = env.sim.model
    data = env.sim.data
    yellow_tokens = ("left_yellow_slab", "right_yellow_slab")
    allowed_other_tokens = ("table", "floor")

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = model.geom_id2name(contact.geom1) or ""
        geom2 = model.geom_id2name(contact.geom2) or ""
        geom1_is_yellow = any(token in geom1 for token in yellow_tokens)
        geom2_is_yellow = any(token in geom2 for token in yellow_tokens)
        if not geom1_is_yellow and not geom2_is_yellow:
            continue
        other = geom2 if geom1_is_yellow else geom1
        if any(token in other for token in allowed_other_tokens):
            continue
        return True, f"{geom1} <-> {geom2}"
    return False, None


def check_failure(env) -> tuple[bool, str]:
    if env._check_obstacle_violation():
        return True, "yellow slab tipped"
    has_contact, contact_name = yellow_contact(env)
    if has_contact:
        return True, f"yellow contact: {contact_name}"
    return False, ""


def append_lerobot_episode(dataset, episode: list[dict], task: str) -> None:
    if dataset is None:
        return
    for frame in episode:
        frame_data = {
            "image": frame["images"][frame["primary_image_camera"]],
            "wrist_image": frame["images"][frame["wrist_image_camera"]],
            "state": frame["state"],
            "actions": frame["action"],
            "observation.state": frame["state"],
            "action": frame["action"],
            "task": task,
        }
        for camera, image in frame["images"].items():
            frame_data[f"observation.images.{camera}"] = image
        dataset.add_frame(frame_data)
    try:
        dataset.save_episode(task=task)
    except TypeError:
        dataset.save_episode()


def save_raw_episode(raw_dir: Path, episode_index: int, episode: list[dict], task: str, metadata: dict) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"episode_{episode_index:06d}.npz"
    states = np.stack([frame["state"] for frame in episode]).astype(np.float32)
    actions = np.stack([frame["action"] for frame in episode]).astype(np.float32)
    images = {
        camera: np.stack([frame["images"][camera] for frame in episode])
        for camera in episode[0]["images"].keys()
    }
    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        task=np.asarray(task),
        metadata=np.asarray(json.dumps(metadata)),
        **{f"images_{camera}": value for camera, value in images.items()},
    )
    return path


def load_raw_episode(path: Path, args: argparse.Namespace) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    cameras = [key[len("images_") :] for key in data.files if key.startswith("images_")]
    episode = []
    for i in range(data["states"].shape[0]):
        images = {camera: data[f"images_{camera}"][i] for camera in cameras}
        episode.append(
            {
                "state": data["states"][i].astype(np.float32),
                "action": data["actions"][i].astype(np.float32),
                "images": images,
                "primary_image_camera": args.primary_image_camera,
                "wrist_image_camera": args.wrist_image_camera,
            }
        )
    return episode


def make_env(args: argparse.Namespace):
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import VisualizationWrapper

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from create_robosuite_upright_blocks_scene import UprightBlocksLift

    controller_config = load_controller_config(default_controller=args.controller)
    env = UprightBlocksLift(
        controller_configs=controller_config,
        has_renderer=not args.dry_run,
        has_offscreen_renderer=True,
        render_camera=args.viewer_camera,
        camera_names=tuple(args.cameras),
        camera_widths=args.image_width,
        camera_heights=args.image_height,
        use_camera_obs=False,
        ignore_done=True,
        horizon=max(args.max_steps, 1000000),
        control_freq=args.control_freq,
        hard_reset=False,
    )
    if not args.dry_run:
        env = VisualizationWrapper(env)
    return env


def make_device(args: argparse.Namespace):
    if args.dry_run:
        class DummyDevice:
            def start_control(self):
                return None

        return DummyDevice()
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        return Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    from robosuite.devices import SpaceMouse

    return SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)


def print_controls(args: argparse.Namespace) -> None:
    print("[controls]")
    print("  keyboard/SpaceMouse teleop uses the selected OSC/IK controller")
    print("  q / device reset: discard current attempt and restart")
    print("  success: red cube on plate and gripper released")
    print("  failure: any non-table contact with a yellow block, or yellow block tipping")
    print("  saved action: observed joint delta [q_next - q_now] plus gripper delta")
    if args.max_steps == 0:
        print("  max steps: unlimited")
    else:
        print(f"  max steps: {args.max_steps}")


def collect_one_attempt(env, device, args: argparse.Namespace) -> tuple[bool, list[dict], str]:
    env.reset()
    if not args.dry_run:
        env.render()
    device.start_control()

    episode: list[dict] = []
    success_hold = -1
    step = 0
    max_steps = 5 if args.dry_run and args.max_steps == 0 else args.max_steps
    while max_steps == 0 or step < max_steps:
        if args.dry_run:
            controller_action = np.zeros(env.action_dim, dtype=np.float32)
        else:
            from robosuite.utils.input_utils import input2action

            controller_action, _ = input2action(
                device=device,
                robot=env.robots[0],
                active_arm="right",
                env_configuration=None,
            )
            if controller_action is None:
                return False, episode, "operator reset"
            controller_action = complete_action(np.asarray(controller_action, dtype=np.float32), env.action_dim)

        state = get_state_vector(env).astype(np.float32)
        images = get_images(env, args.cameras, args.image_width, args.image_height)
        env.step(controller_action)
        if not args.dry_run:
            env.render()
        next_state = get_state_vector(env).astype(np.float32)

        episode.append(
            {
                "state": state,
                "action": (next_state - state).astype(np.float32),
                "images": images,
                "primary_image_camera": args.primary_image_camera,
                "wrist_image_camera": args.wrist_image_camera,
            }
        )

        failed, reason = check_failure(env)
        if failed:
            return False, episode, reason

        if success_hold == 0:
            return True, episode, "success"
        if env._check_success():
            success_hold = args.success_hold if success_hold < 0 else success_hold - 1
        else:
            success_hold = -1

        step += 1

    return False, episode, "max steps reached"


def convert_raw_to_lerobot(args: argparse.Namespace) -> None:
    if args.convert_only is None:
        raise ValueError("--convert-only requires a raw episode directory")
    if args.raw_only:
        raise ValueError("--raw-only cannot be used with --convert-only")

    dataset, dataset_path = create_lerobot_dataset(args)
    if dataset is None:
        raise RuntimeError("LeRobot is required for --convert-only")

    raw_paths = sorted(args.convert_only.glob("episode_*.npz"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw episodes found in {args.convert_only}")

    for path in raw_paths:
        episode = load_raw_episode(path, args)
        task = str(np.load(path, allow_pickle=True)["task"])
        append_lerobot_episode(dataset, episode, task)
        print(f"[info] converted {path.name} ({len(episode)} frames)")
    print(f"[done] LeRobot dataset: {dataset_path}")


def run_collection(args: argparse.Namespace) -> None:
    run_dir = args.output_dir / time.strftime("upright_blocks_lerobot_%Y%m%d_%H%M%S")
    raw_dir = run_dir / "raw"
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir(parents=True, exist_ok=False)

    dataset, dataset_path = create_lerobot_dataset(args)
    metadata = {
        "task": args.task,
        "repo_id": args.repo_id,
        "state": list(STATE_NAMES),
        "action": "delta joint state: next_state - state",
        "cameras": list(args.cameras),
        "success": "red cube on target plate and gripper released",
        "failure": "yellow slab tipped or touched by any non-table geom",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    env = make_env(args)
    device = make_device(args)
    if args.device == "keyboard" and not args.dry_run:
        env.viewer.add_keypress_callback(device.on_press)

    print_controls(args)
    print(f"[info] raw backup: {raw_dir}")
    if dataset_path is not None:
        print(f"[info] LeRobot dataset: {dataset_path}")

    successes = 0
    attempts = 0
    try:
        while successes < args.num_demos:
            attempts += 1
            print(f"[info] attempt {attempts}; saved successes {successes}/{args.num_demos}")
            ok, episode, reason = collect_one_attempt(env, device, args)
            if not episode:
                print(f"[info] discarded empty attempt: {reason}")
                if args.dry_run:
                    break
                continue
            if not ok:
                print(f"[warn] discarded failed attempt ({len(episode)} frames): {reason}")
                if args.dry_run:
                    break
                continue

            raw_path = save_raw_episode(raw_dir, successes, episode, args.task, metadata)
            append_lerobot_episode(dataset, episode, args.task)
            successes += 1
            print(f"[done] saved demo {successes}/{args.num_demos}: {raw_path}")

            if args.dry_run:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print(f"[done] successful demos: {successes}")
    print(f"[done] raw backup: {raw_dir}")
    if dataset_path is not None:
        print(f"[done] LeRobot dataset: {dataset_path}")
    else:
        print("[note] LeRobot was not written in this run.")
        print(f"[note] convert later with: python {Path(__file__).name} --convert-only {raw_dir} --repo-id {args.repo_id}")


def main() -> None:
    args = parse_args()
    if args.convert_only is not None:
        convert_raw_to_lerobot(args)
        return
    run_collection(args)


if __name__ == "__main__":
    main()
