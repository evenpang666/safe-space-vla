#!/usr/bin/env python3
"""Teleoperate the UR5e upright-blocks scene and collect demonstrations."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
from pathlib import Path
import shutil
import time
from glob import glob

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from create_robosuite_upright_blocks_scene import UprightBlocksLift


DEFAULT_OUTPUT_DIR = Path("outputs/robosuite_collision_scene/demos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["keyboard", "spacemouse"], default="keyboard")
    parser.add_argument("--controller", choices=["OSC_POSE", "IK_POSE"], default="OSC_POSE")
    parser.add_argument("--camera", default="frontview", help="Viewer camera used during teleoperation.")
    parser.add_argument("--num-demos", type=int, default=10, help="Number of successful demos to collect.")
    parser.add_argument("--horizon", type=int, default=800, help="Maximum environment steps per attempt.")
    parser.add_argument("--success-hold", type=int, default=10, help="Extra successful steps before ending an episode.")
    parser.add_argument("--collect-freq", type=int, default=1)
    parser.add_argument("--flush-freq", type=int, default=100)
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument(
        "--save-unsuccessful",
        action="store_true",
        help="Keep aborted or failed attempts in demo.hdf5 instead of filtering them out.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the intermediate per-episode robosuite DataCollectionWrapper files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the environment headlessly and take a few zero-action steps for validation.",
    )
    return parser.parse_args()


def make_env(args: argparse.Namespace):
    controller_config = load_controller_config(default_controller=args.controller)
    env = UprightBlocksLift(
        controller_configs=controller_config,
        has_renderer=not args.dry_run,
        has_offscreen_renderer=args.dry_run,
        render_camera=args.camera,
        camera_names=args.camera,
        camera_widths=768,
        camera_heights=512,
        use_camera_obs=False,
        ignore_done=True,
        horizon=args.horizon,
        control_freq=20,
        hard_reset=False,
    )
    env_info = {
        "env_name": "UprightBlocksLift",
        "robots": "UR5e",
        "controller": args.controller,
        "camera": args.camera,
        "task": "pick red cube and place it on the left target plate without tipping either yellow slab",
    }
    return env, env_info


def make_device(args: argparse.Namespace):
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        return Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)

    from robosuite.devices import SpaceMouse

    return SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)


def print_controls(device_name: str) -> None:
    print("[controls]")
    if device_name == "keyboard":
        print("  translate: W/S, A/D, R/F")
        print("  rotate: Z/X, T/G, C/V")
        print("  gripper: space")
        print("  reset / finish current attempt: Q")
    else:
        print("  translate/rotate: SpaceMouse axes")
        print("  gripper: SpaceMouse side buttons")
        print("  reset / finish current attempt: SpaceMouse reset command")
    print("  goal: place the red cube on the left plate without tipping the yellow slabs")


def complete_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    if action.size == action_dim:
        return action
    if action.size < action_dim:
        return np.concatenate([action, np.zeros(action_dim - action.size)])
    return action[:action_dim]


def collect_one_attempt(env, device, args: argparse.Namespace) -> bool:
    from robosuite.utils.input_utils import input2action

    env.reset()
    env.render()
    device.start_control()

    task_completion_hold_count = -1
    for step in range(args.horizon):
        active_robot = env.robots[0]
        action, _ = input2action(
            device=device,
            robot=active_robot,
            active_arm="right",
            env_configuration=None,
        )
        if action is None:
            print("[info] attempt reset by operator")
            return False

        action = complete_action(action, env.action_dim)
        env.step(action)
        env.render()

        if env._check_obstacle_violation():
            print(f"[warn] obstacle violation at step {step}; ending attempt")
            return False

        if task_completion_hold_count == 0:
            print(f"[info] success held for {args.success_hold} steps")
            return True
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = args.success_hold
        else:
            task_completion_hold_count = -1

    print("[warn] horizon reached before success")
    return False


def gather_demonstrations_as_hdf5(
    raw_dir: Path,
    out_dir: Path,
    env_info: dict,
    save_unsuccessful: bool,
) -> int:
    hdf5_path = out_dir / "demo.hdf5"
    if hdf5_path.exists():
        hdf5_path.unlink()

    num_eps = 0
    env_name = "UprightBlocksLift"
    with h5py.File(hdf5_path, "w") as f:
        grp = f.create_group("data")
        for ep_directory in sorted(raw_dir.iterdir()):
            if not ep_directory.is_dir():
                continue
            states = []
            actions = []
            success = False
            for state_file in sorted(glob(str(ep_directory / "state_*.npz"))):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])
                states.extend(dic["states"])
                actions.extend(ai["actions"] for ai in dic["action_infos"])
                success = success or bool(dic["successful"])

            if len(states) == 0:
                continue
            if not success and not save_unsuccessful:
                continue

            # DataCollectionWrapper records one extra state after the final action.
            if len(states) == len(actions) + 1:
                states = states[:-1]
            if len(states) != len(actions):
                print(f"[warn] skipping {ep_directory.name}: states/actions length mismatch")
                continue

            num_eps += 1
            ep_data_grp = grp.create_group(f"demo_{num_eps}")
            xml_path = ep_directory / "model.xml"
            ep_data_grp.attrs["model_file"] = xml_path.read_text(encoding="utf-8")
            ep_data_grp.attrs["successful"] = success
            ep_data_grp.create_dataset("states", data=np.asarray(states))
            ep_data_grp.create_dataset("actions", data=np.asarray(actions))

        now = _datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = json.dumps(env_info)
        grp.attrs["successful_only"] = not save_unsuccessful

    return num_eps


def run_dry_validation(args: argparse.Namespace) -> None:
    env, _ = make_env(args)
    try:
        env.reset()
        for _ in range(5):
            env.step(np.zeros(env.action_dim))
        print(f"[done] dry run ok, action_dim={env.action_dim}")
        print(f"[done] success={env._check_success()}, obstacle_violation={env._check_obstacle_violation()}")
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_validation(args)
        return

    run_dir = args.output_dir / time.strftime("upright_blocks_%Y%m%d_%H%M%S")
    raw_dir = run_dir / "raw"
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir(parents=True, exist_ok=False)

    base_env, env_info = make_env(args)
    env = VisualizationWrapper(base_env)
    device = make_device(args)
    if args.device == "keyboard":
        env.viewer.add_keypress_callback(device.on_press)
    env = DataCollectionWrapper(
        env,
        str(raw_dir),
        collect_freq=args.collect_freq,
        flush_freq=args.flush_freq,
    )

    print_controls(args.device)
    print(f"[info] raw episodes: {raw_dir}")
    print(f"[info] final hdf5: {run_dir / 'demo.hdf5'}")

    successes = 0
    attempts = 0
    try:
        while successes < args.num_demos:
            attempts += 1
            print(f"[info] attempt {attempts}; successful demos {successes}/{args.num_demos}")
            collect_one_attempt(env, device, args)
            env.close()
            saved = gather_demonstrations_as_hdf5(
                raw_dir=raw_dir,
                out_dir=run_dir,
                env_info=env_info,
                save_unsuccessful=args.save_unsuccessful,
            )
            successes = saved if not args.save_unsuccessful else min(saved, args.num_demos)

            if successes < args.num_demos:
                base_env, _ = make_env(args)
                env = VisualizationWrapper(base_env)
                if args.device == "keyboard":
                    env.viewer.add_keypress_callback(device.on_press)
                env = DataCollectionWrapper(
                    env,
                    str(raw_dir),
                    collect_freq=args.collect_freq,
                    flush_freq=args.flush_freq,
                )
    finally:
        try:
            env.close()
        except Exception:
            pass

    if not args.keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
    print(f"[done] saved {successes} demos to {run_dir / 'demo.hdf5'}")


if __name__ == "__main__":
    main()
