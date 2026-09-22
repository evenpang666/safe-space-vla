#!/usr/bin/env python3
"""Repair a preprocessed Quest3 shard whose left 2F-85 state used 1=open.

This is for immutable legacy shards whose source HDF5 has subsequently been
re-recorded or replaced.  It preserves RGB, robot joints, actions and task
text, but deterministically rebuilds mesh-surface points and point-flow
targets from the stored qpos and raw gripper state.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.dual_ur7e_surface import DualUR7eSurfacePointSampler
from safety_module.project_config import DEFAULT_PROJECT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Legacy preprocessed .npz shard; it is never modified.")
    parser.add_argument("--output", type=Path, required=True, help="Corrected .npz shard.")
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    with np.load(args.input, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    required = ("qpos", "gripper_position", "sample_frame_indices", "action_chunks", "surface_arm_count", "points_per_link", "link_names")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{args.input} is not a supported Quest3 surface shard; missing {missing}")
    qpos = np.asarray(payload["qpos"], dtype=np.float32)
    raw_gripper = np.asarray(payload["gripper_position"], dtype=np.float32)
    sample_indices = np.asarray(payload["sample_frame_indices"], dtype=np.int64)
    horizon = int(np.asarray(payload["action_chunks"]).shape[1])
    arm_count = int(np.asarray(payload["surface_arm_count"]).item())
    points_per_link = int(np.asarray(payload["points_per_link"]).item())
    if arm_count != 1 or qpos.ndim != 2 or qpos.shape[1] < 6 or raw_gripper.shape != (len(qpos), 2):
        raise ValueError("This repair is deliberately limited to a left-only surface with a dual-placeholder qpos/gripper shard")
    if np.any(sample_indices < 0) or np.any(sample_indices + horizon >= len(qpos)):
        raise ValueError("sample_frame_indices/action horizon do not fit the stored state timeline")
    surface_gripper = raw_gripper.copy()
    # Legacy semantics: 1=open,0=closed. The mesh requires 0=open,1=closed.
    surface_gripper[:, 0] = 1.0 - surface_gripper[:, 0]
    sampler = DualUR7eSurfacePointSampler(points_per_link=points_per_link, project_config=args.project_config)
    link_names = sampler.link_names(1)
    if tuple(np.asarray(payload["link_names"]).tolist()) != tuple(link_names):
        raise ValueError("Shard link layout differs from the current canonical left UR7e+2F85 sampler")
    fixed = np.empty((len(qpos), len(link_names), points_per_link, 3), dtype=np.float32)
    for frame in range(len(qpos)):
        fixed[frame] = sampler.link_points(qpos[frame, :6], surface_gripper[frame, :1])
    flat = fixed.reshape(len(qpos), -1, 3)
    offsets = np.stack([flat[start + 1 : start + horizon + 1] - flat[start] for start in sample_indices]).astype(np.float32)
    payload.update(
        fixed_link_points=fixed,
        current_link_points=flat[sample_indices],
        target_point_offsets=offsets,
        target_point_mask=np.ones(offsets.shape[:-1], dtype=bool),
        surface_gripper_position=surface_gripper,
        left_gripper_position_convention=np.asarray("opening_fraction"),
        surface_gripper_position_semantics=np.asarray("closing_fraction_0_open_1_closed"),
        gripper_surface_repair=np.asarray("legacy_opening_fraction_inverted_to_closing_fraction"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print({"input": str(args.input), "output": str(args.output), "frames": len(qpos), "samples": len(sample_indices), "points": flat.shape[1], "convention": "opening_fraction -> closing_fraction"})


if __name__ == "__main__":
    main()
