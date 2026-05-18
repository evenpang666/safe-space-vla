#!/usr/bin/env python3
"""Visualize a legacy robot swept-pointcloud model prediction against ground truth."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from libero_reconstruct_pointcloud import save_preview_png, write_ascii_ply  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predicted and ground-truth swept robot point clouds.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="TorchScript model exported by train_robot_pointcloud_model.py")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "robot_pointcloud_world_model" / "visualization",
    )
    parser.add_argument("--preview-points", type=int, default=80000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset)
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    gt_points = np.asarray(data["points"], dtype=np.float32)
    if not 0 <= args.index < len(states):
        raise ValueError(f"--index must be in [0, {len(states) - 1}]")

    device = torch.device(args.device)
    model = torch.jit.load(str(args.model), map_location=device)
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(states[args.index : args.index + 1]).to(device),
            torch.from_numpy(actions[args.index : args.index + 1]).to(device),
        )
    pred_points = pred.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
    target_points = gt_points[args.index].reshape(-1, 3).astype(np.float32)

    pred_color = np.repeat(np.array([[30, 110, 255]], dtype=np.uint8), len(pred_points), axis=0)
    gt_color = np.repeat(np.array([[255, 40, 40]], dtype=np.uint8), len(target_points), axis=0)
    combined_points = np.concatenate((target_points, pred_points), axis=0)
    combined_colors = np.concatenate((gt_color, pred_color), axis=0)

    prefix = f"sample_{args.index:06d}"
    write_ascii_ply(args.output_dir / f"{prefix}_gt_red.ply", target_points, gt_color)
    write_ascii_ply(args.output_dir / f"{prefix}_pred_blue.ply", pred_points, pred_color)
    write_ascii_ply(args.output_dir / f"{prefix}_combined.ply", combined_points, combined_colors)
    save_preview_png(
        args.output_dir / f"{prefix}_combined_preview.png",
        combined_points,
        combined_colors,
        args.preview_points,
    )
    print(f"[done] saved visualization files under: {args.output_dir}")
    print("[info] red = ground truth, blue = prediction")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
