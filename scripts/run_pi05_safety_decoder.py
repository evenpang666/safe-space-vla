#!/usr/bin/env python3
"""Run PI05 latent safety decoder and geometric collision check."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from safety_module.geometric_safety import collision_result_to_dict, predicted_link_points_collision
from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "libero_joint_swept_pointcloud" / "pi05_safety_decoder_prediction.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=Path, required=True)
    parser.add_argument("--safe-space", type=Path, required=True)
    parser.add_argument("--collision-margin", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_checkpoint_model(path: Path, device: torch.device) -> SafetyPointDecoder:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = SafetyPointDecoderConfig.from_dict(payload["config"])
    model = SafetyPointDecoder(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def load_prefix_tokens(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        prefix_tokens = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            prefix_tokens = data["prefix_tokens"]
    else:
        raise ValueError(f"prefix token file must be .npy or .npz, got {path.suffix}")

    prefix_tokens = np.asarray(prefix_tokens, dtype=np.float32)
    if prefix_tokens.ndim == 2:
        prefix_tokens = prefix_tokens[None, ...]
    if prefix_tokens.ndim != 3:
        raise ValueError(f"prefix_tokens must have shape (B, N, D) or (N, D), got {prefix_tokens.shape}")
    return prefix_tokens


@torch.no_grad()
def run_prediction(
    model: SafetyPointDecoder,
    prefix_tokens: np.ndarray,
    safe_space: dict[str, np.ndarray],
    collision_margin: float,
    device: torch.device,
) -> tuple[np.ndarray, object]:
    prefix = torch.as_tensor(np.asarray(prefix_tokens, dtype=np.float32), device=device)
    pred = model(prefix).detach().cpu().numpy().astype(np.float32, copy=False)
    result = predicted_link_points_collision(pred[0], safe_space, collision_margin=collision_margin)
    return pred, result


def save_prediction(path: Path, pred_link_points: np.ndarray, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pred_link_points=pred_link_points, **collision_result_to_dict(result))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_checkpoint_model(args.checkpoint, device)
    prefix_tokens = load_prefix_tokens(args.prefix_tokens)
    safe_space = load_npz_dict(args.safe_space)
    pred_link_points, result = run_prediction(
        model,
        prefix_tokens,
        safe_space,
        collision_margin=args.collision_margin,
        device=device,
    )
    save_prediction(args.output, pred_link_points, result)
    status = "collision" if result.collides else "safe"
    print(f"{status} collision_point_count={result.collision_point_count}")


if __name__ == "__main__":
    main()
