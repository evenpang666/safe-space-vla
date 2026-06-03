#!/usr/bin/env python3
"""Train a PI05 latent safety point decoder from a generated dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig

DEFAULT_DATASET = REPO_ROOT / "outputs" / "libero_joint_swept_pointcloud" / "pi05_safety_decoder_dataset.npz"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "libero_joint_swept_pointcloud" / "pi05_safety_decoder.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=0, help="Transformer feed-forward dim. 0 means 4 * hidden_dim.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_dataset_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        prefix_tokens = torch.as_tensor(np.asarray(data["prefix_tokens"], dtype=np.float32))
        target_link_points = torch.as_tensor(np.asarray(data["target_link_points"], dtype=np.float32))

    if prefix_tokens.ndim != 3:
        raise ValueError(f"prefix_tokens must have shape (S, N, D), got {tuple(prefix_tokens.shape)}")
    if target_link_points.ndim != 5:
        raise ValueError(
            "target_link_points must have shape (S, T, L, P, 3), "
            f"got {tuple(target_link_points.shape)}"
        )
    if prefix_tokens.shape[0] <= 0:
        raise ValueError("dataset must contain at least one sample")
    if prefix_tokens.shape[1] <= 0:
        raise ValueError("prefix_tokens token count must be positive")
    if prefix_tokens.shape[2] <= 0:
        raise ValueError("prefix_tokens token dimension must be positive")
    if target_link_points.shape[1] <= 0:
        raise ValueError("target_link_points horizon must be positive")
    if target_link_points.shape[2] <= 0:
        raise ValueError("target_link_points link count must be positive")
    if target_link_points.shape[3] <= 0:
        raise ValueError("target_link_points points per link must be positive")
    if target_link_points.shape[-1] != 3:
        raise ValueError(f"target_link_points last dimension must be 3, got {target_link_points.shape[-1]}")
    if prefix_tokens.shape[0] != target_link_points.shape[0]:
        raise ValueError("prefix_tokens and target_link_points must have the same first dimension")
    return prefix_tokens, target_link_points


def train_one_epoch(
    model: SafetyPointDecoder,
    optimizer: torch.optim.Optimizer,
    prefix_tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if prefix_tokens.shape[0] != targets.shape[0]:
        raise ValueError("prefix_tokens and targets must have the same first dimension")
    if prefix_tokens.shape[0] <= 0:
        raise ValueError("training dataset must contain at least one sample")

    model.train()
    num_samples = prefix_tokens.shape[0]
    total_loss = 0.0
    order = torch.randperm(num_samples)

    for start in range(0, num_samples, batch_size):
        batch_idx = order[start : start + batch_size]
        batch_prefix = prefix_tokens[batch_idx].to(device=device)
        batch_targets = targets[batch_idx].to(device=device)

        optimizer.zero_grad()
        predictions = model(batch_prefix)
        loss = torch.nn.functional.smooth_l1_loss(predictions, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * int(batch_idx.numel())

    return total_loss / float(num_samples)


def save_checkpoint(
    path: Path,
    model: SafetyPointDecoder,
    config: SafetyPointDecoderConfig,
    epoch: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_payload = config.to_dict()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_payload,
            "epoch": int(epoch),
            "loss": float(loss),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": config_payload,
                "epoch": int(epoch),
                "loss": float(loss),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    prefix_tokens, targets = load_dataset_tensors(args.dataset)
    config = SafetyPointDecoderConfig(
        token_dim=int(prefix_tokens.shape[-1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_tokens=args.max_tokens,
        horizon=int(targets.shape[1]),
        num_links=int(targets.shape[2]),
        points_per_link=int(targets.shape[3]),
        dropout=args.dropout,
    )
    model = SafetyPointDecoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            optimizer,
            prefix_tokens,
            targets,
            batch_size=args.batch_size,
            device=device,
        )
        print(f"epoch={epoch} loss={loss:.6f}")

    save_checkpoint(args.output, model, config, epoch=args.epochs, loss=loss)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
