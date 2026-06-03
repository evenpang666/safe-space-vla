#!/usr/bin/env python3
"""Train SafetyFlowPointModel from PI05 prefix tokens and arm point offsets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_module.safety_flow_point_model import (
    SafetyFlowPointModel,
    flow_matching_loss,
    sample_flow_matching_batch,
)

DEFAULT_DATASET = REPO_ROOT / "outputs" / "pi05_safety_decoder" / "pi05_libero_task0_decoder_dataset.npz"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pi05_safety_decoder" / "pi05_libero_task0_safety_flow_point_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=0, help="Feed-forward dim. 0 means 4 * hidden_dim.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-prefix-tokens", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save an additional checkpoint every N epochs. Use 0 to disable periodic checkpoints.",
    )
    parser.add_argument("--loss-plot", type=Path, default=None, help="PNG path for live loss curve.")
    parser.add_argument("--loss-csv", type=Path, default=None, help="CSV path for epoch loss history.")
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help="Update loss PNG every N epochs. Use 0 to disable PNG updates while still writing CSV.",
    )
    parser.add_argument("--device", default="cpu", help="Use cpu by default; set cuda only with a compatible PyTorch build.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_dataset_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        prefix_tokens = torch.as_tensor(np.asarray(data["prefix_tokens"], dtype=np.float32))
        arm_points = torch.as_tensor(np.asarray(data["arm_points"], dtype=np.float32))
        target_point_offsets = torch.as_tensor(np.asarray(data["target_point_offsets"], dtype=np.float32))

    validate_dataset_tensors(prefix_tokens, arm_points, target_point_offsets)
    return prefix_tokens, arm_points, target_point_offsets


def validate_dataset_tensors(
    prefix_tokens: torch.Tensor,
    arm_points: torch.Tensor,
    target_point_offsets: torch.Tensor,
) -> None:
    if prefix_tokens.ndim != 3:
        raise ValueError(f"prefix_tokens must have shape (S, N, D), got {tuple(prefix_tokens.shape)}")
    if arm_points.ndim != 3 or arm_points.shape[-1] < 3:
        raise ValueError(f"arm_points must have shape (S, K, 3 + C_arm), got {tuple(arm_points.shape)}")
    if target_point_offsets.ndim != 4 or target_point_offsets.shape[-1] != 3:
        raise ValueError(
            "target_point_offsets must have shape (S, T_future, K, 3), "
            f"got {tuple(target_point_offsets.shape)}"
        )
    if prefix_tokens.shape[0] <= 0:
        raise ValueError("dataset must contain at least one sample")
    if prefix_tokens.shape[1] <= 0:
        raise ValueError("prefix_tokens token count must be positive")
    if prefix_tokens.shape[2] <= 0:
        raise ValueError("prefix_tokens token dimension must be positive")
    if arm_points.shape[1] <= 0:
        raise ValueError("arm_points point count must be positive")
    if target_point_offsets.shape[1] <= 0:
        raise ValueError("target_point_offsets horizon must be positive")
    if not (prefix_tokens.shape[0] == arm_points.shape[0] == target_point_offsets.shape[0]):
        raise ValueError("prefix_tokens, arm_points, and target_point_offsets must have the same sample count")
    if arm_points.shape[1] != target_point_offsets.shape[2]:
        raise ValueError(
            f"arm_points point count {arm_points.shape[1]} must match target_point_offsets point count "
            f"{target_point_offsets.shape[2]}"
        )


def build_model_kwargs(
    prefix_tokens: torch.Tensor,
    arm_points: torch.Tensor,
    target_point_offsets: torch.Tensor,
    *,
    hidden_dim: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    max_prefix_tokens: int,
) -> dict:
    validate_dataset_tensors(prefix_tokens, arm_points, target_point_offsets)
    resolved_ffn_dim = int(ffn_dim) if int(ffn_dim) > 0 else int(hidden_dim) * 4
    return {
        "arm_point_dim": int(arm_points.shape[-1]),
        "prefix_dim": int(prefix_tokens.shape[-1]),
        "hidden_dim": int(hidden_dim),
        "n_future": int(target_point_offsets.shape[1]),
        "max_points": int(arm_points.shape[1]),
        "num_encoder_layers": int(num_encoder_layers),
        "num_decoder_layers": int(num_decoder_layers),
        "num_heads": int(num_heads),
        "ffn_dim": resolved_ffn_dim,
        "dropout": float(dropout),
        "max_prefix_tokens": int(max_prefix_tokens),
    }


def train_one_epoch(
    model: SafetyFlowPointModel,
    optimizer: torch.optim.Optimizer,
    prefix_tokens: torch.Tensor,
    arm_points: torch.Tensor,
    target_point_offsets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> float:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    validate_dataset_tensors(prefix_tokens, arm_points, target_point_offsets)

    model.train()
    num_samples = prefix_tokens.shape[0]
    total_loss = 0.0
    order = torch.randperm(num_samples)

    for start in range(0, num_samples, batch_size):
        batch_idx = order[start : start + batch_size]
        batch_prefix = prefix_tokens[batch_idx].to(device=device)
        batch_arm_points = arm_points[batch_idx].to(device=device)
        batch_offsets = target_point_offsets[batch_idx].to(device=device)
        x_s, s, x_0, _v_target = sample_flow_matching_batch(batch_offsets)

        optimizer.zero_grad()
        v_pred = model(
            arm_points=batch_arm_points,
            prefix_tokens=batch_prefix,
            x_s=x_s,
            s=s,
        )
        loss = flow_matching_loss(v_pred, batch_offsets, x_0)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * int(batch_idx.numel())

    return total_loss / float(num_samples)


def save_checkpoint(
    path: Path,
    model: SafetyFlowPointModel,
    model_kwargs: dict,
    epoch: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "SafetyFlowPointModel",
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "epoch": int(epoch),
        "loss": float(loss),
    }
    torch.save(payload, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_type": payload["model_type"],
                "model_kwargs": model_kwargs,
                "epoch": int(epoch),
                "loss": float(loss),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def should_save_periodic_checkpoint(*, epoch: int, checkpoint_every: int) -> bool:
    return int(checkpoint_every) > 0 and int(epoch) > 0 and int(epoch) % int(checkpoint_every) == 0


def periodic_checkpoint_path(output: Path, *, epoch: int) -> Path:
    return output.with_name(f"{output.stem}_epoch{int(epoch):04d}{output.suffix}")


def default_loss_plot_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_loss.png")


def default_loss_csv_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_loss.csv")


def should_update_loss_plot(*, epoch: int, plot_every: int) -> bool:
    return int(plot_every) > 0 and int(epoch) > 0 and int(epoch) % int(plot_every) == 0


def save_loss_history(
    history: list[tuple[int, float]],
    *,
    csv_path: Path,
    plot_path: Path | None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,loss"]
    lines.extend(f"{int(epoch)},{float(loss):.10f}" for epoch, loss in history)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if plot_path is None:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(epoch) for epoch, _loss in history]
    losses = [float(loss) for _epoch, loss in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, color="#1f77b4", linewidth=1.8)
    ax.scatter(epochs[-1:], losses[-1:], color="#d62728", s=18, zorder=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("SafetyFlowPointModel training loss")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}")
    if args.checkpoint_every < 0:
        raise ValueError(f"--checkpoint-every must be >= 0, got {args.checkpoint_every}")
    if args.plot_every < 0:
        raise ValueError(f"--plot-every must be >= 0, got {args.plot_every}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    prefix_tokens, arm_points, target_point_offsets = load_dataset_tensors(args.dataset)
    model_kwargs = build_model_kwargs(
        prefix_tokens,
        arm_points,
        target_point_offsets,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_prefix_tokens=args.max_prefix_tokens,
    )
    model = SafetyFlowPointModel(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss = float("nan")
    loss_history: list[tuple[int, float]] = []
    loss_csv = args.loss_csv or default_loss_csv_path(args.output)
    loss_plot = args.loss_plot or default_loss_plot_path(args.output)
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            optimizer,
            prefix_tokens,
            arm_points,
            target_point_offsets,
            batch_size=args.batch_size,
            device=device,
            grad_clip_norm=args.grad_clip_norm,
        )
        print(f"epoch={epoch} loss={loss:.6f}")
        loss_history.append((epoch, loss))
        save_loss_history(
            loss_history,
            csv_path=loss_csv,
            plot_path=loss_plot if should_update_loss_plot(epoch=epoch, plot_every=args.plot_every) else None,
        )
        if should_save_periodic_checkpoint(epoch=epoch, checkpoint_every=args.checkpoint_every):
            checkpoint_path = periodic_checkpoint_path(args.output, epoch=epoch)
            save_checkpoint(checkpoint_path, model, model_kwargs, epoch=epoch, loss=loss)
            print(f"saved periodic checkpoint to {checkpoint_path}")

    save_checkpoint(args.output, model, model_kwargs, epoch=args.epochs, loss=loss)
    print(f"saved checkpoint to {args.output}")
    print(f"saved loss history to {loss_csv}")
    if args.plot_every > 0:
        print(f"saved loss plot to {loss_plot}")


if __name__ == "__main__":
    main()
