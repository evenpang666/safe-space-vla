#!/usr/bin/env python3
"""Train a 3D geometric collision critic for VLA action chunks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_module.safety_critic import (  # noqa: E402
    PointCloudSafetyCritic,
    collision_critic_loss,
    geometric_safety_cost,
)


class CollisionCriticDataset(Dataset):
    """NPZ dataset for action-conditioned safety critic training.

    Required keys:
        scene_points: ``(N, P, 3)`` current non-robot scene points.
        robot_point_flow: ``(N, H, R, 3)`` deterministic future robot points.

    Optional inputs:
        scene_features: ``(N, P, C)`` per-scene-point features.
        forbidden_mask: ``(N, P)`` or ``(N, H, P)`` points that should not be contacted.
        target_mask: ``(N, P)`` or ``(N, H, P)`` task-relevant contact points.
        future_scene_points: ``(N, H, P, 3)`` dynamic scene prediction/label.

    At least one target is required:
        collision_label: ``(N,)`` chunk-level unsafe label.
        min_distance: ``(N,)`` future minimum safety distance.
        risk_mask: ``(N, H, P)`` point-time risk/forbidden-contact labels.
    """

    def __init__(self, path: Path, require_targets: bool = True) -> None:
        data = np.load(path, allow_pickle=False)
        for key in ("scene_points", "robot_point_flow"):
            if key not in data:
                raise ValueError(f"{path} is missing required key {key!r}")
        self.scene_points = self._tensor(data["scene_points"], "scene_points", ndim=3, last_dim=3)
        self.robot_point_flow = self._tensor(data["robot_point_flow"], "robot_point_flow", ndim=4, last_dim=3)
        if len(self.scene_points) != len(self.robot_point_flow):
            raise ValueError("scene_points and robot_point_flow must have the same sample count")

        self.scene_features = self._optional_tensor(data, "scene_features", ndim=3)
        self.forbidden_mask = self._optional_tensor_flexible(data, "forbidden_mask", valid_ndims=(2, 3))
        self.target_mask = self._optional_tensor_flexible(data, "target_mask", valid_ndims=(2, 3))
        self.future_scene_points = self._optional_tensor(data, "future_scene_points", ndim=4, last_dim=3)
        self.collision_label = self._optional_tensor(data, "collision_label", ndim=1)
        self.min_distance = self._optional_tensor(data, "min_distance", ndim=1)
        self.risk_mask = self._optional_tensor(data, "risk_mask", ndim=3)

        if require_targets and self.collision_label is None and self.min_distance is None and self.risk_mask is None:
            raise ValueError("dataset must contain at least one target: collision_label, min_distance, or risk_mask")

        sample_count = len(self.scene_points)
        for name in (
            "scene_features",
            "forbidden_mask",
            "target_mask",
            "future_scene_points",
            "collision_label",
            "min_distance",
            "risk_mask",
        ):
            value = getattr(self, name)
            if value is not None and len(value) != sample_count:
                raise ValueError(f"{name} must have the same sample count as scene_points")

    @staticmethod
    def _tensor(array: np.ndarray, name: str, ndim: int, last_dim: Optional[int] = None) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(array, dtype=np.float32))
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dims, got {tuple(tensor.shape)}")
        if last_dim is not None and tensor.shape[-1] != last_dim:
            raise ValueError(f"{name} last dim must be {last_dim}, got {tuple(tensor.shape)}")
        return tensor

    @classmethod
    def _optional_tensor(
        cls,
        data: np.lib.npyio.NpzFile,
        key: str,
        ndim: int,
        last_dim: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if key not in data:
            return None
        return cls._tensor(data[key], key, ndim=ndim, last_dim=last_dim)

    @staticmethod
    def _optional_tensor_flexible(
        data: np.lib.npyio.NpzFile,
        key: str,
        valid_ndims: tuple[int, ...],
    ) -> Optional[torch.Tensor]:
        if key not in data:
            return None
        tensor = torch.as_tensor(np.asarray(data[key], dtype=np.float32))
        if tensor.ndim not in valid_ndims:
            raise ValueError(f"{key} must have dims in {valid_ndims}, got {tuple(tensor.shape)}")
        return tensor

    def __len__(self) -> int:
        return int(self.scene_points.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "scene_points": self.scene_points[index],
            "robot_point_flow": self.robot_point_flow[index],
        }
        for name in (
            "scene_features",
            "forbidden_mask",
            "target_mask",
            "future_scene_points",
            "collision_label",
            "min_distance",
            "risk_mask",
        ):
            value = getattr(self, name)
            if value is not None:
                item[name] = value[index]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VLA action-chunk collision critic.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "collision_critic")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--safe-distance", type=float, default=0.03)
    parser.add_argument("--near-distance", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--forbidden-weight", type=float, default=4.0)
    parser.add_argument("--target-contact-weight", type=float, default=0.0)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-positive-weight", type=float, default=8.0)
    parser.add_argument("--lambda-collision", type=float, default=1.0)
    parser.add_argument("--lambda-distance", type=float, default=1.0)
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--lambda-conservative", type=float, default=0.05)
    parser.add_argument("--bootstrap-geometric-labels", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def split_indices(count: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(count, generator=generator).tolist()
    val_count = int(round(count * val_fraction))
    val_count = min(max(val_count, 1 if count > 1 else 0), count - 1 if count > 1 else 0)
    return perm[val_count:], perm[:val_count]


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=torch.float32) for key, value in batch.items()}


def maybe_bootstrap_labels(
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if not args.bootstrap_geometric_labels:
        return batch
    if "collision_label" in batch and "min_distance" in batch and "risk_mask" in batch:
        return batch
    geom = geometric_safety_cost(
        batch.get("future_scene_points", batch["scene_points"]),
        batch["robot_point_flow"],
        safe_distance=args.safe_distance,
        near_distance=args.near_distance,
        temperature=args.temperature,
        forbidden_mask=batch.get("forbidden_mask"),
        target_mask=batch.get("target_mask"),
        forbidden_weight=args.forbidden_weight,
        target_contact_weight=args.target_contact_weight,
    )
    out = dict(batch)
    out.setdefault("collision_label", (geom["min_distance"] < args.safe_distance).to(batch["scene_points"].dtype))
    out.setdefault("min_distance", geom["min_distance"])
    out.setdefault("risk_mask", (geom["risk_heatmap"] > 0.5).to(batch["scene_points"].dtype))
    return out


def run_epoch(
    model: PointCloudSafetyCritic,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    seen = 0

    for raw_batch in loader:
        batch = maybe_bootstrap_labels(to_device(raw_batch, device), args)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(
                batch["scene_points"],
                batch["robot_point_flow"],
                scene_features=batch.get("scene_features"),
                forbidden_mask=batch.get("forbidden_mask"),
                target_mask=batch.get("target_mask"),
                future_scene_points=batch.get("future_scene_points"),
            )
            loss, metrics = collision_critic_loss(
                predictions,
                collision_label=batch.get("collision_label"),
                min_distance_label=batch.get("min_distance"),
                risk_mask=batch.get("risk_mask"),
                unsafe_positive_weight=args.unsafe_positive_weight,
                lambda_collision=args.lambda_collision,
                lambda_distance=args.lambda_distance,
                lambda_risk=args.lambda_risk,
                lambda_conservative=args.lambda_conservative,
                safe_distance=args.safe_distance,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        batch_size = batch["scene_points"].shape[0]
        seen += batch_size
        for key, value in metrics.items():
            scalar = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            totals[key] = totals.get(key, 0.0) + scalar * batch_size
        for key in ("cost", "collision_probability", "min_distance", "uncertainty_cost"):
            totals[key] = totals.get(key, 0.0) + float(predictions[key].detach().mean().cpu()) * batch_size

    if seen == 0:
        return {"loss": float("nan")}
    return {key: value / seen for key, value in totals.items()}


def export_torchscript(model: PointCloudSafetyCritic, output_path: Path, dataset: CollisionCriticDataset, device: torch.device) -> None:
    model.eval()
    example_scene = dataset.scene_points[:1].to(device)
    example_robot = dataset.robot_point_flow[:1].to(device)
    if dataset.scene_features is None:
        traced = torch.jit.trace(model, (example_scene, example_robot), strict=False)
    else:
        example_features = dataset.scene_features[:1].to(device)
        traced = torch.jit.trace(model, (example_scene, example_robot, example_features), strict=False)
    traced.save(str(output_path))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CollisionCriticDataset(args.dataset, require_targets=not args.bootstrap_geometric_labels)
    train_indices, val_indices = split_indices(len(dataset), args.val_fraction, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    feature_dim = 0 if dataset.scene_features is None else int(dataset.scene_features.shape[-1])
    model = PointCloudSafetyCritic(
        scene_feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        safe_distance=args.safe_distance,
        near_distance=args.near_distance,
        temperature=args.temperature,
        forbidden_weight=args.forbidden_weight,
        target_contact_weight=args.target_contact_weight,
        uncertainty_weight=args.uncertainty_weight,
    )
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args)
        val_metrics = run_epoch(model, val_loader, None, device, args)
        selection_loss = val_metrics.get("loss", float("nan"))
        if not np.isfinite(selection_loss):
            selection_loss = train_metrics["loss"]
        if selection_loss < best_val:
            best_val = selection_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            "[epoch {epoch:04d}] train={train:.6f} val={val:.6f} p_col={pcol:.4f} dmin={dmin:.4f} cost={cost:.4f}".format(
                epoch=epoch,
                train=train_metrics.get("loss", float("nan")),
                val=val_metrics.get("loss", float("nan")),
                pcol=val_metrics.get("collision_probability", float("nan")),
                dmin=val_metrics.get("min_distance", float("nan")),
                cost=val_metrics.get("cost", float("nan")),
            )
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.output_dir / "collision_critic_checkpoint.pt"
    script_path = args.output_dir / "collision_critic.pt"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "scene_feature_dim": feature_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "safe_distance": args.safe_distance,
            "near_distance": args.near_distance,
            "temperature": args.temperature,
            "forbidden_weight": args.forbidden_weight,
            "target_contact_weight": args.target_contact_weight,
            "uncertainty_weight": args.uncertainty_weight,
            "best_val_loss": best_val,
        },
        ckpt_path,
    )
    model = model.to(device)
    export_torchscript(model, script_path, dataset, device)
    print(f"[done] saved checkpoint: {ckpt_path}")
    print(f"[done] saved TorchScript model: {script_path}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
