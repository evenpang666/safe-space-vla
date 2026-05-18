#!/usr/bin/env python3
"""Train a point-world model from scene points and deterministic robot point flow."""

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

from safety_module.point_world_model import (  # noqa: E402
    PointWorldModel,
    chamfer_hausdorff_loss,
    point_world_model_loss,
)


class PointWorldDataset(Dataset):
    """NPZ dataset for chunked point-world training.

    Required keys:
        scene_points: ``(N, P, 3)`` current non-robot scene points.
        robot_point_flow: ``(N, H, R, 3)`` deterministic FK/simulator robot points.

    Optional keys:
        scene_features: ``(N, P, C)`` per-point features such as RGB.
        future_scene_points: ``(N, H, P, 3)`` point-correspondence future scene points.
        future_scene_points_unordered: ``(N, H, M, 3)`` future points without correspondence.
        scene_flow: ``(N, H, P, 3)`` direct flow labels.
        affected_mask: ``(N, H, P)`` contacted/moved labels.
    """

    def __init__(self, path: Path) -> None:
        data = np.load(path, allow_pickle=False)
        for key in ("scene_points", "robot_point_flow"):
            if key not in data:
                raise ValueError(f"{path} is missing required key {key!r}")

        self.scene_points = self._tensor(data["scene_points"], "scene_points", ndim=3, last_dim=3)
        self.robot_point_flow = self._tensor(data["robot_point_flow"], "robot_point_flow", ndim=4, last_dim=3)
        if len(self.scene_points) != len(self.robot_point_flow):
            raise ValueError("scene_points and robot_point_flow must have the same sample count")

        self.scene_features = self._optional_tensor(data, "scene_features", ndim=3)
        self.future_scene_points = self._optional_tensor(data, "future_scene_points", ndim=4, last_dim=3)
        self.future_scene_points_unordered = self._optional_tensor(
            data,
            "future_scene_points_unordered",
            ndim=4,
            last_dim=3,
        )
        self.scene_flow = self._optional_tensor(data, "scene_flow", ndim=4, last_dim=3)
        self.affected_mask = self._optional_tensor(data, "affected_mask", ndim=3)

        if (
            self.future_scene_points is None
            and self.future_scene_points_unordered is None
            and self.scene_flow is None
            and self.affected_mask is None
        ):
            raise ValueError(
                "dataset must contain at least one target: future_scene_points, "
                "future_scene_points_unordered, scene_flow, or affected_mask"
            )

        sample_count = len(self.scene_points)
        for name in (
            "scene_features",
            "future_scene_points",
            "future_scene_points_unordered",
            "scene_flow",
            "affected_mask",
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

    def __len__(self) -> int:
        return int(self.scene_points.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "scene_points": self.scene_points[index],
            "robot_point_flow": self.robot_point_flow[index],
        }
        for name in (
            "scene_features",
            "future_scene_points",
            "future_scene_points_unordered",
            "scene_flow",
            "affected_mask",
        ):
            value = getattr(self, name)
            if value is not None:
                item[name] = value[index]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train affected-mask / scene-flow point-world model.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "point_world_model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--contact-radius", type=float, default=0.05)
    parser.add_argument("--predict-uncertainty", action="store_true")
    parser.add_argument("--lambda-flow", type=float, default=1.0)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--lambda-chamfer", type=float, default=1.0)
    parser.add_argument("--lambda-hausdorff", type=float, default=0.1)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-unc", type=float, default=0.0)
    parser.add_argument("--moving-weight", type=float, default=8.0)
    parser.add_argument("--near-robot-weight", type=float, default=3.0)
    parser.add_argument("--contact-weight", type=float, default=5.0)
    parser.add_argument("--moving-threshold", type=float, default=0.01)
    parser.add_argument("--near-robot-radius", type=float, default=0.06)
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


def run_epoch(
    model: PointWorldModel,
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
        batch = to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(
                batch["scene_points"],
                batch["robot_point_flow"],
                batch.get("scene_features"),
            )
            loss, metrics = point_world_model_loss(
                predictions,
                scene_points=batch["scene_points"],
                target_future_points=batch.get("future_scene_points"),
                target_flow=batch.get("scene_flow"),
                target_mask=batch.get("affected_mask"),
                robot_point_flow=batch["robot_point_flow"],
                lambda_flow=args.lambda_flow,
                lambda_mask=args.lambda_mask,
                lambda_smooth=args.lambda_smooth,
                lambda_unc=args.lambda_unc,
                moving_threshold=args.moving_threshold,
                moving_weight=args.moving_weight,
                near_robot_radius=args.near_robot_radius,
                near_robot_weight=args.near_robot_weight,
                contact_weight=args.contact_weight,
            )
            if "future_scene_points_unordered" in batch:
                unordered_loss, unordered_metrics = chamfer_hausdorff_loss(
                    predictions["future_points"],
                    batch["future_scene_points_unordered"],
                    hausdorff_weight=args.lambda_hausdorff,
                )
                loss = loss + float(args.lambda_chamfer) * unordered_loss
                metrics.update(unordered_metrics)
                metrics["loss"] = loss.detach()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        batch_size = batch["scene_points"].shape[0]
        seen += batch_size
        for key, value in metrics.items():
            scalar = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            totals[key] = totals.get(key, 0.0) + scalar * batch_size

    if seen == 0:
        return {"loss": float("nan")}
    return {key: value / seen for key, value in totals.items()}


def export_torchscript(model: PointWorldModel, output_path: Path, dataset: PointWorldDataset, device: torch.device) -> None:
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

    dataset = PointWorldDataset(args.dataset)
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
    predict_flow = (
        dataset.future_scene_points is not None
        or dataset.future_scene_points_unordered is not None
        or dataset.scene_flow is not None
    )
    model = PointWorldModel(
        scene_feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        contact_radius=args.contact_radius,
        predict_flow=predict_flow,
        predict_uncertainty=args.predict_uncertainty,
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
            "[epoch {epoch:04d}] train={train:.6f} val={val:.6f} mask={mask:.6f} flow={flow:.6f}".format(
                epoch=epoch,
                train=train_metrics.get("loss", float("nan")),
                val=val_metrics.get("loss", float("nan")),
                mask=val_metrics.get("mask_loss", float("nan")),
                flow=val_metrics.get("flow_loss", float("nan")),
            )
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.output_dir / "point_world_model_checkpoint.pt"
    script_path = args.output_dir / "point_world_model.pt"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "scene_feature_dim": feature_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "contact_radius": args.contact_radius,
            "predict_flow": predict_flow,
            "predict_uncertainty": args.predict_uncertainty,
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
