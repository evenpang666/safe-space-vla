#!/usr/bin/env python3
"""Jointly fine-tune PyTorch PI05 for UR7e actions and robot-surface flow.

Input shards are produced by ``preprocess_pi05_rgbd_surface_dataset.py``.
Each sample uses task text, current RGB images, qpos, and current robot points;
the targets are a normalized 7-D action chunk and its future point offsets.
Run inside the OpenPI Python environment, for example:

  uv run --project openpi ../scripts/train_pi05_ur7e_surface_pytorch.py \
    --dataset ../outputs/pi05_surface --output ../outputs/pi05_ur7e_joint.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_SRC = REPO_ROOT / "openpi" / "src"
for path in (REPO_ROOT, OPENPI_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True, help="Preprocessed .npz files or directories.")
    parser.add_argument("--output", type=Path, required=True, help="Output PyTorch checkpoint (.pt).")
    parser.add_argument("--pretrained", type=Path, default=None, help="Optional converted PI05 model.safetensors checkpoint.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--point-loss-weight", type=float, default=10.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--action-horizon", type=int, default=None, help="Defaults to the horizon stored in the dataset.")
    parser.add_argument("--max-points", type=int, default=128, help="Stable evenly-spaced point subset; use 0 for all points.")
    parser.add_argument("--point-target", choices=("fixed", "visual"), default="fixed")
    parser.add_argument(
        "--camera-map",
        action="append",
        default=[],
        metavar="RGB_NAME=MODEL_KEY",
        help="For example front=base_0_rgb. Unmapped model views are zero-filled and masked.",
    )
    parser.add_argument("--tokenizer-model", type=Path, default=None, help="Optional local paligemma_tokenizer.model.")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-base", action="store_true", help="Train only the new point-token and point-flow layers.")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def _resolve_shards(paths: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for path in paths:
        if path.is_dir():
            shards.extend(sorted(path.glob("*.npz")))
        elif path.suffix == ".npz":
            shards.append(path)
        else:
            raise ValueError(f"Dataset input must be an .npz or directory: {path}")
    if not shards:
        raise FileNotFoundError("No preprocessed .npz shards found")
    return shards


def _parse_camera_map(values: list[str], rgb_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        source, separator, target = value.partition("=")
        if not separator or source not in rgb_names or target not in MODEL_IMAGE_KEYS:
            raise ValueError(f"Invalid --camera-map {value!r}; use RGB_NAME=one of {MODEL_IMAGE_KEYS}")
        if target in mapping.values():
            raise ValueError(f"Multiple cameras mapped to the same PI05 view: {target}")
        mapping[source] = target
    if not mapping:
        for source, target in zip(rgb_names, MODEL_IMAGE_KEYS, strict=False):
            mapping[source] = target
    return mapping


class _Tokenizer:
    def __init__(self, max_len: int, model_path: Path | None):
        self.max_len = int(max_len)
        self._openpi = None
        if model_path is None:
            from openpi.models import tokenizer as openpi_tokenizer

            self._openpi = openpi_tokenizer.PaligemmaTokenizer(max_len=max_len)
        else:
            import sentencepiece

            self._processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))

    def tokenize(self, prompt: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._openpi is not None:
            return self._openpi.tokenize(prompt, state=state)
        clean = prompt.strip().replace("_", " ").replace("\n", " ")
        bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {clean}, State: {' '.join(map(str, bins))};\nAction: "
        tokens = self._processor.encode(text, add_bos=True)[: self.max_len]
        mask = np.zeros((self.max_len,), dtype=bool)
        mask[: len(tokens)] = True
        result = np.zeros((self.max_len,), dtype=np.int32)
        result[: len(tokens)] = tokens
        return result, mask


@dataclass
class _Episode:
    path: Path
    task_text: str
    qpos: np.ndarray
    actions: np.ndarray
    current_points: np.ndarray
    target_offsets: np.ndarray
    target_mask: np.ndarray
    rgb: dict[str, np.ndarray]
    sample_count: int


class Ur7eSurfaceDataset(Dataset):
    def __init__(
        self,
        shard_paths: list[Path],
        *,
        max_points: int,
        point_target: str,
        action_horizon: int | None,
        tokenizer: _Tokenizer,
        qpos_mean: np.ndarray,
        qpos_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        self.qpos_mean, self.qpos_std = qpos_mean, qpos_std
        self.action_mean, self.action_std = action_mean, action_std
        self.tokenizer = tokenizer
        self.episodes: list[_Episode] = []
        self.index: list[tuple[int, int]] = []
        image_shape: tuple[int, int, int] | None = None
        reference_points: int | None = None
        for path in shard_paths:
            episode = self._load_episode(path, max_points=max_points, point_target=point_target, action_horizon=action_horizon)
            if reference_points is None:
                reference_points = episode.current_points.shape[1]
            elif reference_points != episode.current_points.shape[1]:
                raise ValueError("All shards must resolve to the same selected point count")
            for frames in episode.rgb.values():
                if image_shape is None:
                    image_shape = tuple(frames.shape[1:])
                elif image_shape != tuple(frames.shape[1:]):
                    raise ValueError("All RGB views must share a shape before batching")
            episode_index = len(self.episodes)
            self.episodes.append(episode)
            self.index.extend((episode_index, sample_index) for sample_index in range(episode.sample_count))
        if not self.index or image_shape is None:
            raise ValueError("Dataset contains no samples or RGB frames")
        self.image_shape = image_shape
        self.point_count = int(reference_points)
        self.camera_map: dict[str, str] = {}  # assigned by the caller after CLI validation

    @staticmethod
    def _point_indices(point_count: int, max_points: int) -> np.ndarray:
        if max_points <= 0 or max_points >= point_count:
            return np.arange(point_count, dtype=np.int64)
        return np.linspace(0, point_count - 1, max_points, dtype=np.int64)

    def _load_episode(self, path: Path, *, max_points: int, point_target: str, action_horizon: int | None) -> _Episode:
        with np.load(path, allow_pickle=False) as data:
            required = ("task_text", "qpos", "action_chunks", "sample_frame_indices")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{path} is not a preprocessed PI05 surface shard; missing {missing}")
            if point_target == "visual":
                required_points = ("visual_robot_current_points", "visual_robot_future_offsets", "visual_robot_flow_supervision_mask")
                missing = [key for key in required_points if key not in data]
                if missing:
                    raise ValueError(f"{path} has no visual point target; missing {missing}")
                current = np.asarray(data["visual_robot_current_points"], dtype=np.float32)
                offsets = np.asarray(data["visual_robot_future_offsets"], dtype=np.float32)
                mask = np.asarray(data["visual_robot_flow_supervision_mask"], dtype=bool)
            else:
                current = np.asarray(data["current_link_points"], dtype=np.float32)
                offsets = np.asarray(data["target_point_offsets"], dtype=np.float32)
                mask = np.ones(offsets.shape[:-1], dtype=bool)
            actions = np.asarray(data["action_chunks"], dtype=np.float32)
            sample_frames = np.asarray(data["sample_frame_indices"], dtype=np.int64)
            qpos = np.asarray(data["qpos"], dtype=np.float32)[sample_frames, :6]
            rgb = {key[4:]: np.asarray(data[key], dtype=np.uint8)[sample_frames] for key in data.files if key.startswith("rgb_")}
            task_text = str(np.asarray(data["task_text"]).item())
        if not rgb:
            raise ValueError(f"{path} contains no rgb_* arrays")
        if actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError(f"{path} action_chunks must have shape [N,H,7], got {actions.shape}")
        if action_horizon is not None and actions.shape[1] != action_horizon:
            raise ValueError(f"{path} horizon {actions.shape[1]} does not match --action-horizon={action_horizon}")
        if current.ndim != 3 or current.shape[0] != actions.shape[0] or current.shape[-1] != 3:
            raise ValueError(f"{path} current points must have shape [N,K,3], got {current.shape}")
        if offsets.shape[:3] != (actions.shape[0], actions.shape[1], current.shape[1]) or offsets.shape[-1] != 3:
            raise ValueError(f"{path} point target shapes do not align with action_chunks")
        if mask.shape != offsets.shape[:-1]:
            raise ValueError(f"{path} point target mask must have shape {offsets.shape[:-1]}, got {mask.shape}")
        ids = self._point_indices(current.shape[1], max_points)
        return _Episode(
            path=path,
            task_text=task_text,
            qpos=qpos,
            actions=actions,
            current_points=current[:, ids],
            target_offsets=offsets[:, :, ids],
            target_mask=mask[:, :, ids],
            rgb=rgb,
            sample_count=actions.shape[0],
        )

    def set_camera_map(self, mapping: dict[str, str]) -> None:
        for episode in self.episodes:
            if set(episode.rgb) != set(self.episodes[0].rgb):
                raise ValueError("Every shard must contain the same rgb_* camera names")
        self.camera_map = mapping

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        episode_index, sample_index = self.index[index]
        episode = self.episodes[episode_index]
        qpos = (episode.qpos[sample_index] - self.qpos_mean) / self.qpos_std
        state = np.zeros((32,), dtype=np.float32)
        state[:6] = qpos
        tokens, token_mask = self.tokenizer.tokenize(episode.task_text, state)
        images = {key: np.zeros(self.image_shape, dtype=np.uint8) for key in MODEL_IMAGE_KEYS}
        image_masks = {key: np.asarray(False) for key in MODEL_IMAGE_KEYS}
        for source, target in self.camera_map.items():
            images[target] = episode.rgb[source][sample_index]
            image_masks[target] = np.asarray(True)
        actions = np.zeros((episode.actions.shape[1], 32), dtype=np.float32)
        actions[:, :7] = (episode.actions[sample_index] - self.action_mean) / self.action_std
        return {
            "images": images,
            "image_masks": image_masks,
            "state": state,
            "tokenized_prompt": tokens.astype(np.int64),
            "tokenized_prompt_mask": token_mask,
            "actions": actions,
            "robot_points": episode.current_points[sample_index],
            "joint_positions": qpos,
            "target_point_offsets": episode.target_offsets[sample_index],
            "target_point_mask": episode.target_mask[sample_index],
        }


def _normalization_stats(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    qpos_values, action_values, horizon = [], [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            qpos = np.asarray(data["qpos"], dtype=np.float32)
            sample_frames = np.asarray(data["sample_frame_indices"], dtype=np.int64)
            actions = np.asarray(data["action_chunks"], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError(f"{path} must contain action_chunks[N,H,7]")
        if horizon is not None and horizon != actions.shape[1]:
            raise ValueError("All shards must have the same action horizon")
        horizon = actions.shape[1]
        qpos_values.append(qpos[sample_frames, :6])
        action_values.append(actions.reshape(-1, 7))
    qpos_all, action_all = np.concatenate(qpos_values), np.concatenate(action_values)
    return (
        qpos_all.mean(axis=0).astype(np.float32),
        qpos_all.std(axis=0).clip(1e-6).astype(np.float32),
        action_all.mean(axis=0).astype(np.float32),
        action_all.std(axis=0).clip(1e-6).astype(np.float32),
        int(horizon),
    )


def _to_observation(batch: dict[str, Any], device: torch.device) -> SimpleNamespace:
    # We deliberately use a lightweight observation object to keep this loader
    # independent of JAX. Mirror Observation.from_dict's uint8 NHWC -> float
    # NCHW [-1, 1] conversion before PI05's PyTorch preprocessor sees it.
    images = {
        key: batch["images"][key].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        for key in MODEL_IMAGE_KEYS
    }
    return SimpleNamespace(
        images=images,
        image_masks={key: batch["image_masks"][key].to(device=device, dtype=torch.bool) for key in MODEL_IMAGE_KEYS},
        state=batch["state"].to(device=device, dtype=torch.float32),
        tokenized_prompt=batch["tokenized_prompt"].to(device=device, dtype=torch.long),
        tokenized_prompt_mask=batch["tokenized_prompt_mask"].to(device=device, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def _save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "PI05SafetyPytorch", "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "metadata": metadata}, path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_points < 0:
        raise ValueError("epochs and batch-size must be positive; max-points must be >= 0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    shards = _resolve_shards(args.dataset)
    qpos_mean, qpos_std, action_mean, action_std, detected_horizon = _normalization_stats(shards)
    horizon = detected_horizon if args.action_horizon is None else args.action_horizon
    tokenizer = _Tokenizer(max_len=200, model_path=args.tokenizer_model)
    dataset = Ur7eSurfaceDataset(
        shards,
        max_points=args.max_points,
        point_target=args.point_target,
        action_horizon=horizon,
        tokenizer=tokenizer,
        qpos_mean=qpos_mean,
        qpos_std=qpos_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    rgb_names = sorted(dataset.episodes[0].rgb)
    dataset.set_camera_map(_parse_camera_map(args.camera_map, rgb_names))
    if len(dataset) < args.batch_size:
        raise ValueError(f"Dataset has {len(dataset)} samples, smaller than --batch-size={args.batch_size}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI05SafetyPytorch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenPI dependencies are unavailable. Run this script in the OpenPI environment, for example: "
            "cd openpi && uv run --project . ../scripts/train_pi05_ur7e_surface_pytorch.py ..."
        ) from exc
    config = Pi0Config(action_dim=32, action_horizon=horizon, pi05=True, dtype=args.precision)
    model = PI05SafetyPytorch(config).to(device)
    if args.pretrained is not None:
        from safetensors.torch import load_file

        incompatible = model.load_state_dict(load_file(str(args.pretrained)), strict=False)
        print(f"[weights] missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")
    if args.freeze_base:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("surface_"))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    metadata = {
        "dataset": [str(path) for path in shards], "point_target": args.point_target, "point_count": dataset.point_count,
        "camera_map": dataset.camera_map, "action_mean": action_mean, "action_std": action_std,
        "qpos_mean": qpos_mean, "qpos_std": qpos_std, "action_horizon": horizon,
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_action = total_point = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_losses(
                _to_observation(batch, device),
                batch["actions"].to(device=device, dtype=torch.float32),
                batch["robot_points"].to(device=device, dtype=torch.float32),
                batch["joint_positions"].to(device=device, dtype=torch.float32),
                batch["target_point_offsets"].to(device=device, dtype=torch.float32),
                batch["target_point_mask"].to(device=device, dtype=torch.bool),
                point_loss_weight=args.point_loss_weight,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += float(losses["loss"].detach())
            total_action += float(losses["action_loss"])
            total_point += float(losses["point_loss"])
        batches = len(loader)
        print(f"epoch={epoch:03d} loss={total_loss / batches:.6f} action={total_action / batches:.6f} point={total_point / batches:.6f}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            _save_checkpoint(args.output, model, optimizer, epoch, metadata)


if __name__ == "__main__":
    main()
