#!/usr/bin/env python3
"""Serve a jointly trained Quest3 PI05 action + surface-flow checkpoint."""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "openpi" / "src", REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DEFAULT_TOKENIZER_REPOSITORY = "google/paligemma-3b-pt-224"
DEFAULT_TOKENIZER_FILENAME = "tokenizer.model"


def _cached_huggingface_tokenizer() -> Path | None:
    hub_root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    candidates = sorted((hub_root / "models--google--paligemma-3b-pt-224" / "snapshots").glob("*/tokenizer.model"))
    return candidates[-1] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--tokenizer-model", type=Path, default=None)
    return parser.parse_args()


class _Tokenizer:
    def __init__(self, model_path: Path | None) -> None:
        self.max_len = 200
        if model_path is None:
            model_path = _cached_huggingface_tokenizer()
        if model_path is None:
            try:
                from huggingface_hub import hf_hub_download

                model_path = Path(hf_hub_download(repo_id=DEFAULT_TOKENIZER_REPOSITORY, filename=DEFAULT_TOKENIZER_FILENAME))
            except Exception as exc:
                raise RuntimeError(
                    "PaliGemma tokenizer is absent from the Hugging Face cache and could not be downloaded. "
                    "Copy tokenizer.model to the normal cache or pass --tokenizer-model."
                ) from exc
        import sentencepiece

        self._processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))

    def tokenize(self, prompt: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {prompt.strip().replace('_', ' ').replace(chr(10), ' ')}, State: {' '.join(map(str, bins))};\nAction: "
        tokens = self._processor.encode(text, add_bos=True)[: self.max_len]
        values = np.zeros((self.max_len,), dtype=np.int32)
        mask = np.zeros((self.max_len,), dtype=bool)
        values[: len(tokens)], mask[: len(tokens)] = tokens, True
        return values, mask


class Quest3JointSafetyPolicy:
    def __init__(self, checkpoint: Path, *, device_name: str, num_steps: int, tokenizer_model: Path | None) -> None:
        import torch
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI05SafetyPytorch

        self.torch = torch
        self.device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
        # Keep parameter tensors memory-mapped until ``assign=True`` hands
        # their storage directly to the module.  A conventional torch.load +
        # copy produces a state dict, an initialized model and then a CUDA
        # model at once; that can exhaust both 32 GB host RAM and 16 GB GPUs.
        # The checkpoint is a zip-format torch.save file, which supports mmap.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("model_type") != "PI05SafetyPytorch":
            raise ValueError(f"{checkpoint} is not a jointly trained PI05SafetyPytorch checkpoint")
        self.info = dict(payload["metadata"])
        required = ("action_mean", "action_std", "qpos_mean", "qpos_std", "action_horizon", "logical_action_dim", "joint_dim", "selected_point_indices", "camera_map")
        missing = [key for key in required if key not in self.info]
        if missing:
            raise ValueError(f"Checkpoint metadata is missing {missing}")
        self.action_mean = np.asarray(self.info["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(self.info["action_std"], dtype=np.float32)
        self.qpos_mean = np.asarray(self.info["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.asarray(self.info["qpos_std"], dtype=np.float32)
        self.logical_action_dim = int(self.info["logical_action_dim"])
        self.joint_dim = int(self.info["joint_dim"])
        self.point_indices = np.asarray(self.info["selected_point_indices"], dtype=np.int64)
        self.camera_map = {str(key): str(value) for key, value in self.info["camera_map"].items()}
        config = Pi0Config(
            action_dim=32, action_horizon=int(self.info["action_horizon"]), pi05=True,
            dtype=str(self.info.get("precision", "bfloat16")),
        )
        # Constructing directly on ``meta`` prevents an otherwise-unused
        # ~6 GB CPU parameter allocation.  ``to_empty`` allocates the final
        # parameter storage only on the target device; load_state_dict then
        # copies each memory-mapped tensor in turn, rather than retaining a
        # complete extra CUDA state dict.  Some rotary buffers are not stored
        # in the checkpoint, so assigning tensor storage directly is unsafe.
        with torch.device("meta"):
            self.model = PI05SafetyPytorch(config, joint_dim=self.joint_dim)
        self.model.to_empty(device=self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model.eval()
        self.tokenizer = _Tokenizer(tokenizer_model)
        self.num_steps = int(num_steps)
        self.metadata = {
            "model_type": "PI05SafetyPytorch", "joint_action_and_point_flow": True,
            "coordinate_frame": self.info.get("coordinate_frame", "left_base"),
            "action_horizon": int(self.info["action_horizon"]), "logical_action_dim": self.logical_action_dim,
            "joint_dim": self.joint_dim, "point_count": len(self.point_indices), "camera_map": self.camera_map,
            "surface_model_hash": self.info.get("surface_model_hash"),
            "point_identity_version": self.info.get("point_identity_version"),
            "points_per_link": self.info.get("points_per_link"),
        }

    def _points(self, value: object) -> np.ndarray:
        points = np.asarray(value, dtype=np.float32)
        if points.ndim == 3 and points.shape[-1] == 3:
            points = points.reshape(-1, 3)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"robot_points must have shape [K,3] or [L,P,3], got {points.shape}")
        if len(points) == len(self.point_indices):
            selected = points
        elif len(points) > int(self.point_indices.max(initial=-1)):
            selected = points[self.point_indices]
        else:
            raise ValueError(f"robot_points has {len(points)} rows but checkpoint requires indices through {self.point_indices.max()}")
        if not np.isfinite(selected).all():
            raise ValueError("robot_points contains non-finite values")
        return selected

    def infer(self, obs: dict) -> dict:
        torch = self.torch
        qpos = np.asarray(obs["qpos"], dtype=np.float32).reshape(-1)
        if qpos.shape != (self.joint_dim,) or not np.isfinite(qpos).all():
            raise ValueError(f"qpos must be finite shape [{self.joint_dim}], got {qpos.shape}")
        normalized_qpos = (qpos - self.qpos_mean) / self.qpos_std
        state = np.zeros((32,), dtype=np.float32)
        state[: self.joint_dim] = normalized_qpos
        tokens, token_mask = self.tokenizer.tokenize(str(obs.get("prompt", "")), state)
        available: dict[str, np.ndarray] = {}
        for source, target in self.camera_map.items():
            key = f"rgb_{source}"
            if key in obs:
                image = np.asarray(obs[key], dtype=np.uint8)
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"{key} must be HxWx3 RGB")
                available[target] = image
        if not available:
            raise KeyError(f"No checkpoint camera was provided; expected {[f'rgb_{name}' for name in self.camera_map]}")
        shape = next(iter(available.values())).shape
        images, masks = {}, {}
        for key in MODEL_IMAGE_KEYS:
            image = available.get(key, np.zeros(shape, dtype=np.uint8))
            images[key] = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)[None].permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
            masks[key] = torch.tensor([key in available], dtype=torch.bool, device=self.device)
        observation = SimpleNamespace(
            images=images, image_masks=masks,
            state=torch.from_numpy(state).to(self.device)[None],
            tokenized_prompt=torch.from_numpy(tokens.astype(np.int64)).to(self.device)[None],
            tokenized_prompt_mask=torch.from_numpy(token_mask).to(self.device)[None],
            token_ar_mask=None, token_loss_mask=None,
        )
        current = self._points(obs["robot_points"])
        robot_points = torch.from_numpy(current).to(self.device)[None]
        joints = torch.from_numpy(normalized_qpos).to(self.device)[None]
        with torch.inference_mode():
            normalized_actions, offsets = self.model.sample_actions_and_point_offsets(
                self.device, observation, robot_points, joints, num_steps=self.num_steps
            )
        normalized = normalized_actions[0, :, : self.logical_action_dim].float().cpu().numpy()
        actions = normalized * self.action_std[None, :] + self.action_mean[None, :]
        point_offsets = offsets[0].float().cpu().numpy()
        return {
            "actions": actions.astype(np.float32),
            "point_offsets": point_offsets.astype(np.float32),
            "predicted_robot_points": (current[None, ...] + point_offsets).astype(np.float32),
            "selected_point_indices": self.point_indices,
            "surface_model_hash": self.info.get("surface_model_hash", ""),
            "point_identity_version": self.info.get("point_identity_version", ""),
            "points_per_link": int(self.info.get("points_per_link", 0)),
        }

    def reset(self) -> None:
        return None


def main() -> None:
    args = parse_args()
    from openpi.serving import websocket_policy_server

    policy = Quest3JointSafetyPolicy(args.checkpoint, device_name=args.device, num_steps=args.num_steps, tokenizer_model=args.tokenizer_model)
    logging.info("Serving joint Quest3 PI05 safety policy on %s:%d", socket.gethostname(), args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
