#!/usr/bin/env python3
"""Serve an OpenPI policy over websocket and include PI05 prefix tokens."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
import socket
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENPI_ROOT = REPO_ROOT / "openpi"
OPENPI_SRC = OPENPI_ROOT / "src"
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"
for path in (OPENPI_SRC, OPENPI_CLIENT_SRC, REPO_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--pytorch-device", default=None)
    parser.add_argument(
        "--safety-checkpoint",
        type=Path,
        default=None,
        help="Optional trained safety decoder / SafetyFlowPointModel checkpoint to serve with PI05.",
    )
    parser.add_argument(
        "--safety-device",
        default="auto",
        help="Torch device for --safety-checkpoint. 'auto' chooses cuda when available, otherwise cpu.",
    )
    parser.add_argument("--safety-prediction-steps", type=int, default=10, help="Euler steps for SafetyFlowPointModel.")
    return parser.parse_args()


def resolve_torch_device(device: str | None):
    import torch

    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "gpu":
        return torch.device("cuda")
    return torch.device(device)


def extract_prefix_tokens(policy, obs: dict) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model

    inputs = jax.tree.map(lambda x: x, obs)
    inputs = policy._input_transform(inputs)

    if getattr(policy, "_is_pytorch_model", False):
        import torch

        device = getattr(policy, "_pytorch_device", "cpu")
        tensor_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device)[None, ...], inputs)
        observation = _model.Observation.from_dict(tensor_inputs)
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, _state = policy._model._preprocess_observation(
                observation, train=False
            )
            prefix_tokens, _prefix_pad_masks, _prefix_att_masks = policy._model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        return prefix_tokens[0].detach().to(dtype=torch.float32).cpu().numpy()

    batch_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    observation = _model.Observation.from_dict(batch_inputs)
    prefix_tokens, _prefix_mask, _prefix_ar_mask = policy._model.embed_prefix(observation)
    return np.asarray(prefix_tokens[0], dtype=np.float32)


class PrefixTokenPolicy:
    def __init__(
        self,
        policy,
        *,
        prefix_extractor: Callable = extract_prefix_tokens,
        safety_predictor: Callable | None = None,
    ):
        self._policy = policy
        self._prefix_extractor = prefix_extractor
        self._safety_predictor = safety_predictor
        self.metadata = getattr(policy, "metadata", {})

    def infer(self, obs: dict) -> dict:
        if bool(obs.get("safety_only", False)):
            if self._safety_predictor is None:
                raise RuntimeError("No safety module is loaded; start server with --safety-checkpoint.")
            if "prefix_tokens" not in obs:
                raise KeyError("safety_only request requires 'prefix_tokens'")
            return {
                "pred_link_points": np.asarray(
                    self._safety_predictor(obs["prefix_tokens"], obs.get("current_link_points")),
                    dtype=np.float32,
                )
            }

        result = dict(self._policy.infer(obs))
        result["prefix_tokens"] = np.asarray(self._prefix_extractor(self._policy, obs), dtype=np.float32)
        return result

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if reset is not None:
            reset()


@dataclass
class SafetyModulePredictor:
    model_type: str
    model: object
    device: object
    prediction_steps: int
    config: object | None = None
    model_kwargs: dict | None = None

    @property
    def metadata(self) -> dict:
        metadata = {
            "returns_safety_predictions": True,
            "safety_model_type": self.model_type,
            "safety_device": str(self.device),
        }
        if self.model_type == "flow":
            metadata.update(
                {
                    "safety_flow_max_points": int(self.model.flow_head.max_points),
                    "safety_flow_n_future": int(self.model.flow_head.n_future),
                    "safety_prediction_steps": int(self.prediction_steps),
                }
            )
        elif self.config is not None:
            metadata["safety_decoder_points_per_link"] = int(self.config.points_per_link)
        return metadata

    def __call__(self, prefix_tokens, current_link_points=None) -> np.ndarray:
        import torch

        if self.model_type == "flow":
            if current_link_points is None:
                raise KeyError("SafetyFlowPointModel prediction requires 'current_link_points'")
            from safety_module.safety_flow_point_model import euler_sample

            current = np.asarray(current_link_points, dtype=np.float32)
            if current.ndim != 3 or current.shape[-1] != 3:
                raise ValueError(f"current_link_points must have shape (L, P, 3), got {current.shape}")
            prefix = np.asarray(prefix_tokens, dtype=np.float32)
            if prefix.ndim == 2:
                prefix = prefix[None, ...]
            if prefix.ndim != 3 or prefix.shape[0] != 1:
                raise ValueError(f"prefix_tokens must have shape (N, D) or (1, N, D), got {prefix.shape}")
            arm_points = current.reshape(1, -1, 3)
            if arm_points.shape[1] != int(self.model.flow_head.max_points):
                raise ValueError(
                    f"current arm point count {arm_points.shape[1]} must equal flow model max_points="
                    f"{int(self.model.flow_head.max_points)}"
                )
            with torch.no_grad():
                offsets = euler_sample(
                    model=self.model,
                    arm_points=torch.as_tensor(arm_points, dtype=torch.float32, device=self.device),
                    prefix_tokens=torch.as_tensor(prefix, dtype=torch.float32, device=self.device),
                    n_steps=int(self.prediction_steps),
                    n_future=int(self.model.flow_head.n_future),
                    K=arm_points.shape[1],
                )
            delta = offsets[0].detach().cpu().numpy().astype(np.float32, copy=False)
            return (current[None, :, :, :] + delta.reshape(delta.shape[0], *current.shape)).astype(
                np.float32,
                copy=False,
            )

        prefix = np.asarray(prefix_tokens, dtype=np.float32)
        if prefix.ndim == 2:
            prefix = prefix[None, ...]
        if prefix.ndim != 3 or prefix.shape[0] != 1:
            raise ValueError(f"prefix_tokens must have shape (N, D) or (1, N, D), got {prefix.shape}")
        with torch.no_grad():
            pred = self.model(torch.as_tensor(prefix, dtype=torch.float32, device=self.device))
        return pred[0].detach().cpu().numpy().astype(np.float32, copy=False)


def load_safety_predictor(path: Path, *, device_name: str | None, prediction_steps: int) -> SafetyModulePredictor:
    import torch
    from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig
    from safety_module.safety_flow_point_model import SafetyFlowPointModel

    device = resolve_torch_device(device_name)
    load_kwargs = {"map_location": device}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
    except (TypeError, ValueError):
        pass
    payload = torch.load(path, **load_kwargs)
    model_type = str(payload.get("model_type", "SafetyPointDecoder"))
    if model_type == "SafetyFlowPointModel" or "model_kwargs" in payload:
        model_kwargs = dict(payload["model_kwargs"])
        model = SafetyFlowPointModel(**model_kwargs).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return SafetyModulePredictor(
            model_type="flow",
            model=model,
            device=device,
            prediction_steps=prediction_steps,
            model_kwargs=model_kwargs,
        )

    config = SafetyPointDecoderConfig.from_dict(payload["config"])
    model = SafetyPointDecoder(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return SafetyModulePredictor(
        model_type="decoder",
        model=model,
        device=device,
        prediction_steps=prediction_steps,
        config=config,
    )


def create_policy(args: argparse.Namespace):
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    base_policy = _policy_config.create_trained_policy(
        _config.get_config(args.policy_config),
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    safety_predictor = None
    if args.safety_checkpoint is not None:
        safety_predictor = load_safety_predictor(
            args.safety_checkpoint,
            device_name=args.safety_device,
            prediction_steps=args.safety_prediction_steps,
        )
        logging.info(
            "Loaded %s safety module from %s on %s",
            safety_predictor.model_type,
            args.safety_checkpoint,
            safety_predictor.device,
        )
    return PrefixTokenPolicy(base_policy, safety_predictor=safety_predictor)


def main() -> None:
    args = parse_args()
    from openpi.serving import websocket_policy_server

    policy = create_policy(args)
    hostname = socket.gethostname()
    logging.info("Creating prefix-token policy server on %s:%s", hostname, args.port)
    metadata = {**getattr(policy, "metadata", {}), "returns_prefix_tokens": True}
    safety_predictor = getattr(policy, "_safety_predictor", None)
    if safety_predictor is not None:
        metadata.update(safety_predictor.metadata)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
