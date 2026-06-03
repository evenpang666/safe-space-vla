#!/usr/bin/env python3
"""Serve an OpenPI policy over websocket and include PI05 prefix tokens."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
    return parser.parse_args()


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
    def __init__(self, policy, *, prefix_extractor: Callable = extract_prefix_tokens):
        self._policy = policy
        self._prefix_extractor = prefix_extractor
        self.metadata = getattr(policy, "metadata", {})

    def infer(self, obs: dict) -> dict:
        result = dict(self._policy.infer(obs))
        result["prefix_tokens"] = np.asarray(self._prefix_extractor(self._policy, obs), dtype=np.float32)
        return result

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if reset is not None:
            reset()


def create_policy(args: argparse.Namespace):
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    base_policy = _policy_config.create_trained_policy(
        _config.get_config(args.policy_config),
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    return PrefixTokenPolicy(base_policy)


def main() -> None:
    args = parse_args()
    from openpi.serving import websocket_policy_server

    policy = create_policy(args)
    hostname = socket.gethostname()
    logging.info("Creating prefix-token policy server on %s:%s", hostname, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={**getattr(policy, "metadata", {}), "returns_prefix_tokens": True},
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
