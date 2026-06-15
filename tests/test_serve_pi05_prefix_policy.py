import argparse

import numpy as np
import torch

from scripts import serve_pi05_prefix_policy as server
from scripts.serve_pi05_prefix_policy import PrefixTokenPolicy, resolve_torch_device


class _FakePolicy:
    metadata = {"policy": "fake"}

    def infer(self, obs):
        return {"actions": np.asarray([[1.0, 2.0]], dtype=np.float32)}


def test_prefix_token_policy_adds_prefix_tokens_to_policy_response():
    wrapper = PrefixTokenPolicy(
        _FakePolicy(),
        prefix_extractor=lambda _policy, obs: np.full((3, 4), float(obs["value"]), dtype=np.float32),
    )

    result = wrapper.infer({"value": 2})

    np.testing.assert_allclose(result["actions"], [[1.0, 2.0]])
    np.testing.assert_allclose(result["prefix_tokens"], np.full((3, 4), 2.0, dtype=np.float32))


def test_prefix_token_policy_handles_safety_only_request_without_base_policy_call():
    class _ExplodingPolicy:
        metadata = {}

        def infer(self, _obs):
            raise AssertionError("base policy should not be called for safety_only")

    def fake_safety_predictor(prefix_tokens, current_link_points):
        value = float(np.asarray(prefix_tokens)[0, 0])
        return np.asarray(current_link_points, dtype=np.float32) + value

    wrapper = PrefixTokenPolicy(_ExplodingPolicy(), safety_predictor=fake_safety_predictor)

    result = wrapper.infer(
        {
            "safety_only": True,
            "prefix_tokens": np.asarray([[2.0]], dtype=np.float32),
            "current_link_points": np.zeros((1, 2, 3), dtype=np.float32),
        }
    )

    assert result.keys() == {"pred_link_points"}
    np.testing.assert_allclose(result["pred_link_points"], np.full((1, 2, 3), 2.0, dtype=np.float32))


def test_prefix_token_policy_safety_only_requires_loaded_safety_module():
    wrapper = PrefixTokenPolicy(_FakePolicy())

    try:
        wrapper.infer({"safety_only": True, "prefix_tokens": np.zeros((1, 1), dtype=np.float32)})
    except RuntimeError as exc:
        assert "safety module" in str(exc)
    else:
        raise AssertionError("safety_only request was accepted without a safety module")


def test_resolve_torch_device_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_torch_device("auto") == torch.device("cpu")
    assert resolve_torch_device("gpu") == torch.device("cuda")


def test_create_policy_rewrites_missing_jaxlib_error(monkeypatch):
    def fake_import(name, *args, **kwargs):
        if name == "openpi.policies":
            cause = ModuleNotFoundError("No module named 'jaxlib'")
            raise ModuleNotFoundError("jax requires jaxlib to be installed") from cause
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    args = argparse.Namespace(
        policy_config="pi05_libero",
        checkpoint_dir="gs://openpi-assets/checkpoints/pi05_libero",
        default_prompt=None,
        pytorch_device=None,
        safety_checkpoint=None,
        safety_device="auto",
        safety_prediction_steps=10,
    )
    try:
        server.create_policy(args)
    except RuntimeError as exc:
        message = str(exc)
        assert "jaxlib is missing" in message
        assert "uv run --project openpi" in message
        assert "Python >=3.11" in message
    else:
        raise AssertionError("missing jaxlib error was not rewritten")
