import numpy as np

from scripts.serve_pi05_prefix_policy import PrefixTokenPolicy


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
