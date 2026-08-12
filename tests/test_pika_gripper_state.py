from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.pika_gripper_state import (
    PIKA_MAX_OPENING_M,
    PikaGripperState,
    nearest_pika_state,
    normalize_pika_opening,
)


def test_opening_normalization_supports_explicit_units():
    assert normalize_pika_opening(95.0, unit="mm") == PIKA_MAX_OPENING_M
    assert normalize_pika_opening(0.5, unit="ratio") == PIKA_MAX_OPENING_M * 0.5
    assert normalize_pika_opening(0.02, unit="m") == 0.02
    with pytest.raises(ValueError):
        normalize_pika_opening(95.0, unit="m")


def test_nearest_pika_state_rejects_stale_state():
    states = [PikaGripperState(100_000_000, 0.02), PikaGripperState(130_000_000, 0.04)]
    assert nearest_pika_state(states, timestamp_ns=128_000_000, max_delta_ns=5_000_000).opening_m == 0.04
    with pytest.raises(LookupError):
        nearest_pika_state(states, timestamp_ns=200_000_000, max_delta_ns=10_000_000)
