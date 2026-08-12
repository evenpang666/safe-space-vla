#!/usr/bin/env python3
"""PiKA gripper opening-state normalization and timestamp matching helpers.

PiKA ROS exposes gripper state through ``/gripper/data`` and/or
``/gripper/joint_states``.  Keep the hardware-specific subscriber outside this
module and feed its timestamp plus opening measurement into ``PikaGripperState``.
The resulting metres-valued state can be synchronized to a D435i frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PIKA_MAX_OPENING_M = 0.095


@dataclass(frozen=True)
class PikaGripperState:
    timestamp_ns: int
    opening_m: float
    motor_angle_deg: float | None = None

    def __post_init__(self) -> None:
        if int(self.timestamp_ns) < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not np.isfinite(self.opening_m) or not 0.0 <= float(self.opening_m) <= PIKA_MAX_OPENING_M:
            raise ValueError(f"opening_m must be in [0, {PIKA_MAX_OPENING_M}], got {self.opening_m}")
        if self.motor_angle_deg is not None and not np.isfinite(self.motor_angle_deg):
            raise ValueError("motor_angle_deg must be finite when provided")

    @property
    def opening_ratio(self) -> float:
        return float(self.opening_m) / PIKA_MAX_OPENING_M


def normalize_pika_opening(value: float, *, unit: str) -> float:
    """Convert PiKA's reported opening value into metres with strict bounds."""
    unit = unit.lower().strip()
    if unit == "m":
        opening_m = float(value)
    elif unit == "mm":
        opening_m = float(value) / 1000.0
    elif unit == "ratio":
        opening_m = float(value) * PIKA_MAX_OPENING_M
    else:
        raise ValueError("PiKA opening unit must be one of: m, mm, ratio")
    if not np.isfinite(opening_m) or not 0.0 <= opening_m <= PIKA_MAX_OPENING_M:
        raise ValueError(f"PiKA opening is outside [0, {PIKA_MAX_OPENING_M}] m: {opening_m}")
    return opening_m


def nearest_pika_state(
    states: Iterable[PikaGripperState],
    *,
    timestamp_ns: int,
    max_delta_ns: int = 20_000_000,
) -> PikaGripperState:
    """Find a PiKA state close enough to one image capture timestamp.

    Twenty milliseconds is a conservative initial limit for a 30 Hz RGB-D
    pipeline.  Tighten it after measuring camera/robot transport latency.
    """
    candidates = tuple(states)
    if not candidates:
        raise LookupError("no PiKA gripper states are available")
    target = int(timestamp_ns)
    best = min(candidates, key=lambda item: abs(int(item.timestamp_ns) - target))
    delta_ns = abs(int(best.timestamp_ns) - target)
    if delta_ns > int(max_delta_ns):
        raise LookupError(f"nearest PiKA state is {delta_ns / 1e6:.1f} ms from the image")
    return best
