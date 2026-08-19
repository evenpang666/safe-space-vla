"""Read PiKA's live jaw opening without enabling or commanding the motor."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


class PikaOpeningReader:
    """Read ``Gripper.get_gripper_distance()`` in millimetres over PiKA USB."""

    def __init__(self, port: str, *, max_opening_mm: float = 95.0) -> None:
        self.port = str(port)
        self.max_opening_mm = float(max_opening_mm)
        if self.max_opening_mm <= 0.0:
            raise ValueError("max_opening_mm must be positive")
        self._device = None

    def connect(self) -> None:
        # The vendored SDK is deliberately imported only when a caller opts in.
        # No enable(), set_motor_angle(), or disable() call is made here.
        sdk_root = REPO_ROOT / "real_scripts" / "ur7e_robotiq_d435i_collector" / "pika_sdk"
        if not sdk_root.is_dir():
            raise FileNotFoundError(f"PiKA SDK directory does not exist: {sdk_root}")
        if str(sdk_root) not in sys.path:
            sys.path.insert(0, str(sdk_root))
        from pika.gripper import Gripper

        device = Gripper(self.port)
        if not device.connect():
            raise RuntimeError(f"PiKA gripper did not connect on {self.port}")
        self._device = device

    def opening_mm(self) -> float:
        if self._device is None:
            raise RuntimeError("PiKA opening reader is not connected")
        opening_mm = float(self._device.get_gripper_distance())
        if not np.isfinite(opening_mm) or not 0.0 <= opening_mm <= self.max_opening_mm:
            raise RuntimeError(
                f"PiKA opening readback {opening_mm!r} mm is outside [0, {self.max_opening_mm}]"
            )
        return opening_mm

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.disconnect()
            except Exception:
                pass
            self._device = None
