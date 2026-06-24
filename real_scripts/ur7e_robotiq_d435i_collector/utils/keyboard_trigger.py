from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class CollectorMode(str, Enum):
    IDLE = "idle"
    TELEOP = "teleop"
    RECORDING = "recording"


@dataclass
class CollectorState:
    save_on_teleop_stop: bool = True
    mode: CollectorMode = CollectorMode.IDLE

    def on_teleop_toggle(self) -> str:
        if self.mode == CollectorMode.IDLE:
            self.mode = CollectorMode.TELEOP
            return "teleop_started"
        if self.mode == CollectorMode.TELEOP:
            self.mode = CollectorMode.IDLE
            return "teleop_stopped"
        self.mode = CollectorMode.IDLE
        return "record_saved_and_teleop_stopped" if self.save_on_teleop_stop else "record_discarded_and_teleop_stopped"

    def on_record_toggle(self) -> str:
        if self.mode == CollectorMode.IDLE:
            return "record_ignored"
        if self.mode == CollectorMode.TELEOP:
            self.mode = CollectorMode.RECORDING
            return "record_started"
        self.mode = CollectorMode.TELEOP
        return "record_stopped"


class FunctionKeyListener:
    def __init__(self, on_teleop: Callable[[], None], on_record: Callable[[], None]):
        self.on_teleop = on_teleop
        self.on_record = on_record
        self._listener = None
        self._ready = threading.Event()

    def start(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:
            raise RuntimeError("pynput is required for F2/F3 keyboard capture") from exc

        def on_press(key):
            if key == keyboard.Key.f2:
                self.on_teleop()
            elif key == keyboard.Key.f3:
                self.on_record()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        self._ready.set()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
