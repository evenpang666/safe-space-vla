from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    timestamp_s: float


class RealSenseRGBDCamera:
    def __init__(
        self,
        name: str,
        serial: str,
        width: int,
        height: int,
        fps: int,
        align_depth_to_color: bool = True,
    ):
        self.name = name
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.align_depth_to_color = bool(align_depth_to_color)

        self._pipeline: Any | None = None
        self._align: Any | None = None

    def connect(self) -> None:
        if self._pipeline is not None:
            return

        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise RuntimeError("pyrealsense2 is required for RealSense RGB-D cameras") from exc

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        try:
            pipeline.start(config)
        except Exception as exc:
            raise RuntimeError(f"Failed to start RealSense camera {self.name!r}") from exc

        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color) if self.align_depth_to_color else None

        try:
            for _ in range(10):
                frames = pipeline.wait_for_frames()
                if self._align is not None:
                    self._align.process(frames)
        except Exception:
            self.disconnect()
            raise

    def get_frame(self) -> RGBDFrame:
        if self._pipeline is None:
            raise RuntimeError(f"RealSense camera {self.name!r} is not connected")

        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(f"RealSense camera {self.name!r} returned incomplete RGB-D frames")

        rgb = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        return RGBDFrame(rgb=rgb, depth=depth, timestamp_s=time.time())

    def disconnect(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._align = None
        if pipeline is not None:
            pipeline.stop()


class MultiRGBDCamera:
    def __init__(self, configs: dict[str, dict[str, Any]]):
        self.cameras = {
            name: RealSenseRGBDCamera(
                name=name,
                serial=str(config.get("serial", "")),
                width=int(config["width"]),
                height=int(config["height"]),
                fps=int(config["fps"]),
                align_depth_to_color=bool(config.get("align_depth_to_color", True)),
            )
            for name, config in configs.items()
        }
        self._latest: dict[str, RGBDFrame] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_error: BaseException | None = None

    def connect(self) -> None:
        if self._thread is not None:
            return

        connected: list[RealSenseRGBDCamera] = []
        try:
            for camera in self.cameras.values():
                camera.connect()
                connected.append(camera)

            latest = {name: camera.get_frame() for name, camera in self.cameras.items()}
        except Exception:
            for camera in reversed(connected):
                camera.disconnect()
            raise

        with self._lock:
            self._latest = latest
            self._capture_error = None

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="multi-rgbd-camera", daemon=True)
        self._thread.start()

    def get_latest(self) -> dict[str, dict[str, np.ndarray]]:
        with self._lock:
            if self._capture_error is not None:
                raise RuntimeError("RGB-D camera capture thread failed") from self._capture_error
            latest = dict(self._latest)

        missing = set(self.cameras) - set(latest)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise RuntimeError(f"Missing latest RGB-D frames for camera(s): {missing_names}")

        return {
            name: {
                "rgb": frame.rgb.copy(),
                "depth": frame.depth.copy(),
            }
            for name, frame in latest.items()
        }

    def disconnect(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                print("[Camera] RGB-D capture thread did not stop within timeout; continuing shutdown.")

        for camera in self.cameras.values():
            camera.disconnect()

        with self._lock:
            self._latest = {}
            self._capture_error = None

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                for name, camera in self.cameras.items():
                    if self._stop_event.is_set():
                        break
                    frame = camera.get_frame()
                    with self._lock:
                        self._latest[name] = frame
            except BaseException as exc:
                with self._lock:
                    self._capture_error = exc
                self._stop_event.set()
