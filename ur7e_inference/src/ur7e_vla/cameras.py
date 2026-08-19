from __future__ import annotations

import logging
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

from .config import CameraConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from real_scripts.real_robot_adapter import RGBDFrame

LOG = logging.getLogger(__name__)


class LatestFrame:
    def __init__(self, read_frame: Callable[[], np.ndarray], name: str, max_age_s: float):
        self._read_frame = read_frame
        self._name = name
        self._frame: Optional[np.ndarray] = None
        self._frame_time: Optional[float] = None
        self._error: Optional[Exception] = None
        self._max_age_s = max_age_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"camera-{name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._read_frame()
                if frame is None or frame.size == 0:
                    raise RuntimeError(f"{self._name} returned an empty frame")
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.monotonic()
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = exc
                time.sleep(0.05)

    def get(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                is_fresh = self._frame_time is not None and time.monotonic() - self._frame_time <= self._max_age_s
                if self._frame is not None and is_fresh:
                    return self._frame.copy()
                error = self._error
            if error is not None and time.monotonic() + 0.1 >= deadline:
                raise RuntimeError(f"Failed to read {self._name}") from error
            time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for {self._name}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class LatestRGBDFrame:
    """Latest time-synchronised colour/depth sample from one RealSense pipeline."""

    def __init__(self, read_frame: Callable[[], RGBDFrame], name: str, max_age_s: float):
        self._read_frame = read_frame
        self._name = name
        self._frame: Optional[RGBDFrame] = None
        self._frame_time: Optional[float] = None
        self._error: Optional[Exception] = None
        self._max_age_s = max_age_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"rgbd-camera-{name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._read_frame()
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.monotonic()
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = exc
                time.sleep(0.05)

    def get(self, timeout_s: float = 5.0) -> RGBDFrame:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                is_fresh = self._frame_time is not None and time.monotonic() - self._frame_time <= self._max_age_s
                if self._frame is not None and is_fresh:
                    frame = self._frame
                    return RGBDFrame(
                        frame.camera_name,
                        frame.rgb.copy(),
                        frame.depth_m.copy(),
                        host_timestamp_ns=frame.host_timestamp_ns,
                        device_timestamp_ms=frame.device_timestamp_ms,
                        frame_number=frame.frame_number,
                        timestamp_domain=frame.timestamp_domain,
                    )
                error = self._error
            if error is not None and time.monotonic() + 0.1 >= deadline:
                raise RuntimeError(f"Failed to read {self._name}") from error
            time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for {self._name}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class CameraPair:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._pipeline = None
        self._side_pipeline = None
        self._wrist_pipeline = None
        self._capture = None
        self._exterior: Optional[LatestRGBDFrame] = None
        self._side: Optional[LatestRGBDFrame] = None
        self._wrist: Optional[LatestFrame] = None

    def start(self) -> None:
        if self._exterior is not None or self._side is not None or self._wrist is not None:
            return
        import cv2
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        if self.cfg.realsense_serial:
            rs_cfg.enable_device(self.cfg.realsense_serial)
        else:
            # The exterior stream is the original D435/D435i.  Do not let
            # librealsense select an arbitrary device when a wrist D405 is
            # connected as well.
            d435_devices = []
            for device in rs.context().query_devices():
                name = device.get_info(rs.camera_info.name)
                if "D435" in name.upper():
                    d435_devices.append(device.get_info(rs.camera_info.serial_number))
            if len(d435_devices) == 1:
                self.cfg.realsense_serial = d435_devices[0]
                rs_cfg.enable_device(self.cfg.realsense_serial)
                LOG.info("Auto-selected original front D435i serial %s", self.cfg.realsense_serial)
            elif len(d435_devices) > 1:
                raise RuntimeError(
                    f"Multiple D435/D435i cameras found: {d435_devices}; set cameras.realsense_serial explicitly."
                )
        rs_cfg.enable_stream(
            rs.stream.color,
            self.cfg.realsense_width,
            self.cfg.realsense_height,
            rs.format.bgr8,
            self.cfg.realsense_fps,
        )
        rs_cfg.enable_stream(
            rs.stream.depth,
            self.cfg.realsense_width,
            self.cfg.realsense_height,
            rs.format.z16,
            self.cfg.realsense_fps,
        )
        pipeline.start(rs_cfg)
        self._pipeline = pipeline
        align_to_colour = rs.align(rs.stream.color)

        side_pipeline = None
        side_align_to_colour = None
        if self.cfg.side.enabled:
            side_pipeline = rs.pipeline()
            side_rs_cfg = rs.config()
            side_rs_cfg.enable_device(str(self.cfg.side.serial))
            side_rs_cfg.enable_stream(
                rs.stream.color,
                self.cfg.realsense_width,
                self.cfg.realsense_height,
                rs.format.bgr8,
                self.cfg.realsense_fps,
            )
            side_rs_cfg.enable_stream(
                rs.stream.depth,
                self.cfg.realsense_width,
                self.cfg.realsense_height,
                rs.format.z16,
                self.cfg.realsense_fps,
            )
            try:
                side_pipeline.start(side_rs_cfg)
            except Exception:
                pipeline.stop()
                self._pipeline = None
                raise
            self._side_pipeline = side_pipeline
            side_align_to_colour = rs.align(rs.stream.color)

        wrist_pipeline = None
        capture = None
        if self.cfg.wrist_realsense_serial:
            wrist_pipeline = rs.pipeline()
            wrist_rs_cfg = rs.config()
            wrist_rs_cfg.enable_device(self.cfg.wrist_realsense_serial)
            wrist_rs_cfg.enable_stream(
                rs.stream.color,
                self.cfg.wrist_width,
                self.cfg.wrist_height,
                rs.format.bgr8,
                self.cfg.wrist_fps,
            )
            try:
                wrist_pipeline.start(wrist_rs_cfg)
            except Exception:
                if side_pipeline is not None:
                    side_pipeline.stop()
                    self._side_pipeline = None
                pipeline.stop()
                self._pipeline = None
                raise
            self._wrist_pipeline = wrist_pipeline
        else:
            device = self.cfg.wrist_device
            if isinstance(device, str) and device.strip().lower() == "auto":
                # Normal runtime paths resolve this together with the Gripper
                # serial endpoint. Retain a safe fallback for direct callers:
                # it only succeeds when there is exactly one PiKA UVC camera.
                from .device_discovery import resolve_wrist_camera_device

                device = resolve_wrist_camera_device(None)
                self.cfg.wrist_device = device
            if isinstance(device, str) and device.isdigit():
                device = int(device)
            capture = cv2.VideoCapture(device)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.wrist_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.wrist_height)
            capture.set(cv2.CAP_PROP_FPS, self.cfg.wrist_fps)
            if not capture.isOpened():
                if side_pipeline is not None:
                    side_pipeline.stop()
                    self._side_pipeline = None
                pipeline.stop()
                self._pipeline = None
                raise RuntimeError(f"Cannot open Pika wrist camera {device!r}")
            self._capture = capture

        def read_exterior() -> RGBDFrame:
            host_timestamp_ns = time.monotonic_ns()
            frames = align_to_colour.process(pipeline.wait_for_frames(2000))
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                raise RuntimeError("D435i aligned colour/depth frame missing")
            return RGBDFrame(
                "front",
                cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB),
                np.asanyarray(depth.get_data()).astype(np.float32) * float(depth.get_units()),
                host_timestamp_ns=host_timestamp_ns,
                device_timestamp_ms=float(color.get_timestamp()),
                frame_number=int(color.get_frame_number()),
                timestamp_domain=str(color.get_frame_timestamp_domain()),
            )

        def read_side() -> RGBDFrame:
            if side_pipeline is None or side_align_to_colour is None:
                raise RuntimeError("Side D435i is not configured")
            host_timestamp_ns = time.monotonic_ns()
            frames = side_align_to_colour.process(side_pipeline.wait_for_frames(2000))
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                raise RuntimeError("Side D435i aligned colour/depth frame missing")
            return RGBDFrame(
                "side",
                cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB),
                np.asanyarray(depth.get_data()).astype(np.float32) * float(depth.get_units()),
                host_timestamp_ns=host_timestamp_ns,
                device_timestamp_ms=float(color.get_timestamp()),
                frame_number=int(color.get_frame_number()),
                timestamp_domain=str(color.get_frame_timestamp_domain()),
            )

        def read_wrist() -> np.ndarray:
            if wrist_pipeline is not None:
                frames = wrist_pipeline.wait_for_frames(2000)
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError("Pika wrist RealSense color frame missing")
                return cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)
            assert capture is not None
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Pika wrist camera read failed")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._exterior = LatestRGBDFrame(read_exterior, "D435i", self.cfg.max_frame_age_s)
        if side_pipeline is not None:
            self._side = LatestRGBDFrame(read_side, "side D435i", self.cfg.max_frame_age_s)
        self._wrist = LatestFrame(read_wrist, "Pika wrist camera", self.cfg.max_frame_age_s)
        self._exterior.start()
        if self._side is not None:
            self._side.start()
        self._wrist.start()

    def frames(self) -> tuple[np.ndarray, np.ndarray]:
        if self._exterior is None or self._wrist is None:
            raise RuntimeError("Cameras have not been started")
        return self._exterior.get().rgb, self._wrist.get()

    def front_rgbd_frame(self) -> RGBDFrame:
        """Return one fresh D435i colour/depth pair aligned to the colour image."""
        if self._exterior is None:
            raise RuntimeError("Cameras have not been started")
        return self._exterior.get()

    def side_rgbd_frame(self) -> RGBDFrame:
        """Return one fresh side D435i colour/depth pair aligned to colour."""
        if self._side is None:
            raise RuntimeError("Side D435i is not enabled")
        return self._side.get()

    def wrist_rgb_frame(self) -> np.ndarray:
        if self._wrist is None:
            raise RuntimeError("Cameras have not been started")
        return self._wrist.get()

    def stop(self) -> None:
        if self._exterior:
            self._exterior.stop()
        if self._side:
            self._side.stop()
        if self._wrist:
            self._wrist.stop()
        if self._capture:
            self._capture.release()
        if self._pipeline:
            self._pipeline.stop()
        if self._side_pipeline:
            self._side_pipeline.stop()
        if self._wrist_pipeline:
            self._wrist_pipeline.stop()
        self._pipeline = None
        self._side_pipeline = None
        self._wrist_pipeline = None
        self._capture = None
        self._exterior = None
        self._side = None
        self._wrist = None


def resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x = (size - new_width) // 2
    y = (size - new_height) // 2
    canvas[y : y + new_height, x : x + new_width] = resized.astype(np.uint8)
    return canvas


def list_opencv_cameras(max_index: int = 10) -> list[int]:
    import cv2

    found = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                found.append(index)
        capture.release()
    return found
