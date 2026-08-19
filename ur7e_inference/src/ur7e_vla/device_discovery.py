"""Safe, topology-aware discovery for PiKA serial devices and UVC cameras."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import logging
import os
from pathlib import Path
import re
import time
from typing import Iterable, Literal, Optional

from .config import AppConfig

LOG = logging.getLogger(__name__)

PikaRole = Literal["sense", "gripper"]
_USB_DEVICE_NAME = re.compile(r"\d+-\d+(?:\.\d+)*$")


@dataclass(frozen=True)
class PikaSerialDevices:
    sense: Optional[str]
    gripper: Optional[str]
    unknown: tuple[str, ...]
    inaccessible: tuple[str, ...]


@dataclass(frozen=True)
class UvcCamera:
    device: str
    name: str
    usb_path: Optional[str]


def _is_auto(value: object) -> bool:
    return value is None or not str(value).strip() or str(value).strip().lower() == "auto"


def _existing_path(value: object) -> Optional[str]:
    if _is_auto(value):
        return None
    path = str(value).strip()
    return path if Path(path).exists() else None


def _usb_path(sysfs_path: Path) -> Optional[str]:
    """Return a Linux USB physical port chain, e.g. ``1-2.1``."""
    try:
        resolved = sysfs_path.resolve()
    except OSError:
        return None
    for item in (resolved, *resolved.parents):
        if _USB_DEVICE_NAME.fullmatch(item.name):
            return item.name
    return None


def _serial_usb_path(port: str) -> Optional[str]:
    name = Path(port).name
    node = Path("/sys/class/tty") / name / "device"
    return _usb_path(node) if node.exists() else None


def _serial_candidates(preferred: Iterable[object]) -> list[str]:
    ports: set[str] = set()
    for item in preferred:
        path = _existing_path(item)
        if path:
            ports.add(path)
    try:
        from serial.tools import list_ports

        ports.update(
            port.device
            for port in list_ports.comports()
            if port.device.startswith(("/dev/ttyUSB", "/dev/ttyACM")) and Path(port.device).exists()
        )
    except ImportError:
        LOG.warning("pyserial is unavailable; falling back to /dev/ttyUSB* and /dev/ttyACM* discovery")
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/serial/by-id/*"):
        ports.update(path for path in glob.glob(pattern) if Path(path).exists())
    # /dev/serial/by-id entries are aliases of ttyUSB/ttyACM nodes; canonicalize
    # them so one PiKA device is never probed or reported twice.
    return sorted({str(Path(port).resolve()) for port in ports})


def _classify_payload(payload: dict) -> Optional[PikaRole]:
    if "AS5047" in payload:
        return "sense"
    if "motor" in payload or "motorstatus" in payload:
        return "gripper"
    return None


def _import_serial_comm():
    # Import through the existing PiKA loader so this works with the vendored
    # SDK and does not require the deprecated agx-pypika GUI package.
    from .hardware import _import_pika_gripper

    _import_pika_gripper()
    from pika.serial_comm import SerialComm

    return SerialComm


def discover_pika_serial_devices(
    preferred: Iterable[object] = (), *, probe_timeout_s: float = 1.5
) -> PikaSerialDevices:
    """Classify PiKA serial ports from passive telemetry only.

    No motor command is sent.  Sense firmware periodically publishes an
    ``AS5047`` object; Gripper firmware publishes ``motor``/``motorstatus``.
    """
    candidates = _serial_candidates(preferred)
    if not candidates:
        raise RuntimeError(
            "No Linux serial devices were found for PiKA. Expected /dev/ttyUSB* or /dev/ttyACM*. "
            "The configured Windows-style COM port is not usable on this host."
        )
    SerialComm = _import_serial_comm()
    found: dict[PikaRole, list[str]] = {"sense": [], "gripper": []}
    unknown: list[str] = []
    inaccessible: list[str] = []
    for port in candidates:
        if not Path(port).exists() or not os.access(port, os.R_OK | os.W_OK):
            inaccessible.append(port)
            continue
        comm = SerialComm(port=port, timeout=min(0.25, probe_timeout_s))
        role: Optional[PikaRole] = None
        try:
            if not comm.connect():
                unknown.append(port)
                continue
            comm.start_reading_thread()
            deadline = time.monotonic() + probe_timeout_s
            while time.monotonic() < deadline:
                role = _classify_payload(comm.get_latest_data())
                if role is not None:
                    break
                time.sleep(0.03)
        except Exception as exc:
            LOG.debug("PiKA probe failed on %s: %s", port, exc)
        finally:
            comm.disconnect()
        if role is None:
            unknown.append(port)
        else:
            found[role].append(port)

    for role, ports in found.items():
        if len(ports) > 1:
            raise RuntimeError(f"Found multiple PiKA {role} serial devices: {ports}; disconnect the extra device.")
    result = PikaSerialDevices(
        sense=found["sense"][0] if found["sense"] else None,
        gripper=found["gripper"][0] if found["gripper"] else None,
        unknown=tuple(unknown),
        inaccessible=tuple(inaccessible),
    )
    LOG.info(
        "PiKA serial discovery: Sense=%s (USB %s), Gripper=%s (USB %s), unclassified=%s, inaccessible=%s",
        result.sense,
        _serial_usb_path(result.sense) if result.sense else None,
        result.gripper,
        _serial_usb_path(result.gripper) if result.gripper else None,
        result.unknown or "none",
        result.inaccessible or "none",
    )
    return result


def discover_pika_uvc_cameras() -> list[UvcCamera]:
    """Return capture nodes exposed by PiKA's DECXIN fisheye camera modules."""
    cameras: list[UvcCamera] = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda value: int(value.name[5:])):
        try:
            name = (node / "name").read_text(encoding="utf-8").strip()
            index = (node / "index").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        # ``index=0`` is the video-capture endpoint; the paired metadata node
        # has index=1 and must never be passed to OpenCV.
        if index != "0" or "decxin" not in name.lower():
            continue
        cameras.append(UvcCamera(device=f"/dev/{node.name}", name=name, usb_path=_usb_path(node / "device")))
    LOG.info("PiKA UVC capture candidates: %s", cameras or "none")
    return cameras


def _same_usb_device(left: Optional[str], right: Optional[str]) -> bool:
    return left is not None and right is not None and left == right


def _shared_usb_hub_depth(left: Optional[str], right: Optional[str]) -> int:
    """Depth of the common physical USB hub path, excluding the root bus."""
    if left is None or right is None:
        return 0
    try:
        left_bus, left_chain = left.split("-", maxsplit=1)
        right_bus, right_chain = right.split("-", maxsplit=1)
    except ValueError:
        return 0
    if left_bus != right_bus:
        return 0
    common = 0
    for left_part, right_part in zip(left_chain.split("."), right_chain.split(".")):
        if left_part != right_part:
            break
        common += 1
    return common


def resolve_wrist_camera_device(gripper_port: Optional[str]) -> str:
    """Find the Gripper fisheye capture node using the shared USB topology."""
    cameras = discover_pika_uvc_cameras()
    if not cameras:
        raise RuntimeError("No PiKA UVC wrist camera was found (expected a DECXIN capture device).")
    gripper_usb_path = _serial_usb_path(gripper_port) if gripper_port else None
    matches = [camera for camera in cameras if _same_usb_device(camera.usb_path, gripper_usb_path)]
    if len(matches) == 1:
        LOG.info("Selected PiKA Gripper wrist camera %s (USB %s)", matches[0].device, matches[0].usb_path)
        return matches[0].device
    if len(cameras) == 1:
        LOG.info("Selected the only PiKA wrist camera candidate %s", cameras[0].device)
        return cameras[0].device
    depths = [(_shared_usb_hub_depth(camera.usb_path, gripper_usb_path), camera) for camera in cameras]
    best_depth = max(depth for depth, _ in depths)
    best = [camera for depth, camera in depths if depth == best_depth]
    if best_depth > 0 and len(best) == 1:
        LOG.info(
            "Selected PiKA Gripper wrist camera %s from shared USB hub depth %d (serial=%s, camera=%s)",
            best[0].device,
            best_depth,
            gripper_usb_path,
            best[0].usb_path,
        )
        return best[0].device
    rendered = [f"{camera.device} (USB {camera.usb_path})" for camera in cameras]
    raise RuntimeError(
        "Cannot safely distinguish the PiKA Gripper wrist camera from the PiKA Sense camera. "
        f"Gripper serial USB path: {gripper_usb_path}; camera candidates: {rendered}. "
        "Connect the Gripper serial and camera through the same USB device/hub path, or set cameras.wrist_device "
        "to the correct persistent /dev/v4l/by-path/... video-capture path."
    )


def autodetect_pika_hardware(
    cfg: AppConfig,
    *,
    require_sense: bool,
    require_gripper: bool,
    require_wrist_camera: bool,
) -> None:
    """Resolve configured ``auto``/stale PiKA endpoints before hardware connects."""
    wants_serial = require_sense or (require_gripper and cfg.gripper.enabled)
    devices: Optional[PikaSerialDevices] = None
    if wants_serial:
        devices = discover_pika_serial_devices((cfg.demo.pika_sense_port, cfg.gripper.serial_port))
        if require_sense:
            if devices.sense is None:
                if devices.inaccessible:
                    raise RuntimeError(
                        "Cannot read PiKA serial port(s) "
                        f"{list(devices.inaccessible)}. Add the current user to the dialout group and start a new login session."
                    )
                raise RuntimeError(
                    "PiKA Sense was not identified from serial telemetry. "
                    f"Unclassified ports: {list(devices.unknown) or 'none'}."
                )
            cfg.demo.pika_sense_port = devices.sense
        if require_gripper and cfg.gripper.enabled:
            if devices.gripper is None:
                if devices.inaccessible:
                    raise RuntimeError(
                        "Cannot read PiKA serial port(s) "
                        f"{list(devices.inaccessible)}. Add the current user to the dialout group and start a new login session."
                    )
                raise RuntimeError(
                    "PiKA Gripper was not identified from serial telemetry. "
                    f"Unclassified ports: {list(devices.unknown) or 'none'}."
                )
            cfg.gripper.serial_port = devices.gripper
    if require_wrist_camera and not cfg.cameras.wrist_realsense_serial and _is_auto(cfg.cameras.wrist_device):
        cfg.cameras.wrist_device = resolve_wrist_camera_device(devices.gripper if devices else None)
