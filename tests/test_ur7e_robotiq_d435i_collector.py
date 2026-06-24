from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_collector_module_imports_without_opencv_until_images_are_written():
    from real_scripts.ur7e_robotiq_d435i_collector.collect_rgbd_pika_robotiq import (
        _resolve_config_path,
        resolve_robot_host,
        resolve_task,
    )

    config_path = _resolve_config_path("configs/ur7e_robotiq_d435i.yaml")

    assert config_path.exists()
    assert resolve_task("pick object", {}) == "pick object"
    assert resolve_robot_host(None, {}) == "169.254.26.10"


def test_resolve_robot_host_priority(monkeypatch):
    from real_scripts.ur7e_robotiq_d435i_collector.collect_rgbd_pika_robotiq import (
        resolve_robot_host,
    )

    monkeypatch.setenv("UR_ROBOT_IP", "10.0.0.3")

    assert resolve_robot_host("10.0.0.1", {"robot": {"host": "10.0.0.2"}}) == "10.0.0.1"
    assert resolve_robot_host(None, {"robot": {"host": "10.0.0.2"}}) == "10.0.0.2"
    assert resolve_robot_host(None, {"robot": {"host": ""}}) == "10.0.0.3"


def test_pika_serial_preflight_reports_wrong_serial_package(monkeypatch):
    from real_scripts.ur7e_robotiq_d435i_collector.utils import pika_interface

    fake_serial = types.ModuleType("serial")
    fake_serial.__file__ = "fake/site-packages/serial/__init__.py"
    monkeypatch.setitem(sys.modules, "serial", fake_serial)

    with pytest.raises(RuntimeError, match="pyserial"):
        pika_interface.ensure_pyserial_available()


def test_pika_tracker_preflight_reports_missing_pysurvive(monkeypatch):
    from real_scripts.ur7e_robotiq_d435i_collector.utils import pika_interface

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match="pysurvive"):
        pika_interface.ensure_pysurvive_available()


def test_detect_pika_ports_uses_pyserial_list_ports_on_windows(monkeypatch):
    from real_scripts.ur7e_robotiq_d435i_collector.utils import pika_interface

    monkeypatch.delenv("PIKA_SENSE_PORT", raising=False)
    monkeypatch.delenv("PIKA_GRIPPER_PORT", raising=False)
    monkeypatch.setattr(pika_interface.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(pika_interface, "_list_serial_port_devices", lambda: ["COM7", "COM8"])

    assert pika_interface.detect_pika_ports() == ("COM7", "COM8")


def test_episode_writer_uses_camera_names_from_config(tmp_path: Path):
    from real_scripts.ur7e_robotiq_d435i_collector.utils.episode_writer import EpisodeWriter

    writer = EpisodeWriter(
        dataset_root=tmp_path / "dataset",
        dataset_name="dataset",
        task="task",
        fps=30,
        robot_ip="127.0.0.1",
        camera_config={
            "front": {"serial": "front"},
            "side": {"serial": "side"},
            "wrist": {"serial": "wrist"},
        },
        state_names=[f"joint_{idx}" for idx in range(6)] + ["gripper"],
        action_names=[f"joint_{idx}" for idx in range(6)] + ["gripper"],
    )

    episode_dir = writer.start_episode()

    assert (episode_dir / "rgb" / "front").is_dir()
    assert (episode_dir / "rgb" / "side").is_dir()
    assert (episode_dir / "rgb" / "wrist").is_dir()
    assert (episode_dir / "depth" / "front").is_dir()
    assert (episode_dir / "depth" / "side").is_dir()
    assert (episode_dir / "depth" / "wrist").is_dir()

    writer.end_episode(save=False)


def test_episode_writer_requires_every_configured_camera_before_image_write(tmp_path: Path):
    from real_scripts.ur7e_robotiq_d435i_collector.utils.episode_writer import EpisodeWriter

    writer = EpisodeWriter(
        dataset_root=tmp_path / "dataset",
        dataset_name="dataset",
        task="task",
        fps=30,
        robot_ip="127.0.0.1",
        camera_config={
            "front": {"serial": "front"},
            "wrist": {"serial": "wrist"},
        },
        state_names=[f"joint_{idx}" for idx in range(6)] + ["gripper"],
        action_names=[f"joint_{idx}" for idx in range(6)] + ["gripper"],
    )
    writer.start_episode()

    try:
        writer.add_frame(
            timestamp_s=0.0,
            state=np.zeros(7, dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            frames={
                "front": {
                    "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                    "depth": np.zeros((2, 2), dtype=np.uint16),
                }
            },
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected add_frame to reject missing wrist camera")

    assert "wrist" in message
    writer.end_episode(save=False)
