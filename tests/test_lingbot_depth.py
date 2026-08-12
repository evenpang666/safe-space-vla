import argparse
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.lingbot_depth import add_lingbot_depth_cli_args
from real_scripts.lingbot_depth import LingBotDepthRefiner
from real_scripts.lingbot_depth import normalized_intrinsics
from real_scripts.real_robot_adapter import CameraCalibration
from real_scripts.real_robot_adapter import RGBDFrame


def test_normalized_intrinsics_uses_image_width_and_height_independently():
    intrinsics = np.asarray([[600.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    actual = normalized_intrinsics(intrinsics, width=640, height=480)

    np.testing.assert_allclose(actual, [[600 / 640, 0.0, 320 / 640], [0.0, 500 / 480, 240 / 480], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(intrinsics, [[600.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("width,height", [(0, 480), (640, 0)])
def test_normalized_intrinsics_rejects_non_positive_image_dimensions(width, height):
    with pytest.raises(ValueError, match="positive"):
        normalized_intrinsics(np.eye(3), width=width, height=height)


def test_lingbot_cli_supports_disabling_fp16_for_cpu_inference():
    parser = argparse.ArgumentParser()
    add_lingbot_depth_cli_args(parser)

    args = parser.parse_args(["--lingbot-depth", "--no-lingbot-fp16", "--lingbot-device", "cpu"])

    assert args.lingbot_depth is True
    assert args.lingbot_fp16 is False
    assert args.lingbot_device == "cpu"
    assert args.lingbot_camera_names == ("front", "side")


def test_lingbot_refiner_uses_only_front_and_side_in_configured_order():
    refiner = object.__new__(LingBotDepthRefiner)
    refiner.camera_names = ("front", "side")
    called = []

    def fake_refine_one(frame, calibration):
        called.append((frame.camera_name, calibration.name))
        return frame

    refiner._refine_one = fake_refine_one
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    frames = [
        RGBDFrame("wrist", rgb, np.ones((1, 1), dtype=np.float32)),
        RGBDFrame("side", rgb, np.ones((1, 1), dtype=np.float32)),
        RGBDFrame("front", rgb, np.ones((1, 1), dtype=np.float32)),
    ]
    calibrations = {
        name: CameraCalibration(name, np.eye(3), np.eye(4)) for name in ("front", "side", "wrist")
    }

    refined = refiner.refine(frames, calibrations)

    assert [frame.camera_name for frame in refined] == ["front", "side"]
    assert called == [("front", "front"), ("side", "side")]
