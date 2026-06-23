from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.ur7e_realsense_adapter import (
    D435iCameraConfig,
    DEFAULT_D435I_CAMERA_NAMES,
    RealSenseD435iSource,
    UR7eRealSenseAdapter,
    create_adapter,
)


class FakeController:
    def __init__(self):
        self.connected = False
        self.closed = False
        self.sent_actions = []

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True

    def get_current_joints(self):
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def get_gripper_open_ratio(self):
        return 0.25

    def send_ee_delta_vector(self, action, acceleration, velocity, wait_after_arm_s):
        self.sent_actions.append((list(action), acceleration, velocity, wait_after_arm_s))
        return [0.0] * 7, list(action)


class FakeCameraSource:
    def __init__(self, names=DEFAULT_D435I_CAMERA_NAMES):
        self.names = tuple(names)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self):
        return {
            name: (
                np.full((2, 2, 3), idx, dtype=np.uint8),
                np.full((2, 2), idx + 1.0, dtype=np.float32),
            )
            for idx, name in enumerate(self.names)
        }


class IncrementingCameraSource(FakeCameraSource):
    def __init__(self, names=DEFAULT_D435I_CAMERA_NAMES):
        super().__init__(names)
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return {
            name: (
                np.full((2, 2, 3), self.read_count + idx, dtype=np.uint8),
                np.full((2, 2), self.read_count + idx, dtype=np.float32),
            )
            for idx, name in enumerate(self.names)
        }


class FakeFrame:
    def __init__(self, data, units=1.0):
        self.data = data
        self.units = units

    def get_data(self):
        return self.data

    def get_units(self):
        return self.units


class FakeAlignedFrames:
    def __init__(self):
        self.rgb = np.full((2, 2, 3), 7, dtype=np.uint8)
        self.depth = np.full((2, 2), 2, dtype=np.uint16)

    def get_color_frame(self):
        return FakeFrame(self.rgb)

    def get_depth_frame(self):
        return FakeFrame(self.depth, units=0.001)


class FlakyPipeline:
    def __init__(self):
        self.calls = 0

    def wait_for_frames(self, timeout_ms):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Frame didn't arrive within 5000")
        return object()


class FakeAlign:
    def process(self, frames):
        return FakeAlignedFrames()


def test_ur7e_adapter_sends_pi05_action_as_ee_delta_vector():
    controller = FakeController()
    camera_source = FakeCameraSource()
    adapter = UR7eRealSenseAdapter(
        controller=controller,
        camera_source=camera_source,
        acceleration=0.11,
        velocity=0.022,
        wait_after_arm_s=0.03,
    )
    action = np.asarray([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    adapter.reset()
    adapter.execute_action(action)
    adapter.close()

    assert controller.connected is True
    assert camera_source.started is True
    assert controller.closed is True
    assert camera_source.stopped is True
    assert len(controller.sent_actions) == 1
    sent_action, acceleration, velocity, wait_after = controller.sent_actions[0]
    np.testing.assert_allclose(sent_action, action, rtol=0, atol=1e-7)
    assert acceleration == 0.11
    assert velocity == 0.022
    assert wait_after == 0.03


def test_ur7e_adapter_observation_and_rgbd_frames_default_to_three_d435i_cameras():
    adapter = UR7eRealSenseAdapter(controller=FakeController(), camera_source=FakeCameraSource())

    observation = adapter.get_observation()
    frames = adapter.get_rgbd_frames()

    assert tuple(DEFAULT_D435I_CAMERA_NAMES) == ("front", "side", "wrist")
    np.testing.assert_allclose(observation["qpos"], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(observation["gripper"], [0.25])
    assert set(observation) >= {"front_rgb", "side_rgb", "wrist_rgb"}
    assert [frame.camera_name for frame in frames] == ["front", "side", "wrist"]
    np.testing.assert_allclose([frame.depth_m[0, 0] for frame in frames], [1.0, 2.0, 3.0])


def test_ur7e_adapter_reuses_observation_rgbd_frames_for_same_control_step():
    camera_source = IncrementingCameraSource()
    adapter = UR7eRealSenseAdapter(controller=FakeController(), camera_source=camera_source)

    observation = adapter.get_observation()
    frames = adapter.get_rgbd_frames()

    assert camera_source.read_count == 1
    np.testing.assert_array_equal(observation["front_rgb"], frames[0].rgb)
    np.testing.assert_allclose(frames[0].depth_m, np.full((2, 2), 1.0, dtype=np.float32))


def test_ur7e_adapter_can_use_front_and_wrist_cameras_only():
    adapter = UR7eRealSenseAdapter(
        controller=FakeController(),
        camera_source=FakeCameraSource(names=("front", "wrist")),
        camera_names=("front", "wrist"),
    )

    observation = adapter.get_observation()
    frames = adapter.get_rgbd_frames()

    assert set(observation) >= {"front_rgb", "wrist_rgb"}
    assert "side_rgb" not in observation
    assert [frame.camera_name for frame in frames] == ["front", "wrist"]


def test_realsense_source_retries_transient_wait_for_frames_timeout():
    pipeline = FlakyPipeline()
    source = RealSenseD435iSource(
        cameras=[D435iCameraConfig("front", "front_serial")],
        wait_timeout_ms=123,
        read_retries=2,
    )
    source._rs = object()
    source._pipelines = [("front", pipeline)]
    source._aligns = [FakeAlign()]

    frames = source.read()

    assert pipeline.calls == 2
    rgb, depth_m = frames["front"]
    np.testing.assert_array_equal(rgb, np.full((2, 2, 3), 7, dtype=np.uint8))
    np.testing.assert_allclose(depth_m, np.full((2, 2), 0.002, dtype=np.float32))


def test_create_adapter_requires_serials_for_multiple_realsense_cameras(monkeypatch):
    monkeypatch.delenv("REAL_SENSE_FRONT_SERIAL", raising=False)
    monkeypatch.delenv("REAL_SENSE_WRIST_SERIAL", raising=False)

    try:
        create_adapter(camera_names=("front", "wrist"))
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing serials to fail before starting RealSense pipelines")

    assert "REAL_SENSE_FRONT_SERIAL" in message
    assert "REAL_SENSE_WRIST_SERIAL" in message
