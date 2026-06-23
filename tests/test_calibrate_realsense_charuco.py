import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.calibrate_realsense_charuco import (
    calibration_payload,
    invert_transform,
    save_calibration_json,
)


def test_invert_transform_inverts_rigid_matrix():
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = [1.0, 2.0, 3.0]

    inverse = invert_transform(transform)

    np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-8)


def test_save_calibration_json_writes_demo_compatible_camera_payload(tmp_path: Path):
    intrinsics = np.asarray([[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 3] = [0.1, 0.2, 0.3]

    payload = calibration_payload("front", intrinsics, camera_to_world, model="intel_realsense_d435i")
    path = tmp_path / "calibration.json"
    save_calibration_json(path, payload)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["cameras"]) == {"front"}
    np.testing.assert_allclose(data["cameras"]["front"]["intrinsics"], intrinsics)
    np.testing.assert_allclose(data["cameras"]["front"]["camera_to_world"], camera_to_world)
