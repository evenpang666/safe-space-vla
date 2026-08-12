import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.validate_ur7e_pika_mask_config import validate_mask_config


def _config(tmp_path: Path) -> Path:
    for name in ("full.stl", "body.stl", "finger_a.stl", "finger_b.stl", "calibration.json"):
        (tmp_path / name).write_bytes(b"mesh")
    payload = {
        "schema_version": 1,
        "pika": {
            "full_collision_mesh": "full.stl",
            "body_collision_mesh": "body.stl",
            "finger_mesh_candidates": ["finger_a.stl", "finger_b.stl"],
            "mesh_unit": "mm",
            "flange_to_pika_step_frame": np.eye(4).tolist(),
            "opening_state": {"unit": "mm", "max_opening_m": 0.095},
        },
        "camera_to_base_calibration": "calibration.json",
    }
    path = tmp_path / "mask.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validate_mask_config_requires_real_geometry_and_calibration(tmp_path):
    config_path = _config(tmp_path)
    checked = validate_mask_config(config_path, require_calibration=True)
    assert len(checked) == 5

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pika"]["flange_to_pika_step_frame"] = None
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="not measured"):
        validate_mask_config(config_path, require_calibration=True)
