#!/usr/bin/env python3
"""Validate a UR7e+PiKA mesh-mask configuration before live collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _require_file(value: str, *, base_dir: Path, label: str) -> Path:
    if not value or "REPLACE_" in value:
        raise ValueError(f"{label} is not configured")
    path = Path(value)
    path = path if path.is_absolute() else base_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def validate_mask_config(path: Path, *, require_calibration: bool) -> list[str]:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported or missing schema_version")
    base_dir = path.parent
    pika = config.get("pika", {})
    checked = []
    for key in ("full_collision_mesh", "body_collision_mesh"):
        checked.append(str(_require_file(str(pika.get(key, "")), base_dir=base_dir, label=f"pika.{key}")))
    fingers = pika.get("finger_mesh_candidates", [])
    if len(fingers) != 2:
        raise ValueError("pika.finger_mesh_candidates must contain two mesh paths")
    checked.extend(str(_require_file(str(value), base_dir=base_dir, label="pika.finger_mesh_candidates")) for value in fingers)
    if pika.get("mesh_unit") != "mm":
        raise ValueError("PiKA STEP collision meshes must declare mesh_unit='mm'")
    opening = pika.get("opening_state", {})
    if opening.get("unit") not in {"m", "mm", "ratio"}:
        raise ValueError("pika.opening_state.unit must be m, mm, or ratio")
    if float(opening.get("max_opening_m", 0.0)) != 0.095:
        raise ValueError("PiKA max opening must be 0.095 m unless the installed hardware is verified otherwise")
    transform = pika.get("flange_to_pika_step_frame")
    if transform is None:
        if require_calibration:
            raise ValueError("pika.flange_to_pika_step_frame is not measured")
    else:
        transform_array = np.asarray(transform, dtype=np.float64)
        if transform_array.shape != (4, 4) or not np.isfinite(transform_array).all():
            raise ValueError("pika.flange_to_pika_step_frame must be a finite 4x4 transform")
        if not np.allclose(transform_array[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("pika.flange_to_pika_step_frame has an invalid homogeneous last row")
    calibration = str(config.get("camera_to_base_calibration", ""))
    if require_calibration:
        checked.append(str(_require_file(calibration, base_dir=base_dir, label="camera_to_base_calibration")))
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--ready-for-live-mask",
        action="store_true",
        help="also require measured flange-to-PiKA and camera-to-base calibration",
    )
    args = parser.parse_args()
    checked = validate_mask_config(args.config, require_calibration=args.ready_for_live_mask)
    print("configuration is valid")
    for item in checked:
        print(f"- {item}")


if __name__ == "__main__":
    main()
