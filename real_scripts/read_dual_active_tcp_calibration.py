#!/usr/bin/env python3
"""Read (never command) both UR7e controllers and measure flange→active TCP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.live_dual_ur7e_obstacle_model import rotvec_transform  # noqa: E402
from real_scripts.ur7e_collision_mesh import flange_transform  # noqa: E402
from safety_module.project_config import DEFAULT_PROJECT_CONFIG, load_project_config, resolve_repo_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "calibration" / "dual_flange_to_active_tcp.json")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--interval-s", type=float, default=0.05)
    parser.add_argument("--max-translation-spread-mm", type=float, default=0.5)
    parser.add_argument("--max-rotation-spread-deg", type=float, default=0.2)
    return parser.parse_args()


def _mean_transform(transforms: np.ndarray) -> tuple[np.ndarray, float, float]:
    translations = transforms[:, :3, 3]
    center = np.median(translations, axis=0)
    rotation = Rotation.from_matrix(transforms[:, :3, :3]).mean()
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation.as_matrix(), center
    translation_spread = float(np.max(np.linalg.norm(translations - center, axis=1)) * 1000.0)
    rotation_spread = float(np.max((rotation.inv() * Rotation.from_matrix(transforms[:, :3, :3])).magnitude()) * 180.0 / np.pi)
    return result, translation_spread, rotation_spread


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.interval_s < 0:
        raise ValueError("samples must be >=2 and interval-s must be non-negative")
    config, _ = load_project_config(args.project_config)
    with resolve_repo_path(config["quest3"]["hardware_config"]).open(encoding="utf-8") as handle:
        hardware = yaml.safe_load(handle)
    from rtde_receive import RTDEReceiveInterface

    receivers = {
        name: RTDEReceiveInterface(str(hardware["arms"][name]["robot_ip"]))
        for name in ("left_arm", "right_arm")
    }
    samples = {name: [] for name in receivers}
    try:
        for _ in range(args.samples):
            for name, receiver in receivers.items():
                qpos = np.asarray(receiver.getActualQ(), dtype=np.float64)
                tcp = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
                if qpos.shape != (6,) or tcp.shape != (6,) or not np.isfinite(qpos).all() or not np.isfinite(tcp).all():
                    raise RuntimeError(f"Invalid receive-only RTDE sample from {name}")
                samples[name].append(np.linalg.inv(flange_transform(qpos)) @ rotvec_transform(tcp))
            time.sleep(args.interval_s)
    finally:
        for receiver in receivers.values():
            receiver.disconnect()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "method": "receive_only_rtde_inv_base_T_flange_times_base_T_active_tcp", "arms": {}}
    all_verified = True
    for name, values in samples.items():
        matrix, translation_spread, rotation_spread = _mean_transform(np.stack(values))
        verified = translation_spread <= args.max_translation_spread_mm and rotation_spread <= args.max_rotation_spread_deg
        all_verified &= verified
        result["arms"][name] = {
            "matrix": matrix.tolist(), "sample_count": len(values),
            "translation_max_spread_mm": translation_spread, "rotation_max_spread_deg": rotation_spread,
            "verified": verified,
        }
    result["verified"] = all_verified
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "output": str(output)}, ensure_ascii=False, indent=2))
    if not all_verified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
