#!/usr/bin/env python3
"""Capture front/side D435i colour frames and SDK intrinsics for calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.ur7e_realsense_adapter import D435iCameraConfig
from real_scripts.ur7e_realsense_adapter import RealSenseD435iSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames discarded while auto-exposure settles.")
    return parser.parse_args()


def capture(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = RealSenseD435iSource(
        cameras=(
            D435iCameraConfig("front", os.environ.get("REAL_SENSE_FRONT_SERIAL") or None),
            D435iCameraConfig("side", os.environ.get("REAL_SENSE_SIDE_SERIAL") or None),
        ),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    source.start()
    try:
        if args.warmup_frames < 0:
            raise ValueError("--warmup-frames must be >= 0")
        frames = None
        for _ in range(max(1, int(args.warmup_frames))):
            frames = source.read()
        assert frames is not None
        calibrations = source.get_color_calibrations()
    finally:
        source.stop()

    for name, (rgb, _) in frames.items():
        Image.fromarray(rgb, mode="RGB").save(output_dir / f"{name}_rgb.png")
    payload = {
        "camera_frame": "color",
        "depth_alignment": "depth_to_color",
        "cameras": calibrations,
    }
    intrinsics_path = output_dir / "d435i_color_intrinsics.json"
    intrinsics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return intrinsics_path


def main() -> None:
    path = capture(parse_args())
    print(f"[done] wrote calibration RGB frames and intrinsics to {path.parent}")


if __name__ == "__main__":
    main()
