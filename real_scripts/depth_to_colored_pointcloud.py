#!/usr/bin/env python3
"""Project a metric depth NPY and aligned RGB image into a binary XYZRGB PLY."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth_npy", type=Path, help="H×W depth map in metres")
    parser.add_argument("rgb_image", type=Path, help="aligned RGB image")
    parser.add_argument("output_ply", type=Path)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument(
        "--comment",
        default="",
        help="stored in the PLY header, for calibration provenance",
    )
    args = parser.parse_args()

    depth = np.load(args.depth_npy).astype(np.float32, copy=False)
    rgb = np.asarray(Image.open(args.rgb_image).convert("RGB"))
    if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError(f"depth {depth.shape} and RGB {rgb.shape} are not aligned")
    rows, cols = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= args.max_depth_m)
    z = depth[valid]
    # Image rows grow downward, whereas the viewer and standard 3D camera
    # convention use +Y upward.  Negate the pixel-row projection here so that
    # a physical point near the top of the RGB image also appears at the top
    # of the interactive point-cloud view.
    xyz = np.column_stack(
        ((cols[valid] - args.cx) * z / args.fx, (args.cy - rows[valid]) * z / args.fy, z)
    ).astype(np.float32, copy=False)
    colors = rgb[valid]

    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment {args.comment}\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii", errors="replace")
    vertex_struct = struct.Struct("<fffBBB")
    with args.output_ply.open("wb") as stream:
        stream.write(header)
        for point, color in zip(xyz, colors, strict=True):
            stream.write(vertex_struct.pack(*point, *color))
    print(f"wrote {args.output_ply} with {len(xyz):,} points")


if __name__ == "__main__":
    main()
