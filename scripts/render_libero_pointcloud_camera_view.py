#!/usr/bin/env python3
"""Project a LIBERO world-frame point cloud into a simulator camera view."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import libero_reconstruct_pointcloud as libero_pc  # noqa: E402
from libero_reconstruct_pointcloud import (  # noqa: E402
    create_env,
    load_runtime_dependencies,
    resolve_task,
    settle_scene,
)


DEFAULT_TASK = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pointcloud",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "libero_visible_robot_pointcloud"
        / f"{DEFAULT_TASK}_visible_robot_pointcloud.npz",
        help="Input .npz containing world-frame 'points' and optional 'colors'.",
    )
    parser.add_argument(
        "--task-suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO benchmark suite used to instantiate the camera.",
    )
    parser.add_argument("--task-id", type=int, default=0, help="Task index in the suite.")
    parser.add_argument("--init-state-id", type=int, default=0, help="Initial-state index.")
    parser.add_argument("--bddl-file", type=Path, default=None, help="Optional direct .bddl path.")
    parser.add_argument("--camera-name", default="frontview", help="MuJoCo camera used for projection.")
    parser.add_argument("--width", type=int, default=768, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=768, help="Output height in pixels.")
    parser.add_argument("--point-size", type=float, default=3.0, help="Projected point diameter in pixels.")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="No-op steps after reset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_camera_view_pointcloud",
        help="Directory for rendered PNG outputs.",
    )
    parser.add_argument("--name", default=None, help="Output basename. Defaults to input stem.")
    parser.add_argument(
        "--mujoco-gl",
        choices=["egl", "osmesa", "glfw"],
        default=None,
        help="MuJoCo OpenGL backend. Must be set before robosuite import.",
    )
    return parser.parse_args()


def load_colored_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "points" not in data:
        raise ValueError(f"{path} does not contain a 'points' array")
    points = np.asarray(data["points"], dtype=np.float64).reshape(-1, 3)
    if "colors" in data:
        colors = np.asarray(data["colors"], dtype=np.uint8).reshape(-1, 3)
    else:
        z = points[:, 2]
        denom = max(float(np.ptp(z)), 1e-6)
        normalized = ((z - float(z.min())) / denom * 255.0).astype(np.uint8)
        colors = np.stack([normalized, 255 - normalized, np.full_like(normalized, 180)], axis=1)
    valid = np.isfinite(points).all(axis=1)
    return points[valid], colors[valid]


def render_camera_rgb(sim, camera_name: str, width: int, height: int) -> np.ndarray:
    rgb = sim.render(camera_name=camera_name, width=width, height=height, depth=False)
    if isinstance(rgb, (tuple, list)):
        rgb = rgb[0]
    return np.asarray(rgb, dtype=np.uint8)[::-1]


def project_world_points_to_camera(
    sim,
    camera_name: str,
    width: int,
    height: int,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if libero_pc.camera_utils is None:
        raise RuntimeError("LIBERO camera utilities were not loaded.")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    intrinsic = libero_pc.camera_utils.get_camera_intrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    camera_to_world = libero_pc.camera_utils.get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    world_to_camera = np.linalg.inv(camera_to_world)
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera_points = (world_to_camera @ hom.T).T[:, :3]
    camera_depth = camera_points[:, 2]

    valid = np.isfinite(camera_points).all(axis=1) & (camera_depth > 1e-6)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = intrinsic[0, 0] * camera_points[valid, 0] / camera_depth[valid] + intrinsic[0, 2]
    uv[valid, 1] = intrinsic[1, 1] * camera_points[valid, 1] / camera_depth[valid] + intrinsic[1, 2]
    valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] < width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < height)
    return uv, camera_depth, valid


def rasterize_projected_points(
    width: int,
    height: int,
    uv: np.ndarray,
    camera_depth: np.ndarray,
    colors: np.ndarray,
    valid: np.ndarray,
    point_radius: int,
    background: np.ndarray | None = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    if background is None:
        canvas = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    else:
        canvas = Image.fromarray(np.asarray(background, dtype=np.uint8), mode="RGB")

    draw = ImageDraw.Draw(canvas)
    valid_indices = np.flatnonzero(valid)
    draw_order = valid_indices[np.argsort(camera_depth[valid_indices])[::-1]]
    radius = int(max(point_radius, 0))
    for idx in draw_order:
        x, y = uv[idx]
        color = tuple(int(c) for c in colors[idx])
        if radius <= 0:
            draw.point((float(x), float(y)), fill=color)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return np.asarray(canvas, dtype=np.uint8)


def render_pointcloud_camera_view(args: argparse.Namespace) -> tuple[Path, Path, int]:
    from PIL import Image

    points, colors = load_colored_points(args.pointcloud)
    task_args = SimpleNamespace(
        task_suite=args.task_suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        bddl_file=args.bddl_file,
    )
    bddl_file, _, init_state = resolve_task(task_args)
    env = create_env(bddl_file, args.width, args.height, [args.camera_name])
    try:
        settle_scene(env, init_state, args.num_steps_wait)
        uv, camera_depth, valid = project_world_points_to_camera(
            env.sim,
            args.camera_name,
            args.width,
            args.height,
            points,
        )
        radius = max(1, int(round(args.point_size / 2.0)))
        rgb = render_camera_rgb(env.sim, args.camera_name, args.width, args.height)
        point_image = rasterize_projected_points(
            args.width,
            args.height,
            uv,
            camera_depth,
            colors,
            valid,
            point_radius=radius,
        )
        overlay_image = rasterize_projected_points(
            args.width,
            args.height,
            uv,
            camera_depth,
            colors,
            valid,
            point_radius=radius,
            background=rgb,
        )
    finally:
        env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.pointcloud.stem
    point_path = args.output_dir / f"{name}_{args.camera_name}_projected_points.png"
    overlay_path = args.output_dir / f"{name}_{args.camera_name}_projected_overlay.png"
    Image.fromarray(point_image).save(point_path)
    Image.fromarray(overlay_image).save(overlay_path)
    return point_path, overlay_path, int(np.count_nonzero(valid))


def main() -> None:
    args = parse_args()
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    load_runtime_dependencies()
    point_path, overlay_path, valid_count = render_pointcloud_camera_view(args)
    print(f"[done] projected points in view: {valid_count}")
    print(f"[done] saved projected points: {point_path}")
    print(f"[done] saved projected overlay: {overlay_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
