#!/usr/bin/env python3
"""Reconstruct a point cloud for the custom robosuite upright-blocks scene."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from robosuite.controllers import load_controller_config
from robosuite.utils import camera_utils

from create_robosuite_upright_blocks_scene import UprightBlocksLift


DEFAULT_CAMERAS = ("frontview", "sideview", "leftsideview")
DEFAULT_OUTPUT_DIR = Path("outputs/robosuite_collision_scene")
DEFAULT_WORKSPACE_BOUNDS = (-0.48, 0.48, -0.38, 0.38, 0.74, 1.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-names", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--robot-mask-dilation", type=int, default=2)
    parser.add_argument(
        "--workspace-bounds",
        type=float,
        nargs=6,
        default=list(DEFAULT_WORKSPACE_BOUNDS),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--preview-points", type=int, default=40000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="ur5e_upright_blocks")
    return parser.parse_args()


def model_name(model, kind: str, idx: int) -> str:
    try:
        return getattr(model, f"{kind}_id2name")(idx) or ""
    except Exception:
        names = getattr(model, f"{kind}_names", None)
        if names is not None and idx < len(names):
            return names[idx] or ""
    return ""


def body_descendants(model, root_body_id: int) -> set[int]:
    parent_ids = np.asarray(model.body_parentid)
    descendants = {root_body_id}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parent_ids):
            if body_id not in descendants and int(parent_id) in descendants:
                descendants.add(body_id)
                changed = True
    return descendants


def find_robot_geoms(env: UprightBlocksLift) -> set[int]:
    model = env.sim.model
    robot_bodies: set[int] = set()
    for robot in env.robots:
        root_name = getattr(robot.robot_model, "root_body", None)
        if root_name is None:
            continue
        try:
            root_id = int(model.body_name2id(root_name))
        except Exception:
            continue
        robot_bodies.update(body_descendants(model, root_id))

    geom_ids = set()
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        name = model_name(model, "geom", geom_id).lower()
        if body_id in robot_bodies or any(key in name for key in ("robot", "ur5", "gripper", "finger", "hand")):
            geom_ids.add(geom_id)
    return geom_ids


def render_rgbd(sim, camera_name: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rgb, normalized_depth = sim.render(
        camera_name=camera_name,
        width=width,
        height=height,
        depth=True,
    )
    rgb = np.asarray(rgb[::-1], dtype=np.uint8)
    normalized_depth = np.asarray(normalized_depth[::-1], dtype=np.float32)
    depth_m = camera_utils.get_real_depth_map(sim, normalized_depth)
    return rgb, depth_m


def render_segmentation(sim, camera_name: str, width: int, height: int) -> np.ndarray:
    return np.asarray(
        sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            segmentation=True,
        )[::-1],
        dtype=np.int32,
    )


def mujoco_geom_objtype() -> int | None:
    try:
        import mujoco
    except ImportError:
        return None
    return int(mujoco.mjtObj.mjOBJ_GEOM)


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    for _ in range(max(iterations, 0)):
        padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        mask = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return mask


def robot_pixel_mask(
    sim,
    camera_name: str,
    width: int,
    height: int,
    robot_geom_ids: set[int],
    dilation: int,
) -> np.ndarray:
    segmentation = render_segmentation(sim, camera_name, width, height)
    obj_types = segmentation[..., 0]
    geom_ids = segmentation[..., 1]
    mask = np.isin(geom_ids, list(robot_geom_ids))
    geom_objtype = mujoco_geom_objtype()
    if geom_objtype is not None:
        mask &= obj_types == geom_objtype
    return dilate_mask(mask, dilation)


def depth_to_world_points(
    sim,
    camera_name: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    stride: int,
    max_depth: float,
    keep_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    z = depth_m[::stride, ::stride].reshape(-1)

    intrinsic = camera_utils.get_camera_intrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    camera_to_world = camera_utils.get_camera_extrinsic_matrix(
        sim=sim,
        camera_name=camera_name,
    )

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    u = cols.reshape(-1).astype(np.float32)
    v = rows.reshape(-1).astype(np.float32)
    camera_points = np.stack(
        [(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)],
        axis=1,
    )
    points = (camera_to_world @ camera_points.T).T[:, :3]
    colors = rgb[::stride, ::stride].reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1)
    valid &= np.isfinite(z)
    valid &= z > 0.0
    valid &= z < max_depth
    if keep_mask is not None:
        valid &= keep_mask[::stride, ::stride].reshape(-1)
    return points[valid], colors[valid]


def crop_workspace(points: np.ndarray, colors: np.ndarray, bounds: list[float]) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    keep = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )
    return points[keep], colors[keep]


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_preview_png(path: Path, points: np.ndarray, colors: np.ndarray, max_points: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors.astype(np.float32) / 255.0,
        s=0.6,
        linewidths=0,
    )
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    ax.view_init(elev=25, azim=-65)
    ranges = np.ptp(points, axis=0)
    centers = np.mean(points, axis=0)
    radius = max(float(np.max(ranges)) / 2.0, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = UprightBlocksLift(
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        camera_names=list(args.camera_names),
        camera_widths=args.width,
        camera_heights=args.height,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )

    try:
        env.reset()
        for _ in range(20):
            env.step(np.zeros(env.action_dim))

        robot_geom_ids = find_robot_geoms(env)
        print(f"[info] robot geoms excluded from point cloud: {len(robot_geom_ids)}")
        all_points = []
        all_colors = []
        for camera_name in args.camera_names:
            rgb, depth_m = render_rgbd(env.sim, camera_name, args.width, args.height)
            robot_mask = robot_pixel_mask(
                sim=env.sim,
                camera_name=camera_name,
                width=args.width,
                height=args.height,
                robot_geom_ids=robot_geom_ids,
                dilation=args.robot_mask_dilation,
            )
            points, colors = depth_to_world_points(
                sim=env.sim,
                camera_name=camera_name,
                rgb=rgb,
                depth_m=depth_m,
                stride=max(args.stride, 1),
                max_depth=args.max_depth,
                keep_mask=~robot_mask,
            )
            all_points.append(points)
            all_colors.append(colors)
            np.save(args.output_dir / f"{args.name}_{camera_name}_depth.npy", depth_m)
            np.save(args.output_dir / f"{args.name}_{camera_name}_robot_mask.npy", robot_mask)
            print(f"[info] {camera_name}: {len(points)} points, removed {int(robot_mask.sum())} robot pixels")

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        points, colors = crop_workspace(points, colors, args.workspace_bounds)
        if len(points) == 0:
            raise RuntimeError("point cloud is empty after workspace cropping")

        npz_path = args.output_dir / f"{args.name}_pointcloud.npz"
        ply_path = args.output_dir / f"{args.name}_pointcloud.ply"
        preview_path = args.output_dir / f"{args.name}_pointcloud_preview.png"
        np.savez_compressed(npz_path, points=points.astype(np.float32), colors=colors.astype(np.uint8))
        write_ascii_ply(ply_path, points, colors)
        save_preview_png(preview_path, points, colors, args.preview_points)

        print(f"[done] fused points: {len(points)}")
        print(f"[done] bounds min xyz: {points.min(axis=0)}")
        print(f"[done] bounds max xyz: {points.max(axis=0)}")
        print(f"[done] saved npz: {npz_path}")
        print(f"[done] saved ply: {ply_path}")
        print(f"[done] saved preview: {preview_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
