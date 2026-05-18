#!/usr/bin/env python3
"""Reconstruct and visualize a LIBERO scene point cloud from simulator RGB-D.

The script creates a LIBERO offscreen environment, renders depth from one or
more MuJoCo cameras, back-projects the depth maps into world coordinates, and
saves a fused colored point cloud.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import types
from typing import Iterable

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT_CANDIDATES = (
    REPO_ROOT / "third_party" / "LIBERO",
    REPO_ROOT / "thiry_party" / "LIBERO",
)
for libero_root in LIBERO_ROOT_CANDIDATES:
    if libero_root.exists():
        sys.path.insert(0, str(libero_root))
        break


def install_robosuite_compat() -> None:
    """Install small shims for LIBERO versions written for robosuite <= 1.4.

    The local environment may have robosuite >= 1.5, where SingleArmEnv was
    folded into ManipulationEnv and the constructor argument mount_types became
    base_types. This shim keeps the LIBERO import path working without editing
    third_party code.
    """

    import robosuite

    if not hasattr(robosuite, "load_controller_config"):
        def load_controller_config(default_controller=None, custom_fpath=None):
            if custom_fpath is not None:
                return robosuite.load_part_controller_config(custom_fpath=custom_fpath)
            part_config = robosuite.load_part_controller_config(default_controller=default_controller)
            composite_config = robosuite.load_composite_controller_config(controller="BASIC", robot="Panda")
            composite_config["body_parts"]["right"] = dict(part_config)
            composite_config["body_parts"]["right"]["gripper"] = {"type": "GRIP"}
            composite_config["body_parts"]["left"] = dict(part_config)
            composite_config["body_parts"]["left"]["gripper"] = {"type": "GRIP"}
            return composite_config

        robosuite.load_controller_config = load_controller_config

    try:
        from robosuite.environments.manipulation.single_arm_env import SingleArmEnv  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

    class SingleArmEnv(ManipulationEnv):
        def __init__(self, *args, mount_types="default", **kwargs):
            kwargs.setdefault("base_types", mount_types)
            super().__init__(*args, **kwargs)

    shim = types.ModuleType("robosuite.environments.manipulation.single_arm_env")
    shim.SingleArmEnv = SingleArmEnv
    sys.modules["robosuite.environments.manipulation.single_arm_env"] = shim

    try:
        from robosuite.robots.single_arm import SingleArm  # noqa: F401
    except ModuleNotFoundError:
        from robosuite.robots.fixed_base_robot import FixedBaseRobot

        robot_shim = types.ModuleType("robosuite.robots.single_arm")
        robot_shim.SingleArm = FixedBaseRobot
        sys.modules["robosuite.robots.single_arm"] = robot_shim

benchmark = None
get_libero_path = None
OffScreenRenderEnv = None
camera_utils = None


def load_runtime_dependencies() -> None:
    global benchmark, get_libero_path, OffScreenRenderEnv, camera_utils
    if OffScreenRenderEnv is not None:
        return

    try:
        patch_torch_load_for_libero()
        install_robosuite_compat()
        from libero.libero import benchmark as libero_benchmark
        from libero.libero import get_libero_path as libero_get_libero_path
        from libero.libero.envs import OffScreenRenderEnv as LiberoOffScreenRenderEnv
        from robosuite.utils import camera_utils as robosuite_camera_utils
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            "Missing runtime dependency "
            f"{missing!r}. Install LIBERO requirements first, for example: "
            "pip install -r thiry_party/LIBERO/requirements.txt"
        ) from exc

    benchmark = libero_benchmark
    get_libero_path = libero_get_libero_path
    OffScreenRenderEnv = LiberoOffScreenRenderEnv
    camera_utils = robosuite_camera_utils
    patch_libero_robot_models()


def patch_torch_load_for_libero() -> None:
    """Keep LIBERO init-state loading compatible with PyTorch >= 2.6."""

    try:
        import torch
    except ModuleNotFoundError:
        return

    if getattr(torch.load, "_libero_compat_patched", False):
        return

    original_load = torch.load

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    compat_load._libero_compat_patched = True
    torch.load = compat_load


def patch_libero_robot_models() -> None:
    """Patch LIBERO custom robot models for robosuite >= 1.5 metadata."""

    try:
        from libero.libero.envs.robots.mounted_panda import MountedPanda
        from libero.libero.envs.robots.mounted_ur5e import MountedUR5e
        from libero.libero.envs.robots.on_the_ground_panda import OnTheGroundPanda
        from libero.libero.envs.robots.on_the_ground_ur5e import OnTheGroundUR5E
    except ImportError:
        try:
            from libero.libero.envs.robots.on_the_ground_ur5e import OnTheGroundUR5e
            from libero.libero.envs.robots.mounted_ur5e import MountedUR5e
            from libero.libero.envs.robots.mounted_panda import MountedPanda
            from libero.libero.envs.robots.on_the_ground_panda import OnTheGroundPanda
        except ImportError:
            return
    else:
        OnTheGroundUR5e = OnTheGroundUR5E

    robot_specs = (
        (MountedPanda, "PandaGripper", "default_panda"),
        (OnTheGroundPanda, "PandaGripper", "default_panda"),
        (MountedUR5e, "Robotiq85Gripper", "default_ur5e"),
        (OnTheGroundUR5e, "Robotiq85Gripper", "default_ur5e"),
    )
    for cls, gripper, controller in robot_specs:
        cls.arms = ["right"]
        cls.default_base = property(lambda self: self.default_mount)
        cls.default_gripper = property(lambda self, name=gripper: {"right": name})
        cls.default_controller_config = property(lambda self, name=controller: {"right": name})


DEFAULT_CAMERAS = ("agentview", "frontview", "birdview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a colored 3D point cloud from a LIBERO scene."
    )
    parser.add_argument(
        "--task-suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO benchmark suite to instantiate.",
    )
    parser.add_argument("--task-id", type=int, default=0, help="Task index in the suite.")
    parser.add_argument(
        "--init-state-id",
        type=int,
        default=0,
        help="Initial-state index from the LIBERO task suite.",
    )
    parser.add_argument(
        "--bddl-file",
        type=Path,
        default=None,
        help="Optional direct path to a .bddl file. Overrides --task-suite/--task-id.",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="MuJoCo camera names to fuse.",
    )
    parser.add_argument("--width", type=int, default=256, help="Render width.")
    parser.add_argument("--height", type=int, default=256, help="Render height.")
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=10,
        help="No-op simulator steps after reset so objects settle.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Pixel stride for downsampling before point cloud generation.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=4.0,
        help="Discard points farther than this metric depth from the camera.",
    )
    parser.add_argument(
        "--workspace-bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Optional world-coordinate crop box.",
    )
    parser.add_argument(
        "--include-robot",
        action="store_true",
        help="Keep robot points in the reconstructed cloud. By default robot pixels are removed.",
    )
    parser.add_argument(
        "--robot-mask-dilation",
        type=int,
        default=1,
        help="Dilate the robot segmentation mask by this many pixels before removing it.",
    )
    parser.add_argument(
        "--save-robot-masks",
        action="store_true",
        help="Save per-camera robot masks as .npy files for debugging.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_pointcloud",
        help="Directory for point cloud and visualization outputs.",
    )
    parser.add_argument(
        "--preview-points",
        type=int,
        default=60000,
        help="Maximum number of points drawn in the saved preview image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive Open3D viewer if open3d is installed.",
    )
    parser.add_argument(
        "--mujoco-gl",
        choices=["egl", "osmesa", "glfw"],
        default=None,
        help="MuJoCo OpenGL backend. Must be set before robosuite is imported.",
    )
    return parser.parse_args()


def resolve_task(args: argparse.Namespace) -> tuple[Path, str, np.ndarray | None]:
    if args.bddl_file is not None:
        return args.bddl_file, args.bddl_file.stem, None

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(
            f"--task-id must be in [0, {task_suite.n_tasks - 1}] for {args.task_suite}"
        )

    task = task_suite.get_task(args.task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = task_suite.get_task_init_states(args.task_id)
    if init_states is not None:
        if not 0 <= args.init_state_id < len(init_states):
            raise ValueError(
                f"--init-state-id must be in [0, {len(init_states) - 1}] for task {args.task_id}"
            )
        init_state = init_states[args.init_state_id]
    else:
        init_state = None
    return bddl_file, task.name, init_state


def create_env(bddl_file: Path, width: int, height: int, camera_names: Iterable[str]) -> OffScreenRenderEnv:
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_names=list(camera_names),
        camera_heights=height,
        camera_widths=width,
        camera_depths=True,
    )


def settle_scene(env: OffScreenRenderEnv, init_state: np.ndarray | None, num_steps_wait: int) -> None:
    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
    no_op_action = np.array([0.0] * 6 + [-1.0], dtype=np.float32)
    for _ in range(num_steps_wait):
        env.step(no_op_action)


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


def model_name(model, kind: str, idx: int) -> str:
    try:
        return getattr(model, f"{kind}_id2name")(idx) or ""
    except Exception:
        names = getattr(model, f"{kind}_names", None)
        if names is not None and idx < len(names):
            return names[idx] or ""
    return ""


def model_name2id(model, kind: str, name: str) -> int | None:
    try:
        return int(getattr(model, f"{kind}_name2id")(name))
    except Exception:
        return None


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


def robot_body_ids(env: OffScreenRenderEnv) -> set[int]:
    ids: set[int] = set()
    for robot in env.robots:
        root_name = getattr(robot.robot_model, "root_body", None)
        if root_name is None:
            continue
        root_id = model_name2id(env.sim.model, "body", root_name)
        if root_id is not None:
            ids.update(body_descendants(env.sim.model, root_id))
    return ids


def find_robot_geoms(env: OffScreenRenderEnv) -> set[int]:
    model = env.sim.model
    robot_bodies = robot_body_ids(env)
    geom_ids = set()
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        if robot_bodies and body_id in robot_bodies:
            geom_ids.add(geom_id)
            continue

        name = model_name(model, "geom", geom_id).lower()
        if any(key in name for key in ("robot", "panda", "ur5", "gripper", "finger", "hand", "arm")):
            geom_ids.add(geom_id)
    return geom_ids


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
    keep_mask: np.ndarray | None = None,
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
        [
            (u - cx) * z / fx,
            (v - cy) * z / fy,
            z,
            np.ones_like(z),
        ],
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


def crop_workspace(
    points: np.ndarray,
    colors: np.ndarray,
    bounds: list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        return points, colors
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
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_preview_png(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> None:
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


def show_open3d(points: np.ndarray, colors: np.ndarray) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("--show requires open3d to be installed") from exc

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.visualization.draw_geometries([point_cloud], window_name="LIBERO point cloud")


def main() -> None:
    args = parse_args()
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_runtime_dependencies()

    bddl_file, task_name, init_state = resolve_task(args)
    env = create_env(bddl_file, args.width, args.height, args.camera_names)
    try:
        settle_scene(env, init_state, args.num_steps_wait)
        robot_geom_ids = set() if args.include_robot else find_robot_geoms(env)
        if args.include_robot:
            print("[info] robot filtering disabled; robot points will be kept")
        else:
            print(f"[info] robot geoms excluded from point cloud: {len(robot_geom_ids)}")

        all_points = []
        all_colors = []
        for camera_name in args.camera_names:
            try:
                rgb, depth_m = render_rgbd(env.sim, camera_name, args.width, args.height)
                keep_mask = None
                if not args.include_robot:
                    robot_mask = robot_pixel_mask(
                        sim=env.sim,
                        camera_name=camera_name,
                        width=args.width,
                        height=args.height,
                        robot_geom_ids=robot_geom_ids,
                        dilation=args.robot_mask_dilation,
                    )
                    keep_mask = ~robot_mask
                    if args.save_robot_masks:
                        np.save(args.output_dir / f"{task_name}_{camera_name}_robot_mask.npy", robot_mask)
            except Exception as exc:
                print(f"[warn] skipped camera {camera_name!r}: {exc}")
                continue

            points, colors = depth_to_world_points(
                sim=env.sim,
                camera_name=camera_name,
                rgb=rgb,
                depth_m=depth_m,
                stride=max(args.stride, 1),
                max_depth=args.max_depth,
                keep_mask=keep_mask,
            )
            np.save(args.output_dir / f"{task_name}_{camera_name}_depth.npy", depth_m)
            all_points.append(points)
            all_colors.append(colors)
            if args.include_robot:
                print(f"[info] {camera_name}: {len(points)} points")
            else:
                removed = int(np.count_nonzero(robot_mask))
                print(f"[info] {camera_name}: {len(points)} points, removed {removed} robot pixels")

        if not all_points:
            raise RuntimeError("No point cloud was reconstructed from the requested cameras.")

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        points, colors = crop_workspace(points, colors, args.workspace_bounds)

        safe_task_name = task_name.replace("/", "_")
        npz_path = args.output_dir / f"{safe_task_name}_pointcloud.npz"
        ply_path = args.output_dir / f"{safe_task_name}_pointcloud.ply"
        preview_path = args.output_dir / f"{safe_task_name}_pointcloud_preview.png"

        np.savez_compressed(npz_path, points=points.astype(np.float32), colors=colors.astype(np.uint8))
        write_ascii_ply(ply_path, points, colors)
        save_preview_png(preview_path, points, colors, args.preview_points)

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        print(f"[done] fused points: {len(points)}")
        print(f"[done] bounds min xyz: {mins}")
        print(f"[done] bounds max xyz: {maxs}")
        print(f"[done] saved npz: {npz_path}")
        print(f"[done] saved ply: {ply_path}")
        print(f"[done] saved preview: {preview_path}")

        if args.show:
            show_open3d(points, colors)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
