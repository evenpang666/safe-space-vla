#!/usr/bin/env python3
"""Create and render a robosuite UR5e scene with fragile upright obstacles."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string


DEFAULT_OUTPUT_DIR = Path("outputs/robosuite_collision_scene")


class UprightBlocksLift(SingleArmEnv):
    """UR5e tabletop scene for lifting a red cube between two upright blocks."""

    def __init__(
        self,
        robots="UR5e",
        env_configuration="default",
        controller_configs=None,
        gripper_types="Robotiq85Gripper",
        initialization_noise=None,
        table_full_size=(0.9, 0.7, 0.05),
        table_friction=(1.0, 0.005, 0.0001),
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="frontview",
        camera_heights=512,
        camera_widths=512,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        self.table_full_size = np.array(table_full_size, dtype=np.float64)
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8), dtype=np.float64)
        self.red_cube_size = np.array((0.025, 0.025, 0.025), dtype=np.float64)
        self.yellow_slab_size = np.array((0.012, 0.045, 0.16), dtype=np.float64)
        self.plate_size = np.array((0.05, 0.006), dtype=np.float64)
        self.success_z_tolerance = 0.018
        self.release_action_threshold = 0.0
        self.object_x = 0.14
        self.red_cube_xy = np.array((self.object_x, 0.0), dtype=np.float64)
        self.left_slab_xy = np.array((self.object_x, 0.135), dtype=np.float64)
        self.right_slab_xy = np.array((self.object_x, -0.135), dtype=np.float64)
        self.plate_xy = np.array((self.object_x, -0.245), dtype=np.float64)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        if self._check_success():
            return 1.0
        if self._check_obstacle_violation():
            return -1.0
        return 0.0

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=[1.35, 0.0, 1.32],
            quat=[0.56, 0.43, 0.43, 0.56],
        )
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.72, 0.0, 1.35],
            quat=[0.638, 0.305, 0.305, 0.638],
        )
        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[-0.0565, 1.2761, 1.4880],
            quat=[0.0099, 0.0069, 0.5912, 0.8064],
        )
        mujoco_arena.set_camera(
            camera_name="leftsideview",
            pos=[-0.0565, -1.2761, 1.4880],
            quat=[0.8064, 0.5912, -0.0069, -0.0099],
        )

        self.red_cube = BoxObject(
            name="red_cube",
            size=self.red_cube_size,
            rgba=[1.0, 0.02, 0.02, 1.0],
            density=500,
            friction=[0.8, 0.005, 0.0001],
        )
        self.left_yellow_slab = BoxObject(
            name="left_yellow_slab",
            size=self.yellow_slab_size,
            rgba=[1.0, 0.82, 0.02, 1.0],
            density=140,
            friction=[0.55, 0.005, 0.0001],
        )
        self.right_yellow_slab = BoxObject(
            name="right_yellow_slab",
            size=self.yellow_slab_size,
            rgba=[1.0, 0.82, 0.02, 1.0],
            density=140,
            friction=[0.55, 0.005, 0.0001],
        )
        self.target_plate = CylinderObject(
            name="target_plate",
            size=self.plate_size,
            rgba=[0.92, 0.96, 1.0, 1.0],
            density=700,
            friction=[1.0, 0.01, 0.001],
            joints=None,
        )
        self.target_plate.get_obj().set(
            "pos",
            array_to_string([self.plate_xy[0], self.plate_xy[1], self.table_offset[2] + self.plate_size[1]]),
        )
        self.objects = [
            self.red_cube,
            self.left_yellow_slab,
            self.right_yellow_slab,
            self.target_plate,
        ]

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self.red_cube_body_id = self.sim.model.body_name2id(self.red_cube.root_body)
        self.left_slab_body_id = self.sim.model.body_name2id(self.left_yellow_slab.root_body)
        self.right_slab_body_id = self.sim.model.body_name2id(self.right_yellow_slab.root_body)
        self.target_plate_body_id = self.sim.model.body_name2id(self.target_plate.root_body)

    def _reset_internal(self):
        super()._reset_internal()
        table_z = float(self.table_offset[2])
        placements = {
            self.red_cube: (self.red_cube_xy, self.red_cube_size[2]),
            self.left_yellow_slab: (self.left_slab_xy, self.yellow_slab_size[2]),
            self.right_yellow_slab: (self.right_slab_xy, self.yellow_slab_size[2]),
        }
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for obj, (xy, half_height) in placements.items():
            pos = np.array([xy[0], xy[1], table_z + half_height], dtype=np.float64)
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, quat]))
        self.sim.forward()

    def _check_success(self):
        cube_pos = self.sim.data.body_xpos[self.red_cube_body_id]
        xy_dist = np.linalg.norm(cube_pos[:2] - self.plate_xy)
        expected_cube_z = self.table_offset[2] + 2.0 * self.plate_size[1] + self.red_cube_size[2]
        on_plate_xy = xy_dist < self.plate_size[0] - 0.01
        on_plate_z = abs(float(cube_pos[2] - expected_cube_z)) < self.success_z_tolerance
        return bool(
            on_plate_xy
            and on_plate_z
            and self._check_gripper_released()
            and not self._check_red_cube_gripper_contact()
            and not self._check_obstacle_violation()
        )

    def _check_gripper_released(self):
        robot = self.robots[0]
        if not getattr(robot, "has_gripper", False):
            return True
        gripper = getattr(robot, "gripper", None)
        current_action = getattr(gripper, "current_action", None)
        if current_action is None:
            return False
        return bool(float(np.asarray(current_action).reshape(-1)[0]) <= self.release_action_threshold)

    def _check_red_cube_gripper_contact(self):
        model = self.sim.model
        data = self.sim.data
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = model.geom_id2name(contact.geom1) or ""
            geom2 = model.geom_id2name(contact.geom2) or ""
            red1 = "red_cube" in geom1
            red2 = "red_cube" in geom2
            grip1 = "gripper" in geom1 or "finger" in geom1
            grip2 = "gripper" in geom2 or "finger" in geom2
            if (red1 and grip2) or (red2 and grip1):
                return True
        return False

    def _check_obstacle_violation(self):
        return bool(
            self._is_slab_tipped(self.left_slab_body_id)
            or self._is_slab_tipped(self.right_slab_body_id)
        )

    def _is_slab_tipped(self, body_id: int) -> bool:
        body_rot = self.sim.data.body_xmat[body_id].reshape(3, 3)
        upright_cos = float(body_rot[2, 2])
        return upright_cos < np.cos(np.deg2rad(12.0))


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except ModuleNotFoundError:
        import matplotlib.pyplot as plt

        plt.imsave(path, rgb)


def render_camera(env: SingleArmEnv, camera_name: str, width: int, height: int) -> np.ndarray:
    rgb = env.sim.render(camera_name=camera_name, width=width, height=height)
    return np.asarray(rgb[::-1], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=768, help="Rendered image width.")
    parser.add_argument("--height", type=int, default=512, help="Rendered image height.")
    parser.add_argument("--camera-name", default="frontview", help="MuJoCo camera to render.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="ur5e_upright_blocks_frontview")
    parser.add_argument("--save-model-xml", action="store_true", help="Also save the generated MJCF XML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = UprightBlocksLift(
        controller_configs=controller_config,
        camera_names=args.camera_name,
        camera_widths=args.width,
        camera_heights=args.height,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        render_camera=args.camera_name,
    )

    try:
        env.reset()
        for _ in range(20):
            env.step(np.zeros(env.action_dim))
        rgb = render_camera(env, args.camera_name, args.width, args.height)
        image_path = args.output_dir / f"{args.name}.png"
        save_rgb(image_path, rgb)

        if args.save_model_xml:
            xml_path = args.output_dir / f"{args.name}.xml"
            xml_path.write_text(env.model.get_xml(), encoding="utf-8")

        print(f"[info] saved front view: {image_path}")
        print("[info] objects:")
        for name, body_id in [
            ("red_cube", env.red_cube_body_id),
            ("left_yellow_slab", env.left_slab_body_id),
            ("right_yellow_slab", env.right_slab_body_id),
            ("target_plate", env.target_plate_body_id),
        ]:
            pos = env.sim.data.body_xpos[body_id]
            quat = env.sim.data.body_xquat[body_id]
            print(
                f"  {name}: pos={array_to_string(pos)} quat={array_to_string(quat)}"
            )
        print(f"[info] obstacle_violation={env._check_obstacle_violation()}")
        print(f"[info] success={env._check_success()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
