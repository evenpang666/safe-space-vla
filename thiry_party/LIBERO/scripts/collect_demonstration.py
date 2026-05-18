import argparse
import cv2
import datetime
import h5py
import init_path
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action


import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *


"""Driver class for Xbox controller.

This class provides driver support for Xbox controllers using pygame.
Assumes a standard Xbox controller mapping.

"""

import threading
import time
from collections import namedtuple

import numpy as np
from pynput.keyboard import Controller, Key, Listener

import pygame

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


def scale_to_control(x, min_v=-1.0, max_v=1.0):
    """
    Normalize raw readings to target range. For pygame axes, already -1 to 1, so minimal scaling.
    """
    x = min(max(x, min_v), max_v)
    return x


class Xbox(Device):
    """
    A minimalistic driver class for Xbox controller using pygame.

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(
        self,
        env,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
    ):
        super().__init__()

        print("Opening Xbox device")
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("No joystick connected. Please connect an Xbox controller.")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print("Joystick: %s" % self.joystick.get_name())

        self._display_controls()
        self._reset_internal_state()

        self.grasp_active = False

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 0.05
        


        # launch a new listener thread to listen to Xbox
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        # also add a keyboard for aux controls
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Y button", "reset simulation")
        print_command("X button (press)", "toggle gripper")
        print_command("Left stick (lateral)", "move arm horizontally in x-y plane")
        print_command("LT / RT triggers", "move arm vertically (RT up, LT down)")
        print_command("Right stick (lateral)", "rotate arm yaw / pitch")
        print_command("LB / RB buttons", "rotate arm roll")
        print_command("Control+C", "quit")
        print_command("b", "toggle arm/base mode (if applicable)")
        print_command("s", "switch active arm (if multi-armed robot)")
        print_command("=", "switch active robot (if multi-robot environment)")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        # super()._reset_internal_state()
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False
        self.prev_x = 0

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the controller.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        )  # create local variable to return, then reset internal drotation
        self.last_drotation = np.array(self.raw_drotation)
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
            # base_mode=int(self.base_mode),
        )


    def run(self):
        while True:
            time.sleep(0.01)
            if self._enabled:
                pygame.event.pump()

            # controls for moving position
            if scale_to_control(self.joystick.get_axis(1)) < -0.5:
                self.pos[0] -= self._pos_step * self.pos_sensitivity  # dec x
            if scale_to_control(self.joystick.get_axis(1)) > 0.5:
                self.pos[0] += self._pos_step * self.pos_sensitivity  # inc x
            if scale_to_control(self.joystick.get_axis(0)) < -0.5:
                self.pos[1] -= self._pos_step * self.pos_sensitivity  # dec y
            if scale_to_control(self.joystick.get_axis(0)) > 0.5:
                self.pos[1] += self._pos_step * self.pos_sensitivity  # inc y
            # if scale_to_control(self.joystick.get_axis(4)) > 0:
            if scale_to_control(self.joystick.get_axis(3)) > 0.5:
                self.pos[2] -= self._pos_step * self.pos_sensitivity  # dec z
            # if scale_to_control(self.joystick.get_axis(5)) > 0:
            if scale_to_control(self.joystick.get_axis(3)) < -0.5:
                self.pos[2] += self._pos_step * self.pos_sensitivity  # inc z

            # controls for moving orientation
            if scale_to_control(self.joystick.get_axis(2)) < -0.5:
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] -= 0.1 * self.rot_sensitivity
            if scale_to_control(self.joystick.get_axis(2)) > 0.5:
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] += 0.1 * self.rot_sensitivity
            # if scale_to_control(self.joystick.get_axis(3)) < -0.5:
            if scale_to_control(self.joystick.get_axis(5)) > 0:
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] += 0.1 * self.rot_sensitivity
            # if scale_to_control(self.joystick.get_axis(3)) > 0.5:
            if scale_to_control(self.joystick.get_axis(4)) > 0:
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] -= 0.1 * self.rot_sensitivity
            if self.joystick.get_button(6) > 0:
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] += 0.1 * self.rot_sensitivity
            if self.joystick.get_button(7) > 0:
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] -= 0.1 * self.rot_sensitivity
            if self.joystick.get_button(4):
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
            if self.joystick.get_button(0):
                curr_x = self.joystick.get_button(0)
                if curr_x == 1 and self.prev_x == 0:
                    self.grasp = not self.grasp
                self.prev_x = 1
            if self.joystick.get_button(1):
                curr_x = self.joystick.get_button(1)
                if curr_x == 1 and self.prev_x == 1:
                    self.grasp = not self.grasp
                self.prev_x = 0

            
        

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            # controls for mobile base (only applicable if mobile base present)
            if key.char == "b":
                self.base_modes[self.active_robot] = not self.base_modes[self.active_robot]  # toggle mobile base
            elif key.char == "s":
                self.active_arm_index = (self.active_arm_index + 1) % len(self.all_robot_arms[self.active_robot])
            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots

        except AttributeError as e:
            pass

    def _postprocess_device_outputs(self, dpos, drotation):
        # drotation = drotation * 50
        # dpos = dpos * 125
        drotation = drotation * 1.5
        dpos = dpos * 75

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation








def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, remove_directory=[]
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue

    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    saving = True
    count = 0

    while True:
        count += 1
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # If action is none, then this a reset so we should break
        if action is None:
            print("Break")
            saving = False
            break

        # Run environment step

        env.step(action)
        env.render()
        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    print(count)
    # cleanup for end of data collection episodes
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return saving


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, args, remove_directory=[]
):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="demonstration_data",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=50,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str)

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # Check if we're using a multi-armed environment and use env_configuration argument if so

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )

    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
        env.viewer.add_keypress_callback(device.on_press)
        # env.viewer.add_keyup_callback(device.on_release)
        # env.viewer.add_keyrepeat_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "xbox":
        # from robosuite.devices.xbox import Xbox
        device = Xbox(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' , 'xbox', or 'spacemouse'."
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"
        + language_instruction.replace(" ", "_").strip('""'),
    )

    os.makedirs(new_dir)

    # collect demonstrations

    remove_directory = []
    i = 0
    while i < args.num_demonstration:
        print(i)
        saving = collect_human_trajectory(
            env, device, args.arm, args.config, problem_info, remove_directory
        )
        if saving:
            print(remove_directory)
            gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, args, remove_directory
            )
            i += 1
