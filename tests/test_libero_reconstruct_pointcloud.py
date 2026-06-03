import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import libero_reconstruct_pointcloud


def test_parse_args_rejects_include_robot_with_only_robot(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["libero_reconstruct_pointcloud.py", "--include-robot", "--only-robot"],
    )

    with pytest.raises(SystemExit):
        libero_reconstruct_pointcloud.parse_args()


def test_parse_args_defaults_to_scene_without_robot(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["libero_reconstruct_pointcloud.py"])

    args = libero_reconstruct_pointcloud.parse_args()

    assert not args.include_robot
    assert not args.only_robot


def test_normalize_libero_import_paths_removes_inner_package_path(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    libero_root = repo_root / "openpi" / "third_party" / "libero"
    inner_package_parent = libero_root / "libero"
    monkeypatch.setattr(sys, "path", [str(inner_package_parent), "/tmp/other"])

    libero_reconstruct_pointcloud.normalize_libero_import_paths()

    assert str(inner_package_parent) not in sys.path
    assert sys.path[0] == str(libero_root)


def test_patch_torch_load_for_libero_does_not_add_weights_only_to_old_torch(monkeypatch):
    calls = []

    class FakeTorch:
        pass

    def old_load(*args, **kwargs):
        calls.append(kwargs.copy())
        if "weights_only" in kwargs:
            raise TypeError("'weights_only' is an invalid keyword argument for Unpickler()")
        return "loaded"

    fake_torch = FakeTorch()
    fake_torch.load = old_load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    libero_reconstruct_pointcloud.patch_torch_load_for_libero()

    assert fake_torch.load("init_state.pt") == "loaded"
    assert calls == [{}]


def test_patch_torch_load_for_libero_adds_weights_only_to_new_torch(monkeypatch):
    calls = []

    class FakeTorch:
        pass

    def new_load(path, *, weights_only=True):
        calls.append({"path": path, "weights_only": weights_only})
        return "loaded"

    fake_torch = FakeTorch()
    fake_torch.load = new_load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    libero_reconstruct_pointcloud.patch_torch_load_for_libero()

    assert fake_torch.load("init_state.pt") == "loaded"
    assert calls == [{"path": "init_state.pt", "weights_only": False}]


def test_patch_libero_robot_models_keeps_string_metadata_for_legacy_single_arm(monkeypatch):
    class SingleArm:
        pass

    class MountedPanda:
        default_mount = "RethinkMount"

    class OnTheGroundPanda:
        default_mount = "RethinkMount"

    class MountedUR5e:
        default_mount = "RethinkMount"

    class OnTheGroundUR5E:
        default_mount = "RethinkMount"

    single_arm_module = types.ModuleType("robosuite.robots.single_arm")
    single_arm_module.SingleArm = SingleArm
    monkeypatch.setitem(sys.modules, "robosuite.robots.single_arm", single_arm_module)

    modules = {
        "libero.libero.envs.robots.mounted_panda": ("MountedPanda", MountedPanda),
        "libero.libero.envs.robots.on_the_ground_panda": ("OnTheGroundPanda", OnTheGroundPanda),
        "libero.libero.envs.robots.mounted_ur5e": ("MountedUR5e", MountedUR5e),
        "libero.libero.envs.robots.on_the_ground_ur5e": ("OnTheGroundUR5E", OnTheGroundUR5E),
    }
    for module_name, (class_name, cls) in modules.items():
        module = types.ModuleType(module_name)
        setattr(module, class_name, cls)
        monkeypatch.setitem(sys.modules, module_name, module)

    libero_reconstruct_pointcloud.patch_libero_robot_models()

    assert MountedPanda().default_gripper == "PandaGripper"
    assert MountedPanda().default_controller_config == "default_panda"
    assert MountedUR5e().default_gripper == "Robotiq85Gripper"
    assert MountedUR5e().default_controller_config == "default_ur5e"
