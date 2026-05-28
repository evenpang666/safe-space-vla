import sys
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
