from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.calibrate_d435i_to_ur_base import camera_to_base_from_board_pose
from real_scripts.calibrate_d435i_to_ur_base import rigid_transform_from_correspondences


def test_rigid_transform_from_correspondences_maps_board_points_into_ur_base():
    board_points = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]])
    transform = np.eye(4)
    transform[:3, :3] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform[:3, 3] = [0.3, -0.2, 0.5]
    base_points = (transform[:3, :3] @ board_points.T).T + transform[:3, 3]

    actual = rigid_transform_from_correspondences(board_points, base_points)

    np.testing.assert_allclose(actual, transform, atol=1e-8)


def test_camera_to_base_uses_documented_board_pose_composition():
    board_to_base = np.eye(4)
    board_to_base[:3, 3] = [1.0, 2.0, 3.0]
    board_to_camera = np.eye(4)
    board_to_camera[:3, 3] = [0.1, 0.2, 0.3]

    actual = camera_to_base_from_board_pose(board_to_base, board_to_camera)

    np.testing.assert_allclose(actual[:3, 3], [0.9, 1.8, 2.7])
