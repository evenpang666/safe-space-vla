import numpy as np

from scripts.libero_joint_swept_pointcloud import build_link_segments


def test_build_link_segments_supports_panda_seven_joint_chain():
    anchor_path = np.zeros((2, 8, 3), dtype=np.float64)
    anchor_path[:, :, 2] = np.arange(8, dtype=np.float64)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)

    segments = build_link_segments(anchor_path, rotations, gripper_width=0.08)

    assert segments.shape == (2, 8, 2, 3)
    np.testing.assert_allclose(segments[0, 0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(segments[0, 6], [[0.0, 0.0, 6.0], [0.0, 0.0, 7.0]])
    np.testing.assert_allclose(segments[0, 7], [[-0.04, 0.0, 7.0], [0.04, 0.0, 7.0]])
    assert np.dot(segments[0, 6, 1] - segments[0, 6, 0], segments[0, 7, 1] - segments[0, 7, 0]) == 0.0
