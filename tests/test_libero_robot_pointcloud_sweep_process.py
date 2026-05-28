import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.libero_robot_pointcloud_sweep_process import (
    YELLOW_LINE_RGBA,
    temporal_line_colors,
    temporal_line_segments,
)


def test_temporal_line_segments_connect_matching_points_between_steps():
    points_by_step = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )

    segments = temporal_line_segments(points_by_step)
    colors = temporal_line_colors(len(segments))

    assert segments.shape == (4, 2, 3)
    np.testing.assert_allclose(segments[0], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(segments[1], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    np.testing.assert_allclose(segments[2], [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(colors[0], YELLOW_LINE_RGBA)
