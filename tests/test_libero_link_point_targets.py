import numpy as np
import pytest

from scripts.libero_link_point_targets import (
    flatten_link_points,
    sample_link_points_from_segments,
)


def test_sample_link_points_from_segments_keeps_time_link_point_topology():
    segment_path = np.asarray(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]],
            ],
            [
                [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]],
            ],
        ],
        dtype=np.float64,
    )

    points = sample_link_points_from_segments(segment_path, points_per_link=3)

    assert points.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(points[0, 0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(points[1, 1], [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]])


def test_flatten_link_points_preserves_row_major_time_link_point_order():
    link_points = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)

    flat = flatten_link_points(link_points)

    assert flat.shape == (24, 3)
    np.testing.assert_array_equal(flat[0], link_points[0, 0, 0])
    np.testing.assert_array_equal(flat[4], link_points[0, 1, 0])
    np.testing.assert_array_equal(flat[-1], link_points[1, 2, 3])


def test_sample_link_points_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="points_per_link must be >= 2"):
        sample_link_points_from_segments(np.zeros((2, 1, 2, 3)), points_per_link=1)

    with pytest.raises(ValueError, match="segment_path must have shape"):
        sample_link_points_from_segments(np.zeros((2, 1, 3)), points_per_link=2)


def test_flatten_link_points_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="link_points must have shape"):
        flatten_link_points(np.zeros((2, 3, 4)))
