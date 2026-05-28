import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_libero_pointcloud_camera_view import rasterize_projected_points


def test_rasterize_projected_points_draws_near_points_over_far_points():
    uv = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    camera_depth = np.array([10.0, 1.0], dtype=np.float64)
    colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    valid = np.array([True, True])

    image = rasterize_projected_points(
        width=5,
        height=5,
        uv=uv,
        camera_depth=camera_depth,
        colors=colors,
        valid=valid,
        point_radius=0,
    )

    np.testing.assert_array_equal(image[2, 2], [0, 0, 255])
