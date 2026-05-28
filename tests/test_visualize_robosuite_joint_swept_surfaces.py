import numpy as np

from scripts.visualize_robosuite_joint_swept_surfaces import (
    overlay_mask,
    rasterize_polygons,
    sample_swept_surface_points,
)


def test_rasterize_polygons_fills_projected_sweep_area():
    polygons = np.asarray(
        [
            [[2.0, 2.0], [7.0, 2.0], [7.0, 7.0], [2.0, 7.0]],
        ],
        dtype=np.float64,
    )

    mask = rasterize_polygons(polygons, height=10, width=10)

    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert mask[4, 4] == 255
    assert mask[0, 0] == 0


def test_overlay_mask_blends_only_masked_pixels():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.asarray([[0, 255], [0, 0]], dtype=np.uint8)

    overlay = overlay_mask(rgb, mask, color=(100, 50, 0), alpha=0.5)

    assert overlay.dtype == np.uint8
    np.testing.assert_array_equal(overlay[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(overlay[0, 1], [50, 25, 0])


def test_sample_swept_surface_points_samples_link_and_time_axes():
    segment_path = np.asarray(
        [
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
        ],
        dtype=np.float64,
    )

    points, link_ids, step_ids = sample_swept_surface_points(
        segment_path,
        link_samples=3,
        time_samples=2,
    )

    assert points.shape == (6, 3)
    np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(points[1], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(points[-1], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(link_ids, np.zeros(6, dtype=np.int64))
    np.testing.assert_array_equal(step_ids, np.zeros(6, dtype=np.int64))
