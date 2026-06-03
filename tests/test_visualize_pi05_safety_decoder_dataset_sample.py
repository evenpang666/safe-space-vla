from pathlib import Path

import numpy as np

from scripts.visualize_pi05_safety_decoder_dataset_sample import (
    box_faces_from_corners,
    default_output_path,
    flatten_link_points_for_plot,
    load_safe_space_obbs,
    load_dataset_sample,
)


def test_flatten_link_points_for_plot_preserves_time_link_point_order():
    link_points = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)

    points, link_ids, step_ids = flatten_link_points_for_plot(link_points)

    assert points.shape == (24, 3)
    np.testing.assert_array_equal(points[0], link_points[0, 0, 0])
    np.testing.assert_array_equal(points[4], link_points[0, 1, 0])
    np.testing.assert_array_equal(points[-1], link_points[1, 2, 3])
    np.testing.assert_array_equal(link_ids[:8], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(step_ids[:12], np.zeros(12, dtype=np.int64))
    np.testing.assert_array_equal(step_ids[12:], np.ones(12, dtype=np.int64))


def test_flatten_link_points_for_plot_selects_one_time_index():
    link_points = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)

    points, link_ids, step_ids = flatten_link_points_for_plot(link_points, time_index=1)

    assert points.shape == (12, 3)
    np.testing.assert_array_equal(points[0], link_points[1, 0, 0])
    np.testing.assert_array_equal(link_ids, np.repeat(np.arange(3), 4))
    np.testing.assert_array_equal(step_ids, np.ones(12, dtype=np.int64))


def test_load_dataset_sample_returns_metadata(tmp_path: Path):
    dataset = tmp_path / "decoder_dataset.npz"
    np.savez_compressed(
        dataset,
        target_link_points=np.zeros((2, 5, 3, 4, 3), dtype=np.float32),
        target_source=np.asarray("rollout_surface"),
        prefix_tokens=np.zeros((2, 7, 8), dtype=np.float32),
        action_chunks=np.zeros((2, 10, 7), dtype=np.float32),
        rollout_ids=np.asarray([4, 5], dtype=np.int64),
        step_ids=np.asarray([10, 11], dtype=np.int64),
        task_ids=np.asarray([0, 0], dtype=np.int64),
        link_names=np.asarray(["a", "b", "c"]),
    )

    sample = load_dataset_sample(dataset, 1)

    assert sample.index == 1
    assert sample.target_link_points.shape == (5, 3, 4, 3)
    assert sample.rollout_id == 5
    assert sample.step_id == 11
    assert sample.link_names.tolist() == ["a", "b", "c"]


def test_load_dataset_sample_rejects_non_rollout_surface_dataset(tmp_path: Path):
    dataset = tmp_path / "decoder_dataset.npz"
    np.savez_compressed(
        dataset,
        target_link_points=np.zeros((1, 5, 3, 4, 3), dtype=np.float32),
        target_source=np.asarray("action_fk"),
    )

    try:
        load_dataset_sample(dataset, 0)
    except ValueError as exc:
        assert "rollout_surface" in str(exc)
    else:
        raise AssertionError("non-rollout-surface dataset was accepted")


def test_default_output_path_uses_sample_index():
    path = default_output_path(Path("outputs/pi05_safety_decoder/dataset.npz"), 7)

    assert path == Path("outputs/pi05_safety_decoder/dataset_sample0007_link_points.png")


def test_load_safe_space_obbs_reads_corners(tmp_path: Path):
    safe_space = tmp_path / "safe_space.npz"
    corners = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    np.savez_compressed(safe_space, obstacle_box_corners=corners)

    loaded = load_safe_space_obbs(safe_space)

    np.testing.assert_array_equal(loaded, corners)


def test_box_faces_from_corners_returns_six_quad_faces():
    corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    faces = box_faces_from_corners(corners)

    assert len(faces) == 6
    assert all(face.shape == (4, 3) for face in faces)
    np.testing.assert_array_equal(faces[0], corners[[0, 1, 2, 3]])
