import numpy as np

from scripts.libero_joint_swept_pointcloud import (
    build_link_segments,
    build_geom_skeleton_segments,
    central_axis_segment_for_geom,
    cumulative_swept_point_frame_indices,
    cumulative_projected_point_frames,
    detect_swept_obstacle_collision,
    link_color_ids_from_body_names,
    parse_args,
    points_inside_oriented_boxes,
)


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


def test_parse_args_defaults_to_geom_skeleton(monkeypatch):
    monkeypatch.setattr("sys.argv", ["libero_joint_swept_pointcloud.py"])

    args = parse_args()

    assert args.skeleton_source == "geom"


def test_parse_args_accepts_video_output(monkeypatch):
    monkeypatch.setattr("sys.argv", ["libero_joint_swept_pointcloud.py", "--save-video", "--video-fps", "18"])

    args = parse_args()

    assert args.save_video
    assert args.video_fps == 18


def test_cumulative_swept_point_frame_indices_accumulates_by_step():
    step_ids = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)

    frame_indices = cumulative_swept_point_frame_indices(step_ids)

    assert len(frame_indices) == 3
    np.testing.assert_array_equal(frame_indices[0], [0, 1])
    np.testing.assert_array_equal(frame_indices[1], [0, 1, 2])
    np.testing.assert_array_equal(frame_indices[2], [0, 1, 2, 3, 4])


def test_cumulative_projected_point_frames_draws_only_new_step_points():
    uv = np.asarray([[1.0, 1.0], [3.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    valid = np.asarray([True, True, True])
    colors = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    step_ids = np.asarray([0, 1, 1], dtype=np.int64)

    frames = list(
        cumulative_projected_point_frames(
            uv=uv,
            valid=valid,
            colors=colors,
            step_ids=step_ids,
            width=5,
            height=5,
            point_radius=0,
        )
    )

    assert len(frames) == 2
    assert np.count_nonzero(np.any(frames[0] != 255, axis=2)) == 1
    assert np.count_nonzero(np.any(frames[1] != 255, axis=2)) == 3


def test_central_axis_segment_for_capsule_uses_envelope_center_axis():
    segment = central_axis_segment_for_geom(
        position=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
        size=np.asarray([0.1, 0.5, 0.0], dtype=np.float64),
        geom_kind="capsule",
    )

    np.testing.assert_allclose(segment, [[1.0, 2.0, 2.5], [1.0, 2.0, 3.5]])


def test_central_axis_segment_for_box_uses_longest_envelope_axis():
    segment = central_axis_segment_for_geom(
        position=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
        size=np.asarray([0.1, 0.3, 0.2], dtype=np.float64),
        geom_kind="box",
    )

    np.testing.assert_allclose(segment, [[0.0, -0.3, 0.0], [0.0, 0.3, 0.0]])


def test_central_axis_segment_for_mesh_uses_precomputed_local_axis():
    segment = central_axis_segment_for_geom(
        position=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
        size=np.zeros(3, dtype=np.float64),
        geom_kind="mesh",
        local_segment=np.asarray([[-0.2, 0.0, 0.1], [0.4, 0.0, 0.1]], dtype=np.float64),
    )

    np.testing.assert_allclose(segment, [[0.8, 2.0, 3.1], [1.4, 2.0, 3.1]])


def test_build_geom_skeleton_segments_tracks_each_geom_center_axis_over_time():
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, None], 4, axis=0).reshape(2, 2, 3, 3)
    sizes = np.asarray([[0.05, 0.2, 0.0], [0.1, 0.4, 0.0]], dtype=np.float64)

    segments = build_geom_skeleton_segments(positions, rotations, sizes, ["capsule", "cylinder"])

    assert segments.shape == (2, 2, 2, 3)
    np.testing.assert_allclose(segments[0, 0], [[0.0, 0.0, -0.2], [0.0, 0.0, 0.2]])
    np.testing.assert_allclose(segments[1, 1], [[1.0, 1.0, -0.4], [1.0, 1.0, 0.4]])


def test_link_color_ids_group_multiple_geoms_on_the_same_link():
    color_ids, link_names = link_color_ids_from_body_names(
        ["robot0_link0", "robot0_link0", "robot0_link1", "gripper0_leftfinger", "robot0_link1"]
    )

    np.testing.assert_array_equal(color_ids, [0, 0, 1, 2, 1])
    np.testing.assert_array_equal(link_names, ["robot0_link0", "robot0_link1", "gripper0_leftfinger"])


def test_points_inside_oriented_boxes_detects_inflated_box_hits():
    points = np.asarray(
        [
            [0.49, 0.0, 0.0],
            [0.56, 0.0, 0.0],
            [0.70, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    centers = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    axes = np.asarray([np.eye(3)], dtype=np.float64)
    half_sizes = np.asarray([[0.5, 0.5, 0.5]], dtype=np.float64)

    inside = points_inside_oriented_boxes(points, centers, axes, half_sizes, margin=0.06)

    np.testing.assert_array_equal(inside, [True, True, False])


def test_detect_swept_obstacle_collision_uses_safe_space_occupied_grid():
    swept_points = np.asarray(
        [
            [0.25, 0.25, 0.25],
            [1.25, 1.25, 1.25],
        ],
        dtype=np.float64,
    )
    occupied_grid = np.zeros((3, 3, 3), dtype=bool)
    occupied_grid[1, 1, 1] = True
    safe_space = {
        "workspace_bounds": np.asarray([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], dtype=np.float32),
        "voxel_size": np.asarray(1.0, dtype=np.float32),
        "occupied_grid": occupied_grid,
    }

    result = detect_swept_obstacle_collision(swept_points, safe_space)

    assert result.collides
    assert result.method == "occupied_grid"
    assert result.collision_point_count == 1
    np.testing.assert_array_equal(result.colliding_point_indices, [1])


def test_detect_swept_obstacle_collision_reports_no_hit_for_empty_space():
    swept_points = np.asarray([[2.25, 2.25, 2.25]], dtype=np.float64)
    occupied_grid = np.zeros((3, 3, 3), dtype=bool)
    occupied_grid[1, 1, 1] = True
    safe_space = {
        "workspace_bounds": np.asarray([0.0, 3.0, 0.0, 3.0, 0.0, 3.0], dtype=np.float32),
        "voxel_size": np.asarray(1.0, dtype=np.float32),
        "occupied_grid": occupied_grid,
    }

    result = detect_swept_obstacle_collision(swept_points, safe_space)

    assert not result.collides
    assert result.collision_point_count == 0
