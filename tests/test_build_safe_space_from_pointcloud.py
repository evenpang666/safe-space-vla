import numpy as np

from scripts import build_safe_space_from_pointcloud as builder


def test_component_boxes_can_use_face_connectivity_to_separate_diagonal_obstacles():
    points = np.asarray(
        [
            [0.05, 0.05, 0.05],
            [0.06, 0.05, 0.05],
            [0.05, 0.06, 0.05],
            [0.15, 0.15, 0.05],
            [0.16, 0.15, 0.05],
            [0.15, 0.16, 0.05],
        ],
        dtype=np.float32,
    )
    bounds = np.asarray([0.0, 0.3, 0.0, 0.3, 0.0, 0.2], dtype=np.float32)

    merged = builder.component_boxes_from_tabletop_points(
        points=points,
        bounds=bounds,
        table_z=0.0,
        component_voxel_size=0.1,
        component_connectivity=26,
        min_component_points=1,
        box_margin=0.0,
        box_shape="cuboid",
        box_orientation="axis_aligned",
    )
    separated = builder.component_boxes_from_tabletop_points(
        points=points,
        bounds=bounds,
        table_z=0.0,
        component_voxel_size=0.1,
        component_connectivity=6,
        min_component_points=1,
        box_margin=0.0,
        box_shape="cuboid",
        box_orientation="axis_aligned",
    )

    assert merged[2].shape == (1, 3)
    assert separated[2].shape == (2, 3)
