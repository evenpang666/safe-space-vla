from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_libero_frontview_figures import render_frontview_figures
from scripts.render_libero_frontview_figures import select_display_workspace_bounds


def test_render_frontview_figures_writes_swept_and_obb_pngs(tmp_path: Path) -> None:
    swept_path = tmp_path / "swept.npz"
    scene_path = tmp_path / "scene.npz"
    obb_path = tmp_path / "obb.npz"
    output_dir = tmp_path / "figures"

    rng = np.random.default_rng(0)
    swept_points = rng.normal(size=(128, 3)).astype(np.float32)
    scene_points = rng.normal(size=(160, 3)).astype(np.float32)
    scene_points[:, 2] += 1.0
    colors = np.full((160, 3), 180, dtype=np.uint8)
    box_corners = np.array(
        [
            [
                [-0.2, -0.1, 0.9],
                [0.2, -0.1, 0.9],
                [0.2, 0.1, 0.9],
                [-0.2, 0.1, 0.9],
                [-0.2, -0.1, 1.2],
                [0.2, -0.1, 1.2],
                [0.2, 0.1, 1.2],
                [-0.2, 0.1, 1.2],
            ]
        ],
        dtype=np.float32,
    )

    np.savez_compressed(swept_path, points=swept_points)
    np.savez_compressed(scene_path, points=scene_points, colors=colors)
    np.savez_compressed(
        obb_path,
        workspace_bounds=np.array([-1, 1, -1, 1, 0, 2], dtype=np.float32),
        obstacle_box_corners=box_corners,
        obstacle_box_point_counts=np.array([42], dtype=np.int64),
    )

    swept_png, obb_png = render_frontview_figures(
        swept_pointcloud=swept_path,
        scene_pointcloud=scene_path,
        obb_safe_space=obb_path,
        output_dir=output_dir,
        name="demo",
        max_points=1000,
    )

    assert swept_png.exists()
    assert obb_png.exists()
    assert swept_png.stat().st_size > 0
    assert obb_png.stat().st_size > 0


def test_select_display_workspace_bounds_prefers_saved_display_bounds() -> None:
    workspace_bounds = np.array([-1, 1, -1, 1, 0, 2], dtype=np.float32)
    display_bounds = np.array([-0.5, 0.5, -0.3, 0.3, 0.75, 2], dtype=np.float32)

    selected = select_display_workspace_bounds(
        workspace_bounds=workspace_bounds,
        display_workspace_bounds=display_bounds,
        table_z=np.array(0.7, dtype=np.float32),
    )

    np.testing.assert_allclose(selected, display_bounds)
