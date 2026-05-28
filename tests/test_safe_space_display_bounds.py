import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_safe_space_from_pointcloud import table_aligned_display_bounds


def test_table_aligned_display_bounds_uses_table_xy_and_table_z_bottom():
    workspace_bounds = np.array([-1.2, 1.3, -0.8, 0.9, 0.1, 1.8], dtype=np.float32)
    table_xy_bounds = np.array([-0.5, 0.5, -0.35, 0.35], dtype=np.float32)

    display_bounds = table_aligned_display_bounds(
        workspace_bounds,
        table_z=0.72,
        table_xy_bounds=table_xy_bounds,
    )

    np.testing.assert_allclose(display_bounds, [-0.5, 0.5, -0.35, 0.35, 0.72, 1.8])
