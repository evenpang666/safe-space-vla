from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.libero_link_point_targets import flatten_link_points

if TYPE_CHECKING:
    from scripts.libero_joint_swept_pointcloud import CollisionResult


def predicted_link_points_collision(
    pred_link_points: np.ndarray,
    safe_space: dict[str, np.ndarray],
    collision_margin: float = 0.0,
) -> "CollisionResult":
    from scripts.libero_joint_swept_pointcloud import detect_swept_obstacle_collision

    link_points = np.asarray(pred_link_points)
    if link_points.ndim == 5:
        if link_points.shape[0] != 1:
            raise ValueError(f"pred_link_points must contain a single sample, got batch size {link_points.shape[0]}")
        link_points = link_points[0]
    points = flatten_link_points(link_points)
    return detect_swept_obstacle_collision(points, safe_space, collision_margin=collision_margin)


def collision_result_to_dict(result: "CollisionResult") -> dict[str, Any]:
    return {
        "collision": bool(result.collides),
        "collision_method": str(result.method),
        "collision_margin": float(result.collision_margin),
        "collision_point_count": int(result.collision_point_count),
        "collision_point_indices": np.asarray(result.colliding_point_indices, dtype=np.int64),
    }
