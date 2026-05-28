#!/usr/bin/env python3
"""Fixed-topology robot link-point target utilities."""

from __future__ import annotations

import numpy as np


def sample_link_points_from_segments(segment_path: np.ndarray, points_per_link: int) -> np.ndarray:
    """Sample fixed point indices along each link segment at every time step.

    Args:
        segment_path: Array with shape ``(T, L, 2, 3)``.
        points_per_link: Number of ordered samples per link segment.

    Returns:
        Array with shape ``(T, L, points_per_link, 3)``.
    """
    try:
        point_count = int(points_per_link)
    except (TypeError, ValueError):
        raise ValueError("points_per_link must be an integer") from None
    if point_count != points_per_link:
        raise ValueError("points_per_link must be an integer")
    if points_per_link < 2:
        raise ValueError("points_per_link must be >= 2")
    segment_path = np.asarray(segment_path, dtype=np.float64)
    if segment_path.ndim != 4 or segment_path.shape[-2:] != (2, 3):
        raise ValueError(f"segment_path must have shape (T, L, 2, 3), got {segment_path.shape}")

    u = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
    start = segment_path[:, :, 0, :]
    end = segment_path[:, :, 1, :]
    points = (1.0 - u[None, None, :, None]) * start[:, :, None, :]
    points += u[None, None, :, None] * end[:, :, None, :]
    return points.astype(np.float32)


def flatten_link_points(link_points: np.ndarray) -> np.ndarray:
    """Flatten ``(T, L, P, 3)`` link points to ``(T*L*P, 3)`` for collision checks."""
    link_points = np.asarray(link_points)
    if link_points.ndim != 4 or link_points.shape[-1] != 3:
        raise ValueError(f"link_points must have shape (T, L, P, 3), got {link_points.shape}")
    return link_points.reshape(-1, 3)
