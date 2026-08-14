"""Dependency-light CBF-QP projection for UR7e fixed-identity surface points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrientedBox:
    center: np.ndarray  # [3]
    axes: np.ndarray  # [3,3], columns are box axes
    half_sizes: np.ndarray  # [3]


@dataclass(frozen=True)
class PointFlowConstraint:
    normal: np.ndarray
    current_h: float
    predicted_h: float
    link_id: int
    point_id: int
    box_id: int


def point_jacobian_fd(sampler, qpos: np.ndarray, *, epsilon_rad: float = 1e-4) -> np.ndarray:
    """Central finite-difference d(surface point)/d(q), shape ``(L,P,3,6)``."""
    q = np.asarray(qpos, dtype=np.float64).reshape(6)
    if epsilon_rad <= 0.0:
        raise ValueError("epsilon_rad must be positive")
    jacobian = np.empty((*sampler.link_points(q).shape, 6), dtype=np.float32)
    for joint in range(6):
        plus, minus = q.copy(), q.copy()
        plus[joint] += epsilon_rad
        minus[joint] -= epsilon_rad
        jacobian[..., joint] = (sampler.link_points(plus) - sampler.link_points(minus)) / (2.0 * epsilon_rad)
    return jacobian


def select_point_flow_constraints(
    current_link_points: np.ndarray,
    predicted_link_points: np.ndarray,
    boxes: list[OrientedBox],
    *,
    collision_margin_m: float = 0.0,
    trigger_margin_m: float = 0.02,
    max_constraints: int = 32,
) -> list[PointFlowConstraint]:
    current = np.asarray(current_link_points, dtype=np.float32)
    predicted = np.asarray(predicted_link_points, dtype=np.float32)
    if predicted.ndim != 4 or current.shape != predicted.shape[1:]:
        raise ValueError(f"Expected current (L,P,3) and predicted (T,L,P,3), got {current.shape}, {predicted.shape}")
    constraints: list[PointFlowConstraint] = []
    seen: set[tuple[int, int, int]] = set()
    for box_id, box in enumerate(boxes):
        center = np.asarray(box.center, dtype=np.float32).reshape(3)
        axes = np.asarray(box.axes, dtype=np.float32).reshape(3, 3)
        half = np.asarray(box.half_sizes, dtype=np.float32).reshape(3) + float(collision_margin_m)
        current_local = (current - center) @ axes
        predicted_local = (predicted - center) @ axes
        dangerous = np.all(np.abs(predicted_local) <= (half + float(trigger_margin_m)), axis=-1)
        for time_index, link_id, point_id in np.argwhere(dangerous):
            key = (int(link_id), int(point_id), box_id)
            if key in seen:
                continue
            seen.add(key)
            local_current = current_local[link_id, point_id]
            local_predicted = predicted_local[time_index, link_id, point_id]
            axis = int(np.argmax(np.maximum(np.abs(local_current) / np.maximum(half, 1e-6), np.abs(local_predicted) / np.maximum(half, 1e-6))))
            sign = 1.0 if local_predicted[axis] >= 0.0 else -1.0
            normal = sign * axes[:, axis]
            current_h = sign * float(local_current[axis]) - float(half[axis])
            predicted_h = sign * float(local_predicted[axis]) - float(half[axis])
            constraints.append(PointFlowConstraint(normal.astype(np.float32), current_h, predicted_h, int(link_id), int(point_id), box_id))
    return sorted(constraints, key=lambda item: min(item.current_h, item.predicted_h))[: int(max_constraints)]


def project_joint_delta_qp(
    nominal_delta: np.ndarray,
    jacobian: np.ndarray,
    constraints: list[PointFlowConstraint],
    *,
    lower_delta: np.ndarray,
    upper_delta: np.ndarray,
    alpha: float = 1.0,
    iterations: int = 32,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve a bounded least-change QP by cyclic projection onto CBF halfspaces."""
    nominal = np.asarray(nominal_delta, dtype=np.float64).reshape(6)
    low = np.asarray(lower_delta, dtype=np.float64).reshape(6)
    high = np.asarray(upper_delta, dtype=np.float64).reshape(6)
    if np.any(low > high):
        raise ValueError("joint-delta lower bounds exceed upper bounds")
    delta = np.clip(nominal, low, high)
    if not constraints:
        return delta.astype(np.float32), {"triggered": False, "success": True, "constraint_count": 0, "max_violation": 0.0}
    rows, rhs = [], []
    for item in constraints:
        grad = np.asarray(item.normal, dtype=np.float64) @ np.asarray(jacobian[item.link_id, item.point_id], dtype=np.float64)
        if float(np.linalg.norm(grad)) <= 1e-10:
            continue
        # h(q + dq) >= (1-alpha) h(q); predicted penetration makes this
        # conservative even before the current point reaches the box.
        barrier_h = min(float(item.current_h), float(item.predicted_h))
        rows.append(grad)
        rhs.append(-float(alpha) * barrier_h)
    if not rows:
        return delta.astype(np.float32), {"triggered": True, "success": False, "constraint_count": len(constraints), "max_violation": float("inf")}
    matrix = np.stack(rows)
    rhs_array = np.asarray(rhs)
    for _ in range(max(int(iterations), 1)):
        for row, bound in zip(matrix, rhs_array, strict=True):
            violation = float(bound - row @ delta)
            if violation > 0.0:
                delta += violation * row / max(float(row @ row), 1e-12)
                delta = np.clip(delta, low, high)
    violations = rhs_array - matrix @ delta
    max_violation = max(float(np.max(violations)), 0.0)
    return delta.astype(np.float32), {
        "triggered": True,
        "success": max_violation <= 1e-4,
        "constraint_count": len(rows),
        "max_violation": max_violation,
    }
