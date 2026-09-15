"""CBF-QP projection for UR7e fixed-identity surface points."""

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
    time_id: int = 0


@dataclass(frozen=True)
class PointPairFlowConstraint:
    """Minimum-separation barrier between one point on each robot arm."""

    normal: np.ndarray
    current_h: float
    predicted_h: float
    first_point_id: int
    second_point_id: int
    time_id: int = 0


def point_jacobian_fd(
    sampler, qpos: np.ndarray, *, gripper_position: np.ndarray | None = None,
    point_indices: np.ndarray | None = None, epsilon_rad: float = 1e-4,
) -> np.ndarray:
    """Central finite-difference d(surface point)/d(q), shape ``(...,3,J)``."""
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if q.size not in (6, 12):
        raise ValueError("qpos must contain six or twelve UR joint angles")
    if epsilon_rad <= 0.0:
        raise ValueError("epsilon_rad must be positive")
    def points(value: np.ndarray) -> np.ndarray:
        try:
            result = sampler.link_points(value, gripper_position)
        except TypeError:
            result = sampler.link_points(value)
        result = np.asarray(result, dtype=np.float32)
        if point_indices is not None:
            result = result.reshape(-1, 3)[np.asarray(point_indices, dtype=np.int64)]
        return result
    jacobian = np.empty((*points(q).shape, q.size), dtype=np.float32)
    for joint in range(q.size):
        plus, minus = q.copy(), q.copy()
        plus[joint] += epsilon_rad
        minus[joint] -= epsilon_rad
        jacobian[..., joint] = (points(plus) - points(minus)) / (2.0 * epsilon_rad)
    return jacobian


def _box_signed_distance(local: np.ndarray, half: np.ndarray) -> np.ndarray:
    delta = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
    inside = np.minimum(np.max(delta, axis=-1), 0.0)
    return outside + inside


def select_point_flow_constraints(
    current_link_points: np.ndarray,
    predicted_link_points: np.ndarray,
    boxes: list[OrientedBox],
    *,
    collision_margin_m: float = 0.0,
    trigger_margin_m: float = 0.02,
    max_constraints: int = 32,
) -> list[PointFlowConstraint]:
    current_source = np.asarray(current_link_points, dtype=np.float32)
    predicted_source = np.asarray(predicted_link_points, dtype=np.float32)
    if current_source.ndim not in (2, 3) or current_source.shape[-1] != 3:
        raise ValueError(f"Expected current (K,3) or (L,P,3), got {current_source.shape}")
    if predicted_source.shape[1:] != current_source.shape:
        raise ValueError(f"Prediction {predicted_source.shape} does not follow current shape {current_source.shape}")
    link_shape = current_source.shape[:-1]
    current = current_source.reshape(-1, 3)
    predicted = predicted_source.reshape(predicted_source.shape[0], -1, 3)
    constraints: list[PointFlowConstraint] = []
    seen: set[tuple[int, int, int]] = set()
    for box_id, box in enumerate(boxes):
        center = np.asarray(box.center, dtype=np.float32).reshape(3)
        axes = np.asarray(box.axes, dtype=np.float32).reshape(3, 3)
        half = np.asarray(box.half_sizes, dtype=np.float32).reshape(3) + float(collision_margin_m)
        current_local = (current - center) @ axes
        predicted_local = (predicted - center) @ axes
        distances = _box_signed_distance(predicted_local, half)
        dangerous = distances <= float(trigger_margin_m)
        for time_index, flat_point_id in np.argwhere(dangerous):
            key = (0, int(flat_point_id), box_id)
            if key in seen:
                continue
            seen.add(key)
            local_current = current_local[flat_point_id]
            local_predicted = predicted_local[time_index, flat_point_id]
            axis = int(np.argmax(np.maximum(np.abs(local_current) / np.maximum(half, 1e-6), np.abs(local_predicted) / np.maximum(half, 1e-6))))
            sign = 1.0 if local_predicted[axis] >= 0.0 else -1.0
            normal = sign * axes[:, axis]
            current_h = float(_box_signed_distance(local_current[None], half)[0])
            predicted_h = float(distances[time_index, flat_point_id])
            if len(link_shape) == 2:
                link_id, point_id = np.unravel_index(int(flat_point_id), link_shape)
            else:
                link_id, point_id = 0, int(flat_point_id)
            constraints.append(PointFlowConstraint(normal.astype(np.float32), current_h, predicted_h, int(link_id), int(point_id), box_id, int(time_index)))
    return sorted(constraints, key=lambda item: min(item.current_h, item.predicted_h))[: int(max_constraints)]


def select_inter_arm_constraints(
    current_points: np.ndarray,
    predicted_points: np.ndarray,
    left_point_mask: np.ndarray,
    *,
    minimum_distance_m: float = 0.035,
    trigger_margin_m: float = 0.02,
    max_constraints: int = 24,
) -> list[PointPairFlowConstraint]:
    """Select predicted left/right point pairs that violate arm separation."""
    from scipy.spatial import cKDTree

    current = np.asarray(current_points, dtype=np.float32).reshape(-1, 3)
    predicted = np.asarray(predicted_points, dtype=np.float32).reshape(len(predicted_points), -1, 3)
    left_mask = np.asarray(left_point_mask, dtype=bool).reshape(-1)
    if len(current) != len(left_mask) or predicted.shape[1:] != current.shape:
        raise ValueError("Inter-arm point arrays and left_point_mask do not align")
    left_ids, right_ids = np.flatnonzero(left_mask), np.flatnonzero(~left_mask)
    if not len(left_ids) or not len(right_ids):
        return []
    threshold = float(minimum_distance_m + trigger_margin_m)
    candidates: dict[tuple[int, int], PointPairFlowConstraint] = {}
    for time_id, points in enumerate(predicted):
        neighbours = cKDTree(points[right_ids]).query_ball_point(points[left_ids], r=threshold)
        for local_left, local_rights in enumerate(neighbours):
            first = int(left_ids[local_left])
            for local_right in local_rights:
                second = int(right_ids[local_right])
                delta = points[first] - points[second]
                distance = float(np.linalg.norm(delta))
                current_delta = current[first] - current[second]
                current_distance = float(np.linalg.norm(current_delta))
                normal = delta / max(distance, 1e-8)
                item = PointPairFlowConstraint(
                    normal.astype(np.float32), current_distance - minimum_distance_m,
                    distance - minimum_distance_m, first, second, time_id,
                )
                key = (first, second)
                if key not in candidates or item.predicted_h < candidates[key].predicted_h:
                    candidates[key] = item
    return sorted(candidates.values(), key=lambda item: min(item.current_h, item.predicted_h))[: int(max_constraints)]


def project_joint_delta_qp(
    nominal_delta: np.ndarray,
    jacobian: np.ndarray,
    constraints: list[PointFlowConstraint | PointPairFlowConstraint],
    *,
    lower_delta: np.ndarray,
    upper_delta: np.ndarray,
    alpha: float = 1.0,
    iterations: int = 32,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Solve ``min 0.5||dq-dq_nom||²`` with bounds and linear CBF constraints."""
    nominal = np.asarray(nominal_delta, dtype=np.float64).reshape(-1)
    low = np.asarray(lower_delta, dtype=np.float64).reshape(-1)
    high = np.asarray(upper_delta, dtype=np.float64).reshape(-1)
    if nominal.shape != low.shape or nominal.shape != high.shape:
        raise ValueError("nominal and joint-delta bounds must have the same shape")
    if np.any(low > high):
        raise ValueError("joint-delta lower bounds exceed upper bounds")
    delta = np.clip(nominal, low, high)
    if not constraints:
        return delta.astype(np.float32), {"triggered": False, "success": True, "constraint_count": 0, "max_violation": 0.0}
    rows, rhs = [], []
    for item in constraints:
        if isinstance(item, PointPairFlowConstraint):
            flat_jacobian = np.asarray(jacobian, dtype=np.float64).reshape(-1, 3, nominal.size)
            relative = flat_jacobian[item.first_point_id] - flat_jacobian[item.second_point_id]
            grad = np.asarray(item.normal, dtype=np.float64) @ relative
        else:
            if np.asarray(jacobian).ndim == 3:
                point_jacobian = jacobian[item.point_id]
            else:
                point_jacobian = jacobian[item.link_id, item.point_id]
            grad = np.asarray(item.normal, dtype=np.float64) @ np.asarray(point_jacobian, dtype=np.float64)
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
    from scipy.optimize import Bounds, LinearConstraint, minimize

    result = minimize(
        fun=lambda value: 0.5 * float((value - nominal) @ (value - nominal)),
        x0=delta,
        jac=lambda value: value - nominal,
        method="SLSQP",
        bounds=Bounds(low, high),
        constraints=(LinearConstraint(matrix, rhs_array, np.full_like(rhs_array, np.inf)),),
        options={"maxiter": max(int(iterations), 1), "ftol": 1e-10, "disp": False},
    )
    delta = np.asarray(result.x, dtype=np.float64)
    violations = rhs_array - matrix @ delta
    max_violation = max(float(np.max(violations)), 0.0)
    return delta.astype(np.float32), {
        "triggered": True,
        "success": bool(result.success) and max_violation <= 1e-4,
        "constraint_count": len(rows),
        "max_violation": max_violation,
        "solver": "scipy_slsqp_qp",
        "solver_iterations": int(result.nit),
    }
