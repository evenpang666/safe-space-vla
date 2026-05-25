"""Differentiable joint-space swept-surface loss against obstacle OBBs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_UR5E_LINK_RADII = (0.045, 0.045, 0.04, 0.035, 0.03, 0.03, 0.025)


def load_obstacle_boxes(
    path: Union[str, Path],
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    """Load OBB fields from a safe-space .npz or a minimal boxes .npz."""

    data = np.load(path)
    center_key = "obstacle_box_centers" if "obstacle_box_centers" in data else "centers"
    axes_key = "obstacle_box_axes" if "obstacle_box_axes" in data else "axes"
    half_key = "obstacle_box_half_sizes" if "obstacle_box_half_sizes" in data else "half_sizes"
    required = (center_key, axes_key, half_key)
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing obstacle box arrays: {missing}")

    centers = np.asarray(data[center_key], dtype=np.float32)
    axes = np.asarray(data[axes_key], dtype=np.float32)
    half_sizes = np.asarray(data[half_key], dtype=np.float32)
    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError(f"box centers must have shape (M, 3), got {centers.shape}")
    if axes.shape != (centers.shape[0], 3, 3):
        raise ValueError(f"box axes must have shape (M, 3, 3), got {axes.shape}")
    if half_sizes.shape != centers.shape:
        raise ValueError(f"box half_sizes must have shape (M, 3), got {half_sizes.shape}")

    if indices:
        idx = np.asarray(indices, dtype=np.int64)
        centers = centers[idx]
        axes = axes[idx]
        half_sizes = half_sizes[idx]
    if centers.shape[0] == 0:
        raise ValueError("at least one obstacle box is required")

    return {"centers": centers, "axes": axes, "half_sizes": half_sizes}


def parse_box_indices(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    if value is None or value.strip() == "":
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _quat_to_rot(quat_wxyz: torch.Tensor) -> torch.Tensor:
    q = quat_wxyz / torch.clamp(torch.linalg.norm(quat_wxyz), min=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    two = q.new_tensor(2.0)
    return torch.stack(
        (
            torch.stack((1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)), dim=-1),
            torch.stack((two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)), dim=-1),
            torch.stack((two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def _axis_angle_to_rot(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-8)
    x, y, z = axis.unbind(dim=-1)
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c
    return torch.stack(
        (
            torch.stack((c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s), dim=-1),
            torch.stack((y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s), dim=-1),
            torch.stack((z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c), dim=-1),
        ),
        dim=-2,
    )


class UR5eSweptSegments(nn.Module):
    """Pure-Torch robosuite-style UR5e FK to seven swept line segments."""

    def __init__(
        self,
        base_position: Sequence[float] = (-0.61, 0.0, 0.912),
        gripper_width: float = 0.085,
    ) -> None:
        super().__init__()
        self.gripper_width = float(gripper_width)
        self.register_buffer("base_position", torch.tensor(base_position, dtype=torch.float32))
        self.register_buffer(
            "body_pos",
            torch.tensor(
                [
                    [0.0, 0.0, 0.163],
                    [0.0, 0.138, 0.0],
                    [0.0, -0.131, 0.425],
                    [0.0, 0.0, 0.392],
                    [0.0, 0.127, 0.0],
                    [0.0, 0.0, 0.1],
                    [0.0, 0.098, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.145],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "body_quat",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.70710678, 0.0, 0.70710678, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.70710678, 0.0, 0.70710678, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.70710678, -0.70710678, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.70710678, 0.0, 0.0, -0.70710678],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "joint_axes",
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, joint_path: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``segment_path`` and ``anchor_path``.

        Args:
            joint_path: ``(B, T, 6)`` UR5e joint angles in radians.
        """

        if joint_path.shape[-1] < 6:
            raise ValueError(f"joint_path must contain 6 joints, got {tuple(joint_path.shape)}")
        joint_path = joint_path[..., :6]
        batch_shape = joint_path.shape[:-1]
        flat_q = joint_path.reshape(-1, 6)
        n = flat_q.shape[0]
        dtype = flat_q.dtype
        device = flat_q.device

        p = self.base_position.to(device=device, dtype=dtype).expand(n, 3)
        r = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
        anchors = []
        eef_rot = None

        for body_idx in range(9):
            pos = self.body_pos[body_idx].to(device=device, dtype=dtype)
            fixed_rot = _quat_to_rot(self.body_quat[body_idx].to(device=device, dtype=dtype))
            p = p + torch.einsum("bij,j->bi", r, pos)
            r = torch.matmul(r, fixed_rot.expand(n, 3, 3))
            if body_idx < 6:
                axis = self.joint_axes[body_idx].to(device=device, dtype=dtype)
                joint_rot = _axis_angle_to_rot(axis, flat_q[:, body_idx])
                r = torch.matmul(r, joint_rot)
            if body_idx < 6 or body_idx == 8:
                anchors.append(p)
            if body_idx == 8:
                eef_rot = r

        anchor_path = torch.stack(anchors, dim=1).reshape(*batch_shape, 7, 3)
        eef_rot = eef_rot.reshape(*batch_shape, 3, 3)

        segments = []
        for link_idx in range(6):
            segments.append(torch.stack((anchor_path[..., link_idx, :], anchor_path[..., link_idx + 1, :]), dim=-2))

        eef = anchor_path[..., -1, :]
        gripper_x = F.normalize(eef_rot[..., :, 0], dim=-1)
        half = 0.5 * self.gripper_width
        gripper_segment = torch.stack((eef - half * gripper_x, eef + half * gripper_x), dim=-2)
        segments.append(gripper_segment)
        segment_path = torch.stack(segments, dim=-3)
        return segment_path, anchor_path


class JointSweptBoxSafetyLoss(nn.Module):
    """Penalize swept UR5e link surfaces that approach obstacle OBBs."""

    def __init__(
        self,
        obstacle_box_path: Union[str, Path],
        obstacle_box_indices: Optional[Sequence[int]] = None,
        margin: float = 0.04,
        action_scale: float = 1.0,
        surface_samples: int = 5,
        topk: int = 128,
        gripper_width: float = 0.085,
        base_position: Sequence[float] = (-0.61, 0.0, 0.912),
        link_radii: Sequence[float] = DEFAULT_UR5E_LINK_RADII,
        power: float = 2.0,
    ) -> None:
        super().__init__()
        boxes = load_obstacle_boxes(obstacle_box_path, obstacle_box_indices)
        if surface_samples < 2:
            raise ValueError("surface_samples must be >= 2")
        if len(link_radii) != 7:
            raise ValueError(f"link_radii must contain 7 values, got {len(link_radii)}")
        self.margin = float(margin)
        self.action_scale = float(action_scale)
        self.surface_samples = int(surface_samples)
        self.topk = int(topk)
        self.power = float(power)
        self.fk = UR5eSweptSegments(base_position=base_position, gripper_width=gripper_width)
        self.register_buffer("box_centers", torch.as_tensor(boxes["centers"], dtype=torch.float32))
        self.register_buffer("box_axes", torch.as_tensor(boxes["axes"], dtype=torch.float32))
        self.register_buffer("box_half_sizes", torch.as_tensor(boxes["half_sizes"], dtype=torch.float32))
        self.register_buffer("link_radii", torch.as_tensor(link_radii, dtype=torch.float32))
        grid = torch.linspace(0.0, 1.0, self.surface_samples, dtype=torch.float32)
        vv, uu = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("surface_u", uu.reshape(-1))
        self.register_buffer("surface_v", vv.reshape(-1))

    def integrate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] < 6:
            raise ValueError(f"state must contain 6 joint angles, got {tuple(state.shape)}")
        if actions.shape[-1] < 6:
            raise ValueError(f"actions must contain 6 joint deltas, got {tuple(actions.shape)}")
        q0 = state[..., :6].to(dtype=actions.dtype)
        dq = actions[..., :6] * self.action_scale
        q_future = q0[:, None, :] + torch.cumsum(dq, dim=1)
        return torch.cat((q0[:, None, :], q_future), dim=1)

    def swept_surface_points(self, segment_path: torch.Tensor) -> torch.Tensor:
        s0 = segment_path[:, :-1]
        s1 = segment_path[:, 1:]
        p00 = s0[..., 0, :]
        p01 = s0[..., 1, :]
        p10 = s1[..., 0, :]
        p11 = s1[..., 1, :]
        u = self.surface_u.to(device=segment_path.device, dtype=segment_path.dtype).view(1, 1, 1, -1, 1)
        v = self.surface_v.to(device=segment_path.device, dtype=segment_path.dtype).view(1, 1, 1, -1, 1)
        points = (
            (1.0 - u) * (1.0 - v) * p00.unsqueeze(-2)
            + u * (1.0 - v) * p01.unsqueeze(-2)
            + (1.0 - u) * v * p10.unsqueeze(-2)
            + u * v * p11.unsqueeze(-2)
        )
        return points

    def box_sdf(self, points: torch.Tensor) -> torch.Tensor:
        centers = self.box_centers.to(device=points.device, dtype=points.dtype)
        axes = self.box_axes.to(device=points.device, dtype=points.dtype)
        half_sizes = self.box_half_sizes.to(device=points.device, dtype=points.dtype)
        delta = points.unsqueeze(-2) - centers
        local = torch.einsum("...mi,mij->...mj", delta, axes)
        q = torch.abs(local) - half_sizes
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(q.amax(dim=-1), max=0.0)
        return outside + inside

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        joint_path = self.integrate_actions(state, actions)
        segment_path, _ = self.fk(joint_path)
        swept_points = self.swept_surface_points(segment_path)
        sdf = self.box_sdf(swept_points)
        radii = self.link_radii.to(device=actions.device, dtype=actions.dtype).view(1, 1, 7, 1, 1)
        clearance = sdf - radii
        violation = torch.relu(self.margin - clearance)
        per_sample = violation if self.power == 1.0 else violation.pow(self.power)
        flat = per_sample.reshape(per_sample.shape[0], -1)
        if self.topk > 0 and flat.shape[1] > self.topk:
            loss = torch.topk(flat, k=self.topk, dim=1).values.mean()
        else:
            loss = flat.mean()

        if not return_info:
            return loss

        with torch.no_grad():
            min_clearance = clearance.amin()
            info: Dict[str, Any] = {
                "min_sdf": min_clearance,
                "mean_sdf": clearance.mean(),
                "unsafe_ratio": (clearance < 0.0).to(torch.float32).mean(),
                "margin_violation_ratio": (clearance < self.margin).to(torch.float32).mean(),
                "max_violation": violation.amax(),
                "min_clearance": min_clearance,
                "num_obstacle_boxes": torch.as_tensor(self.box_centers.shape[0], device=actions.device),
                "swept_points": torch.as_tensor(swept_points.numel() // 3, device=actions.device),
            }
        return loss, info
