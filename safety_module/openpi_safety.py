"""Adapters for adding deterministic robot point-flow safety losses to OpenPI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .safety_loss import SafetyLoss


def _stats_dim(stats: Any) -> int:
    if getattr(stats, "mean", None) is not None:
        return int(np.asarray(stats.mean).shape[-1])
    if getattr(stats, "q01", None) is not None:
        return int(np.asarray(stats.q01).shape[-1])
    raise ValueError("normalization stats must contain mean/std or q01/q99")


def _as_tensor(value: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def unnormalize_tensor(
    tensor: torch.Tensor,
    stats: Any,
    use_quantiles: bool,
    real_dim: Optional[int] = None,
) -> torch.Tensor:
    """Torch equivalent of openpi.transforms.Unnormalize for the leading real dimensions."""

    if stats is None:
        return tensor

    dim = _stats_dim(stats) if real_dim is None else min(real_dim, _stats_dim(stats))
    dim = min(dim, tensor.shape[-1])
    if dim <= 0:
        return tensor

    out = tensor.clone()
    target = out[..., :dim]
    if use_quantiles:
        q01 = _as_tensor(np.asarray(stats.q01)[..., :dim], tensor.device, tensor.dtype)
        q99 = _as_tensor(np.asarray(stats.q99)[..., :dim], tensor.device, tensor.dtype)
        out[..., :dim] = (target + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    else:
        mean = _as_tensor(np.asarray(stats.mean)[..., :dim], tensor.device, tensor.dtype)
        std = _as_tensor(np.asarray(stats.std)[..., :dim], tensor.device, tensor.dtype)
        out[..., :dim] = target * (std + 1e-6) + mean
    return out


def make_fibonacci_sphere_offsets(num_points: int, radius: float) -> torch.Tensor:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    indices = torch.arange(num_points, dtype=torch.float32)
    golden_angle = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))
    z = 1.0 - 2.0 * (indices + 0.5) / num_points
    radial = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = golden_angle * indices
    unit = torch.stack((torch.cos(theta) * radial, torch.sin(theta) * radial, z), dim=-1)
    return unit * float(radius)


class EefSpherePointCloud(nn.Module):
    """Simple differentiable LIBERO baseline: swept spheres around predicted EEF positions.

    This constrains the end-effector swept volume, not the full robot body. Use
    an FK/URDF-based robot point-flow module for the full-arm safety constraint.
    """

    def __init__(self, radius: float = 0.05, num_points: int = 64, action_scale: float = 1.0) -> None:
        super().__init__()
        self.action_scale = float(action_scale)
        offsets = make_fibonacci_sphere_offsets(num_points, radius)
        self.register_buffer("offsets", offsets)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] < 3:
            raise ValueError(f"state must contain eef xyz in the first 3 dims, got {tuple(state.shape)}")
        if actions.shape[-1] < 3:
            raise ValueError(f"actions must contain delta xyz in the first 3 dims, got {tuple(actions.shape)}")
        eef0 = state[..., :3]
        delta_xyz = actions[..., :3] * self.action_scale
        eef_positions = eef0[:, None, :] + torch.cumsum(delta_xyz, dim=1)
        return eef_positions[:, :, None, :] + self.offsets.to(device=actions.device, dtype=actions.dtype)


class TorchscriptRobotPointCloud(nn.Module):
    """Load a deterministic FK robot point-flow module saved with torch.jit.save.

    The loaded module must implement ``model(state, actions) -> points``. In the
    recommended architecture this module is FK/action integration, optionally
    with a small residual correction. Returned points may be shaped ``(B, N, 3)``
    or ``(B, H, N, 3)``; any tensor whose last dimension is world xyz is accepted
    by ``SafetyLoss``.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__()
        self.model = torch.jit.load(str(path))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        points = self.model(state, actions)
        if isinstance(points, dict):
            points = points["points"]
        if points.shape[-1] != 3:
            raise ValueError(f"robot point model must return (..., 3), got {tuple(points.shape)}")
        return points


class TorchscriptSafetyCritic(nn.Module):
    """Load a TorchScript collision critic saved from ``PointCloudSafetyCritic``."""

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__()
        self.model = torch.jit.load(str(path))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(
        self,
        scene_points: torch.Tensor,
        robot_point_flow: torch.Tensor,
        scene_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if scene_features is None:
            outputs = self.model(scene_points, robot_point_flow)
        else:
            outputs = self.model(scene_points, robot_point_flow, scene_features)
        if isinstance(outputs, torch.Tensor):
            outputs = {"cost": outputs}
        return outputs


class OpenPISafetyLoss(nn.Module):
    """Combine OpenPI unnormalization, deterministic robot point flow, and SDF loss."""

    def __init__(
        self,
        sdf_path: Union[str, Path],
        data_config: Any,
        margin: float = 0.03,
        robot_pointcloud_model_path: Optional[Union[str, Path]] = None,
        robot_pointcloud_mode: str = "torchscript",
        eef_sphere_radius: float = 0.05,
        eef_sphere_points: int = 64,
        eef_action_scale: float = 1.0,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.sdf_loss = SafetyLoss.from_npz(sdf_path, margin=margin, device=device)
        self.use_quantiles = bool(getattr(data_config, "use_quantile_norm", False))
        norm_stats = getattr(data_config, "norm_stats", None) or {}
        self.state_stats = norm_stats.get("state") if isinstance(norm_stats, dict) else None
        self.action_stats = norm_stats.get("actions") if isinstance(norm_stats, dict) else None
        self.state_dim = state_dim
        self.action_dim = action_dim

        if robot_pointcloud_mode == "torchscript":
            if robot_pointcloud_model_path is None:
                raise ValueError("robot_pointcloud_model_path is required when robot_pointcloud_mode='torchscript'")
            self.robot_pointcloud = TorchscriptRobotPointCloud(robot_pointcloud_model_path)
        elif robot_pointcloud_mode == "eef_sphere":
            self.robot_pointcloud = EefSpherePointCloud(
                radius=eef_sphere_radius,
                num_points=eef_sphere_points,
                action_scale=eef_action_scale,
            )
        else:
            raise ValueError(f"unsupported robot_pointcloud_mode: {robot_pointcloud_mode}")

    def forward(self, normalized_state: torch.Tensor, normalized_actions: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        state = unnormalize_tensor(
            normalized_state,
            self.state_stats,
            self.use_quantiles,
            real_dim=self.state_dim,
        )
        actions = unnormalize_tensor(
            normalized_actions,
            self.action_stats,
            self.use_quantiles,
            real_dim=self.action_dim,
        )
        robot_points = self.robot_pointcloud(state, actions)
        loss, info = self.sdf_loss(robot_points, return_info=True)
        info["robot_points"] = robot_points.shape[-2] if robot_points.ndim >= 3 else robot_points.numel() // 3
        return loss, info


class OpenPICollisionCritic(nn.Module):
    """OpenPI adapter for action-chunk collision cost.

    This module is used after a VLA/action head proposes a chunk. It converts
    normalized state/actions back to robot units, generates deterministic robot
    point flow, and evaluates the learned/geometric collision critic against the
    current non-robot scene point cloud.
    """

    def __init__(
        self,
        data_config: Any,
        robot_pointcloud_model_path: Union[str, Path],
        critic_model_path: Union[str, Path],
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        cost_weight: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.use_quantiles = bool(getattr(data_config, "use_quantile_norm", False))
        norm_stats = getattr(data_config, "norm_stats", None) or {}
        self.state_stats = norm_stats.get("state") if isinstance(norm_stats, dict) else None
        self.action_stats = norm_stats.get("actions") if isinstance(norm_stats, dict) else None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cost_weight = float(cost_weight)
        self.robot_pointcloud = TorchscriptRobotPointCloud(robot_pointcloud_model_path)
        self.critic = TorchscriptSafetyCritic(critic_model_path)
        if device is not None:
            self.to(device=device)

    def forward(
        self,
        normalized_state: torch.Tensor,
        normalized_actions: torch.Tensor,
        scene_points: torch.Tensor,
        scene_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        state = unnormalize_tensor(
            normalized_state,
            self.state_stats,
            self.use_quantiles,
            real_dim=self.state_dim,
        )
        actions = unnormalize_tensor(
            normalized_actions,
            self.action_stats,
            self.use_quantiles,
            real_dim=self.action_dim,
        )
        robot_point_flow = self.robot_pointcloud(state, actions)
        outputs = self.critic(scene_points, robot_point_flow, scene_features)
        cost = outputs["cost"].mean() * self.cost_weight
        info: Dict[str, Any] = {
            "critic_cost": outputs["cost"].detach().mean(),
            "robot_point_flow": robot_point_flow.shape,
        }
        for key in ("collision_probability", "min_distance", "uncertainty_cost", "geometric_cost"):
            value = outputs.get(key)
            if value is not None:
                info[key] = value.detach().mean()
        return cost, info
