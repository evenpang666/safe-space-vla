"""Legacy/residual robot swept-pointcloud models.

The preferred safety architecture is deterministic robot geometry first:
integrate the action chunk to future joint states, run FK/URDF geometry, and
produce robot point flow with shape ``(B, H, R, 3)``. The learned models in this
file are kept for compatibility and for residual correction when the nominal FK
trajectory has delay, compliance, controller, or calibration error.

For legacy OpenPI integration, an exported TorchScript module may still expose:

    model(state, actions) -> points

where ``points`` has shape ``(B, N, 3)`` or ``(B, H, N, 3)`` in world
coordinates. New learned scene dynamics should use ``PointWorldModel`` instead
of learning robot geometry from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RobotSweptPointCloudModel(nn.Module):
    """Legacy MLP point-set predictor for robot swept volumes.

    The model consumes the current low-dimensional robot state and a future
    action chunk, then predicts a fixed-size point set approximating all robot
    geometry swept by that chunk. Prefer deterministic FK for the primary robot
    geometry layer; use this only as a baseline or when no geometry source is
    available.
    """

    def __init__(
        self,
        state_dim: int,
        action_horizon: int,
        action_dim: int,
        num_points: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_horizon <= 0 or action_dim <= 0 or num_points <= 0:
            raise ValueError("state_dim, action_horizon, action_dim, and num_points must be positive")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.state_dim = int(state_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.num_points = int(num_points)

        input_dim = self.state_dim + self.action_horizon * self.action_dim
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, self.num_points * 3))
        self.net = nn.Sequential(*layers)

        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_horizon, self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_horizon, self.action_dim))
        self.register_buffer("point_mean", torch.zeros(3))
        self.register_buffer("point_std", torch.ones(3))

    def set_normalization(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        point_mean: torch.Tensor,
        point_std: torch.Tensor,
    ) -> None:
        """Copy normalization statistics into buffers."""

        with torch.no_grad():
            self.state_mean.copy_(state_mean.reshape(self.state_dim))
            self.state_std.copy_(torch.clamp(state_std.reshape(self.state_dim), min=1e-6))
            self.action_mean.copy_(action_mean.reshape(self.action_horizon, self.action_dim))
            self.action_std.copy_(torch.clamp(action_std.reshape(self.action_horizon, self.action_dim), min=1e-6))
            self.point_mean.copy_(point_mean.reshape(3))
            self.point_std.copy_(torch.clamp(point_std.reshape(3), min=1e-6))

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state = state[..., : self.state_dim]
        actions = actions[..., : self.action_horizon, : self.action_dim]

        norm_state = (state - self.state_mean.to(device=state.device, dtype=state.dtype)) / self.state_std.to(
            device=state.device, dtype=state.dtype
        )
        norm_actions = (
            actions - self.action_mean.to(device=actions.device, dtype=actions.dtype)
        ) / self.action_std.to(device=actions.device, dtype=actions.dtype)

        flat_actions = norm_actions.reshape(norm_actions.shape[0], self.action_horizon * self.action_dim)
        features = torch.cat((norm_state, flat_actions), dim=-1)
        norm_points = self.net(features).reshape(features.shape[0], self.num_points, 3)
        return norm_points * self.point_std.to(device=norm_points.device, dtype=norm_points.dtype) + self.point_mean.to(
            device=norm_points.device, dtype=norm_points.dtype
        )


class ResidualRobotPointFlowModel(nn.Module):
    """Learn small residual corrections on top of deterministic FK points.

    This implements the recommended fallback for systems where nominal FK is
    biased by latency, compliance, controller tracking error, or calibration
    drift:

        corrected_points = fk_points + residual(state, actions, fk_points)

    The residual is intentionally bounded so the deterministic geometry remains
    the dominant source of robot shape and motion.
    """

    def __init__(
        self,
        state_dim: int,
        action_horizon: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        max_residual: float = 0.03,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_horizon <= 0 or action_dim <= 0:
            raise ValueError("state_dim, action_horizon, and action_dim must be positive")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.state_dim = int(state_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.max_residual = float(max_residual)

        input_dim = self.state_dim + self.action_horizon * self.action_dim + 3
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, actions: torch.Tensor, fk_points: torch.Tensor) -> torch.Tensor:
        if fk_points.shape[-1] != 3:
            raise ValueError(f"fk_points last dimension must be 3, got {tuple(fk_points.shape)}")
        state = state[..., : self.state_dim]
        actions = actions[..., : self.action_horizon, : self.action_dim]
        batch = fk_points.shape[0]
        point_shape = fk_points.shape[1:-1]
        flat_points = fk_points.reshape(batch, -1, 3)
        flat_count = flat_points.shape[1]

        flat_actions = actions.reshape(batch, self.action_horizon * self.action_dim)
        global_features = torch.cat((state, flat_actions), dim=-1)
        global_features = global_features[:, None].expand(batch, flat_count, global_features.shape[-1])
        residual_input = torch.cat((global_features, flat_points), dim=-1)
        residual = torch.tanh(self.net(residual_input)) * self.max_residual
        return (flat_points + residual).reshape(batch, *point_shape, 3)


def chamfer_distance(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    target_coverage_weight: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric-weighted Chamfer distance for conservative coverage.

    ``target_coverage_weight > 1`` makes missed ground-truth swept points more
    expensive than predicting extra points, which is the safer error profile for
    collision avoidance.
    """

    distances = torch.cdist(pred_points, target_points)
    pred_to_target = distances.min(dim=2).values.pow(2).mean()
    target_to_pred = distances.min(dim=1).values.pow(2).mean()
    loss = pred_to_target + float(target_coverage_weight) * target_to_pred
    return loss, pred_to_target, target_to_pred
