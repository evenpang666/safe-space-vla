"""Small safe-RL utilities for VLA residual post-training.

These modules are intentionally VLA-agnostic. They operate on action chunks,
log-probabilities, rewards, and safety costs so they can be used with OpenPI,
LoRA adapters, or a separate residual action head.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualActionCorrection(nn.Module):
    """Bounded residual policy head: ``a_safe = a_vla + delta``.

    The caller supplies context features from the VLA, robot state, language
    embedding, or risk heatmap encoder. Keeping this as a residual head lets the
    original SFT/IL policy remain mostly intact.
    """

    def __init__(
        self,
        context_dim: int,
        horizon: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        max_delta: float = 0.05,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.max_delta = float(max_delta)

        layers: list[nn.Module] = []
        dim = int(context_dim)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, self.horizon * self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, base_actions: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if base_actions.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"base_actions must end with {(self.horizon, self.action_dim)}, got {tuple(base_actions.shape)}"
            )
        residual = torch.tanh(self.net(context)).reshape(*base_actions.shape[:-2], self.horizon, self.action_dim)
        residual = residual * self.max_delta
        return base_actions + residual, residual


class LagrangeMultiplier(nn.Module):
    """Non-negative multiplier for CMDP-style safety constraints."""

    def __init__(self, initial_value: float = 1.0, max_value: Optional[float] = None) -> None:
        super().__init__()
        if initial_value < 0.0:
            raise ValueError("initial_value must be non-negative")
        self.raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(initial_value))) + 1e-6))
        self.max_value = None if max_value is None else float(max_value)

    def forward(self) -> torch.Tensor:
        value = F.softplus(self.raw)
        if self.max_value is not None:
            value = torch.clamp(value, max=self.max_value)
        return value

    @torch.no_grad()
    def manual_update(self, mean_cost: torch.Tensor, cost_limit: float, lr: float) -> torch.Tensor:
        """Projected dual ascent update: lambda <- [lambda + lr(C-eps)]_+."""

        current = self.forward()
        updated = torch.clamp(current + float(lr) * (mean_cost.detach() - float(cost_limit)), min=0.0)
        if self.max_value is not None:
            updated = torch.clamp(updated, max=self.max_value)
        self.raw.copy_(torch.log(torch.expm1(updated) + 1e-6))
        return updated


def ppo_lagrangian_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    task_advantage: torch.Tensor,
    safety_cost: torch.Tensor,
    lagrange_multiplier: torch.Tensor,
    cost_limit: float,
    reference_log_prob: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2,
    kl_weight: float = 0.01,
    entropy_weight: float = 0.0,
    normalize_advantage: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """PPO-style objective with a Lagrangian safety penalty.

    This is the scalar objective for updating an action head, LoRA, or residual
    policy. ``reference_log_prob`` should come from the frozen SFT VLA when
    available; it keeps RL from erasing the original skill.
    """

    task_advantage = task_advantage.to(new_log_prob)
    safety_cost = safety_cost.to(new_log_prob)
    safety_advantage = safety_cost - float(cost_limit)
    advantage = task_advantage - lagrange_multiplier.detach().to(new_log_prob) * safety_advantage
    if normalize_advantage and advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / torch.clamp(advantage.std(unbiased=False), min=1e-6)

    ratio = torch.exp(new_log_prob - old_log_prob.to(new_log_prob))
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio)) * advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    kl_loss = new_log_prob.new_tensor(0.0)
    approx_kl = (old_log_prob.to(new_log_prob) - new_log_prob).mean()
    if reference_log_prob is not None and kl_weight:
        kl_loss = (new_log_prob - reference_log_prob.to(new_log_prob)).mean()
        policy_loss = policy_loss + float(kl_weight) * kl_loss
    if entropy is not None and entropy_weight:
        policy_loss = policy_loss - float(entropy_weight) * entropy.to(new_log_prob).mean()

    metrics: Dict[str, Any] = {
        "loss": policy_loss.detach(),
        "mean_task_advantage": task_advantage.mean().detach(),
        "mean_safety_cost": safety_cost.mean().detach(),
        "mean_safety_violation": safety_advantage.mean().detach(),
        "lagrange_multiplier": lagrange_multiplier.detach(),
        "approx_kl_old": approx_kl.detach(),
        "reference_kl_proxy": kl_loss.detach(),
        "clip_fraction": ((ratio - 1.0).abs() > float(clip_ratio)).to(torch.float32).mean().detach(),
    }
    return policy_loss, metrics
