"""Collision critic and runtime shield utilities for VLA action chunks.

The critic consumes the deterministic robot point flow produced by FK/simulator
geometry plus the current scene point cloud. It is intended as a geometric
safety cost model for action reranking, shielding, and constrained RL
post-training of an already fine-tuned VLA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .point_world_model import _mlp, normalize_robot_point_flow


TensorDict = Dict[str, torch.Tensor]


def normalize_scene_point_flow(scene_points: torch.Tensor, horizon: int) -> torch.Tensor:
    """Return scene points as ``(B, H, N, 3)``.

    ``(B, N, 3)`` is treated as a static scene repeated over the action chunk.
    ``(B, H, N, 3)`` can be used for dynamic obstacles or point-world predicted
    future scene points.
    """

    if scene_points.shape[-1] != 3:
        raise ValueError(f"scene_points last dim must be 3, got {tuple(scene_points.shape)}")
    if scene_points.ndim == 3:
        return scene_points[:, None].expand(scene_points.shape[0], horizon, scene_points.shape[1], 3)
    if scene_points.ndim == 4:
        if scene_points.shape[1] != horizon:
            raise ValueError(f"scene_points horizon {scene_points.shape[1]} does not match robot horizon {horizon}")
        return scene_points
    raise ValueError(f"scene_points must have shape (B, N, 3) or (B, H, N, 3), got {tuple(scene_points.shape)}")


def normalize_point_mask(mask: Optional[torch.Tensor], scene_points: torch.Tensor, horizon: int) -> Optional[torch.Tensor]:
    """Return an optional point mask as ``(B, H, N)``."""

    if mask is None:
        return None
    if mask.ndim == 2:
        return mask[:, None].expand(mask.shape[0], horizon, mask.shape[1])
    if mask.ndim == 3:
        if mask.shape[1] != horizon:
            raise ValueError(f"mask horizon {mask.shape[1]} does not match robot horizon {horizon}")
        return mask
    raise ValueError(f"mask must have shape (B, N) or (B, H, N), got {tuple(mask.shape)}")


def scene_collision_weights(
    scene_points: torch.Tensor,
    horizon: int,
    forbidden_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    forbidden_weight: float = 4.0,
    target_contact_weight: float = 0.0,
) -> torch.Tensor:
    """Build per-scene-point collision weights.

    Target points can be down-weighted or ignored so manipulation contacts are
    not treated the same as forbidden contacts. Forbidden points receive a
    stronger weight because false negatives are more costly there.
    """

    base = scene_points if scene_points.ndim == 3 else scene_points[:, 0]
    weights = torch.ones(base.shape[:2], device=base.device, dtype=base.dtype)

    target = normalize_point_mask(target_mask, base, horizon) if target_mask is not None else None
    forbidden = normalize_point_mask(forbidden_mask, base, horizon) if forbidden_mask is not None else None
    weights = weights[:, None].expand(base.shape[0], horizon, base.shape[1]).clone()

    if target is not None:
        weights = torch.where(target > 0.5, weights.new_full((), float(target_contact_weight)), weights)
    if forbidden is not None:
        weights = weights + (forbidden > 0.5).to(weights.dtype) * float(forbidden_weight)
    return weights


def scene_robot_distances(
    scene_points: torch.Tensor,
    robot_point_flow: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute nearest distances between scene and robot points.

    Returns:
        scene_to_robot: ``(B, H, N)`` nearest robot distance for each scene point.
        robot_to_scene: ``(B, H, R)`` nearest scene distance for each robot point.
    """

    robot_point_flow = normalize_robot_point_flow(robot_point_flow)
    batch, horizon, robot_count, _ = robot_point_flow.shape
    scene_flow = normalize_scene_point_flow(scene_points, horizon)
    scene_count = scene_flow.shape[2]

    flat_scene = scene_flow.reshape(batch * horizon, scene_count, 3)
    flat_robot = robot_point_flow.reshape(batch * horizon, robot_count, 3)
    distances = torch.cdist(flat_scene, flat_robot)
    scene_to_robot = distances.amin(dim=2).reshape(batch, horizon, scene_count)
    robot_to_scene = distances.amin(dim=1).reshape(batch, horizon, robot_count)
    return scene_to_robot, robot_to_scene


def masked_min_distance(distances: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    masked = torch.where(weights > 0.0, distances, distances.new_full((), 1e6))
    min_distance = masked.amin(dim=-1)
    no_valid_points = (weights > 0.0).sum(dim=-1) == 0
    return torch.where(no_valid_points, distances.new_full(min_distance.shape, 1e6), min_distance)


def geometric_safety_cost(
    scene_points: torch.Tensor,
    robot_point_flow: torch.Tensor,
    safe_distance: float = 0.03,
    near_distance: float = 0.08,
    temperature: float = 0.01,
    forbidden_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    collision_weight: float = 1.0,
    near_weight: float = 0.25,
    forbidden_weight: float = 4.0,
    target_contact_weight: float = 0.0,
    reduce_horizon: str = "sum",
) -> TensorDict:
    """Differentiable geometric collision cost for one action chunk.

    The returned ``cost`` is per sample ``(B,)``. ``risk_heatmap`` keeps the
    time dimension and can be rendered or fed back to a policy as safety tokens.
    """

    if reduce_horizon not in {"sum", "mean", "max"}:
        raise ValueError(f"unsupported reduce_horizon: {reduce_horizon}")
    robot_point_flow = normalize_robot_point_flow(robot_point_flow)
    horizon = robot_point_flow.shape[1]
    scene_flow = normalize_scene_point_flow(scene_points, horizon)
    scene_to_robot, robot_to_scene = scene_robot_distances(scene_flow, robot_point_flow)
    weights = scene_collision_weights(
        scene_flow,
        horizon,
        forbidden_mask=forbidden_mask,
        target_mask=target_mask,
        forbidden_weight=forbidden_weight,
        target_contact_weight=target_contact_weight,
    )

    tau = max(float(temperature), 1e-6)
    raw_risk = torch.sigmoid((float(safe_distance) - scene_to_robot) / tau)
    weighted_risk = raw_risk * weights
    point_norm = torch.clamp((weights > 0.0).to(weights.dtype).sum(dim=-1), min=1.0)

    min_distance_per_step = masked_min_distance(scene_to_robot, weights)
    min_distance = min_distance_per_step.amin(dim=-1)
    collision_per_step = weighted_risk.amax(dim=-1)
    near_violation = torch.relu(float(near_distance) - scene_to_robot) * weights
    near_per_step = near_violation.sum(dim=-1) / point_norm

    forbidden = normalize_point_mask(forbidden_mask, scene_flow[:, 0], horizon)
    if forbidden is None:
        forbidden_per_step = collision_per_step.new_zeros(collision_per_step.shape)
    else:
        forbidden_weights = (forbidden > 0.5).to(collision_per_step.dtype)
        denom = torch.clamp(forbidden_weights.sum(dim=-1), min=1.0)
        forbidden_per_step = (raw_risk * forbidden_weights).sum(dim=-1) / denom

    per_step_cost = (
        float(collision_weight) * collision_per_step
        + float(near_weight) * near_per_step
        + float(forbidden_weight) * forbidden_per_step
    )
    if reduce_horizon == "sum":
        cost = per_step_cost.sum(dim=-1)
    elif reduce_horizon == "mean":
        cost = per_step_cost.mean(dim=-1)
    else:
        cost = per_step_cost.amax(dim=-1)

    collision_probability = 1.0 - torch.prod(torch.clamp(1.0 - collision_per_step, min=1e-6, max=1.0), dim=-1)
    return {
        "cost": cost,
        "collision_probability": collision_probability,
        "min_distance": min_distance,
        "min_distance_per_step": min_distance_per_step,
        "collision_per_step": collision_per_step,
        "near_cost_per_step": near_per_step,
        "forbidden_contact_per_step": forbidden_per_step,
        "risk_heatmap": raw_risk,
        "weighted_risk_heatmap": weighted_risk,
        "scene_to_robot_distance": scene_to_robot,
        "robot_to_scene_distance": robot_to_scene,
    }


class PointCloudSafetyCritic(nn.Module):
    """Learned 3D collision critic on top of deterministic robot point flow.

    Outputs:
        collision_logit: chunk-level unsafe logit, ``(B,)``.
        collision_probability: sigmoid of the unsafe logit.
        min_distance: predicted future obstacle distance, ``(B,)``.
        risk_logits: per-point risk logits, ``(B, H, N)``.
        risk_heatmap: per-point risk probabilities, ``(B, H, N)``.
        cost: conservative scalar cost for RL/reranking, ``(B,)``.
    """

    def __init__(
        self,
        scene_feature_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 4,
        safe_distance: float = 0.03,
        near_distance: float = 0.08,
        temperature: float = 0.01,
        forbidden_weight: float = 4.0,
        target_contact_weight: float = 0.0,
        uncertainty_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.scene_feature_dim = int(scene_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.safe_distance = float(safe_distance)
        self.near_distance = float(near_distance)
        self.temperature = float(temperature)
        self.forbidden_weight = float(forbidden_weight)
        self.target_contact_weight = float(target_contact_weight)
        self.uncertainty_weight = float(uncertainty_weight)

        point_input_dim = (
            3
            + self.scene_feature_dim
            + 1  # scene-to-robot min distance
            + 1  # radial proximity kernel
            + 3  # robot centroid relative position
            + 3  # robot centroid chunk displacement
            + 1  # normalized time
            + 2  # forbidden and target masks
        )
        self.point_trunk = _mlp(point_input_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.chunk_head = _mlp(hidden_dim * 2 + 5, hidden_dim, 3, num_layers=3)

    def forward(
        self,
        scene_points: torch.Tensor,
        robot_point_flow: torch.Tensor,
        scene_features: Optional[torch.Tensor] = None,
        forbidden_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        future_scene_points: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        robot_point_flow = normalize_robot_point_flow(robot_point_flow)
        batch, horizon, _, _ = robot_point_flow.shape
        collision_scene = scene_points if future_scene_points is None else future_scene_points
        scene_flow = normalize_scene_point_flow(collision_scene, horizon)
        base_scene = scene_points if scene_points.ndim == 3 else scene_points[:, 0]
        if base_scene.ndim != 3:
            raise ValueError(f"scene_points must have shape (B, N, 3), got {tuple(scene_points.shape)}")
        if base_scene.shape[0] != batch:
            raise ValueError("scene_points and robot_point_flow must have the same batch size")
        scene_count = base_scene.shape[1]

        if scene_features is None:
            if self.scene_feature_dim:
                raise ValueError("scene_features is required when scene_feature_dim > 0")
            scene_features_h = base_scene.new_zeros(batch, horizon, scene_count, 0)
        else:
            if scene_features.shape != (batch, scene_count, self.scene_feature_dim):
                raise ValueError(
                    "scene_features must have shape "
                    f"{(batch, scene_count, self.scene_feature_dim)}, got {tuple(scene_features.shape)}"
                )
            scene_features_h = scene_features[:, None].expand(batch, horizon, scene_count, self.scene_feature_dim)

        scene_to_robot, _ = scene_robot_distances(scene_flow, robot_point_flow)
        radial = torch.exp(-scene_to_robot / max(self.near_distance, 1e-6))
        robot_centroid = robot_point_flow.mean(dim=2)
        robot_delta = robot_centroid - robot_centroid[:, :1]
        rel_centroid = robot_centroid[:, :, None, :] - scene_flow
        rel_delta = robot_delta[:, :, None, :].expand(batch, horizon, scene_count, 3)
        time = torch.linspace(0.0, 1.0, horizon, device=base_scene.device, dtype=base_scene.dtype)
        time = time.view(1, horizon, 1, 1).expand(batch, horizon, scene_count, 1)

        forbidden = normalize_point_mask(forbidden_mask, base_scene, horizon)
        target = normalize_point_mask(target_mask, base_scene, horizon)
        if forbidden is None:
            forbidden = base_scene.new_zeros(batch, horizon, scene_count)
        else:
            forbidden = forbidden.to(device=base_scene.device, dtype=base_scene.dtype)
        if target is None:
            target = base_scene.new_zeros(batch, horizon, scene_count)
        else:
            target = target.to(device=base_scene.device, dtype=base_scene.dtype)

        point_features = torch.cat(
            (
                scene_flow,
                scene_features_h,
                scene_to_robot[..., None],
                radial[..., None],
                rel_centroid,
                rel_delta,
                time,
                forbidden[..., None],
                target[..., None],
            ),
            dim=-1,
        )
        hidden = self.point_trunk(point_features.reshape(batch * horizon * scene_count, -1))
        hidden = hidden.reshape(batch, horizon, scene_count, self.hidden_dim)
        risk_logits = self.risk_head(hidden).squeeze(-1)
        uncertainty = F.softplus(self.uncertainty_head(hidden).squeeze(-1))

        weights = scene_collision_weights(
            base_scene,
            horizon,
            forbidden_mask=forbidden_mask,
            target_mask=target_mask,
            forbidden_weight=self.forbidden_weight,
            target_contact_weight=self.target_contact_weight,
        ).to(device=base_scene.device, dtype=base_scene.dtype)
        masked_hidden = hidden * weights[..., None]
        denom = torch.clamp(weights.sum(dim=(1, 2), keepdim=True), min=1.0)
        mean_pool = masked_hidden.sum(dim=(1, 2)) / denom.squeeze(-1)
        max_pool = hidden.amax(dim=2).amax(dim=1)

        geom = geometric_safety_cost(
            collision_scene,
            robot_point_flow,
            safe_distance=self.safe_distance,
            near_distance=self.near_distance,
            temperature=self.temperature,
            forbidden_mask=forbidden_mask,
            target_mask=target_mask,
            forbidden_weight=self.forbidden_weight,
            target_contact_weight=self.target_contact_weight,
        )
        geom_features = torch.stack(
            (
                geom["cost"],
                geom["collision_probability"],
                geom["min_distance"],
                geom["collision_per_step"].amax(dim=-1),
                geom["near_cost_per_step"].mean(dim=-1),
            ),
            dim=-1,
        )
        chunk_raw = self.chunk_head(torch.cat((mean_pool, max_pool, geom_features), dim=-1))
        geom_logit = torch.logit(torch.clamp(geom["collision_probability"], 1e-5, 1.0 - 1e-5))
        collision_logit = chunk_raw[:, 0] + geom_logit
        min_distance = geom["min_distance"] + 0.05 * torch.tanh(chunk_raw[:, 1])
        cost_residual = F.softplus(chunk_raw[:, 2])
        uncertainty_cost = (uncertainty * weights).sum(dim=(1, 2)) / torch.clamp(weights.sum(dim=(1, 2)), min=1.0)
        cost = geom["cost"] + cost_residual + self.uncertainty_weight * uncertainty_cost

        return {
            "cost": cost,
            "collision_logit": collision_logit,
            "collision_probability": torch.sigmoid(collision_logit),
            "min_distance": min_distance,
            "risk_logits": risk_logits,
            "risk_heatmap": torch.sigmoid(risk_logits),
            "uncertainty": uncertainty,
            "uncertainty_cost": uncertainty_cost,
            "geometric_cost": geom["cost"],
            "geometric_collision_probability": geom["collision_probability"],
            "geometric_min_distance": geom["min_distance"],
            "geometric_risk_heatmap": geom["risk_heatmap"],
        }


def collision_critic_loss(
    predictions: TensorDict,
    collision_label: Optional[torch.Tensor] = None,
    min_distance_label: Optional[torch.Tensor] = None,
    risk_mask: Optional[torch.Tensor] = None,
    unsafe_positive_weight: float = 8.0,
    lambda_collision: float = 1.0,
    lambda_distance: float = 1.0,
    lambda_risk: float = 1.0,
    lambda_conservative: float = 0.05,
    safe_distance: float = 0.03,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Loss for a conservative safety critic.

    Unsafe positives receive larger BCE weight to reduce false negatives.
    ``lambda_conservative`` penalizes low predicted risk when labels are absent
    or ambiguous by nudging the predicted minimum distance below the margin only
    when the model already sees high geometric risk.
    """

    total = predictions["cost"].new_tensor(0.0)
    metrics: Dict[str, Any] = {}

    if collision_label is not None and lambda_collision:
        labels = collision_label.to(device=predictions["collision_logit"].device, dtype=predictions["collision_logit"].dtype)
        labels = labels.reshape_as(predictions["collision_logit"])
        weight = torch.where(labels > 0.5, labels.new_full((), float(unsafe_positive_weight)), labels.new_ones(()))
        loss = F.binary_cross_entropy_with_logits(predictions["collision_logit"], labels, weight=weight)
        total = total + float(lambda_collision) * loss
        metrics["collision_bce"] = loss.detach()
        with torch.no_grad():
            pred_unsafe = predictions["collision_probability"] > 0.5
            metrics["collision_accuracy"] = (pred_unsafe == (labels > 0.5)).to(torch.float32).mean()

    if min_distance_label is not None and lambda_distance:
        target = min_distance_label.to(device=predictions["min_distance"].device, dtype=predictions["min_distance"].dtype)
        target = target.reshape_as(predictions["min_distance"])
        distance_loss = F.huber_loss(predictions["min_distance"], target)
        total = total + float(lambda_distance) * distance_loss
        metrics["distance_loss"] = distance_loss.detach()
        metrics["distance_l1"] = torch.abs(predictions["min_distance"] - target).mean().detach()

    if risk_mask is not None and lambda_risk:
        target_risk = risk_mask.to(device=predictions["risk_logits"].device, dtype=predictions["risk_logits"].dtype)
        risk_weight = torch.where(
            target_risk > 0.5,
            target_risk.new_full((), float(unsafe_positive_weight)),
            target_risk.new_ones(()),
        )
        risk_loss = F.binary_cross_entropy_with_logits(predictions["risk_logits"], target_risk, weight=risk_weight)
        total = total + float(lambda_risk) * risk_loss
        metrics["risk_bce"] = risk_loss.detach()

    if lambda_conservative:
        geom_prob = predictions.get("geometric_collision_probability")
        if geom_prob is not None:
            margin_violation = torch.relu(float(safe_distance) - predictions["min_distance"])
            conservative_loss = (geom_prob.detach() * torch.relu(0.5 - predictions["collision_probability"])).mean()
            conservative_loss = conservative_loss + margin_violation.mean()
            total = total + float(lambda_conservative) * conservative_loss
            metrics["conservative_loss"] = conservative_loss.detach()

    metrics["loss"] = total.detach()
    return total, metrics


@dataclass
class RerankResult:
    index: torch.Tensor
    action: torch.Tensor
    score: torch.Tensor
    cost: torch.Tensor
    accepted: torch.Tensor


def rerank_action_chunks(
    actions: torch.Tensor,
    safety_cost: torch.Tensor,
    task_score: Optional[torch.Tensor] = None,
    safety_weight: float = 1.0,
    max_cost: Optional[float] = None,
) -> RerankResult:
    """Select the best candidate action chunk under a safety cost.

    Args:
        actions: ``(B, K, H, A)`` candidate chunks.
        safety_cost: ``(B, K)`` critic cost for each chunk.
        task_score: optional ``(B, K)`` VLA log-prob or task critic score.
        max_cost: if set, chunks above the threshold are rejected. If all
            chunks are rejected, the least unsafe chunk is selected and
            ``accepted`` is false for that batch element.
    """

    if actions.ndim < 4:
        raise ValueError(f"actions must have shape (B, K, H, A...), got {tuple(actions.shape)}")
    if safety_cost.shape != actions.shape[:2]:
        raise ValueError(f"safety_cost must have shape {tuple(actions.shape[:2])}, got {tuple(safety_cost.shape)}")
    if task_score is None:
        task_score = safety_cost.new_zeros(safety_cost.shape)
    if task_score.shape != safety_cost.shape:
        raise ValueError("task_score and safety_cost must have the same shape")

    score = task_score - float(safety_weight) * safety_cost
    if max_cost is None:
        masked_score = score
        accepted = torch.ones(actions.shape[0], device=actions.device, dtype=torch.bool)
    else:
        valid = safety_cost <= float(max_cost)
        accepted = valid.any(dim=1)
        masked_score = torch.where(valid, score, score.new_full((), -1e9))
        fallback_score = -safety_cost
        masked_score = torch.where(accepted[:, None], masked_score, fallback_score)

    index = masked_score.argmax(dim=1)
    batch_index = torch.arange(actions.shape[0], device=actions.device)
    return RerankResult(
        index=index,
        action=actions[batch_index, index],
        score=score[batch_index, index],
        cost=safety_cost[batch_index, index],
        accepted=accepted,
    )


def action_smoothness_cost(actions: torch.Tensor) -> torch.Tensor:
    """Chunk-level jerk cost, returned as ``(B,)`` or ``(B, K)``."""

    if actions.shape[-2] < 3:
        return actions.new_zeros(actions.shape[:-2])
    jerk = actions[..., 2:, :] - 2.0 * actions[..., 1:-1, :] + actions[..., :-2, :]
    return jerk.pow(2).mean(dim=(-1, -2))


def joint_limit_cost(
    q: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Soft joint-limit violation cost for rollout joint positions."""

    lower = torch.relu((q_min.to(q) + float(margin)) - q)
    upper = torch.relu(q - (q_max.to(q) - float(margin)))
    return (lower.pow(2) + upper.pow(2)).mean(dim=tuple(range(1, q.ndim)))
