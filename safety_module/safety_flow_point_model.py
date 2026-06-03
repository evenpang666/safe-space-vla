from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous flow-time embedding for s in [0, 1]."""

    def __init__(self, hidden_dim: int, max_period: float = 10000.0):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        self.hidden_dim = int(hidden_dim)
        self.max_period = float(max_period)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: [B] or [B, 1]
        if s.ndim == 1:
            s_flat = s
        elif s.ndim == 2 and s.shape[1] == 1:
            s_flat = s[:, 0]
        else:
            raise ValueError(f"s must have shape [B] or [B, 1], got {tuple(s.shape)}")

        half_dim = self.hidden_dim // 2
        if half_dim == 0:
            return s_flat[:, None]

        exponent = -math.log(self.max_period) * torch.arange(
            half_dim,
            device=s_flat.device,
            dtype=s_flat.dtype,
        ) / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        args = s_flat[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = F.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))
        return embedding


class MLP(nn.Module):
    """Small configurable MLP used by point, time, and velocity projections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}")

        layers: list[nn.Module] = []
        dims = [input_dim]
        if num_layers > 1:
            dims.extend([hidden_dim] * (num_layers - 1))
        dims.append(output_dim)
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArmPointTokenEmbedding(nn.Module):
    """Embed current robot-arm local point tokens."""

    def __init__(self, arm_point_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        if arm_point_dim < 3:
            raise ValueError(f"arm_point_dim must be >= 3, got {arm_point_dim}")
        self.arm_point_dim = int(arm_point_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(arm_point_dim) - 3
        self.xyz_pos_mlp = MLP(3, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.feature_mlp = (
            MLP(self.feature_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
            if self.feature_dim > 0
            else None
        )
        self.arm_modality_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, arm_points: torch.Tensor) -> torch.Tensor:
        # arm_points: [B, K, 3 + C_arm]
        if arm_points.ndim != 3:
            raise ValueError(f"arm_points must have shape [B, K, D], got {tuple(arm_points.shape)}")
        if arm_points.shape[-1] != self.arm_point_dim:
            raise ValueError(
                f"arm_points last dimension must be arm_point_dim={self.arm_point_dim}, "
                f"got {arm_points.shape[-1]}"
            )

        xyz = arm_points[..., :3]  # [B, K, 3]
        xyz_embedding = self.xyz_pos_mlp(xyz)  # [B, K, hidden_dim]
        if self.feature_mlp is None:
            feature_embedding = torch.zeros_like(xyz_embedding)  # [B, K, hidden_dim]
        else:
            feature_embedding = self.feature_mlp(arm_points[..., 3:])  # [B, K, hidden_dim]
        return feature_embedding + xyz_embedding + self.arm_modality_embedding[None, None, :]


class PrefixTokenAdapter(nn.Module):
    """Project VLM prefix tokens into the safety model hidden space."""

    def __init__(
        self,
        prefix_dim: int,
        hidden_dim: int,
        max_prefix_tokens: int = 1024,
        use_position_embedding: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if prefix_dim <= 0:
            raise ValueError(f"prefix_dim must be > 0, got {prefix_dim}")
        if max_prefix_tokens <= 0:
            raise ValueError(f"max_prefix_tokens must be > 0, got {max_prefix_tokens}")
        self.prefix_dim = int(prefix_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_prefix_tokens = int(max_prefix_tokens)
        self.use_position_embedding = bool(use_position_embedding)
        self.proj = nn.Linear(prefix_dim, hidden_dim)
        self.prefix_modality_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.position_embedding = (
            nn.Parameter(torch.zeros(max_prefix_tokens, hidden_dim))
            if use_position_embedding
            else None
        )
        self.dropout = nn.Dropout(dropout)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        # prefix_tokens: [B, L, d_vlm]
        if prefix_tokens.ndim != 3:
            raise ValueError(f"prefix_tokens must have shape [B, L, D], got {tuple(prefix_tokens.shape)}")
        if prefix_tokens.shape[-1] != self.prefix_dim:
            raise ValueError(
                f"prefix_tokens last dimension must be prefix_dim={self.prefix_dim}, "
                f"got {prefix_tokens.shape[-1]}"
            )
        if prefix_tokens.shape[1] > self.max_prefix_tokens:
            raise ValueError(
                f"prefix token count {prefix_tokens.shape[1]} exceeds max_prefix_tokens={self.max_prefix_tokens}"
            )

        tokens = self.proj(prefix_tokens)  # [B, L, hidden_dim]
        tokens = tokens + self.prefix_modality_embedding[None, None, :]
        if self.position_embedding is not None:
            tokens = tokens + self.position_embedding[: prefix_tokens.shape[1]][None, :, :]
        return self.dropout(tokens)


class PrefixPointEncoder(nn.Module):
    """Jointly encode current arm geometry and VLM prefix conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        _validate_attention_dims(hidden_dim, num_heads)
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, got {num_layers}")
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4
        if num_layers == 0:
            self.encoder = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        H_arm: torch.Tensor,
        H_prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # H_arm: [B, K, hidden_dim], H_prefix: [B, L, hidden_dim]
        if H_arm.ndim != 3 or H_prefix.ndim != 3:
            raise ValueError("H_arm and H_prefix must both have shape [B, N, hidden_dim]")
        if H_arm.shape[0] != H_prefix.shape[0] or H_arm.shape[-1] != H_prefix.shape[-1]:
            raise ValueError(
                f"H_arm and H_prefix batch/hidden dimensions must match, got "
                f"{tuple(H_arm.shape)} and {tuple(H_prefix.shape)}"
            )

        num_arm_points = H_arm.shape[1]
        encoder_input = torch.cat([H_arm, H_prefix], dim=1)  # [B, K + L, hidden_dim]
        memory = self.encoder(encoder_input)  # [B, K + L, hidden_dim]
        H_arm_enc = memory[:, :num_arm_points, :]  # [B, K, hidden_dim]
        H_prefix_enc = memory[:, num_arm_points:, :]  # [B, L, hidden_dim]
        return H_arm_enc, H_prefix_enc, memory


class FlowPointDecoderLayer(nn.Module):
    """Custom decoder layer for flow-matching future point tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        _validate_attention_dims(hidden_dim, num_heads)
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.geom_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.prefix_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.norm_geom = nn.LayerNorm(hidden_dim)
        self.norm_prefix = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        H_arm_enc: torch.Tensor,
        H_prefix_enc: torch.Tensor,
    ) -> torch.Tensor:
        # z: [B, n_future * K, hidden_dim]
        self_out, _ = self.self_attn(query=z, key=z, value=z, need_weights=False)
        z = self.norm_self(z + self.dropout(self_out))

        # H_arm_enc: [B, K, hidden_dim]
        geom_out, _ = self.geom_cross_attn(query=z, key=H_arm_enc, value=H_arm_enc, need_weights=False)
        z = self.norm_geom(z + self.dropout(geom_out))

        # H_prefix_enc: [B, L, hidden_dim]
        prefix_out, _ = self.prefix_cross_attn(
            query=z,
            key=H_prefix_enc,
            value=H_prefix_enc,
            need_weights=False,
        )
        z = self.norm_prefix(z + self.dropout(prefix_out))

        ffn_out = self.ffn(z)
        z = self.norm_ffn(z + self.dropout(ffn_out))
        return z


class FlowPointHead(nn.Module):
    """Flow Matching point head that predicts velocity for future offsets."""

    def __init__(
        self,
        hidden_dim: int,
        n_future: int,
        max_points: int,
        num_decoder_layers: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_future <= 0:
            raise ValueError(f"n_future must be > 0, got {n_future}")
        if max_points <= 0:
            raise ValueError(f"max_points must be > 0, got {max_points}")
        if num_decoder_layers <= 0:
            raise ValueError(f"num_decoder_layers must be > 0, got {num_decoder_layers}")
        self.hidden_dim = int(hidden_dim)
        self.n_future = int(n_future)
        self.max_points = int(max_points)
        self.x_mlp = MLP(3, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.future_step_embedding = nn.Embedding(n_future, hidden_dim)
        self.point_identity_embedding = nn.Embedding(max_points, hidden_dim)
        self.decoder_layers = nn.ModuleList(
            [
                FlowPointDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.velocity_head = MLP(hidden_dim, hidden_dim, 3, num_layers=2, dropout=dropout)

    def forward(
        self,
        x_s: torch.Tensor,
        s: torch.Tensor,
        H_arm_enc: torch.Tensor,
        H_prefix_enc: torch.Tensor,
    ) -> torch.Tensor:
        # x_s: [B, n_future, K, 3]
        if x_s.ndim != 4 or x_s.shape[-1] != 3:
            raise ValueError(f"x_s must have shape [B, n_future, K, 3], got {tuple(x_s.shape)}")
        batch_size, n_future, num_points, _ = x_s.shape
        if n_future != self.n_future:
            raise ValueError(f"x_s n_future must be {self.n_future}, got {n_future}")
        if num_points > self.max_points:
            raise ValueError(f"point count {num_points} exceeds max_points={self.max_points}")
        if H_arm_enc.shape[:2] != (batch_size, num_points):
            raise ValueError(
                f"H_arm_enc must have shape [B, K, hidden_dim] matching x_s, got {tuple(H_arm_enc.shape)}"
            )

        x_s_flat = x_s.reshape(batch_size, n_future * num_points, 3)  # [B, n_future * K, 3]
        z = self.x_mlp(x_s_flat)  # [B, n_future * K, hidden_dim]

        future_ids = torch.arange(n_future, device=x_s.device).repeat_interleave(num_points)
        point_ids = torch.arange(num_points, device=x_s.device).repeat(n_future)
        z = z + self.future_step_embedding(future_ids)[None, :, :]
        z = z + self.point_identity_embedding(point_ids)[None, :, :]

        time_emb = self.time_mlp(self.time_embedding(s.to(device=x_s.device, dtype=x_s.dtype)))
        z = z + time_emb[:, None, :]  # [B, n_future * K, hidden_dim]

        for layer in self.decoder_layers:
            z = layer(z, H_arm_enc, H_prefix_enc)  # [B, n_future * K, hidden_dim]

        v_pred_flat = self.velocity_head(z)  # [B, n_future * K, 3]
        return v_pred_flat.reshape(batch_size, n_future, num_points, 3)


class SafetyFlowPointModel(nn.Module):
    """Prefix-conditioned Flow Matching model for future arm point offsets."""

    def __init__(
        self,
        arm_point_dim: int,
        prefix_dim: int,
        hidden_dim: int,
        n_future: int,
        max_points: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        max_prefix_tokens: int = 1024,
        use_prefix_position_embedding: bool = True,
    ):
        super().__init__()
        _validate_attention_dims(hidden_dim, num_heads)
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4
        self.arm_point_embedding = ArmPointTokenEmbedding(
            arm_point_dim=arm_point_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.prefix_adapter = PrefixTokenAdapter(
            prefix_dim=prefix_dim,
            hidden_dim=hidden_dim,
            max_prefix_tokens=max_prefix_tokens,
            use_position_embedding=use_prefix_position_embedding,
            dropout=dropout,
        )
        self.encoder = PrefixPointEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.flow_head = FlowPointHead(
            hidden_dim=hidden_dim,
            n_future=n_future,
            max_points=max_points,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        arm_points: torch.Tensor,
        prefix_tokens: torch.Tensor,
        x_s: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        # arm_points: [B, K, 3 + C_arm]
        # prefix_tokens: [B, L, d_vlm]
        # x_s: [B, n_future, K, 3]
        # s: [B] or [B, 1]
        param = next(self.parameters())
        arm_points = arm_points.to(device=param.device, dtype=param.dtype)
        prefix_tokens = prefix_tokens.to(device=param.device, dtype=param.dtype)
        x_s = x_s.to(device=param.device, dtype=param.dtype)
        s = s.to(device=param.device, dtype=param.dtype)

        H_arm = self.arm_point_embedding(arm_points)  # [B, K, hidden_dim]
        H_prefix = self.prefix_adapter(prefix_tokens)  # [B, L, hidden_dim]
        H_arm_enc, H_prefix_enc, _ = self.encoder(H_arm, H_prefix)
        return self.flow_head(x_s, s, H_arm_enc, H_prefix_enc)  # [B, n_future, K, 3]


def flow_matching_loss(v_pred: torch.Tensor, x_1: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
    # v_target: [B, n_future, K, 3]
    v_target = x_1.to(device=v_pred.device, dtype=v_pred.dtype) - x_0.to(device=v_pred.device, dtype=v_pred.dtype)
    return F.mse_loss(v_pred, v_target)


def sample_flow_matching_batch(x_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_1: [B, n_future, K, 3]
    if x_1.ndim != 4 or x_1.shape[-1] != 3:
        raise ValueError(f"x_1 must have shape [B, n_future, K, 3], got {tuple(x_1.shape)}")
    batch_size = x_1.shape[0]
    x_0 = torch.randn_like(x_1)  # [B, n_future, K, 3]
    s = torch.rand(batch_size, device=x_1.device, dtype=x_1.dtype)  # [B]
    s_view = s.view(batch_size, 1, 1, 1)
    x_s = (1.0 - s_view) * x_0 + s_view * x_1  # [B, n_future, K, 3]
    v_target = x_1 - x_0  # [B, n_future, K, 3]
    return x_s, s, x_0, v_target


@torch.no_grad()
def euler_sample(
    model: SafetyFlowPointModel,
    arm_points: torch.Tensor,
    prefix_tokens: torch.Tensor,
    n_steps: int,
    n_future: int,
    K: int,
) -> torch.Tensor:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    if n_future <= 0:
        raise ValueError(f"n_future must be > 0, got {n_future}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    param = next(model.parameters())
    batch_size = arm_points.shape[0]
    x_s = torch.randn(
        batch_size,
        n_future,
        K,
        3,
        device=param.device,
        dtype=param.dtype,
    )
    arm_points = arm_points.to(device=param.device, dtype=param.dtype)
    prefix_tokens = prefix_tokens.to(device=param.device, dtype=param.dtype)
    dt = 1.0 / float(n_steps)

    was_training = model.training
    model.eval()
    try:
        for step in range(n_steps):
            s_value = torch.full((batch_size,), step * dt, device=param.device, dtype=param.dtype)
            v_pred = model(
                arm_points=arm_points,
                prefix_tokens=prefix_tokens,
                x_s=x_s,
                s=s_value,
            )
            x_s = x_s + dt * v_pred
    finally:
        model.train(was_training)
    return x_s  # [B, n_future, K, 3]


# Future training hooks can add collision_loss, smoothness_loss, and
# rigid_link_consistency_loss here without changing the forward contract.


def _validate_attention_dims(hidden_dim: int, num_heads: int) -> None:
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be > 0, got {num_heads}")
    if hidden_dim % num_heads != 0:
        raise ValueError(
            f"hidden_dim must be divisible by num_heads, got hidden_dim={hidden_dim} and num_heads={num_heads}"
        )


def _count_parameters(modules: Iterable[nn.Module]) -> int:
    return sum(param.numel() for module in modules for param in module.parameters())


def main() -> None:
    B = 2
    K = 128
    L = 64
    C_arm = 0
    d_vlm = 768
    n_future = 8
    hidden_dim = 256
    num_encoder_layers = 4
    num_decoder_layers = 4
    num_heads = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix_tokens = torch.randn(B, L, d_vlm, device=device)  # [B, L, 768]
    arm_points = torch.randn(B, K, 3 + C_arm, device=device)  # [B, K, 3]
    x_1 = torch.randn(B, n_future, K, 3, device=device)  # [B, n_future, K, 3]

    model = SafetyFlowPointModel(
        arm_point_dim=3 + C_arm,
        prefix_dim=d_vlm,
        hidden_dim=hidden_dim,
        n_future=n_future,
        max_points=K,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
    ).to(device)

    x_s, s, x_0, _ = sample_flow_matching_batch(x_1)
    v_pred = model(
        arm_points=arm_points,
        prefix_tokens=prefix_tokens,
        x_s=x_s,
        s=s,
    )
    loss = flow_matching_loss(v_pred, x_1, x_0)
    delta_pred = euler_sample(
        model=model,
        arm_points=arm_points,
        prefix_tokens=prefix_tokens,
        n_steps=10,
        n_future=n_future,
        K=K,
    )

    print(v_pred.shape)
    print(loss.item())
    print(delta_pred.shape)
    print(f"parameters: {_count_parameters([model])}")


if __name__ == "__main__":
    main()
