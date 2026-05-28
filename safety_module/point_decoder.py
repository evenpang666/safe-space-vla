from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SafetyPointDecoderConfig:
    token_dim: int
    hidden_dim: int
    num_layers: int
    horizon: int
    num_links: int
    points_per_link: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.token_dim <= 0:
            raise ValueError(f"token_dim must be > 0, got {self.token_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, got {self.num_layers}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")
        if self.num_links <= 0:
            raise ValueError(f"num_links must be > 0, got {self.num_links}")
        if self.points_per_link <= 0:
            raise ValueError(f"points_per_link must be > 0, got {self.points_per_link}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {self.dropout}")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "SafetyPointDecoderConfig":
        return cls(
            token_dim=int(payload["token_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_layers=int(payload["num_layers"]),
            horizon=int(payload["horizon"]),
            num_links=int(payload["num_links"]),
            points_per_link=int(payload["points_per_link"]),
            dropout=float(payload.get("dropout", 0.0)),
        )


def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (B, N, D), got {tuple(tokens.shape)}")
    if mask is None:
        return tokens.mean(dim=1)
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}")
    mask_f = mask.to(dtype=tokens.dtype, device=tokens.device)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (tokens * mask_f[:, :, None]).sum(dim=1) / denom
    return pooled


class SafetyPointDecoder(nn.Module):
    def __init__(self, config: SafetyPointDecoderConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_dim = int(config.token_dim)
        for _ in range(int(config.num_layers)):
            layers.append(nn.Linear(in_dim, int(config.hidden_dim)))
            layers.append(nn.GELU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(float(config.dropout)))
            in_dim = int(config.hidden_dim)
        out_dim = int(config.horizon) * int(config.num_links) * int(config.points_per_link) * 3
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, prefix_tokens: torch.Tensor, prefix_mask: torch.Tensor | None = None) -> torch.Tensor:
        if prefix_tokens.shape[-1] != int(self.config.token_dim):
            raise ValueError(
                f"prefix_tokens last dimension must match token_dim={self.config.token_dim}, "
                f"got {prefix_tokens.shape[-1]}"
            )
        param = next(self.parameters())
        prefix_tokens = prefix_tokens.to(device=param.device, dtype=param.dtype)
        pooled = masked_mean_pool(prefix_tokens, prefix_mask)
        raw = self.net(pooled)
        return raw.reshape(
            prefix_tokens.shape[0],
            int(self.config.horizon),
            int(self.config.num_links),
            int(self.config.points_per_link),
            3,
        )
