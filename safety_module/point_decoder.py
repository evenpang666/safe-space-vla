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
    num_heads: int = 8
    ffn_dim: int = 0
    max_tokens: int = 1024
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
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads, got hidden_dim={self.hidden_dim} "
                f"and num_heads={self.num_heads}"
            )
        if self.ffn_dim < 0:
            raise ValueError(f"ffn_dim must be >= 0, got {self.ffn_dim}")
        if self.ffn_dim == 0:
            object.__setattr__(self, "ffn_dim", int(self.hidden_dim) * 4)
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
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
            num_heads=int(payload.get("num_heads", 8)),
            ffn_dim=int(payload.get("ffn_dim", 0)),
            max_tokens=int(payload.get("max_tokens", 1024)),
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
        self.token_projection = nn.Linear(int(config.token_dim), int(config.hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(int(config.max_tokens), int(config.hidden_dim)))
        self.input_dropout = nn.Dropout(float(config.dropout))
        if config.num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(config.hidden_dim),
                nhead=int(config.num_heads),
                dim_feedforward=int(config.ffn_dim),
                dropout=float(config.dropout),
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(config.num_layers))
        else:
            self.transformer = nn.Identity()
        out_dim = int(config.horizon) * int(config.num_links) * int(config.points_per_link) * 3
        self.output_head = nn.Linear(int(config.hidden_dim), out_dim)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, prefix_tokens: torch.Tensor, prefix_mask: torch.Tensor | None = None) -> torch.Tensor:
        if prefix_tokens.ndim != 3:
            raise ValueError(f"prefix_tokens must have shape (B, N, D), got {tuple(prefix_tokens.shape)}")
        if prefix_tokens.shape[-1] != int(self.config.token_dim):
            raise ValueError(
                f"prefix_tokens last dimension must match token_dim={self.config.token_dim}, "
                f"got {prefix_tokens.shape[-1]}"
            )
        if prefix_tokens.shape[1] > int(self.config.max_tokens):
            raise ValueError(
                f"prefix token count {prefix_tokens.shape[1]} exceeds max_tokens={self.config.max_tokens}"
            )
        if prefix_mask is not None and prefix_mask.shape != prefix_tokens.shape[:2]:
            raise ValueError(
                f"prefix_mask must have shape {tuple(prefix_tokens.shape[:2])}, got {tuple(prefix_mask.shape)}"
            )
        param = next(self.parameters())
        prefix_tokens = prefix_tokens.to(device=param.device, dtype=param.dtype)
        projected = self.token_projection(prefix_tokens)
        positions = self.position_embedding[: prefix_tokens.shape[1]].to(dtype=projected.dtype)
        encoded_input = self.input_dropout(projected + positions[None, :, :])
        padding_mask = None
        if prefix_mask is not None:
            prefix_mask = prefix_mask.to(device=param.device, dtype=torch.bool)
            padding_mask = ~prefix_mask
        if isinstance(self.transformer, nn.Identity):
            encoded = self.transformer(encoded_input)
        else:
            encoded = self.transformer(encoded_input, src_key_padding_mask=padding_mask)
        pooled = masked_mean_pool(encoded, prefix_mask)
        raw = self.output_head(pooled)
        return raw.reshape(
            prefix_tokens.shape[0],
            int(self.config.horizon),
            int(self.config.num_links),
            int(self.config.points_per_link),
            3,
        )
