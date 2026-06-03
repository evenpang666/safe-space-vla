import pytest
import torch

from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig, masked_mean_pool


def test_masked_mean_pool_ignores_invalid_tokens():
    tokens = torch.tensor([[[1.0, 3.0], [9.0, 9.0], [5.0, 7.0]]])
    mask = torch.tensor([[True, False, True]])

    pooled = masked_mean_pool(tokens, mask)

    torch.testing.assert_close(pooled, torch.tensor([[3.0, 5.0]]))


def test_safety_point_decoder_outputs_fixed_topology_points():
    config = SafetyPointDecoderConfig(
        token_dim=6,
        hidden_dim=16,
        num_layers=2,
        horizon=4,
        num_links=3,
        points_per_link=5,
    )
    model = SafetyPointDecoder(config)
    prefix_tokens = torch.randn(2, 7, 6)
    prefix_mask = torch.ones(2, 7, dtype=torch.bool)

    points = model(prefix_tokens, prefix_mask)

    assert points.shape == (2, 4, 3, 5, 3)
    assert points.dtype == torch.float32


def test_safety_point_decoder_uses_transformer_encoder_before_linear_head():
    config = SafetyPointDecoderConfig(
        token_dim=6,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
        max_tokens=8,
        horizon=4,
        num_links=3,
        points_per_link=5,
    )

    model = SafetyPointDecoder(config)

    assert isinstance(model.transformer, torch.nn.TransformerEncoder)
    assert isinstance(model.output_head, torch.nn.Linear)
    assert model.output_head.out_features == 4 * 3 * 5 * 3


def test_safety_point_decoder_config_round_trips_transformer_fields():
    config = SafetyPointDecoderConfig(
        token_dim=6,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        max_tokens=128,
        horizon=4,
        num_links=3,
        points_per_link=5,
        dropout=0.1,
    )

    restored = SafetyPointDecoderConfig.from_dict(config.to_dict())

    assert restored == config


def test_safety_point_decoder_can_fit_one_tiny_batch():
    torch.manual_seed(0)
    config = SafetyPointDecoderConfig(
        token_dim=4,
        hidden_dim=32,
        num_layers=3,
        horizon=2,
        num_links=2,
        points_per_link=3,
    )
    model = SafetyPointDecoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    prefix_tokens = torch.randn(4, 5, 4)
    target = torch.randn(4, 2, 2, 3, 3)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        pred = model(prefix_tokens)
        loss = torch.nn.functional.smooth_l1_loss(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]


def test_safety_point_decoder_config_rejects_invalid_horizon():
    with pytest.raises(ValueError, match="horizon"):
        SafetyPointDecoderConfig(
            token_dim=4,
            hidden_dim=32,
            num_layers=3,
            horizon=0,
            num_links=2,
            points_per_link=3,
        )


def test_safety_point_decoder_config_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        SafetyPointDecoderConfig(
            token_dim=4,
            hidden_dim=32,
            num_layers=3,
            horizon=2,
            num_links=2,
            points_per_link=3,
            dropout=1.0,
        )


def test_safety_point_decoder_config_rejects_hidden_dim_not_divisible_by_heads():
    with pytest.raises(ValueError, match="hidden_dim.*num_heads"):
        SafetyPointDecoderConfig(
            token_dim=4,
            hidden_dim=30,
            num_layers=3,
            num_heads=8,
            horizon=2,
            num_links=2,
            points_per_link=3,
        )


def test_safety_point_decoder_rejects_wrong_prefix_token_dim():
    config = SafetyPointDecoderConfig(
        token_dim=4,
        hidden_dim=32,
        num_layers=3,
        horizon=2,
        num_links=2,
        points_per_link=3,
    )
    model = SafetyPointDecoder(config)
    prefix_tokens = torch.randn(4, 5, 5)

    with pytest.raises(ValueError, match="token_dim"):
        model(prefix_tokens)


def test_safety_point_decoder_rejects_more_prefix_tokens_than_position_table():
    config = SafetyPointDecoderConfig(
        token_dim=4,
        hidden_dim=32,
        num_layers=3,
        max_tokens=4,
        horizon=2,
        num_links=2,
        points_per_link=3,
    )
    model = SafetyPointDecoder(config)

    with pytest.raises(ValueError, match="max_tokens"):
        model(torch.randn(1, 5, 4))


def test_safety_point_decoder_double_returns_float64_output():
    config = SafetyPointDecoderConfig(
        token_dim=4,
        hidden_dim=32,
        num_layers=3,
        horizon=2,
        num_links=2,
        points_per_link=3,
    )
    model = SafetyPointDecoder(config).double()
    prefix_tokens = torch.randn(4, 5, 4, dtype=torch.float64)

    points = model(prefix_tokens)

    assert points.dtype == torch.float64
