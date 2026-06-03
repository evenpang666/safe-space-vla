import pytest
import torch

from safety_module.safety_flow_point_model import (
    ArmPointTokenEmbedding,
    FlowPointDecoderLayer,
    SafetyFlowPointModel,
    SinusoidalTimeEmbedding,
    euler_sample,
    flow_matching_loss,
    sample_flow_matching_batch,
)


def test_sinusoidal_time_embedding_accepts_vector_or_column_time():
    embedding = SinusoidalTimeEmbedding(hidden_dim=16)

    vector_out = embedding(torch.tensor([0.0, 1.0]))
    column_out = embedding(torch.tensor([[0.0], [1.0]]))

    assert vector_out.shape == (2, 16)
    torch.testing.assert_close(vector_out, column_out)


def test_arm_point_embedding_supports_xyz_only_points():
    module = ArmPointTokenEmbedding(arm_point_dim=3, hidden_dim=32)
    arm_points = torch.randn(2, 5, 3)

    tokens = module(arm_points)

    assert tokens.shape == (2, 5, 32)


def test_decoder_layer_has_separate_self_geometry_and_prefix_attention():
    layer = FlowPointDecoderLayer(hidden_dim=32, num_heads=4, ffn_dim=64)

    assert isinstance(layer.self_attn, torch.nn.MultiheadAttention)
    assert isinstance(layer.geom_cross_attn, torch.nn.MultiheadAttention)
    assert isinstance(layer.prefix_cross_attn, torch.nn.MultiheadAttention)

    z = torch.randn(2, 6, 32)
    h_arm = torch.randn(2, 5, 32)
    h_prefix = torch.randn(2, 4, 32)

    out = layer(z, h_arm, h_prefix)

    assert out.shape == z.shape


def test_safety_flow_point_model_forward_loss_and_sampling_shapes():
    torch.manual_seed(0)
    batch_size = 2
    num_points = 7
    prefix_len = 5
    prefix_dim = 11
    n_future = 3
    model = SafetyFlowPointModel(
        arm_point_dim=3,
        prefix_dim=prefix_dim,
        hidden_dim=32,
        n_future=n_future,
        max_points=num_points,
        num_encoder_layers=1,
        num_decoder_layers=2,
        num_heads=4,
        ffn_dim=64,
    )
    arm_points = torch.randn(batch_size, num_points, 3)
    prefix_tokens = torch.randn(batch_size, prefix_len, prefix_dim)
    x_1 = torch.randn(batch_size, n_future, num_points, 3)

    x_s, s, x_0, v_target = sample_flow_matching_batch(x_1)
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
        n_steps=4,
        n_future=n_future,
        K=num_points,
    )

    assert v_target.shape == x_1.shape
    assert v_pred.shape == x_1.shape
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert delta_pred.shape == x_1.shape


def test_safety_flow_point_model_rejects_point_count_beyond_embedding_table():
    model = SafetyFlowPointModel(
        arm_point_dim=3,
        prefix_dim=8,
        hidden_dim=16,
        n_future=2,
        max_points=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        ffn_dim=32,
    )

    with pytest.raises(ValueError, match="max_points"):
        model(
            arm_points=torch.randn(1, 5, 3),
            prefix_tokens=torch.randn(1, 3, 8),
            x_s=torch.randn(1, 2, 5, 3),
            s=torch.tensor([0.5]),
        )
