"""Trainable safety modules for robot link-point prediction."""

from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig
from safety_module.safety_flow_point_model import (
    ArmPointTokenEmbedding,
    FlowPointDecoderLayer,
    FlowPointHead,
    MLP,
    PrefixPointEncoder,
    PrefixTokenAdapter,
    SafetyFlowPointModel,
    SinusoidalTimeEmbedding,
    euler_sample,
    flow_matching_loss,
    sample_flow_matching_batch,
)


def predicted_link_points_collision(*args, **kwargs):
    from safety_module.geometric_safety import predicted_link_points_collision as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "ArmPointTokenEmbedding",
    "FlowPointDecoderLayer",
    "FlowPointHead",
    "MLP",
    "PrefixPointEncoder",
    "PrefixTokenAdapter",
    "SafetyFlowPointModel",
    "SafetyPointDecoder",
    "SafetyPointDecoderConfig",
    "SinusoidalTimeEmbedding",
    "euler_sample",
    "flow_matching_loss",
    "predicted_link_points_collision",
    "sample_flow_matching_batch",
]
