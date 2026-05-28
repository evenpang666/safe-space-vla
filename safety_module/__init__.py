"""Trainable safety modules for robot link-point prediction."""

from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig


def predicted_link_points_collision(*args, **kwargs):
    from safety_module.geometric_safety import predicted_link_points_collision as _impl

    return _impl(*args, **kwargs)


__all__ = ["SafetyPointDecoder", "SafetyPointDecoderConfig", "predicted_link_points_collision"]
