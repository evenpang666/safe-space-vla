from pathlib import Path

import numpy as np
import torch

from scripts.run_pi05_safety_decoder import load_checkpoint_model, run_prediction
from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig


def test_run_prediction_returns_points_and_geometric_collision(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    config = SafetyPointDecoderConfig(
        token_dim=4,
        hidden_dim=8,
        num_layers=1,
        horizon=1,
        num_links=1,
        points_per_link=2,
    )
    model = SafetyPointDecoder(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict(), "epoch": 1, "loss": 0.0},
        checkpoint,
    )

    loaded = load_checkpoint_model(checkpoint, torch.device("cpu"))
    prefix_tokens = np.zeros((1, 3, 4), dtype=np.float32)
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        "obstacle_box_axes": np.asarray([np.eye(3)], dtype=np.float64),
        "obstacle_box_half_sizes": np.asarray([[0.1, 0.1, 0.1]], dtype=np.float64),
    }

    pred, result = run_prediction(loaded, prefix_tokens, safe_space, collision_margin=0.0, device=torch.device("cpu"))

    assert pred.shape == (1, 1, 1, 2, 3)
    assert result.collides is True
