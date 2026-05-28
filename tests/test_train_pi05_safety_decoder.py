from pathlib import Path
import json

import numpy as np
import torch

from scripts.train_pi05_safety_decoder import load_dataset_tensors, save_checkpoint, train_one_epoch
from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig


def test_load_dataset_tensors_reads_prefix_and_targets(tmp_path: Path):
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        prefix_tokens=np.zeros((3, 4, 5), dtype=np.float32),
        target_link_points=np.zeros((3, 2, 6, 3, 3), dtype=np.float32),
    )

    prefix, targets = load_dataset_tensors(dataset)

    assert prefix.shape == (3, 4, 5)
    assert targets.shape == (3, 2, 6, 3, 3)
    assert prefix.dtype == torch.float32


def test_train_one_epoch_decreases_loss_on_tiny_dataset():
    torch.manual_seed(0)
    prefix = torch.randn(8, 4, 5)
    targets = torch.randn(8, 2, 3, 2, 3)
    config = SafetyPointDecoderConfig(token_dim=5, hidden_dim=32, num_layers=2, horizon=2, num_links=3, points_per_link=2)
    model = SafetyPointDecoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    first = train_one_epoch(model, optimizer, prefix, targets, batch_size=4, device=torch.device("cpu"))
    second = train_one_epoch(model, optimizer, prefix, targets, batch_size=4, device=torch.device("cpu"))

    assert second <= first


def test_save_checkpoint_writes_pt_and_json_sidecar(tmp_path: Path):
    path = tmp_path / "decoder.pt"
    config = SafetyPointDecoderConfig(token_dim=5, hidden_dim=32, num_layers=2, horizon=2, num_links=3, points_per_link=2)
    model = SafetyPointDecoder(config)

    save_checkpoint(path, model, config, epoch=7, loss=0.25)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert path.exists()
    assert "model_state_dict" in checkpoint
    assert checkpoint["config"] == config.to_dict()
    assert checkpoint["epoch"] == 7
    assert checkpoint["loss"] == 0.25

    sidecar_path = path.with_suffix(".json")
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["epoch"] == 7
    assert sidecar["loss"] == 0.25
    assert sidecar["config"]["token_dim"] == 5
