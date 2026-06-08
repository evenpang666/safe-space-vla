import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from safety_module.safety_flow_point_model import SafetyFlowPointModel
from scripts.train_pi05_safety_flow_point_model import (
    apply_prefix_ablation,
    build_model_kwargs,
    default_loss_csv_path,
    default_loss_plot_path,
    save_loss_history,
    periodic_checkpoint_path,
    load_dataset_tensors,
    save_checkpoint,
    should_save_periodic_checkpoint,
    should_update_loss_plot,
    train_one_epoch,
)


def test_train_flow_script_help_runs_when_invoked_by_path():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/train_pi05_safety_flow_point_model.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout
    assert "--num-encoder-layers" in result.stdout
    assert "--num-decoder-layers" in result.stdout
    assert "--num-heads" in result.stdout
    assert "--prefix-ablation" in result.stdout


def test_load_dataset_tensors_reads_flow_fields(tmp_path: Path):
    dataset = tmp_path / "flow_dataset.npz"
    np.savez_compressed(
        dataset,
        prefix_tokens=np.zeros((3, 4, 5), dtype=np.float32),
        arm_points=np.zeros((3, 6, 3), dtype=np.float32),
        target_point_offsets=np.zeros((3, 2, 6, 3), dtype=np.float32),
    )

    prefix, arm_points, offsets = load_dataset_tensors(dataset)

    assert prefix.shape == (3, 4, 5)
    assert arm_points.shape == (3, 6, 3)
    assert offsets.shape == (3, 2, 6, 3)
    assert prefix.dtype == torch.float32


def test_load_dataset_tensors_rejects_mismatched_point_count(tmp_path: Path):
    dataset = tmp_path / "flow_dataset.npz"
    np.savez_compressed(
        dataset,
        prefix_tokens=np.zeros((3, 4, 5), dtype=np.float32),
        arm_points=np.zeros((3, 6, 3), dtype=np.float32),
        target_point_offsets=np.zeros((3, 2, 7, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="point count"):
        load_dataset_tensors(dataset)


def test_build_model_kwargs_uses_dataset_shapes():
    prefix = torch.zeros(3, 4, 5)
    arm_points = torch.zeros(3, 6, 3)
    offsets = torch.zeros(3, 2, 6, 3)

    kwargs = build_model_kwargs(
        prefix,
        arm_points,
        offsets,
        hidden_dim=32,
        num_encoder_layers=1,
        num_decoder_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.1,
        max_prefix_tokens=16,
    )

    assert kwargs["prefix_dim"] == 5
    assert kwargs["arm_point_dim"] == 3
    assert kwargs["n_future"] == 2
    assert kwargs["max_points"] == 6
    assert kwargs["num_decoder_layers"] == 2


def test_apply_prefix_ablation_zero_returns_zero_prefix_without_mutating_input():
    prefix = torch.randn(2, 3, 4)
    original = prefix.clone()

    ablated = apply_prefix_ablation(prefix, "zero")

    assert torch.count_nonzero(ablated) == 0
    assert ablated.shape == prefix.shape
    assert ablated.dtype == prefix.dtype
    assert ablated.device == prefix.device
    torch.testing.assert_close(prefix, original)


def test_apply_prefix_ablation_none_returns_original_tensor():
    prefix = torch.randn(2, 3, 4)

    ablated = apply_prefix_ablation(prefix, "none")

    assert ablated is prefix


def test_train_one_epoch_updates_flow_model_parameters_on_tiny_dataset():
    torch.manual_seed(0)
    prefix = torch.randn(4, 3, 5)
    arm_points = torch.randn(4, 6, 3)
    offsets = torch.randn(4, 2, 6, 3)
    model = SafetyFlowPointModel(
        arm_point_dim=3,
        prefix_dim=5,
        hidden_dim=16,
        n_future=2,
        max_points=6,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        ffn_dim=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_one_epoch(
        model,
        optimizer,
        prefix,
        arm_points,
        offsets,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(previous, current) for previous, current in zip(before, model.parameters(), strict=True))


def test_save_checkpoint_writes_flow_model_metadata(tmp_path: Path):
    path = tmp_path / "flow_model.pt"
    model_kwargs = {
        "arm_point_dim": 3,
        "prefix_dim": 5,
        "hidden_dim": 16,
        "n_future": 2,
        "max_points": 6,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 4,
        "ffn_dim": 32,
        "dropout": 0.0,
        "max_prefix_tokens": 16,
    }
    model = SafetyFlowPointModel(**model_kwargs)

    save_checkpoint(path, model, model_kwargs, epoch=3, loss=0.5, training_metadata={"prefix_ablation": "zero"})

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["model_type"] == "SafetyFlowPointModel"
    assert checkpoint["model_kwargs"] == model_kwargs
    assert checkpoint["epoch"] == 3
    assert checkpoint["loss"] == 0.5
    assert checkpoint["training_metadata"]["prefix_ablation"] == "zero"

    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["model_type"] == "SafetyFlowPointModel"
    assert sidecar["model_kwargs"]["prefix_dim"] == 5
    assert sidecar["training_metadata"]["prefix_ablation"] == "zero"


def test_periodic_checkpoint_path_inserts_epoch_before_suffix():
    path = periodic_checkpoint_path(Path("outputs/model.pt"), epoch=100)

    assert path == Path("outputs/model_epoch0100.pt")


def test_should_save_periodic_checkpoint_uses_positive_interval():
    assert should_save_periodic_checkpoint(epoch=100, checkpoint_every=100)
    assert should_save_periodic_checkpoint(epoch=200, checkpoint_every=100)
    assert not should_save_periodic_checkpoint(epoch=99, checkpoint_every=100)
    assert not should_save_periodic_checkpoint(epoch=100, checkpoint_every=0)


def test_default_loss_artifact_paths_follow_output_stem():
    output = Path("outputs/model.pt")

    assert default_loss_plot_path(output) == Path("outputs/model_loss.png")
    assert default_loss_csv_path(output) == Path("outputs/model_loss.csv")


def test_should_update_loss_plot_uses_positive_interval():
    assert should_update_loss_plot(epoch=5, plot_every=5)
    assert should_update_loss_plot(epoch=10, plot_every=5)
    assert not should_update_loss_plot(epoch=4, plot_every=5)
    assert not should_update_loss_plot(epoch=5, plot_every=0)


def test_save_loss_history_writes_csv_and_png(tmp_path: Path):
    csv_path = tmp_path / "loss.csv"
    png_path = tmp_path / "loss.png"

    save_loss_history(
        [(1, 2.0), (2, 1.5), (3, 1.0)],
        csv_path=csv_path,
        plot_path=png_path,
    )

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "epoch,loss",
        "1,2.0000000000",
        "2,1.5000000000",
        "3,1.0000000000",
    ]
    assert png_path.exists()
    assert png_path.stat().st_size > 0
