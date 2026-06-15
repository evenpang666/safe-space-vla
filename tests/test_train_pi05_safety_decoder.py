import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.train_pi05_safety_decoder import load_dataset_tensors, resolve_dataset_paths, save_checkpoint, train_one_epoch
from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig


def test_train_script_help_runs_when_invoked_by_path():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/train_pi05_safety_decoder.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout
    assert "--num-heads" in result.stdout
    assert "--ffn-dim" in result.stdout
    assert "--max-tokens" in result.stdout


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


def test_load_dataset_tensors_concatenates_multiple_dataset_files(tmp_path: Path):
    first = tmp_path / "task000.npz"
    second = tmp_path / "task001.npz"
    np.savez_compressed(
        first,
        prefix_tokens=np.full((2, 4, 5), 1.0, dtype=np.float32),
        target_link_points=np.full((2, 2, 6, 3, 3), 10.0, dtype=np.float32),
    )
    np.savez_compressed(
        second,
        prefix_tokens=np.full((3, 4, 5), 2.0, dtype=np.float32),
        target_link_points=np.full((3, 2, 6, 3, 3), 20.0, dtype=np.float32),
    )

    prefix, targets = load_dataset_tensors([first, second])

    assert prefix.shape == (5, 4, 5)
    assert targets.shape == (5, 2, 6, 3, 3)
    assert torch.all(prefix[:2] == 1.0)
    assert torch.all(prefix[2:] == 2.0)
    assert torch.all(targets[:2] == 10.0)
    assert torch.all(targets[2:] == 20.0)


def test_resolve_dataset_paths_expands_directory_in_sorted_order(tmp_path: Path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "libero_spatial_task001.npz").write_bytes(b"")
    (shard_dir / "libero_spatial_task000.npz").write_bytes(b"")

    paths = resolve_dataset_paths([shard_dir])

    assert paths == [
        shard_dir / "libero_spatial_task000.npz",
        shard_dir / "libero_spatial_task001.npz",
    ]


def test_load_dataset_tensors_rejects_multiple_datasets_with_mismatched_shapes(tmp_path: Path):
    first = tmp_path / "task000.npz"
    second = tmp_path / "task001.npz"
    np.savez_compressed(
        first,
        prefix_tokens=np.zeros((2, 4, 5), dtype=np.float32),
        target_link_points=np.zeros((2, 2, 6, 3, 3), dtype=np.float32),
    )
    np.savez_compressed(
        second,
        prefix_tokens=np.zeros((2, 5, 5), dtype=np.float32),
        target_link_points=np.zeros((2, 2, 6, 3, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="non-sample dimensions"):
        load_dataset_tensors([first, second])


def test_load_dataset_tensors_rejects_empty_samples(tmp_path: Path):
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        prefix_tokens=np.zeros((0, 4, 5), dtype=np.float32),
        target_link_points=np.zeros((0, 2, 6, 3, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="at least one sample"):
        load_dataset_tensors(dataset)


@pytest.mark.parametrize(
    ("prefix_shape", "target_shape", "match"),
    [
        ((3, 0, 5), (3, 2, 6, 3, 3), "prefix_tokens token count"),
        ((3, 4, 0), (3, 2, 6, 3, 3), "prefix_tokens token dimension"),
        ((3, 4, 5), (3, 0, 6, 3, 3), "target_link_points horizon"),
        ((3, 4, 5), (3, 2, 0, 3, 3), "target_link_points link count"),
        ((3, 4, 5), (3, 2, 6, 0, 3), "target_link_points points per link"),
    ],
)
def test_load_dataset_tensors_rejects_zero_dimensions(
    tmp_path: Path,
    prefix_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    match: str,
):
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        prefix_tokens=np.zeros(prefix_shape, dtype=np.float32),
        target_link_points=np.zeros(target_shape, dtype=np.float32),
    )

    with pytest.raises(ValueError, match=match):
        load_dataset_tensors(dataset)


def test_train_one_epoch_updates_parameters_on_tiny_dataset():
    torch.manual_seed(0)
    prefix = torch.randn(8, 4, 5)
    targets = torch.randn(8, 2, 3, 2, 3)
    config = SafetyPointDecoderConfig(token_dim=5, hidden_dim=32, num_layers=2, horizon=2, num_links=3, points_per_link=2)
    model = SafetyPointDecoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_one_epoch(model, optimizer, prefix, targets, batch_size=len(prefix), device=torch.device("cpu"))

    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(previous, current) for previous, current in zip(before, model.parameters(), strict=True))


def test_train_one_epoch_rejects_empty_dataset():
    config = SafetyPointDecoderConfig(token_dim=5, hidden_dim=32, num_layers=2, horizon=2, num_links=3, points_per_link=2)
    model = SafetyPointDecoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    prefix = torch.empty(0, 4, 5)
    targets = torch.empty(0, 2, 3, 2, 3)

    with pytest.raises(ValueError, match="at least one sample"):
        train_one_epoch(model, optimizer, prefix, targets, batch_size=4, device=torch.device("cpu"))


def test_train_one_epoch_rejects_non_positive_batch_size():
    config = SafetyPointDecoderConfig(token_dim=5, hidden_dim=32, num_layers=2, horizon=2, num_links=3, points_per_link=2)
    model = SafetyPointDecoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    prefix = torch.randn(2, 4, 5)
    targets = torch.randn(2, 2, 3, 2, 3)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        train_one_epoch(model, optimizer, prefix, targets, batch_size=0, device=torch.device("cpu"))


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
