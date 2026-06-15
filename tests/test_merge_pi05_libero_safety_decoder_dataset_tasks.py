from pathlib import Path

import numpy as np

from scripts import merge_pi05_libero_safety_decoder_dataset_tasks as merger
from scripts.collect_pi05_libero_safety_decoder_dataset import CollectedSampleBuffer, save_collected_dataset


def _write_task_shard(path: Path, *, task_id: int) -> None:
    buffer = CollectedSampleBuffer()
    buffer.append(
        prefix_tokens=np.full((5, 8), task_id, dtype=np.float32),
        action_chunk=np.full((10, 7), task_id + 0.1, dtype=np.float32),
        start_joint_vector=np.full((7,), task_id + 0.2, dtype=np.float32),
        target_link_points=np.full((11, 3, 4, 3), task_id + 0.3, dtype=np.float32),
        task_id=task_id,
        rollout_id=0,
        step_id=task_id * 10,
    )
    save_collected_dataset(
        path,
        buffer=buffer,
        link_names=np.asarray(["link0", "link1", "link2"]),
        task_suite="libero_spatial",
        points_per_link=4,
        samples_per_action=1,
        policy_config="pi05_libero",
        checkpoint_dir="checkpoint",
    )


def test_parse_args_defaults_to_libero_spatial_task_merge(monkeypatch):
    monkeypatch.setattr(merger.sys, "argv", ["merge_pi05_libero_safety_decoder_dataset_tasks.py"])

    args = merger.parse_args()

    assert args.task_suite == "libero_spatial"
    assert args.task_ids == ["all"]
    assert args.input_dir is None
    assert args.output == merger.DEFAULT_OUTPUT


def test_resolve_merge_task_ids_expands_known_suite_all():
    assert merger.resolve_merge_task_ids(task_suite="libero_spatial", task_ids=["all"]) == list(range(10))
    assert merger.resolve_merge_task_ids(task_suite="libero_spatial", task_ids=["2", "0"]) == [2, 0]


def test_main_merges_selected_task_shards(monkeypatch, tmp_path: Path):
    output = tmp_path / "pi05_libero_decoder_dataset.npz"
    shard_dir = tmp_path / "pi05_libero_decoder_dataset_tasks"
    shard_dir.mkdir()
    for task_id in [0, 1]:
        _write_task_shard(shard_dir / f"libero_spatial_task{task_id:03d}.npz", task_id=task_id)

    monkeypatch.setattr(
        merger.sys,
        "argv",
        [
            "merge_pi05_libero_safety_decoder_dataset_tasks.py",
            "--task-suite",
            "libero_spatial",
            "--task-ids",
            "0",
            "1",
            "--input-dir",
            str(shard_dir),
            "--output",
            str(output),
        ],
    )

    merger.main()

    with np.load(output, allow_pickle=False) as data:
        assert data["prefix_tokens"].shape == (2, 5, 8)
        assert data["task_ids"].tolist() == [0, 1]
        assert data["step_ids"].tolist() == [0, 10]
