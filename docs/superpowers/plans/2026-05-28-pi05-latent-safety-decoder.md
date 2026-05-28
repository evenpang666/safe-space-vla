# PI05 Latent Safety Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trainable module that maps PI05 VLM prefix latent tokens to future fixed-topology robot link points, then decides `collision` / `safe` using deterministic geometry against OBBs or occupied grids.

**Architecture:** Keep the learned model narrowly scoped: `prefix_tokens -> pred_link_points [B, T, L, P, 3]`. FK target generation and OBB collision stay outside the neural network, reusing the existing LIBERO swept-link geometry and collision functions. First implementation supports offline `prefix_tokens` stored in `.npz`; online PI05 extraction can call the same model/inference API once prefix tensors are available.

**Tech Stack:** Python 3.11 for model/training code, PyTorch for decoder training, NumPy for dataset files and FK targets, existing LIBERO/MuJoCo helpers for FK, existing `detect_swept_obstacle_collision()` for geometry safety checks.

---

## File Structure

- Create `scripts/libero_link_point_targets.py`: pure NumPy utilities for converting link skeleton segments `[T, L, 2, 3]` into fixed-topology points `[T, L, P, 3]`.
- Create `tests/test_libero_link_point_targets.py`: unit tests for fixed-topology sampling and validation.
- Create `safety_module/__init__.py`: package export for the decoder and geometry wrapper.
- Create `safety_module/point_decoder.py`: PyTorch config and model for `prefix_tokens -> link_points`.
- Create `tests/test_safety_point_decoder.py`: shape, masking, and trainability tests for the decoder.
- Create `safety_module/geometric_safety.py`: small inference wrapper that flattens predicted link points and calls existing geometric collision utilities.
- Create `tests/test_safety_geometric_safety.py`: OBB-based safety/collision tests for predicted link points.
- Create `scripts/build_pi05_safety_decoder_dataset.py`: offline dataset builder from `prefix_tokens`, real `action_chunks`, and start joint vectors to FK target link points.
- Create `tests/test_build_pi05_safety_decoder_dataset.py`: schema and pure helper tests for dataset builder logic.
- Create `scripts/train_pi05_safety_decoder.py`: train the decoder from an `.npz` dataset.
- Create `tests/test_train_pi05_safety_decoder.py`: smoke test for one optimizer step and checkpoint contents.
- Create `scripts/run_pi05_safety_decoder.py`: inference CLI that loads prefix tokens, model checkpoint, and safespace `.npz`, then writes predicted points and safety result.
- Create `tests/test_run_pi05_safety_decoder.py`: smoke test for inference using a tiny synthetic checkpoint and OBB.
- Modify `README.md`: add a concise section describing dataset build, training, and inference commands.

---

### Task 1: Fixed-Topology Link Point Targets

**Files:**
- Create: `scripts/libero_link_point_targets.py`
- Test: `tests/test_libero_link_point_targets.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_libero_link_point_targets.py`:

```python
import numpy as np
import pytest

from scripts.libero_link_point_targets import (
    flatten_link_points,
    sample_link_points_from_segments,
)


def test_sample_link_points_from_segments_keeps_time_link_point_topology():
    segment_path = np.asarray(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]],
            ],
            [
                [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]],
            ],
        ],
        dtype=np.float64,
    )

    points = sample_link_points_from_segments(segment_path, points_per_link=3)

    assert points.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(points[0, 0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(points[1, 1], [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]])


def test_flatten_link_points_preserves_row_major_time_link_point_order():
    link_points = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)

    flat = flatten_link_points(link_points)

    assert flat.shape == (24, 3)
    np.testing.assert_array_equal(flat[0], link_points[0, 0, 0])
    np.testing.assert_array_equal(flat[4], link_points[0, 1, 0])
    np.testing.assert_array_equal(flat[-1], link_points[1, 2, 3])


def test_sample_link_points_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="points_per_link must be >= 2"):
        sample_link_points_from_segments(np.zeros((2, 1, 2, 3)), points_per_link=1)

    with pytest.raises(ValueError, match="segment_path must have shape"):
        sample_link_points_from_segments(np.zeros((2, 1, 3)), points_per_link=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_libero_link_point_targets.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'scripts.libero_link_point_targets'`.

- [ ] **Step 3: Implement the target utility**

Create `scripts/libero_link_point_targets.py`:

```python
#!/usr/bin/env python3
"""Fixed-topology robot link-point target utilities."""

from __future__ import annotations

import numpy as np


def sample_link_points_from_segments(segment_path: np.ndarray, points_per_link: int) -> np.ndarray:
    """Sample fixed point indices along each link segment at every time step.

    Args:
        segment_path: Array with shape ``(T, L, 2, 3)``.
        points_per_link: Number of ordered samples per link segment.

    Returns:
        Array with shape ``(T, L, points_per_link, 3)``.
    """
    if points_per_link < 2:
        raise ValueError("points_per_link must be >= 2")
    segment_path = np.asarray(segment_path, dtype=np.float64)
    if segment_path.ndim != 4 or segment_path.shape[-2:] != (2, 3):
        raise ValueError(f"segment_path must have shape (T, L, 2, 3), got {segment_path.shape}")

    u = np.linspace(0.0, 1.0, int(points_per_link), dtype=np.float64)
    start = segment_path[:, :, 0, :]
    end = segment_path[:, :, 1, :]
    points = (1.0 - u[None, None, :, None]) * start[:, :, None, :]
    points += u[None, None, :, None] * end[:, :, None, :]
    return points.astype(np.float32)


def flatten_link_points(link_points: np.ndarray) -> np.ndarray:
    """Flatten ``(T, L, P, 3)`` link points to ``(T*L*P, 3)`` for collision checks."""
    link_points = np.asarray(link_points)
    if link_points.ndim != 4 or link_points.shape[-1] != 3:
        raise ValueError(f"link_points must have shape (T, L, P, 3), got {link_points.shape}")
    return link_points.reshape(-1, 3)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_libero_link_point_targets.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/libero_link_point_targets.py tests/test_libero_link_point_targets.py
git commit -m "Add fixed topology link point targets"
```

---

### Task 2: Safety Point Decoder Model

**Files:**
- Create: `safety_module/__init__.py`
- Create: `safety_module/point_decoder.py`
- Test: `tests/test_safety_point_decoder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_safety_point_decoder.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_safety_point_decoder.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'safety_module'`.

- [ ] **Step 3: Implement the model**

Create `safety_module/__init__.py`:

```python
"""Trainable safety modules for robot link-point prediction."""

from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig

__all__ = ["SafetyPointDecoder", "SafetyPointDecoderConfig"]
```

Create `safety_module/point_decoder.py`:

```python
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
    dropout: float = 0.0

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
        layers: list[nn.Module] = []
        in_dim = int(config.token_dim)
        for _ in range(int(config.num_layers)):
            layers.append(nn.Linear(in_dim, int(config.hidden_dim)))
            layers.append(nn.GELU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(float(config.dropout)))
            in_dim = int(config.hidden_dim)
        out_dim = int(config.horizon) * int(config.num_links) * int(config.points_per_link) * 3
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, prefix_tokens: torch.Tensor, prefix_mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = masked_mean_pool(prefix_tokens.to(dtype=torch.float32), prefix_mask)
        raw = self.net(pooled)
        return raw.reshape(
            prefix_tokens.shape[0],
            int(self.config.horizon),
            int(self.config.num_links),
            int(self.config.points_per_link),
            3,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_safety_point_decoder.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add safety_module/__init__.py safety_module/point_decoder.py tests/test_safety_point_decoder.py
git commit -m "Add PI05 latent point decoder model"
```

---

### Task 3: Geometric Safety Wrapper

**Files:**
- Create: `safety_module/geometric_safety.py`
- Modify: `safety_module/__init__.py`
- Test: `tests/test_safety_geometric_safety.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_safety_geometric_safety.py`:

```python
import numpy as np

from safety_module.geometric_safety import predicted_link_points_collision


def test_predicted_link_points_collision_reports_obb_hit():
    pred = np.asarray([[[[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]]]], dtype=np.float32)
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.2, 0.0, 0.0]], dtype=np.float64),
        "obstacle_box_axes": np.asarray([np.eye(3)], dtype=np.float64),
        "obstacle_box_half_sizes": np.asarray([[0.05, 0.05, 0.05]], dtype=np.float64),
    }

    result = predicted_link_points_collision(pred, safe_space)

    assert result.collides is True
    assert result.method == "oriented_boxes"
    assert result.collision_point_count == 1


def test_predicted_link_points_collision_reports_safe_when_no_point_overlaps():
    pred = np.asarray([[[[[1.0, 1.0, 1.0], [1.2, 1.0, 1.0]]]]], dtype=np.float32)
    safe_space = {
        "obstacle_box_centers": np.asarray([[0.2, 0.0, 0.0]], dtype=np.float64),
        "obstacle_box_axes": np.asarray([np.eye(3)], dtype=np.float64),
        "obstacle_box_half_sizes": np.asarray([[0.05, 0.05, 0.05]], dtype=np.float64),
    }

    result = predicted_link_points_collision(pred, safe_space)

    assert result.collides is False
    assert result.collision_point_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_safety_geometric_safety.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'safety_module.geometric_safety'`.

- [ ] **Step 3: Implement the wrapper**

Create `safety_module/geometric_safety.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np

from scripts.libero_joint_swept_pointcloud import CollisionResult, detect_swept_obstacle_collision
from scripts.libero_link_point_targets import flatten_link_points


def predicted_link_points_collision(
    pred_link_points: np.ndarray,
    safe_space: dict[str, np.ndarray],
    collision_margin: float = 0.0,
) -> CollisionResult:
    points = flatten_link_points(np.asarray(pred_link_points))
    return detect_swept_obstacle_collision(points, safe_space, collision_margin=collision_margin)


def collision_result_to_dict(result: CollisionResult) -> dict[str, Any]:
    return {
        "collision": bool(result.collides),
        "collision_method": str(result.method),
        "collision_margin": float(result.collision_margin),
        "collision_point_count": int(result.collision_point_count),
        "collision_point_indices": np.asarray(result.colliding_point_indices, dtype=np.int64),
    }
```

Modify `safety_module/__init__.py`:

```python
"""Trainable safety modules for robot link-point prediction."""

from safety_module.geometric_safety import predicted_link_points_collision
from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig

__all__ = ["SafetyPointDecoder", "SafetyPointDecoderConfig", "predicted_link_points_collision"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_safety_geometric_safety.py tests/test_libero_link_point_targets.py -v
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add safety_module/__init__.py safety_module/geometric_safety.py tests/test_safety_geometric_safety.py
git commit -m "Add geometric safety wrapper for predicted points"
```

---

### Task 4: FK Dataset Builder

**Files:**
- Create: `scripts/build_pi05_safety_decoder_dataset.py`
- Test: `tests/test_build_pi05_safety_decoder_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_pi05_safety_decoder_dataset.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from scripts.build_pi05_safety_decoder_dataset import (
    DatasetConfig,
    load_seed_samples,
    save_decoder_dataset,
    validate_seed_arrays,
)


def test_validate_seed_arrays_accepts_matching_prefix_actions_and_joints():
    prefix = np.zeros((3, 5, 7), dtype=np.float32)
    actions = np.zeros((3, 4, 6), dtype=np.float32)
    start_joints = np.zeros((3, 6), dtype=np.float32)

    validate_seed_arrays(prefix, actions, start_joints)


def test_validate_seed_arrays_rejects_mismatched_sample_count():
    prefix = np.zeros((3, 5, 7), dtype=np.float32)
    actions = np.zeros((2, 4, 6), dtype=np.float32)
    start_joints = np.zeros((3, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="same first dimension"):
        validate_seed_arrays(prefix, actions, start_joints)


def test_load_seed_samples_reads_required_arrays(tmp_path: Path):
    path = tmp_path / "seed_samples.npz"
    np.savez_compressed(
        path,
        prefix_tokens=np.ones((2, 3, 4), dtype=np.float32),
        action_chunks=np.ones((2, 5, 7), dtype=np.float32),
        start_joint_vectors=np.ones((2, 7), dtype=np.float32),
    )

    prefix, actions, start_joints = load_seed_samples(path)

    assert prefix.shape == (2, 3, 4)
    assert actions.shape == (2, 5, 7)
    assert start_joints.shape == (2, 7)


def test_save_decoder_dataset_writes_expected_schema(tmp_path: Path):
    output = tmp_path / "decoder_dataset.npz"
    config = DatasetConfig(task_suite="libero_spatial", task_id=0, init_state_id=0, points_per_link=4)
    save_decoder_dataset(
        output,
        prefix_tokens=np.zeros((2, 3, 4), dtype=np.float32),
        action_chunks=np.zeros((2, 5, 7), dtype=np.float32),
        start_joint_vectors=np.zeros((2, 7), dtype=np.float32),
        target_link_points=np.zeros((2, 6, 8, 4, 3), dtype=np.float32),
        link_names=np.asarray(["link0", "link1"]),
        config=config,
    )

    with np.load(output) as data:
        assert data["prefix_tokens"].shape == (2, 3, 4)
        assert data["target_link_points"].shape == (2, 6, 8, 4, 3)
        assert data["link_names"].tolist() == ["link0", "link1"]
        assert int(data["points_per_link"]) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_build_pi05_safety_decoder_dataset.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'scripts.build_pi05_safety_decoder_dataset'`.

- [ ] **Step 3: Implement dataset builder helpers and CLI**

Create `scripts/build_pi05_safety_decoder_dataset.py`:

```python
#!/usr/bin/env python3
"""Build PI05 latent safety-decoder datasets from prefix tokens and real action chunks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from libero_joint_swept_pointcloud import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    geom_skeleton_path,
    get_arm_qpos_indices,
    integrate_joint_path,
    joint_limits,
    load_runtime_dependencies,
    normalize_action_chunk,
)
from libero_link_point_targets import sample_link_points_from_segments  # noqa: E402
from libero_reconstruct_pointcloud import create_env, resolve_task, settle_scene  # noqa: E402
import libero_reconstruct_pointcloud as libero_pc  # noqa: E402


@dataclass(frozen=True)
class DatasetConfig:
    task_suite: str
    task_id: int
    init_state_id: int
    points_per_link: int
    samples_per_action: int = 1
    mujoco_gl: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-samples", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--bddl-file", type=Path, default=None)
    parser.add_argument("--points-per-link", type=int, default=8)
    parser.add_argument("--samples-per-action", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "pi05_safety_decoder_dataset.npz")
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default=None)
    return parser.parse_args()


def validate_seed_arrays(prefix_tokens: np.ndarray, action_chunks: np.ndarray, start_joint_vectors: np.ndarray) -> None:
    if prefix_tokens.ndim != 3:
        raise ValueError(f"prefix_tokens must have shape (S, N, D), got {prefix_tokens.shape}")
    if action_chunks.ndim != 3:
        raise ValueError(f"action_chunks must have shape (S, T, A), got {action_chunks.shape}")
    if start_joint_vectors.ndim != 2:
        raise ValueError(f"start_joint_vectors must have shape (S, J), got {start_joint_vectors.shape}")
    if not (prefix_tokens.shape[0] == action_chunks.shape[0] == start_joint_vectors.shape[0]):
        raise ValueError("prefix_tokens, action_chunks, and start_joint_vectors must have the same first dimension")


def load_seed_samples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        prefix_tokens = np.asarray(data["prefix_tokens"], dtype=np.float32)
        action_chunks = np.asarray(data["action_chunks"], dtype=np.float32)
        start_joint_vectors = np.asarray(data["start_joint_vectors"], dtype=np.float32)
    validate_seed_arrays(prefix_tokens, action_chunks, start_joint_vectors)
    return prefix_tokens, action_chunks, start_joint_vectors


def save_decoder_dataset(
    output: Path,
    *,
    prefix_tokens: np.ndarray,
    action_chunks: np.ndarray,
    start_joint_vectors: np.ndarray,
    target_link_points: np.ndarray,
    link_names: np.ndarray,
    config: DatasetConfig,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prefix_tokens=np.asarray(prefix_tokens, dtype=np.float32),
        action_chunks=np.asarray(action_chunks, dtype=np.float32),
        start_joint_vectors=np.asarray(start_joint_vectors, dtype=np.float32),
        target_link_points=np.asarray(target_link_points, dtype=np.float32),
        link_names=np.asarray(link_names),
        task_suite=np.asarray(config.task_suite),
        task_id=np.asarray(config.task_id),
        init_state_id=np.asarray(config.init_state_id),
        points_per_link=np.asarray(config.points_per_link),
        samples_per_action=np.asarray(config.samples_per_action),
    )


def fk_target_link_points(
    env,
    qpos_indices: np.ndarray,
    geom_ids: np.ndarray,
    start_joint_vector: np.ndarray,
    action_chunk: np.ndarray,
    points_per_link: int,
    samples_per_action: int,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actions = normalize_action_chunk(action_chunk, len(qpos_indices))
    joint_path = integrate_joint_path(start_joint_vector, actions, low, high, samples_per_action)
    segment_path, _geom_kinds, _geom_names, _color_ids, link_names = geom_skeleton_path(
        env,
        qpos_indices,
        np.asarray(geom_ids, dtype=np.int64),
        joint_path,
    )
    return sample_link_points_from_segments(segment_path, points_per_link), link_names


def main() -> None:
    args = parse_args()
    if args.points_per_link < 2:
        raise ValueError("--points-per-link must be >= 2")
    if args.samples_per_action < 1:
        raise ValueError("--samples-per-action must be >= 1")
    if args.mujoco_gl is not None:
        import os

        os.environ["MUJOCO_GL"] = args.mujoco_gl

    prefix_tokens, action_chunks, start_joint_vectors = load_seed_samples(args.seed_samples)
    load_runtime_dependencies()
    bddl_file, _task_name, init_state = resolve_task(args)
    env = create_env(bddl_file, width=64, height=64, camera_names=["agentview"])
    try:
        settle_scene(env, init_state, num_steps=10)
        qpos_indices = get_arm_qpos_indices(env)
        low, high = joint_limits(env.sim, qpos_indices)
        geom_ids = libero_pc.find_robot_geoms(env)
        targets = []
        link_names = np.asarray([])
        for sample_idx in range(prefix_tokens.shape[0]):
            target, link_names = fk_target_link_points(
                env,
                qpos_indices,
                np.asarray(geom_ids, dtype=np.int64),
                start_joint_vectors[sample_idx],
                action_chunks[sample_idx],
                args.points_per_link,
                args.samples_per_action,
                low,
                high,
            )
            targets.append(target)
        save_decoder_dataset(
            args.output,
            prefix_tokens=prefix_tokens,
            action_chunks=action_chunks,
            start_joint_vectors=start_joint_vectors,
            target_link_points=np.stack(targets).astype(np.float32),
            link_names=link_names,
            config=DatasetConfig(
                task_suite=args.task_suite,
                task_id=args.task_id,
                init_state_id=args.init_state_id,
                points_per_link=args.points_per_link,
                samples_per_action=args.samples_per_action,
                mujoco_gl=args.mujoco_gl,
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_build_pi05_safety_decoder_dataset.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_pi05_safety_decoder_dataset.py tests/test_build_pi05_safety_decoder_dataset.py
git commit -m "Add PI05 safety decoder dataset builder"
```

---

### Task 5: Training Script

**Files:**
- Create: `scripts/train_pi05_safety_decoder.py`
- Test: `tests/test_train_pi05_safety_decoder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_pi05_safety_decoder.py`:

```python
from pathlib import Path

import numpy as np
import torch

from scripts.train_pi05_safety_decoder import load_dataset_tensors, train_one_epoch
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_train_pi05_safety_decoder.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'scripts.train_pi05_safety_decoder'`.

- [ ] **Step 3: Implement training script**

Create `scripts/train_pi05_safety_decoder.py`:

```python
#!/usr/bin/env python3
"""Train PI05 prefix-token safety decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_dataset_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        prefix = torch.from_numpy(np.asarray(data["prefix_tokens"], dtype=np.float32))
        targets = torch.from_numpy(np.asarray(data["target_link_points"], dtype=np.float32))
    if prefix.ndim != 3:
        raise ValueError(f"prefix_tokens must have shape (S, N, D), got {tuple(prefix.shape)}")
    if targets.ndim != 5 or targets.shape[-1] != 3:
        raise ValueError(f"target_link_points must have shape (S, T, L, P, 3), got {tuple(targets.shape)}")
    if prefix.shape[0] != targets.shape[0]:
        raise ValueError("prefix_tokens and target_link_points must have the same sample count")
    return prefix, targets


def train_one_epoch(
    model: SafetyPointDecoder,
    optimizer: torch.optim.Optimizer,
    prefix: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    model.train()
    order = torch.randperm(prefix.shape[0])
    total = 0.0
    count = 0
    for start in range(0, prefix.shape[0], batch_size):
        idx = order[start : start + batch_size]
        batch_prefix = prefix[idx].to(device)
        batch_targets = targets[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch_prefix)
        loss = F.smooth_l1_loss(pred, batch_targets)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * batch_prefix.shape[0]
        count += batch_prefix.shape[0]
    return total / max(count, 1)


def save_checkpoint(path: Path, model: SafetyPointDecoder, config: SafetyPointDecoderConfig, epoch: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "epoch": int(epoch),
            "loss": float(loss),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps({"config": config.to_dict(), "epoch": int(epoch), "loss": float(loss)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    prefix, targets = load_dataset_tensors(args.dataset)
    config = SafetyPointDecoderConfig(
        token_dim=int(prefix.shape[-1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        horizon=int(targets.shape[1]),
        num_links=int(targets.shape[2]),
        points_per_link=int(targets.shape[3]),
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    model = SafetyPointDecoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, prefix, targets, args.batch_size, device)
        print(f"epoch={epoch} loss={loss:.6f}")
    save_checkpoint(args.output, model, config, args.epochs, loss)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_train_pi05_safety_decoder.py tests/test_safety_point_decoder.py -v
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_pi05_safety_decoder.py tests/test_train_pi05_safety_decoder.py
git commit -m "Add PI05 safety decoder training script"
```

---

### Task 6: Inference CLI

**Files:**
- Create: `scripts/run_pi05_safety_decoder.py`
- Test: `tests/test_run_pi05_safety_decoder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_pi05_safety_decoder.py`:

```python
from pathlib import Path

import numpy as np
import torch

from scripts.run_pi05_safety_decoder import load_checkpoint_model, run_prediction
from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig


def test_run_prediction_returns_points_and_geometric_collision(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    config = SafetyPointDecoderConfig(token_dim=4, hidden_dim=8, num_layers=1, horizon=1, num_links=1, points_per_link=2)
    model = SafetyPointDecoder(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict(), "epoch": 1, "loss": 0.0}, checkpoint)

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_run_pi05_safety_decoder.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'scripts.run_pi05_safety_decoder'`.

- [ ] **Step 3: Implement inference CLI**

Create `scripts/run_pi05_safety_decoder.py`:

```python
#!/usr/bin/env python3
"""Run PI05 latent safety decoder and geometric collision check."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from safety_module.geometric_safety import collision_result_to_dict, predicted_link_points_collision
from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig
from scripts.libero_joint_swept_pointcloud import load_npz_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=Path, required=True)
    parser.add_argument("--safe-space", type=Path, required=True)
    parser.add_argument("--collision-margin", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint_model(path: Path, device: torch.device) -> SafetyPointDecoder:
    payload = torch.load(path, map_location=device)
    config = SafetyPointDecoderConfig.from_dict(payload["config"])
    model = SafetyPointDecoder(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def load_prefix_tokens(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        prefix = np.load(path)
    else:
        with np.load(path, allow_pickle=False) as data:
            prefix = data["prefix_tokens"]
    prefix = np.asarray(prefix, dtype=np.float32)
    if prefix.ndim == 2:
        prefix = prefix[None, ...]
    if prefix.ndim != 3:
        raise ValueError(f"prefix_tokens must have shape (B, N, D) or (N, D), got {prefix.shape}")
    return prefix


@torch.no_grad()
def run_prediction(
    model: SafetyPointDecoder,
    prefix_tokens: np.ndarray,
    safe_space: dict[str, np.ndarray],
    collision_margin: float,
    device: torch.device,
) -> tuple[np.ndarray, object]:
    prefix = torch.from_numpy(np.asarray(prefix_tokens, dtype=np.float32)).to(device)
    pred = model(prefix).detach().cpu().numpy().astype(np.float32)
    result = predicted_link_points_collision(pred[0], safe_space, collision_margin=collision_margin)
    return pred, result


def save_prediction(path: Path, pred_link_points: np.ndarray, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result_payload = collision_result_to_dict(result)
    np.savez_compressed(path, pred_link_points=pred_link_points, **result_payload)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_checkpoint_model(args.checkpoint, device)
    prefix_tokens = load_prefix_tokens(args.prefix_tokens)
    safe_space = load_npz_dict(args.safe_space)
    pred, result = run_prediction(model, prefix_tokens, safe_space, args.collision_margin, device)
    save_prediction(args.output, pred, result)
    print("collision" if result.collides else "safe")
    print(f"collision_point_count={result.collision_point_count}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_run_pi05_safety_decoder.py tests/test_safety_geometric_safety.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pi05_safety_decoder.py tests/test_run_pi05_safety_decoder.py
git commit -m "Add PI05 safety decoder inference CLI"
```

---

### Task 7: README Usage

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add documentation section**

Append this section after the LIBERO swept-point section:

```markdown
## 3. PI05 latent safety decoder

第一版 safety decoder 使用 PI05 VLM `prefix_tokens` 预测未来连杆点，不直接预测碰撞分类。安全信号由预测点和障碍物 OBB / occupied grid 的几何重叠计算得到。

准备训练 seed 数据，要求 `.npz` 至少包含：

```text
prefix_tokens: shape [S, N, D]
action_chunks: shape [S, T, A]
start_joint_vectors: shape [S, J]
```

用真实 action chunk 经 FK 生成训练目标：

```bash
$LIBERO_PY scripts/build_pi05_safety_decoder_dataset.py \
  --seed-samples outputs/pi05_prefix_seed_samples.npz \
  --task-suite libero_spatial \
  --task-id 0 \
  --points-per-link 8 \
  --samples-per-action 1 \
  --output outputs/pi05_safety_decoder/libero_spatial_task0_decoder_dataset.npz \
  --mujoco-gl egl
```

训练 decoder：

```bash
python scripts/train_pi05_safety_decoder.py \
  --dataset outputs/pi05_safety_decoder/libero_spatial_task0_decoder_dataset.npz \
  --output outputs/pi05_safety_decoder/decoder.pt \
  --hidden-dim 512 \
  --num-layers 4 \
  --epochs 50 \
  --batch-size 64
```

推理并用几何计算输出 `collision` 或 `safe`：

```bash
python scripts/run_pi05_safety_decoder.py \
  --checkpoint outputs/pi05_safety_decoder/decoder.pt \
  --prefix-tokens outputs/pi05_prefix_tokens/current_prefix_tokens.npz \
  --safe-space outputs/safe_space/${TASK}_tabletop_xy_oriented_obstacle_obb_safe_space.npz \
  --collision-margin 0.01 \
  --output outputs/pi05_safety_decoder/current_safety_result.npz
```
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
python -m pytest \
  tests/test_libero_link_point_targets.py \
  tests/test_safety_point_decoder.py \
  tests/test_safety_geometric_safety.py \
  tests/test_build_pi05_safety_decoder_dataset.py \
  tests/test_train_pi05_safety_decoder.py \
  tests/test_run_pi05_safety_decoder.py
```

Expected: PASS for all tests.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document PI05 latent safety decoder workflow"
```

---

### Task 8: Final Verification

**Files:**
- Verify all files from Tasks 1-7.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
python -m pytest \
  tests/test_libero_link_point_targets.py \
  tests/test_safety_point_decoder.py \
  tests/test_safety_geometric_safety.py \
  tests/test_build_pi05_safety_decoder_dataset.py \
  tests/test_train_pi05_safety_decoder.py \
  tests/test_run_pi05_safety_decoder.py
```

Expected: PASS.

- [ ] **Step 2: Run existing swept-point tests to catch regressions**

Run:

```bash
python -m pytest tests/test_libero_joint_swept_pointcloud.py -v
```

Expected: PASS.

- [ ] **Step 3: Compile new scripts**

Run:

```bash
python -m py_compile \
  scripts/libero_link_point_targets.py \
  scripts/build_pi05_safety_decoder_dataset.py \
  scripts/train_pi05_safety_decoder.py \
  scripts/run_pi05_safety_decoder.py \
  safety_module/point_decoder.py \
  safety_module/geometric_safety.py
```

Expected: command exits with status 0.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: no uncommitted changes from this plan. Pre-existing unrelated changes may remain if they were present before execution.

---

## Self-Review

- Spec coverage: covered prefix-token input, fixed-topology point prediction, FK target generation from real action chunks, no classification head, deterministic geometry safety decision, training, inference, and README workflow.
- Placeholder scan: no placeholder tokens or unspecified implementation steps remain.
- Type consistency: the plan uses `prefix_tokens [S/B, N, D]`, `target_link_points [S/B, T, L, P, 3]`, and `pred_link_points [B, T, L, P, 3]` consistently across model, training, inference, and collision wrapper.
