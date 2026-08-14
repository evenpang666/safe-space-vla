"""Standalone real-robot dataset schema for VLA-conditioned surface point flow.

This module intentionally does not import simulation collectors.  All position
fields are expressed in the calibrated UR controller base frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


COORDINATE_FRAME = "ur_base"
OFFSET_FRAME = "ur_base_delta"


def derive_flow_targets(link_trajectory: np.ndarray) -> dict[str, np.ndarray]:
    trajectory = np.asarray(link_trajectory, dtype=np.float32)
    if trajectory.ndim != 4 or trajectory.shape[-1] != 3 or trajectory.shape[0] < 2:
        raise ValueError(f"link_trajectory must have shape (T>=2, L, P, 3), got {trajectory.shape}")
    current = trajectory[0]
    future_offsets = trajectory[1:] - current[None, ...]
    return {
        "current_link_points": current.astype(np.float32),
        "future_link_offsets": future_offsets.astype(np.float32),
        "arm_points": current.reshape(-1, 3).astype(np.float32),
        "target_point_offsets": future_offsets.reshape(future_offsets.shape[0], -1, 3).astype(np.float32),
    }


@dataclass
class RealSafetyDatasetBuffer:
    prefix_tokens: list[np.ndarray] = field(default_factory=list)
    action_chunks: list[np.ndarray] = field(default_factory=list)
    start_joint_vectors: list[np.ndarray] = field(default_factory=list)
    target_link_points: list[np.ndarray] = field(default_factory=list)
    current_link_points: list[np.ndarray] = field(default_factory=list)
    future_link_offsets: list[np.ndarray] = field(default_factory=list)
    arm_points: list[np.ndarray] = field(default_factory=list)
    target_point_offsets: list[np.ndarray] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)
    step_ids: list[int] = field(default_factory=list)

    def append(
        self,
        *,
        prefix_tokens: np.ndarray,
        action_chunk: np.ndarray,
        start_joint_vector: np.ndarray,
        link_trajectory: np.ndarray,
        episode_id: int,
        step_id: int,
    ) -> None:
        targets = derive_flow_targets(link_trajectory)
        prefix = np.asarray(prefix_tokens, dtype=np.float32)
        actions = np.asarray(action_chunk, dtype=np.float32)
        if prefix.ndim != 2 or actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected prefix (N,D) and absolute 7-D action chunk, got {prefix.shape}, {actions.shape}")
        if actions.shape[0] != targets["future_link_offsets"].shape[0]:
            raise ValueError("Action horizon and real surface trajectory horizon differ")
        self.prefix_tokens.append(prefix)
        self.action_chunks.append(actions)
        self.start_joint_vectors.append(np.asarray(start_joint_vector, dtype=np.float32).reshape(6))
        self.target_link_points.append(np.asarray(link_trajectory, dtype=np.float32))
        self.current_link_points.append(targets["current_link_points"])
        self.future_link_offsets.append(targets["future_link_offsets"])
        self.arm_points.append(targets["arm_points"])
        self.target_point_offsets.append(targets["target_point_offsets"])
        self.episode_ids.append(int(episode_id))
        self.step_ids.append(int(step_id))

    def __len__(self) -> int:
        return len(self.prefix_tokens)


def save_real_safety_dataset(
    output: Path,
    *,
    buffer: RealSafetyDatasetBuffer,
    link_names: np.ndarray,
    point_ids: np.ndarray,
    local_link_points: np.ndarray,
    surface_model_hash: str,
    point_identity_version: str,
    policy_config: str,
    action_mode: str,
    control_hz: float,
) -> None:
    if len(buffer) == 0:
        raise ValueError("No complete real-robot future-horizon samples were collected")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prefix_tokens=np.stack(buffer.prefix_tokens).astype(np.float32),
        action_chunks=np.stack(buffer.action_chunks).astype(np.float32),
        start_joint_vectors=np.stack(buffer.start_joint_vectors).astype(np.float32),
        target_link_points=np.stack(buffer.target_link_points).astype(np.float32),
        current_link_points=np.stack(buffer.current_link_points).astype(np.float32),
        future_link_offsets=np.stack(buffer.future_link_offsets).astype(np.float32),
        arm_points=np.stack(buffer.arm_points).astype(np.float32),
        target_point_offsets=np.stack(buffer.target_point_offsets).astype(np.float32),
        episode_ids=np.asarray(buffer.episode_ids, dtype=np.int64),
        step_ids=np.asarray(buffer.step_ids, dtype=np.int64),
        link_names=np.asarray(link_names),
        point_ids=np.asarray(point_ids, dtype=np.int32),
        local_link_points=np.asarray(local_link_points, dtype=np.float32),
        coordinate_frame=np.asarray(COORDINATE_FRAME),
        target_link_points_frame=np.asarray(COORDINATE_FRAME),
        current_link_points_frame=np.asarray(COORDINATE_FRAME),
        arm_points_frame=np.asarray(COORDINATE_FRAME),
        future_link_offsets_frame=np.asarray(OFFSET_FRAME),
        target_point_offsets_frame=np.asarray(OFFSET_FRAME),
        surface_model_hash=np.asarray(surface_model_hash),
        point_identity_version=np.asarray(point_identity_version),
        task_suite=np.asarray("real_ur7e"),
        skeleton_source=np.asarray("ur7e_pika_collision_surface"),
        target_source=np.asarray("measured_real_rollout_surface"),
        policy_config=np.asarray(policy_config),
        action_mode=np.asarray(action_mode),
        control_hz=np.asarray(float(control_hz), dtype=np.float32),
    )
