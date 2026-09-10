#!/usr/bin/env python3
"""Preprocess a left-arm VR HDF5 episode into fixed real collision-surface flow.

The Quest recorder stores left UR7e joint states, Robotiq register-normalized
opening, actions, and RGB-D, but no right-arm state.  This tool deliberately
exports only the recorded left arm: seven official UR7e collision links plus
the ten upstream Robotiq 2F-85 collision-mesh links.  It never fabricates a
stationary right arm.

The source episode does not record the active TCP transform.  Pass the static
``flange_to_active_tcp`` matrix measured from the unchanged pendant TCP.  The
matrix is embedded in the result, making the output auditable and repeatable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.robotiq_vendor_visual_models import (  # noqa: E402
    TWO_F85_COLLISION,
    TWO_F85_LINK_NAMES,
    TWO_F85_MESH_NAMES,
    robotiq_2f85_link_points_in_gripper_base,
)
from real_scripts.ur7e_collision_mesh import (  # noqa: E402
    UR7eCollisionSurfacePointSampler,
    flange_transform,
)


UR_LINK_NAMES = tuple(f"left_ur7e_{name}" for name in ("base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3"))
POINT_IDENTITY_VERSION = "left_ur7e_robotiq_2f85_collision_surface_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="Quest left-VR HDF5 episode.")
    parser.add_argument("--output", type=Path, required=True, help="Compressed surface-flow NPZ output.")
    parser.add_argument("--visualization", type=Path, required=True, help="Self-contained Plotly HTML animation output.")
    parser.add_argument("--preview", type=Path, default=None, help="Optional static PNG overview of the first, middle, and final frame.")
    parser.add_argument("--flange-to-active-tcp-json", type=Path, required=True, help="JSON/YAML 4x4 matrix from the active pendant TCP.")
    parser.add_argument("--points-per-link", type=int, default=128)
    parser.add_argument("--future-horizon", type=int, default=8)
    parser.add_argument("--robotiq-base-to-active-tcp-m", type=float, default=.1493, help="Upstream 2F-85 gripper-base to active-TCP length used by the live mesh model.")
    parser.add_argument("--visualization-max-frames", type=int, default=151)
    parser.add_argument("--visualization-points-per-link", type=int, default=12)
    return parser.parse_args()


def _load_matrix(path: Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        value = yaml.safe_load(text)
    if isinstance(value, dict):
        value = value.get("matrix", value.get("flange_to_active_tcp", value.get("transform")))
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0., 0., 0., 1.)):
        raise ValueError(f"{path} must contain a finite 4x4 homogeneous matrix")
    return matrix


def _translation(z_m: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = float(z_m)
    return transform


def _transform_link_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return ((transform[:3, :3] @ values.reshape(-1, 3).T).T + transform[:3, 3]).reshape(values.shape).astype(np.float32)


def _surface_hash(ur_hash: str, flange_to_active_tcp: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(ur_hash).encode())
    hasher.update(np.asarray(flange_to_active_tcp, dtype=np.float64).tobytes())
    for name in TWO_F85_MESH_NAMES:
        hasher.update((TWO_F85_COLLISION / name).read_bytes())
    return hasher.hexdigest()


def _sample_indices(action_valid: np.ndarray, *, horizon: int) -> np.ndarray:
    valid = np.asarray(action_valid, dtype=bool)
    if len(valid) <= horizon:
        return np.empty((0,), dtype=np.int64)
    return np.asarray([start for start in range(len(valid) - horizon) if bool(valid[start : start + horizon].all())], dtype=np.int64)


def _make_html(
    output: Path,
    *,
    fixed_link_points: np.ndarray,
    link_names: tuple[str, ...],
    timestamps_ns: np.ndarray,
    max_frames: int,
    points_per_link: int,
) -> None:
    import plotly.graph_objects as go

    trajectory = np.asarray(fixed_link_points, dtype=np.float32)
    frame_indices = np.unique(np.linspace(0, len(trajectory) - 1, min(int(max_frames), len(trajectory)), dtype=np.int64))
    sample_indices = np.linspace(0, trajectory.shape[2] - 1, min(int(points_per_link), trajectory.shape[2]), dtype=np.int64)
    palette = ("#8ec5ff", "#36d9b0", "#f9bf4d", "#ff8a65", "#df8cff", "#73a7ff", "#c0df61", "#f15bb5", "#ff6b6b", "#ffd166", "#06d6a0", "#4cc9f0", "#9b5de5", "#f72585", "#90be6d", "#f9844a", "#c77dff")

    def traces(index: int) -> list[go.Scatter3d]:
        return [
            go.Scatter3d(
                x=trajectory[index, link, sample_indices, 0], y=trajectory[index, link, sample_indices, 1], z=trajectory[index, link, sample_indices, 2],
                mode="markers", name=link_names[link], marker={"size": 3.0, "color": palette[link % len(palette)]}, hoverinfo="name",
            )
            for link in range(trajectory.shape[1])
        ]

    frames = [go.Frame(name=str(int(index)), data=traces(int(index)), layout={"title": {"text": f"left arm surface flow · frame {index + 1}/{len(trajectory)} · t={(timestamps_ns[index] - timestamps_ns[0]) / 1e9:.2f}s"}}) for index in frame_indices]
    figure = go.Figure(
        data=traces(int(frame_indices[0])),
        frames=frames,
        layout=go.Layout(
            title=f"Left UR7e + Robotiq 2F-85 fixed collision-surface flow ({len(trajectory)} recorded frames)",
            paper_bgcolor="#080a0d", plot_bgcolor="#080a0d", font={"color": "#e6edf3"}, showlegend=True,
            legend={"font": {"size": 10}},
            scene={"bgcolor": "#080a0d", "aspectmode": "data", "xaxis": {"title": "left_base X (m)"}, "yaxis": {"title": "left_base Y (m)"}, "zaxis": {"title": "left_base Z (m)"}},
            updatemenus=[{"type": "buttons", "showactive": False, "x": .01, "y": .02, "buttons": [
                {"label": "播放", "method": "animate", "args": [None, {"frame": {"duration": 65, "redraw": True}, "fromcurrent": True}]},
                {"label": "暂停", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
            ]}],
            sliders=[{"active": 0, "x": .15, "len": .8, "y": .02, "steps": [{"label": str(int(index)), "method": "animate", "args": [[str(int(index))], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for index in frame_indices]}],
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True, auto_open=False)


def _make_preview(output: Path, *, fixed_link_points: np.ndarray, link_names: tuple[str, ...]) -> None:
    """Write a compact static companion to the interactive animation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectory = np.asarray(fixed_link_points, dtype=np.float32)
    indices = (0, len(trajectory) // 2, len(trajectory) - 1)
    palette = ("#8ec5ff", "#36d9b0", "#f9bf4d", "#ff8a65", "#df8cff", "#73a7ff", "#c0df61", "#f15bb5", "#ff6b6b", "#ffd166", "#06d6a0", "#4cc9f0", "#9b5de5", "#f72585", "#90be6d", "#f9844a", "#c77dff")
    points = trajectory[list(indices)].reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) * .5
    radius = float(np.max(high - low) * .55)
    figure = plt.figure(figsize=(16, 5.4), facecolor="#080a0d")
    for axis_index, frame_index in enumerate(indices, start=1):
        axis = figure.add_subplot(1, 3, axis_index, projection="3d", facecolor="#080a0d")
        for link_index, name in enumerate(link_names):
            cloud = trajectory[frame_index, link_index]
            axis.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=2.0, color=palette[link_index % len(palette)], label=name if axis_index == 1 else None)
        axis.set_title(f"frame {frame_index + 1}/{len(trajectory)}", color="#e6edf3")
        axis.set_xlim(center[0] - radius, center[0] + radius); axis.set_ylim(center[1] - radius, center[1] + radius); axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_xlabel("X (m)", color="#c8d1dc"); axis.set_ylabel("Y (m)", color="#c8d1dc"); axis.set_zlabel("Z (m)", color="#c8d1dc")
        axis.tick_params(colors="#aeb7c2", labelsize=7)
        axis.view_init(elev=23, azim=-58)
    legend = figure.legend(loc="center left", bbox_to_anchor=(.91, .5), frameon=False, fontsize=7)
    for text in legend.get_texts(): text.set_color("#e6edf3")
    figure.suptitle("Left UR7e + Robotiq 2F-85 fixed collision-surface flow", color="#f2f6fa", fontsize=14)
    figure.tight_layout(rect=(0, 0, .88, .94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.points_per_link < 2 or args.future_horizon < 1 or args.visualization_max_frames < 1 or args.visualization_points_per_link < 1:
        raise ValueError("points-per-link, future-horizon, and visualization counts must be positive")
    if args.robotiq_base_to_active_tcp_m <= 0.0:
        raise ValueError("robotiq-base-to-active-tcp-m must be positive")
    flange_to_active_tcp = _load_matrix(args.flange_to_active_tcp_json)
    # ``robotiq_2f85_points_in_active_tcp`` maps root-frame mesh points into
    # active-TCP coordinates with a -149.3 mm Z translation.  Compose that
    # same root→TCP map with ^flangeT_tcp; do not invert it, or the gripper is
    # displaced by roughly twice its fingertip length.
    flange_to_gripper_root = flange_to_active_tcp @ _translation(-args.robotiq_base_to_active_tcp_m)
    with h5py.File(args.episode, "r") as source:
        required = ("observations/qpos", "observations/gripper_position", "action", "action_valid", "executed_action", "gripper_action", "timestamps_ns")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Episode is missing required datasets: {missing}")
        qpos = np.asarray(source["observations/qpos"], dtype=np.float32)
        gripper = np.asarray(source["observations/gripper_position"], dtype=np.float32).reshape(-1)
        action = np.asarray(source["action"], dtype=np.float32)
        action_valid = np.asarray(source["action_valid"], dtype=bool)
        executed_action = np.asarray(source["executed_action"], dtype=np.float32)
        gripper_action = np.asarray(source["gripper_action"], dtype=np.float32)
        timestamps_ns = np.asarray(source["timestamps_ns"], dtype=np.int64)
        source_attrs = {str(key): source.attrs[key] for key in ("mode", "created_utc", "action_definition", "executed_action_definition") if key in source.attrs}
    length = len(qpos)
    if qpos.shape != (length, 6) or action.shape != (length, 6) or executed_action.shape != (length, 6) or gripper.shape != (length,) or action_valid.shape != (length,) or timestamps_ns.shape != (length,):
        raise ValueError("Episode datasets have inconsistent left-arm shapes")
    if not np.isfinite(qpos).all() or not np.isfinite(gripper).all() or not np.isfinite(action).all() or not np.isfinite(executed_action).all():
        raise ValueError("Episode contains non-finite left-arm values")

    ur_sampler = UR7eCollisionSurfacePointSampler(points_per_link=args.points_per_link)
    # Use the measured, normalized Robotiq position convention used by the
    # collector: 0=open and 1=closed; upstream joint travel is 0..0.8 rad.
    fixed = np.empty((length, len(UR_LINK_NAMES) + len(TWO_F85_LINK_NAMES), args.points_per_link, 3), dtype=np.float32)
    for frame_index, (joint_position, close_fraction) in enumerate(zip(qpos, gripper, strict=True)):
        fixed[frame_index, : len(UR_LINK_NAMES)] = ur_sampler.link_points(joint_position)
        gripper_local = robotiq_2f85_link_points_in_gripper_base(
            finger_angle_rad=float(np.clip(close_fraction, 0.0, 1.0) * .8), samples_per_link=args.points_per_link
        )
        fixed[frame_index, len(UR_LINK_NAMES) :] = _transform_link_points(gripper_local, flange_transform(joint_position) @ flange_to_gripper_root)

    sample_indices = _sample_indices(action_valid, horizon=args.future_horizon)
    if len(sample_indices) == 0:
        raise ValueError("No complete future-horizon action windows in episode")
    arm_points = fixed.reshape(length, -1, 3)
    future_offsets = np.stack([arm_points[index + 1 : index + args.future_horizon + 1] - arm_points[index] for index in sample_indices]).astype(np.float32)
    link_names = UR_LINK_NAMES + TWO_F85_LINK_NAMES
    link_ids = np.arange(len(link_names), dtype=np.int32)[:, None]
    point_ids = np.stack((np.broadcast_to(link_ids, (len(link_names), args.points_per_link)), np.broadcast_to(np.arange(args.points_per_link, dtype=np.int32)[None, :], (len(link_names), args.points_per_link))), axis=-1)
    local_link_points = np.concatenate((ur_sampler.local_link_points, robotiq_2f85_link_points_in_gripper_base(finger_angle_rad=0.0, samples_per_link=args.points_per_link)), axis=0).astype(np.float32)
    control_hz = float(1e9 / np.median(np.diff(timestamps_ns))) if len(timestamps_ns) > 1 else float("nan")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format_version=np.asarray(1, dtype=np.int32),
        source_episode=np.asarray(str(Path(args.episode).resolve())),
        source_attrs=np.asarray(json.dumps(source_attrs, default=str)),
        coordinate_frame=np.asarray("left_base"),
        source_arm=np.asarray("left_arm"),
        right_arm_present=np.asarray(False),
        depth_removed=np.asarray(True),
        qpos=qpos,
        gripper_position=gripper,
        gripper_joint_angle_rad=np.clip(gripper, 0.0, 1.0).astype(np.float32) * .8,
        actions=action,
        executed_actions=executed_action,
        gripper_actions=gripper_action,
        action_valid=action_valid,
        timestamps_ns=timestamps_ns,
        control_hz=np.asarray(control_hz, dtype=np.float32),
        fixed_link_points=fixed,
        arm_points=arm_points,
        link_names=np.asarray(link_names),
        point_ids=point_ids,
        local_link_points=local_link_points,
        point_identity_version=np.asarray(POINT_IDENTITY_VERSION),
        surface_model_hash=np.asarray(_surface_hash(ur_sampler.mesh_model_hash, flange_to_active_tcp)),
        flange_to_active_tcp=flange_to_active_tcp.astype(np.float64),
        flange_to_robotiq_root=flange_to_gripper_root.astype(np.float64),
        robotiq_base_to_active_tcp_m=np.asarray(args.robotiq_base_to_active_tcp_m, dtype=np.float32),
        sample_frame_indices=sample_indices,
        action_chunks=np.stack([action[index : index + args.future_horizon] for index in sample_indices]).astype(np.float32),
        executed_action_chunks=np.stack([executed_action[index : index + args.future_horizon] for index in sample_indices]).astype(np.float32),
        current_link_points=fixed[sample_indices],
        future_link_offsets=future_offsets.reshape(len(sample_indices), args.future_horizon, len(link_names), args.points_per_link, 3),
        target_point_offsets=future_offsets,
    )
    _make_html(Path(args.visualization), fixed_link_points=fixed, link_names=link_names, timestamps_ns=timestamps_ns, max_frames=args.visualization_max_frames, points_per_link=args.visualization_points_per_link)
    if args.preview is not None:
        _make_preview(Path(args.preview), fixed_link_points=fixed, link_names=link_names)
    print(json.dumps({"frames": length, "flow_windows": len(sample_indices), "links": len(link_names), "points_per_link": args.points_per_link, "stable_points": int(arm_points.shape[1]), "control_hz": control_hz, "right_arm_present": False, "output": str(output), "visualization": str(args.visualization), "preview": "" if args.preview is None else str(args.preview)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
