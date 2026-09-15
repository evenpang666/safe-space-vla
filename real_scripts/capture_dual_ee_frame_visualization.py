#!/usr/bin/env python3
"""Capture read-only dual-UR7e flange/active-TCP frame visualizations.

This is the first, non-moving step for aligning end-effector mounting
directions.  It writes a front-camera overlay and a 3-D Plotly HTML scene in
``left_base``.  It never opens RTDE control or sends robot/gripper commands.

The repository includes vendor Robotiq 2F-85 and EPick CAD/URDF models.  The
output shows the two measured frames used to place those models:

* ``flange``: UR7e FK frame from the current receive-only joint state;
* ``active_tcp``: the pose currently configured on each robot controller.

Axes follow the standard convention: X red, Y green, Z blue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.live_dual_ur7e_obstacle_model import load_arms, rotvec_transform
from real_scripts.robotiq_vendor_visual_models import robotiq_2f85_points_in_active_tcp, robotiq_epick_points_in_active_tcp
from real_scripts.ur7e_collision_mesh import flange_transform
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource


AXIS_COLORS_BGR = ((45, 45, 235), (70, 190, 70), (235, 135, 45))  # X/Y/Z
AXIS_NAMES = ("X", "Y", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", type=Path, default=REPO_ROOT.parent / "quest3_collect" / "config" / "hardware_dual_teleop.yaml")
    parser.add_argument("--front-serial", default=None)
    parser.add_argument("--width", type=int, default=1280); parser.add_argument("--height", type=int, default=720); parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=15); parser.add_argument("--axis-length-m", type=float, default=.100)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "dual_ee_frames")
    return parser.parse_args()


def project(points: np.ndarray, camera_to_world: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera = (np.linalg.inv(camera_to_world) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    valid = camera[:, 2] > 1e-5
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = K[0, 0] * camera[valid, 0] / camera[valid, 2] + K[0, 2]
    uv[valid, 1] = K[1, 1] * camera[valid, 1] / camera[valid, 2] + K[1, 2]
    return uv, valid


def frame_points(transform: np.ndarray, length_m: float) -> np.ndarray:
    """Return origin, +X, +Y, +Z in the target world frame."""
    basis = np.vstack((np.zeros(3), np.eye(3) * float(length_m)))
    return (transform[:3, :3] @ basis.T).T + transform[:3, 3]


def draw_frame(image: np.ndarray, transform: np.ndarray, *, name: str, camera_to_world: np.ndarray, K: np.ndarray, label_color: tuple[int, int, int]) -> None:
    points = frame_points(transform, .100)
    uv, valid = project(points, camera_to_world, K)
    height, width = image.shape[:2]
    visible = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not visible[0]:
        return
    origin = tuple(np.rint(uv[0]).astype(int))
    cv2.circle(image, origin, 5, label_color, -1, lineType=cv2.LINE_AA)
    cv2.putText(image, name, (origin[0] + 7, origin[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, .55, label_color, 2, cv2.LINE_AA)
    for index, (axis, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS_BGR), start=1):
        if visible[index]:
            end = tuple(np.rint(uv[index]).astype(int)); cv2.arrowedLine(image, origin, end, color, 3, cv2.LINE_AA, tipLength=.14)
            cv2.putText(image, axis, (end[0] + 3, end[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)


def draw_model_points(image: np.ndarray, points: np.ndarray, *, camera_to_world: np.ndarray, K: np.ndarray, color: tuple[int, int, int]) -> None:
    """Overlay a sparse URDF collision-surface sample without hiding the camera image."""
    uv, valid = project(points[::3], camera_to_world, K); height, width = image.shape[:2]
    for point in uv[valid]:
        x, y = np.rint(point).astype(int)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), 1, color, -1, lineType=cv2.LINE_AA)


def plotly_html(frames: list[tuple[str, np.ndarray, str]], models: list[tuple[str, np.ndarray, str]], output: Path, *, axis_length_m: float) -> None:
    import plotly.graph_objects as go
    figure = go.Figure()
    axis_colors = ("#ef4444", "#22c55e", "#3b82f6")
    for name, transform, kind in frames:
        points = frame_points(transform, axis_length_m)
        for index, color in enumerate(axis_colors, start=1):
            figure.add_trace(go.Scatter3d(x=(points[0,0],points[index,0]), y=(points[0,1],points[index,1]), z=(points[0,2],points[index,2]), mode="lines+markers", line={"color":color,"width":8}, marker={"size":3}, name=f"{name} {AXIS_NAMES[index-1]}", legendgroup=name, hovertemplate=f"{name} {AXIS_NAMES[index-1]}<extra></extra>"))
        figure.add_trace(go.Scatter3d(x=(points[0,0],), y=(points[0,1],), z=(points[0,2],), mode="text", text=(name,), textposition="top center", textfont={"color":"#e5e7eb"}, showlegend=False, hoverinfo="skip"))
    for name, points, color in models:
        figure.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode="markers", name=name, marker={"size":2.1,"color":color,"opacity":.62}, hoverinfo="skip"))
    figure.update_layout(title="Dual UR7e end-effector frames (left_base): X=red, Y=green, Z=blue", paper_bgcolor="#080a0d", plot_bgcolor="#080a0d", font={"color":"#e5e7eb"}, scene={"bgcolor":"#080a0d","aspectmode":"data","xaxis_title":"left_base X (m)","yaxis_title":"left_base Y (m)","zaxis_title":"left_base Z (m)","camera":{"eye":{"x":1.4,"y":-1.4,"z":1.1}}}, margin={"l":0,"r":0,"t":45,"b":0})
    # Keep the artifact self-contained: the robot workstation need not have
    # Internet access when an operator opens it beside the workcell.
    figure.write_html(output, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = parse_args()
    # load_arms uses these optional fields even though this visualization does
    # not use any end-effector proxy transform.
    args.left_tool_from_tcp = None; args.right_tool_from_tcp = None
    left, right, calibration, calibration_report = load_arms(args)
    from rtde_receive import RTDEReceiveInterface
    receivers = {arm.name: RTDEReceiveInterface(arm.ip) for arm in (left, right)}
    source = RealSenseD435iSource(cameras=(D435iCameraConfig("front", args.front_serial or calibration_report["camera_serial"]),), width=args.width, height=args.height, fps=args.fps)
    try:
        qpos = {arm.name: np.asarray(receivers[arm.name].getActualQ(), dtype=np.float64) for arm in (left, right)}
        tcp_pose = {arm.name: np.asarray(receivers[arm.name].getActualTCPPose(), dtype=np.float64) for arm in (left, right)}
        if any(value.shape != (6,) or not np.isfinite(value).all() for value in (*qpos.values(), *tcp_pose.values())):
            raise RuntimeError("RTDE returned an invalid joint or TCP pose")
        frames: list[tuple[str, np.ndarray, str]] = []
        model_colours_bgr = {left.name: (210, 45, 210), right.name: (0, 210, 255)}
        model_colours_html = {left.name: "#d22dd2", right.name: "#ffd200"}
        models: list[tuple[str, np.ndarray, str]] = []
        report: dict = {"coordinate_frame": "left_base", "axis_convention": {"x": "red", "y": "green", "z": "blue"}, "calibration": calibration_report, "model_frame_status": "URDF root frames are active TCP frames.  Upstream Robotiq mesh surfaces are overlaid to inspect model-axis direction against the physical tool.", "arms": {}}
        for arm in (left, right):
            flange = arm.base_to_left @ flange_transform(qpos[arm.name])
            active_tcp = arm.base_to_left @ rotvec_transform(tcp_pose[arm.name])
            flange_to_tcp = np.linalg.inv(flange) @ active_tcp
            # These factory model base-to-TCP distances are intrinsic to the
            # end-effector.  Do not substitute flange-to-TCP: that includes
            # the physical wrist mount and would shift the CAD mesh backward.
            local_model = robotiq_2f85_points_in_active_tcp() if arm.name == left.name else robotiq_epick_points_in_active_tcp()
            model_points = (active_tcp[:3, :3] @ local_model.T).T + active_tcp[:3, 3]
            frames.extend(((f"{arm.name} flange", flange, "flange"), (f"{arm.name} URDF root = active TCP", active_tcp, "urdf_root")))
            models.append((f"{arm.name} {arm.tool} URDF collision", model_points, model_colours_html[arm.name]))
            report["arms"][arm.name] = {"tool": arm.tool, "visual_mesh_source": "PickNik Robotics upstream Robotiq description package", "urdf_root": "controller active TCP", "q_rad": qpos[arm.name].tolist(), "flange_in_left_base": flange.tolist(), "urdf_root_in_left_base": active_tcp.tolist(), "flange_to_urdf_root": flange_to_tcp.tolist(), "urdf_surface_point_count": int(len(model_points))}
        source.start()
        for _ in range(max(1, args.warmup_frames)): source.read()
        image = source.read()["front"].rgb.copy()
        for arm in (left, right):
            matching = next(points for name, points, _ in models if name.startswith(arm.name + " "))
            draw_model_points(image, matching, camera_to_world=calibration.camera_to_world, K=calibration.intrinsics, color=model_colours_bgr[arm.name])
        for name, transform, kind in frames:
            draw_frame(image, transform, name=name, camera_to_world=calibration.camera_to_world, K=calibration.intrinsics, label_color=(0, 230, 255) if kind == "flange" else (255, 255, 255))
    finally:
        source.stop()
        for receiver in receivers.values(): receiver.disconnect()
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    overlay = output_dir / "front_ee_frames_overlay.png"; cv2.imwrite(str(overlay), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    scene = output_dir / "ee_urdf_frames_3d.html"; plotly_html(frames, models, scene, axis_length_m=args.axis_length_m)
    report_path = output_dir / "ee_frame_snapshot.json"; report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"front_overlay": str(overlay), "scene_3d": str(scene), "frame_snapshot": str(report_path), "model_frame_status": report["model_frame_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
