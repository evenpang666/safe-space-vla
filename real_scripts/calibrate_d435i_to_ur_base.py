#!/usr/bin/env python3
"""Estimate fixed D435i colour-camera poses in the UR base frame from ChArUco."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_transform(value: Any, *, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must have final row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal with determinant +1")
    return transform


def rigid_transform_from_correspondences(board_points_m: np.ndarray, base_points_m: np.ndarray) -> np.ndarray:
    """Return ``^base T_board`` from paired non-collinear 3D point samples."""
    source = np.asarray(board_points_m, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(base_points_m, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("board_points_m and base_points_m must have matching shapes [N, 3] with N >= 3")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    if np.linalg.matrix_rank(source_zero) < 2:
        raise ValueError("board point correspondences must not be collinear")
    u, _, vh = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1] *= -1.0
        rotation = vh.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def camera_to_base_from_board_pose(board_to_base: np.ndarray, board_to_camera: np.ndarray) -> np.ndarray:
    """Compute ``^base T_camera = ^base T_board · inverse(^camera T_board)``."""
    return _as_transform(board_to_base, name="board_to_base") @ np.linalg.inv(_as_transform(board_to_camera, name="board_to_camera"))


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_board_to_base(*, transform_path: Path | None, correspondences_path: Path | None) -> np.ndarray:
    if (transform_path is None) == (correspondences_path is None):
        raise ValueError("Provide exactly one of --board-to-base-json or --board-base-correspondences-json")
    if transform_path is not None:
        payload = _load_json(transform_path)
        return _as_transform(payload.get("board_to_base", payload), name="board_to_base")
    payload = _load_json(correspondences_path)
    transform = rigid_transform_from_correspondences(payload["board_points_m"], payload["base_points_m"])
    residual = np.asarray(payload["base_points_m"], dtype=np.float64) - (
        transform[:3, :3] @ np.asarray(payload["board_points_m"], dtype=np.float64).T
    ).T - transform[:3, 3]
    rms_m = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if rms_m > 0.003:
        raise ValueError(f"board/base correspondence RMS is {rms_m * 1000:.1f} mm; re-measure before calibration")
    return transform


def _charuco_pose(
    image_path: Path,
    *,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    dictionary_name: str,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
) -> tuple[np.ndarray, float, int]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("ChArUco calibration requires opencv-contrib-python with cv2.aruco.") from exc
    if not hasattr(cv2, "aruco"):
        raise ImportError("This OpenCV installation has no cv2.aruco; install opencv-contrib-python.")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown OpenCV ArUco dictionary {dictionary_name!r}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image {image_path}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard((int(squares_x), int(squares_y)), float(square_length_m), float(marker_length_m), dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(image)
    if ids is None or len(ids) < 6:
        raise RuntimeError(f"Detected fewer than 6 ChArUco corners in {image_path}; improve board visibility")
    object_points, image_points = board.matchImagePoints(corners, ids)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, intrinsics, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError(f"solvePnP failed for {image_path}")
    reprojection, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsics, distortion)
    rms_px = float(np.sqrt(np.mean(np.sum((reprojection.reshape(-1, 2) - image_points.reshape(-1, 2)) ** 2, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    board_to_camera = np.eye(4, dtype=np.float64)
    board_to_camera[:3, :3] = rotation
    board_to_camera[:3, 3] = tvec.reshape(3)
    return board_to_camera, rms_px, int(len(ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory from capture_d435i_calibration_frame.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--board-to-base-json", type=Path, help="JSON with a board_to_base 4x4 matrix.")
    group.add_argument("--board-base-correspondences-json", type=Path, help="JSON with board_points_m and TCP-measured base_points_m.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-names", nargs="+", default=("front", "side"))
    parser.add_argument("--dictionary", default="DICT_5X5_100")
    parser.add_argument("--squares-x", type=int, required=True)
    parser.add_argument("--squares-y", type=int, required=True)
    parser.add_argument("--square-length-m", type=float, required=True)
    parser.add_argument("--marker-length-m", type=float, required=True)
    parser.add_argument("--max-reprojection-rms-px", type=float, default=0.8)
    return parser.parse_args()


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    if args.square_length_m <= 0.0 or args.marker_length_m <= 0.0 or args.marker_length_m >= args.square_length_m:
        raise ValueError("Require 0 < --marker-length-m < --square-length-m")
    if args.squares_x < 2 or args.squares_y < 2:
        raise ValueError("--squares-x and --squares-y must be >= 2")
    input_dir = Path(args.input_dir)
    intrinsics_payload = _load_json(input_dir / "d435i_color_intrinsics.json")
    camera_payloads = intrinsics_payload["cameras"]
    board_to_base = load_board_to_base(
        transform_path=args.board_to_base_json,
        correspondences_path=args.board_base_correspondences_json,
    )
    output_cameras: dict[str, Any] = {}
    for name in args.camera_names:
        if name not in camera_payloads:
            raise KeyError(f"No captured intrinsics for camera {name!r}")
        item = camera_payloads[name]
        intrinsics = np.asarray(item["intrinsics"], dtype=np.float64)
        distortion = np.asarray(item.get("distortion", np.zeros(5)), dtype=np.float64).reshape(-1, 1)
        board_to_camera, rms_px, corner_count = _charuco_pose(
            input_dir / f"{name}_rgb.png",
            intrinsics=intrinsics,
            distortion=distortion,
            dictionary_name=args.dictionary,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length_m=args.square_length_m,
            marker_length_m=args.marker_length_m,
        )
        if rms_px > args.max_reprojection_rms_px:
            raise RuntimeError(f"{name} ChArUco reprojection RMS is {rms_px:.3f}px; exceeds {args.max_reprojection_rms_px:.3f}px")
        output_cameras[str(name)] = {
            "model": "intel_realsense_d435i",
            "frame": "ur_base",
            "depth_alignment": "depth_to_color",
            "intrinsics": intrinsics.tolist(),
            "distortion": distortion.reshape(-1).tolist(),
            "camera_to_world": camera_to_base_from_board_pose(board_to_base, board_to_camera).tolist(),
            "charuco_corner_count": corner_count,
            "charuco_reprojection_rms_px": rms_px,
        }
    result = {
        "coordinate_frame": "ur_base",
        "board_to_base": board_to_base.tolist(),
        "cameras": output_cameras,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = calibrate(parse_args())
    print(f"[done] wrote {len(result['cameras'])} camera-to-UR-base calibrations")


if __name__ == "__main__":
    main()
