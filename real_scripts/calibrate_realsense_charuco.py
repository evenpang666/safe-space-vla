#!/usr/bin/env python3
"""Estimate RealSense calibration using a ChArUco board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT = REPO_ROOT / "real_scripts" / "ur7e_d435i_camera_calibration.generated.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--wait-timeout-ms", type=int, default=10000)
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-length-m", type=float, required=True)
    parser.add_argument("--marker-length-m", type=float, required=True)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument(
        "--world-frame",
        choices=("board",),
        default="board",
        help="Currently writes camera_to_world with world equal to the ChArUco board frame.",
    )
    return parser.parse_args()


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def calibration_payload(
    camera_name: str,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    model: str = "intel_realsense_d435i",
) -> dict:
    return {
        "cameras": {
            str(camera_name): {
                "model": str(model),
                "intrinsics": np.asarray(intrinsics, dtype=np.float64).tolist(),
                "camera_to_world": np.asarray(camera_to_world, dtype=np.float64).tolist(),
            }
        }
    }


def save_calibration_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("ChArUco calibration requires opencv-contrib-python: pip install opencv-contrib-python") from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Your OpenCV build lacks cv2.aruco. Install opencv-contrib-python.")
    return cv2


def _aruco_dictionary(cv2, name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def create_charuco_board(
    *,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str,
):
    cv2 = _require_cv2()
    dictionary = _aruco_dictionary(cv2, dictionary_name)
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard(
            (int(squares_x), int(squares_y)),
            float(square_length_m),
            float(marker_length_m),
            dictionary,
        )
    return cv2.aruco.CharucoBoard_create(
        int(squares_x),
        int(squares_y),
        float(square_length_m),
        float(marker_length_m),
        dictionary,
    )


def capture_realsense_color_frames(
    *,
    serial: str | None,
    width: int,
    height: int,
    fps: int,
    samples: int,
    wait_timeout_ms: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("This script requires pyrealsense2.") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(str(serial))
    config.enable_stream(rs.stream.color, int(width), int(height), rs.format.rgb8, int(fps))
    profile = pipeline.start(config)
    try:
        frames: list[np.ndarray] = []
        for _ in range(max(1, int(samples))):
            frame_set = pipeline.wait_for_frames(int(wait_timeout_ms))
            color_frame = frame_set.get_color_frame()
            if not color_frame:
                continue
            frames.append(np.ascontiguousarray(np.asarray(color_frame.get_data(), dtype=np.uint8)))
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsics = np.asarray([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return frames, intrinsics
    finally:
        pipeline.stop()


def estimate_camera_to_board_from_frames(
    frames_rgb: list[np.ndarray],
    intrinsics: np.ndarray,
    *,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str,
) -> np.ndarray:
    cv2 = _require_cv2()
    board = create_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square_length_m,
        marker_length_m=marker_length_m,
        dictionary_name=dictionary_name,
    )
    dictionary = _aruco_dictionary(cv2, dictionary_name)
    detector_params = cv2.aruco.DetectorParameters()
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    camera_to_board_samples: list[np.ndarray] = []

    for rgb in frames_rgb:
        gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)
        if ids is None or len(ids) == 0:
            continue
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if charuco_ids is None or len(charuco_ids) < 4:
            continue
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            camera_matrix,
            dist_coeffs,
            None,
            None,
        )
        if not ok:
            continue
        board_to_camera = transform_from_rvec_tvec(rvec, tvec)
        camera_to_board_samples.append(invert_transform(board_to_camera))

    if not camera_to_board_samples:
        raise RuntimeError("No valid ChArUco poses detected. Move the board into view and retry.")
    translations = np.asarray([sample[:3, 3] for sample in camera_to_board_samples], dtype=np.float64)
    result = camera_to_board_samples[-1].copy()
    result[:3, 3] = translations.mean(axis=0)
    return result


def main() -> None:
    args = parse_args()
    frames, intrinsics = capture_realsense_color_frames(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        samples=args.samples,
        wait_timeout_ms=args.wait_timeout_ms,
    )
    camera_to_board = estimate_camera_to_board_from_frames(
        frames,
        intrinsics,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length_m,
        marker_length_m=args.marker_length_m,
        dictionary_name=args.dictionary,
    )
    payload = calibration_payload(args.camera_name, intrinsics, camera_to_board)
    save_calibration_json(args.output, payload)
    print(f"[done] wrote {args.camera_name!r} camera_to_world with world=charuco_board to {args.output}")


if __name__ == "__main__":
    main()
