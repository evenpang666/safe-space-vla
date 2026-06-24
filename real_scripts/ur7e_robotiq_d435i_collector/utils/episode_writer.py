from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeWriter:
    def __init__(
        self,
        dataset_root: str | Path,
        dataset_name: str,
        task: str,
        fps: int,
        robot_ip: str,
        camera_config: dict[str, Any],
        state_names: list[str],
        action_names: list[str],
        pika_sense_config: dict[str, Any] | None = None,
        robotiq_config: dict[str, Any] | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.task = task
        self.fps = int(fps)
        self.robot_ip = robot_ip
        self.camera_config = camera_config
        self.camera_names = tuple(str(name) for name in camera_config)
        if not self.camera_names:
            raise ValueError("camera_config must define at least one camera")
        self.pika_sense_config = dict(pika_sense_config or {})
        self.robotiq_config = dict(robotiq_config or {})
        self.state_names = list(state_names)
        self.action_names = list(action_names)

        self.episode_index = self._next_episode_index()
        self.episode_dir: Path | None = None
        self.rows: list[dict[str, Any]] = []
        self.started_at = ""

    def _next_episode_index(self) -> int:
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        indices: list[int] = []
        for path in self.dataset_root.glob("episode_*"):
            if not path.is_dir():
                continue
            try:
                indices.append(int(path.name.split("_", maxsplit=1)[1]))
            except (IndexError, ValueError):
                pass
        return max(indices) + 1 if indices else 0

    def start_episode(self) -> Path:
        if self.episode_dir is not None:
            raise RuntimeError("Episode already active")

        self.episode_dir = self.dataset_root / f"episode_{self.episode_index:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        for kind in ("rgb", "depth"):
            for camera in self.camera_names:
                (self.episode_dir / kind / camera).mkdir(parents=True, exist_ok=True)

        self.rows = []
        self.started_at = datetime.now(timezone.utc).isoformat()
        return self.episode_dir

    def add_frame(
        self,
        timestamp_s: float,
        state: np.ndarray,
        action: np.ndarray,
        frames: dict[str, dict[str, np.ndarray]],
    ) -> None:
        if self.episode_dir is None:
            raise RuntimeError("Call start_episode() before add_frame()")

        for camera in self.camera_names:
            if camera not in frames or "rgb" not in frames[camera] or "depth" not in frames[camera]:
                raise ValueError(f"Missing complete RGB-D frame for camera {camera!r}")

        state_array = np.asarray(state, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        if state_array.shape[0] != len(self.state_names):
            raise ValueError("state length must match state_names")
        if action_array.shape[0] != len(self.action_names):
            raise ValueError("action length must match action_names")
        if state_array.shape[0] < 7 or action_array.shape[0] < 7:
            raise ValueError("state and action must include 6 joints plus gripper")

        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required to write RGB-D episode images") from exc

        frame_index = len(self.rows)
        paths: dict[str, str] = {}
        for camera in self.camera_names:
            rgb_rel = f"rgb/{camera}/{frame_index:06d}.png"
            depth_rel = f"depth/{camera}/{frame_index:06d}.png"
            rgb = frames[camera]["rgb"]
            depth = frames[camera]["depth"]

            # The camera wrapper provides OpenCV-style BGR frames; imwrite expects
            # BGR input and stores a normal RGB PNG on disk.
            if not cv2.imwrite(str(self.episode_dir / rgb_rel), rgb):
                raise OSError(f"Failed to write RGB frame: {rgb_rel}")
            depth_u16 = np.asarray(depth, dtype=np.uint16)
            if not cv2.imwrite(str(self.episode_dir / depth_rel), depth_u16):
                raise OSError(f"Failed to write depth frame: {depth_rel}")

            paths[f"rgb.{camera}"] = rgb_rel
            paths[f"depth.{camera}"] = depth_rel

        row = {
            "frame_index": frame_index,
            "timestamp_s": float(timestamp_s),
            "task": self.task,
            "observation.state": state_array.tolist(),
            "action": action_array.tolist(),
            "joint_positions": state_array[:6].tolist(),
            "target_q": action_array[:6].tolist(),
            "gripper_position": float(state_array[6]),
            "gripper_action": float(action_array[6]),
        }
        row.update(paths)
        self.rows.append(row)

    def end_episode(self, save: bool) -> Path:
        if self.episode_dir is None:
            raise RuntimeError("No active episode")

        episode_dir = self.episode_dir
        if not save or not self.rows:
            shutil.rmtree(episode_dir, ignore_errors=True)
            self.episode_dir = None
            self.rows = []
            return episode_dir

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas and pyarrow are required to write frames.parquet") from exc

        pd.DataFrame(self.rows).to_parquet(episode_dir / "frames.parquet", index=False)
        meta = {
            "dataset_name": self.dataset_name,
            "episode_index": self.episode_index,
            "task": self.task,
            "started_at": self.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "fps": self.fps,
            "robot_ip": self.robot_ip,
            "pika_sense_config": self.pika_sense_config,
            "robotiq_config": self.robotiq_config,
            "camera_config": self.camera_config,
            "camera_names": list(self.camera_names),
            "state_names": self.state_names,
            "action_names": self.action_names,
            "depth_encoding": "RealSense raw uint16 Z16 aligned to color",
        }
        (episode_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self.episode_index += 1
        self.episode_dir = None
        self.rows = []
        return episode_dir
