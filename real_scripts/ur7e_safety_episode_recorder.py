"""Asynchronous, loss-visible raw recorder for real UR7e safety episodes.

The recorder deliberately stores raw RGB-D and timestamp metadata before any
depth repair or point-cloud processing.  This makes camera/robot alignment and
offline reconstruction reproducible without placing filesystem work on the
robot-control path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import shutil
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from real_scripts.real_robot_adapter import RGBDFrame


@dataclass(frozen=True)
class PolicyQueryRecord:
    query_id: int
    request_timestamp_ns: int
    response_timestamp_ns: int
    prefix_tokens: np.ndarray
    action_chunk: np.ndarray


@dataclass(frozen=True)
class ControlTickRecord:
    tick_id: int
    frames: tuple[RGBDFrame, ...]
    qpos: np.ndarray
    gripper_state: np.ndarray
    robot_state_timestamp_ns: int
    policy_query_id: int | None
    action_index: int | None
    commanded_action: np.ndarray | None


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class UR7eSafetyEpisodeRecorder:
    """Writes raw episode records from a bounded background queue.

    ``record_*`` never waits for disk I/O.  A full queue either raises (the
    safe default) or returns ``False`` when ``drop_on_backpressure`` is opted
    into.  Callers must surface a failed enqueue as a data-quality event.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        queue_size: int = 128,
        drop_on_backpressure: bool = False,
        manifest: Mapping[str, Any] | None = None,
        calibration_path: Path | None = None,
    ) -> None:
        if int(queue_size) < 1:
            raise ValueError("queue_size must be >= 1")
        self.output_dir = Path(output_dir)
        self.drop_on_backpressure = bool(drop_on_backpressure)
        self._queue: queue.Queue[tuple[str, object] | None] = queue.Queue(maxsize=int(queue_size))
        self._writer_error: BaseException | None = None
        self._closed = False
        self._dropped_records = 0
        self._written_ticks = 0
        self._written_queries = 0
        self.output_dir.mkdir(parents=True, exist_ok=False)
        manifest_payload = dict(manifest or {})
        manifest_payload.update({"status": "recording", "format_version": 1})
        _atomic_write_json(self.output_dir / "manifest.json", manifest_payload)
        if calibration_path is not None:
            source = Path(calibration_path)
            if not source.is_file():
                raise FileNotFoundError(f"Calibration file does not exist: {source}")
            shutil.copy2(source, self.output_dir / "calibration.json")
        self._worker = threading.Thread(target=self._run, name="ur7e-episode-writer", daemon=True)
        self._worker.start()

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    def _check_writer(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("Episode recorder writer failed") from self._writer_error
        if self._closed:
            raise RuntimeError("Episode recorder is already closed")

    def _enqueue(self, kind: str, item: object) -> bool:
        self._check_writer()
        try:
            self._queue.put_nowait((kind, item))
            return True
        except queue.Full:
            self._dropped_records += 1
            if self.drop_on_backpressure:
                return False
            raise RuntimeError("Episode recorder queue is full; refusing to hide a missing raw record")

    def record_policy_query(
        self,
        *,
        query_id: int,
        request_timestamp_ns: int,
        response_timestamp_ns: int,
        prefix_tokens: np.ndarray,
        action_chunk: np.ndarray,
    ) -> bool:
        record = PolicyQueryRecord(
            query_id=int(query_id),
            request_timestamp_ns=int(request_timestamp_ns),
            response_timestamp_ns=int(response_timestamp_ns),
            prefix_tokens=np.asarray(prefix_tokens, dtype=np.float32).copy(),
            action_chunk=np.asarray(action_chunk, dtype=np.float32).copy(),
        )
        return self._enqueue("policy_query", record)

    def record_tick(
        self,
        *,
        tick_id: int,
        frames: Sequence[RGBDFrame],
        qpos: np.ndarray,
        gripper_state: np.ndarray,
        robot_state_timestamp_ns: int,
        policy_query_id: int | None,
        action_index: int | None,
        commanded_action: np.ndarray | None,
    ) -> bool:
        copied_frames = tuple(
            RGBDFrame(
                frame.camera_name,
                np.asarray(frame.rgb, dtype=np.uint8).copy(),
                np.asarray(frame.depth_m, dtype=np.float32).copy(),
                host_timestamp_ns=frame.host_timestamp_ns,
                device_timestamp_ms=frame.device_timestamp_ms,
                frame_number=frame.frame_number,
                timestamp_domain=frame.timestamp_domain,
            )
            for frame in frames
        )
        names = [frame.camera_name for frame in copied_frames]
        if len(names) != len(set(names)):
            raise ValueError(f"Each tick must contain at most one frame per camera, got {names}")
        record = ControlTickRecord(
            tick_id=int(tick_id),
            frames=copied_frames,
            qpos=np.asarray(qpos, dtype=np.float32).reshape(-1).copy(),
            gripper_state=np.asarray(gripper_state, dtype=np.float32).reshape(-1).copy(),
            robot_state_timestamp_ns=int(robot_state_timestamp_ns),
            policy_query_id=None if policy_query_id is None else int(policy_query_id),
            action_index=None if action_index is None else int(action_index),
            commanded_action=None if commanded_action is None else np.asarray(commanded_action, dtype=np.float32).reshape(-1).copy(),
        )
        return self._enqueue("tick", record)

    def _write_policy_query(self, record: PolicyQueryRecord) -> None:
        output = self.output_dir / "policy_queries" / f"{record.query_id:06d}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                prefix_tokens=record.prefix_tokens,
                action_chunk=record.action_chunk,
                request_timestamp_ns=np.asarray(record.request_timestamp_ns, dtype=np.int64),
                response_timestamp_ns=np.asarray(record.response_timestamp_ns, dtype=np.int64),
            )
        os.replace(temporary, output)
        self._written_queries += 1

    def _write_tick(self, record: ControlTickRecord) -> None:
        tick_dir = self.output_dir / "frames" / f"{record.tick_id:06d}"
        tick_dir.mkdir(parents=True, exist_ok=False)
        _atomic_save_npy(tick_dir / "qpos.npy", record.qpos)
        _atomic_save_npy(tick_dir / "gripper_state.npy", record.gripper_state)
        if record.commanded_action is not None:
            _atomic_save_npy(tick_dir / "commanded_action.npy", record.commanded_action)
        frames_meta: dict[str, Any] = {}
        host_times = []
        for frame in record.frames:
            _atomic_save_npy(tick_dir / f"{frame.camera_name}_rgb.npy", frame.rgb)
            _atomic_save_npy(tick_dir / f"{frame.camera_name}_depth_raw_m.npy", frame.depth_m)
            if frame.host_timestamp_ns is not None:
                host_times.append(frame.host_timestamp_ns)
            frames_meta[frame.camera_name] = {
                "host_timestamp_ns": frame.host_timestamp_ns,
                "device_timestamp_ms": frame.device_timestamp_ms,
                "frame_number": frame.frame_number,
                "timestamp_domain": frame.timestamp_domain,
            }
        skew_ns = None if len(host_times) < 2 else int(max(host_times) - min(host_times))
        _atomic_write_json(
            tick_dir / "metadata.json",
            {
                "tick_id": record.tick_id,
                "robot_state_timestamp_ns": record.robot_state_timestamp_ns,
                "policy_query_id": record.policy_query_id,
                "action_index": record.action_index,
                "camera_host_skew_ns": skew_ns,
                "frames": frames_meta,
            },
        )
        self._written_ticks += 1

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                kind, record = item
                if kind == "policy_query":
                    self._write_policy_query(record)  # type: ignore[arg-type]
                elif kind == "tick":
                    self._write_tick(record)  # type: ignore[arg-type]
                else:
                    raise ValueError(f"Unknown recorder item type: {kind}")
        except BaseException as exc:  # propagate on the control thread at close/next enqueue
            self._writer_error = exc

    def close(self) -> None:
        if self._closed:
            return
        self._queue.put(None)
        self._worker.join()
        self._closed = True
        if self._writer_error is not None:
            raise RuntimeError("Episode recorder writer failed") from self._writer_error
        manifest_path = self.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "complete",
                "written_ticks": self._written_ticks,
                "written_policy_queries": self._written_queries,
                "dropped_records": self._dropped_records,
            }
        )
        _atomic_write_json(manifest_path, manifest)

    def __enter__(self) -> "UR7eSafetyEpisodeRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
