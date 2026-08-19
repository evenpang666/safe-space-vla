from __future__ import annotations

import logging
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from .config import AppConfig
from .demo_collection import LiveDemoCollector, PendingEpisode, save_pending_episode

LOG = logging.getLogger(__name__)


@contextmanager
def _without_unsupported_socks_proxy() -> Iterator[None]:
    """Hide only legacy ``socks://`` proxy variables while Gradio starts.

    httpx accepts ``socks5://`` but rejects the legacy ``socks://`` spelling.
    The latter is common in shell proxy configurations and otherwise prevents
    Gradio from importing before the local web server even starts.  This only
    changes environment variables inside the collector process; the calling
    shell is never modified.
    """
    removed: dict[str, str] = {}
    for name in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(name)
        if value and value.lower().startswith("socks://"):
            removed[name] = value
            os.environ.pop(name, None)
    try:
        yield
    finally:
        os.environ.update(removed)


def _import_gradio():
    try:
        with _without_unsupported_socks_proxy():
            import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for collect-demo. Install it with "
            "`python -m pip install gradio` in the active environment."
        ) from exc
    return gr


class DemoCollectorWeb:
    """Thread-safe Web control surface for the live demonstration collector."""

    def __init__(self, cfg: AppConfig, task: str, execute: bool):
        self.cfg = cfg
        self.initial_task = task
        self.execute = execute
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.teleop_stop = threading.Event()
        self.teleop_pause = threading.Event()
        self.record_event = threading.Event()
        self._lock = threading.RLock()
        self.pending: Optional[PendingEpisode] = None
        self.record_task: Optional[str] = None
        self.teleop_running = False
        self.teleop_starting = False
        self.teleop_paused = False
        self.recording = False
        self.saving = False
        self._status = (
            "Enter a task, confirm the cell is clear, then start teleoperation."
            if execute
            else "Dry run: restart with --execute before physical teleoperation can start."
        )

    def _worker(self, name: str, fn: Callable[[], object]) -> None:
        def work() -> None:
            try:
                self.events.put((name, fn()))
            except BaseException as exc:
                LOG.exception("Demo collection operation failed")
                self.events.put(("error", exc))

        threading.Thread(target=work, name=f"demo-{name}", daemon=True).start()

    def _drain_events(self) -> None:
        """Apply worker notifications on the Gradio request/timer thread."""
        try:
            while True:
                kind, payload = self.events.get_nowait()
                with self._lock:
                    if kind == "teleop_ready":
                        self.teleop_starting = False
                        self.teleop_running = True
                        self._status = "Teleoperation active. Enter a task and start recording when ready."
                    elif kind == "teleop_paused":
                        self.teleop_paused = True
                        self.recording = False
                        self._status = "Teleoperation paused. Hardware remains connected; click Resume Teleoperation to continue."
                    elif kind == "teleop_resumed":
                        self.teleop_paused = False
                        self._status = "Teleoperation resumed with a fresh Sensor/TCP anchor."
                    elif kind == "record_progress":
                        if self.recording:
                            self._status = f"Recording episode: {int(payload)} frames. Teleoperation is active."
                    elif kind == "record_done":
                        self.pending = payload  # type: ignore[assignment]
                        self.recording = False
                        frame_count = len(self.pending.frames) if self.pending is not None else 0
                        if self.pending is None or self.record_task is None:
                            self._status = "Recording stopped, but no episode task is available for saving."
                        else:
                            pending, task = self.pending, self.record_task
                            self.saving = True
                            self._status = f"Recording stopped: {frame_count} frames. Saving episode…"
                            self._worker("save_done", lambda: save_pending_episode(pending, task, self.cfg.demo))
                    elif kind == "save_done":
                        episode_path = payload  # type: ignore[assignment]
                        if self.pending is not None:
                            self.pending.discard()
                        self.pending = None
                        self.record_task = None
                        self.saving = False
                        self._status = f"Saved file episode to {episode_path}. Teleoperation remains active."
                    elif kind == "teleop_done":
                        self.teleop_starting = False
                        self.teleop_running = False
                        self.teleop_paused = False
                        self.recording = False
                        self.record_event.clear()
                        if self.pending is None:
                            self._status = "Teleoperation paused; hardware has been released."
                    elif kind == "error":
                        self.teleop_starting = False
                        self.teleop_running = False
                        self.teleop_paused = False
                        self.recording = False
                        self.saving = False
                        self.record_event.clear()
                        self.teleop_stop.set()
                        self._status = f"Operation failed: {payload}"
        except queue.Empty:
            return

    def status(self) -> str:
        self._drain_events()
        with self._lock:
            state = []
            if self.teleop_starting:
                state.append("connecting")
            if self.teleop_running:
                state.append("teleoperation paused" if self.teleop_paused else "teleoperation active")
            if self.recording:
                state.append("recording")
            if self.pending is not None:
                state.append("episode staged")
            if self.saving:
                state.append("saving")
            return f"{self._status}\n\nState: {', '.join(state) if state else 'idle'}"

    def status_with_record_button(self):
        gr = _import_gradio()
        status = self.status()
        with self._lock:
            label = "Stop & Save Recording" if self.recording else "Start Recording"
        return status, gr.Button(value=label)

    def start_teleoperation(self, confirmed: bool) -> tuple[str, bool]:
        self._drain_events()
        with self._lock:
            if not self.execute:
                self._status = "Physical teleoperation is disabled. Restart with --execute."
            elif self.teleop_running or self.teleop_starting:
                self._status = "Teleoperation is already starting or active."
            elif not confirmed:
                self._status = "Confirm that the cell is clear and E-stop is reachable before starting teleoperation."
            else:
                self.teleop_starting = True
                self.teleop_stop.clear()
                self.teleop_pause.clear()
                self.record_event.clear()
                self._status = "Connecting Pika Sense, UR7e, gripper, and cameras…"

                def run_teleop() -> None:
                    LiveDemoCollector(self.cfg).run(
                        self.teleop_stop,
                        self.record_event,
                        lambda pending: self.events.put(("record_done", pending)),
                        lambda: self.events.put(("teleop_ready", None)),
                        lambda count: self.events.put(("record_progress", count))
                        if count % self.cfg.demo.fps == 0
                        else None,
                        pause_event=self.teleop_pause,
                        on_paused=lambda: self.events.put(("teleop_paused", None)),
                        on_resumed=lambda: self.events.put(("teleop_resumed", None)),
                    )

                self._worker("teleop_done", run_teleop)
        return self.status(), False

    def toggle_pause_teleoperation(self):
        """Soft-pause or resume without tearing down any hardware connection."""
        gr = _import_gradio()
        self._drain_events()
        with self._lock:
            if self.teleop_running and self.teleop_paused:
                self.teleop_pause.clear()
                self.teleop_paused = False
                self._status = "Resuming teleoperation; resetting the Sensor/TCP anchor…"
                label = "Pause Teleoperation"
            elif self.teleop_running:
                self.record_event.clear()
                self.recording = False
                self.teleop_paused = True
                self.teleop_pause.set()
                self._status = "Pausing teleoperation; hardware remains connected…"
                label = "Resume Teleoperation"
            else:
                self._status = "Teleoperation is not active."
                label = "Pause Teleoperation"
        return self.status(), gr.Button(value=label)

    def toggle_recording(self, task: str):
        """Start recording, or stop the active recording and save it immediately."""
        gr = _import_gradio()
        self._drain_events()
        task = task.strip()
        with self._lock:
            if self.recording:
                self.recording = False
                self.record_event.clear()
                self._status = "Stopping recording; the completed episode will be saved automatically…"
                label = "Start Recording"
            elif not task:
                self._status = "Task description cannot be empty."
                label = "Start Recording"
            elif not self.teleop_running or self.teleop_paused:
                self._status = "Resume teleoperation before recording."
                label = "Start Recording"
            elif self.pending is not None or self.saving:
                self._status = "The previous episode is still being saved."
                label = "Start Recording"
            else:
                self.record_task = task
                self.recording = True
                self.record_event.set()
                self._status = "Recording episode while teleoperation continues…"
                label = "Stop & Save Recording"
        return self.status(), gr.Button(value=label)

    def save(self) -> str:
        self._drain_events()
        with self._lock:
            if self.pending is None or self.record_task is None:
                self._status = "No staged episode is available to save."
            elif self.saving:
                self._status = "Episode saving is already in progress."
            else:
                pending, task = self.pending, self.record_task
                self.saving = True
                self._status = "Encoding and saving the episode; teleoperation remains active."
                self._worker("save_done", lambda: save_pending_episode(pending, task, self.cfg.demo))
        return self.status()

    def discard(self) -> str:
        self._drain_events()
        with self._lock:
            if self.pending is None:
                self._status = "No staged episode is available to discard."
            elif self.saving:
                self._status = "Cannot discard while the episode is being saved."
            else:
                self.pending.discard()
                self.pending = None
                self.record_task = None
                self._status = "Episode discarded. Teleoperation remains active."
        return self.status()

    def shutdown(self) -> None:
        """Stop hardware on process shutdown (Ctrl+C / service stop)."""
        with self._lock:
            self.record_event.clear()
            self.teleop_stop.set()
            self.teleop_pause.clear()
            if self.pending is not None and not self.saving:
                self.pending.discard()
                self.pending = None

    def build(self):
        gr = _import_gradio()
        with gr.Blocks(title="UR7e / PiKA demonstration collection") as page:
            gr.Markdown(
                "# UR7e / PiKA 示教采集\n"
                "此页面可远程控制真机。开始前确认急停可达、工作区清空，且 UR 示教器处于 Remote Control。"
            )
            task = gr.Textbox(label="Task description", value=self.initial_task, placeholder="e.g. pick cube")
            confirmed = gr.Checkbox(label="我已确认工作区清空且急停可达", value=False)
            status = gr.Textbox(label="Status", value=self.status(), lines=4, interactive=False)
            with gr.Row():
                start = gr.Button("Start Teleoperation", variant="primary")
                pause = gr.Button("Pause Teleoperation")
                record = gr.Button("Start Recording", variant="primary")

            start.click(self.start_teleoperation, inputs=confirmed, outputs=[status, confirmed])
            pause.click(self.toggle_pause_teleoperation, outputs=[status, pause])
            record.click(self.toggle_recording, inputs=task, outputs=[status, record])
            gr.Timer(0.5).tick(self.status_with_record_button, outputs=[status, record], queue=False)
        return page


class DemoCollectorProcessWeb:
    """Persistent Gradio control plane for one isolated collection worker at a time."""

    def __init__(self, cfg: AppConfig, task: str, execute: bool, config_path: Path):
        self.cfg = cfg
        self.initial_task = task
        self.execute = execute
        self.config_path = config_path
        self._lock = threading.RLock()
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._process: Optional[subprocess.Popen[str]] = None
        self._status = "Enter a task, confirm the cell is clear, then start a new isolated teleoperation worker."
        self._last_worker_line = ""
        self._last_worker_command = ""
        self._teleop_ready = False
        self._recording_requested = False
        self._recording_started = False
        self._episode_saved = False

    def _running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                with self._lock:
                    if kind == "line":
                        line = str(payload).strip()
                        if line:
                            self._last_worker_line = line
                            if "[demo-worker] READY" in line:
                                self._teleop_ready = True
                                self._status = (
                                    "Teleoperation active. Restore the robot state, then click Start Recording when ready."
                                )
                            elif "[demo-worker] RECORDING_STARTED" in line:
                                self._recording_started = True
                                self._status = "Recording active. Stop Teleoperation & Save will save this episode."
                            elif "[demo-worker] RECORD_REQUESTED" in line:
                                self._recording_requested = True
                                self._status = "Recording requested; waiting for cameras to start…"
                            elif "[demo-worker] SAVED" in line:
                                self._episode_saved = True
                                self._status = f"Episode saved: {line.removeprefix('[demo-worker] SAVED ').strip()}"
                            elif "[demo-worker] FAILED" in line:
                                self._status = f"Worker failed: {line.removeprefix('[demo-worker] FAILED ').strip()}"
                    elif kind == "exit":
                        code = int(payload)
                        self._process = None
                        episode_saved = self._episode_saved
                        self._teleop_ready = False
                        self._recording_requested = False
                        self._recording_started = False
                        if code == 0:
                            self._status = (
                                "Worker stopped normally. The recorded episode was saved. Click Start to run a new session."
                                if episode_saved
                                else "Worker stopped normally. No completed recording was saved."
                            )
                        elif "[demo-worker] FAILED" not in self._last_worker_line:
                            self._status = f"Worker exited unexpectedly (code {code}). Click Start to launch a clean new worker."
        except queue.Empty:
            return

    def _watch_worker(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            # The Web UI owns the child's stdout so it can report failures in
            # the browser.  Mirror it to the launcher terminal as well: this
            # makes a Start click observable without opening browser devtools.
            print(
                f"[worker-log] {line}",
                end="" if line.endswith("\n") else "\n",
                flush=True,
            )
            self._events.put(("line", line))
        code = process.wait()
        print(f"[worker-log] exited with code {code}", flush=True)
        self._events.put(("exit", code))

    def status_with_buttons(self):
        gr = _import_gradio()
        self._drain_events()
        with self._lock:
            state = "running" if self._running() else "idle"
            detail = f"\n\nState: {state}"
            if self._last_worker_command:
                detail += f"\nCommand: {self._last_worker_command}"
            if self._last_worker_line:
                detail += f"\nWorker: {self._last_worker_line}"
            start_label = "Stop Teleoperation & Save" if self._running() else "Start Teleoperation"
            record_enabled = self._teleop_ready and not self._recording_requested
            record_label = "Recording…" if self._recording_requested else "Start Recording"
            return (
                f"{self._status}{detail}",
                gr.Button(value=start_label),
                gr.Button(value=record_label, interactive=record_enabled),
            )

    def start_or_stop(self, task: str):
        gr = _import_gradio()
        self._drain_events()
        with self._lock:
            if self._running():
                assert self._process is not None
                self._status = "Stop requested. Waiting for the worker to save the current episode…"
                if os.name == "posix":
                    os.killpg(self._process.pid, signal.SIGINT)
                else:
                    self._process.send_signal(signal.SIGINT)
                status, start_button, record_button = self.status_with_buttons()
                return status, start_button, record_button
            task = task.strip()
            if not self.execute:
                self._status = "Physical teleoperation is disabled. Restart the Web UI with --execute."
            elif not task:
                self._status = "Task description cannot be empty."
            else:
                command = [
                    sys.executable,
                    "-m",
                    "ur7e_vla.cli",
                    "collect-demo-worker",
                    "--config",
                    str(self.config_path),
                    "--task",
                    task,
                    "--execute",
                    "--wait-for-record",
                ]
                self._last_worker_command = shlex.join(command)
                self._last_worker_line = ""
                self._teleop_ready = False
                self._recording_requested = False
                self._recording_started = False
                self._episode_saved = False
                self._status = f"Starting isolated worker for task: {task!r}…"
                print(f"[collector-web] starting: {self._last_worker_command}", flush=True)
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=os.name == "posix",
                )
                threading.Thread(
                    target=self._watch_worker,
                    args=(self._process,),
                    name="demo-worker-log",
                    daemon=True,
                ).start()
        status, start_button, record_button = self.status_with_buttons()
        return status, start_button, record_button

    def start_recording(self):
        gr = _import_gradio()
        self._drain_events()
        with self._lock:
            if not self._running() or self._process is None:
                self._status = "Start Teleoperation first."
            elif not self._teleop_ready:
                self._status = "Teleoperation is still connecting; wait for READY before starting recording."
            elif self._recording_requested:
                self._status = "Recording has already been requested."
            elif self._process.stdin is None:
                self._status = "Worker control channel is unavailable; stop and start teleoperation again."
            else:
                try:
                    self._process.stdin.write("record\n")
                    self._process.stdin.flush()
                    self._recording_requested = True
                    self._status = "Recording requested; waiting for cameras to start…"
                    print("[collector-web] sent worker control: record", flush=True)
                except (BrokenPipeError, OSError):
                    self._status = "Worker exited before recording could start. Click Start Teleoperation to retry."
        return self.status_with_buttons()

    def shutdown(self) -> None:
        with self._lock:
            if self._running():
                assert self._process is not None
                if os.name == "posix":
                    os.killpg(self._process.pid, signal.SIGINT)
                else:
                    self._process.send_signal(signal.SIGINT)

    def build(self):
        gr = _import_gradio()
        with gr.Blocks(title="UR7e / PiKA demonstration collection") as page:
            gr.Markdown(
                "# UR7e / PiKA 示教采集\n"
                "依次执行 Start Teleoperation → Start Recording → Stop Teleoperation & Save。"
            )
            task = gr.Textbox(label="Task description", value=self.initial_task, placeholder="e.g. pick cube")
            initial_status = self._status + "\n\nState: idle"
            status = gr.Textbox(label="Status", value=initial_status, lines=5, interactive=False)
            start_stop = gr.Button("Start Teleoperation", variant="primary")
            start_record = gr.Button(
                value="Start Recording",
                interactive=False,
                variant="secondary",
            )
            start_stop.click(
                self.start_or_stop,
                inputs=[task],
                outputs=[status, start_stop, start_record],
            )
            start_record.click(self.start_recording, outputs=[status, start_stop, start_record])
            gr.Timer(0.5).tick(
                self.status_with_buttons,
                outputs=[status, start_stop, start_record],
                queue=False,
            )
        return page


def launch_demo_gradio(
    cfg: AppConfig,
    task: str,
    execute: bool,
    *,
    config_path: Path | None = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
) -> None:
    """Serve the demonstration collector locally or on a LAN interface."""
    if not 1 <= server_port <= 65535:
        raise ValueError("--server-port must be in 1..65535")
    controller = DemoCollectorProcessWeb(cfg, task, execute, config_path or (Path.cwd() / "config.yaml"))
    page = controller.build()
    try:
        # Keep a legacy socks:// proxy out of Gradio's httpx startup checks.
        # It is restored when the process exits, while no parent shell is changed.
        with _without_unsupported_socks_proxy():
            page.queue(default_concurrency_limit=8).launch(
                server_name=server_name,
                server_port=server_port,
                inbrowser=False,
                share=False,
                show_error=True,
            )
    finally:
        controller.shutdown()
        page.close()
