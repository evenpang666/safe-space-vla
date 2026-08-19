"""Compatibility imports for the former Tk demo collection module.

``collect-demo`` now uses :mod:`ur7e_vla.demo_gradio`, so it can be operated
from a browser without a local X11/desktop display.  This module keeps the
former launch function for external callers, but it now opens the Web UI too.
"""

from .config import AppConfig
from .demo_gradio import DemoCollectorWeb, launch_demo_gradio

# Retain the former public symbol while routing new callers to the Web UI.
DemoCollectorGUI = DemoCollectorWeb


def launch_demo_gui(cfg: AppConfig, task: str, execute: bool) -> None:
    launch_demo_gradio(cfg, task, execute)
