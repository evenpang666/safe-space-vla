#!/usr/bin/env python3
"""Evaluate PI05 safety predictions online in LIBERO.

The script connects to ``scripts/serve_pi05_prefix_policy.py`` for pi05_libero
actions and prefix tokens. When the server also advertises a safety module, it
uses server-side safety predictions; otherwise it falls back to a local
checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safety_module.point_decoder import SafetyPointDecoder, SafetyPointDecoderConfig
from safety_module.safety_flow_point_model import SafetyFlowPointModel, euler_sample


REPO_SCRIPT_DIR = REPO_ROOT / "scripts"


def load_repo_script_module(module_name: str):
    module_path = REPO_SCRIPT_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise ModuleNotFoundError(f"Could not find local script module at {module_path}")
    qualified_name = f"_safety_module_local_scripts.{module_name}"
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


collector = load_repo_script_module("collect_pi05_libero_safety_decoder_dataset")


DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "pi05_safety_decoder" / "pi05_libero_task0_decoder.pt"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pi05_safety_decoder" / "pi05_libero_task0_decoder_online_eval.npz"
DEFAULT_VIDEO_OUTPUT = REPO_ROOT / "outputs" / "pi05_safety_decoder" / "pi05_libero_task0_decoder_online_eval.mp4"
COORDINATE_FRAME = "mujoco_world"
FIXED_LINK_TOPOLOGY_COUNT = 7


@dataclass
class LoadedSafetyModel:
    model_type: str
    model: SafetyPointDecoder | SafetyFlowPointModel
    config: object | None = None
    model_kwargs: dict | None = None


@dataclass
class VideoFrameBuffer:
    enabled: bool
    frames: list[np.ndarray] = field(default_factory=list)

    def append(self, frame: np.ndarray) -> None:
        if self.enabled:
            self.frames.append(np.asarray(frame, dtype=np.uint8))


@dataclass(frozen=True)
class PointFlowCbfConstraint:
    time_index: int
    link_id: int
    point_id: int
    obb_id: int
    face_axis: int
    normal: np.ndarray
    h: float
    current_point: np.ndarray
    predicted_point: np.ndarray


@dataclass(frozen=True)
class CbfQpProjectionResult:
    action: np.ndarray
    success: bool
    max_violation: float
    iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Local trained decoder checkpoint. Used when the policy server does not serve safety predictions.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output evaluation .npz.")
    parser.add_argument("--video-output", type=Path, default=DEFAULT_VIDEO_OUTPUT, help="Output MP4 task video.")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 task video generation.")
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--video-camera", default="agentview")
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=320)
    parser.add_argument("--video-point-radius", type=int, default=1)
    parser.add_argument(
        "--safe-space",
        type=Path,
        default=None,
        help="Optional static safe-space .npz with obstacle OBBs. Used only when --no-realtime-obbs is set.",
    )
    parser.add_argument(
        "--realtime-obbs",
        dest="realtime_obbs",
        action="store_true",
        default=True,
        help="Build obstacle OBBs from the current LIBERO RGB-D state during evaluation.",
    )
    parser.add_argument(
        "--no-realtime-obbs",
        dest="realtime_obbs",
        action="store_false",
        help="Disable runtime OBB generation and use --safe-space if provided.",
    )
    parser.add_argument("--collision-margin", type=float, default=0.02, help="Extra OBB margin for collision checks.")
    parser.add_argument(
        "--obb-camera-names",
        nargs="+",
        default=["frontview", "sideview", "agentview"],
        help="MuJoCo cameras fused for runtime obstacle OBB generation.",
    )
    parser.add_argument("--obb-width", type=int, default=256)
    parser.add_argument("--obb-height", type=int, default=256)
    parser.add_argument("--obb-stride", type=int, default=2)
    parser.add_argument("--obb-max-depth", type=float, default=4.0)
    parser.add_argument("--obb-robot-mask-dilation", type=int, default=2)
    parser.add_argument("--obb-workspace-bounds", nargs=6, type=float, default=None)
    parser.add_argument("--obb-workspace-mode", choices=["table", "pointcloud"], default="table")
    parser.add_argument("--obb-workspace-margin", type=float, default=0.02)
    parser.add_argument("--obb-table-z", type=float, default=None)
    parser.add_argument("--obb-table-slab-height", type=float, default=0.02)
    parser.add_argument("--obb-table-obstacle-min-height", type=float, default=0.02)
    parser.add_argument("--obb-table-obstacle-max-height", type=float, default=0.35)
    parser.add_argument("--obb-component-voxel-size", type=float, default=0.02)
    parser.add_argument(
        "--obb-component-connectivity",
        type=int,
        choices=[6, 18, 26],
        default=6,
        help=(
            "3D voxel connectivity for realtime OBB component grouping. "
            "6 separates obstacles more aggressively; 26 preserves legacy corner-connected grouping."
        ),
    )
    parser.add_argument("--obb-min-component-points", type=int, default=40)
    parser.add_argument("--obb-box-margin", type=float, default=0.01)
    parser.add_argument("--obb-box-shape", choices=["cuboid", "cube"], default="cuboid")
    parser.add_argument("--obb-box-orientation", choices=["axis_aligned", "xy_oriented", "pca_3d"], default="xy_oriented")
    parser.add_argument("--obb-voxel-size", type=float, default=0.04)
    parser.add_argument(
        "--obb-target-geom-name-patterns",
        nargs="*",
        default=["eval_scene_obstacle", "wine_bottle", "winebottle"],
        help=(
            "Case-insensitive geom/body name substrings kept for realtime OBBs. "
            "Default keeps the inserted wine-bottle obstacle only. Use 'all' to restore legacy tabletop OBBs."
        ),
    )
    parser.add_argument("--prediction-steps", type=int, default=10, help="Euler ODE steps for SafetyFlowPointModel.")
    parser.add_argument(
        "--enable-cbf-qp",
        action="store_true",
        help="Filter executed PI05 actions with a point-flow-triggered CBF-QP projection.",
    )
    parser.add_argument("--cbf-alpha", type=float, default=1.0, help="CBF class-K gain for active OBB constraints.")
    parser.add_argument(
        "--cbf-action-space",
        choices=["auto", "joint_delta", "cartesian_delta", "cartesian_action"],
        default="auto",
        help=(
            "Action parameterization used by CBF-QP. 'joint_delta' optimizes arm joint deltas; "
            "'cartesian_delta' maps OSC_POSITION xyz deltas through an EEF Jacobian; "
            "'cartesian_action' directly optimizes executable OSC_POSITION xyz actions; "
            "'auto' uses cartesian_action for LIBERO OSC_POSITION action_dim=4 or 7 and joint_delta otherwise."
        ),
    )
    parser.add_argument(
        "--cbf-cartesian-damping",
        type=float,
        default=1e-4,
        help="Damping for pseudo-inverse mapping from Cartesian xyz action to joint deltas.",
    )
    parser.add_argument(
        "--cbf-trigger-margin",
        type=float,
        default=0.02,
        help="Extra OBB margin used only to trigger CBF-QP from predicted future point flow.",
    )
    parser.add_argument("--cbf-max-constraints", type=int, default=32)
    parser.add_argument("--cbf-finite-difference-eps", type=float, default=1e-4)
    parser.add_argument("--cbf-projection-iterations", type=int, default=12)
    parser.add_argument(
        "--cbf-trigger-source",
        choices=["predicted_point_flow", "current_pointcloud"],
        default="predicted_point_flow",
        help="Active-set source for CBF-QP constraints.",
    )
    parser.add_argument(
        "--cbf-include-current-points",
        action="store_true",
        help=(
            "When using predicted point-flow CBF, also add current-frame robot pointcloud constraints. "
            "By default, predicted point-flow CBF only uses future predicted point constraints."
        ),
    )
    parser.add_argument(
        "--cbf-correction-target",
        choices=["current_action", "predicted_frame_action"],
        default="current_action",
        help=(
            "Which action to correct when predicted point flow enters an OBB. "
            "'current_action' preserves the legacy behavior; 'predicted_frame_action' "
            "corrects the action at current_offset + predicted collision time_index."
        ),
    )
    parser.add_argument(
        "--cbf-action-lower",
        nargs="*",
        type=float,
        default=None,
        help="Optional scalar or per-variable lower bound for CBF QP variables.",
    )
    parser.add_argument(
        "--cbf-action-upper",
        nargs="*",
        type=float,
        default=None,
        help="Optional scalar or per-variable upper bound for CBF QP variables.",
    )
    parser.add_argument(
        "--cbf-fallback",
        choices=["projected", "zero", "nominal"],
        default="projected",
        help=(
            "Action fallback when CBF projection cannot fully satisfy all constraints. "
            "'projected' keeps the best-effort projection, preserving tangent motion; "
            "'zero' stops the optimized variables; 'nominal' executes the unfiltered action."
        ),
    )
    parser.add_argument(
        "--safety-prediction-source",
        choices=["auto", "remote", "local"],
        default="auto",
        help="Use remote server-side safety predictions when available, or local checkpoint inference.",
    )
    parser.add_argument("--policy-server-host", default="127.0.0.1", help="pi05 prefix policy websocket host.")
    parser.add_argument("--policy-server-port", type=int, default=8000, help="pi05 prefix policy websocket port.")
    parser.add_argument("--task-suite", default="libero_spatial", choices=sorted(collector.TASK_SUITE_MAX_STEPS))
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--scene-obstacle",
        choices=["none", "wine_bottle"],
        default="wine_bottle",
        help="Optionally insert a physical obstacle into the LIBERO scene before evaluation.",
    )
    parser.add_argument(
        "--scene-obstacle-xy",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Optional x y placement for --scene-obstacle. Defaults to the table center.",
    )
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--samples-per-action", type=int, default=1)
    parser.add_argument("--points-per-link", type=int, default=24, help="Flow model FK samples per clean arm link.")
    parser.add_argument(
        "--skeleton-source",
        choices=["surface", "anchors", "geom"],
        default="surface",
        help=(
            "'surface' uses fixed surface points on robot0_link1..link7; "
            "'anchors' uses the clean 7-link robot0_link0..link7 skeleton; "
            "'geom' uses all robot geom axes."
        ),
    )
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default="egl")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device_name(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if device == "gpu" else device


def select_safety_prediction_source(requested: str, server_metadata: dict | None) -> str:
    if requested not in {"auto", "remote", "local"}:
        raise ValueError(f"Unknown safety prediction source: {requested}")
    metadata = server_metadata or {}
    remote_available = bool(metadata.get("returns_safety_predictions", False))
    if requested == "auto":
        return "remote" if remote_available else "local"
    if requested == "remote" and not remote_available:
        raise RuntimeError(
            "The connected policy server does not advertise safety predictions. "
            "Start scripts/serve_pi05_prefix_policy.py with --safety-checkpoint, or use "
            "--safety-prediction-source local."
        )
    return requested


def infer_flow_points_per_link(
    *,
    max_points: int,
    skeleton_source: str,
    requested_points_per_link: int,
) -> int:
    """Infer the FK points-per-link needed by a trained flow model."""
    max_points = int(max_points)
    requested_points_per_link = int(requested_points_per_link)
    if skeleton_source not in {"surface", "anchors"}:
        return requested_points_per_link
    if max_points % FIXED_LINK_TOPOLOGY_COUNT != 0:
        raise ValueError(
            f"flow model max_points={max_points} is not divisible by "
            f"{FIXED_LINK_TOPOLOGY_COUNT} {skeleton_source} links"
        )
    return max_points // FIXED_LINK_TOPOLOGY_COUNT


def load_decoder_checkpoint(path: Path, device: torch.device) -> SafetyPointDecoder:
    load_kwargs = {"map_location": device}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
    except (TypeError, ValueError):
        pass
    payload = torch.load(path, **load_kwargs)
    config = SafetyPointDecoderConfig.from_dict(payload["config"])
    model = SafetyPointDecoder(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def load_safety_model_checkpoint(path: Path, device: torch.device) -> LoadedSafetyModel:
    load_kwargs = {"map_location": device}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
    except (TypeError, ValueError):
        pass
    payload = torch.load(path, **load_kwargs)
    model_type = str(payload.get("model_type", "SafetyPointDecoder"))
    if model_type == "SafetyFlowPointModel" or "model_kwargs" in payload:
        model_kwargs = dict(payload["model_kwargs"])
        model = SafetyFlowPointModel(**model_kwargs).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return LoadedSafetyModel(model_type="flow", model=model, model_kwargs=model_kwargs)

    config = SafetyPointDecoderConfig.from_dict(payload["config"])
    model = SafetyPointDecoder(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return LoadedSafetyModel(model_type="decoder", model=model, config=config)


@torch.no_grad()
def predict_link_points(model: SafetyPointDecoder, prefix_tokens: np.ndarray, device: torch.device) -> np.ndarray:
    prefix_tokens = np.array(prefix_tokens, dtype=np.float32, copy=True)
    if prefix_tokens.ndim == 2:
        prefix_tokens = prefix_tokens[None, ...]
    if prefix_tokens.ndim != 3 or prefix_tokens.shape[0] != 1:
        raise ValueError(f"prefix_tokens must have shape (N, D) or (1, N, D), got {prefix_tokens.shape}")
    prefix = torch.as_tensor(prefix_tokens, dtype=torch.float32, device=device)
    pred = model(prefix)
    return pred[0].detach().cpu().numpy().astype(np.float32, copy=False)


def absolute_link_points_from_offsets(offsets: np.ndarray, current_link_points: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float32)
    current_link_points = np.asarray(current_link_points, dtype=np.float32)
    if current_link_points.ndim != 3 or current_link_points.shape[-1] != 3:
        raise ValueError(f"current_link_points must have shape (L, P, 3), got {current_link_points.shape}")
    if offsets.ndim != 3 or offsets.shape[-1] != 3:
        raise ValueError(f"offsets must have shape (T, K, 3), got {offsets.shape}")
    link_count, points_per_link, _ = current_link_points.shape
    if offsets.shape[1] != link_count * points_per_link:
        raise ValueError(
            f"offset point count {offsets.shape[1]} does not match current link topology "
            f"{link_count} * {points_per_link}"
        )
    return current_link_points[None, :, :, :] + offsets.reshape(offsets.shape[0], link_count, points_per_link, 3)


@torch.no_grad()
def predict_safety_flow_link_points(
    model: SafetyFlowPointModel,
    prefix_tokens: np.ndarray,
    current_link_points: np.ndarray,
    *,
    device: torch.device,
    prediction_steps: int,
) -> np.ndarray:
    current_link_points = np.asarray(current_link_points, dtype=np.float32)
    if current_link_points.ndim != 3 or current_link_points.shape[-1] != 3:
        raise ValueError(f"current_link_points must have shape (L, P, 3), got {current_link_points.shape}")
    prefix_tokens = np.array(prefix_tokens, dtype=np.float32, copy=True)
    if prefix_tokens.ndim == 2:
        prefix_tokens = prefix_tokens[None, ...]
    if prefix_tokens.ndim != 3 or prefix_tokens.shape[0] != 1:
        raise ValueError(f"prefix_tokens must have shape (N, D) or (1, N, D), got {prefix_tokens.shape}")
    arm_points = current_link_points.reshape(1, -1, 3)
    if arm_points.shape[1] != int(model.flow_head.max_points):
        raise ValueError(
            f"current arm point count {arm_points.shape[1]} must equal flow model max_points="
            f"{int(model.flow_head.max_points)}. Use the same --points-per-link as training."
        )
    delta = euler_sample(
        model=model,
        arm_points=torch.as_tensor(arm_points, dtype=torch.float32, device=device),
        prefix_tokens=torch.as_tensor(prefix_tokens, dtype=torch.float32, device=device),
        n_steps=prediction_steps,
        n_future=int(model.flow_head.n_future),
        K=arm_points.shape[1],
    )
    offsets = delta[0].detach().cpu().numpy().astype(np.float32, copy=False)
    return absolute_link_points_from_offsets(offsets, current_link_points).astype(np.float32, copy=False)


def query_remote_safety_prediction(policy, *, prefix_tokens: np.ndarray, current_link_points: np.ndarray) -> np.ndarray:
    result = policy.infer(
        {
            "safety_only": True,
            "prefix_tokens": np.asarray(prefix_tokens, dtype=np.float32),
            "current_link_points": np.asarray(current_link_points, dtype=np.float32),
        }
    )
    if "pred_link_points" not in result:
        raise KeyError("Remote safety response must contain 'pred_link_points'")
    return np.asarray(result["pred_link_points"], dtype=np.float32)


def prediction_link_ids(pred_link_points: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_link_points)
    if pred.ndim != 4:
        raise ValueError(f"pred_link_points must have shape (T, L, P, 3), got {pred.shape}")
    link_count = pred.shape[1]
    points_per_link = pred.shape[2]
    return np.tile(np.repeat(np.arange(link_count, dtype=np.int64), points_per_link), pred.shape[0])


def load_safe_space_for_video(path: Path) -> dict[str, np.ndarray]:
    required = (
        "obstacle_box_centers",
        "obstacle_box_axes",
        "obstacle_box_half_sizes",
        "obstacle_box_corners",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"safe-space file is missing required OBB arrays: {missing}")
        safe_space = {key: np.asarray(data[key], dtype=np.float32) for key in required}
    if safe_space["obstacle_box_centers"].ndim != 2 or safe_space["obstacle_box_centers"].shape[-1] != 3:
        raise ValueError("obstacle_box_centers must have shape (N, 3)")
    if safe_space["obstacle_box_axes"].shape != (safe_space["obstacle_box_centers"].shape[0], 3, 3):
        raise ValueError("obstacle_box_axes must have shape (N, 3, 3)")
    if safe_space["obstacle_box_half_sizes"].shape != safe_space["obstacle_box_centers"].shape:
        raise ValueError("obstacle_box_half_sizes must have shape (N, 3)")
    if safe_space["obstacle_box_corners"].shape != (safe_space["obstacle_box_centers"].shape[0], 8, 3):
        raise ValueError("obstacle_box_corners must have shape (N, 8, 3)")
    return safe_space


def empty_obstacle_safe_space() -> dict[str, np.ndarray]:
    return {
        "obstacle_box_centers": np.zeros((0, 3), dtype=np.float32),
        "obstacle_box_axes": np.zeros((0, 3, 3), dtype=np.float32),
        "obstacle_box_half_sizes": np.zeros((0, 3), dtype=np.float32),
        "obstacle_box_corners": np.zeros((0, 8, 3), dtype=np.float32),
        "obstacle_box_point_counts": np.zeros((0,), dtype=np.int64),
    }


def safe_space_obb_arrays(safe_space: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(safe_space["obstacle_box_centers"], dtype=np.float32)
    axes = np.asarray(safe_space["obstacle_box_axes"], dtype=np.float32)
    half_sizes = np.asarray(safe_space["obstacle_box_half_sizes"], dtype=np.float32)
    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError("obstacle_box_centers must have shape (N, 3)")
    if axes.shape != (centers.shape[0], 3, 3):
        raise ValueError(
            "obstacle_box_centers, obstacle_box_axes, and obstacle_box_half_sizes "
            "must describe the same number of OBBs"
        )
    if half_sizes.shape != centers.shape:
        raise ValueError(
            "obstacle_box_centers, obstacle_box_axes, and obstacle_box_half_sizes "
            "must describe the same number of OBBs"
        )
    return centers, axes, half_sizes


def normalize_obb_target_geom_name_patterns(patterns) -> tuple[str, ...]:
    if patterns is None:
        return ()
    normalized = tuple(str(item).strip().lower() for item in patterns if str(item).strip())
    if any(item in {"all", "none", "*"} for item in normalized):
        return ()
    return normalized


def _model_name(model, kind: str, idx: int) -> str:
    try:
        return str(getattr(model, f"{kind}_id2name")(int(idx)) or "")
    except Exception:
        names = getattr(model, f"{kind}_names", None)
        if names is not None and int(idx) < len(names):
            return str(names[int(idx)] or "")
    return ""


def realtime_obb_target_geom_ids(model, patterns) -> set[int] | None:
    """Resolve realtime OBB target geom ids by matching geom or parent body names.

    ``None`` means no target filtering, while an empty set means filtering was
    requested but no MuJoCo geom matched the requested patterns.
    """
    normalized = normalize_obb_target_geom_name_patterns(patterns)
    if not normalized:
        return None

    target_ids: set[int] = set()
    geom_bodyid = np.asarray(getattr(model, "geom_bodyid", []), dtype=np.int64).reshape(-1)
    for geom_id in range(int(getattr(model, "ngeom", len(geom_bodyid)))):
        names = [_model_name(model, "geom", geom_id).lower()]
        if geom_id < len(geom_bodyid):
            names.append(_model_name(model, "body", int(geom_bodyid[geom_id])).lower())
        if any(pattern in name for pattern in normalized for name in names):
            target_ids.add(int(geom_id))
    return target_ids


def geom_pixel_mask_from_segmentation(
    *,
    libero_pc,
    sim,
    camera_name: str,
    width: int,
    height: int,
    geom_ids: set[int],
) -> np.ndarray:
    if not geom_ids:
        return np.zeros((int(height), int(width)), dtype=bool)
    segmentation = np.asarray(
        libero_pc.render_segmentation(sim, camera_name, int(width), int(height)),
        dtype=np.int32,
    )
    obj_types = segmentation[..., 0]
    rendered_geom_ids = segmentation[..., 1]
    mask = np.isin(rendered_geom_ids, list(geom_ids))
    geom_objtype = None
    if hasattr(libero_pc, "mujoco_geom_objtype"):
        geom_objtype = libero_pc.mujoco_geom_objtype()
    if geom_objtype is not None:
        mask &= obj_types == int(geom_objtype)
    return mask


def real_robot_obstacle_collision(
    env,
    *,
    robot_geom_ids,
    target_geom_name_patterns=None,
) -> dict[str, object]:
    robot_geom_set = set(int(x) for x in np.asarray(list(robot_geom_ids), dtype=np.int64).reshape(-1))
    empty = {"collision": False, "contact_count": 0, "contact_geom_pairs": np.zeros((0, 2), dtype=np.int64)}
    if not hasattr(env, "sim") or not hasattr(env.sim, "data") or not hasattr(env.sim.data, "contact"):
        return empty
    if not hasattr(env.sim, "model"):
        return empty
    target_patterns = normalize_obb_target_geom_name_patterns(target_geom_name_patterns)
    target_geom_set = realtime_obb_target_geom_ids(env.sim.model, target_patterns) if target_patterns else None
    if target_geom_set is not None and not target_geom_set:
        return empty
    if target_geom_set is None:
        target_geom_set = {
            int(geom_id)
            for geom_id in range(int(getattr(env.sim.model, "ngeom", 0)))
            if int(geom_id) not in robot_geom_set
        }

    contacts = getattr(env.sim.data, "contact", [])
    ncon = int(getattr(env.sim.data, "ncon", len(contacts)))
    pairs: list[tuple[int, int]] = []
    for contact_index in range(ncon):
        contact = contacts[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        geom1_robot = geom1 in robot_geom_set
        geom2_robot = geom2 in robot_geom_set
        geom1_target = geom1 in target_geom_set
        geom2_target = geom2 in target_geom_set
        if (geom1_robot and geom2_target) or (geom2_robot and geom1_target):
            pairs.append((geom1, geom2))
    pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return {
        "collision": bool(len(pair_array) > 0),
        "contact_count": int(len(pair_array)),
        "contact_geom_pairs": pair_array,
    }


def build_realtime_safe_space_from_env(
    *,
    env,
    libero_pc,
    safe_space_builder,
    camera_names: tuple[str, ...] | list[str],
    width: int,
    height: int,
    stride: int,
    max_depth: float,
    robot_geom_ids,
    robot_mask_dilation: int,
    workspace_bounds: list[float] | tuple[float, ...] | np.ndarray | None,
    workspace_mode: str,
    workspace_margin: float,
    table_z: float | None,
    table_slab_height: float,
    table_obstacle_min_height: float,
    table_obstacle_max_height: float,
    component_voxel_size: float,
    component_connectivity: int,
    min_component_points: int,
    box_margin: float,
    box_shape: str,
    box_orientation: str,
    voxel_size: float,
    target_geom_name_patterns=None,
) -> dict[str, np.ndarray]:
    """Rebuild obstacle OBBs from the current simulator RGB-D state.

    Returns the same OBB fields as ``build_safe_space_from_pointcloud.py`` so
    video drawing and point-flow collision checks can share one geometry path.
    """
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    robot_geom_set = set(int(x) for x in np.asarray(list(robot_geom_ids), dtype=np.int64).reshape(-1))
    target_patterns = normalize_obb_target_geom_name_patterns(target_geom_name_patterns)
    target_geom_set = realtime_obb_target_geom_ids(env.sim.model, target_patterns) if target_patterns else None
    if target_geom_set is not None and not target_geom_set:
        print(f"[warn] no realtime OBB target geoms matched patterns {tuple(target_geom_name_patterns)!r}")
        return empty_obstacle_safe_space()
    for camera_name in camera_names:
        try:
            rgb, depth_m = libero_pc.render_rgbd(env.sim, camera_name, int(width), int(height))
            if target_geom_set is None:
                robot_mask = libero_pc.robot_pixel_mask(
                    sim=env.sim,
                    camera_name=camera_name,
                    width=int(width),
                    height=int(height),
                    robot_geom_ids=robot_geom_set,
                    dilation=int(robot_mask_dilation),
                )
                keep_mask = ~robot_mask
            else:
                keep_mask = geom_pixel_mask_from_segmentation(
                    libero_pc=libero_pc,
                    sim=env.sim,
                    camera_name=camera_name,
                    width=int(width),
                    height=int(height),
                    geom_ids=target_geom_set,
                )
        except Exception as exc:
            print(f"[warn] skipped realtime OBB camera {camera_name!r}: {exc}")
            continue
        points, colors = libero_pc.depth_to_world_points(
            sim=env.sim,
            camera_name=camera_name,
            rgb=rgb,
            depth_m=depth_m,
            stride=max(int(stride), 1),
            max_depth=float(max_depth),
            keep_mask=keep_mask,
        )
        if len(points) > 0:
            all_points.append(np.asarray(points, dtype=np.float32))
            all_colors.append(np.asarray(colors, dtype=np.uint8))

    if not all_points:
        return empty_obstacle_safe_space()

    points = np.concatenate(all_points, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(all_colors, axis=0).astype(np.uint8, copy=False)
    if workspace_bounds is not None:
        points, _colors = libero_pc.crop_workspace(points, colors, list(workspace_bounds))
    if len(points) == 0:
        return empty_obstacle_safe_space()

    resolved_table_z = float(table_z) if table_z is not None else float(
        safe_space_builder.estimate_table_z(points, float(voxel_size))
    )
    if workspace_bounds is not None:
        bounds = np.asarray(workspace_bounds, dtype=np.float32)
    elif workspace_mode == "table":
        bounds, _table_slab_points = safe_space_builder.estimate_table_workspace_bounds(
            points=points,
            margin=float(workspace_margin),
            table_z=resolved_table_z,
            slab_height=float(table_slab_height),
            voxel_size=float(voxel_size),
        )
    else:
        bounds = safe_space_builder.estimate_pointcloud_workspace_bounds(
            points=points,
            margin=float(workspace_margin),
            make_cube=False,
        )
    bounds = np.asarray(bounds, dtype=np.float32)
    bounds[4] = max(float(bounds[4]), resolved_table_z)

    tabletop_points = safe_space_builder.tabletop_obstacle_points(
        points=points,
        bounds=bounds,
        table_z=resolved_table_z,
        min_height=float(table_obstacle_min_height),
        max_height=float(table_obstacle_max_height),
    )
    if len(tabletop_points) == 0:
        return empty_obstacle_safe_space()

    (
        _box_mins,
        _box_maxs,
        box_centers,
        box_axes,
        box_half_sizes,
        box_corners,
        point_counts,
    ) = safe_space_builder.component_boxes_from_tabletop_points(
        points=tabletop_points,
        bounds=bounds,
        table_z=resolved_table_z,
        component_voxel_size=float(component_voxel_size),
        component_connectivity=int(component_connectivity),
        min_component_points=int(min_component_points),
        box_margin=float(box_margin),
        box_shape=box_shape,
        box_orientation=box_orientation,
    )
    return {
        "obstacle_box_centers": np.asarray(box_centers, dtype=np.float32),
        "obstacle_box_axes": np.asarray(box_axes, dtype=np.float32),
        "obstacle_box_half_sizes": np.asarray(box_half_sizes, dtype=np.float32),
        "obstacle_box_corners": np.asarray(box_corners, dtype=np.float32),
        "obstacle_box_point_counts": np.asarray(point_counts, dtype=np.int64),
    }


def point_flow_obb_collision(
    pred_link_points: np.ndarray,
    safe_space: dict[str, np.ndarray] | None,
    *,
    collision_margin: float = 0.0,
) -> dict[str, object]:
    if safe_space is None:
        return {
            "collision": False,
            "collision_point_count": 0,
            "collision_point_indices": np.zeros((0,), dtype=np.int64),
        }
    points = np.asarray(pred_link_points, dtype=np.float32).reshape(-1, 3)
    centers, axes, half_sizes = safe_space_obb_arrays(safe_space)
    if len(centers) == 0 or len(points) == 0:
        colliding = np.zeros((0,), dtype=np.int64)
    else:
        inside_any = np.zeros((points.shape[0],), dtype=bool)
        margin = float(collision_margin)
        for center, box_axes, box_half_sizes in zip(centers, axes, half_sizes):
            local = (points - center) @ box_axes
            inside_any |= np.all(np.abs(local) <= (box_half_sizes + margin + 1e-6), axis=-1)
        colliding = np.flatnonzero(inside_any).astype(np.int64)
    return {
        "collision": bool(len(colliding) > 0),
        "collision_point_count": int(len(colliding)),
        "collision_point_indices": colliding,
    }


def point_flow_obb_cbf_constraints(
    pred_link_points: np.ndarray,
    current_link_points: np.ndarray,
    safe_space: dict[str, np.ndarray] | None,
    *,
    collision_margin: float = 0.0,
    trigger_margin: float = 0.02,
    max_constraints: int = 32,
    allowed_time_indices: set[int] | frozenset[int] | None = None,
) -> list[PointFlowCbfConstraint]:
    """Select CBF constraints from predicted future robot-surface point flow.

    The prediction is only used as a trigger and active-set selector. Each
    returned constraint is anchored at the corresponding current FK surface
    point, so the downstream QP can use a current-state Jacobian.
    """
    if safe_space is None or max_constraints <= 0:
        return []
    pred = np.asarray(pred_link_points, dtype=np.float32)
    current = np.asarray(current_link_points, dtype=np.float32)
    if pred.ndim != 4 or pred.shape[-1] != 3:
        raise ValueError(f"pred_link_points must have shape (T, L, P, 3), got {pred.shape}")
    if current.shape != pred.shape[1:]:
        raise ValueError(f"current_link_points must have shape {pred.shape[1:]}, got {current.shape}")

    centers, axes, half_sizes = safe_space_obb_arrays(safe_space)
    if len(centers) == 0:
        return []

    constraints: list[PointFlowCbfConstraint] = []
    seen: set[tuple[int, int, int]] = set()
    allowed_times = None if allowed_time_indices is None else {int(item) for item in allowed_time_indices}
    cbf_margin = max(float(collision_margin), 0.0)
    trigger = cbf_margin + max(float(trigger_margin), 0.0)
    for obb_id, (center, box_axes, box_half_sizes) in enumerate(zip(centers, axes, half_sizes)):
        pred_local = (pred - center) @ box_axes
        trigger_half_sizes = box_half_sizes + trigger
        dangerous = np.all(np.abs(pred_local) <= (trigger_half_sizes + 1e-6), axis=-1)
        for time_index, link_id, point_id in np.argwhere(dangerous):
            if allowed_times is not None and int(time_index) not in allowed_times:
                continue
            key = (int(link_id), int(point_id), int(obb_id))
            if key in seen:
                continue
            seen.add(key)
            current_point = current[int(link_id), int(point_id)]
            current_local = (current_point - center) @ box_axes
            cbf_half_sizes = box_half_sizes + cbf_margin
            ratios = np.abs(current_local) / np.maximum(cbf_half_sizes, 1e-6)
            face_axis = int(np.argmax(ratios))
            sign = 1.0 if float(current_local[face_axis]) >= 0.0 else -1.0
            if abs(float(current_local[face_axis])) < 1e-9:
                pred_value = float(pred_local[int(time_index), int(link_id), int(point_id), face_axis])
                sign = 1.0 if pred_value >= 0.0 else -1.0
            normal = sign * np.asarray(box_axes[:, face_axis], dtype=np.float32)
            current_h = sign * float(current_local[face_axis]) - float(cbf_half_sizes[face_axis])
            predicted_h = (
                sign * float(pred_local[int(time_index), int(link_id), int(point_id), face_axis])
                - float(cbf_half_sizes[face_axis])
            )
            h = min(current_h, predicted_h)
            constraints.append(
                PointFlowCbfConstraint(
                    time_index=int(time_index),
                    link_id=int(link_id),
                    point_id=int(point_id),
                    obb_id=int(obb_id),
                    face_axis=face_axis,
                    normal=normal.astype(np.float32, copy=False),
                    h=float(h),
                    current_point=np.asarray(current_point, dtype=np.float32),
                    predicted_point=np.asarray(pred[int(time_index), int(link_id), int(point_id)], dtype=np.float32),
                )
            )
            if len(constraints) >= int(max_constraints):
                return sorted(constraints, key=lambda item: (item.h, item.time_index))
    return sorted(constraints, key=lambda item: (item.h, item.time_index))[: int(max_constraints)]


def current_point_obb_cbf_constraints(
    current_link_points: np.ndarray,
    safe_space: dict[str, np.ndarray] | None,
    *,
    collision_margin: float = 0.0,
    trigger_margin: float = 0.02,
    max_constraints: int = 32,
) -> list[PointFlowCbfConstraint]:
    """Select CBF constraints from current robot-surface points near OBBs."""
    if safe_space is None or max_constraints <= 0:
        return []
    current = np.asarray(current_link_points, dtype=np.float32)
    if current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError(f"current_link_points must have shape (L, P, 3), got {current.shape}")

    centers, axes, half_sizes = safe_space_obb_arrays(safe_space)
    if len(centers) == 0:
        return []

    constraints: list[PointFlowCbfConstraint] = []
    cbf_margin = max(float(collision_margin), 0.0)
    trigger = max(float(trigger_margin), 0.0)
    for obb_id, (center, box_axes, box_half_sizes) in enumerate(zip(centers, axes, half_sizes)):
        local = (current - center) @ box_axes
        cbf_half_sizes = box_half_sizes + cbf_margin
        ratios = np.abs(local) / np.maximum(cbf_half_sizes, 1e-6)
        face_axes = np.argmax(ratios, axis=-1)
        for link_id in range(current.shape[0]):
            for point_id in range(current.shape[1]):
                face_axis = int(face_axes[link_id, point_id])
                local_value = float(local[link_id, point_id, face_axis])
                sign = 1.0 if local_value >= 0.0 else -1.0
                h = sign * local_value - float(cbf_half_sizes[face_axis])
                if h > trigger:
                    continue
                normal = sign * np.asarray(box_axes[:, face_axis], dtype=np.float32)
                current_point = np.asarray(current[link_id, point_id], dtype=np.float32)
                constraints.append(
                    PointFlowCbfConstraint(
                        time_index=0,
                        link_id=int(link_id),
                        point_id=int(point_id),
                        obb_id=int(obb_id),
                        face_axis=face_axis,
                        normal=normal.astype(np.float32, copy=False),
                        h=float(h),
                        current_point=current_point,
                        predicted_point=current_point,
                    )
                )
                if len(constraints) >= int(max_constraints):
                    return sorted(constraints, key=lambda item: item.h)
    return sorted(constraints, key=lambda item: item.h)[: int(max_constraints)]


def predicted_frame_action_cbf_constraints(
    pred_link_points: np.ndarray,
    current_link_points: np.ndarray,
    safe_space: dict[str, np.ndarray] | None,
    *,
    collision_margin: float = 0.0,
    trigger_margin: float = 0.02,
    max_constraints: int = 32,
    allowed_time_indices: set[int] | frozenset[int] | None = None,
    include_current_points: bool = False,
) -> list[PointFlowCbfConstraint]:
    """Select predicted-frame constraints for chunk-indexed CBF."""
    if max_constraints <= 0:
        return []
    allowed_times = None if allowed_time_indices is None else {int(item) for item in allowed_time_indices}
    if include_current_points and (allowed_times is None or 0 in allowed_times):
        current_constraints = current_point_obb_cbf_constraints(
            current_link_points,
            safe_space,
            collision_margin=collision_margin,
            trigger_margin=trigger_margin,
            max_constraints=max_constraints,
        )
    else:
        current_constraints = []
    predicted_constraints = point_flow_obb_cbf_constraints(
        pred_link_points,
        current_link_points,
        safe_space,
        collision_margin=collision_margin,
        trigger_margin=trigger_margin,
        max_constraints=max_constraints,
        allowed_time_indices=allowed_time_indices,
    )
    constraints = current_constraints + predicted_constraints
    return sorted(constraints, key=lambda item: (item.h, item.time_index))[: int(max_constraints)]


def cbf_action_indices_from_constraints(
    constraints: list[PointFlowCbfConstraint],
    *,
    current_action_offset: int,
    action_count: int,
) -> list[int]:
    offset = int(current_action_offset)
    count = int(action_count)
    if offset < 0:
        raise ValueError(f"current_action_offset must be >= 0, got {offset}")
    if count < 0:
        raise ValueError(f"action_count must be >= 0, got {count}")
    return sorted(
        {
            offset + int(item.time_index)
            for item in constraints
            if 0 <= offset + int(item.time_index) < count
        }
    )


def apply_frame_indexed_cbf_corrections(
    action_chunk: np.ndarray,
    constraints: list[PointFlowCbfConstraint],
    *,
    current_action_offset: int,
    correct_action_fn,
) -> tuple[np.ndarray, dict[str, object]]:
    chunk = np.asarray(action_chunk, dtype=np.float64).copy()
    if chunk.ndim != 2:
        raise ValueError(f"action_chunk must have shape (T, A), got {chunk.shape}")
    action_indices = cbf_action_indices_from_constraints(
        constraints,
        current_action_offset=int(current_action_offset),
        action_count=int(chunk.shape[0]),
    )
    info: dict[str, object] = {
        "triggered": bool(action_indices),
        "constraint_count": int(len(constraints)),
        "success": True,
        "max_violation": 0.0,
        "corrected_action_indices": action_indices,
    }
    for action_index in action_indices:
        selected = [
            item
            for item in constraints
            if int(current_action_offset) + int(item.time_index) == int(action_index)
        ]
        corrected_action, correction_info = correct_action_fn(int(action_index), chunk[action_index].copy(), selected)
        chunk[action_index] = np.asarray(corrected_action, dtype=np.float64).reshape(chunk.shape[1])
        info["success"] = bool(info["success"]) and bool(correction_info.get("success", True))
        info["max_violation"] = max(float(info["max_violation"]), float(correction_info.get("max_violation", 0.0)))
        if "action_space" not in info and "action_space" in correction_info:
            info["action_space"] = correction_info["action_space"]
    return chunk, info


def cbf_corrected_action_masks(cbf_infos: list[dict[str, object]], *, action_count: int) -> np.ndarray:
    masks = np.zeros((len(cbf_infos), int(action_count)), dtype=bool)
    for sample_id, info in enumerate(cbf_infos):
        for action_index in info.get("corrected_action_indices", []):
            action_index = int(action_index)
            if 0 <= action_index < masks.shape[1]:
                masks[sample_id, action_index] = True
    return masks


def solve_cbf_qp_projection(
    nominal_action: np.ndarray,
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    iterations: int = 12,
    tolerance: float = 1e-7,
) -> CbfQpProjectionResult:
    """Project a nominal action onto linear CBF halfspaces.

    This is a dependency-light first-version QP projection. For one active
    halfspace it is the exact Euclidean projection. For several halfspaces it
    alternates projections and bound clipping, which is sufficient as a
    conservative online filter and easy to replace with OSQP later.
    """
    x = np.asarray(nominal_action, dtype=np.float64).reshape(-1).copy()
    a = np.asarray(a_matrix, dtype=np.float64)
    b = np.asarray(b_vector, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return CbfQpProjectionResult(action=x, success=True, max_violation=0.0, iterations=0)
    if a.ndim != 2 or a.shape[1] != x.size:
        raise ValueError(f"a_matrix must have shape (N, {x.size}), got {a.shape}")
    if b.shape != (a.shape[0],):
        raise ValueError(f"b_vector must have shape ({a.shape[0]},), got {b.shape}")

    lo = np.full_like(x, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.full_like(x, np.inf) if upper is None else np.asarray(upper, dtype=np.float64).reshape(-1)
    if lo.shape != x.shape or hi.shape != x.shape:
        raise ValueError(f"lower/upper bounds must have shape {x.shape}")
    x = np.minimum(np.maximum(x, lo), hi)

    completed_iterations = 0
    for iteration in range(max(int(iterations), 1)):
        completed_iterations = iteration + 1
        for row, rhs in zip(a, b):
            denom = float(row @ row)
            if denom <= 1e-12:
                continue
            violation = float(rhs - row @ x)
            if violation > 0.0:
                x = x + (violation / denom) * row
                x = np.minimum(np.maximum(x, lo), hi)
        if np.all((a @ x) >= (b - tolerance)):
            break

    halfspace_violation = float(np.max(np.maximum(b - a @ x, 0.0))) if len(b) else 0.0
    lower_violation = float(np.max(np.maximum(lo - x, 0.0))) if len(x) else 0.0
    upper_violation = float(np.max(np.maximum(x - hi, 0.0))) if len(x) else 0.0
    max_violation = max(halfspace_violation, lower_violation, upper_violation)
    return CbfQpProjectionResult(
        action=x.astype(np.float64, copy=False),
        success=bool(max_violation <= tolerance),
        max_violation=max_violation,
        iterations=completed_iterations,
    )


def cbf_qp_action_from_projection(
    projection: CbfQpProjectionResult,
    *,
    qp_to_action,
    qp_nominal: np.ndarray,
    nominal: np.ndarray,
    fallback: str,
) -> np.ndarray:
    if projection.success or fallback == "projected":
        return np.asarray(qp_to_action(projection.action), dtype=np.float64)
    if fallback == "zero":
        zero_qp_action = np.zeros_like(np.asarray(qp_nominal, dtype=np.float64))
        return np.asarray(qp_to_action(zero_qp_action), dtype=np.float64)
    if fallback == "nominal":
        return np.asarray(nominal, dtype=np.float64).copy()
    raise ValueError(f"Unsupported CBF fallback: {fallback}")


def _resolve_optional_action_bound(bound: list[float] | tuple[float, ...] | np.ndarray | None, dim: int) -> np.ndarray | None:
    if bound is None:
        return None
    arr = np.asarray(bound, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return np.full((dim,), float(arr[0]), dtype=np.float64)
    if arr.size < dim:
        raise ValueError(f"CBF action bound needs 1 or at least {dim} values, got {arr.size}")
    return arr[:dim].astype(np.float64, copy=False)


def resolve_cbf_action_space(requested: str, *, action_dim: int, arm_dim: int) -> str:
    if requested not in {"auto", "joint_delta", "cartesian_delta", "cartesian_action"}:
        raise ValueError(f"Unsupported CBF action space: {requested}")
    if requested != "auto":
        return requested
    # LIBERO / PI05 uses OSC_POSITION-like executable actions: either compact
    # xyz+gripper (4-D) or xyz+orientation+gripper (7-D). These are not arm
    # joint deltas even when action_dim equals the number of arm joints.
    return "cartesian_action" if int(action_dim) in (4, 7) and int(arm_dim) >= 7 else "joint_delta"


def executable_libero_action(action: np.ndarray, *, action_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    action_dim = int(action_dim)
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    if action_dim == 4 and action.size > 4:
        return np.concatenate((action[:3], action[-1:]), axis=0).astype(np.float64, copy=False)
    if action.size < action_dim:
        raise ValueError(f"action needs at least {action_dim} values, got {action.size}")
    return action[:action_dim].astype(np.float64, copy=False)


def damped_pseudoinverse(matrix: np.ndarray, *, damping: float = 1e-4) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {matrix.shape}")
    damping = float(damping)
    if damping < 0.0:
        raise ValueError(f"damping must be >= 0, got {damping}")
    if damping == 0.0:
        return np.linalg.pinv(matrix)
    rows = matrix.shape[0]
    return matrix.T @ np.linalg.inv(matrix @ matrix.T + (damping * damping) * np.eye(rows, dtype=np.float64))


def cartesian_delta_action_to_joint_delta(
    nominal_action: np.ndarray,
    eef_jacobian: np.ndarray,
    *,
    arm_dim: int,
    damping: float = 1e-4,
):
    nominal = np.asarray(nominal_action, dtype=np.float64).reshape(-1)
    if nominal.size < 3:
        raise ValueError(f"Cartesian action needs at least 3 xyz values, got {nominal.size}")
    jacobian = np.asarray(eef_jacobian, dtype=np.float64)
    arm_dim = int(arm_dim)
    if jacobian.shape != (3, arm_dim):
        raise ValueError(f"eef_jacobian must have shape (3, {arm_dim}), got {jacobian.shape}")
    pseudo_inverse = damped_pseudoinverse(jacobian, damping=float(damping))
    joint_delta = pseudo_inverse @ nominal[:3]

    def joint_delta_to_action(safe_joint_delta: np.ndarray) -> np.ndarray:
        safe_joint_delta = np.asarray(safe_joint_delta, dtype=np.float64).reshape(-1)
        if safe_joint_delta.shape != (arm_dim,):
            raise ValueError(f"safe_joint_delta must have shape ({arm_dim},), got {safe_joint_delta.shape}")
        safe = nominal.copy()
        safe[:3] = jacobian @ safe_joint_delta
        return safe

    return joint_delta.astype(np.float64, copy=False), joint_delta_to_action


def cartesian_action_to_qp_action(nominal_action: np.ndarray, *, action_dim: int):
    executable = executable_libero_action(nominal_action, action_dim=int(action_dim))
    if executable.size < 3:
        raise ValueError(f"Cartesian executable action needs at least 3 xyz values, got {executable.size}")
    action_xyz = executable[:3].astype(np.float64, copy=True)

    def xyz_to_action(safe_xyz: np.ndarray) -> np.ndarray:
        safe_xyz = np.asarray(safe_xyz, dtype=np.float64).reshape(-1)
        if safe_xyz.shape != (3,):
            raise ValueError(f"safe_xyz must have shape (3,), got {safe_xyz.shape}")
        safe = executable.copy()
        safe[:3] = safe_xyz
        return safe

    return action_xyz, xyz_to_action


def finite_difference_eef_position_jacobian(
    eef_position_fn,
    q: np.ndarray,
    *,
    eps: float = 1e-4,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if eps <= 0.0:
        raise ValueError("finite-difference eps must be positive")
    jacobian = np.zeros((3, q.size), dtype=np.float64)
    for joint_idx in range(q.size):
        delta = np.zeros_like(q)
        delta[joint_idx] = float(eps)
        pos_plus = np.asarray(eef_position_fn(q + delta), dtype=np.float64).reshape(-1)
        pos_minus = np.asarray(eef_position_fn(q - delta), dtype=np.float64).reshape(-1)
        if pos_plus.shape != (3,) or pos_minus.shape != (3,):
            raise ValueError("eef_position_fn must return shape (3,)")
        jacobian[:, joint_idx] = (pos_plus - pos_minus) / (2.0 * float(eps))
    return jacobian


def eef_body_position(env, *, body_name: str = "gripper0_eef") -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def finite_difference_point_jacobians(
    point_position_fn,
    q: np.ndarray,
    point_keys: list[tuple[int, int]],
    *,
    eps: float = 1e-4,
) -> dict[tuple[int, int], np.ndarray]:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if eps <= 0.0:
        raise ValueError("finite-difference eps must be positive")
    unique_keys = sorted(set((int(link_id), int(point_id)) for link_id, point_id in point_keys))
    jacobians = {key: np.zeros((3, q.size), dtype=np.float64) for key in unique_keys}
    for joint_idx in range(q.size):
        delta = np.zeros_like(q)
        delta[joint_idx] = float(eps)
        points_plus = np.asarray(point_position_fn(q + delta), dtype=np.float64)
        points_minus = np.asarray(point_position_fn(q - delta), dtype=np.float64)
        if points_plus.shape != points_minus.shape or points_plus.ndim != 3 or points_plus.shape[-1] != 3:
            raise ValueError("point_position_fn must return link points with shape (L, P, 3)")
        diff = (points_plus - points_minus) / (2.0 * float(eps))
        for key in unique_keys:
            jacobians[key][:, joint_idx] = diff[key[0], key[1]]
    return jacobians


def finite_difference_action_point_jacobians(
    action_position_fn,
    action_xyz: np.ndarray,
    point_keys: list[tuple[int, int]],
    *,
    eps: float = 1e-4,
) -> dict[tuple[int, int], np.ndarray]:
    action_xyz = np.asarray(action_xyz, dtype=np.float64).reshape(-1)
    if action_xyz.shape != (3,):
        raise ValueError(f"action_xyz must have shape (3,), got {action_xyz.shape}")
    if eps <= 0.0:
        raise ValueError("finite-difference eps must be positive")
    unique_keys = sorted(set((int(link_id), int(point_id)) for link_id, point_id in point_keys))
    jacobians = {key: np.zeros((3, action_xyz.size), dtype=np.float64) for key in unique_keys}
    for action_idx in range(action_xyz.size):
        delta = np.zeros_like(action_xyz)
        delta[action_idx] = float(eps)
        points_plus = np.asarray(action_position_fn(action_xyz + delta), dtype=np.float64)
        points_minus = np.asarray(action_position_fn(action_xyz - delta), dtype=np.float64)
        if points_plus.shape != points_minus.shape or points_plus.ndim != 3 or points_plus.shape[-1] != 3:
            raise ValueError("action_position_fn must return link points with shape (L, P, 3)")
        diff = (points_plus - points_minus) / (2.0 * float(eps))
        for key in unique_keys:
            jacobians[key][:, action_idx] = diff[key[0], key[1]]
    return jacobians


def snapshot_env_runtime_state(env) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for name in ("timestep", "cur_time", "_elapsed_steps"):
        if hasattr(env, name):
            snapshot[name] = getattr(env, name)
    return snapshot


def restore_env_runtime_state(env, snapshot: dict[str, object]) -> None:
    for name, value in snapshot.items():
        setattr(env, name, value)


def filter_action_with_pointflow_cbf_qp(
    *,
    env,
    dataset_builder,
    qpos_indices: np.ndarray,
    geom_ids: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    nominal_action: np.ndarray,
    action_dim: int,
    current_link_points: np.ndarray,
    pred_link_points: np.ndarray,
    safe_space: dict[str, np.ndarray] | None,
    points_per_link: int,
    samples_per_action: int,
    skeleton_source: str,
    collision_margin: float,
    trigger_margin: float,
    alpha: float,
    max_constraints: int,
    finite_difference_eps: float,
    projection_iterations: int,
    action_lower: list[float] | tuple[float, ...] | np.ndarray | None,
    action_upper: list[float] | tuple[float, ...] | np.ndarray | None,
    fallback: str,
    trigger_source: str = "predicted_point_flow",
    action_space: str = "auto",
    cartesian_damping: float = 1e-4,
    precomputed_constraints: list[PointFlowCbfConstraint] | None = None,
    allowed_time_indices: set[int] | frozenset[int] | None = None,
    include_current_points: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    nominal = np.asarray(nominal_action, dtype=np.float64).reshape(-1)
    arm_dim = int(len(qpos_indices))
    resolved_action_space = resolve_cbf_action_space(action_space, action_dim=int(action_dim), arm_dim=arm_dim)
    if resolved_action_space == "joint_delta" and nominal.size < arm_dim:
        raise ValueError(f"joint_delta CBF action needs at least {arm_dim} arm values, got {nominal.size}")
    if resolved_action_space == "cartesian_delta" and nominal.size < 3:
        raise ValueError(f"cartesian_delta CBF action needs at least 3 xyz values, got {nominal.size}")
    if precomputed_constraints is not None:
        constraints = list(precomputed_constraints)
    elif trigger_source == "predicted_point_flow":
        constraints = predicted_frame_action_cbf_constraints(
            pred_link_points,
            current_link_points,
            safe_space,
            collision_margin=collision_margin,
            trigger_margin=trigger_margin,
            max_constraints=max_constraints,
            allowed_time_indices=allowed_time_indices,
            include_current_points=include_current_points,
        )
    elif trigger_source == "current_pointcloud":
        constraints = current_point_obb_cbf_constraints(
            current_link_points,
            safe_space,
            collision_margin=collision_margin,
            trigger_margin=trigger_margin,
            max_constraints=max_constraints,
        )
    else:
        raise ValueError(f"Unsupported CBF trigger source: {trigger_source}")
    info: dict[str, object] = {
        "triggered": bool(constraints),
        "constraint_count": int(len(constraints)),
        "success": True,
        "max_violation": 0.0,
        "trigger_source": trigger_source,
        "action_space": resolved_action_space,
    }
    if not constraints:
        return nominal.astype(np.float64, copy=False), info

    q = np.asarray(env.sim.data.qpos[qpos_indices], dtype=np.float64).reshape(-1)
    low = np.asarray(low, dtype=np.float64).reshape(-1)[:arm_dim]
    high = np.asarray(high, dtype=np.float64).reshape(-1)[:arm_dim]
    lower = low - q
    upper = high - q
    user_lower = _resolve_optional_action_bound(action_lower, arm_dim)
    user_upper = _resolve_optional_action_bound(action_upper, arm_dim)
    if user_lower is not None:
        lower = np.maximum(lower, user_lower)
    if user_upper is not None:
        upper = np.minimum(upper, user_upper)

    zero_action_chunk = np.zeros((0, arm_dim), dtype=np.float64)

    def set_q_and_forward(q_vector: np.ndarray) -> None:
        swept = dataset_builder.import_script_module("libero_joint_swept_pointcloud")
        swept.set_arm_joint_vector(env.sim, qpos_indices, np.asarray(q_vector, dtype=np.float64))

    def point_position_fn(q_vector: np.ndarray) -> np.ndarray:
        def target_builder():
            return dataset_builder.fk_target_link_points(
                env,
                qpos_indices,
                geom_ids,
                np.asarray(q_vector, dtype=np.float64),
                zero_action_chunk,
                int(points_per_link),
                int(samples_per_action),
                low,
                high,
                skeleton_source,
            )

        target, _link_names = collector.compute_fk_target_preserving_sim_state(env, target_builder)
        return np.asarray(target[0], dtype=np.float64)

    point_keys = [(item.link_id, item.point_id) for item in constraints]
    if resolved_action_space == "joint_delta":
        qp_nominal = nominal[:arm_dim]
        qp_lower = lower
        qp_upper = upper

        def qp_to_action(safe_joint_delta: np.ndarray) -> np.ndarray:
            safe = nominal.copy()
            safe[:arm_dim] = np.asarray(safe_joint_delta, dtype=np.float64).reshape(-1)
            return safe

        jacobians = finite_difference_point_jacobians(
            point_position_fn,
            q,
            point_keys,
            eps=float(finite_difference_eps),
        )
    elif resolved_action_space == "cartesian_delta":
        def eef_position_fn(q_vector: np.ndarray) -> np.ndarray:
            def position_builder():
                set_q_and_forward(q_vector)
                return eef_body_position(env)

            return collector.compute_fk_target_preserving_sim_state(env, position_builder)

        eef_jacobian = finite_difference_eef_position_jacobian(
            eef_position_fn,
            q,
            eps=float(finite_difference_eps),
        )
        qp_nominal, qp_to_action = cartesian_delta_action_to_joint_delta(
            nominal,
            eef_jacobian,
            arm_dim=arm_dim,
            damping=float(cartesian_damping),
        )
        qp_lower = lower
        qp_upper = upper

        jacobians = finite_difference_point_jacobians(
            point_position_fn,
            q,
            point_keys,
            eps=float(finite_difference_eps),
        )
    else:
        qp_nominal, qp_to_action = cartesian_action_to_qp_action(nominal, action_dim=int(action_dim))
        qp_lower = np.full_like(qp_nominal, -np.inf, dtype=np.float64)
        qp_upper = np.full_like(qp_nominal, np.inf, dtype=np.float64)
        user_lower = _resolve_optional_action_bound(action_lower, qp_nominal.size)
        user_upper = _resolve_optional_action_bound(action_upper, qp_nominal.size)
        if user_lower is not None:
            qp_lower = np.maximum(qp_lower, user_lower)
        if user_upper is not None:
            qp_upper = np.minimum(qp_upper, user_upper)

        def action_position_fn(action_xyz: np.ndarray) -> np.ndarray:
            executable_action = qp_to_action(action_xyz)
            env_runtime_state = snapshot_env_runtime_state(env)
            snapshot = collector.snapshot_sim_state(env.sim)
            try:
                env.step(executable_action.tolist())
                q_after = np.asarray(env.sim.data.qpos[qpos_indices], dtype=np.float64).reshape(-1)
                return point_position_fn(q_after)
            finally:
                collector.restore_sim_state(env.sim, snapshot)
                restore_env_runtime_state(env, env_runtime_state)

        jacobians = finite_difference_action_point_jacobians(
            action_position_fn,
            qp_nominal,
            point_keys,
            eps=float(finite_difference_eps),
        )
    a_rows = []
    b_values = []
    for item in constraints:
        jacobian = jacobians[(item.link_id, item.point_id)]
        a_rows.append(np.asarray(item.normal, dtype=np.float64) @ jacobian)
        b_values.append(-float(alpha) * float(item.h))

    projection = solve_cbf_qp_projection(
        qp_nominal,
        np.asarray(a_rows, dtype=np.float64),
        np.asarray(b_values, dtype=np.float64),
        lower=qp_lower,
        upper=qp_upper,
        iterations=int(projection_iterations),
    )
    safe = cbf_qp_action_from_projection(
        projection,
        qp_to_action=qp_to_action,
        qp_nominal=qp_nominal,
        nominal=nominal,
        fallback=fallback,
    )

    info.update(
        {
            "success": bool(projection.success),
            "max_violation": float(projection.max_violation),
            "iterations": int(projection.iterations),
            "qp_nominal_norm": float(np.linalg.norm(qp_nominal)),
            "fallback": str(fallback),
        }
    )
    return safe.astype(np.float64, copy=False), info


def draw_projected_obbs(
    frame: np.ndarray,
    *,
    sim,
    swept,
    camera_name: str,
    width: int,
    height: int,
    obb_corners: np.ndarray | None,
    color: tuple[int, int, int] = (255, 120, 25),
    line_width: int = 2,
) -> np.ndarray:
    if obb_corners is None or len(obb_corners) == 0:
        return np.asarray(frame, dtype=np.uint8)
    from PIL import Image, ImageDraw

    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    corners = np.asarray(obb_corners, dtype=np.float32).reshape(-1, 8, 3)
    flat_corners = corners.reshape(-1, 3)
    uv, valid = swept.project_world_points_to_camera_pixels(sim, camera_name, width, height, flat_corners)
    uv = uv.reshape(corners.shape[0], 8, 2)
    valid = valid.reshape(corners.shape[0], 8)
    for box_idx in range(corners.shape[0]):
        for i0, i1 in edges:
            if not (valid[box_idx, i0] and valid[box_idx, i1]):
                continue
            p0 = tuple(float(v) for v in uv[box_idx, i0])
            p1 = tuple(float(v) for v in uv[box_idx, i1])
            draw.line((p0, p1), fill=color, width=max(int(line_width), 1))
    return np.asarray(image, dtype=np.uint8)


def append_prediction_video_frame(
    buffer: VideoFrameBuffer,
    *,
    env,
    swept,
    pred_link_points: np.ndarray,
    obb_corners: np.ndarray | None = None,
    collision_result: dict[str, object] | None = None,
    real_collision_result: dict[str, object] | None = None,
    camera_name: str,
    width: int,
    height: int,
    point_radius: int,
    rollout_id: int,
    step_id: int,
    sample_id: int,
) -> None:
    if not buffer.enabled:
        return
    pred = np.asarray(pred_link_points, dtype=np.float32)
    points = pred.reshape(-1, 3)
    link_ids = prediction_link_ids(pred)
    colors = swept.point_colors(link_ids)
    background = swept.render_camera_rgb(env.sim, camera_name, width, height)
    frame = swept.projected_point_image(
        env.sim,
        camera_name,
        width,
        height,
        points,
        colors,
        point_radius,
        background=background,
    )
    frame = draw_projected_obbs(
        frame,
        sim=env.sim,
        swept=swept,
        camera_name=camera_name,
        width=width,
        height=height,
        obb_corners=obb_corners,
    )
    frame = annotate_video_frame(
        frame,
        rollout_id=rollout_id,
        step_id=step_id,
        sample_id=sample_id,
        collision_result=collision_result,
        real_collision_result=real_collision_result,
    )
    buffer.append(frame)


def annotate_video_frame(
    frame: np.ndarray,
    *,
    rollout_id: int,
    step_id: int,
    sample_id: int,
    collision_result: dict[str, object] | None = None,
    real_collision_result: dict[str, object] | None = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    text = f"rollout {rollout_id} | step {step_id} | pred {sample_id}"
    x0, y0 = 6, 6
    bbox = draw.textbbox((x0, y0), text)
    draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(0, 0, 0))
    draw.text((x0, y0), text, fill=(255, 255, 255))
    if real_collision_result is not None:
        real_collides = bool(real_collision_result.get("collision", False))
        contact_count = int(real_collision_result.get("contact_count", 0))
        real_status = f"REAL COLLISION | contacts: {contact_count}" if real_collides else "NO REAL COLLISION | contacts: 0"
        real_fill = (220, 100, 0) if real_collides else (0, 90, 120)
        x_real, y_real = 6, max(6, image.height - 44)
        real_bbox = draw.textbbox((x_real, y_real), real_status)
        draw.rectangle(
            (real_bbox[0] - 4, real_bbox[1] - 3, real_bbox[2] + 4, real_bbox[3] + 3),
            fill=real_fill,
        )
        draw.text((x_real, y_real), real_status, fill=(255, 255, 255))
    if collision_result is not None:
        collides = bool(collision_result.get("collision", False))
        count = int(collision_result.get("collision_point_count", 0))
        status = f"POSSIBLE COLLISION | points in OBB: {count}" if collides else "SAFE | points in OBB: 0"
        fill = (180, 0, 0) if collides else (0, 110, 30)
        x1, y1 = 6, max(6, image.height - 24)
        status_bbox = draw.textbbox((x1, y1), status)
        draw.rectangle(
            (status_bbox[0] - 4, status_bbox[1] - 3, status_bbox[2] + 4, status_bbox[3] + 3),
            fill=fill,
        )
        draw.text((x1, y1), status, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def compute_point_error_metrics(pred_link_points: np.ndarray, target_link_points: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_link_points, dtype=np.float32)
    target = np.asarray(target_link_points, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"prediction shape {pred.shape} does not match target shape {target.shape}")
    diff = pred - target
    l2 = np.linalg.norm(diff, axis=-1)
    return {
        "mse": float(np.mean(diff * diff)),
        "mean_l2": float(np.mean(l2)),
        "max_l2": float(np.max(l2)),
    }


def stack_metric_dicts(sample_metrics: list[dict[str, float]]) -> dict[str, np.ndarray]:
    sample_mse = np.asarray([metrics["mse"] for metrics in sample_metrics], dtype=np.float32)
    sample_mean_l2 = np.asarray([metrics["mean_l2"] for metrics in sample_metrics], dtype=np.float32)
    sample_max_l2 = np.asarray([metrics["max_l2"] for metrics in sample_metrics], dtype=np.float32)
    return {
        "sample_mse": sample_mse,
        "sample_mean_l2": sample_mean_l2,
        "sample_max_l2": sample_max_l2,
        "mean_mse": np.asarray(float(sample_mse.mean()), dtype=np.float32),
        "mean_l2": np.asarray(float(sample_mean_l2.mean()), dtype=np.float32),
        "max_l2": np.asarray(float(sample_max_l2.max()), dtype=np.float32),
    }


def cbf_action_delta_norms(nominal_actions: np.ndarray, executed_actions: np.ndarray) -> np.ndarray:
    nominal = np.asarray(nominal_actions, dtype=np.float32)
    executed = np.asarray(executed_actions, dtype=np.float32)
    if nominal.shape != executed.shape:
        raise ValueError(f"nominal_actions and executed_actions must have the same shape, got {nominal.shape} and {executed.shape}")
    if nominal.ndim != 2:
        raise ValueError(f"nominal_actions must have shape (S, A), got {nominal.shape}")
    return np.linalg.norm(executed - nominal, axis=1).astype(np.float32)


def save_evaluation(
    output: Path,
    *,
    pred_link_points: np.ndarray,
    target_link_points: np.ndarray,
    prefix_tokens_shape: np.ndarray,
    action_chunks: np.ndarray,
    metrics: dict[str, np.ndarray],
    rollout_ids: np.ndarray,
    step_ids: np.ndarray,
    link_names: np.ndarray,
    coordinate_frame: str,
    collision_flags: np.ndarray | None = None,
    collision_point_counts: np.ndarray | None = None,
    real_collision_flags: np.ndarray | None = None,
    real_collision_contact_counts: np.ndarray | None = None,
    nominal_actions: np.ndarray | None = None,
    executed_actions: np.ndarray | None = None,
    cbf_action_delta_norms: np.ndarray | None = None,
    cbf_triggered: np.ndarray | None = None,
    cbf_success: np.ndarray | None = None,
    cbf_constraint_counts: np.ndarray | None = None,
    cbf_max_violations: np.ndarray | None = None,
    cbf_corrected_action_masks: np.ndarray | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        pred_link_points=np.asarray(pred_link_points, dtype=np.float32),
        target_link_points=np.asarray(target_link_points, dtype=np.float32),
        prefix_tokens_shape=np.asarray(prefix_tokens_shape, dtype=np.int64),
        action_chunks=np.asarray(action_chunks, dtype=np.float32),
        rollout_ids=np.asarray(rollout_ids, dtype=np.int64),
        step_ids=np.asarray(step_ids, dtype=np.int64),
        link_names=np.asarray(link_names),
        coordinate_frame=np.asarray(coordinate_frame),
        pred_link_points_frame=np.asarray(coordinate_frame),
        target_link_points_frame=np.asarray(coordinate_frame),
        **metrics,
    )
    if collision_flags is not None:
        payload["collision_flags"] = np.asarray(collision_flags, dtype=bool)
    if collision_point_counts is not None:
        payload["collision_point_counts"] = np.asarray(collision_point_counts, dtype=np.int64)
    if real_collision_flags is not None:
        payload["real_collision_flags"] = np.asarray(real_collision_flags, dtype=bool)
    if real_collision_contact_counts is not None:
        payload["real_collision_contact_counts"] = np.asarray(real_collision_contact_counts, dtype=np.int64)
    if nominal_actions is not None:
        payload["nominal_actions"] = np.asarray(nominal_actions, dtype=np.float32)
    if executed_actions is not None:
        payload["executed_actions"] = np.asarray(executed_actions, dtype=np.float32)
    if cbf_action_delta_norms is not None:
        payload["cbf_action_delta_norms"] = np.asarray(cbf_action_delta_norms, dtype=np.float32)
    if cbf_triggered is not None:
        payload["cbf_triggered"] = np.asarray(cbf_triggered, dtype=bool)
    if cbf_success is not None:
        payload["cbf_success"] = np.asarray(cbf_success, dtype=bool)
    if cbf_constraint_counts is not None:
        payload["cbf_constraint_counts"] = np.asarray(cbf_constraint_counts, dtype=np.int64)
    if cbf_max_violations is not None:
        payload["cbf_max_violations"] = np.asarray(cbf_max_violations, dtype=np.float32)
    if cbf_corrected_action_masks is not None:
        payload["cbf_corrected_action_masks"] = np.asarray(cbf_corrected_action_masks, dtype=bool)
    np.savez_compressed(output, **payload)


def validate_pred_target_shape(pred: np.ndarray, target: np.ndarray) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            "decoder output shape does not match FK target shape. "
            f"pred={pred.shape}, target={target.shape}. "
            "Use a decoder trained with the same points_per_link / samples_per_action / skeleton_source."
        )


def evaluate_online(args: argparse.Namespace) -> dict[str, object]:
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")
    if args.num_rollouts <= 0:
        raise ValueError("--num-rollouts must be > 0")
    if args.replan_steps <= 0:
        raise ValueError("--replan-steps must be > 0")
    if args.samples_per_action < 1:
        raise ValueError("--samples-per-action must be >= 1")
    if args.points_per_link < 2:
        raise ValueError("--points-per-link must be >= 2")
    if args.prediction_steps <= 0:
        raise ValueError("--prediction-steps must be > 0")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be > 0")
    if args.collision_margin < 0.0:
        raise ValueError("--collision-margin must be >= 0")
    if getattr(args, "cbf_trigger_margin", 0.0) < 0.0:
        raise ValueError("--cbf-trigger-margin must be >= 0")
    if getattr(args, "cbf_max_constraints", 1) <= 0:
        raise ValueError("--cbf-max-constraints must be > 0")
    if getattr(args, "cbf_finite_difference_eps", 1e-4) <= 0.0:
        raise ValueError("--cbf-finite-difference-eps must be > 0")
    if getattr(args, "cbf_projection_iterations", 1) <= 0:
        raise ValueError("--cbf-projection-iterations must be > 0")
    if getattr(args, "cbf_alpha", 1.0) < 0.0:
        raise ValueError("--cbf-alpha must be >= 0")
    if getattr(args, "cbf_cartesian_damping", 1e-4) < 0.0:
        raise ValueError("--cbf-cartesian-damping must be >= 0")
    if getattr(args, "scene_obstacle", "none") == "none" and getattr(args, "scene_obstacle_xy", None) is not None:
        raise ValueError("--scene-obstacle-xy requires --scene-obstacle wine_bottle")
    if args.realtime_obbs:
        if args.obb_width <= 0 or args.obb_height <= 0:
            raise ValueError("--obb-width and --obb-height must be > 0")
        if args.obb_stride <= 0:
            raise ValueError("--obb-stride must be > 0")
        if args.obb_max_depth <= 0.0:
            raise ValueError("--obb-max-depth must be > 0")
        if args.obb_table_obstacle_max_height <= args.obb_table_obstacle_min_height:
            raise ValueError("--obb-table-obstacle-max-height must be greater than --obb-table-obstacle-min-height")
        if args.obb_component_voxel_size <= 0.0:
            raise ValueError("--obb-component-voxel-size must be > 0")
        if args.obb_component_connectivity not in (6, 18, 26):
            raise ValueError("--obb-component-connectivity must be one of 6, 18, or 26")
        if args.obb_min_component_points <= 0:
            raise ValueError("--obb-min-component-points must be > 0")
        if args.obb_voxel_size <= 0.0:
            raise ValueError("--obb-voxel-size must be > 0")

    collector.ensure_third_party_paths()
    if args.mujoco_gl is not None:
        collector.os.environ["MUJOCO_GL"] = args.mujoco_gl

    dataset_builder = collector.load_repo_script_module("build_pi05_safety_decoder_dataset")
    swept = dataset_builder.import_script_module("libero_joint_swept_pointcloud")
    libero_pc = dataset_builder.import_script_module("libero_reconstruct_pointcloud")
    swept.load_runtime_dependencies()
    if args.realtime_obbs and hasattr(libero_pc, "load_runtime_dependencies"):
        libero_pc.load_runtime_dependencies()
    safe_space_builder = load_repo_script_module("build_safe_space_from_pointcloud") if args.realtime_obbs else None

    np.random.seed(args.seed)
    policy = collector.load_remote_policy(host=args.policy_server_host, port=args.policy_server_port)
    get_metadata = getattr(policy, "get_server_metadata", None)
    server_metadata = dict(get_metadata()) if callable(get_metadata) else {}
    safety_prediction_source = select_safety_prediction_source(
        getattr(args, "safety_prediction_source", "auto"),
        server_metadata,
    )

    device = torch.device(resolve_device_name(getattr(args, "device", "auto")))
    loaded_model = None
    if safety_prediction_source == "local":
        loaded_model = load_safety_model_checkpoint(args.checkpoint, device)
        model_type = loaded_model.model_type
    else:
        model_type = str(server_metadata.get("safety_model_type", ""))
        if model_type not in {"flow", "decoder"}:
            raise ValueError(f"Remote safety server returned unsupported safety_model_type={model_type!r}")

    flow_points_per_link = args.points_per_link
    if model_type == "flow":
        flow_max_points = (
            int(loaded_model.model.flow_head.max_points)
            if loaded_model is not None
            else int(server_metadata["safety_flow_max_points"])
        )
        flow_points_per_link = infer_flow_points_per_link(
            max_points=flow_max_points,
            skeleton_source=args.skeleton_source,
            requested_points_per_link=args.points_per_link,
        )
        if flow_points_per_link != args.points_per_link:
            print(
                f"[info] overriding --points-per-link {args.points_per_link} -> {flow_points_per_link} "
                f"to match flow checkpoint max_points={flow_max_points}"
            )

    task_suite = collector.create_libero_task_suite(args.task_suite)
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    max_steps = args.max_steps if args.max_steps is not None else collector.default_max_steps(args.task_suite)

    obstacle_xy = None
    if getattr(args, "scene_obstacle_xy", None) is not None:
        obstacle_xy = (float(args.scene_obstacle_xy[0]), float(args.scene_obstacle_xy[1]))
    scene_obstacle = collector.SceneObstacleSpec(kind=getattr(args, "scene_obstacle", "none"), xy=obstacle_xy)
    env, task_description = collector.create_libero_env(
        task,
        resolution=args.env_resolution,
        seed=args.seed,
        scene_obstacle=scene_obstacle,
    )
    pred_samples: list[np.ndarray] = []
    target_samples: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    nominal_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    prefix_shapes: list[tuple[int, ...]] = []
    metrics: list[dict[str, float]] = []
    rollout_ids: list[int] = []
    step_ids: list[int] = []
    link_names = np.asarray([])
    video_buffer = VideoFrameBuffer(enabled=not args.no_video)
    video_path = args.video_output
    static_safe_space = load_safe_space_for_video(args.safe_space) if args.safe_space is not None else None
    collision_results: list[dict[str, object]] = []
    real_collision_results: list[dict[str, object]] = []
    cbf_infos: list[dict[str, object]] = []

    try:
        qpos_indices = swept.get_arm_qpos_indices(env)
        low, high = swept.joint_limits(env.sim, qpos_indices)
        geom_ids = libero_pc.find_robot_geoms(env)
        geom_ids_array = collector.robot_geom_ids_array(geom_ids)
        dummy_action = collector.make_dummy_action(env)

        for rollout_id in range(args.num_rollouts):
            if len(pred_samples) >= args.max_samples:
                break
            env.reset()
            init_state = initial_states[rollout_id % len(initial_states)]
            init_state = collector.adapt_init_state_for_scene_obstacle(init_state, env, scene_obstacle)
            obs = env.set_init_state(init_state)
            refreshed_obs = collector.reset_scene_obstacle_pose(env, scene_obstacle, refresh_observation=True)
            if refreshed_obs is not None:
                obs = refreshed_obs
            for _ in range(args.num_steps_wait):
                obs, _reward, done, _info = env.step(dummy_action)
                if done:
                    break
            refreshed_obs = collector.reset_scene_obstacle_pose(env, scene_obstacle, refresh_observation=True)
            if refreshed_obs is not None:
                obs = refreshed_obs

            step_id = 0
            done = False
            control_action_chunk = None
            control_prefix_tokens = None
            control_action_offset = 0
            control_replan_offset = 0
            while not done and step_id < max_steps and len(pred_samples) < args.max_samples:
                element = collector.build_libero_policy_input(
                    obs,
                    prompt=task_description,
                    resize_size=args.resize_size,
                )
                need_control_query = (
                    control_action_chunk is None
                    or control_action_offset >= len(control_action_chunk)
                    or control_replan_offset >= args.replan_steps
                )
                chunk_aligned_cbf = (
                    bool(getattr(args, "enable_cbf_qp", False))
                    and getattr(args, "cbf_correction_target", "predicted_frame_action") == "predicted_frame_action"
                    and getattr(args, "cbf_trigger_source", "predicted_point_flow") == "predicted_point_flow"
                )
                if need_control_query or not chunk_aligned_cbf:
                    action_chunk, prefix_tokens = collector.query_policy_action_and_prefix(
                        policy,
                        element,
                        remote_prefix_tokens=True,
                    )
                    if action_chunk.shape[0] <= 0:
                        raise ValueError("policy returned an empty action_chunk")
                else:
                    action_chunk = np.asarray(control_action_chunk, dtype=np.float32)
                    prefix_tokens = np.asarray(control_prefix_tokens, dtype=np.float32)
                if need_control_query:
                    control_action_chunk = np.asarray(action_chunk, dtype=np.float64).copy()
                    control_prefix_tokens = np.asarray(prefix_tokens, dtype=np.float32).copy()
                    control_action_offset = 0
                    control_replan_offset = 0
                start_joint_vector = np.asarray(env.sim.data.qpos[qpos_indices], dtype=np.float32)

                def target_builder():
                    points_per_link = (
                        int(loaded_model.config.points_per_link)
                        if loaded_model is not None and model_type == "decoder"
                        else int(flow_points_per_link)
                    )
                    return dataset_builder.fk_target_link_points(
                        env,
                        qpos_indices,
                        geom_ids_array,
                        start_joint_vector,
                        action_chunk,
                        points_per_link,
                        args.samples_per_action,
                        low,
                        high,
                        args.skeleton_source,
                    )

                target, link_names = collector.compute_fk_target_preserving_sim_state(env, target_builder)
                current_link_points = target[0]
                target_future = target
                if model_type == "flow":
                    target_future = target[1:]
                    if safety_prediction_source == "remote":
                        pred = query_remote_safety_prediction(
                            policy,
                            prefix_tokens=prefix_tokens,
                            current_link_points=current_link_points,
                        )
                    else:
                        pred = predict_safety_flow_link_points(
                            loaded_model.model,
                            prefix_tokens,
                            current_link_points,
                            device=device,
                            prediction_steps=args.prediction_steps,
                        )
                else:
                    if safety_prediction_source == "remote":
                        pred = query_remote_safety_prediction(
                            policy,
                            prefix_tokens=prefix_tokens,
                            current_link_points=current_link_points,
                        )
                    else:
                        pred = predict_link_points(loaded_model.model, prefix_tokens, device)
                validate_pred_target_shape(pred, target_future)
                if args.realtime_obbs:
                    safe_space = build_realtime_safe_space_from_env(
                        env=env,
                        libero_pc=libero_pc,
                        safe_space_builder=safe_space_builder,
                        camera_names=tuple(args.obb_camera_names),
                        width=args.obb_width,
                        height=args.obb_height,
                        stride=args.obb_stride,
                        max_depth=args.obb_max_depth,
                        robot_geom_ids=geom_ids_array,
                        robot_mask_dilation=args.obb_robot_mask_dilation,
                        workspace_bounds=args.obb_workspace_bounds,
                        workspace_mode=args.obb_workspace_mode,
                        workspace_margin=args.obb_workspace_margin,
                        table_z=args.obb_table_z,
                        table_slab_height=args.obb_table_slab_height,
                        table_obstacle_min_height=args.obb_table_obstacle_min_height,
                        table_obstacle_max_height=args.obb_table_obstacle_max_height,
                        component_voxel_size=args.obb_component_voxel_size,
                        component_connectivity=args.obb_component_connectivity,
                        min_component_points=args.obb_min_component_points,
                        box_margin=args.obb_box_margin,
                        box_shape=args.obb_box_shape,
                        box_orientation=args.obb_box_orientation,
                        voxel_size=args.obb_voxel_size,
                        target_geom_name_patterns=args.obb_target_geom_name_patterns,
                    )
                else:
                    safe_space = static_safe_space
                obb_corners = None if safe_space is None else safe_space["obstacle_box_corners"]
                collision_result = point_flow_obb_collision(
                    pred,
                    safe_space,
                    collision_margin=args.collision_margin,
                )
                real_collision_result = real_robot_obstacle_collision(
                    env,
                    robot_geom_ids=geom_ids_array,
                    target_geom_name_patterns=getattr(args, "obb_target_geom_name_patterns", None),
                )

                pred_samples.append(pred)
                target_samples.append(target_future)
                action_chunks.append(np.asarray(action_chunk, dtype=np.float32))
                prefix_shapes.append(tuple(np.asarray(prefix_tokens).shape))
                metrics.append(compute_point_error_metrics(pred, target_future))
                collision_results.append(collision_result)
                real_collision_results.append(real_collision_result)
                rollout_ids.append(rollout_id)
                step_ids.append(step_id)

                latest = metrics[-1]
                collision_text = (
                    f" collision=YES points={collision_result['collision_point_count']}"
                    if collision_result["collision"]
                    else " collision=NO"
                )
                real_collision_text = (
                    f" real_collision=YES contacts={real_collision_result['contact_count']}"
                    if real_collision_result["collision"]
                    else " real_collision=NO"
                )
                print(
                    f"[eval] sample={len(pred_samples)}/{args.max_samples} "
                    f"rollout={rollout_id} step={step_id} "
                    f"mean_l2={latest['mean_l2']:.6f} max_l2={latest['max_l2']:.6f}"
                    f"{collision_text}{real_collision_text}"
                )

                append_prediction_video_frame(
                    video_buffer,
                    env=env,
                    swept=swept,
                    pred_link_points=pred,
                    obb_corners=obb_corners,
                    collision_result=collision_result if safe_space is not None else None,
                    real_collision_result=real_collision_result,
                    camera_name=args.video_camera,
                    width=args.video_width,
                    height=args.video_height,
                    point_radius=args.video_point_radius,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    sample_id=len(pred_samples),
                )
                control_action = np.asarray(control_action_chunk[control_action_offset], dtype=np.float64).copy()
                action_dim = int(getattr(env, "action_dim", control_action.size))
                executed_action = control_action.copy()
                cbf_info: dict[str, object] = {
                    "triggered": False,
                    "constraint_count": 0,
                    "success": True,
                    "max_violation": 0.0,
                    "corrected_action_indices": [],
                }
                if getattr(args, "enable_cbf_qp", False):
                    cbf_trigger_source = getattr(args, "cbf_trigger_source", "predicted_point_flow")
                    cbf_correction_target = getattr(args, "cbf_correction_target", "predicted_frame_action")
                    if cbf_correction_target == "predicted_frame_action" and cbf_trigger_source == "predicted_point_flow":
                        chunk_constraints = predicted_frame_action_cbf_constraints(
                            pred,
                            current_link_points,
                            safe_space,
                            collision_margin=args.collision_margin,
                            trigger_margin=getattr(args, "cbf_trigger_margin", 0.02),
                            max_constraints=getattr(args, "cbf_max_constraints", 32),
                            include_current_points=getattr(args, "cbf_include_current_points", False),
                        )

                        def correct_chunk_action(action_index: int, nominal_chunk_action: np.ndarray, selected_constraints):
                            safe_action, action_info = filter_action_with_pointflow_cbf_qp(
                                env=env,
                                dataset_builder=dataset_builder,
                                qpos_indices=qpos_indices,
                                geom_ids=geom_ids_array,
                                low=low,
                                high=high,
                                nominal_action=nominal_chunk_action,
                                action_dim=action_dim,
                                current_link_points=current_link_points,
                                pred_link_points=pred,
                                safe_space=safe_space,
                                points_per_link=int(current_link_points.shape[1]),
                                samples_per_action=args.samples_per_action,
                                skeleton_source=args.skeleton_source,
                                collision_margin=args.collision_margin,
                                trigger_margin=getattr(args, "cbf_trigger_margin", 0.02),
                                alpha=getattr(args, "cbf_alpha", 1.0),
                                max_constraints=len(selected_constraints),
                                finite_difference_eps=getattr(args, "cbf_finite_difference_eps", 1e-4),
                                projection_iterations=getattr(args, "cbf_projection_iterations", 12),
                                action_lower=getattr(args, "cbf_action_lower", None),
                                action_upper=getattr(args, "cbf_action_upper", None),
                                fallback=getattr(args, "cbf_fallback", "zero"),
                                trigger_source=cbf_trigger_source,
                                action_space=getattr(args, "cbf_action_space", "auto"),
                                cartesian_damping=getattr(args, "cbf_cartesian_damping", 1e-4),
                                precomputed_constraints=list(selected_constraints),
                                include_current_points=getattr(args, "cbf_include_current_points", False),
                            )
                            safe_action = np.asarray(safe_action, dtype=np.float64).reshape(-1)
                            nominal_chunk_action = np.asarray(nominal_chunk_action, dtype=np.float64).reshape(-1)
                            if safe_action.shape != nominal_chunk_action.shape:
                                if int(action_dim) == 4 and safe_action.size == 4 and nominal_chunk_action.size > 4:
                                    patched = nominal_chunk_action.copy()
                                    patched[:3] = safe_action[:3]
                                    patched[-1] = safe_action[-1]
                                    safe_action = patched
                                else:
                                    raise ValueError(
                                        "CBF-corrected action shape does not match chunk row shape: "
                                        f"safe={safe_action.shape}, chunk={nominal_chunk_action.shape}"
                                    )
                            return safe_action, action_info

                        control_action_chunk, cbf_info = apply_frame_indexed_cbf_corrections(
                            control_action_chunk,
                            chunk_constraints,
                            current_action_offset=control_action_offset,
                            correct_action_fn=correct_chunk_action,
                        )
                        executed_action = np.asarray(control_action_chunk[control_action_offset], dtype=np.float64).copy()
                    else:
                        executed_action, cbf_info = filter_action_with_pointflow_cbf_qp(
                            env=env,
                            dataset_builder=dataset_builder,
                            qpos_indices=qpos_indices,
                            geom_ids=geom_ids_array,
                            low=low,
                            high=high,
                            nominal_action=control_action,
                            action_dim=action_dim,
                            current_link_points=current_link_points,
                            pred_link_points=pred,
                            safe_space=safe_space,
                            points_per_link=int(current_link_points.shape[1]),
                            samples_per_action=args.samples_per_action,
                            skeleton_source=args.skeleton_source,
                            collision_margin=args.collision_margin,
                            trigger_margin=getattr(args, "cbf_trigger_margin", 0.02),
                            alpha=getattr(args, "cbf_alpha", 1.0),
                            max_constraints=getattr(args, "cbf_max_constraints", 32),
                            finite_difference_eps=getattr(args, "cbf_finite_difference_eps", 1e-4),
                            projection_iterations=getattr(args, "cbf_projection_iterations", 12),
                            action_lower=getattr(args, "cbf_action_lower", None),
                            action_upper=getattr(args, "cbf_action_upper", None),
                            fallback=getattr(args, "cbf_fallback", "zero"),
                            trigger_source=cbf_trigger_source,
                            action_space=getattr(args, "cbf_action_space", "auto"),
                            cartesian_damping=getattr(args, "cbf_cartesian_damping", 1e-4),
                            include_current_points=getattr(args, "cbf_include_current_points", False),
                        )
                    if bool(cbf_info.get("triggered", False)):
                        print(
                            f"[cbf] constraints={int(cbf_info.get('constraint_count', 0))} "
                            f"success={bool(cbf_info.get('success', True))} "
                            f"max_violation={float(cbf_info.get('max_violation', 0.0)):.3e} "
                            f"action_space={cbf_info.get('action_space', 'unknown')} "
                            f"corrected_actions={cbf_info.get('corrected_action_indices', [])}"
                        )
                nominal_executable_action = executable_libero_action(control_action, action_dim=action_dim)
                executed_executable_action = executable_libero_action(executed_action, action_dim=action_dim)
                nominal_actions.append(np.asarray(nominal_executable_action, dtype=np.float32))
                executed_actions.append(np.asarray(executed_executable_action, dtype=np.float32))
                cbf_infos.append(cbf_info)
                obs, _reward, done, _info = env.step(executed_executable_action.tolist())
                step_id += 1
                control_action_offset += 1
                control_replan_offset += 1
    finally:
        env.close()

    if not pred_samples:
        raise RuntimeError("No evaluation samples were collected")

    stacked_metrics = stack_metric_dicts(metrics)
    result = {
        "pred_link_points": np.stack(pred_samples).astype(np.float32),
        "target_link_points": np.stack(target_samples).astype(np.float32),
        "prefix_tokens_shape": np.asarray(prefix_shapes, dtype=np.int64),
        "action_chunks": np.stack(action_chunks).astype(np.float32),
        "nominal_actions": np.stack(nominal_actions).astype(np.float32),
        "executed_actions": np.stack(executed_actions).astype(np.float32),
        "cbf_action_delta_norms": cbf_action_delta_norms(
            np.stack(nominal_actions).astype(np.float32),
            np.stack(executed_actions).astype(np.float32),
        ),
        "metrics": stacked_metrics,
        "rollout_ids": np.asarray(rollout_ids, dtype=np.int64),
        "step_ids": np.asarray(step_ids, dtype=np.int64),
        "link_names": np.asarray(link_names),
        "model_type": model_type,
        "safety_prediction_source": safety_prediction_source,
        "video_frames": video_buffer.frames,
        "video_path": video_path,
        "collision_flags": np.asarray([item["collision"] for item in collision_results], dtype=bool),
        "collision_point_counts": np.asarray(
            [item["collision_point_count"] for item in collision_results],
            dtype=np.int64,
        ),
        "real_collision_flags": np.asarray([item["collision"] for item in real_collision_results], dtype=bool),
        "real_collision_contact_counts": np.asarray(
            [item["contact_count"] for item in real_collision_results],
            dtype=np.int64,
        ),
        "cbf_triggered": np.asarray([bool(item.get("triggered", False)) for item in cbf_infos], dtype=bool),
        "cbf_success": np.asarray([bool(item.get("success", True)) for item in cbf_infos], dtype=bool),
        "cbf_constraint_counts": np.asarray(
            [int(item.get("constraint_count", 0)) for item in cbf_infos],
            dtype=np.int64,
        ),
        "cbf_max_violations": np.asarray(
            [float(item.get("max_violation", 0.0)) for item in cbf_infos],
            dtype=np.float32,
        ),
        "cbf_corrected_action_masks": cbf_corrected_action_masks(
            cbf_infos,
            action_count=int(np.stack(action_chunks).shape[1]),
        ),
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate_online(args)
    save_evaluation(
        args.output,
        pred_link_points=result["pred_link_points"],
        target_link_points=result["target_link_points"],
        prefix_tokens_shape=result["prefix_tokens_shape"],
        action_chunks=result["action_chunks"],
        metrics=result["metrics"],
        rollout_ids=result["rollout_ids"],
        step_ids=result["step_ids"],
        link_names=result["link_names"],
        coordinate_frame=COORDINATE_FRAME,
        collision_flags=result["collision_flags"],
        collision_point_counts=result["collision_point_counts"],
        real_collision_flags=result["real_collision_flags"],
        real_collision_contact_counts=result["real_collision_contact_counts"],
        nominal_actions=result["nominal_actions"],
        executed_actions=result["executed_actions"],
        cbf_action_delta_norms=result["cbf_action_delta_norms"],
        cbf_triggered=result["cbf_triggered"],
        cbf_success=result["cbf_success"],
        cbf_constraint_counts=result["cbf_constraint_counts"],
        cbf_max_violations=result["cbf_max_violations"],
        cbf_corrected_action_masks=result["cbf_corrected_action_masks"],
    )
    if not args.no_video:
        swept = load_repo_script_module("libero_joint_swept_pointcloud")
        swept.write_rgb_frames_mp4(args.video_output, result["video_frames"], fps=args.video_fps)
    metrics = result["metrics"]
    print(
        f"[done] saved evaluation: {args.output} "
        f"mean_l2={float(metrics['mean_l2']):.6f} "
        f"mean_mse={float(metrics['mean_mse']):.6f} "
        f"max_l2={float(metrics['max_l2']):.6f}"
    )
    if not args.no_video:
        print(f"[done] saved task prediction video: {args.video_output}")


if __name__ == "__main__":
    main()
