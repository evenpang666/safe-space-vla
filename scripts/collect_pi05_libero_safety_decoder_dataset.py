#!/usr/bin/env python3
"""Collect pi05_libero prefix-token to future-link-point datasets in LIBERO.

The saved ``.npz`` keeps the original absolute link-point targets and also
stores fields directly consumable by ``SafetyFlowPointModel``:

``prefix_tokens [S, N, D] + arm_points [S, K, 3] -> target_point_offsets [S, T_future, K, 3]``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENPI_ROOT_CANDIDATES = (
    REPO_ROOT / "openpi",
    REPO_ROOT / "third_party" / "openpi",
    REPO_ROOT / "thiry_party" / "openpi",
)
OPENPI_ROOT = next((path for path in OPENPI_ROOT_CANDIDATES if path.exists()), OPENPI_ROOT_CANDIDATES[0])
OPENPI_SRC = OPENPI_ROOT / "src"
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"
LIBERO_ROOT_CANDIDATES = (
    OPENPI_ROOT / "third_party" / "libero",
    REPO_ROOT / "third_party" / "LIBERO",
    REPO_ROOT / "thiry_party" / "LIBERO",
)
LIBERO_ROOT = next((path for path in LIBERO_ROOT_CANDIDATES if path.exists()), LIBERO_ROOT_CANDIDATES[0])
REPO_SCRIPT_DIR = REPO_ROOT / "scripts"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "pi05_safety_decoder" / "pi05_libero_decoder_dataset.npz"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
COORDINATE_FRAME = "mujoco_world"
OFFSET_FRAME = "mujoco_world_delta"
TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

EVAL_SCENE_WINE_BOTTLE_CATEGORY = "eval_scene_wine_bottle_obstacle"
EVAL_SCENE_WINE_BOTTLE_SCALE = 1.8


@dataclass(frozen=True)
class SceneObstacleSpec:
    kind: str = "none"
    xy: tuple[float, float] | None = None

    @property
    def enabled(self) -> bool:
        return self.kind != "none"


def _format_bddl_float(value: float) -> str:
    return f"{float(value):.6g}"


def _format_xml_floats(values: list[float]) -> str:
    return " ".join(_format_bddl_float(value) for value in values)


def _scaled_xml_vector(raw: str | None, scale: float) -> str | None:
    if raw is None:
        return None
    values = [float(item) * float(scale) for item in raw.split()]
    return _format_xml_floats(values)


def materialize_eval_scene_wine_bottle_xml(
    source_xml: Path,
    *,
    output_dir: Path,
    scale: float = EVAL_SCENE_WINE_BOTTLE_SCALE,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xml = output_dir / "eval_scene_wine_bottle_obstacle.xml"
    tree = ET.parse(source_xml)
    root = tree.getroot()
    root.set("model", "eval_scene_wine_bottle_obstacle")

    source_dir = source_xml.parent
    for asset in root.findall(".//asset/*"):
        file_attr = asset.get("file")
        if file_attr:
            asset.set("file", str((source_dir / file_attr).resolve()))
        if asset.tag == "mesh" and asset.get("scale") is not None:
            asset.set("scale", _scaled_xml_vector(asset.get("scale"), scale))

    for element in root.findall(".//geom") + root.findall(".//site"):
        for attr_name in ("pos", "size"):
            scaled = _scaled_xml_vector(element.get(attr_name), scale)
            if scaled is not None:
                element.set(attr_name, scaled)

    tree.write(output_xml, encoding="unicode")
    return output_xml


def register_eval_scene_obstacle_objects(*, scale: float = EVAL_SCENE_WINE_BOTTLE_SCALE) -> None:
    ensure_third_party_paths()
    from libero.libero.envs.base_object import OBJECTS_DICT, register_object
    from robosuite.models.objects import MujocoXMLObject

    if EVAL_SCENE_WINE_BOTTLE_CATEGORY in OBJECTS_DICT:
        return

    source_xml = (
        LIBERO_ROOT
        / "libero"
        / "libero"
        / "assets"
        / "turbosquid_objects"
        / "wine_bottle"
        / "wine_bottle.xml"
    )
    output_xml = materialize_eval_scene_wine_bottle_xml(
        source_xml,
        output_dir=Path(os.environ.get("TMPDIR", "/tmp")) / "safety_module_libero_scene_obstacles" / "assets",
        scale=scale,
    )

    class EvalSceneWineBottleObstacle(MujocoXMLObject):
        def __init__(self, name=EVAL_SCENE_WINE_BOTTLE_CATEGORY, joints=None):
            super().__init__(
                str(output_xml),
                name=name,
                joints=joints,
                obj_type="all",
                duplicate_collision_geoms=False,
            )
            self.category_name = EVAL_SCENE_WINE_BOTTLE_CATEGORY
            self.rotation = (0, 0)
            self.rotation_axis = "z"
            self.object_properties = {"vis_site_names": {}}

    EvalSceneWineBottleObstacle.__name__ = "EvalSceneWineBottleObstacle"
    register_object(EvalSceneWineBottleObstacle)


def _find_bddl_section_span(text: str, section_name: str) -> tuple[int, int]:
    match = re.search(r"\(:" + re.escape(section_name) + r"\b", text)
    if match is None:
        raise ValueError(f"BDDL text is missing :{section_name} section")
    start = int(match.start())
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, idx + 1
    raise ValueError(f"BDDL :{section_name} section is not balanced")


def _insert_before_bddl_section_close(text: str, section_name: str, insertion: str) -> str:
    _start, end = _find_bddl_section_span(text, section_name)
    return text[: end - 1] + insertion + text[end - 1 :]


def patch_bddl_with_scene_obstacle(
    bddl_text: str,
    obstacle: SceneObstacleSpec,
    *,
    region_half_extent: float = 0.01,
) -> str:
    """Add an optional scene obstacle to a LIBERO BDDL problem.

    The inserted object is intentionally not added to ``:obj_of_interest`` so
    task success predicates keep their original semantics.
    """
    if not obstacle.enabled:
        return bddl_text
    if obstacle.kind != "wine_bottle":
        raise ValueError(f"Unsupported scene obstacle kind: {obstacle.kind!r}")
    if "eval_scene_obstacle_1" in bddl_text:
        return bddl_text
    xy = obstacle.xy if obstacle.xy is not None else (0.0, 0.0)
    x, y = float(xy[0]), float(xy[1])
    half = float(region_half_extent)
    x0, y0 = x - half, y - half
    x1, y1 = x + half, y + half

    region = (
        "\n"
        "      (eval_scene_obstacle_region\n"
        "          (:target main_table)\n"
        "          (:ranges (\n"
        f"              ({_format_bddl_float(x0)} {_format_bddl_float(y0)} "
        f"{_format_bddl_float(x1)} {_format_bddl_float(y1)})\n"
        "            )\n"
        "          )\n"
        "      )"
    )
    fixture = f"\n    eval_scene_obstacle_1 - {EVAL_SCENE_WINE_BOTTLE_CATEGORY}"
    init = "\n    (On eval_scene_obstacle_1 main_table_eval_scene_obstacle_region)"

    patched = _insert_before_bddl_section_close(bddl_text, "regions", region)
    patched = _insert_before_bddl_section_close(patched, "fixtures", fixture)
    return _insert_before_bddl_section_close(patched, "init", init)


def materialize_scene_obstacle_bddl(
    task_bddl_file: Path,
    obstacle: SceneObstacleSpec | None,
    *,
    output_dir: Path | None = None,
) -> Path:
    if obstacle is None or not obstacle.enabled:
        return task_bddl_file
    output_dir = output_dir or (Path(os.environ.get("TMPDIR", "/tmp")) / "safety_module_libero_scene_obstacles")
    output_dir.mkdir(parents=True, exist_ok=True)
    xy_suffix = "center" if obstacle.xy is None else f"{_format_bddl_float(obstacle.xy[0])}_{_format_bddl_float(obstacle.xy[1])}"
    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{obstacle.kind}_{xy_suffix}")
    output_path = output_dir / f"{task_bddl_file.stem}_{safe_suffix}.bddl"
    patched = patch_bddl_with_scene_obstacle(task_bddl_file.read_text(), obstacle)
    output_path.write_text(patched)
    return output_path


def _joint_state_widths(joint_type: int) -> tuple[int, int]:
    # MuJoCo joint type ids used by robosuite bindings: free, ball, slide, hinge.
    if int(joint_type) == 0:
        return 7, 6
    if int(joint_type) == 1:
        return 4, 3
    return 1, 1


def _scene_obstacle_joint_spans(model) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    qpos_spans: list[tuple[int, int]] = []
    qvel_spans: list[tuple[int, int]] = []
    for joint_id in range(int(getattr(model, "njnt", 0))):
        joint_name = model.joint_id2name(joint_id)
        if not joint_name or "eval_scene_obstacle_" not in str(joint_name):
            continue
        q_width, v_width = _joint_state_widths(int(model.jnt_type[joint_id]))
        q_start = int(model.jnt_qposadr[joint_id])
        v_start = int(model.jnt_dofadr[joint_id])
        qpos_spans.append((q_start, q_start + q_width))
        qvel_spans.append((v_start, v_start + v_width))
    return sorted(qpos_spans), sorted(qvel_spans)


def _copy_old_state_around_spans(
    *,
    adapted: np.ndarray,
    target_offset: int,
    target_size: int,
    old_values: np.ndarray,
    preserved_spans: list[tuple[int, int]],
) -> None:
    old_offset = 0
    cursor = 0
    for span_start, span_end in preserved_spans:
        if span_start < cursor or span_end > target_size:
            raise ValueError(f"invalid or overlapping scene obstacle state span {(span_start, span_end)}")
        segment_len = span_start - cursor
        adapted[target_offset + cursor : target_offset + span_start] = old_values[old_offset : old_offset + segment_len]
        old_offset += segment_len
        cursor = span_end
    segment_len = target_size - cursor
    adapted[target_offset + cursor : target_offset + target_size] = old_values[old_offset : old_offset + segment_len]
    old_offset += segment_len
    if old_offset != old_values.size:
        raise ValueError(
            f"scene obstacle state copy consumed {old_offset} old values, expected {old_values.size}"
        )


def adapt_init_state_for_scene_obstacle(
    init_state: np.ndarray,
    env,
    obstacle: SceneObstacleSpec | None,
) -> np.ndarray:
    """Pad an original LIBERO init state after adding free-joint obstacles.

    Benchmark init states are saved for the original BDDL model. Adding one
    free-joint scene object appends 7 qpos and 6 qvel entries, so the old flat
    state no longer matches the patched model. We copy the original model state
    prefix and keep the new obstacle qpos/qvel from the just-reset environment.
    """
    state = np.asarray(init_state)
    if obstacle is None or not obstacle.enabled:
        return state

    if hasattr(env, "get_sim_state"):
        current = np.asarray(env.get_sim_state(), dtype=state.dtype)
    else:
        current = np.asarray(env.sim.get_state().flatten(), dtype=state.dtype)
    if state.size == current.size:
        return state
    if state.size > current.size:
        raise ValueError(
            f"scene obstacle init state adaptation expected old state <= current state, "
            f"got old={state.size}, current={current.size}"
        )

    model = env.sim.model
    qpos_spans, qvel_spans = _scene_obstacle_joint_spans(model)
    if not qpos_spans or not qvel_spans:
        raise ValueError("scene obstacle is enabled, but no eval_scene_obstacle joints were found in the model")

    extra = int(current.size - state.size)
    preserved_qpos = sum(span_end - span_start for span_start, span_end in qpos_spans)
    preserved_qvel = sum(span_end - span_start for span_start, span_end in qvel_spans)
    if extra != preserved_qpos + preserved_qvel:
        raise ValueError(
            f"scene obstacle init state size delta does not match obstacle joints: "
            f"old={state.size}, current={current.size}, delta={extra}, "
            f"obstacle_qpos={preserved_qpos}, obstacle_qvel={preserved_qvel}"
        )

    new_nq = int(model.nq)
    new_nv = int(model.nv)
    old_nq = new_nq - preserved_qpos
    old_nv = new_nv - preserved_qvel
    expected_old_size = 1 + old_nq + old_nv
    if state.size != expected_old_size:
        raise ValueError(
            f"cannot infer scene obstacle state layout: old={state.size}, "
            f"expected={expected_old_size}, current={current.size}"
        )

    adapted = current.copy()
    adapted[0] = state[0]
    old_qvel_start = 1 + old_nq
    new_qvel_start = 1 + new_nq
    _copy_old_state_around_spans(
        adapted=adapted,
        target_offset=1,
        target_size=new_nq,
        old_values=state[1 : 1 + old_nq],
        preserved_spans=qpos_spans,
    )
    _copy_old_state_around_spans(
        adapted=adapted,
        target_offset=new_qvel_start,
        target_size=new_nv,
        old_values=state[old_qvel_start : old_qvel_start + old_nv],
        preserved_spans=qvel_spans,
    )
    return adapted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", default="pi05_libero", help="OpenPI training config name.")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT, help="OpenPI policy checkpoint directory.")
    parser.add_argument("--task-suite", default="libero_spatial", choices=sorted(TASK_SUITE_MAX_STEPS))
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Task ids to collect into one dataset. Use 'all' to collect every task in the suite.",
    )
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=512, help="Maximum replan samples to collect.")
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=512,
        help="Maximum replan samples to collect for each task. Defaults to --max-samples.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Rollout step cap after settling.")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="No-op steps before policy control.")
    parser.add_argument("--replan-steps", type=int, default=10, help="Executed steps per predicted action chunk.")
    parser.add_argument("--resize-size", type=int, default=224, help="OpenPI image input size.")
    parser.add_argument("--env-resolution", type=int, default=256, help="LIBERO render resolution before resizing.")
    parser.add_argument("--points-per-link", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default="egl")
    parser.add_argument("--pytorch-device", default=None, help="Device for PyTorch OpenPI checkpoints.")
    parser.add_argument(
        "--policy-server-host",
        default=None,
        help="Optional websocket policy server host. If set, actions and prefix_tokens are read from the server.",
    )
    parser.add_argument("--policy-server-port", type=int, default=8000)
    return parser.parse_args()


def ensure_third_party_paths() -> None:
    inner_libero_parents = [libero_root / "libero" for libero_root in LIBERO_ROOT_CANDIDATES]
    sys.path[:] = [
        path
        for path in sys.path
        if not any(Path(path).resolve() == inner_libero_parent.resolve() for inner_libero_parent in inner_libero_parents)
    ]
    for path in (OPENPI_SRC, OPENPI_CLIENT_SRC, LIBERO_ROOT, REPO_ROOT):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


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


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(quat[3]) / den)


def resize_uint8_image(image: np.ndarray, resize_size: int) -> np.ndarray:
    ensure_third_party_paths()
    from openpi_client import image_tools

    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))


def build_libero_policy_input(
    obs: dict,
    *,
    prompt: str,
    resize_size: int,
    image_resizer: Callable[[np.ndarray, int], np.ndarray] | None = None,
) -> dict:
    image_resizer = image_resizer or resize_uint8_image
    base_image = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1])
    wrist_image = np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1])

    return {
        "observation/image": image_resizer(base_image, resize_size),
        "observation/wrist_image": image_resizer(wrist_image, resize_size),
        "observation/state": np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
            )
        ).astype(np.float32),
        "prompt": str(prompt),
    }


def load_openpi_policy(
    *,
    policy_config: str,
    checkpoint_dir: str,
    default_prompt: str | None,
    pytorch_device: str | None,
):
    ensure_third_party_paths()
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    return _policy_config.create_trained_policy(
        _config.get_config(policy_config),
        checkpoint_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )


def load_remote_policy(*, host: str, port: int):
    ensure_third_party_paths()
    from openpi_client import websocket_client_policy as _websocket_client_policy

    return _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)


def extract_policy_prefix_tokens(policy, element: dict) -> np.ndarray:
    """Run only the PI05 prefix encoder and return one sample of prefix embeddings."""
    ensure_third_party_paths()
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model

    inputs = jax.tree.map(lambda x: x, element)
    inputs = policy._input_transform(inputs)

    if getattr(policy, "_is_pytorch_model", False):
        import torch

        device = getattr(policy, "_pytorch_device", "cpu")
        tensor_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device)[None, ...], inputs)
        observation = _model.Observation.from_dict(tensor_inputs)
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, _state = policy._model._preprocess_observation(
                observation, train=False
            )
            prefix_tokens, _prefix_pad_masks, _prefix_att_masks = policy._model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        return prefix_tokens[0].detach().to(dtype=torch.float32).cpu().numpy()

    batch_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    observation = _model.Observation.from_dict(batch_inputs)
    prefix_tokens, _prefix_mask, _prefix_ar_mask = policy._model.embed_prefix(observation)
    return np.asarray(prefix_tokens[0], dtype=np.float32)


def query_policy_action_and_prefix(
    policy,
    element: dict,
    *,
    remote_prefix_tokens: bool,
    local_prefix_extractor: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result = policy.infer(element)
    if "actions" not in result:
        raise KeyError("Policy response must contain 'actions'")
    action_chunk = np.asarray(result["actions"], dtype=np.float32)

    if remote_prefix_tokens:
        if "prefix_tokens" not in result:
            raise KeyError(
                "Remote policy response must contain 'prefix_tokens'. "
                "Start scripts/serve_pi05_prefix_policy.py instead of OpenPI's default serve_policy.py."
            )
        prefix_tokens = np.asarray(result["prefix_tokens"], dtype=np.float32)
    else:
        extractor = local_prefix_extractor or extract_policy_prefix_tokens
        prefix_tokens = np.asarray(extractor(policy, element), dtype=np.float32)

    return action_chunk, prefix_tokens


def derive_flow_point_targets(target_link_points: np.ndarray) -> dict[str, np.ndarray]:
    target_link_points = np.asarray(target_link_points, dtype=np.float32)
    if target_link_points.ndim != 4 or target_link_points.shape[-1] != 3:
        raise ValueError(f"target_link_points must have shape (T, L, P, 3), got {target_link_points.shape}")
    if target_link_points.shape[0] < 2:
        raise ValueError("target_link_points must include the current step and at least one future step")

    current_link_points = target_link_points[0].astype(np.float32)  # [L, P, 3]
    future_link_points = target_link_points[1:].astype(np.float32)  # [T_future, L, P, 3]
    future_link_offsets = future_link_points - current_link_points[None, :, :, :]  # [T_future, L, P, 3]
    arm_points = current_link_points.reshape(-1, 3).astype(np.float32)  # [K, 3]
    target_point_offsets = future_link_offsets.reshape(future_link_offsets.shape[0], -1, 3).astype(np.float32)
    return {
        "current_link_points": current_link_points,
        "future_link_offsets": future_link_offsets.astype(np.float32),
        "arm_points": arm_points,
        "target_point_offsets": target_point_offsets,
    }


@dataclass
class CollectedSampleBuffer:
    prefix_tokens: list[np.ndarray] = field(default_factory=list)
    action_chunks: list[np.ndarray] = field(default_factory=list)
    start_joint_vectors: list[np.ndarray] = field(default_factory=list)
    target_link_points: list[np.ndarray] = field(default_factory=list)
    current_link_points: list[np.ndarray] = field(default_factory=list)
    future_link_offsets: list[np.ndarray] = field(default_factory=list)
    arm_points: list[np.ndarray] = field(default_factory=list)
    target_point_offsets: list[np.ndarray] = field(default_factory=list)
    task_ids: list[int] = field(default_factory=list)
    rollout_ids: list[int] = field(default_factory=list)
    step_ids: list[int] = field(default_factory=list)
    _shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def append(
        self,
        *,
        prefix_tokens: np.ndarray,
        action_chunk: np.ndarray,
        start_joint_vector: np.ndarray,
        target_link_points: np.ndarray,
        current_link_points: np.ndarray | None = None,
        future_link_offsets: np.ndarray | None = None,
        arm_points: np.ndarray | None = None,
        target_point_offsets: np.ndarray | None = None,
        task_id: int,
        rollout_id: int,
        step_id: int,
    ) -> None:
        prefix_tokens = np.asarray(prefix_tokens, dtype=np.float32)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        start_joint_vector = np.asarray(start_joint_vector, dtype=np.float32).reshape(-1)
        target_link_points = np.asarray(target_link_points, dtype=np.float32)
        derived = derive_flow_point_targets(target_link_points)
        current_link_points = (
            derived["current_link_points"]
            if current_link_points is None
            else np.asarray(current_link_points, dtype=np.float32)
        )
        future_link_offsets = (
            derived["future_link_offsets"]
            if future_link_offsets is None
            else np.asarray(future_link_offsets, dtype=np.float32)
        )
        arm_points = derived["arm_points"] if arm_points is None else np.asarray(arm_points, dtype=np.float32)
        target_point_offsets = (
            derived["target_point_offsets"]
            if target_point_offsets is None
            else np.asarray(target_point_offsets, dtype=np.float32)
        )

        if prefix_tokens.ndim != 2:
            raise ValueError(f"prefix_tokens must have shape (N, D), got {prefix_tokens.shape}")
        if action_chunk.ndim != 2:
            raise ValueError(f"action_chunk must have shape (T, A), got {action_chunk.shape}")
        if target_link_points.ndim != 4 or target_link_points.shape[-1] != 3:
            raise ValueError(f"target_link_points must have shape (T, L, P, 3), got {target_link_points.shape}")
        if current_link_points.shape != target_link_points.shape[1:]:
            raise ValueError(
                f"current_link_points must have shape {target_link_points.shape[1:]}, "
                f"got {current_link_points.shape}"
            )
        expected_future_shape = (target_link_points.shape[0] - 1,) + target_link_points.shape[1:]
        if future_link_offsets.shape != expected_future_shape:
            raise ValueError(
                f"future_link_offsets must have shape {expected_future_shape}, got {future_link_offsets.shape}"
            )
        if arm_points.ndim != 2 or arm_points.shape[-1] != 3:
            raise ValueError(f"arm_points must have shape (K, 3), got {arm_points.shape}")
        if target_point_offsets.ndim != 3 or target_point_offsets.shape[-1] != 3:
            raise ValueError(f"target_point_offsets must have shape (T_future, K, 3), got {target_point_offsets.shape}")

        self._check_shape("prefix_tokens", prefix_tokens.shape)
        self._check_shape("action_chunk", action_chunk.shape)
        self._check_shape("start_joint_vector", start_joint_vector.shape)
        self._check_shape("target_link_points", target_link_points.shape)
        self._check_shape("current_link_points", current_link_points.shape)
        self._check_shape("future_link_offsets", future_link_offsets.shape)
        self._check_shape("arm_points", arm_points.shape)
        self._check_shape("target_point_offsets", target_point_offsets.shape)

        self.prefix_tokens.append(prefix_tokens)
        self.action_chunks.append(action_chunk)
        self.start_joint_vectors.append(start_joint_vector)
        self.target_link_points.append(target_link_points)
        self.current_link_points.append(current_link_points)
        self.future_link_offsets.append(future_link_offsets)
        self.arm_points.append(arm_points)
        self.target_point_offsets.append(target_point_offsets)
        self.task_ids.append(int(task_id))
        self.rollout_ids.append(int(rollout_id))
        self.step_ids.append(int(step_id))

    def _check_shape(self, name: str, shape: tuple[int, ...]) -> None:
        expected = self._shapes.setdefault(name, shape)
        if expected != shape:
            raise ValueError(f"{name} shape changed from {expected} to {shape}")

    def __len__(self) -> int:
        return len(self.prefix_tokens)


@dataclass
class ReplanSampleRecord:
    prefix_tokens: np.ndarray
    action_chunk: np.ndarray
    start_joint_vector: np.ndarray
    task_id: int
    rollout_id: int
    step_id: int


def surface_trajectory_target(surface_frames: np.ndarray, *, start_step: int, horizon: int) -> np.ndarray:
    """Slice [current + future] surface points from a recorded executed trajectory.

    surface_frames: [T_recorded, L, P, 3]
    return: [horizon + 1, L, P, 3]
    """
    surface_frames = np.asarray(surface_frames, dtype=np.float32)
    if surface_frames.ndim != 4 or surface_frames.shape[-1] != 3:
        raise ValueError(f"surface_frames must have shape (T, L, P, 3), got {surface_frames.shape}")
    if surface_frames.shape[0] == 0:
        raise ValueError("surface_frames must contain at least one frame")
    if start_step < 0:
        raise ValueError("start_step must be >= 0")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    end_step = start_step + horizon
    if end_step >= surface_frames.shape[0]:
        raise ValueError(
            f"surface trajectory does not contain a complete future horizon: "
            f"start_step={start_step}, horizon={horizon}, frame_count={surface_frames.shape[0]}"
        )
    return surface_frames[start_step : end_step + 1].astype(np.float32)


def append_surface_trajectory_samples(
    buffer: CollectedSampleBuffer,
    *,
    records: list[ReplanSampleRecord],
    surface_frames: np.ndarray,
    link_names: np.ndarray,
    max_samples: int,
) -> int:
    appended = 0
    for record in records:
        if len(buffer) >= max_samples:
            break
        action_chunk = np.asarray(record.action_chunk, dtype=np.float32)
        if record.step_id + action_chunk.shape[0] >= np.asarray(surface_frames).shape[0]:
            continue
        target_link_points = surface_trajectory_target(
            surface_frames,
            start_step=record.step_id,
            horizon=action_chunk.shape[0],
        )
        buffer.append(
            prefix_tokens=record.prefix_tokens,
            action_chunk=action_chunk,
            start_joint_vector=record.start_joint_vector,
            target_link_points=target_link_points,
            task_id=record.task_id,
            rollout_id=record.rollout_id,
            step_id=record.step_id,
        )
        appended += 1
        print(
            f"[collect] sample={len(buffer)}/{max_samples} "
            f"rollout={record.rollout_id} step={record.step_id} "
            f"prefix={np.asarray(record.prefix_tokens).shape} target={target_link_points.shape}"
        )
    return appended


def save_collected_dataset(
    output: Path,
    *,
    buffer: CollectedSampleBuffer,
    link_names: np.ndarray,
    task_suite: str,
    points_per_link: int,
    samples_per_action: int,
    policy_config: str,
    checkpoint_dir: str,
    skeleton_source: str = "surface",
    target_source: str = "rollout_surface",
) -> None:
    if len(buffer) == 0:
        raise ValueError("No samples were collected; refusing to write an empty dataset")

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
        task_ids=np.asarray(buffer.task_ids, dtype=np.int64),
        rollout_ids=np.asarray(buffer.rollout_ids, dtype=np.int64),
        step_ids=np.asarray(buffer.step_ids, dtype=np.int64),
        link_names=np.asarray(link_names),
        coordinate_frame=np.asarray(COORDINATE_FRAME),
        target_link_points_frame=np.asarray(COORDINATE_FRAME),
        current_link_points_frame=np.asarray(COORDINATE_FRAME),
        future_link_offsets_frame=np.asarray(OFFSET_FRAME),
        arm_points_frame=np.asarray(COORDINATE_FRAME),
        target_point_offsets_frame=np.asarray(OFFSET_FRAME),
        task_suite=np.asarray(task_suite),
        points_per_link=np.asarray(points_per_link),
        samples_per_action=np.asarray(samples_per_action),
        skeleton_source=np.asarray(skeleton_source),
        target_source=np.asarray(target_source),
        policy_config=np.asarray(policy_config),
        checkpoint_dir=np.asarray(checkpoint_dir),
    )


def compute_fk_target_preserving_sim_state(env, target_builder: Callable[[], tuple[np.ndarray, np.ndarray]]):
    qpos = np.asarray(env.sim.data.qpos).copy()
    qvel = np.asarray(env.sim.data.qvel).copy()
    try:
        return target_builder()
    finally:
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()


def _restore_optional_array(target, value: np.ndarray | None) -> None:
    if value is not None and target is not None:
        target[:] = value


def snapshot_sim_state(sim) -> dict[str, object]:
    state = sim.get_state() if hasattr(sim, "get_state") else None
    data = sim.data
    return {
        "state": state,
        "qpos": np.asarray(data.qpos).copy() if hasattr(data, "qpos") else None,
        "qvel": np.asarray(data.qvel).copy() if hasattr(data, "qvel") else None,
        "ctrl": np.asarray(data.ctrl).copy() if hasattr(data, "ctrl") else None,
        "mocap_pos": np.asarray(data.mocap_pos).copy() if hasattr(data, "mocap_pos") else None,
        "mocap_quat": np.asarray(data.mocap_quat).copy() if hasattr(data, "mocap_quat") else None,
    }


def restore_sim_state(sim, snapshot: dict[str, object]) -> None:
    if snapshot["state"] is not None and hasattr(sim, "set_state"):
        sim.set_state(snapshot["state"])
    data = sim.data
    _restore_optional_array(getattr(data, "qpos", None), snapshot["qpos"])
    _restore_optional_array(getattr(data, "qvel", None), snapshot["qvel"])
    _restore_optional_array(getattr(data, "ctrl", None), snapshot["ctrl"])
    _restore_optional_array(getattr(data, "mocap_pos", None), snapshot["mocap_pos"])
    _restore_optional_array(getattr(data, "mocap_quat", None), snapshot["mocap_quat"])
    sim.forward()


def collect_rollout_surface_target(
    env,
    action_chunk: np.ndarray,
    *,
    surface_snapshot: Callable[[], np.ndarray],
    link_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"action_chunk must have shape (T, A), got {action_chunk.shape}")

    target_frames = [np.asarray(surface_snapshot(), dtype=np.float32)]
    done = False
    for action in action_chunk:
        if not done:
            env_action = np.asarray(action, dtype=np.float64)
            action_dim = int(getattr(env, "action_dim", env_action.size))
            _obs, _reward, done, _info = env.step(env_action[:action_dim].tolist())
        target_frames.append(np.asarray(surface_snapshot(), dtype=np.float32))
    return np.stack(target_frames).astype(np.float32), np.asarray(link_names)


def compute_rollout_surface_target_preserving_sim_state(
    env,
    action_chunk: np.ndarray,
    *,
    surface_snapshot: Callable[[], np.ndarray],
    link_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    snapshot = snapshot_sim_state(env.sim)
    try:
        return collect_rollout_surface_target(
            env,
            action_chunk,
            surface_snapshot=surface_snapshot,
            link_names=link_names,
        )
    finally:
        restore_sim_state(env.sim, snapshot)


def robot_geom_ids_array(geom_ids) -> np.ndarray:
    if isinstance(geom_ids, set):
        geom_ids = sorted(geom_ids)
    return np.asarray(geom_ids, dtype=np.int64)


def make_dummy_action(env) -> np.ndarray:
    action_dim = int(getattr(env, "action_dim", 7))
    action = np.zeros(action_dim, dtype=np.float64)
    if action_dim > 0:
        action[min(6, action_dim - 1)] = -1.0
    return action


def create_libero_task_suite(task_suite_name: str):
    ensure_third_party_paths()
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    return benchmark_dict[task_suite_name]()


def create_libero_env(
    task,
    *,
    resolution: int,
    seed: int,
    scene_obstacle: SceneObstacleSpec | None = None,
):
    ensure_third_party_paths()
    if scene_obstacle is not None and scene_obstacle.enabled:
        register_eval_scene_obstacle_objects()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    task_bddl_file = materialize_scene_obstacle_bddl(task_bddl_file, scene_obstacle)
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, str(task.language)


def default_max_steps(task_suite: str) -> int:
    try:
        return TASK_SUITE_MAX_STEPS[task_suite]
    except KeyError as exc:
        raise ValueError(f"Unknown task suite: {task_suite}") from exc


def resolve_task_ids(*, task_id: int, task_ids: list[str] | tuple[str, ...] | None, n_tasks: int) -> list[int]:
    n_tasks = int(n_tasks)
    if n_tasks <= 0:
        raise ValueError(f"task suite must contain at least one task, got {n_tasks}")
    if task_ids is None:
        resolved = [int(task_id)]
    else:
        raw_ids = [str(item) for item in task_ids]
        lowered = [item.lower() for item in raw_ids]
        if "all" in lowered:
            if len(raw_ids) != 1:
                raise ValueError("--task-ids all cannot be combined with explicit task ids")
            resolved = list(range(n_tasks))
        else:
            try:
                resolved = [int(item) for item in raw_ids]
            except ValueError as exc:
                raise ValueError("--task-ids entries must be integers or 'all'") from exc
    for item in resolved:
        if not 0 <= int(item) < n_tasks:
            raise ValueError(f"task id {item} must be in [0, {n_tasks - 1}]")
    if len(set(resolved)) != len(resolved):
        raise ValueError("--task-ids must not contain duplicates")
    return resolved


def resolve_max_samples_per_task(*, max_samples: int, max_samples_per_task: int | None) -> int:
    if max_samples_per_task is None:
        return int(max_samples)
    if int(max_samples_per_task) <= 0:
        raise ValueError("--max-samples-per-task must be > 0")
    return int(max_samples_per_task)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_rollouts <= 0:
        raise ValueError("--num-rollouts must be > 0")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")
    if args.max_samples_per_task is not None and args.max_samples_per_task <= 0:
        raise ValueError("--max-samples-per-task must be > 0")
    if args.replan_steps <= 0:
        raise ValueError("--replan-steps must be > 0")
    if args.points_per_link < 2:
        raise ValueError("--points-per-link must be >= 2")


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_third_party_paths()
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    dataset_builder = load_repo_script_module("build_pi05_safety_decoder_dataset")

    swept = dataset_builder.import_script_module("libero_joint_swept_pointcloud")
    libero_pc = dataset_builder.import_script_module("libero_reconstruct_pointcloud")
    swept.load_runtime_dependencies()

    np.random.seed(args.seed)
    task_suite = create_libero_task_suite(args.task_suite)
    task_ids = resolve_task_ids(task_id=args.task_id, task_ids=args.task_ids, n_tasks=task_suite.n_tasks)
    max_samples_per_task = resolve_max_samples_per_task(
        max_samples=args.max_samples,
        max_samples_per_task=args.max_samples_per_task,
    )
    max_steps = args.max_steps if args.max_steps is not None else default_max_steps(args.task_suite)
    remote_prefix_tokens = args.policy_server_host is not None
    if remote_prefix_tokens:
        policy = load_remote_policy(host=args.policy_server_host, port=args.policy_server_port)
    else:
        policy = load_openpi_policy(
            policy_config=args.policy_config,
            checkpoint_dir=args.checkpoint_dir,
            default_prompt=None,
            pytorch_device=args.pytorch_device,
        )

    buffer = CollectedSampleBuffer()
    link_names = np.asarray([])
    for task_id in task_ids:
        task_start_sample_count = len(buffer)
        task_sample_limit = task_start_sample_count + max_samples_per_task
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = create_libero_env(task, resolution=args.env_resolution, seed=args.seed)
        print(
            f"[task] collecting task_id={task_id} "
            f"target_samples={max_samples_per_task} prompt={task_description!r}"
        )
        try:
            qpos_indices = swept.get_arm_qpos_indices(env)
            geom_ids = libero_pc.find_robot_geoms(env)
            geom_ids_array = robot_geom_ids_array(geom_ids)
            dummy_action = make_dummy_action(env)
            local_surface_points, surface_template_geom_ids, surface_link_names = (
                dataset_builder.build_link_surface_template(
                    env.sim.model,
                    geom_ids_array,
                    args.points_per_link,
                    np.random.default_rng(0),
                )
            )

            def surface_snapshot():
                return dataset_builder.transform_link_surface_template(
                    env.sim,
                    local_surface_points,
                    surface_template_geom_ids,
                )

            for rollout_id in range(args.num_rollouts):
                if len(buffer) >= task_sample_limit:
                    break
                env.reset()
                init_state = initial_states[rollout_id % len(initial_states)]
                obs = env.set_init_state(init_state)
                for _ in range(args.num_steps_wait):
                    obs, _reward, done, _info = env.step(dummy_action)
                    if done:
                        break

                step_id = 0
                done = False
                rollout_records: list[ReplanSampleRecord] = []
                surface_frames: list[np.ndarray] = []

                control_action_chunk = None
                control_action_offset = 0
                control_replan_offset = 0
                while not done and step_id < max_steps:
                    surface_frames.append(np.asarray(surface_snapshot(), dtype=np.float32))
                    element = build_libero_policy_input(obs, prompt=task_description, resize_size=args.resize_size)
                    need_control_query = (
                        control_action_chunk is None
                        or control_action_offset >= len(control_action_chunk)
                        or control_replan_offset >= args.replan_steps
                    )
                    if need_control_query:
                        action_chunk, prefix_tokens = query_policy_action_and_prefix(
                            policy,
                            element,
                            remote_prefix_tokens=remote_prefix_tokens,
                        )
                        control_action_chunk = action_chunk
                        control_action_offset = 0
                        control_replan_offset = 0
                    elif len(buffer) + len(rollout_records) < task_sample_limit:
                        action_chunk, prefix_tokens = query_policy_action_and_prefix(
                            policy,
                            element,
                            remote_prefix_tokens=remote_prefix_tokens,
                        )
                    else:
                        action_chunk = control_action_chunk
                        prefix_tokens = None
                    start_joint_vector = np.asarray(env.sim.data.qpos[qpos_indices], dtype=np.float32)

                    if prefix_tokens is not None and len(buffer) + len(rollout_records) < task_sample_limit:
                        rollout_records.append(
                            ReplanSampleRecord(
                                prefix_tokens=prefix_tokens,
                                action_chunk=action_chunk,
                                start_joint_vector=start_joint_vector,
                                task_id=task_id,
                                rollout_id=rollout_id,
                                step_id=step_id,
                            )
                        )

                    actions_to_execute = [control_action_chunk[control_action_offset]]
                    for action in actions_to_execute:
                        env_action = np.asarray(action, dtype=np.float64)
                        action_dim = int(getattr(env, "action_dim", env_action.size))
                        obs, _reward, done, _info = env.step(env_action[:action_dim].tolist())
                        step_id += 1
                        control_action_offset += 1
                        control_replan_offset += 1
                        if done or step_id >= max_steps:
                            break

                link_names = surface_link_names
                if surface_frames:
                    append_surface_trajectory_samples(
                        buffer,
                        records=rollout_records,
                        surface_frames=np.stack(surface_frames).astype(np.float32),
                        link_names=link_names,
                        max_samples=task_sample_limit,
                    )
            print(f"[task] task_id={task_id} saved_samples={len(buffer) - task_start_sample_count}")
        finally:
            env.close()

    save_collected_dataset(
        args.output,
        buffer=buffer,
        link_names=link_names,
        task_suite=args.task_suite,
        points_per_link=args.points_per_link,
        samples_per_action=1,
        policy_config=args.policy_config,
        checkpoint_dir=args.checkpoint_dir,
        skeleton_source="surface",
        target_source="rollout_surface",
    )
    print(f"[done] saved {len(buffer)} samples from {len(task_ids)} tasks to {args.output}")


if __name__ == "__main__":
    main()
