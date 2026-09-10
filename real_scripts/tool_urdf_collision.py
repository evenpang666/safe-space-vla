"""Small fixed-joint collision-URDF sampler for live RGB-D robot masking.

Only collision primitives are intentionally supported: a deployment should
fail loudly rather than silently ignoring a mesh or moving joint.  This keeps
the live model deterministic and independent of ROS/xacro at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), (-sp, cp * sr, cp * cr))
    return result


def _origin(node: ET.Element | None) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    if node is None:
        return value
    value[:3, 3] = [float(item) for item in node.get("xyz", "0 0 0").split()]
    rpy = [float(item) for item in node.get("rpy", "0 0 0").split()]
    return value @ _rpy(*rpy)


@dataclass(frozen=True)
class CollisionPrimitive:
    transform: np.ndarray
    kind: str
    values: tuple[float, ...]


def load_fixed_collision_urdf(path: Path) -> list[CollisionPrimitive]:
    """Load fixed collision primitive poses in the URDF root-link frame."""
    root = ET.parse(path).getroot()
    links = {node.get("name"): node for node in root.findall("link")}
    if not links:
        raise ValueError(f"{path} has no links")
    children: set[str] = set(); adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        kind = joint.get("type")
        if kind != "fixed":
            raise ValueError(f"{path}: joint {joint.get('name')} is {kind!r}; live collision URDF must use fixed joints")
        parent = joint.find("parent"); child = joint.find("child")
        if parent is None or child is None or parent.get("link") not in links or child.get("link") not in links:
            raise ValueError(f"{path}: joint {joint.get('name')} references a missing link")
        parent_name, child_name = parent.get("link"), child.get("link")
        children.add(child_name); adjacency.setdefault(parent_name, []).append((child_name, _origin(joint.find("origin"))))
    roots = set(links) - children
    if len(roots) != 1:
        raise ValueError(f"{path}: expected one root link, found {sorted(roots)}")
    poses = {next(iter(roots)): np.eye(4, dtype=np.float64)}; pending = list(poses)
    while pending:
        parent = pending.pop()
        for child, transform in adjacency.get(parent, []):
            if child in poses:
                raise ValueError(f"{path}: link {child} has multiple parents")
            poses[child] = poses[parent] @ transform; pending.append(child)
    if len(poses) != len(links):
        raise ValueError(f"{path}: collision-link graph is disconnected")
    result: list[CollisionPrimitive] = []
    for name, link in links.items():
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError(f"{path}: collision in {name} must contain exactly one geometry")
            shape = geometry[0]; transform = poses[name] @ _origin(collision.find("origin"))
            if shape.tag == "box":
                result.append(CollisionPrimitive(transform, "box", tuple(float(item) for item in shape.get("size", "").split())))
            elif shape.tag == "cylinder":
                result.append(CollisionPrimitive(transform, "cylinder", (float(shape.get("radius", "nan")), float(shape.get("length", "nan")))))
            elif shape.tag == "sphere":
                result.append(CollisionPrimitive(transform, "sphere", (float(shape.get("radius", "nan")),)))
            else:
                raise ValueError(f"{path}: unsupported collision geometry {shape.tag!r} in {name}; use box/cylinder/sphere")
    if not result:
        raise ValueError(f"{path} has no collision primitives")
    return result


def _box_surface(size: np.ndarray, resolution_m: float) -> np.ndarray:
    half = np.asarray(size, dtype=np.float64) * .5
    step = max(float(resolution_m), .002)
    axes = [np.arange(-value, value + step * .5, step) for value in half]
    faces = []
    for fixed in range(3):
        other = [item for item in range(3) if item != fixed]
        first, second = np.meshgrid(axes[other[0]], axes[other[1]], indexing="ij")
        for sign in (-1., 1.):
            points = np.zeros((first.size, 3), dtype=np.float64); points[:, fixed] = sign * half[fixed]; points[:, other[0]] = first.ravel(); points[:, other[1]] = second.ravel(); faces.append(points)
    return np.unique(np.concatenate(faces), axis=0)


def _cylinder_surface(radius: float, length: float, resolution_m: float) -> np.ndarray:
    radius, length = float(radius), float(length)
    step = max(float(resolution_m), .002); count = max(16, int(np.ceil(2. * np.pi * radius / step))); theta = np.linspace(0., 2. * np.pi, count, endpoint=False); z = np.arange(-length * .5, length * .5 + step * .5, step)
    angle, height = np.meshgrid(theta, z, indexing="ij"); side = np.column_stack((radius * np.cos(angle.ravel()), radius * np.sin(angle.ravel()), height.ravel()))
    radial = np.arange(0., radius + step * .5, step); cap_angle, cap_radius = np.meshgrid(theta, radial, indexing="ij"); caps = [np.column_stack((cap_radius.ravel() * np.cos(cap_angle.ravel()), cap_radius.ravel() * np.sin(cap_angle.ravel()), np.full(cap_angle.size, sign * length * .5))) for sign in (-1., 1.)]
    return np.concatenate((side, *caps))


def _sphere_surface(radius: float, resolution_m: float) -> np.ndarray:
    count = max(12, int(np.ceil(np.pi * float(radius) / max(float(resolution_m), .002)))); polar = np.linspace(0., np.pi, count); azimuth = np.linspace(0., 2. * np.pi, count * 2, endpoint=False); p, a = np.meshgrid(polar, azimuth, indexing="ij")
    return np.column_stack((radius * np.sin(p.ravel()) * np.cos(a.ravel()), radius * np.sin(p.ravel()) * np.sin(a.ravel()), radius * np.cos(p.ravel())))


def sample_collision_surface(path: Path, *, margin_m: float = 0.0, resolution_m: float = .009) -> np.ndarray:
    """Sample a conservative surface in the root-link coordinates of ``path``."""
    margin = max(float(margin_m), 0.)
    surfaces = []
    for primitive in load_fixed_collision_urdf(path):
        if primitive.kind == "box":
            local = _box_surface(np.asarray(primitive.values) + 2. * margin, resolution_m)
        elif primitive.kind == "cylinder":
            local = _cylinder_surface(primitive.values[0] + margin, primitive.values[1] + 2. * margin, resolution_m)
        else:
            local = _sphere_surface(primitive.values[0] + margin, resolution_m)
        world = (primitive.transform[:3, :3] @ local.T).T + primitive.transform[:3, 3]; surfaces.append(world)
    return np.concatenate(surfaces).astype(np.float32)
