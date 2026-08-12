"""Official UR7e collision-mesh FK helpers in the UR controller base frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLISION_ROOT = REPO_ROOT / "assets" / "robot_models" / "ur_description" / "meshes" / "ur5e" / "collision"
MESH_NAMES = ("base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3")


def _translation(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (x, y, z)
    return result


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((1, 0, 0, 0), (0, c, -s, 0), (0, s, c, 0), (0, 0, 0, 1)), dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, -s, 0, 0), (s, c, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, 0, s, 0), (0, 1, 0, 0), (-s, 0, c, 0), (0, 0, 0, 1)), dtype=np.float64)


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw rotation: Rz(yaw) Ry(pitch) Rx(roll)."""
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def link_and_collision_transforms(qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Return ``^base T_collision_mesh`` for official UR7e collision meshes.

    The transforms are the official ur7e kinematics YAML joint origins plus
    collision origins from ``ur_macro.xacro``.  ``base`` is in the controller
    base convention used by the RTDE joint angles and camera calibration.
    """
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if q.size < 6:
        raise ValueError("qpos must contain six UR joint angles")
    q = q[:6]
    links: dict[str, np.ndarray] = {}
    links["base"] = np.eye(4, dtype=np.float64)
    shoulder = _translation(0.0, 0.0, 0.1625) @ _rotation_z(q[0])
    upperarm = shoulder @ _rotation_x(np.pi / 2.0) @ _rotation_z(q[1])
    forearm = upperarm @ _translation(-0.425, 0.0, 0.0) @ _rotation_z(q[2])
    wrist1 = forearm @ _translation(-0.3922, 0.0, 0.1333) @ _rotation_z(q[3])
    wrist2 = wrist1 @ _translation(0.0, -0.0997, 0.0) @ _rotation_x(np.pi / 2.0) @ _rotation_z(q[4])
    wrist3 = wrist2 @ _translation(0.0, 0.0996, 0.0) @ _rpy(np.pi / 2.0, np.pi, np.pi) @ _rotation_z(q[5])
    links.update({"shoulder": shoulder, "upperarm": upperarm, "forearm": forearm, "wrist1": wrist1, "wrist2": wrist2, "wrist3": wrist3})
    return {
        "base": links["base"],
        "shoulder": shoulder @ _rotation_z(np.pi),
        "upperarm": upperarm @ _translation(0.0, 0.0, 0.138) @ _rpy(np.pi / 2.0, 0.0, -np.pi / 2.0),
        "forearm": forearm @ _translation(0.0, 0.0, 0.007) @ _rpy(np.pi / 2.0, 0.0, -np.pi / 2.0),
        "wrist1": wrist1 @ _translation(0.0, 0.0, -0.127) @ _rotation_x(np.pi / 2.0),
        "wrist2": wrist2 @ _translation(0.0, 0.0, -0.0997),
        "wrist3": wrist3 @ _translation(0.0, -0.0005, -0.0989) @ _rotation_x(np.pi / 2.0),
    }


def flange_transform(qpos: np.ndarray) -> np.ndarray:
    """Return ``^base T_flange`` using the fixed transform in ur_macro.xacro."""
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if q.size < 6:
        raise ValueError("qpos must contain six UR joint angles")
    q = q[:6]
    shoulder = _translation(0.0, 0.0, 0.1625) @ _rotation_z(q[0])
    upperarm = shoulder @ _rotation_x(np.pi / 2.0) @ _rotation_z(q[1])
    forearm = upperarm @ _translation(-0.425, 0.0, 0.0) @ _rotation_z(q[2])
    wrist1 = forearm @ _translation(-0.3922, 0.0, 0.1333) @ _rotation_z(q[3])
    wrist2 = wrist1 @ _translation(0.0, -0.0997, 0.0) @ _rotation_x(np.pi / 2.0) @ _rotation_z(q[4])
    wrist3 = wrist2 @ _translation(0.0, 0.0996, 0.0) @ _rpy(np.pi / 2.0, np.pi, np.pi) @ _rotation_z(q[5])
    return wrist3 @ _rpy(0.0, -np.pi / 2.0, -np.pi / 2.0)


def load_collision_meshes() -> dict[str, trimesh.Trimesh]:
    return {name: trimesh.load_mesh(COLLISION_ROOT / f"{name}.stl", process=False) for name in MESH_NAMES}


def world_mesh_vertices(qpos: np.ndarray) -> dict[str, np.ndarray]:
    transforms = link_and_collision_transforms(qpos)
    result: dict[str, np.ndarray] = {}
    for name, mesh in load_collision_meshes().items():
        vertices = np.concatenate((np.asarray(mesh.vertices, dtype=np.float64), np.ones((len(mesh.vertices), 1))), axis=1)
        result[name] = (transforms[name] @ vertices.T).T[:, :3]
    return result


def sample_collision_surface_points(qpos: np.ndarray, *, samples_per_face: int = 4) -> np.ndarray:
    """Deterministically sample the official collision surfaces in ``base``."""
    transforms = link_and_collision_transforms(qpos)
    points: list[np.ndarray] = []
    # A deterministic barycentric pattern covers each triangle without random
    # flicker between real-time frames.
    count = max(1, int(samples_per_face))
    # Build a regular barycentric lattice with at least ``count`` samples.
    # The former four hard-coded positions silently capped every request at
    # four samples per face, leaving visible striping on the cylindrical links.
    subdivisions = 1
    while (subdivisions + 1) * (subdivisions + 2) // 2 < count:
        subdivisions += 1
    barycentric: list[tuple[float, float, float]] = []
    for i in range(subdivisions + 1):
        for j in range(subdivisions + 1 - i):
            a = (i + 1.0 / 3.0) / (subdivisions + 1)
            b = (j + 1.0 / 3.0) / (subdivisions + 1)
            c = 1.0 - a - b
            barycentric.append((a, b, c))
    for name, mesh in load_collision_meshes().items():
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = vertices[np.asarray(mesh.faces, dtype=np.int64)]
        local = np.concatenate([a * triangles[:, 0] + b * triangles[:, 1] + c * triangles[:, 2] for a, b, c in barycentric], axis=0)
        homogeneous = np.concatenate((local, np.ones((len(local), 1))), axis=1)
        points.append((transforms[name] @ homogeneous.T).T[:, :3])
    return np.concatenate(points, axis=0).astype(np.float32)


def sample_mesh_surface_points(mesh: trimesh.Trimesh, transform: np.ndarray, *, samples_per_face: int = 4) -> np.ndarray:
    """Deterministically sample one mesh surface after a supplied 4x4 pose."""
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
    count = max(1, int(samples_per_face))
    subdivisions = 1
    while (subdivisions + 1) * (subdivisions + 2) // 2 < count:
        subdivisions += 1
    local = np.concatenate(
        [
            ((i + 1.0 / 3.0) / (subdivisions + 1)) * triangles[:, 0]
            + ((j + 1.0 / 3.0) / (subdivisions + 1)) * triangles[:, 1]
            + (1.0 - (i + 1.0 / 3.0) / (subdivisions + 1) - (j + 1.0 / 3.0) / (subdivisions + 1)) * triangles[:, 2]
            for i in range(subdivisions + 1)
            for j in range(subdivisions + 1 - i)
        ],
        axis=0,
    )
    homogeneous = np.concatenate((local, np.ones((len(local), 1))), axis=1)
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3].astype(np.float32)


def render_surface_points_depth(
    points: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    *,
    width: int,
    height: int,
    splat_radius_pixels: int = 2,
) -> np.ndarray:
    """Z-buffer a base-frame point-sampled mesh into a colour-camera depth map."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    camera = (world_to_camera @ np.concatenate((points, np.ones((len(points), 1))), axis=1).T).T[:, :3]
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z > 1e-5)
    fx, fy, cx, cy = float(intrinsics[0, 0]), float(intrinsics[1, 1]), float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u = np.rint(fx * camera[:, 0] / z + cx).astype(np.int64)
    v = np.rint(fy * camera[:, 1] / z + cy).astype(np.int64)
    valid &= (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    depth = np.full((int(height), int(width)), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v[valid], u[valid]), z[valid].astype(np.float32))
    radius = max(0, int(splat_radius_pixels))
    if radius:
        source = depth.copy()
        padded = np.pad(source, ((radius, radius), (radius, radius)), constant_values=np.inf)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx or dy:
                    depth = np.minimum(depth, padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width])
    depth[~np.isfinite(depth)] = 0.0
    return depth


def occupied_collision_voxels(
    qpos: np.ndarray,
    *,
    voxel_pitch_m: float = 0.006,
    exterior_margin_m: float = 0.015,
    extra_meshes: tuple[tuple[trimesh.Trimesh, np.ndarray], ...] = (),
) -> tuple[np.ndarray, float]:
    """Return centres of filled collision-volume voxels in the UR base frame.

    Vendor collision STLs are not perfectly watertight. ``trimesh.voxelized``
    followed by ``fill`` repairs small openings into a conservative enclosed
    volume; a Chebyshev-neighbour expansion supplies the requested exterior
    pose/depth tolerance.  This is deliberately a *volume* representation,
    not a surface z-buffer, so points inside the robot are also rejected.
    """
    pitch = float(voxel_pitch_m)
    margin = float(exterior_margin_m)
    if pitch <= 0.0 or margin < 0.0:
        raise ValueError("voxel_pitch_m must be > 0 and exterior_margin_m must be >= 0")
    transforms = link_and_collision_transforms(qpos)
    mesh_items: list[tuple[trimesh.Trimesh, np.ndarray]] = []
    mesh_items.extend((mesh, transforms[name]) for name, mesh in load_collision_meshes().items())
    mesh_items.extend(extra_meshes)
    indices: list[np.ndarray] = []
    for mesh, transform in mesh_items:
        world_mesh = mesh.copy()
        world_mesh.apply_transform(np.asarray(transform, dtype=np.float64))
        # PiKA meshes use millimetres at source; callers pass a metres-scaled mesh.
        filled = world_mesh.voxelized(pitch).fill()
        if len(filled.points):
            indices.append(np.rint(np.asarray(filled.points, dtype=np.float64) / pitch).astype(np.int64))
    if not indices:
        return np.zeros((0, 3), dtype=np.float32), pitch
    occupied = np.unique(np.concatenate(indices, axis=0), axis=0)
    radius = int(np.ceil(margin / pitch))
    if radius:
        offsets = np.stack(
            np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        occupied = np.unique((occupied[:, None, :] + offsets[None, :, :]).reshape(-1, 3), axis=0)
    return (occupied.astype(np.float64) * pitch).astype(np.float32), pitch


def collision_volume_keep_mask(points_base: np.ndarray, occupied_voxel_centres: np.ndarray, *, voxel_pitch_m: float) -> np.ndarray:
    """True for points outside a precomputed filled-and-expanded collision volume."""
    points = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    centres = np.asarray(occupied_voxel_centres, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or len(centres) == 0:
        return np.ones(len(points), dtype=bool)
    pitch = float(voxel_pitch_m)
    occupied = {tuple(value) for value in np.rint(centres / pitch).astype(np.int64)}
    point_indices = np.rint(points / pitch).astype(np.int64)
    return np.fromiter((tuple(value) not in occupied for value in point_indices), dtype=bool, count=len(point_indices))


def render_collision_depth(
    qpos: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    *,
    width: int,
    height: int,
    samples_per_face: int = 4,
    splat_radius_pixels: int = 2,
) -> np.ndarray:
    """Rasterize sampled URDF collision surfaces to a conservative z-buffer."""
    points = sample_collision_surface_points(qpos, samples_per_face=samples_per_face)
    return render_surface_points_depth(
        points, camera_to_world, intrinsics, width=width, height=height, splat_radius_pixels=splat_radius_pixels,
    )
