#!/usr/bin/env python3
"""Live browser viewer for LingBot-fused RGB-D, UR7e/PiKA mask, and tabletop OBBs.

The robot connection is receive-only: this program calls RTDEReceiveInterface
for ``actual_q`` and never opens an RTDE control interface or sends motion.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import webbrowser

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.cluster_tabletop_objects import _expanded_obb
from real_scripts.demo_record_ur7e_safety_overlay_video import UprightOBB
from real_scripts.lingbot_depth import LingBotDepthRefiner
from real_scripts.real_robot_adapter import RGBDFrame, depth_to_world_points, load_camera_calibrations, robot_depth_keep_mask, voxel_downsample_points
from real_scripts.reconstruct_realsense_pointcloud import _cluster_points_3d, estimate_dominant_plane
from real_scripts.ur7e_collision_mesh import (
    collision_surface_samples,
    collision_volume_keep_mask,
    flange_transform,
    mesh_local_filled_voxel_indices,
    occupied_collision_voxels_from_local_indices,
    render_collision_depth,
    render_surface_points_depth,
    mesh_surface_samples,
)
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>UR7e live fused scene</title>
<style>html,body{margin:0;height:100%;background:#080a0d;color:#e6edf3;font:14px Arial}#hud{position:fixed;z-index:2;left:12px;top:10px;background:#000a;padding:9px 12px;border-radius:7px;line-height:1.55}canvas{width:100vw;height:100vh;display:block;cursor:grab}canvas:active{cursor:grabbing}.red{color:#ff5045}.green{color:#4cff90}</style></head>
<body><div id="hud"><b>UR7e live LingBot fused scene</b><br><span id="s">waiting for first processed frame…</span><br><span class="red">red: UR7e + PiKA</span> · <span class="green">green: tabletop obstacle OBB</span><br>drag: rotate · wheel: zoom</div><canvas id="c"></canvas>
<script>
const C=document.getElementById('c'),X=C.getContext('2d'),S=document.getElementById('s');let D=null,yaw=.82,pitch=-.52,dist=2.1,last;
function resize(){C.width=innerWidth*devicePixelRatio;C.height=innerHeight*devicePixelRatio;X.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0)}addEventListener('resize',resize);resize();
function centred(p){const c=D?.view_center||[0,0,0];return[p[0]-c[0],p[1]-c[1],p[2]-c[2]]}
function rot(p){p=centred(p);let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);let x=cy*p[0]-sy*p[1],z=sy*p[0]+cy*p[1],y=cp*p[1]-sp*z;return[x,y,sp*p[1]+cp*z]}
function cameraDistance(){return Math.max(dist,(D?.view_radius||0)*1.55)}
function proj(p){let q=rot(p),d=cameraDistance(),f=Math.min(innerWidth,innerHeight)*.82;return[innerWidth/2+f*q[0]/(q[2]+d),innerHeight/2-f*q[1]/(q[2]+d),q[2]]}
function visible(q){return q[2]>-cameraDistance()+.02}
function line(a,b,col){let u=proj(a),v=proj(b);if(!visible(u)||!visible(v))return;X.strokeStyle=col;X.beginPath();X.moveTo(u[0],u[1]);X.lineTo(v[0],v[1]);X.stroke()}
function drawAxes(){let c=D.view_center,r=Math.max(.25,D.view_radius*.35);line(c,[c[0]+r,c[1],c[2]],'#ff4b4b');line(c,[c[0],c[1]+r,c[2]],'#58ff70');line(c,[c[0],c[1],c[2]+r],'#579cff')}
function draw(){X.clearRect(0,0,innerWidth,innerHeight);if(!D)return;let all=[];for(const group of [D.environment,D.robot])for(let i=0;i<group.points.length;i++)all.push([proj(group.points[i]),group.colors[i]]);all.sort((a,b)=>b[0][2]-a[0][2]);let d=cameraDistance();for(const [q,c] of all){if(!visible(q))continue;let depth=Math.max(.08,q[2]+d),size=Math.max(1,Math.min(4,5.2/depth)),fade=Math.max(.28,Math.min(1,1.35-depth/(d*1.7)));X.fillStyle=`rgba(${c[0]},${c[1]},${c[2]},${fade})`;X.fillRect(q[0]-size*.5,q[1]-size*.5,size,size)}X.lineWidth=2;drawAxes();for(const b of D.obbs){let e=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];for(const z of e)line(b.corners[z[0]],b.corners[z[1]],'#43ff83')}}
async function poll(){try{let r=await fetch('/snapshot.json?'+Date.now());if(r.ok){D=await r.json();S.textContent=`frame ${D.frame} · ${D.status} · environment ${D.environment.points.length} · robot ${D.robot.points.length} · OBB ${D.obbs.length}`;draw()}}catch(e){S.textContent='connection retrying…'}setTimeout(poll,1000)}poll();
C.addEventListener('pointerdown',e=>last=[e.clientX,e.clientY]);addEventListener('pointerup',()=>last=null);addEventListener('pointermove',e=>{if(!last)return;yaw+=(e.clientX-last[0])*.008;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY];draw()});C.addEventListener('wheel',e=>{dist=Math.max(.2,Math.min(8,dist+e.deltaY*.001));draw();e.preventDefault()},{passive:false});addEventListener('keydown',e=>{if(e.key==='1'){yaw=0;pitch=0;draw()}if(e.key==='2'){yaw=.78;pitch=-.75;draw()}if(e.key.toLowerCase()==='r'){yaw=.82;pitch=-.52;dist=Math.max(.45,(D?.view_radius||.8)*1.55);draw()}});
</script></body></html>"""

# Native WebGL 3-D viewer.  This deliberately supersedes the small Canvas
# prototype above: Scatter3d uses an actual depth buffer and orbit camera, so
# a tabletop cannot be visually flattened by a hand-written projection.
HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>UR7e live fused scene</title>
<script src="/plotly.min.js"></script>
<style>html,body,#scene{margin:0;width:100%;height:100%;overflow:hidden;background:#080a0d;color:#e6edf3;font:14px Arial}#hud{position:fixed;z-index:2;left:12px;top:10px;background:#000c;padding:10px 13px;border-radius:7px;line-height:1.6;pointer-events:none}.red{color:#ff5045}.green{color:#4cff90}.dim{color:#9aa8b8}</style></head>
<body><div id="hud"><b>UR7e live fused scene — WebGL 3-D</b><br><span id="s">waiting for first processed frame...</span><br><span class="red">red: UR7e + PiKA</span> · <span class="green">green: full 3-D tabletop obstacle OBB</span><br><span class="dim">drag: orbit · wheel: zoom · double click: reset view</span></div><div id="scene"></div>
<script>
const G=document.getElementById('scene'),S=document.getElementById('s');let rendered=false,viewToken=0;
const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
function pointTrace(group,name,size){const p=group.points,c=group.colors;return {type:'scatter3d',mode:'markers',name,x:p.map(a=>a[0]),y:p.map(a=>a[1]),z:p.map(a=>a[2]),marker:{size,opacity:.82,color:c.map(v=>`rgb(${v[0]},${v[1]},${v[2]})`)},hoverinfo:'skip'}}
function boxTraces(boxes){let out=[];for(let n=0;n<boxes.length;n++){let q=boxes[n].corners;let x=[],y=[],z=[];for(const[e,f]of edges){x.push(q[e][0],q[f][0],null);y.push(q[e][1],q[f][1],null);z.push(q[e][2],q[f][2],null)}out.push({type:'scatter3d',mode:'lines',name:`obstacle ${n+1}`,x,y,z,line:{color:'#43ff83',width:6},hoverinfo:'skip'})}return out}
function layout(d){return {uirevision:'orbit-'+viewToken,showlegend:false,paper_bgcolor:'#080a0d',plot_bgcolor:'#080a0d',margin:{l:0,r:0,t:0,b:0},scene:{bgcolor:'#080a0d',aspectmode:'data',camera:{projection:{type:'perspective'},eye:{x:1.65,y:-1.65,z:1.25},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1}},xaxis:{title:'UR base X (m)',gridcolor:'#27313e',zerolinecolor:'#526273'},yaxis:{title:'UR base Y (m)',gridcolor:'#27313e',zerolinecolor:'#526273'},zaxis:{title:'UR base Z (m)',gridcolor:'#27313e',zerolinecolor:'#526273'}}}}
function render(d){let traces=[pointTrace(d.environment,'environment',0.65),pointTrace(d.robot,'UR7e + PiKA',1.0),...boxTraces(d.obbs)];Plotly.react(G,traces,layout(d),{displaylogo:false,responsive:true,scrollZoom:true});rendered=true;let span=(d.axis_ranges||[]).map(v=>v.toFixed(2)).join(' × ');S.textContent=`frame ${d.frame} · ${d.status} · environment ${d.environment.points.length} · robot ${d.robot.points.length} · OBB ${d.obbs.length} · xyz span ${span} m`}
async function poll(){try{const r=await fetch('/snapshot.json?'+Date.now());if(r.ok)render(await r.json())}catch(e){S.textContent='connection retrying...'}setTimeout(poll,1000)}poll();
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-ip", default="169.254.175.10")
    p.add_argument("--front-serial", default=os.environ.get("REAL_SENSE_FRONT_SERIAL", "405622074939"))
    p.add_argument("--side-serial", default=os.environ.get("REAL_SENSE_SIDE_SERIAL", "348522070576"))
    p.add_argument("--calibration", type=Path, default=REPO_ROOT / "real_scripts" / "ur7e_d435i_camera_calibration.json")
    p.add_argument("--pika-mount-transform-json", type=Path, default=REPO_ROOT / "outputs" / "ur7e_d435i_live_lingbot_urdf" / "pika_mount_candidate_unverified.json")
    p.add_argument("--pika-mesh", type=Path, default=REPO_ROOT / "assets" / "robot_models" / "pika_gripper" / "collision" / "pika_gripper_full_collision.stl")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--width", type=int, default=1280); p.add_argument("--height", type=int, default=720); p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--lingbot-device", default="cuda", help="LingBot device; use cpu only for diagnostics.")
    p.add_argument("--volume-pitch-m", type=float, default=.006); p.add_argument("--volume-margin-m", type=float, default=.015)
    p.add_argument("--point-stride", type=int, default=3, help="Pixel stride for live fusion; 2 is denser, 4 is faster.")
    p.add_argument("--urdf-samples-per-face", type=int, default=9, help="Cached URDF surface samples per triangle for the depth mask.")
    p.add_argument("--display-max-points", type=int, default=60000)
    p.add_argument("--workspace-bounds", nargs=6, type=float, default=(-0.2, 1.0, -0.8, 0.4, -0.1, 0.45), metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"), help="UR-base workcell crop for display, obstacles, and fusion diagnostics.")
    p.add_argument("--outlier-neighbour-m", type=float, default=.020); p.add_argument("--cluster-radius-m", type=float, default=.015); p.add_argument("--min-cluster-points", type=int, default=100); p.add_argument("--attachment-distance-m", type=float, default=.050)
    p.add_argument("--once", action="store_true", help="Process one frame and write a snapshot without serving a browser.")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def load_transform(path: Path) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8")).get("flange_to_pika_step_frame")
    t = np.asarray(value, dtype=np.float64)
    if t.shape != (4, 4) or not np.allclose(t[3], (0, 0, 0, 1)):
        raise ValueError("PiKA transform must be a 4x4 homogeneous matrix")
    return t


def cap(points: np.ndarray, colors: np.ndarray, limit: int) -> tuple[list, list]:
    if len(points) > limit:
        ind = np.linspace(0, len(points) - 1, limit, dtype=np.int64); points, colors = points[ind], colors[ind]
    return points.astype(float).tolist(), colors.astype(int).tolist()


def view_parameters(*point_sets: np.ndarray) -> tuple[list[float], float]:
    """Return a robust world-space focus and scale for the browser viewer."""
    nonempty = [np.asarray(points, dtype=np.float64).reshape(-1, 3) for points in point_sets if len(points)]
    if not nonempty:
        return [0.0, 0.0, 0.0], 0.8
    points = np.concatenate(nonempty, axis=0)
    # Ignore the farthest depth speckles for framing only; all points still draw.
    low, high = np.percentile(points, (2.0, 98.0), axis=0)
    return ((low + high) * 0.5).astype(float).tolist(), max(float(np.linalg.norm(high - low) * 0.5), 0.25)


def tabletop_aligned_obb(points: np.ndarray, tabletop_normal: np.ndarray, tabletop_offset: float) -> UprightOBB | None:
    """Fit a full 3-D obstacle OBB whose third axis is the fitted tabletop normal.

    The UR controller base frame is not assumed to have its Z axis normal to the
    physical table.  This is important for both the displayed box height and
    safety volume: a fixed-world-Z "upright" box can collapse into a sheet.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        return None
    normal = np.asarray(tabletop_normal, dtype=np.float64).reshape(3)
    normal /= max(np.linalg.norm(normal), 1e-12)
    # Select the least-parallel world axis, then form an orthonormal tabletop basis.
    seed = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    first = np.cross(seed, normal); first /= max(np.linalg.norm(first), 1e-12)
    second = np.cross(normal, first)
    plane = points @ np.column_stack((first, second))
    centered = plane - plane.mean(axis=0)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    _, eigvec = np.linalg.eigh(covariance)
    planar_axes = np.column_stack((first, second)) @ eigvec[:, ::-1]
    rotation = np.column_stack((planar_axes, normal))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 1] *= -1.0
    local = points @ rotation
    local_min, local_max = local.min(axis=0), local.max(axis=0)
    # Obstacles are defined as connected to the table, so close the lower face
    # to the fitted plane instead of leaving an artificial floating thin sheet.
    local_min[2] = min(local_min[2], -float(tabletop_offset))
    extents = np.maximum(local_max - local_min, 1e-4)
    local_center = (local_min + local_max) * 0.5
    center = rotation @ local_center
    signs = np.asarray(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)), dtype=np.float64)
    corners = center[None, :] + (0.5 * signs * extents[None, :]) @ rotation.T
    return UprightOBB(center=center.astype(np.float32), rotation=rotation.astype(np.float32), extents=extents.astype(np.float32), corners=corners.astype(np.float32), point_count=int(len(points)))


def cached_filled_voxel_indices(
    cache_path: Path,
    mesh: object,
    *,
    mesh_source: Path,
    voxel_pitch_m: float,
) -> np.ndarray:
    """Load a local filled-mesh voxel cache, rebuilding only when needed."""
    signature = np.asarray((float(voxel_pitch_m), float(mesh_source.stat().st_mtime_ns), float(mesh_source.stat().st_size)))
    try:
        cached = np.load(cache_path)
        if np.array_equal(cached["signature"], signature):
            return np.asarray(cached["indices"], dtype=np.int64)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        pass
    indices = mesh_local_filled_voxel_indices(mesh, voxel_pitch_m=voxel_pitch_m)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, signature=signature, indices=indices)
    return indices


def main() -> None:
    args = parse_args(); calibrations = load_camera_calibrations(args.calibration)
    import trimesh
    from rtde_receive import RTDEReceiveInterface
    pika_mesh = trimesh.load_mesh(args.pika_mesh, process=False); pika_mesh.vertices = np.asarray(pika_mesh.vertices, dtype=np.float64) * .001
    pika_mount = load_transform(args.pika_mount_transform_json)
    # Expensive mesh work is invariant to joint position, so do it once.
    from real_scripts.ur7e_collision_mesh import COLLISION_ROOT, load_collision_meshes
    cache_dir = REPO_ROOT / "outputs" / "live_ur7e_mesh_voxel_cache"
    local_volume_indices = {
        name: cached_filled_voxel_indices(cache_dir / f"{name}_{args.volume_pitch_m:.6f}.npz", mesh, mesh_source=COLLISION_ROOT / f"{name}.stl", voxel_pitch_m=args.volume_pitch_m)
        for name, mesh in load_collision_meshes().items()
    }
    pika_volume_indices = cached_filled_voxel_indices(cache_dir / f"pika_{args.volume_pitch_m:.6f}.npz", pika_mesh, mesh_source=args.pika_mesh, voxel_pitch_m=args.volume_pitch_m)
    urdf_surface_samples = collision_surface_samples(samples_per_face=args.urdf_samples_per_face)
    pika_surface_samples = mesh_surface_samples(pika_mesh, samples_per_face=4)
    latest: dict = {"frame": 0, "status": "initializing", "environment": {"points": [], "colors": []}, "robot": {"points": [], "colors": []}, "obbs": []}
    lock = threading.Lock(); stop = threading.Event()

    def process_loop() -> None:
        nonlocal latest
        receiver = RTDEReceiveInterface(str(args.robot_ip)); source = RealSenseD435iSource(cameras=(D435iCameraConfig("front", str(args.front_serial)), D435iCameraConfig("side", str(args.side_serial))), width=args.width, height=args.height, fps=args.fps)
        refiner = LingBotDepthRefiner(device=args.lingbot_device, use_fp16=str(args.lingbot_device).lower().startswith("cuda")); frame_index = 0
        try:
            source.start()
            for _ in range(max(1, args.warmup_frames)): source.read()
            while not stop.is_set():
                frame_started = time.perf_counter()
                q0 = np.asarray(receiver.getActualQ(), dtype=np.float32); captured = source.read(); q1 = np.asarray(receiver.getActualQ(), dtype=np.float32); q = (q0 + q1) * .5
                refined = refiner.refine([RGBDFrame(name, captured[name][0], captured[name][1]) for name in ("front", "side")], calibrations); repair_seconds = refiner.last_inference_seconds
                pika_to_base = flange_transform(q) @ pika_mount
                volume, pitch = occupied_collision_voxels_from_local_indices(q, local_indices=local_volume_indices, voxel_pitch_m=args.volume_pitch_m, exterior_margin_m=args.volume_margin_m, extra_local_indices=(pika_volume_indices, pika_to_base))
                robot_sets=[]; env_sets=[]
                for f in refined:
                    name=f.camera_name; rendered=render_collision_depth(q,calibrations[name].camera_to_world,calibrations[name].intrinsics,width=f.rgb.shape[1],height=f.rgb.shape[0],samples_per_face=args.urdf_samples_per_face,splat_radius_pixels=2,local_samples=urdf_surface_samples)
                    pika_points=(pika_to_base[:3,:3]@pika_surface_samples.T).T+pika_to_base[:3,3]; pdepth=render_surface_points_depth(pika_points,calibrations[name].camera_to_world,calibrations[name].intrinsics,width=f.rgb.shape[1],height=f.rgb.shape[0],splat_radius_pixels=3)
                    rendered=np.where((rendered>0)&(pdepth>0),np.minimum(rendered,pdepth),np.maximum(rendered,pdepth)); keep=robot_depth_keep_mask(f.depth_m,rendered,absolute_tolerance_m=.025,relative_tolerance=.03,dilation_pixels=4)
                    rgbd=RGBDFrame(name,f.rgb,f.depth_m); ap,ac=depth_to_world_points(rgbd,calibrations[name],stride=args.point_stride,max_depth=2.5); ep,ec=depth_to_world_points(rgbd,calibrations[name],stride=args.point_stride,max_depth=2.5,keep_mask=keep); rp,rc=depth_to_world_points(rgbd,calibrations[name],stride=args.point_stride,max_depth=2.5,keep_mask=~keep)
                    vk_all=collision_volume_keep_mask(ap,volume,voxel_pitch_m=pitch); vk_env=collision_volume_keep_mask(ep,volume,voxel_pitch_m=pitch); robot_sets.append((np.concatenate((rp,ap[~vk_all])),np.concatenate((rc,ac[~vk_all])))); env_sets.append((ep[vk_env],ec[vk_env]))
                def fuse(items): return voxel_downsample_points(np.concatenate([a for a,_ in items]),np.concatenate([b for _,b in items]),voxel_size=.005)
                robot, rcol=fuse(robot_sets); env, ecol=fuse(env_sets)
                xmin,xmax,ymin,ymax,zmin,zmax = (float(v) for v in args.workspace_bounds)
                in_workcell=(env[:,0]>=xmin)&(env[:,0]<=xmax)&(env[:,1]>=ymin)&(env[:,1]<=ymax)&(env[:,2]>=zmin)&(env[:,2]<=zmax)
                env,ecol=env[in_workcell],ecol[in_workcell]
                # Fit tabletop locally, remove isolated speckles, then retain only 3-D components touching the tabletop.
                local=env[(env[:,0]>-.2)&(env[:,0]<1)&(env[:,1]>-.8)&(env[:,1]<.4)&(env[:,2]>-.1)&(env[:,2]<.12)]; normal,offset,_=estimate_dominant_plane(local,threshold=.012,ransac_iterations=250)
                if normal[2]<0: normal,offset=-normal,-offset
                # OBB input is strictly the already robot-filtered ``env`` cloud.
                # Exclude the tabletop plane itself, retain only points above it,
                # then require each *3-D* component to reach the attachment band.
                # Thus red UR7e/PiKA points, bare table, and suspended clusters
                # can never create a bounding box.
                h=env.astype(np.float64)@normal+offset
                object_candidate=(h>=.025)&(h<=.35)&(env[:,0]>-.2)&(env[:,0]<1)&(env[:,1]>-.8)&(env[:,1]<.4)
                op,oc,oh=env[object_candidate],ecol[object_candidate],h[object_candidate]
                if len(op)>1:
                    d,_=cKDTree(op).query(op,k=2,workers=-1); good=d[:,1]<=args.outlier_neighbour_m; op,oc,oh=op[good],oc[good],oh[good]
                obbs=[]; tabletop_connected_components=0; floating_components=0
                for ind in _cluster_points_3d(op,cluster_radius=args.cluster_radius_m,min_cluster_points=args.min_cluster_points):
                    if oh[ind].min()<=args.attachment_distance_m:
                        tabletop_connected_components+=1
                        box=tabletop_aligned_obb(op[ind], normal, offset)
                        if box is not None: obbs.append(_expanded_obb(box,.008))
                    else:
                        floating_components+=1
                frame_index+=1; view_center,view_radius=view_parameters(env,robot); ep,ec=cap(env,ecol,args.display_max_points); rp,rc=cap(robot,np.tile(np.array([[255,55,45]],np.uint8),(len(robot),1)),args.display_max_points//3)
                all_for_ranges = np.concatenate((env, robot), axis=0)
                axis_ranges = np.ptp(all_for_ranges, axis=0).astype(float).tolist() if len(all_for_ranges) else [0.0, 0.0, 0.0]
                snapshot={"frame":frame_index,"status":f"LingBot {refiner.last_inference_seconds:.1f}s · qΔ {float(np.max(np.abs(q1-q0))):.5f} rad · {len(obbs)} tabletop-connected obstacles","environment":{"points":ep,"colors":ec},"robot":{"points":rp,"colors":rc},"obbs":[{"corners":b.corners.astype(float).tolist()} for b in obbs]}
                snapshot["status"] = f"frame {time.perf_counter()-frame_started:.1f}s (LingBot {repair_seconds:.1f}s) · q delta {float(np.max(np.abs(q1-q0))):.5f} rad · {len(obbs)} tabletop-connected obstacles · {floating_components} floating excluded"
                snapshot["view_center"] = view_center; snapshot["view_radius"] = view_radius; snapshot["axis_ranges"] = axis_ranges
                with lock: latest=snapshot
                if args.once: break
        except Exception as e:
            with lock: latest={**latest,"status":f"error: {type(e).__name__}: {e}"}
            raise
        finally:
            source.stop(); receiver.disconnect()

    worker=threading.Thread(target=process_loop,daemon=True); worker.start()
    if args.once:
        worker.join(); print(json.dumps(latest)[0:500]); return
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/snapshot.json'):
                with lock: data=json.dumps(latest).encode('utf-8')
                self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
            elif self.path.startswith('/plotly.min.js'):
                try:
                    import plotly
                    asset = Path(plotly.__file__).resolve().parent / 'package_data' / 'plotly.min.js'
                    data = asset.read_bytes()
                    self.send_response(200); self.send_header('Content-Type','application/javascript'); self.send_header('Cache-Control','public, max-age=86400'); self.end_headers(); self.wfile.write(data)
                except Exception as exc:
                    self.send_error(500, f'Could not load local Plotly asset: {exc}')
            else:
                self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(HTML.encode('utf-8'))
        def log_message(self, *args): pass
    server=ThreadingHTTPServer(('127.0.0.1',int(args.port)),Handler); url=f'http://127.0.0.1:{args.port}'
    print(f'[viewer] {url}  (Ctrl+C to stop)')
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: stop.set(); server.shutdown()


if __name__ == '__main__': main()
