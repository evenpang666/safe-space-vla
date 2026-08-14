#!/usr/bin/env python3
"""Capture two D435i frames and quantify their calibrated overlap alignment.

This is a read-only diagnostic: it moves neither the UR7e nor either camera.
It writes a source-coloured PLY (front cyan, side amber) and a JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_scripts.real_robot_adapter import RGBDFrame, depth_to_world_points, load_camera_calibrations
from real_scripts.reconstruct_realsense_pointcloud import save_ply_ascii
from real_scripts.ur7e_realsense_adapter import D435iCameraConfig, RealSenseD435iSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--front-serial', default='405622074939')
    p.add_argument('--side-serial', default='348522070576')
    p.add_argument('--calibration', type=Path, default=REPO_ROOT/'real_scripts'/'ur7e_d435i_camera_calibration.json')
    p.add_argument('--output-dir', type=Path, default=REPO_ROOT/'outputs'/'dual_d435i_fusion_diagnostic')
    p.add_argument('--width', type=int, default=1280); p.add_argument('--height', type=int, default=720); p.add_argument('--fps', type=int, default=30); p.add_argument('--warmup-frames', type=int, default=30)
    p.add_argument('--stride', type=int, default=4); p.add_argument('--max-depth-m', type=float, default=2.5)
    p.add_argument('--workspace-bounds', nargs=6, type=float, default=(-.2,1.,-.8,.4,-.1,.45))
    return p.parse_args()


def main() -> None:
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); calibrations=load_camera_calibrations(args.calibration)
    source=RealSenseD435iSource(cameras=(D435iCameraConfig('front',args.front_serial),D435iCameraConfig('side',args.side_serial)),width=args.width,height=args.height,fps=args.fps)
    source.start()
    try:
        for _ in range(max(1,args.warmup_frames)): captured=source.read()
    finally:
        source.stop()
    clouds={}
    for name in ('front','side'):
        rgb,depth=captured[name]
        points,_=depth_to_world_points(RGBDFrame(name,rgb,depth),calibrations[name],stride=args.stride,max_depth=args.max_depth_m)
        xmin,xmax,ymin,ymax,zmin,zmax=args.workspace_bounds
        keep=(points[:,0]>=xmin)&(points[:,0]<=xmax)&(points[:,1]>=ymin)&(points[:,1]<=ymax)&(points[:,2]>=zmin)&(points[:,2]<=zmax)
        clouds[name]=points[keep]
    # Nearest neighbours are only meaningful in the common visible workspace;
    # report robust percentiles rather than letting single depth edge pixels dominate.
    distance,_=cKDTree(clouds['side']).query(clouds['front'],k=1,workers=-1)
    report={'coordinate_frame':'ur_base','front_point_count':int(len(clouds['front'])),'side_point_count':int(len(clouds['side'])),'front_to_side_nearest_m':{k:float(v) for k,v in zip(('p10','p50','p90','p95'),np.percentile(distance,(10,50,90,95)))},'interpretation':'p50 under 0.015 m and p90 under 0.035 m normally indicate useful two-camera alignment; inspect source_coloured_overlap.ply for structured double layers.'}
    colors=np.concatenate((np.tile(np.array([[30,220,255]],np.uint8),(len(clouds['front']),1)),np.tile(np.array([[255,180,35]],np.uint8),(len(clouds['side']),1))))
    save_ply_ascii(args.output_dir/'source_coloured_overlap.ply',np.concatenate((clouds['front'],clouds['side'])),colors)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
