# Real UR SafetyModule Collection

This directory contains the real-robot counterpart of the LIBERO safety dataset
collector. The default hardware adapter targets a UR7e/UR e-Series arm with two
scene Intel RealSense D435i cameras plus one wrist-mounted D435i camera:
`front`, `side`, and `wrist`.

Start the PI05 prefix policy server from an OpenPI environment:

```bash
uv run --project openpi scripts/serve_pi05_prefix_policy.py \
  --policy-config pi05_ur7 \
  --checkpoint-dir /path/to/pi05_ur7_checkpoint \
  --port 8000
```

Then run the collector with a hardware adapter:

```bash
export UR_ROBOT_IP=169.254.26.10
export REAL_SENSE_FRONT_SERIAL=front_camera_serial
export REAL_SENSE_SIDE_SERIAL=side_camera_serial
export REAL_SENSE_WRIST_SERIAL=wrist_camera_serial

python real_scripts/collect_pi05_real_safety_dataset.py \
  --prompt "pick up the block" \
  --adapter real_scripts.ur7e_realsense_adapter:create_adapter \
  --camera-calibration real_scripts/ur7e_d435i_camera_calibration.example.json \
  --output outputs/pi05_safety_decoder/pi05_real_ur_safety_dataset.npz
```

The adapter must provide `front`, `side`, and `wrist` RGB-D frames, six UR joint
positions, gripper state, and action execution. For
`real_scripts.ur7e_realsense_adapter:create_adapter`, each PI05 action is sent
to `UR7eVectorController.send_ee_delta_vector()` as
`[dx_mm, dy_mm, dz_mm, droll, dpitch, dyaw, g]`; xyz is in millimeters in the
gripper local frame, rotations are radians, and `g` is an absolute gripper
command in `[0, 1]`. The collector uses UR FK to create the
fixed-topology `current_link_points`, `target_link_points`, `arm_points`, and
`target_point_offsets` fields required by the existing SafetyModule training
code.

## Passive overlay demo

If the UR7e is controlled by another fixed script, use the passive demo recorder
to validate the real-time geometry pipeline without sending any robot actions:

```bash
python real_scripts/demo_record_ur7e_safety_overlay_video.py \
  --adapter real_scripts.ur7e_realsense_adapter:create_adapter \
  --camera-calibration real_scripts/ur7e_d435i_camera_calibration.example.json \
  --workspace-bounds -0.8 0.8 -0.8 0.8 -0.05 0.8 \
  --table-z 0.0 \
  --output outputs/real_ur_safety_overlay_demo.mp4
```

The video is rendered from the `front` camera. Cyan points are UR7e FK surface
points, orange points are robot-filtered tabletop obstacle points, and green
wireframes are upright OBBs estimated from clustered obstacle points. Tune
`--table-z`, `--min-obstacle-height`, `--max-obstacle-height`,
`--cluster-radius`, and `--robot-filter-radius` for the real table/camera setup.

The calibration JSON values above are placeholders. Replace each D435i
intrinsic matrix and `camera_to_world` transform with calibrated values before
using the point cloud or OBB outputs for safety decisions.
