# Real UR7e Safety Geometry Pipeline

This directory implements the real-hardware geometry chain used by the safety
module:

```text
front / side D435i RGB-D
  → (optional) LingBot-Depth metric depth repair
  → back-project with colour-stream intrinsics
  → transform each cloud into the UR base frame
  → voxel fusion
  → remove the UR7e/PiKA with mesh-depth consistency (or capsule fallback)
  → workspace/tabletop filtering and obstacle OBBs
```

`front` and `side` are the two fixed scene cameras used for the safety point
cloud. `wrist` remains available to PI05, but is intentionally not fused: its
pose changes with the robot and needs a separate dynamic extrinsic pipeline.

## 0. Safety and coordinate convention

- Keep the robot stopped or in reduced-speed/manual mode while collecting
  calibration data. The passive demo never sends actions, but the collector
  does.
- All distances, depths, workspace limits, link radii, and transforms are in
  **metres**.
- `camera_to_world` in the calibration JSON means `^base T_camera`: it maps a
  point in the RGB-aligned D435i colour-camera frame into the UR controller's
  `base` frame. In this workflow, `world == ur_base`.
- The depth stream is aligned to the colour stream in
  `RealSenseD435iSource`; therefore both PnP and depth back-projection use the
  **colour** intrinsics at the live stream resolution.
- The two USB devices are read independently. The included fusion path is for
  static obstacles and a robot pose sampled in the same control cycle. For
  fast-moving obstacles or metrology, use D435i hardware synchronization (or
  record and reject frames outside a measured timestamp skew) before fusion.

Set the fixed-camera serials before using any live command:

```bash
export UR_ROBOT_IP=169.254.26.10
export REAL_SENSE_FRONT_SERIAL=front_camera_serial
export REAL_SENSE_SIDE_SERIAL=right_camera_serial
export REAL_SENSE_WRIST_SERIAL=wrist_camera_serial  # PI05 only
```

## 1. Camera calibration: D435i colour frame to UR base

The checked-in `ur7e_d435i_camera_calibration.example.json` is only a schema
example. It has placeholder identities and must never be used for a safety
decision.

### 1.1 Prepare the calibration target

Mount one rigid ChArUco board in the common workspace where both fixed cameras
can see it. Record its exact parameters: OpenCV dictionary, `squares_x`,
`squares_y`, square side length, and marker side length. The board must not move
after its pose in the robot base has been measured.

Determine `^base T_board` by either:

1. Using a calibrated TCP probe to touch at least four non-collinear known
   ChArUco board corners and recording their positions from the UR controller;
   provide paired board-frame and base-frame coordinates below; or
2. Providing a previously surveyed `board_to_base` 4×4 transform.

For the probe route, create `outputs/calibration/board_base_points.json` using
the same ChArUco board coordinate convention as OpenCV:

```json
{
  "board_points_m": [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.04, 0.04, 0.0]],
  "base_points_m": [[0.41, -0.17, 0.02], [0.41, -0.13, 0.02], [0.37, -0.17, 0.02], [0.37, -0.13, 0.02]]
}
```

The numbers are illustrative. The script rejects a correspondence fit whose
RMS exceeds 3 mm; remeasure rather than accepting it. Ensure the TCP's tool
offset is calibrated before probing.

### 1.2 Capture actual D435i intrinsics and board images

This captures one settled RGB frame per camera and exports the RealSense SDK
intrinsics and distortion coefficients for exactly that stream configuration:

```bash
python real_scripts/capture_d435i_calibration_frame.py \
  --output-dir outputs/calibration/d435i \
  --width 640 --height 480 --fps 30
```

Check that `front_rgb.png` and `side_rgb.png` contain many sharp board corners.
Retake them if there is blur, glare, or partial visibility.

### 1.3 Solve and write the calibration JSON

Install `opencv-contrib-python` (not just base OpenCV) because ChArUco needs
`cv2.aruco`, then run with the physical board dimensions:

```bash
python real_scripts/calibrate_d435i_to_ur_base.py \
  --input-dir outputs/calibration/d435i \
  --board-base-correspondences-json outputs/calibration/board_base_points.json \
  --squares-x 7 --squares-y 5 \
  --square-length-m 0.040 --marker-length-m 0.030 \
  --output real_scripts/ur7e_d435i_camera_calibration.json
```

The script solves `^camera T_board` with PnP and writes:

```text
^base T_camera = ^base T_board · inverse(^camera T_board)
```

for both cameras. It rejects high ChArUco reprojection error by default. The
output JSON can be passed directly to all commands below.

### 1.4 Validate the extrinsics before enabling safety actions

Run the passive overlay command in section 4 with a board or rigid object in
both views. Its point clouds should overlap in the base frame. Also probe a few
visible physical points with the TCP and compare their base coordinates to the
reconstructed cloud. Increase `--robot-filter-margin` and all downstream safety
margins by the measured residual; do not compensate with arbitrary transforms
or ICP alone.

## 2. Optional LingBot-Depth repair

Install [LingBot-Depth](https://github.com/Robbyant/lingbot-depth) in the same
runtime that executes the safety geometry process. The upstream project pins
PyTorch/xFormers, so validate compatibility with the OpenPI CUDA environment
before altering a production deployment.

```bash
git clone https://github.com/Robbyant/lingbot-depth /opt/lingbot-depth
python -m pip install -e /opt/lingbot-depth
```

`--lingbot-depth` loads the recommended v0.5 model once, then for every frame:

1. sends RGB in `[0, 1]`, raw D435i depth in metres, and normalized colour
   intrinsics to `MDMModel.infer`;
2. removes non-finite/invalid output; and
3. returns repaired `front` and `side` frames only for reconstruction.

Default FP16 inference needs CUDA. For functional CPU diagnostics use
`--lingbot-device cpu --no-lingbot-fp16`; it is not appropriate for a live
control loop.

## 3. Fusion and robot-point screening

The online tools default to:

- `--scene-camera-names front side`: fixed cameras only;
- `--pointcloud-voxel-size 0.005`: average overlapping samples in 5 mm voxels;
- `--robot-filter-mode capsules`: test each point against the continuous
  distance to every FK link segment, avoiding the gaps in the old sparse-link
  sampler; and
- `--robot-filter-margin 0.010`: an extra 10 mm for model, calibration, and
  depth uncertainty.

The seven default capsule radii are conservative initial values for the
UR7e links and gripper. They are not a replacement for your actual end-effector
geometry: measure the installed gripper/tool, then supply all values with
`--robot-link-radii BASE SHOULDER UPPER FOREARM WRIST1 WRIST2 GRIPPER`.

For the final production-grade mask, replace/augment the capsule filter with a
URDF collision-mesh depth render in each camera before fusion. The official
`ur_description` package provides UR7e link meshes; add your gripper/tool mesh
and robot-specific kinematics calibration. The capsule filter is the complete,
dependency-free runtime fallback included in this repository and is safer than
the previous sparse-point radius test, but it cannot model cables or unmodelled
tooling exactly.

### 3.1 PiKA-ready assets and mesh-depth interface

The repository already contains the preparation that does not require a ChArUco
board:

```text
assets/robot_models/ur_description/                 official UR7e description
assets/robot_models/pika_gripper/collision/          PiKA collision STL exports
assets/robot_models/ur7e_pika/                       Xacro wrapper and config template
```

The PiKA STEP source unit is millimetres. Every PiKA mesh must therefore use
the URDF scale `0.001 0.001 0.001`. Before live use, copy
`assets/robot_models/ur7e_pika/ur7e_pika_mask.example.json` to a non-example
configuration, fill the measured `flange_to_pika_step_frame`, and replace the
camera calibration placeholder. The preflight check is deliberately split:

```bash
# Works now: validates source meshes and PiKA state convention.
python real_scripts/validate_ur7e_pika_mask_config.py \
  assets/robot_models/ur7e_pika/ur7e_pika_mask.example.json

# Run only after measuring PiKA mounting and camera-to-base transforms.
python real_scripts/validate_ur7e_pika_mask_config.py \
  assets/robot_models/ur7e_pika/ur7e_pika_mask.json --ready-for-live-mask
```

Normalize messages from PiKA `/gripper/data` or `/gripper/joint_states` with
`PikaGripperState` in `real_scripts/pika_gripper_state.py`. Its opening range
is checked as 0–0.095 m, and `nearest_pika_state` rejects a state more than
20 ms from an RGB-D timestamp by default.

For each camera, render **only** the robot collision geometry into an
RGB-camera depth image. Pass that render to `fuse_rgbd_frames(...,
rendered_robot_depths={"front": front_render, "side": side_render})`. The
included `robot_depth_keep_mask` removes a measurement only when it agrees with
the rendered robot surface (default: `8 mm + 1%` depth tolerance and 1-pixel
dilation). It retains an object physically in front of the gripper, unlike a
silhouette-only mask.

### 3.2 First-day procedure after the board arrives

1. Capture D435i colour intrinsics and board frames, then produce the camera
   calibration JSON using section 1.
2. Measure `^flange T_pika_step_frame` with PiKA installed; also replace the
   generic UR7e kinematics YAML with calibration extracted from this controller.
3. At fully open and fully closed PiKA positions, verify the two candidate
   finger meshes and encode their motion. Keep the full PiKA mesh fixed until
   this check is complete.
4. Render the assembled collision model for front and side, call the mesh-depth
   filter before projection/fusion, then run the passive overlay in section 4.
5. Check a held object and an empty gripper from both cameras. Raise depth
   tolerance only from measured residuals; do not delete every pixel behind a
   robot silhouette.

## 4. Passive real-time verification

The following command only reads robot/camera state, repairs depth, fuses the
two scene cameras, removes the robot, and writes an overlay video:

```bash
python real_scripts/demo_record_ur7e_safety_overlay_video.py \
  --adapter real_scripts.ur7e_realsense_adapter:create_adapter \
  --camera-calibration real_scripts/ur7e_d435i_camera_calibration.json \
  --lingbot-depth --lingbot-device cuda:0 \
  --scene-camera-names front side \
  --workspace-bounds -0.8 0.8 -0.8 0.8 -0.05 0.8 \
  --table-z 0.0 \
  --debug-npz outputs/lingbot_depth_debug.npz \
  --output outputs/lingbot_depth_two_view_overlay.mp4
```

Cyan points in the overlay are FK link samples used for visualization; orange
points are the post-filter environment cloud; green boxes are tabletop OBBs.
Review the saved video and debug NPZ before connecting this geometry to an
automatic safety decision.

## 5. PI05 collection with the same geometry

Start the prefix-token policy server from OpenPI, then run:

```bash
python real_scripts/collect_pi05_real_safety_dataset.py \
  --prompt "pick up the block" \
  --adapter real_scripts.ur7e_realsense_adapter:create_adapter \
  --camera-calibration real_scripts/ur7e_d435i_camera_calibration.json \
  --lingbot-depth --lingbot-device cuda:0 \
  --scene-camera-names front side \
  --debug-pointcloud-output outputs/pi05_safety_decoder/real_geometry_debug.npz \
  --output outputs/pi05_safety_decoder/pi05_real_ur_safety_dataset.npz
```

The PI05 observation still contains `front`, `side`, and `wrist` RGB images;
only the safety geometry chain excludes wrist. Supplying different camera name
lists to LingBot and fusion is rejected deliberately, so a repaired/fused scene
cannot silently omit or mix a view.
