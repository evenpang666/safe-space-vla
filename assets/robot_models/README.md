# Robot geometry assets for real-world point-cloud filtering

## Downloaded sources

- `ur_description/`: official Universal Robots ROS 2 description repository,
  shallow-cloned from the `jazzy` branch at
  `39242984dc8d1fff9584c922c17c69c58df3591d`.
  - UR7e configuration: `config/ur7e/`.
  - UR7e's official configuration intentionally references the common e-Series
    geometry in `meshes/ur5e/`; see `config/ur7e/visual_parameters.yaml`.
  - Use the `collision/*.stl` mesh files for depth masking, and the `visual/*.dae`
    files only for visualization.
- `pika_gripper/PiKA-Gripper-STEP.stp`: public PiKA Gripper CAD STEP download.
  - `collision/` contains a 1.5 mm-tessellated collision export, its 16 source
    components, full/body meshes, and per-component STEP-frame bounds.
  - `finger_a_candidate.stl` and `finger_b_candidate.stl` are the two symmetric
    long-finger candidates. Their joint axes and open/close mapping are not
    assumed; verify them against the physical PiKA before moving-finger masking.
- `ur7e_pika/urdf/ur7e_pika_collision.urdf.xacro`: collision-only UR7e + PiKA
  wrapper. Pass the measured PiKA mounting transform and collision-mesh path
  when expanding it with Xacro.
- `ur7e_pika/ur7e_pika_mask.example.json`: ready-to-fill mask configuration.
  Validate its downloaded geometry now with:

  ```bash
  python real_scripts/validate_ur7e_pika_mask_config.py \
    assets/robot_models/ur7e_pika/ur7e_pika_mask.example.json
  ```

## Still required before a mesh-based robot mask can be enabled

1. Record the installed adapter and PiKA flange transform,
   `^flange T_pika_step_frame`. Do not use a nominal transform when an adapter
   plate is present.
2. Verify the two exported finger candidates at fully open and fully closed
   PiKA positions, then encode their real joint axes and state mapping.
3. Replace the default UR7e kinematics YAML with the calibration extracted from
   the individual robot controller.
4. After the D435i-to-base calibration, fill both transforms and run:

  ```bash
  python real_scripts/validate_ur7e_pika_mask_config.py \
    assets/robot_models/ur7e_pika/ur7e_pika_mask.json --ready-for-live-mask
  ```

5. Render the collision URDF in each calibrated D435i camera and remove only
   pixels whose rendered robot depth agrees with measured depth.

Source URLs:

- https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
- https://www.scengrobotics.com/downloads/pika/PiKA-Gripper-STEP.stp
