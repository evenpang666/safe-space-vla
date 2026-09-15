# Robot geometry assets

当前主线使用：

- `ur_description/meshes/ur5e/collision/`：官方 UR e-Series collision mesh；
- `vendor/ros2_robotiq_gripper/`：左臂 Robotiq 2F-85 上游 collision mesh；
- `vendor/ros2_epick_gripper/`：右臂 Robotiq EPick 上游 body mesh；
- `robotiq_2f85/urdf/` 与 `robotiq_epick/urdf/`：安装 adapter、吸盘等固定
  collision primitive，用于补足 vendor mesh 未覆盖部分。

运行时几何由 `real_scripts/dual_ur7e_surface.py` 与
`real_scripts/live_dual_ur7e_obstacle_model.py` 读取。所有安全几何必须使用实测
`flange→active_tcp`，并统一到 `left_base`。

`pika_gripper/` 和 `ur7e_pika/` 是早期单臂实验留下的资产，不属于当前
2F-85/EPick 双臂部署逻辑；保留它们仅用于结果复现，不应被当前入口加载。
