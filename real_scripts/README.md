# 真实 UR SafetyModule 采集与调试

本目录包含 LIBERO safety dataset collector 的真机版本。默认硬件适配器面向
UR7e/UR e-Series 机械臂，以及 Intel RealSense D435i 深度相机。采集脚本默认
仍支持 `front`、`side`、`wrist` 三个相机；当前真机 demo 推荐只使用 `front`
和 `wrist`。

先在 OpenPI 环境中启动 PI05 prefix policy server：

```bash
uv run --project openpi scripts/serve_pi05_prefix_policy.py \
  --policy-config pi05_ur7 \
  --checkpoint-dir /path/to/pi05_ur7_checkpoint \
  --port 8000
```

然后使用硬件适配器运行采集脚本：

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

适配器需要提供 RGB-D 帧、UR 六个关节角、夹爪状态和动作执行接口。对于
`real_scripts.ur7e_realsense_adapter:create_adapter`，每个 PI05 动作会作为
`[dx_mm, dy_mm, dz_mm, droll, dpitch, dyaw, g]` 传给
`UR7eVectorController.send_ee_delta_vector()`。其中 xyz 单位是毫米，位于夹爪局部
坐标系；旋转单位是弧度；`g` 是 `[0, 1]` 范围内的绝对夹爪命令。采集脚本会使用
UR FK 生成训练 SafetyModule 所需的固定拓扑字段：`current_link_points`、
`target_link_points`、`arm_points` 和 `target_point_offsets`。

## 无标定 RealSense 点云重建

在调试 robot/world 外参前，建议先验证每个 RealSense 深度流本身是否正常。下面
的脚本直接在相机坐标系中重建点云，不需要 OpenCV 或标定板：

```bat
python real_scripts\reconstruct_realsense_pointcloud.py ^
  --serial 348522070576 ^
  --camera-name front ^
  --width 640 ^
  --height 480 ^
  --fps 30 ^
  --stride 2 ^
  --max-depth 3.0 ^
  --depth-vis-max 4.0 ^
  --viewer-max-points 60000 ^
  --tabletop-bounds -0.65 0.65 -0.05 0.18 0.45 1.45 ^
  --table-plane-threshold 0.015 ^
  --min-plane-distance 0.03 ^
  --max-plane-distance 0.30 ^
  --obb-cluster-radius 0.08 ^
  --obb-min-cluster-points 32 ^
  --use-rtde-qpos ^
  --robot-ip 169.254.26.10 ^
  --camera-calibration real_scripts\ur7e_d435i_camera_calibration.generated.json ^
  --robot-filter-radius 0.05 ^
  --robot-points-per-link 128 ^
  --output-dir outputs\realsense_pointcloud
```

脚本会写出以下文件：

- `front_pointcloud.npz`：相机坐标系下的 `points`、`colors`、`intrinsics` 和 `depth_m`。
- `front_pointcloud.ply`：彩色点云，可用 CloudCompare、MeshLab 或 Open3D 打开。
- `front_pointcloud_viewer.html`：离线交互式 3D 点云查看器，可在浏览器中拖动旋转。
- `front_rgb.png`：采集到的 RGB 图。
- `front_depth_vis.png`：带米制 colorbar 的彩色深度图。
- `front_topdown_camera_points.png`：相机坐标系 X/Z 俯视预览图。
- `front_tabletop_pointcloud_viewer.html`：相机坐标 ROI 裁剪后的桌面区域交互点云。
- `front_tabletop_obbs_viewer.html`：剔除桌面平面后的障碍物点云，并叠加绿色 OBB 线框。
- `front_tabletop_obbs.json`：相机坐标系下的 OBB 中心、旋转、尺寸、角点和估计桌面平面。
- `front_robot_observed_points_viewer.html`：RealSense 实际看到且靠近 FK UR7e 模型的机械臂表面点。
- `front_non_robot_points_viewer.html`：剔除可见机械臂点后的非机械臂点云。
- `front_fk_robot_model_points_viewer.html`：转换到相机坐标系下的完整 FK UR7e 模型点。

如果要独立检查腕部相机，可改用
`--serial wrist_camera_serial --camera-name wrist`。这些点云都在相机坐标系中，
不是 UR base/world 坐标系。建议先调 `--tabletop-bounds`，只保留可见桌面区域
并去掉远处背景。在 RealSense 相机坐标系里，`x` 是水平方向，`y` 是图像向下，
`z` 是深度，单位为米。

机械臂筛选建议使用 `--use-rtde-qpos`，它会通过
`UR7eVectorController.get_current_joints()` 从 RTDE 直接读取当前六个 UR 关节角。
也可以设置 `UR_ROBOT_IP`，从而省略 `--robot-ip`。机械臂筛选依赖真实
`camera_to_world` 外参；如果 `front_fk_robot_model_points_viewer.html` 中的 FK 模型
和实际机械臂点云没有重合，应先修正相机外参，再调 `--robot-filter-radius`。

## ChArUco 标定板标定

运行标定脚本前需要安装带 ArUco/ChArUco 支持的 OpenCV：

```bat
pip install opencv-contrib-python
```

然后用 RealSense 拍摄 ChArUco 标定板：

```bat
python real_scripts\calibrate_realsense_charuco.py ^
  --serial front_camera_serial ^
  --camera-name front ^
  --square-length-m 0.035 ^
  --marker-length-m 0.026 ^
  --squares-x 7 ^
  --squares-y 5 ^
  --samples 30 ^
  --output real_scripts\ur7e_d435i_camera_calibration.generated.json
```

该脚本会写出 demo 可用的 JSON，其中 `camera_to_world` 的 `world` 是 ChArUco
标定板坐标系。如果希望使用 UR base 作为 world，需要把标定板放置或测量到 UR base
坐标系中，然后组合变换：

```text
camera_to_ur_base = board_to_ur_base @ camera_to_board
```

对于腕部相机，机械臂运动时不能使用固定的 `camera_to_world`。应先估计固定的手眼
变换 `camera_to_tool`，然后在每一帧根据当前关节角计算：

```text
camera_to_world = tool_to_world(qpos) @ camera_to_tool
```

## UR7e overlay demo

overlay demo 用于验证真机实时几何流程，默认只使用 `front` 和 `wrist` 两个 RGB-D
相机。渲染视角始终使用 `front` 相机 RGB 图。青色点表示 UR7e FK 表面点，橙色点
表示过滤机械臂后的桌面障碍物点，绿色线框表示从障碍物聚类估计出的竖直 OBB。

Windows `cmd` 环境变量：

```bat
set UR_ROBOT_IP=169.254.26.10
set REAL_SENSE_FRONT_SERIAL=front_camera_serial
set REAL_SENSE_WRIST_SERIAL=wrist_camera_serial
set REAL_SENSE_WAIT_TIMEOUT_MS=10000
set REAL_SENSE_READ_RETRIES=5
```

采集一张 front 视角 overlay 图片，不移动机械臂：

```bat
python real_scripts\demo_record_ur7e_safety_overlay_video.py ^
  --output-mode image ^
  --adapter real_scripts.ur7e_realsense_adapter:create_adapter ^
  --camera-calibration real_scripts\ur7e_d435i_camera_calibration.example.json ^
  --workspace-bounds -0.8 0.8 -0.8 0.8 -0.05 0.8 ^
  --table-z 0.0 ^
  --debug-image-dir outputs\debug_overlay ^
  --output outputs\real_ur_safety_overlay_demo.png
```

设置 `--debug-image-dir` 后，demo 会额外写出中间证据文件：

- `front_rgb.png`：front 相机原始 RGB 图。
- `overlay_full.png`：RGB 图上叠加机械臂点、障碍物点和 OBB。
- `overlay_robot_only.png`：只叠加 FK 机械臂点。
- `overlay_obstacles_only.png`：只叠加桌面障碍物点。
- `overlay_obbs_only.png`：只叠加 OBB 线框。
- `topdown_scene_points.png`：融合 RGB-D 点云的 world XY 俯视图。
- `topdown_robot_points.png`：FK 机械臂点的 world XY 俯视图。
- `topdown_obstacle_points.png`：筛选出的桌面障碍物点 world XY 俯视图。
- `debug_summary.json`：点数、投影点数和 OBB 数量统计。

录制视频并让机械臂执行小幅确定性随机末端增量动作：

```bat
python real_scripts\demo_record_ur7e_safety_overlay_video.py ^
  --output-mode video ^
  --adapter real_scripts.ur7e_realsense_adapter:create_adapter ^
  --camera-calibration real_scripts\ur7e_d435i_camera_calibration.example.json ^
  --workspace-bounds -0.8 0.8 -0.8 0.8 -0.05 0.8 ^
  --table-z 0.0 ^
  --duration-sec 8 ^
  --random-action-count 8 ^
  --random-xyz-mm 10 ^
  --random-rot 0.03 ^
  --random-seed 0 ^
  --output outputs\real_ur_safety_overlay_demo.mp4
```

如需更换相机组合，传入 `--camera-names`，例如 `--camera-names front wrist`。
如需只录制视频而不让机械臂执行 demo 动作，传入 `--no-demo-actions`。如需覆盖随机
视频动作，可传入一个或多个 7D 动作：

```bat
python real_scripts\demo_record_ur7e_safety_overlay_video.py ^
  --output-mode video ^
  --adapter real_scripts.ur7e_realsense_adapter:create_adapter ^
  --camera-calibration real_scripts\ur7e_d435i_camera_calibration.example.json ^
  --demo-action 10 0 0 0 0 0 0.5 ^
  --demo-action -10 0 0 0 0 0 0.5
```

根据真实桌面和相机位置调节 `--table-z`、`--min-obstacle-height`、
`--max-obstacle-height`、`--cluster-radius` 和 `--robot-filter-radius`。

示例标定 JSON 中的数值只是占位符。真正用于点云、OBB 或安全决策前，必须替换为
每个 D435i 的真实内参矩阵和 `camera_to_world` 外参。
