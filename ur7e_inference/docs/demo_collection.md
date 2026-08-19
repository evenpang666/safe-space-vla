# Pika Sensor 数据采集

采集器当前生成独立文件 episode：两路 RGB 图像、6 维实际关节角、实际夹爪状态和 7 维绝对 action（6 轴目标关节角与夹爪目标）。暂不写入 LeRobot；可在离线预处理阶段转换。

## 数据与夹爪约定

夹爪数据统一为 `0=张开、1=闭合`。实时采集同时启用 `demo.gripper_invert` 和 `gripper.invert`：手柄到实体夹爪的动作方向保持与旧采集流程一致，仅写入数据集的数值改为此约定。

```yaml
gripper:
  policy_min: 0.0
  policy_max: 1.0
  invert: true

demo:
  gripper_invert: true
```

不要将旧的 `0=闭合、1=张开` episode 与新 episode 混合训练。

## 配置

编辑 `config.yaml` 的 `demo`：

- `output_dir`：停止录制后保存的原有 PNG/NPY episode 根目录。为安全点云预处理新增对齐的前视深度、夹爪开口和标定副本；原有 RGB、关节、动作与时间戳文件名不变。
- `pika_sense_port` 和 `gripper.serial_port`：默认 `auto`。程序只读取串口遥测来区分 PiKA Sense（`AS5047`）和 PiKA Gripper（`motor`），不会在发现阶段下发电机命令。不要在 Linux 配置 `COM6`；如需固定设备，请使用 `/dev/serial/by-id/...`。
- `tracker_connect_attempts` 和 `tracker_reconnect_backoff_s`：Tracker 启动超时时，释放当前 libsurvive 上下文并自动重试的次数和等待时间。默认执行两次干净启动；若输出仍为 `LH0/LH1` 而没有手持设备，检查 Tracker 供电/配对和 USB 接收器，而不是继续重连 UR。
- `cameras.wrist_device`：默认 `auto`，按 PiKA Gripper 串口与 DECXIN 鱼眼相机的 USB 拓扑选择腕部相机；`realsense_serial` 保持为原 front D435i。若两个 PiKA 鱼眼相机无法由 USB 拓扑区分，程序会停止并列出候选 `/dev/video*`，此时固定为对应的 `/dev/v4l/by-path/...`。
- `gripper_distance_min_mm/max_mm`、`translation_scale`、`rotation_scale`：按当前设备标定。
- `sensor_to_tool_rpy`：Sensor 到工具坐标的固定旋转，默认值与 `RobotControl` 一致。
- `max_translation_m`、`max_rotation_rad`、`max_ik_joint_step_rad`：真机机械臂安全边界。实时 IK 目标按 `min(max_ik_joint_step_rad, robot.max_joint_step_rad, robot.max_joint_speed_rad_s / demo.fps)` 限速；超出时会分多帧安全逼近，而不是中止采集。Pika 夹爪则与推理端一致，直接跟随 Sensor 的位置指令，不使用 `max_gripper_step` 限速。
- `gripper.max_angle_rad`：Pika 夹爪电机通常使用 `0.0`（闭合）至约 `1.7 rad`（张开）；使用 `1.0` 会截断近 40% 行程。实时手部开合也按 `max_gripper_step` 逐帧限速。
- `gripper_closed_rad`、`gripper_open_rad`：Pika Sense 的 AS5047 原始编码器范围。实时遥操直接以该弧度范围映射到 Pika 电机（与 RobotControl 一致），不使用非线性的估算毫米距离；默认 `0.0` 为闭合、`1.7` 为张开。
- `tracker_position_deadband_m`、`tracker_orientation_deadband_rad`：连续两帧内低于这些值的 Lighthouse 微抖会保持上一安全目标，避免静止 Sensor 触发 IK 分支跳变。默认分别为 2 mm 和约 1.15 度；不要用增大 `max_ik_joint_step_rad` 来掩盖持续跳变。

实时采集以开始时的 Sensor 位姿和 UR TCP 位姿建立相对锚点，不需要绝对 Lighthouse 到 UR 的手眼标定。`calibrate-demo` 仅保留给旧的离线轨迹回放工具。

## 与 VLA 推理 GUI 的区别

`ur7e-vla collect-demo` 的 Gradio Web 页面用于 Pika Sensor 遥操和独立文件 episode 采集；
`ur7e-vla vla-gui` 用于执行远程 OpenPI/VLA 策略。后者可实时更新任务文本、切换同步/
异步推理、独立录制双相机视频，并通过 `Restore initial state` 选项配合 `Apply Task` 执行安全恢复；
它不会生成示教数据集 episode。

采集 Web 页面常驻运行，不直接持有 UR/PiKA 硬件。输入任务描述（`--task` 仅用于预填）后，点击 **Start Teleoperation** 会直接启动独立命令 `ur7e-vla collect-demo-worker --config … --task … --execute --wait-for-record`，由该命令连接硬件并开始遥操，但不会打开相机或写入数据。此时可恢复机器人状态；遥操 READY 后，点击 **Start Recording** 才会开始写入 episode。最后点击 **Stop Teleoperation & Save**，worker 正常停止；若已经开始录制则保存当前 episode 后退出。若 worker 崩溃，Web 页面仍保留，状态显示退出原因后可直接再次点击 Start 启动一个干净的新 worker。

每次正常停止都会创建新的顺序编号 episode。例如任务 `pick cube` 会保存为：

```text
outputs/ur7e_demo_episodes/
└── pick_cube/
    ├── episode_001/
    │   ├── front_rgb/              # 保持原有 PNG 文件
    │   ├── wrist_rgb/              # 保持原有 PNG 文件
    │   ├── front_depth_m/          # 新增，与 front_rgb 按帧名对应的 .npy 深度米图
    │   ├── joints.npy
    │   ├── gripper.npy
    │   ├── gripper_opening_mm.npy  # 新增，N×1，单位 mm
    │   ├── actions.npy
    │   ├── timestamps_s.npy
    │   ├── metadata.json
    │   └── calibration.json        # 新增，标定副本
    └── episode_002/
```

任务名称会规范化为小写下划线形式。录制期间先写入 `demo.staging_dir`；若进程被强制杀死、断电或 Python 崩溃，暂存目录不会自动完成保存，不能交给预处理器。

采集前必须在 `config.yaml` 配置 `demo.safety_calibration_path` 和 `demo.safety_front_calibration_name`。默认配置已使用前视 D435i `405622074939` 与 `outputs/calibration/session_01/camera_calibration.json`。RGB 与深度按相机原始分辨率保存；预处理器会读取实际尺寸，只有当旧 episode 的保存尺寸与标定流尺寸不同才缩放临时内参：

```bash
python ../real_scripts/preprocess_pi05_rgbd_surface_dataset.py \
  --episode-dir outputs/ur7e_demo_episodes/pick_cube/episode_002 \
  --output outputs/pi05_surface/pick_cube_episode_002.npz \
  --pika-mount-transform-json ../outputs/calibration/pika_mount_from_tcp_provisional.json \
  --scene-camera-names front
```

```bash
ur7e-vla collect-demo --config config.yaml --task "pick cube" --execute \
  --server-name 0.0.0.0 --server-port 7860
```

默认页面只监听 `127.0.0.1:7860`，可在同一台主机浏览器打开 `http://127.0.0.1:7860`。没有桌面环境或需从另一台主机访问时，显式绑定局域网接口；运行后在另一台主机打开 `http://<采集主机的局域网-IP>:7860`：

```bash
ur7e-vla collect-demo --config config.yaml --task "pick cube" --execute \
  --server-name 0.0.0.0 --server-port 7860
```

页面不再提供登录认证。请只在受信任的隔离局域网使用；如通过 SSH 隧道访问则保留默认 `127.0.0.1` 绑定即可。首次部署时，如当前环境未装 Gradio：

```bash
python -m pip install 'gradio>=6.24,<7' 'socksio>=1.0'
```

Web 操作：

1. 输入任务，点击 **Start Teleoperation**。启动前请自行确认工作区已清空、急停可达且示教器为 Remote Control。
2. 等待状态显示 READY。此阶段可以移动 Pika Sensor 遥控 UR7e、恢复初始状态；不会采集相机或写入 episode。
3. 准备好后点击 **Start Recording**。开合 Sensor 会同步控制实体夹爪，并按 `demo.fps` 采集帧。
4. 点击 **Stop Teleoperation & Save** 结束并自动保存 episode；等待状态显示 worker 已停止后，再开始下一段录制。
5. 若 worker 报错或退出，页面不会退出；修正硬件问题后再次点击“Start Teleoperation”即可重新运行采集命令。

通常由 Web 页面启动 worker；若需在没有网页时诊断采集链路，也可以直接运行：

```bash
ur7e-vla collect-demo-worker --config config.yaml --task "pick cube" --execute
```

采集会在 Tracker 数据陈旧、工作区越界、逆解失败、关节越限、目标跳变或相机异常时停止。它不包含碰撞规划；首次使用应缩小动作范围并保持急停可达。
