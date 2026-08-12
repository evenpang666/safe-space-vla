# UR7e + Robotiq 2F-85 + 双 D435i RGB-D 数据采集

本目录提供一个已接入当前 `safe-space-vla` 项目的数据采集程序，用 PikaSense 遥操作 UR7e 和 Robotiq 2F-85，并同步保存 Intel RealSense D435i 的 RGB-D 数据。默认配置使用 `front` / `wrist` 两路相机；如需增加 `side`，在配置文件的 `cameras` 下添加同名相机即可，写盘目录会自动跟随配置。

## 功能

- PikaSense 控制 UR7e 末端位姿。
- PikaSense 编码器控制 Robotiq 2F-85 开合。
- `F2` 开始/停止机器人遥操作控制。
- `F3` 开始/停止录制。录制前需要先按 `F2` 进入遥操作状态。
- 每帧保存：
  - `front` / `wrist` 两路 RGB 图像；
  - `front` / `wrist` 两路 depth 图像；
  - UR7e 六个关节角和 Robotiq 实际位置；
  - action：目标关节角和 Robotiq 目标指令；
  - 任务描述 `task`。

## 安装

建议在项目根目录中创建独立 Python 虚拟环境：

```bash
cd safe-space-vla
python -m venv .venv-collector
source .venv-collector/bin/activate
pip install -r real_scripts/ur7e_robotiq_d435i_collector/requirements.txt
```

如果启动时看到 `module 'serial' has no attribute 'Serial'` 或
`module 'serial' has no attribute 'SerialException'`，说明当前环境导入的不是
`pyserial`。在同一个 `(safety)` 环境里执行：

```bash
python -m pip uninstall serial
python -m pip install pyserial
```

本采集脚本使用 Pika Vive tracker 位姿输入来遥操作 UR7e，因此还需要按原项目文档安装和配置 `libsurvive` / `pysurvive`。本目录下的 `pika_sdk/README.md` 和 `pika_sdk/Pika SDK API 文档.md` 也说明了 Vive Tracker 功能依赖 `pysurvive`。如果启动时看到 `Missing dependency 'pysurvive'` 或 `初始化Vive Tracker失败: 未找到pysurvive库`，需要先修复 Vive tracker 环境；否则 PikaSense 编码器可能能读到，但机械臂不会获得 Pika 位姿输入。

快速检查当前 Python 环境是否能导入 Vive tracker 依赖：

```bash
python -c "import pysurvive; print('pysurvive ok')"
```

若失败，在同一个 `(safety)` 环境中安装：

```bash
python -m pip install pysurvive
```

如果 `pip install pysurvive` 编译或运行失败，先按 `pika_sdk/README.md` 安装 `libsurvive`，并确认 Vive 基站 / Tracker 可被系统识别后再运行采集脚本。

## 配置

编辑：

```bash
real_scripts/ur7e_robotiq_d435i_collector/configs/ur7e_robotiq_d435i.yaml
```

采集前至少确认这些字段：

- `robot.host`：UR7e 控制器 IP，例如 `192.168.1.10`。
- `pika_sense.port`：PikaSense 串口；为空时程序会尝试自动检测。Windows 上也可以显式填写 `COM7` 这类端口。
- `cameras.front.serial`：前视 D435i 序列号。
- `cameras.wrist.serial`：腕部 D435i 序列号。
- `collection.task`：默认任务描述；也可以运行时用 `--task` 覆盖。

常用的其他字段：

- `collection.output_dir`：数据集输出根目录，默认 `datasets`。
- `collection.dataset_name`：默认数据集名，可被 `--dataset-name` 覆盖。
- `collection.fps`：采集帧率，默认 `30`。
- `robotiq_gripper.port`：Robotiq URCap socket 端口，默认 `63352`。

## 运行

在项目根目录中运行：

```bash
python -m real_scripts.ur7e_robotiq_d435i_collector.collect_rgbd_pika_robotiq \
  --config real_scripts/ur7e_robotiq_d435i_collector/configs/ur7e_robotiq_d435i.yaml \
  --task "pick up the red block and place it in the tray" \
  --dataset-name pick_red_block
```

程序启动后会依次连接 UR7e、PikaSense、Robotiq 和两台 D435i。终端打印 `Ready. F2 toggles teleop; F3 toggles recording.` 后即可操作。

## 按键

- `F2`：切换遥操作。第一次按下后 PikaSense 开始控制 UR7e，PikaSense 编码器开始控制 Robotiq 2F-85；再次按下停止遥操作。
- `F3`：切换录制。需要先按 `F2`。第一次按下开始新 episode；再次按下保存当前 episode。
- `Ctrl+C`：退出程序。若此时有未保存的活跃 episode，程序会丢弃该 episode 并清理硬件连接。

## 输出

默认输出到：

```text
datasets/<dataset-name>/
```

每次保存会生成一个 episode：

```text
datasets/pick_red_block/
└── episode_000000/
    ├── meta.json
    ├── frames.parquet
    ├── rgb/
    │   ├── front/
    │   │   ├── 000000.png
    │   │   └── 000001.png
    │   └── wrist/
    │       ├── 000000.png
    │       └── 000001.png
    └── depth/
        ├── front/
        │   ├── 000000.png
        │   └── 000001.png
        └── wrist/
            ├── 000000.png
            └── 000001.png
```

其中：

- `meta.json`：数据集名、episode 编号、任务描述、帧率、机器人 IP、相机配置、state/action 字段名和 depth 编码说明。
- `frames.parquet`：每帧索引、时间戳、任务描述、`observation.state`、`action`、关节角、夹爪状态以及对应 RGB/depth 文件相对路径。
- `rgb/front` 和 `rgb/wrist`：两路彩色图。
- `depth/front` 和 `depth/wrist`：两路 `uint16` depth PNG，编码为 RealSense 原始 Z16，并按配置对齐到 color。

## 烟测

这些检查按从轻到重的顺序执行。完整硬件运行前，UR7e 必须处于 remote mode，且配置中的机器人 IP、PikaSense 串口、两台 D435i 序列号都必须正确。

### 1. Python import 检查

```bash
python - <<'PY'
from real_scripts.ur7e_robotiq_d435i_collector.collect_rgbd_pika_robotiq import load_config, resolve_task
from real_scripts.ur7e_robotiq_d435i_collector.utils.camera_rgbd import MultiRGBDCamera
from real_scripts.ur7e_robotiq_d435i_collector.utils.episode_writer import EpisodeWriter
from real_scripts.ur7e_robotiq_d435i_collector.utils.gripper_adapters import RobotiqGripperAdapter
from real_scripts.ur7e_robotiq_d435i_collector.utils.pika_interface import PikaSense
from real_scripts.ur7e_robotiq_d435i_collector.utils.robot_interface import UR7eInterface

cfg = load_config("real_scripts/ur7e_robotiq_d435i_collector/configs/ur7e_robotiq_d435i.yaml")
print("imports ok")
print("task:", resolve_task("smoke test task", cfg))
PY
```

预期看到 `imports ok`，且没有 import error。

### 2. RealSense 序列号枚举

```bash
python - <<'PY'
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
if not devices:
    raise SystemExit("no RealSense devices found")

for dev in devices:
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
    print(f"{name}: serial={serial}, usb={usb_type}")
PY
```

确认输出里有两台 D435i，并把对应序列号填到 `cameras.front.serial` 和 `cameras.wrist.serial`。

### 3. Robotiq 夹爪 sweep，机械臂不移动

当前仓库没有原项目的 `collect/test_pikasense_robotiq_gripper.py`。如果需要在完整采集前单独验证夹爪，请使用本地已有的 Robotiq socket 测试脚本，或在交互式 Python 中只实例化 `RobotiqGripperAdapter` 做小幅开合 sweep；不要同时发送 UR7e 运动指令。

### 4. 完整硬件短采集

确认 UR remote mode、Robotiq URCap 程序已运行、PikaSense / Vive tracker 正常、两台 D435i 使用 USB 3 连接后，再运行一次短采集：

```bash
python -m real_scripts.ur7e_robotiq_d435i_collector.collect_rgbd_pika_robotiq \
  --config real_scripts/ur7e_robotiq_d435i_collector/configs/ur7e_robotiq_d435i.yaml \
  --task "hardware smoke test" \
  --dataset-name hardware_smoke_test
```

操作顺序：

1. 按 `F2` 进入遥操作。
2. 按 `F3` 录制 3 到 5 秒。
3. 再按 `F3` 保存。
4. 再按 `F2` 停止遥操作。
5. 用 `Ctrl+C` 退出。

检查 `datasets/hardware_smoke_test/episode_000000/` 下是否有 `meta.json`、`frames.parquet`、`rgb/front`、`rgb/wrist`、`depth/front`、`depth/wrist`。
