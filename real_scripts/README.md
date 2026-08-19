# UR7e 实机标定与点云流程

本目录用于固定 RealSense D435i、UR7e 和 ChArUco 板的标定、点云生成与机器人点剔除。

```text
RealSense RGB-D → 深度修正（可选）→ UR base 点云 → 多相机融合（可选）
→ UR7e/PiKA 点剔除 → 环境点云 / OBB
```

## 约定与安全

- 所有长度、深度和位姿均使用米；`camera_to_world` 表示 `^base T_camera`。
- 标定和采集时保持机器人静止，保留急停可达；实机结果必须复核后才能接入安全控制。
- 深度已对齐到彩色图，因此标定和点云投影始终使用对应分辨率的彩色内参。
- 多相机 USB 帧并非硬件同步；仅用于静态场景或低速场景。高速测量请使用硬件同步或时间戳筛选。

运行环境需要 `pyrealsense2`、`ur-rtde`、`opencv-contrib-python`（包含 `cv2.aruco`）、`trimesh`，以及可选的 LingBot-Depth 依赖。

```bash
cd /path/to/safety-module
python -m pip install opencv-contrib-python ur-rtde trimesh
```

## 快速开始

### 1. 一体化标定

先在 PolyScope 中完成实际探针的 TCP 工具偏置标定；本脚本用该已校准探针建立 ChArUco 板到 UR base 的变换，并不计算新的法兰到工具 TCP 偏置。

将 ChArUco 板固定在所有相机都可见的位置，记录其真实参数后运行：

```bash
export UR_ROBOT_IP=169.254.175.10

python real_scripts/calibrate_ur7e_realsense_integrated.py \
  --serials 405622074939 348522070576 \
  --robot-ip "$UR_ROBOT_IP" \
  --output-dir outputs/calibration/session_01 \
  --squares-x 10 --squares-y 7 \
  --square-length-m 0.037 \
  --marker-length-m 0.027 \
  --dictionary DICT_5X5_100
```

脚本依次执行：

1. 读取所选 RealSense 彩色流内参与畸变参数；
2. 启动全部相机，检查 ChArUco 可见性、四个指定内角点和重投影误差；
3. 计算每台相机相对 ChArUco 板的外参；
4. 提示操作者依次触碰四个内角点，回车后读取 TCP 的 UR base 坐标；
5. 拟合 `^base T_board`。若四点误差超限，脚本会指出最可疑的角点并只要求重采该点；
6. 写出最终标定文件。

主要输出：

```text
outputs/calibration/session_01/
├── camera_calibration.json            # 后续点云流程使用此文件
├── realsense_color_intrinsics.json
├── camera_extrinsics_charuco_board.json
└── board_base_correspondences.json
```

`camera_calibration.json` 记录相机序列号、标定时的宽高/FPS、`^base T_camera` 和融合模式。不要使用仓库中的 `*.example.json` 做实机安全决策。

### 2. 生成点云

```bash
python real_scripts/capture_fuse_separate_ur7e_live.py \
  --robot-ip "$UR_ROBOT_IP" \
  --calibration outputs/calibration/session_01/camera_calibration.json \
  --output-dir outputs/ur7e_live_scene_1 \
  --pika-max-opening-mm 110
```

脚本会从标定文件自动读取相机序列号与标定流配置：

- 标定文件只有一台相机：直接生成该相机的 UR-base 点云，不融合。
- 标定文件有多台相机：采集所有已标定相机，执行原有的体素融合。

之后的 LingBot 深度修正、UR7e/PiKA 深度与体积剔除、环境点云导出逻辑相同。结果包括：

```text
outputs/ur7e_live_scene/
├── lingbot_fused_scene.ply
├── lingbot_fused_scene_viewer.html
├── environment_without_ur7e.ply
├── *_urdf_removed_overlay.png
└── summary.json                         # 含 point_cloud_mode 和 fusion_enabled
```

旧标定文件若不含序列号，可补充映射：

```bash
python real_scripts/capture_fuse_separate_ur7e_live.py \
  --calibration real_scripts/ur7e_d435i_camera_calibration.json \
  --camera-serial front=405622074939 \
  --camera-serial side=348522070576
```

## PI05 联合训练数据预处理

原始 PI05 episode 由 `UR7eSafetyEpisodeRecorder` 保存 RGB、深度、关节、夹爪、动作和时间戳。深度不能直接作为未来点流标签：可见的机器人点会因遮挡而改变数量和顺序。请离线运行下面的预处理器；它利用标定和深度重建机器人观测点用于质量审计，但输出只保存 RGB、状态、动作和有固定 ID 的碰撞面点集。

```bash
python real_scripts/preprocess_pi05_rgbd_surface_dataset.py \
  --episode-dir outputs/raw_episode_001 \
  --output outputs/pi05_surface/episode_001.npz \
  --pika-mount-transform-json <measured-flange-to-pika.json> \
  --scene-camera-names front side \
  --scene-camera-map front=123456789012 \
  --scene-camera-map side=234567890123
```

如果 episode 相机名与标定文件键相同，可省略 `--scene-camera-map`。输出不含深度数组，包含：

- `rgb_*`、`task_text`、`qpos`、`gripper_state`、`actions`；
- `fixed_link_points[T, L, P, 3]` 与稳定的 `point_ids[L, P, 2]`；
- `current_link_points[N, K, 3]`、`action_chunks[N, H, 7]` 和 `target_point_offsets[N, H, K, 3]`；
- `observed_robot_point_counts[T, C]` 与时间戳偏差，用于拒绝标定差或遮挡严重的 episode。

### 两种机器人表面点目标

预处理器保留两种互补的点表示，不能把它们混为同一个监督目标：

| 表示 | 核心字段 | 点的来源 | 适用场景 |
| --- | --- | --- | --- |
| 模型碰撞网格固定点 | `fixed_link_points`、`point_ids` | UR7e/PiKA collision mesh 的确定性表面采样 + 每帧 FK | 安全几何、保守碰撞、关节条件、无缺失的确定性点流 |
| PointWorld 真实表面点 | `visual_robot_tracks`、`visual_robot_visible_mask` | 跨帧 RGB 轨迹 + 实测深度反投影；模型仅作门控 | 真实外观/执行残差、附着物、标定偏差和可见表面点流监督 |

#### 模型碰撞网格固定点（保留的默认安全预处理）

这是默认且必须保留的安全几何路径。每个 UR7e collision link 与 PiKA
表面各采样 `--points-per-link` 个确定性点；`point_ids[link_id, point_id]`
在所有帧恒定，空间位置随实测关节角和夹爪开度更新。相机深度只用于
`observed_robot_point_counts` 审计，**不会**改变、删除或补充
`fixed_link_points`。

双视角 legacy demo 的命令如下（只有 front RGB-D 时删除 `side` 和对应的
`--scene-camera-map`）：

```bash
conda run -n safety python real_scripts/preprocess_pi05_rgbd_surface_dataset.py \
  --episode-dir ur7e_inference/outputs/ur7e_demo_episodes/pick_cube/episode_003 \
  --output outputs/ur7e_demo_surface/pick_cube_episode_003_fixed_surface_dual_view.npz \
  --pika-mount-transform-json outputs/calibration/pika_mount_from_tcp_provisional.json \
  --scene-camera-names front side \
  --scene-camera-map side=348522070576 \
  --points-per-link 128 --future-horizon 8
```

该输出包含 `fixed_link_points[T, L, P, 3]`、`point_ids[L, P, 2]`、
`local_link_points`、`target_point_offsets` 和 `future_link_offsets`。可用
下面的脚本检查模型点在相机上的投影；传入 `--side-camera-name` 时为三栏
front/side/UR-base 视频，省略时为 front/UR-base 视频：

```bash
conda run -n safety python real_scripts/visualize_ur7e_fixed_surface_trajectory.py \
  --surface-npz outputs/ur7e_demo_surface/pick_cube_episode_003_fixed_surface_dual_view.npz \
  --calibration ur7e_inference/outputs/ur7e_demo_episodes/pick_cube/episode_003/calibration.json \
  --camera-name 405622074939 --side-camera-name 348522070576 \
  --output outputs/ur7e_demo_surface/pick_cube_episode_003_fixed_surface_dual_view_trajectory.mp4 \
  --preview outputs/ur7e_demo_surface/pick_cube_episode_003_fixed_surface_dual_view_preview.png \
  --fps 15
```

该视频用于检验 FK、相机标定与 PiKA 安装变换；即使视觉重合，也不能把该
模型点视频当作实测表面点云。

#### PointWorld 真实机器人表面轨迹

##### 推荐流程：PointWorld 式真实 RGB-D 表面点（GPU）

`fixed_link_points` 是 UR7e/PiKA 碰撞网格上固定 ID 的模型点，用于
运动学条件、机器人区域门控和保守碰撞判断；它**不是**相机实测表面。
若训练或分析需要真实机器人表面点，应使用下面的 PointWorld 流程：

```text
首帧真实深度 + FK 深度一致性 → 机器人表面 pixel seeds
→ CoTracker3 跨帧保持 seed ID → 每帧真实深度反投影到 UR base
→ FK 深度一致性门控 → visual_robot_tracks + visible mask
```

模型只参与 seed 的机器人区域筛选和每帧一致性门控；输出的
`visual_robot_tracks` 坐标来自真实 RGB 跟踪像素与真实深度。被遮挡、
深度无效或跟踪漂移的 seed 不会被重排或替换，而是保留原列 ID 并把
`visual_robot_visible_mask` 置为 `False`。

先在 `safety` conda 环境确认 CUDA 可见，再下载一次官方 CoTracker3
checkpoint（后续 episode 复用该文件）：

```bash
conda run -n safety python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'

mkdir -p outputs/ur7e_demo_surface/cotracker
curl -L --fail -o outputs/ur7e_demo_surface/cotracker/cotracker3_scaled_offline.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

以一个同时具有 front/side RGB-D 的 legacy demo episode 为例，先运行
基础 safety 预处理一次。该步骤提供 FK 门控用的机器人表面，但不把它
作为真实点流标签：

```bash
conda run -n safety python real_scripts/preprocess_pi05_rgbd_surface_dataset.py \
  --episode-dir ur7e_inference/outputs/ur7e_demo_episodes/pick_cube/episode_003 \
  --output outputs/ur7e_demo_surface/pick_cube_episode_003_fixed_surface_dual_view.npz \
  --pika-mount-transform-json outputs/calibration/pika_mount_from_tcp_provisional.json \
  --scene-camera-names front side \
  --scene-camera-map side=348522070576 \
  --points-per-link 128 --future-horizon 8
```

随后每个**固定相机**独立生成一套 512 个 stable seed 的 CoTracker3
轨迹。两个相机的 seed ID 是独立集合，不能直接当作同一个物理点：

```bash
TRACKER_CKPT=outputs/ur7e_demo_surface/cotracker/cotracker3_scaled_offline.pth
BASE_SURFACE=outputs/ur7e_demo_surface/pick_cube_episode_003_fixed_surface_dual_view.npz
EPISODE=ur7e_inference/outputs/ur7e_demo_episodes/pick_cube/episode_003

conda run -n safety python real_scripts/generate_cotracker_robot_tracks.py \
  --episode-dir "$EPISODE" --surface-npz "$BASE_SURFACE" \
  --camera front --calibration-camera 405622074939 \
  --output outputs/ur7e_demo_surface/pick_cube_episode_003_front_cotracker_dense_tracks.npz \
  --checkpoint "$TRACKER_CKPT" --tracker cotracker --device cuda \
  --max-seeds 512 --seed-stride 2 --tracking-scale 0.5 --chunk-size 32

conda run -n safety python real_scripts/generate_cotracker_robot_tracks.py \
  --episode-dir "$EPISODE" --surface-npz "$BASE_SURFACE" \
  --camera side --calibration-camera 348522070576 \
  --output outputs/ur7e_demo_surface/pick_cube_episode_003_side_cotracker_dense_tracks.npz \
  --checkpoint "$TRACKER_CKPT" --tracker cotracker --device cuda \
  --max-seeds 512 --seed-stride 2 --tracking-scale 0.5 --chunk-size 32
```

将每一套 2-D 轨迹重新送入预处理器，得到真实深度反投影的 3-D 点流。
由于每个相机各有独立的 stable ID，本版本分别写出两个 NPZ：

```bash
conda run -n safety python real_scripts/preprocess_pi05_rgbd_surface_dataset.py \
  --episode-dir "$EPISODE" \
  --output outputs/ur7e_demo_surface/pick_cube_episode_003_pointworld_dense_front.npz \
  --pika-mount-transform-json outputs/calibration/pika_mount_from_tcp_provisional.json \
  --scene-camera-names front side --scene-camera-map side=348522070576 \
  --robot-tracks outputs/ur7e_demo_surface/pick_cube_episode_003_front_cotracker_dense_tracks.npz \
  --robot-tracks-camera front --points-per-link 128 --future-horizon 8

conda run -n safety python real_scripts/preprocess_pi05_rgbd_surface_dataset.py \
  --episode-dir "$EPISODE" \
  --output outputs/ur7e_demo_surface/pick_cube_episode_003_pointworld_dense_side.npz \
  --pika-mount-transform-json outputs/calibration/pika_mount_from_tcp_provisional.json \
  --scene-camera-names front side --scene-camera-map side=348522070576 \
  --robot-tracks outputs/ur7e_demo_surface/pick_cube_episode_003_side_cotracker_dense_tracks.npz \
  --robot-tracks-camera side --points-per-link 128 --future-horizon 8
```

渲染真实点而不是模型点。传入 `--side-npz` 时得到 front、side、UR base
三栏视频；单相机 episode（例如只有 front RGB-D 的 episode_002）则省略
该参数，生成 front 与 UR base 两栏视频：

```bash
conda run -n safety python real_scripts/visualize_pointworld_robot_tracks.py \
  --front-npz outputs/ur7e_demo_surface/pick_cube_episode_003_pointworld_dense_front.npz \
  --side-npz outputs/ur7e_demo_surface/pick_cube_episode_003_pointworld_dense_side.npz \
  --output outputs/ur7e_demo_surface/pick_cube_episode_003_pointworld_dense_measured_surface_trajectory.mp4 \
  --preview outputs/ur7e_demo_surface/pick_cube_episode_003_pointworld_dense_measured_surface_preview.png \
  --fps 15
```

`--max-seeds` 是每个相机的理论最大 stable point 数；视频中实际显示的点
通常更少，这是正常现象：门控会排除遮挡、无效深度以及与 FK 深度不一致
的跟踪像素。不要以补点、重采样或重排 ID 来填满不可见区域。

若要采用 PointWorld 式的“同一初始像素 seed 在各帧的可见表面对应”，先在**一个固定相机**的全段 RGB 上运行 CoTracker（或任意 2D tracker），再把结果保存成 NPZ：

```text
tracks_xy[T, M, 2]   # 颜色图像像素坐标 (u, v)，第 m 列为永久 seed ID
visibility[T, M]     # 跟踪器可见性
confidence[T, M]     # 可选；缺失时视为 1
tick_ids[T]          # raw episode 的 tick_id，必须唯一
seed_xy[M, 2]        # 可选；缺失时使用 tracks_xy[0]
```

传入该文件后，预处理器以实测深度回投每个轨迹点，并要求其深度与 FK 渲染的 UR7e/PiKA 表面一致；遮挡、漂移、深度无效或低置信度的点只会变为 `False`，绝不会删除或改变 ID：

```bash
python real_scripts/preprocess_pi05_rgbd_surface_dataset.py \
  --episode-dir outputs/raw_episode_001 \
  --output outputs/pi05_surface/episode_001.npz \
  --pika-mount-transform-json <measured-flange-to-pika.json> \
  --scene-camera-names front side \
  --scene-camera-map front=123456789012 \
  --scene-camera-map side=234567890123 \
  --robot-tracks outputs/tracks/episode_001_front.npz \
  --robot-tracks-camera front
```

输出新增 `visual_robot_track_xy[T,M,2]`、`visual_robot_tracks[T,M,3]`、`visual_robot_visible_mask[T,M]`、`visual_robot_future_offsets[N,H,M,3]` 和 `visual_robot_flow_supervision_mask[N,H,M]`。损失必须只在最后一个 mask 为真的位置计算。`fixed_link_points` 仍是 FK 得到的确定性机器人运动条件；视觉轨迹适合监督真实表面残差、附着物或标定/执行误差，不能取代 FK。

`collect_ur7e_vla_surface_dataset.py` 现在要求 `--camera-calibration`，并把该文件复制到 raw episode 中，供预处理器默认读取。PyTorch PI05 已有联合点流头；若部署 JAX/Flax PI05，则必须实现同构头部，不能直接复用原先独立 PyTorch SafetyModule 的权重。

### PyTorch PI05 联合训练

仓库的 PyTorch PI05 已增加 `PI05SafetyPytorch`：关节角与稳定点集进入 PI05 的非因果 prefix；同一个 action-expert 后缀同时输出动作流匹配速度和未来点偏移。用预处理后的真实 episode 训练：

```bash
cd openpi
uv run --project . ../scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset ../outputs/pi05_surface \
  --output ../outputs/pi05_ur7e_joint/last.pt \
  --pretrained /path/to/pi05_pytorch/model.safetensors \
  --camera-map front=base_0_rgb \
  --camera-map side=left_wrist_0_rgb \
  --point-target fixed \
  --max-points 128 \
  --batch-size 1 --epochs 20
```

`--max-points 128` 对每个 sample 使用固定的等间隔点下标；需要全量点云时设为 `0`，但 Transformer 前缀长度和显存会显著增加。`--point-target visual` 使用已接入 CoTracker 轨迹并由深度/FK 门控后的 `visual_robot_*` 字段；此时点流损失只在其可见性 mask 为真的位置计算。checkpoint 会保存真实 UR 动作与 qpos 的均值/标准差；推理时必须将前 7 维预测动作反归一化，未来点云为 `current_points + predicted_offsets`。

当前 PiKA 预处理会把 body 与两指分开采样，并依记录的开口距离做对称线性平移；这仍是近似运动学。用于精确点流或安全边界前，必须标定真实手指运动学与 `^flange T_pika_step_frame`，并复核投影视频。

## 可选功能

### LingBot-Depth

安装 LingBot-Depth 后可用于离线或在线深度修正。CUDA 默认使用 FP16；CPU 诊断应明确关闭 FP16：

```bash
python real_scripts/demo_record_ur7e_safety_overlay_video.py \
  --adapter real_scripts.ur7e_realsense_adapter:create_adapter \
  --camera-calibration outputs/calibration/session_01/camera_calibration.json \
  --lingbot-depth --lingbot-device cuda:0 \
  --output outputs/overlay.mp4
```

### 浏览实时场景

```bash
python real_scripts/live_ur7e_scene_viewer.py \
  --robot-ip "$UR_ROBOT_IP" \
  --calibration outputs/calibration/session_01/camera_calibration.json \
  --pika-mount-transform-json outputs/calibration/pika_mount_from_tcp_provisional.json \
  --bind-host 0.0.0.0
```

该工具只读取相机和 RTDE 状态；使用 `Ctrl+C` 停止。启用 PiKA 前，必须先测量并验证 `^flange T_pika_step_frame`，示例变换不可用于生产或安全决策。

## 故障排查

- **ChArUco 检测失败**：确认字典、棋盘格数和边长参数准确；增大棋盘在画面中的占比，避免反光与模糊。
- **四点 TCP 拟合失败**：重新触碰脚本指出的内角点；确认探针 TCP 工具偏置已经在 PolyScope 中完成。
- **相机无法打开**：检查序列号、USB 带宽/供电，以及是否有其他进程占用设备。
- **点云错位**：确认点云流分辨率未被命令行覆盖；默认会使用标定文件记录的宽高与 FPS。
- **机器人点未完全剔除**：检查 `*_urdf_removed_overlay.png`；只根据实测误差调整深度容差与体积外扩边界。
