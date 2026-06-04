# Safety Module Architecture


## 0. 环境安装

### 0.1 基础 safety module 和 openpi 环境

```bash
conda create -n safety python=3.11
conda activate safety

cd openpi
uv sync
uv pip install -e .
pip install chex pytest
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 0.2 LIBERO / MuJoCo 数据采集环境

LIBERO 相关脚本用于重建场景点云、采样 robot point flow、生成 action chunk 数据集。建议单独建 Python 3.8 环境：

```bash
uv venv --python 3.8 openpi/examples/libero/.venv
source openpi/examples/libero/.venv/bin/activate
uv pip sync openpi/examples/libero/requirements.txt openpi/third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e openpi/packages/openpi-client
uv pip install -e openpi/third_party/libero
uv pip install h5py

export PYTHONPATH=$PYTHONPATH:$PWD/openpi/third_party/libero
```

无显示器服务器上运行 MuJoCo/LIBERO 时通常需要 EGL：

```bash
export MUJOCO_GL=egl
```

若本机没有 EGL/GPU 渲染，先用 `MUJOCO_GL=osmesa` 或在有显示环境的机器上采集点云；这部分取决于机器的 MuJoCo/OpenGL 驱动配置。



## 1. 重建当前场景点云

从 RGB-D / 多视角深度融合出当前场景点云，默认去掉机器人自身点：

```bash
python scripts/libero_reconstruct_pointcloud.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --camera-names agentview sideview leftsideview \
  --width 256 \
  --height 256 \
  --stride 2 \
  --output-dir outputs/libero_pointcloud
```

## 2. 生成机器人点流和障碍物obb

### 2.1 LIBERO 点云和 safespace 图片生成

下面这组命令用于生成论文/调试时检查的 LIBERO 图片。建议先设置公共环境变量：

```bash
export MUJOCO_GL=egl
export PYTHONPATH=$PWD:$PWD/openpi/third_party/libero:$PYTHONPATH
LIBERO_PY=/home/evan/anaconda3/envs/libero/bin/python
TASK=pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
```

重建三视角场景点云，用于后续障碍物 OBB / safespace 建模：

```bash
$LIBERO_PY scripts/libero_reconstruct_pointcloud.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --camera-names frontview sideview leftsideview \
  --width 256 \
  --height 256 \
  --stride 2 \
  --output-dir outputs/libero_pointcloud \
  --mujoco-gl egl
```

生成桌面障碍物 OBB 和 safespace 图。预览图里的黑色虚线框使用 `display_workspace_bounds`，底面是桌面平面 `table_z`，底面大小是桌面 x/y 范围；`workspace_bounds` 仍用于 voxel grid / SDF 计算。

```bash
$LIBERO_PY scripts/build_safe_space_from_pointcloud.py \
  --pointcloud outputs/libero_pointcloud/${TASK}_pointcloud.npz \
  --obstacle-mode tabletop_boxes \
  --box-orientation xy_oriented \
  --box-shape cuboid \
  --component-voxel-size 0.02 \
  --min-component-points 40 \
  --box-margin 0.01 \
  --voxel-size 0.04 \
  --draw-pointcloud \
  --output-dir outputs/safe_space \
  --name ${TASK}_tabletop_xy_oriented_obstacle_obb
```

主要输出：

```text
outputs/safe_space/${TASK}_tabletop_xy_oriented_obstacle_obb_safe_space.npz
outputs/safe_space/${TASK}_tabletop_xy_oriented_obstacle_obb_safe_space_preview.png
```

生成左、右、前三个深度视角融合的机械臂可见表面点云：

```bash
$LIBERO_PY scripts/libero_reconstruct_pointcloud.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --camera-names frontview sideview leftsideview \
  --width 512 \
  --height 512 \
  --stride 1 \
  --max-depth 4.0 \
  --only-robot \
  --save-robot-masks \
  --output-dir outputs/libero_visible_robot_pointcloud \
  --mujoco-gl egl
```

把这个三视角融合 3D 点云投影到 `frontview` 正面相机视角：

```bash
$LIBERO_PY scripts/render_libero_pointcloud_camera_view.py \
  --pointcloud outputs/libero_visible_robot_pointcloud/${TASK}_visible_robot_pointcloud.npz \
  --task-suite libero_spatial \
  --task-id 0 \
  --camera-name frontview \
  --width 768 \
  --height 768 \
  --point-size 3 \
  --output-dir outputs/libero_visible_robot_pointcloud_frontview_render \
  --name ${TASK}_three_view_visible_robot_pointcloud \
  --mujoco-gl egl
```

主要输出：

```text
outputs/libero_visible_robot_pointcloud_frontview_render/${TASK}_three_view_visible_robot_pointcloud_frontview_projected_points.png
outputs/libero_visible_robot_pointcloud_frontview_render/${TASK}_three_view_visible_robot_pointcloud_frontview_projected_overlay.png
```

生成 LIBERO / Franka 机械臂 skeleton 扫略点云。默认 `--skeleton-source geom` 会在每个 FK 采样时刻读取机器人 MuJoCo geoms 的包络中心轴：capsule / cylinder 用几何体中心轴，box / ellipsoid 用最长中心轴，mesh 用编译后顶点包围盒最长中心轴；这比原来的 `robot0_link0..robot0_link7` 关节锚点连线更接近机械臂包裹面的中心。可视化时同一连杆 body 下的多个 geom 使用同一种颜色，不同连杆使用不同颜色。需要对比旧逻辑时可加 `--skeleton-source anchors`。

```bash
$LIBERO_PY scripts/libero_joint_swept_pointcloud.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --horizon 200 \
  --action-scale 0.12 \
  --skeleton-source geom \
  --samples-per-action 8 \
  --swept-point-link-samples 8 \
  --swept-point-time-samples 2 \
  --safe-space outputs/safe_space/${TASK}_tabletop_xy_oriented_obstacle_obb_safe_space.npz \
  --collision-margin 0.0 \
  --save-video \
  --video-fps 12 \
  --frontview-width 768 \
  --frontview-height 768 \
  --frontview-point-size 3 \
  --output-dir outputs/libero_joint_swept_pointcloud \
  --mujoco-gl egl
```

主要输出：

```text
outputs/libero_joint_swept_pointcloud/${TASK}_joint_link_swept_frontview_swept_points.png
outputs/libero_joint_swept_pointcloud/${TASK}_joint_link_swept_frontview_swept_points_overlay.png
outputs/libero_joint_swept_pointcloud/${TASK}_joint_link_swept_frontview_swept_points_3d.png
outputs/libero_joint_swept_pointcloud/${TASK}_joint_link_swept_frontview_swept_points.mp4
outputs/libero_joint_swept_pointcloud/${TASK}_joint_link_swept.npz
```

其中 `*_frontview_swept_points.mp4` 会在 `frontview` 相机图像上按时间累计显示连杆点，点数随帧递增；`*_frontview_swept_points_3d.png` 标题会写明 `collision: YES/NO`；`*.npz` 内保存 `collision`、`collision_method`、`collision_point_count` 和 `collision_swept_point_indices`。


## 3. 可视化

```bash
python scripts/visualize_pi05_safety_decoder_dataset_sample.py \
  --dataset outputs/pi05_safety_decoder/pi05_libero_task0_decoder_dataset.npz \
  --sample-index 0 \
  --time-index 10
```


## 4. PI05 latent safety decoder

第一版 safety decoder 使用 PI05 VLM `prefix_tokens` 预测未来连杆点，不直接预测碰撞分类。安全信号由预测点和障碍物 OBB / occupied grid 的几何重叠计算得到。

直接在 LIBERO 中运行 `pi05_libero` 推理，并同步保存每个真实控制时间步的 prefix token、PI05 action chunk、当前关节 qpos、当前机械臂表面点，以及真实执行轨迹中的未来表面点偏移量。采集脚本固定使用 rollout surface 点云流：先按 `--replan-steps` 执行任务 rollout，并在每个真实仿真时间步记录一次机械臂表面点云；rollout 结束后，再为每个时间步样本从这条表面点云轨迹中切出当前帧和未来 `len(action_chunk)` 帧。

OpenPI / safety 环境窗口，启动会额外返回 `prefix_tokens` 的 websocket policy server：

```bash
conda activate safety
export PYTHONPATH=$PWD:$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src:$PYTHONPATH

python scripts/serve_pi05_prefix_policy.py \
  --policy-config pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000
```

LIBERO 环境窗口，连接上面的 server，负责仿真、表面点云采集和保存训练数据：

```bash
source openpi/examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD:$PWD/openpi/packages/openpi-client/src:$PWD/openpi/third_party/libero:$PYTHONPATH
export MUJOCO_GL=egl

python scripts/collect_pi05_libero_safety_decoder_dataset.py \
  --policy-server-host 127.0.0.1 \
  --policy-server-port 8000 \
  --task-suite libero_spatial \
  --task-id 0 \
  --num-rollouts 5 \
  --max-samples 256 \
  --replan-steps 5 \
  --points-per-link 128 \
  --output outputs/pi05_safety_decoder/pi05_libero_task0_decoder_dataset.npz \
  --mujoco-gl egl
```

如果不用 websocket server，也可以在单个同时安装了 OpenPI 和 LIBERO 依赖的环境里省略 `--policy-server-host`，让采集脚本本地加载 policy。

输出 `.npz` 可直接用于训练，关键字段为：

```text
prefix_tokens: shape [S, N, D]
action_chunks: shape [S, T, A]
start_joint_vectors: shape [S, J]
target_link_points: shape [S, T_fk, L, P, 3]
current_link_points: shape [S, L, P, 3]
future_link_offsets: shape [S, T_action, L, P, 3]
arm_points: shape [S, K, 3], K = L * P
target_point_offsets: shape [S, T_action, K, 3]
```

其中 `T_action == len(action_chunk)`，`pi05_libero` 默认是 10；因此 `target_link_points` 包含当前帧和 10 个未来真实执行帧，长度为 11。采集只保留拥有完整未来窗口的时间步：如果一条轨迹记录了 150 个 surface 帧、未来长度为 10，就会生成 `150 - 10 = 140` 条样本。

采集脚本固定使用 rollout surface 点云流：在仿真中对 `robot0_link1..robot0_link7` 的机械臂表面固定采样，因此 `L=7`，每个 link 有 `P=points_per_link` 个固定身份点。当前点云和未来点云都来自真实执行 rollout 中同一批 link-local 表面采样点在各时间步的 MuJoCo world 坐标，适合用 MSE / Flow Matching 学习 `future - current` 偏移。真实部署时可用真实机器人关节状态和 URDF / mesh 的同一套固定采样点通过 FK 生成当前机械臂点云；障碍物仍由 RGB-D 点云 / OBB 重建。

其中 `arm_points` 是 `SafetyFlowPointModel` 的当前机械臂局部点云输入，`target_point_offsets` 是 Flow Matching 的真实 `x_1 = P_arm_future - P_arm_current`。`target_link_points` 仍保留完整绝对坐标路径；第 0 帧是当前点，后续帧用于计算 `future_link_offsets`。

`target_link_points` / `current_link_points` / `arm_points` 使用 MuJoCo world 坐标系保存，字段 `coordinate_frame=mujoco_world` / `target_link_points_frame=mujoco_world` / `arm_points_frame=mujoco_world` 会写入 `.npz`。偏移量字段使用 `mujoco_world_delta`。由 `scripts/libero_reconstruct_pointcloud.py` 重建出的障碍物点云，以及 `scripts/build_safe_space_from_pointcloud.py` 生成的 OBB / safe-space 也使用同一 `mujoco_world` 坐标系。

旧的离线 seed + action chunk FK 建集路径不再作为当前 safety flow 数据采集入口。当前训练数据应通过上面的 rollout surface 采集脚本生成。

训练 Transformer decoder。模型先把 PI05 prefix token 投影到 `hidden_dim`，经过 TransformerEncoder，再由线性层映射为未来连杆点：

```bash
python scripts/train_pi05_safety_decoder.py \
  --dataset outputs/pi05_safety_decoder/pi05_libero_task0_decoder_dataset.npz \
  --output outputs/pi05_safety_decoder/decoder.pt \
  --hidden-dim 512 \
  --num-layers 4 \
  --num-heads 8 \
  --ffn-dim 2048 \
  --max-tokens 1024 \
  --epochs 50 \
  --batch-size 4
```

训练新的 Flow Matching 点云安全模型 `SafetyFlowPointModel`。该模型使用 `prefix_tokens + arm_points` 作为条件，学习 `target_point_offsets` 的 velocity field：

```bash
python scripts/train_pi05_safety_flow_point_model.py \
  --dataset outputs/pi05_safety_decoder/pi05_libero_task0_decoder_dataset.npz \
  --output outputs/pi05_safety_decoder/pi05_libero_task0_safety_flow_point_model.pt \
  --hidden-dim 256 \
  --num-encoder-layers 4 \
  --num-decoder-layers 4 \
  --num-heads 8 \
  --ffn-dim 1024 \
  --epochs 100 \
  --batch-size 2 \
  --lr 1e-4 \
  --device cpu
```

在线验证 `SafetyFlowPointModel` 时分两个窗口运行。窗口 1 在 `safety` 环境启动完整 PI05 policy + prefix token server：

```bash
conda activate safety
export PYTHONPATH=$PWD:$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src:$PYTHONPATH

python scripts/serve_pi05_prefix_policy.py \
  --policy-config pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --safety-checkpoint outputs/pi05_safety_decoder/pi05_libero_task0_safety_flow_point_surface_model_6_2_18_15.pt \
  --port 8000
```

窗口 2 在 LIBERO 环境运行在线验证。脚本会连接窗口 1 的 server，在 LIBERO 中执行任务，并生成包含机械臂执行画面、每个真实时间步预测未来 point flow、场景 OBB 方框和碰撞提示的 MP4：

```bash
source openpi/examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD:$PWD/openpi/packages/openpi-client/src:$PWD/openpi/third_party/libero:$PYTHONPATH
export MUJOCO_GL=egl

python scripts/evaluate_pi05_safety_decoder_on_libero.py \
  --policy-server-host 127.0.0.1 \
  --policy-server-port 8000 \
  --task-suite libero_spatial \
  --task-id 0 \
  --num-rollouts 1 \
  --max-samples 16 \
  --replan-steps 5 \
  --points-per-link 128 \
  --prediction-steps 10 \
  --realtime-obbs \
  --obb-camera-names frontview sideview leftsideview \
  --obb-width 160 \
  --obb-height 160 \
  --obb-stride 4 \
  --collision-margin 0.0 \
  --output outputs/pi05_safety_decoder/pi05_libero_task0_flow_online_eval.npz \
  --video-output outputs/pi05_safety_decoder/pi05_libero_task0_flow_online_eval.mp4 \
  --mujoco-gl egl
```

默认验证会在每个采样时间步用当前 LIBERO RGB-D 状态实时重建障碍物 OBB，并用同一组 OBB 绘制视频方框和判断未来 point flow 是否进入 OBB。若需要复用预生成的静态 OBB 文件，可改用 `--no-realtime-obbs --safe-space outputs/safe_space/${TASK}_tabletop_xy_oriented_obstacle_obb_safe_space.npz`。

当 websocket server 通过 `--safety-checkpoint` 启动时，验证脚本会根据 server metadata 自动使用远端 safety module 预测未来 point flow；验证环境不再需要选择 `cpu` / `cuda`。如果 server 没有加载 safety module，验证脚本会回退到本地 `--checkpoint`，并用 `--device auto` 自动选择设备。

推理并用几何计算输出 `collision` 或 `safe`：

```bash
python scripts/run_pi05_safety_decoder.py \
  --checkpoint outputs/pi05_safety_decoder/decoder.pt \
  --prefix-tokens outputs/pi05_prefix_tokens/current_prefix_tokens.npz \
  --safe-space outputs/safe_space/${TASK}_tabletop_xy_oriented_obstacle_obb_safe_space.npz \
  --collision-margin 0.01 \
  --output outputs/pi05_safety_decoder/current_safety_result.npz
```

## 5. Upright Blocks LeRobot Demo Collection

当前 UR5e upright-blocks 场景的数据采集脚本：

```bash
NUMBA_DISABLE_JIT=1 /home/evan/anaconda3/envs/libero/bin/python \
  scripts/collect_upright_blocks_lerobot_demos.py \
  --repo-id local/ur5e_upright_blocks \
  --num-demos 50 \
  --overwrite
```

默认采集逻辑：

- 图像：保存 `frontview/sideview/leftsideview`，并额外提供 OpenPI LIBERO 兼容的 `image=frontview`、`wrist_image=leftsideview`。
- 任务描述：每帧和每个 episode 固定保存 `pick up the red cube and place it on the plate without touching the yellow blocks`。
- 状态：`state` / `observation.state` 为 6 个 UR5e arm joints + 1 个 gripper scalar。
- 动作：`actions` / `action` 为实际观测到的 joint delta，即 `next_state - state`，不是 OSC 控制器命令。
- 成功：红色物块在盘子上、夹爪已释放且不再接触红块，连续保持 `--success-hold` 步后才保存。
- 失败：黄色立方体倾倒，或黄色立方体和任意非桌面几何发生接触，当前 attempt 直接丢弃并重来。
- 时间：`--max-steps 0` 表示不限时，这是默认值。

如果采集环境里没有 `lerobot` 包，脚本会先保存 raw `.npz` 备份；之后可在 OpenPI 环境转换：

```bash
cd openpi
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  ../../scripts/collect_upright_blocks_lerobot_demos.py \
  --convert-only ../../outputs/robosuite_collision_scene/lerobot_demos/<run>/raw \
  --repo-id local/ur5e_upright_blocks \
  --overwrite
```
