# Safety Module Architecture

这个模块现在按三层组织：

1. **确定性机器人几何层**：从当前关节/状态和 action chunk 得到未来机器人点流 `robot_point_flow: (B, H, R, 3)`。优先使用 URDF/FK、控制器积分或仿真器几何，不再把“机器人 swept pointcloud”作为主要学习目标。只有在延迟、柔顺、控制误差或标定误差明显时，才用 `ResidualRobotPointFlowModel` 学小 residual。
2. **接触/影响区域预测**：`PointWorldModel(scene_points, robot_point_flow)` 输出 `mask_logits: (B, H, N)`，回答当前场景点 `X_t` 中哪些点会在未来被机器人影响。
3. **场景点流预测**：同一个 `PointWorldModel` 可输出 `flow: (B, H, N, 3)` 和 `future_points = X_t + flow`。有点级对应时用 weighted Huber/L1；没有对应时应在外层使用 Chamfer/Hausdorff/occupancy/SDF 目标。

核心接口：

```python
from safety_module import PointWorldModel, point_world_model_loss

outputs = model(scene_points, robot_point_flow, scene_features=None)
loss, metrics = point_world_model_loss(
    outputs,
    scene_points=scene_points,
    target_future_points=future_scene_points,
    target_mask=affected_mask,
    robot_point_flow=robot_point_flow,
)
```

`point_world_model_loss` 会提高 moving points、near-robot points 和 near-contact points 的权重，避免训练被大量静态场景点支配。

## 0. 环境安装

建议把环境拆成三类，不要把 OpenPI、LIBERO 和本仓库的轻量 PyTorch 训练脚本硬塞进同一个 Python 环境。OpenPI 当前要求 Python 3.11 和 `torch==2.7.1`，而 LIBERO/robosuite 依赖更偏 Python 3.8、旧版 Torch/MuJoCo。

### 0.1 基础 safety module 环境

这个环境用于训练 `PointWorldModel`、`PointCloudSafetyCritic`、构建 SDF、跑单元 smoke test 和导出 TorchScript：

```bash
conda create -n safety-module python=3.10 -y
conda activate safety-module

# CPU 版本足够跑数据处理和小规模 smoke test；有 CUDA 时可按本机驱动换成对应 PyTorch wheel。
pip install numpy scipy torch torchvision tqdm matplotlib

# 让 scripts/ 可以直接 import safety_module。
export PYTHONPATH=$PWD:$PYTHONPATH
```

安装后做一次快速检查：

```bash
python -m compileall safety_module scripts
python - <<'PY'
import torch
from safety_module import PointCloudSafetyCritic, geometric_safety_cost

scene = torch.randn(2, 128, 3)
robot = torch.randn(2, 10, 64, 3)
critic = PointCloudSafetyCritic()
out = critic(scene, robot)
geom = geometric_safety_cost(scene, robot)
print(out["cost"].shape, geom["min_distance"].shape)
PY
```

### 0.2 LIBERO / MuJoCo 数据采集环境

LIBERO 相关脚本用于重建场景点云、采样 robot point flow、生成 action chunk 数据集。建议单独建 Python 3.8 环境：

```bash
conda create -n libero python=3.8 -y
conda activate libero

pip install -r thiry_party/LIBERO/requirements.txt
pip install -e thiry_party/LIBERO

# 如果 pip resolver 对少数旧依赖失败，可以单独补装：
pip install bddl==1.0.1 easydict==1.9 future==0.18.2

export PYTHONPATH=$PWD:$PWD/thiry_party/LIBERO:$PYTHONPATH
```

无显示器服务器上运行 MuJoCo/LIBERO 时通常需要 EGL：

```bash
export MUJOCO_GL=egl
```

若本机没有 EGL/GPU 渲染，先用 `MUJOCO_GL=osmesa` 或在有显示环境的机器上采集点云；这部分取决于机器的 MuJoCo/OpenGL 驱动配置。

### 0.3 OpenPI 训练环境

OpenPI 使用 `uv` 管理依赖。推荐在 `thiry_party/openpi` 内按它自己的 workspace 安装：

```bash
cd thiry_party/openpi
uv sync
uv pip install -r examples/libero/requirements.txt
cd ../..
```

运行 OpenPI 训练命令时，保持项目根目录可被 import：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
cd thiry_party/openpi
uv run scripts/train_pytorch.py pi05_libero --help
```

如果只是在 OpenPI 里复用本仓库导出的 TorchScript safety loss / collision critic，关键产物是：

```text
outputs/fk_robot_point_flow.pt              FK/仿真器导出的 robot point-flow TorchScript
outputs/collision_critic/collision_critic.pt 训练后的 collision critic TorchScript
outputs/safe_space/*.npz                    SDF safe-space 文件
```

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

## 2. 生成确定性机器人点流

LIBERO 中先用仿真器几何生成每个未来步的 robot point flow。输出里 `robot_point_flow` 是主字段，`points` 只是 legacy union swept cloud：

```bash
/home/evan/anaconda3/envs/libero/bin/python scripts/collect_libero_robot_swept_dataset.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --num-samples 20000 \
  --horizon 10 \
  --action-scale 0.35 \
  --random-prefix-steps 5 \
  --reset-every 5 \
  --points-per-geom 80 \
  --target-points 1024 \
  --disable-nonrobot-collisions \
  --mujoco-gl egl \
  --output outputs/robot_point_flow/libero_spatial_task0_robot_flow.npz
```

可视化单个 action chunk 的机器人 swept volume：

```bash
/home/evan/anaconda3/envs/libero/bin/python scripts/libero_robot_swept_pointcloud.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --horizon 50 \
  --points-per-geom 800 \
  --include-initial \
  --mujoco-gl egl
```

### 2.1 LIBERO 点云和 safespace 图片生成

下面这组命令用于生成论文/调试时检查的 LIBERO 图片。建议先设置公共环境变量：

```bash
export MUJOCO_GL=egl
export PYTHONPATH=$PWD:$PWD/thiry_party/LIBERO:$PYTHONPATH
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

生成 LIBERO / Franka 机械臂关节连线扫略点云。这个脚本不是采样机器人 mesh 几何，而是用 FK 得到 `robot0_link0..robot0_link7` 的关节锚点连线，并额外加入沿夹爪 x 轴方向的末端夹爪宽度线段。

```bash
$LIBERO_PY scripts/libero_joint_swept_pointcloud.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --horizon 50 \
  --samples-per-action 8 \
  --swept-point-link-samples 8 \
  --swept-point-time-samples 2 \
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
outputs/libero_joint_swept_pointcloud/${TASK}_joint_link_swept.npz
```

生成随机 `action_chunk` 长度为 10 的机械臂点云扫略过程图。这里使用固定 robot geom 表面模板并随仿真 rollout 变换，因此相邻 step 的同索引点有稳定对应关系，图中黄色线连接每两步间的对应点。

```bash
$LIBERO_PY scripts/libero_robot_pointcloud_sweep_process.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --horizon 10 \
  --points-per-geom 300 \
  --action-scale 0.18 \
  --seed 0 \
  --plot-elev 18 \
  --plot-azim 0 \
  --max-line-points 3000 \
  --output-dir outputs/libero_robot_pointcloud_sweep_process \
  --mujoco-gl egl
```

主要输出：

```text
outputs/libero_robot_pointcloud_sweep_process/${TASK}_random10_robot_pointcloud_process_frontview_yellow_lines.png
outputs/libero_robot_pointcloud_sweep_process/${TASK}_random10_robot_pointcloud_process.npz
outputs/libero_robot_pointcloud_sweep_process/${TASK}_random10_robot_pointcloud_process.ply
outputs/libero_robot_pointcloud_sweep_process/${TASK}_random10_robot_pointcloud_process_actions.npy
```

把机械臂扫略点云和障碍物 OBB 渲染成正面检查图：

如果 `outputs/libero_robot_swept_pointcloud/${TASK}_robot_swept.npz` 还不存在，先运行本节前面的 `scripts/libero_robot_swept_pointcloud.py` swept volume 命令。

```bash
$LIBERO_PY scripts/render_libero_frontview_figures.py \
  --swept-pointcloud outputs/libero_robot_swept_pointcloud/${TASK}_robot_swept.npz \
  --scene-pointcloud outputs/libero_pointcloud/${TASK}_pointcloud.npz \
  --obb-safe-space outputs/safe_space/${TASK}_tabletop_xy_oriented_obstacle_obb_safe_space.npz \
  --output-dir outputs/libero_frontview_figures \
  --name ${TASK}
```

主要输出：

```text
outputs/libero_frontview_figures/${TASK}_frontview_swept_pointcloud.png
outputs/libero_frontview_figures/${TASK}_frontview_obstacle_pointcloud_obb.png
```


## 3. Upright Blocks LeRobot Demo Collection

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
cd thiry_party/openpi
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  ../../scripts/collect_upright_blocks_lerobot_demos.py \
  --convert-only ../../outputs/robosuite_collision_scene/lerobot_demos/<run>/raw \
  --repo-id local/ur5e_upright_blocks \
  --overwrite
```
