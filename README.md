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
  --camera-names agentview frontview birdview \
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

## 3. 训练 Point World Model

训练数据建议按 chunk 存：

```text
scene_points:        (N, P, 3)       当前非机器人场景点云 X_t
robot_point_flow:    (N, H, R, 3)    FK/仿真器机器人未来点流
affected_mask:       (N, H, P)       可选，接触/移动标签
future_scene_points: (N, H, P, 3)    可选，有点级对应时使用
future_scene_points_unordered: (N, H, M, 3) 可选，无点级对应时用 Chamfer/Hausdorff
scene_flow:          (N, H, P, 3)    可选，直接 flow 标签
scene_features:      (N, P, C)       可选，例如 RGB
```

第一阶段可以只训练 `affected_mask`：

```bash
python scripts/train_point_world_model.py \
  --dataset outputs/point_world_dataset/libero_task0_mask_dataset.npz \
  --output-dir outputs/point_world_model_mask \
  --epochs 100 \
  --batch-size 8 \
  --lambda-flow 0.0 \
  --lambda-mask 1.0
```

第二阶段加入 scene flow：

```bash
python scripts/train_point_world_model.py \
  --dataset outputs/point_world_dataset/libero_task0_flow_dataset.npz \
  --output-dir outputs/point_world_model_flow \
  --epochs 100 \
  --batch-size 8 \
  --lambda-flow 1.0 \
  --lambda-mask 1.0 \
  --lambda-smooth 0.05 \
  --lambda-chamfer 1.0 \
  --lambda-hausdorff 0.1 \
  --moving-weight 8.0 \
  --near-robot-weight 3.0 \
  --contact-weight 5.0
```

## 4. OpenPI Safety Loss

`OpenPISafetyLoss` 仍用于约束机器人点流不要进入 unsafe SDF。推荐传入 FK/URDF 导出的 TorchScript：

```bash
cd thiry_party/openpi
uv run scripts/train_pytorch.py pi05_libero \
  --exp_name libero_safe_fk_robot_flow \
  --overwrite \
  --safety.sdf_path ../../outputs/safe_space/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_pointcloud_safe_space_sdf.npz \
  --safety.weight 0.05 \
  --safety.margin 0.03 \
  --safety.warmup_steps 2000 \
  --safety.robot_pointcloud_mode torchscript \
  --safety.robot_pointcloud_model_path ../../outputs/fk_robot_point_flow.pt \
  --safety.state_dim 8 \
  --safety.action_dim 7
```

`eef_sphere` 模式只适合先跑通链路：

```bash
uv run scripts/train_pytorch.py pi05_libero \
  --exp_name libero_safe_eef \
  --overwrite \
  --safety.sdf_path ../../outputs/safe_space/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_pointcloud_safe_space_sdf.npz \
  --safety.weight 0.05 \
  --safety.margin 0.03 \
  --safety.warmup_steps 2000 \
  --safety.robot_pointcloud_mode eef_sphere \
  --safety.eef_sphere_radius 0.06 \
  --safety.eef_sphere_points 64 \
  --safety.state_dim 8 \
  --safety.action_dim 7
```

## 5. Collision Critic / VLA Safety Shield

现在可以把确定性 robot point flow 和当前场景点云组合成一个 VLA action-chunk critic：

```python
from safety_module import PointCloudSafetyCritic, geometric_safety_cost, rerank_action_chunks

# scene_points: (B, P, 3)
# robot_point_flow: (B, H, R, 3), from FK/URDF/simulator
critic = PointCloudSafetyCritic()
outputs = critic(scene_points, robot_point_flow, forbidden_mask=forbidden_mask, target_mask=target_mask)

# outputs contains:
# cost:                  (B,)
# collision_probability: (B,)
# min_distance:          (B,)
# risk_heatmap:          (B, H, P)
```

无需训练时可以先用几何 baseline：

```python
geom = geometric_safety_cost(
    scene_points,
    robot_point_flow,
    safe_distance=0.03,
    forbidden_mask=forbidden_mask,
    target_mask=target_mask,
)
```

训练 critic 的数据建议按 action chunk 存：

```text
scene_points:      (N, P, 3)       当前非机器人场景点云
robot_point_flow:  (N, H, R, 3)    确定性未来机器人点流
forbidden_mask:    (N, P) or (N, H, P) 可选，禁止接触点
target_mask:       (N, P) or (N, H, P) 可选，允许任务接触点
future_scene_points: (N, H, P, 3)  可选，动态障碍/点世界预测
collision_label:   (N,)            可选，chunk 是否 unsafe
min_distance:      (N,)            可选，未来最小安全距离
risk_mask:         (N, H, P)       可选，点级风险热图标签
```

训练命令：

```bash
python scripts/train_collision_critic.py \
  --dataset outputs/collision_critic_dataset/libero_task0_safety_chunks.npz \
  --output-dir outputs/collision_critic \
  --epochs 100 \
  --batch-size 8 \
  --safe-distance 0.03 \
  --unsafe-positive-weight 8.0 \
  --lambda-collision 1.0 \
  --lambda-distance 1.0 \
  --lambda-risk 1.0
```

如果暂时只有几何点云、没有人工或仿真碰撞标签，可以先用几何伪标签 warm start：

```bash
python scripts/train_collision_critic.py \
  --dataset outputs/collision_critic_dataset/libero_task0_chunks_without_labels.npz \
  --output-dir outputs/collision_critic_bootstrap \
  --bootstrap-geometric-labels
```

Runtime shield / action reranking 的最小用法：

```python
# actions: (B, K, H, A), K 个 VLA 候选 action chunks
# costs:   (B, K), critic 对每个 chunk 的 cost
# task_scores: (B, K), 可用 VLA log-prob 或任务 critic 分数
result = rerank_action_chunks(
    actions,
    safety_cost=costs,
    task_score=task_scores,
    safety_weight=1.0,
    max_cost=0.5,
)
safe_action = result.action
```

Safe RL 后训练建议只更新 LoRA、action head 或 residual head。`safe_rl.py` 提供了：

- `ResidualActionCorrection`: `a_safe = a_vla + bounded_delta`
- `LagrangeMultiplier`: CMDP 约束乘子
- `ppo_lagrangian_loss`: PPO-style objective with safety cost and KL-to-SFT policy

这部分刻意不绑定某个 VLA 实现；接 OpenPI 时，把冻结 VLA 的 `log_prob`、旧策略 `log_prob`、任务 advantage 和 `PointCloudSafetyCritic.cost` 传进去即可。

`OpenPICollisionCritic` 可在 OpenPI 侧复用同一个 FK robot point-flow TorchScript 和 critic TorchScript，输入 normalized state/action chunk 与 `scene_points`，输出可反传到 action head/LoRA 的 collision cost。

## 6. Legacy Learned Robot Pointcloud

`scripts/train_robot_pointcloud_model.py` 和 `RobotSweptPointCloudModel` 仍保留，但不再是推荐主路线。它们只适合作 baseline，或在没有 FK/URDF 几何源时临时使用。若 FK 存在但有系统误差，应改用 `ResidualRobotPointFlowModel` 学 bounded residual，而不是从零学习机械臂几何。
