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


## 4. safety flow decoder

safety decoder 使用 PI05 VLM `prefix_tokens` 预测未来点云，不直接预测碰撞分类。安全信号由预测点和障碍物 OBB / occupied grid 的几何重叠计算得到。

直接在 LIBERO 中运行 `pi05_libero` 推理，并同步保存每个真实控制时间步的 prefix token、PI05 action chunk、当前关节 qpos、当前机械臂表面点，以及真实执行轨迹中的未来表面点偏移量。采集脚本固定使用 rollout surface 点云流：先按 `--replan-steps` 执行任务 rollout，并在每个真实仿真时间步记录一次机械臂表面点云；rollout 结束后，再为每个时间步样本从这条表面点云轨迹中切出当前帧和未来 `len(action_chunk)` 帧。

OpenPI / safety 环境窗口，启动会额外返回 `prefix_tokens` 的 websocket policy server：

```bash
python scripts/serve_pi05_prefix_policy.py \
  --policy-config pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --port 8000
```

LIBERO 环境窗口，连接上面的 server，负责仿真、表面点云采集和保存训练数据：

```bash
python scripts/collect_pi05_libero_safety_decoder_dataset.py \
  --policy-server-host 127.0.0.1 \
  --policy-server-port 8000 \
  --task-suite libero_spatial \
  --task-ids 0 1 2 \
  --num-rollouts 5 \
  --max-samples 256 \
  --replan-steps 5 \
  --points-per-link 256 \
  --output outputs/pi05_safety_decoder/xxx.npz 
```

若要一次采集 `libero_spatial` 下全部任务，并合并成一个可直接训练的 `.npz`：

```bash
python scripts/collect_pi05_libero_safety_decoder_dataset.py \
  --policy-server-host 127.0.0.1 \
  --policy-server-port 8000 \
  --task-suite libero_spatial \
  --task-ids all \
  --num-rollouts 5 \
  --max-samples-per-task 256 \
  --replan-steps 5 \
  --points-per-link 256 \
  --output outputs/pi05_safety_decoder/xxx.npz
```

也可以只采集一部分任务，例如 `--task-ids 0 1 2 3`。输出文件中的 `task_ids` 字段会记录每条样本来自哪个 LIBERO task。

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


Flow Matching 点云安全模型 `SafetyFlowPointModel`。该模型使用 `prefix_tokens + arm_points` 作为条件，学习 `target_point_offsets` 的 velocity field：

```bash
python scripts/train_pi05_safety_flow_point_model.py \
  --dataset outputs/pi05_safety_decoder/xxx.npz \
  --output outputs/pi05_safety_decoder/xxx.pt \
  --hidden-dim 256 \
  --num-encoder-layers 4 \
  --num-decoder-layers 4 \
  --num-heads 8 \
  --ffn-dim 1024 \
  --epochs 128 \
  --batch-size 4 \
  --lr 1e-4
```

训练“去掉 prefix 条件”的 ablation 时，保持同一份数据和模型结构，只在训练批次中把 `prefix_tokens` 置零，并在 checkpoint metadata 中记录 `prefix_ablation=zero`


在线验证 `SafetyFlowPointModel` 时分两个窗口运行。窗口 1 在 `safety` 环境启动完整 PI05 policy + prefix token server：

```bash
python scripts/serve_pi05_prefix_policy.py \
  --policy-config pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --safety-checkpoint outputs/pi05_safety_decoder/xxx.pt \
  --port 8000
```

窗口 2 在 LIBERO 环境运行在线验证。脚本会连接窗口 1 的 server，在 LIBERO 中执行任务，并生成包含机械臂执行画面、每个真实时间步预测未来 point flow、场景 OBB 方框和碰撞提示的 MP4。视频底部的 `POSSIBLE COLLISION` 来自预测点云进入 OBB 的几何判断；`REAL COLLISION` 来自 MuJoCo contact 中机器人 geom 与目标障碍物 geom 的真实接触，并会同步写入 `.npz` 的 `real_collision_flags` 和 `real_collision_contact_counts`：

```bash
python scripts/evaluate_pi05_safety_decoder_on_libero.py   \
--policy-server-host 127.0.0.1  \
 --policy-server-port 8000   \
 --task-suite libero_spatial   \
 --task-id 0   \
 --num-rollouts 1   \
 --max-samples 256   \
 --replan-steps 5   \
 --points-per-link 256   \
 --prediction-steps 10   \
 --realtime-obbs   \
 --enable-cbf-qp   \
 --output outputs/pi05_safety_decoder/pi05_libero_task0_eval.npz   \
 --video-output outputs/pi05_safety_decoder/pi05_libero_task0_eval.mp4   \
 --mujoco-gl egl
```

<!-- python scripts/evaluate_pi05_safety_decoder_on_libero.py   --policy-server-host 127.0.0.1   --policy-server-port 8000   --task-suite libero_spatial   --task-id 0   --num-rollouts 1   --max-samples 256   --replan-steps 5   --points-per-link 128   --prediction-steps 10   --realtime-obbs    --output outputs/pi05_safety_decoder/pi05_libero_task0_eval.npz   --video-output outputs/pi05_safety_decoder/pi05_libero_task0_eval.mp4   --scene-obstacle-xy 0.0 0.04 -->

CBF-QP 默认使用 `--cbf-action-space auto`：当 LIBERO 环境 `action_dim == 4` 或 `action_dim == 7` 时，脚本按 OSC_POSITION 语义处理 PI05 动作，直接在可执行的笛卡尔 `xyz` action 空间做 QP，点云对 action 的 Jacobian 由“扰动 `xyz` -> 临时 `env.step` -> 重建下一帧机械臂点云”的有限差分估计，并保留原 orientation / gripper；其他环境默认沿用 joint-delta CBF。可用 `--cbf-action-space joint_delta` 强制使用关节增量模式，或用 `--cbf-action-space cartesian_delta` 选择上一版“笛卡尔动作先通过末端 Jacobian / pseudo-inverse 转 nominal 关节增量，再映射回 `xyz`”的模式。通常不建议在 LIBERO OSC_POSITION 实验里强制 `cartesian_delta`，因为它会把可执行的 `xyz` 动作绕到关节空间再映射回来，切向动作更容易被多约束投影改变。

当前默认 CBF-QP 的 active set 只来自预测未来点云：如果预测未来第 `k` 帧点云进入 OBB，脚本会修正 active action chunk 中 `current_offset + k` 对应的 action，其中未来第 `0` 帧会映射到当前 action。修正后的 chunk 会缓存到对应 action 真正执行时使用；这避免了“未来帧碰撞却总是修改当前 action”的错位。默认不会额外检查当前帧机械臂点云；若需要把当前帧点云约束也混入预测 point-flow CBF，可加 `--cbf-include-current-points`。若需要复现实验中的旧行为，可显式使用 `--cbf-correction-target current_action`。

CBF-QP 当前实现的数学形式如下。对第 $j$ 个 OBB，设中心为 $c_j$，世界系轴为 $R_j[:, a]$，半边长为 $d_j$，碰撞 margin 为 $m$，则使用膨胀半边长 $\bar d_j=d_j+m$。对一个被预测点云触发的机械臂表面点 $p_i$，先用当前帧点 $p_i^0$ 选择最近需要远离的 OBB face：

$$
\begin{aligned}
y_i &= R_j^\top (p_i^0 - c_j), \\
a^\star &= \arg\max_a \frac{|y_i[a]|}{\bar d_j[a]}, \\
s &= \operatorname{sign}(y_i[a^\star]), \\
n_i &= s R_j[:, a^\star], \\
h_i &= n_i^\top (p_i^0 - c_j) - \bar d_j[a^\star].
\end{aligned}
$$

$h_i \ge 0$ 表示该点在所选 face 外侧，$h_i < 0$ 表示已经进入膨胀 OBB。默认约束来自预测未来第 $k$ 帧点云，并会计算同一 face 方向上的预测 barrier：

$$
h_i^{(k)} = n_i^\top (p_i^k - c_j) - \bar d_j[a^\star],
\qquad
\tilde h_i = \min(h_i, h_i^{(k)}).
$$

这样当未来预测点已经进入 OBB 时，QP 会看到 $\tilde h_i < 0$，而不是只看到当前点仍在 OBB 外侧。若显式加 `--cbf-include-current-points`，当前帧点云触发的约束直接使用 $h_i$。设 QP 变量为 $u$，它可以是关节增量、可执行 Cartesian `xyz` 动作，或旧版 Cartesian-to-joint 变量；点云对该变量的一步 Jacobian 为 $J_i = \partial p_i^+ / \partial u$。脚本使用一阶线性化的离散 CBF 约束：

$$
\begin{aligned}
\tilde h_i(p_i^+) &\approx \tilde h_i + n_i^\top J_i u, \\
\tilde h_i + n_i^\top J_i u &\ge (1 - \alpha) \tilde h_i, \\
n_i^\top J_i u &\ge -\alpha \tilde h_i.
\end{aligned}
$$

因此每个 active point 生成一行 $A_i = n_i^\top J_i$、$b_i = -\alpha \tilde h_i$。最终求解的是保持 nominal action 尽量不变的投影问题：

$$
\begin{aligned}
u_{\mathrm{safe}}
&= \arg\min_u \frac{1}{2}\lVert u - u_{\mathrm{nom}}\rVert_2^2 \\
\text{s.t.}\quad
A u &\ge b, \\
u_{\mathrm{lower}} &\le u \le u_{\mathrm{upper}}.
\end{aligned}
$$

单个 face 约束只会去掉朝 OBB 内部的法向分量，理论上会保留沿 OBB 表面的切向分量，也就是“滑过 OBB”。之前默认 `--cbf-fallback zero` 时，如果多点 / 多面约束在有限迭代内没有完全满足，脚本会把 QP 变量整体清零，导致切向分量也被抹掉，看起来就无法滑过 OBB。现在默认 `--cbf-fallback projected`，即使 `success=False` 也会执行当前 best-effort 投影；如果需要旧的保守停止行为，可以显式设置 `--cbf-fallback zero`。

运行“无预测 point-flow、仅当前机械臂点云触发 CBF-QP”的 eval ablation 时，复用同一验证脚本，打开 CBF-QP 并把 active-set 来源切到当前点云 `--cbf-trigger-source current_pointcloud`.

默认验证会在每个采样时间步用当前 LIBERO RGB-D 状态实时重建障碍物 OBB，并用同一组 OBB 绘制视频方框和判断未来 point flow 是否进入 OBB。实时 OBB 默认通过 MuJoCo segmentation 只保留名称匹配 `eval_scene_obstacle` / `wine_bottle` / `winebottle` 的 geom/body 像素，因此当前只会给插入的 wine bottle 建 OBB；若要恢复旧的整张桌面障碍物建模，可使用 `--obb-target-geom-name-patterns all`。实时 OBB 默认使用 `--obb-component-connectivity 6`，只把共享面的体素连成同一障碍物，比旧的 26 邻接更不容易把相邻障碍物混成一个大 OBB；需要更细的建模时可同时减小 `--obb-stride`、`--obb-component-voxel-size` 和 `--obb-box-margin`。若需要复用预生成的静态 OBB 文件，可改用 `--no-realtime-obbs --safe-space outputs/safe_space/${TASK}_tabletop_xy_oriented_obstacle_obb_safe_space.npz`。

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
