# Safety Module Architecture


## 0. 环境安装

### 0.1 基础 safety module 和 openpi 环境

```bash
conda create -n safety python=3.11
conda activate safety

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
