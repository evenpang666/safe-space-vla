import os
import sys
import numpy as np
import imageio
import h5py
import cv2
import matplotlib.pyplot as plt

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import argparse



def blacken_scene_keep_robot(sim):
    """把场景中除机器人以外的 geom 与 site 设为透明。
    保持机器人相关几何的颜色不变，以便直接渲染出含背景的图像并通过检测透明区域以外的像素得到机器人掩码。
    该函数通过名称启发式判断机器人相关的 geom/site（包含 'robot','panda','gripper','hand','arm' 等关键字）。"""
    # 尝试获取几何体数量
    ngeom = getattr(sim.model, 'ngeom', None)
    if ngeom is None:
        ngeom = len(getattr(sim.model, 'geom_pos', []))

    for gid in range(ngeom):
        # 尝试获取 geom 名称
        gname = ''
        try:
            gname = sim.model.geom_id2name(gid)
        except Exception:
            try:
                gname = sim.model.geom_names[gid]
            except Exception:
                gname = ''

        lname = gname.lower() if gname else ''
        is_robot = any(k in lname for k in ('robot', 'panda', 'gripper', 'hand', 'arm'))

        

        # 非机器人 geom 设为透明
        if not is_robot:
            try:
                sim.model.geom_rgba[gid] = np.array([0.0, 0.0, 0.0, 0.0])
            except Exception:
                # 有些绑定可能不支持直接写入 rgba，忽略错误
                pass
        
        if is_robot:
            try:
                sim.model.geom_rgba[gid] = np.array([1.0, 1.0, 1.0, 1.0])
            except Exception:
                # 有些绑定可能不支持直接写入 rgba，忽略错误
                pass

        if gname=='floor' or 'wall' in gname.lower():
            try:
                sim.model.geom_rgba[gid] = np.array([0.0, 0.0, 0.0, 1.0])
            except Exception:
                # 有些绑定可能不支持直接写入 rgba，忽略错误
                pass

    # 处理 site：非机器人相关 site 设为透明
    if hasattr(sim.model, 'site_names'):
        for i, name in enumerate(sim.model.site_names):
            if name=='flat_stove_1_burner' or name=='gripper0_grip_site' or name=='gripper0_grip_site_cylinder':
                try:
                    sim.model.site_rgba[i] = np.array([0.0, 0.0, 0.0, 0.0])
                except Exception:
                    pass


def set_robot_qpos_from_jointpos(sim, joint_pos_seq, frame_idx):
    """尝试把 joint_pos（shape T x N）写入 sim.data.qpos 对应 robot joints。
    本函数做一些启发式匹配：匹配包含 robot 的 joint 名称顺序。返回 True/False 是否成功。"""
    if joint_pos_seq is None:
        return False

    # 读取 joint 名称并挑选与 robot 相关的 joints（heuristic）
    try:
        njnt = sim.model.njnt
    except Exception:
        njnt = len(getattr(sim.model, 'jnt_qposadr', []))

    robot_joint_ids = []
    jnames = []
    for jid in range(njnt):
        try:
            jn = sim.model.joint_id2name(jid)
        except Exception:
            try:
                jn = sim.model.joint_names[jid]
            except Exception:
                jn = ''
        jnames.append(jn)
        if 'robot' in jn.lower() or 'panda' in jn.lower() or 'right' in jn.lower() or 'left' in jn.lower():
            robot_joint_ids.append(jid)

    # 如未找到，则尝试选前 N joints
    if len(robot_joint_ids) == 0:
        robot_joint_ids = list(range(min(7, njnt)))

    qpos = sim.data.qpos.copy()

    # joint_pos_seq 可能是 (T, N)
    if frame_idx >= len(joint_pos_seq):
        return False

    cur = joint_pos_seq[frame_idx]
    # 如果长度匹配则直接填入
    if len(cur) == len(robot_joint_ids):
        for i, jid in enumerate(robot_joint_ids):
            adr = int(sim.model.jnt_qposadr[jid])
            qpos[adr] = float(cur[i])
        sim.data.qpos[:] = qpos
        try:
            sim.forward()
        except Exception:
            pass
        return True

    # 否则，尝试把 cur 的前 len(robot_joint_ids) 值写入
    if len(cur) >= len(robot_joint_ids):
        for i, jid in enumerate(robot_joint_ids):
            adr = int(sim.model.jnt_qposadr[jid])
            qpos[adr] = float(cur[i])
        sim.data.qpos[:] = qpos
        try:
            sim.forward()
        except Exception:
            pass
        return True

    return False



def show_rendered_samples(demo_file, demo_key=None, n=5, figsize=(8, 4)):
    """在一个窗口中展示指定 demo 的前 n 帧 `img_mask` 与 `masked_rgb`。

    如果 `demo_key` 为 None，则使用文件中的第一个 demo key（通常为 'demo_0'）。
    """
    print('Showing rendered samples from:', demo_file)
    if not os.path.exists(demo_file):
        print('Demo file not found:', demo_file)
        return

    with h5py.File(demo_file, 'r') as f:
        if 'data' not in f:
            print('No data group in file')
            return

        demo_keys = list(f['data'].keys())
        if len(demo_keys) == 0:
            print('No demos in file')
            return

        if demo_key is None:
            demo_key = demo_keys[0]

        if f'data/{demo_key}' not in f:
            print('Demo key not found:', demo_key)
            return

        data_grp = f[f'data/{demo_key}']
        if 'rendered' not in data_grp:
            print('No rendered group for', demo_key)
            return

        rg = data_grp['rendered']
        if 'img_mask' not in rg or 'masked_rgb' not in rg:
            print('rendered missing img_mask or masked_rgb for', demo_key)
            return

        img_mask_ds = rg['img_mask']
        rgb_ds = rg['masked_rgb']

        total = img_mask_ds.shape[0]
        m = min(n, total)
        if m == 0:
            print('No frames to show for', demo_key)
            return

        fig, axes = plt.subplots(m, 2, figsize=(figsize[0], figsize[1] * m))
        if m == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(m):
            mask = img_mask_ds[i]
            rgb = rgb_ds[i]

            ax_mask = axes[i, 0]
            ax_rgb = axes[i, 1]

            ax_mask.imshow(mask, cmap='gray', vmin=0, vmax=255)
            ax_mask.set_title(f'{demo_key} frame {i} mask')
            ax_mask.axis('off')

            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f'{demo_key} frame {i} masked_rgb')
            ax_rgb.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    #input args about demo_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', type=str, default="/home/evan/projects/data/libero_spatial", help='Path to the demo file')
    # args about resolution
    parser.add_argument('--camera_height', type=int, default=128, help='Camera height resolution')
    parser.add_argument('--camera_width', type=int, default=128, help='Camera width resolution')
    # args about list robots
    parser.add_argument('--robots', type=str, nargs='+', default=["Panda"], help='robots')
    args = parser.parse_args()


    # env 设置（尽量与原始录制一致的 camera）
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task_nums = len(task_suite.get_task_names())

    for i in range(task_nums):
        task = task_suite.get_task(i)
        demo_file_path = args.demo_path + '/' + task.name + "_demo.hdf5"
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": args.camera_height,
            "camera_widths": args.camera_width,
            "robots": args.robots,
        }
        env = OffScreenRenderEnv(**env_args)
        sim = env.sim

        # 可选：如果你知道想要的 camera 外参，可以在这里设置，否则使用 env 的默认
        try:
            camera_pos = np.array([0.6586131746834771, 0.0, 1.6103500240372423])
            camera_quat = np.array([
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ])
            cam_id = sim.model.camera_name2id("frontview")
            sim.model.cam_pos[cam_id] = camera_pos
            sim.model.cam_quat[cam_id] = camera_quat
        except Exception:
            print("Warning: failed to set camera pose; using environment defaults.")


        with h5py.File(demo_file_path, 'a') as f:
            if 'data' not in f:
                # print(f.keys())
                print('No data group in file:', demo_file_path)
                continue
            demo_keys = list(f['data'].keys())

            for demo_path in demo_keys:
                ds = f[f'data/{demo_path}/obs']
                print(f'Processing task: {task_suite.get_task(i).name}, demo: {demo_path}')

                # try joint_states, else fallback to robot_states -> EE
                joint_pos_seq = None
                if 'joint_states' in ds:
                    joint_pos_seq = ds['joint_states'][:]
                    print(f'  loaded joint_states shape {joint_pos_seq.shape}')
                else:
                    if 'robot_states' in ds:
                        print('  joint_states not found; will use robot_states (EE fallback)')
                    else:
                        print('  No joint_states or robot_states found; skipping demo')
                        continue

                # prepare scene once per demo
                blacken_scene_keep_robot(sim)

                ee_seq = None
                if 'robot_states' in ds:
                    robot_states = ds['robot_states'][:]
                    if robot_states.shape[1] >= 9:
                        ee_seq = [(rs[2:5], rs[5:9]) for rs in robot_states]

                # determine frame count
                if joint_pos_seq is not None:
                    T = len(joint_pos_seq)
                elif ee_seq is not None:
                    T = len(ee_seq)
                else:
                    print(f'  No per-frame data for {demo_path}, skipping')
                    continue

                img_masks = []
                masked_rgbs = []

                for t in range(T):
                    success = False
                    if joint_pos_seq is not None:
                        success = set_robot_qpos_from_jointpos(sim, joint_pos_seq, t)

                    if not success and ee_seq is not None:
                        ee_pos, ee_quat = ee_seq[t]
                        try:
                            sim.data.set_mocap_pos('gripper0_right_gripper', ee_pos)
                            sim.data.set_mocap_quat('gripper0_right_gripper', ee_quat)
                        except Exception:
                            try:
                                sim.data.set_mocap_pos('robot0_right_hand', ee_pos)
                                sim.data.set_mocap_quat('robot0_right_hand', ee_quat)
                            except Exception:
                                pass
                        for _ in range(5):
                            sim.step()

                    # render (agentview) and compute mask
                    rgb = sim.render(width=128, height=128, camera_name='frontview', depth=False)
                    if isinstance(rgb, (tuple, list)):
                        rgb = rgb[0]

                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    _, img_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

                    # compose masked_rgb from original stored agentview_rgb (keep robot pixels)
                    rgb_orig = ds['agentview_rgb'][t]
                    robot_bool = (img_mask == 0)
                    masked_rgb = np.zeros_like(rgb_orig, dtype=np.uint8)
                    try:
                        masked_rgb[robot_bool] = rgb_orig[robot_bool]
                    except Exception:
                        # channel-wise fallback
                        for c in range(rgb_orig.shape[2]):
                            ch = rgb_orig[:, :, c]
                            ch_masked = np.zeros_like(ch, dtype=np.uint8)
                            ch_masked[robot_bool] = ch[robot_bool]
                            masked_rgb[:, :, c] = ch_masked

                    img_masks.append(img_mask.astype(np.uint8))
                    masked_rgbs.append(masked_rgb.astype(np.uint8))

                # write back for this demo (overwrite rendered group if exists)
                data_grp = f[f'data/{demo_path}']
                if 'rendered' in data_grp:
                    del data_grp['rendered']
                rg = data_grp.create_group('rendered')

                if len(img_masks) > 0:
                    masks_arr = np.stack(img_masks, axis=0)
                    rg.create_dataset('img_mask', data=masks_arr, compression='gzip')
                else:
                    rg.create_dataset('img_mask', data=np.zeros((0,)), compression='gzip')

                if len(masked_rgbs) > 0:
                    rgb_arr = np.stack(masked_rgbs, axis=0)
                    rg.create_dataset('masked_rgb', data=rgb_arr, compression='gzip')
                else:
                    rg.create_dataset('masked_rgb', data=np.zeros((0,)), compression='gzip')

                rg.attrs['num_frames'] = masks_arr.shape[0] if len(img_masks) > 0 else 0
                print(f'Wrote rendered data for {demo_path} (frames={rg.attrs["num_frames"]})')
        
        #env close
        env.close()


    show_rendered_samples(demo_file=demo_file_path)
