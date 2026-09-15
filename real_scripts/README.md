# Real dual-UR7e runtime

当前实机入口只有两条：

- `live_dual_ur7e_obstacle_model.py`：front D455、双臂/真实 EE 网格剔除、
  LingBot-Depth、桌面物体 OBB，以及 v2 HTTP 安全快照；全程只读。
- `run_ur7e_vla_safety_executor.py`：消费联合 PI05 输出和 v2 快照，执行 12-DOF
  CBF-QP；默认 dry-run，未核验右 TCP 时禁止实机执行。

共享固定点 FK 在 `dual_ur7e_surface.py`，CBF 几何在 `real_cbf_qp.py`。默认
配置与完整命令见仓库根目录 [README](../README.md)。

标定脚本、点云工具和可视化脚本仍作为当前硬件维护工具保留，但不再构成另一套
训练/部署主线。
