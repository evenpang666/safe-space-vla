请帮我修改当前的VLA 安全模块，名称为 SafetyFlowPointModel。该模块采用“Prefix-Conditioned Point Encoder + Flow Matching Point Head”的结构，用于根据 VLM 输出的 prefix_token 和当前机械臂局部点云，生成未来 n 步机械臂局部点云轨迹偏移量。

一、任务背景

我正在构建一个接入 VLA 模型后的安全建模模块。VLM backbone 已经输出 prefix_token，该 token 包含任务语义、视觉上下文或高层规划信息。当前系统还可以获得当前时间步的机械臂局部点云 P_arm。

本模块的目标是学习未来 n 步机械臂局部点云的运动趋势。模型不直接输出动作，也不直接输出未来点云绝对坐标，而是使用 Flow Matching 建模未来点云轨迹偏移量。

模型输入仅包括：

1. VLM 输出的 prefix_token：
   prefix_tokens: [B, L, d_vlm]

2. 当前机械臂局部点云：
   arm_points: [B, K, 3 + C_arm]

3. Flow Matching 中的 noisy future point offset：
   x_s: [B, n_future, K, 3]

4. Flow time：
   s: [B] 或 [B, 1]

模型输出：

1. 预测的 velocity field：
   v_pred: [B, n_future, K, 3]

Flow Matching 的目标定义为：

x_1 = ΔP_future = P_arm_future - P_arm_current

x_0 ~ N(0, I)

x_s = (1 - s) * x_0 + s * x_1

v_target = x_1 - x_0

模型学习：

v_theta(x_s, s, prefix_tokens, arm_points) ≈ v_target

训练损失：

L_FM = MSE(v_pred, v_target)

二、整体结构

请实现如下模块：

1. Arm Point Token Embedding

   * 输入当前机械臂局部点云：
     arm_points: [B, K, 3 + C_arm]

   * 每个点的前三维是 xyz，其余维度是可选点特征。

   * 用 MLP 将点特征映射到 hidden_dim。

   * xyz 通过单独的 positional MLP 映射到 hidden_dim。

   * point embedding = feature_embedding + xyz_pos_embedding + arm_modality_embedding。

   * 输出：
     H_arm: [B, K, hidden_dim]

2. Prefix Token Adapter

   * 输入：
     prefix_tokens: [B, L, d_vlm]

   * 通过 Linear 投影到 hidden_dim。

   * 加 learnable prefix modality embedding。

   * 可选加 1D token position embedding。

   * 输出：
     H_prefix: [B, L, hidden_dim]

3. Multimodal Transformer Encoder

   * 将机械臂点云 token 和 prefix token 拼接：

     encoder_input = concat(H_arm, H_prefix)

   * 输入 TransformerEncoder。

   * 输出 memory。

   * 同时 split 出：

     H_arm_enc: [B, K, hidden_dim]

     H_prefix_enc: [B, L, hidden_dim]

   * 其中 H_arm_enc 表示当前机械臂局部几何状态，H_prefix_enc 表示 VLM prefix 条件信息。

4. Flow Matching Point Head

   * 输入 noisy future offset：

     x_s: [B, n_future, K, 3]

   * 将其 reshape 为：

     x_s_flat: [B, n_future * K, 3]

   * 用 MLP_x 映射到 hidden_dim。

   * 加 future step embedding：表示第几个未来时间步。

   * 加 point identity embedding：表示第几个机械臂采样点。

   * 加 flow time embedding：表示 Flow Matching 时间 s。

   * 得到 decoder query：

     Z_s: [B, n_future * K, hidden_dim]

5. Transformer Decoder

   请实现自定义 decoder layer，不要直接使用 nn.TransformerDecoder。

   每一层结构如下：

   a. Self-Attention over noisy future point tokens

   Query = Z_s
   Key = Z_s
   Value = Z_s

   用于建模未来点云轨迹内部的时空关系，包括同一时间步不同点之间的空间关系，以及同一点跨未来时间步的运动关系。

   b. Geometry Cross-Attention to current arm point tokens

   Query = future point tokens
   Key = H_arm_enc
   Value = H_arm_enc

   用于读取当前机械臂局部点云的几何状态。

   c. Prefix Cross-Attention to VLM prefix tokens

   Query = updated future point tokens
   Key = H_prefix_enc
   Value = H_prefix_enc

   用于读取 VLM prefix 中的任务语义、视觉上下文或高层规划信息。

   d. FFN

   e. 每个 attention 和 FFN 后都需要 residual connection 和 LayerNorm。

   Decoder layer 的整体顺序为：

   Self-Attention
   → Geometry Cross-Attention
   → Prefix Cross-Attention
   → FFN

6. Velocity Head

   * 将 decoder 输出：

     Z_out: [B, n_future * K, hidden_dim]

   * 通过 MLP 映射到：

     v_pred_flat: [B, n_future * K, 3]

   * reshape 为：

     v_pred: [B, n_future, K, 3]

三、需要实现的文件结构

请生成一个完整、可运行、结构清晰的代码文件，并保证其可以接在pi05模型中，命名为：

safety_flow_point_model.py

文件中至少包含以下类和函数：

1. SinusoidalTimeEmbedding

   * 输入 s: [B] 或 [B, 1]
   * 输出 [B, hidden_dim]
   * 用于 Flow Matching 的连续时间嵌入。

2. MLP

   * 通用多层感知机模块。

3. ArmPointTokenEmbedding

   * 输入：
     arm_points: [B, K, 3 + C_arm]

   * 输出：
     H_arm: [B, K, hidden_dim]

4. PrefixTokenAdapter

   * 输入：
     prefix_tokens: [B, L, d_vlm]

   * 输出：
     H_prefix: [B, L, hidden_dim]

5. PrefixPointEncoder

   * 输入：
     H_arm
     H_prefix

   * 拼接后送入 TransformerEncoder。

   * 输出：
     H_arm_enc
     H_prefix_enc
     memory

6. FlowPointDecoderLayer

   * 自定义 decoder layer。

   * 包含：
     self_attn
     geom_cross_attn
     prefix_cross_attn
     ffn
     layer norms

7. FlowPointHead

   * 输入：
     x_s
     s
     H_arm_enc
     H_prefix_enc

   * 输出：
     v_pred

8. SafetyFlowPointModel

   * 总模型。

   * forward 输入：
     arm_points
     prefix_tokens
     x_s
     s

   * forward 输出：
     v_pred

9. flow_matching_loss

   * 输入：
     v_pred
     x_1
     x_0

   * 输出：
     MSE(v_pred, x_1 - x_0)

10. sample_flow_matching_batch

* 输入真实未来点云偏移 x_1。

* 随机采样 x_0 和 s。

* 返回：
  x_s
  s
  x_0
  v_target

11. euler_sample

* 用训练好的模型从 Gaussian noise 出发，用 Euler ODE 积分生成未来点云偏移。

* 输入：
  model
  arm_points
  prefix_tokens
  n_steps
  n_future
  K

* 输出：
  delta_p_future: [B, n_future, K, 3]

四、张量形状约定

请在代码中用注释明确标注每一步张量形状。

主要张量形状如下：

prefix_tokens: [B, L, d_vlm]
arm_points:    [B, K, 3 + C_arm]
x_s:           [B, n_future, K, 3]
s:             [B] or [B, 1]

v_pred:        [B, n_future, K, 3]

其中：

B = batch size
L = prefix token 数量
K = 机械臂局部点云采样点数量
d_vlm = VLM token hidden dimension
n_future = 未来预测步数
C_arm = 每个机械臂点的额外特征维度，可为 0

五、实现细节要求

1. 使用 PyTorch。
2. 使用 batch_first=True。
3. 所有 attention 使用 nn.MultiheadAttention。
4. 所有模块要支持 GPU。
5. 不要依赖 PyTorch Geometric。
6. 当前版本不需要实现 kNN local attention，先实现全局 geometry cross-attention。
7. 代码应当可以直接 import。
8. 提供一个 main 测试函数，用随机张量测试 forward、loss 和 sampling 是否能跑通。
9. main 中请使用如下默认超参数：

B = 2
K = 128
L = 64
C_arm = 0
d_vlm = 768
n_future = 8
hidden_dim = 256
num_encoder_layers = 4
num_decoder_layers = 4
num_heads = 8

10. 测试时构造：

prefix_tokens: [B, L, 768]
arm_points: [B, K, 3]
x_1: [B, n_future, K, 3]

然后执行：

x_s, s, x_0, v_target = sample_flow_matching_batch(x_1)

v_pred = model(
arm_points=arm_points,
prefix_tokens=prefix_tokens,
x_s=x_s,
s=s
)

loss = flow_matching_loss(v_pred, x_1, x_0)

delta_pred = euler_sample(
model=model,
arm_points=arm_points,
prefix_tokens=prefix_tokens,
n_steps=10,
n_future=n_future,
K=K
)

打印：

v_pred.shape
loss.item()
delta_pred.shape

六、建模注意事项

1. 模型预测的是未来机械臂局部点云轨迹偏移量的 velocity field，不是直接预测未来点云绝对坐标。

2. 真实目标 x_1 应定义为：

   x_1 = P_arm_future - P_arm_current

3. 当前版本假设机械臂局部点云采样点具有固定身份，即第 i 个点在不同时间步对应机械臂表面或 link mesh 上的同一个采样点。因此可以使用 MSE 监督。

4. 如果未来点云点顺序不固定，后续应改用 Chamfer Distance，但当前实现先假设点顺序固定。

5. 当前版本先不实现 collision loss、smoothness loss、rigid consistency loss，但请在代码中预留接口或注释说明未来可以加入：

   * collision_loss
   * smoothness_loss
   * rigid_link_consistency_loss

6. 由于输入中没有显式场景点云、关节状态或 VLA 原始动作，模型只能根据 VLM prefix_token 和当前机械臂局部点云预测未来点云偏移。因此代码中不要定义 scene_points、q、a_vla 相关输入或模块。

7. 该模块后续可以作为 VLA safety adapter 的一部分，用生成的 delta_p_future 估计未来机械臂局部点云运动趋势，并进一步用于碰撞风险判断、轨迹约束或动作修正。
