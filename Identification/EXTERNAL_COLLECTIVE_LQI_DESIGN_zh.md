# 外部集电 + 4 输出 LQI 部署设计实验（Plan A）

## 1. 目的与契约

为把 `Identification` 的 LQI 控制器部署到 PX4（`px4_trans` 现有
`micoair_h743-v2_nncontrol` 框架），按方案 A 建立独立设计实验：上桨由外部
"飞手"（高度真值闭环）拥有，LQI 只输出下桨与三路舵机。该契约与现有神经策略
`residual_4` 的所有权边界一致，不改变飞手保上桨的安全语义。

```text
外部高度PID（真值高度/垂速，等价于飞手/悬停观察者）
        │  上桨油门 [0,1]
        ▼
  ┌──────────────────┐
  │ 4输出 LQI        │── 下桨 [0,1]、舵机1/2/3 [-1,1]
  │ 13状态命令观察者  │
  └──────────────────┘
```

控制器语义：

- 状态：`[roll_error, pitch_error, p, q, r, 上桨滤波, 下桨滤波,
  舵1/2/3滤波, integral_roll, integral_pitch, integral_yaw_rate]`，共 13 维；
- 五个执行器状态全部是命令驱动观察者：上桨状态由外部油门驱动，下桨与舵机状态
  由 LQI 最终限幅命令驱动，不使用 ESC 转速反馈和舵角反馈；
- 参考系与训练一致：世界系 NED、机体系 FRD、Hamilton wxyz（body→world）；
- 固定 500 Hz 单步时基，增益按 2 ms 离散化；
- 饱和时冻结积分，与原有 LQI 行为一致；
- 舵机观察者提供 `linear`（与增益合成严格一致）和 `nonlinear`（死区/回差/
  最大速率）两种模式，后者更接近真机。

## 2. 与现有链路的隔离

- 不修改 `Controller/src/flight_controller`（不注册新控制器类型，不影响 WebUI、
  Train、旧 LQI）；
- 不修改 `SimEnv`；
- 不修改 `px4_trans`；
- 新增内容：
  - `Identification/src/flight_identification/external_collective_lqi.py`
    （控制器、高度飞手、增益合成、配对评估 CLI）；
  - `Identification/configs/external_collective_lqi_design_v1.yaml`
    （复用 `sim2real_micro_coaxial` 仿真与 `lqi_sim2real_micro_coaxial` 权重）；
  - 本设计文档。

## 3. 增益合成

13 状态离散模型由标称 `sim2real_micro_coaxial` 参数生成
（`control_evaluation._discrete_model`，含积分增广）。把 B 矩阵去掉上桨列，
得到 4 输入模型 `(A, B_ls)`，用相同的 Q 和去掉上桨行的 R 求解 DARE：

```text
K4 = inv(R4 + B_ls^T S B_ls) B_ls^T S A          (4 x 13)
```

上桨状态（状态 5）由外部输入驱动，是不可控但稳定的模式；DARE 仍可解，
状态反馈通过下桨/舵机补偿上桨对机体的扰动。评估同时包含：

- `external4_linear`：重新合成的 K4，线性命令观察者；
- `external4_nonlinear`：K4 + 含死区/回差/速率限制的舵机观察者；
- `external4_dropped_nonlinear`：原 5 输入 K5 去掉上桨行后的 4 行增益
  （对比"直接裁剪"与"重新合成"）；
- `external4_oracle_nonlinear`：逐机体真值参数重新合成的 K4（结构上界）；
- `nominal5`：原 5 输出 LQI 悬停模式（配对基线）。

标称线性结果（与文档一致）：5 输入与 4 输入闭环极点半径均为约 `0.99650`，
上桨通道的角加速度效能很小，去掉该控制通道不改变线性标称极点；
但 K4 与 K5 裁剪行最大差约 `0.45`，非线性行为必须由配对评估决定。

## 4. 评估协议

- 参数组：从与 `lqr_sim2real_micro_repeated8_v1` 相同的条件可行域采样器
  （`_sample_sim2real_labels`）**重新采样**，不依赖旧 5 输出控制器生成的
  轨迹和其成功筛选，避免数据集偏置；
- 每变体重建 SimEnv，共享同一参数组、初始姿态/角速度、随机数序列
  （严格配对）；
- 初态：最大倾角 15°、三轴角速度 ±1 rad/s，时长 2 s；
- 指标：安全率、严格 2 s 收敛率、末态姿态误差/角速度中位数、
  饱和占比及其按参数组聚类的 bootstrap 95% CI、局部线性极点半径；
- 输出：`--output` 指向的 JSON 报告。

## 5. 已知边界（部署时仍需处理）

- 高度环使用真值高度；PX4 上对应 EKF `vehicle_local_position`，需验证估计
  延迟与噪声下的高度闭环；
- 观察者时间常数/斜率取标称值，真机需台架或 ESC telemetry 标定；
- 舵机安装拓扑（机尾 180°、右前 60°、左前 −60°）与 PX4 输出 201/202/203
  的物理方向必须一致，否则增益符号反向；
- 上桨反向旋转由 ESC/接线层保证，`PWM_MAIN_REV` 不能替代；
- 本实验只设计/验证控制器，`deployment_decision` 固定为 `design_only`，
  不授权任何实机输出。

## 6. 运行

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.external_collective_lqi \
  --experiment-config Identification/configs/external_collective_lqi_design_v1.yaml \
  --output Identification/runs/external_collective_lqi_design_v1/control_audit.json \
  --device cuda:0 --parameter-groups 128 \
  --evaluation-initial-conditions 8 --duration-s 2.0
```

小规模 CPU 冒烟：

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.external_collective_lqi \
  --experiment-config Identification/configs/external_collective_lqi_design_v1.yaml \
  --output /tmp/external_lqi_design_smoke.json \
  --device cpu --parameter-groups 8 \
  --evaluation-initial-conditions 4 --duration-s 2.0
```

## 7. 后续部署步骤（本实验通过后）

1. 固化 K4 + 观察者参数（trim、slope、tau、积分限幅）到 manifest，生成黄金向量；
2. 把 4 输出 LQI 填入 `px4_trans/px4/src/modules/nn_control/LqrControllerBackend.cpp`
   （上桨取 `rc_throttle` 映射，与 `NeuralControllerBackend` 相同）；
3. 通过现有 MAVLink 验证协议做 bit-exact 对比；
4. 按 SIL → HIL → 无桨台架 → 系留 → 受控试飞的安全阶梯放行，
   `actuatorOutputAllowed()` 保持 false 直到证据齐全。
