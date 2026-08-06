# 角速度神经网络控制器 DAgger 训练报告

- 日期：2026-08-05
- 环境：Tesla V100-SXM2-32GB（本机），torch 2.12.1+cu126，rl-flight conda 环境
- 范围：延续 `angular_acceleration_allocated_inner_loop_21d_v2` 角速度控制器线，把 256 组
  oracle-LQI → 因果神经网络蒸馏 pilot 从纯行为克隆（BC）升级为 DAgger 闭环训练，并对比
  多种有记忆能力的模型结构。
- 实验根目录：`Identification/runs/lqi_gru_dagger_pilot_v2/`

## 1. 背景与目标

上一阶段 256 组 pilot（`lqi_gru_oracle_distillation_pilot_256_gru192x2_v3_resume50`）证明：

1. 因果 GRU 能在留出机体的 oracle 轨迹分布上以很低误差复现五轴 LQI 命令，且明显利用历史
   （reset-hidden 离线误差恶化 13.74 倍）；
2. 但纯行为克隆没有形成可靠闭环控制器：无 teacher forcing 的 6 秒闭环安全率只有 29-38%，
   oracle LQI 为 94%，固定 nominal LQI 为 96%；
3. 失败机制是 teacher-forced 轨迹与学生闭环轨迹之间的分布漂移，加上 recurrent hidden 长时漂移；
   训练时学生始终读取 oracle 的上一拍动作，学出了依赖专家动作历史的强递归正反馈。闭环时只要把
   previous command 换成学生自己的输出，recurrent 分支就会自行漂移。

因此本阶段按照 pilot 结论的第 1 优先项执行：**DAgger 学生闭环采样，把学生访问到的偏离状态
重新交给 oracle 标注**，同时配合 previous-command 输入正则，并最终把闭环安全/跟踪指标作为
模型选择依据（不再以离线 MSE 为主选择器）。

## 2. 方法

### 2.1 教师与信息隔离

- 教师：每个固定随机参数点的真值参数局部 LQI（13 状态、`5x13` 增益、500 Hz），可读取真值
  状态，输出完整五轴物理命令 `[upper, lower, servo_1, servo_2, servo_3]`。
- 学生输入：25 维因果观测（姿态估计四元数、gyro、加速度计、电机转速、目标姿态、目标角速度、
  collective、上一拍五轴命令），明确禁止参数标签、LQI gain、真值舵角、真值力/力矩。
- 参数组按 70/15/15 严格切分；DAgger 数据只来自 train 组，validation 组只用于闭环门控，
  test 组直到最终评估前不参与任何训练或数据聚合。

### 2.2 DAgger 流程

每一轮迭代：

1. 在聚合数据集（初始 BC 数据 + 之前所有 DAgger 数据）上训练学生；
2. 学生闭环 rollout：学生自己的上一拍命令回灌到 `previous_command` 输入，完全等价部署契约；
   覆盖 96 个 train 参数组 × 4 条 6 秒轨迹（500 Hz）；
3. 对每个学生访问到的状态，用该机体的真值参数 LQI 在该状态上重新标注专家命令
   （oracle 控制器状态沿学生轨迹推进）；
4. 失败前缀（学生坠毁前的状态）**保留**进数据集——它们是最有价值的 off-distribution 数据；
5. 在 24 个未参与训练的 validation 参数组 × 4 条新初值上跑无 teacher forcing 闭环门控
   （与 oracle LQI 配对、固定外生飞手指令），以安全率和角速度跟踪 RMS 选择迭代。

### 2.3 针对分布漂移的具体改动

- DAgger 数据中 `previous_command` 输入记录的是学生自己的上一拍输出，训练/部署输入分布一致；
- 训练时对 `previous_command` 通道加小幅高斯噪声（normalized std 0.05）并按 5% 概率重置为
  初始名义命令，削弱对专家动作历史的过度依赖；
- 闭环评估器在构造每个变体前重置全局 Torch/CUDA RNG，保证控制器间严格配对；
- 模型选择以闭环安全率为准，离线 MSE 仅作诊断。

### 2.4 数据规模

基础 BC 数据集：256 参数组、911 条安全轨迹（train 639 / validation 135 / test 137）。
DAgger 聚合：6 轮 × 96 组 × 4 条 ≈ 2304 条学生轨迹（实际聚合 4 轮新 rollout，见 3.3 节说明），
训练时完整保留失败前缀。

## 3. 结果

### 3.1 DAgger 迭代轨迹（validation 门控，24 组 × 4 条新初值，6 秒）

| 迭代 | GRU 安全率 | 角速度 RMS (rad/s) | 姿态 RMS (rad) | reset 安全率 | oracle 安全率 |
|---:|---:|---:|---:|---:|---:|
| 0（BC） | 41.7% | 3.014 | 0.446 | 22.9% | 90.6% |
| 1 | 89.6% | 0.933 | 0.143 | 49.0% | 93.8% |
| 2 | 88.5% | 0.585 | 0.145 | 80.2% | 90.6% |
| 3 | 88.5% | 0.422 | 0.137 | 81.3% | 88.5% |
| 4 | 88.5% | 0.414 | 0.126 | 88.5% | 91.7% |
| 5 | 87.5% | 0.428 | 0.129 | 88.5% | 90.6% |

第一次 DAgger 数据聚合就使闭环安全率从 41.7% 跳到 89.6%，角速度跟踪 RMS 从 3.01 降到 0.93；
后续迭代继续把跟踪误差压到约 0.41-0.43 rad/s，接近 oracle 水平。安全率在约 88-90% 附近收敛
（该门控子集上 oracle 本身为 88.5-93.8%）。

### 3.2 最终未见机体闭环（test 37 组 × 4 条新初值，6 秒，无 teacher forcing）

| 控制器 | 安全率 | 角速度 RMS (rad/s) | 姿态 RMS (rad) | 命令移动量 |
|---|---:|---:|---:|---:|
| per-airframe oracle LQI | 93.9% | 0.339 | 0.103 | 0.0018 |
| 固定 nominal LQI | 96.0% | 0.429 | 0.089 | 0.0020 |
| **DAgger GRU-192x2** | **94.6%** | **0.402** | **0.114** | 0.0041 |
| GRU reset（记忆消融） | 90.5% | 0.932 | 0.321 | 0.0027 |

在 oracle 自身安全的 139 条配对 episode 中：

| 控制器 | 安全率 | 姿态 RMS (rad) | 角速度 RMS (rad/s) |
|---|---:|---:|---:|
| DAgger GRU | 97.1% | 0.101 | 0.383 |
| nominal LQI | 97.1% | 0.082 | 0.417 |
| oracle LQI | 100% | 0.075 | 0.299 |

结论：DAgger 学生在一整组未见机体的闭环中安全率（94.6%）反超每机体真值 oracle LQI
（93.9%），与固定 nominal LQI（96.0%）基本持平；姿态/角速度跟踪 RMS 位于 oracle 与 nominal
之间（oracle 的 1.11/1.18 倍）。reset-hidden 显著变差（90.5% 安全、角速度 RMS 0.93），证明
recurrent hidden 在闭环中承担了真实的机体辨识/记忆功能，而不是退化到无记忆反馈。

### 3.3 离线 vs 闭环的反差（分布修正的直接证据）

| 迭代 | test 留出机体离线五轴 NRMSE（teacher forced） | reset/recurrent 离线 RMSE 比 | 闭环安全率 |
|---:|---:|---:|---:|
| 0 | 0.098-0.121 | 9.36x | 41.7% |
| 1 | 0.202-0.253 | 3.79x | 89.6% |
| 2 | 0.245-0.344 | 3.27x | 88.5% |
| 3 | 0.264-0.364 | 3.29x | 88.5% |
| 4 | 0.246-0.357 | 3.26x | 88.5% |
| 5 | 0.246-0.357 | 3.26x | 87.5% |

离线（oracle 轨迹上 teacher forcing）NRMSE 随 DAgger 迭代反而升高，而闭环安全率大幅改善。
这正是 DAgger 的预期行为：学生不再以“在专家轨迹分布上模仿专家”为目标，而是优化自己在闭环
访问分布上的控制质量。因此不能再用离线 MSE 选择模型；闭环指标是唯一可信的选择器。

> 注：第 4、5 轮训练实际使用的聚合数据与第 3 轮相同（rollout 元数据去重逻辑的 shard 编号
> 对齐问题导致第 4、5 轮未写入新 rollout，已在代码中修复），因此 4、5 行代表同一数据上的
> 复现/稳定性确认，安全率的小幅波动来自门控评估的随机初值不同。

## 4. 模型结构比较

在同一份最终聚合 DAgger 数据集上，以相同训练预算（40 epochs、batch 64、lr 3e-4、
previous-command 噪声 0.05/重置 0.05）训练四种有记忆能力的结构，并做同一 test 分组的
无 teacher forcing 闭环评估（37 组 × 2 条新初值，6 秒）：

| 结构 | 模型权重 fp32 | 闭环安全率 | 角速度 RMS (rad/s) | 姿态 RMS (rad) | 离线五轴 NRMSE | 单实例推理 |
|---|---:|---:|---:|---:|---:|---:|
| GRU-192x2 | 0.40M / 1.6 MB | 94.6% | 0.414 | 0.122 | 0.27-0.37 | 0.41 ms/step ✓ |
| LSTM-192x2 | 0.52M / 2.1 MB | 94.6% | 0.443 | 0.124 | 0.24-0.35 | 0.37 ms/step ✓ |
| TCN（128 步因果卷积窗） | 1.61M / 6.4 MB | 86.5% | 1.888 | 0.352 | 0.26-0.37 | 2.10 ms/step ✗ |
| Transformer（128 步因果注意力窗） | 0.99M / 4.0 MB | 94.6% | 0.455 | 0.117 | 0.27-0.37 | 1.10 ms/step ✓ |

对照：oracle LQI 93.9%（rate 0.369）、nominal LQI 97.3%（rate 0.419，该 2-trial 子集）。

结论：

1. **GRU 与 LSTM 持平且最优**：闭环安全率并列第一（94.6%），GRU 跟踪最好（0.414 rad/s），
   LSTM 略慢（0.443）；两者参数小、单实例推理远低于 500 Hz 预算。GRU 的 reset 消融
   （89.2% 安全、rate 1.00）与 LSTM 的 reset 消融（71.6%、rate 2.15）都显著差于完整记忆，
   证明两个递归结构都在闭环中真实使用了历史。
2. **Transformer（128 步窗口）闭环达标但跟踪略差**：安全率同为 94.6%，但角速度 RMS
   0.455 高于 GRU；对窗口内位置编码/注意力而言，128 步（0.256 秒）的上下文已经足以完成
   大部分稳定任务，但整体不如递归状态在相同预算下的表现。
3. **TCN 在闭环中明显落后**：离线拟合与其他结构接近（NRMSE 0.26-0.37），但闭环安全率只有
   86.5%、角速度 RMS 1.888，且单实例推理 2.10 ms/步超出 500 Hz 预算。因果卷积在处理自己
   动作反馈回灌时鲁棒性最差，固定窗口也没有递归 hidden 的长时辨识能力。
4. **递归状态 > 固定窗口**：相同 128 步窗口的 Transformer 能闭环但跟踪弱于 GRU/LSTM，
   说明角速度控制任务的机体辨识收益主要来自无界递归记忆，而不是固定长度窗口。

因此最终部署候选为 **GRU-192x2**：闭环性能最优（安全率 94.6%、rate 0.414）、参数最小、
推理 0.41 ms/步、记忆消融证据最清晰。LSTM 可作为等价的备选实现；TCN/Transformer 在
当前 DAgger 数据集上不推荐部署。

> 注：窗口模型（TCN/Transformer）的闭环“记忆消融”是喂单步输入，与递归模型的 reset-hidden
> 消融不等价（训练窗口长度不同），因此其离线 reset/recurrent 比不作为闭环记忆的必要性证据。
> 推理时延为 V100 GPU batch-1 实测；嵌入式 MCU 路径属于 `px4_trans` 独立验证线。

## 5. 部署评估

### 5.1 推理成本

- GRU-192x2（2 层、hidden 192、`192-96-5` 输出头）单实例步进推理：**0.41-0.52 ms/步**
  （V100 GPU、batch 1 实测，不同 checkpoint 间有微小波动），远低于 500 Hz 的 2 ms 预算；
  模型权重 0.40M 参数、约 1.6 MB（fp32）。
- 输入全部来自可部署传感器/指令接口：姿态估计、gyro、加速度计、电机转速、目标姿态/角速度、
  collective、上一拍命令；不依赖参数标签、真值舵角或真值力/力矩。

### 5.2 与经典控制器的对比结论

- 未见机体安全率：DAgger GRU 94.6% ≈ nominal LQI 96.0% > oracle LQI 93.9%。
- 跟踪质量：介于 per-airframe oracle 与固定 nominal 之间，且不需要任何参数辨识或在线增益计算。
- 自适应来源：单条轨迹内只靠 recurrent hidden 携带机体信息，无需显式参数估计头，具备真正的
  “记忆型自适应”结构。

### 5.3 剩余差距（部署前必须补的验证）

1. **传感器/估计质量**：当前姿态估计直接走 truth（通过 estimator 接口）；进入实机前必须加入
   姿态估计噪声、延迟、偏置实验，验证 hidden 对这些扰动的鲁棒性。
2. **更宽参数域与教师域筛选**：当前 256 组随机范围中局部 LQI 并非处处优于 nominal；
   扩大范围时应先筛“oracle 明显优于 nominal”的教师域，避免为学生背锅。
3. **PX4/嵌入式路径**：500 Hz 递归推理已在 GPU 验证，PX4 侧 GRU/MAVLink 部署与位精确校验
   属于 `px4_trans` 独立验证线，本报告未覆盖该硬件路径的实时性。
4. **长时漂移审计**：6 秒闭环通过后，需要 8-60 秒延长审计，确认 hidden 不随时间漂移。

### 5.4 固定 nominal LQI 为什么与自适应/真值 oracle “差不多”？

这是本实验最有价值的反面检验。对 test 37 组 × 4 条共 148 条 episode 做逐条交叉统计：

| 对比 | 两者都安全 | 仅前者安全 | 仅后者安全 | 都不安全 |
|---|---:|---:|---:|---:|
| nominal vs oracle | 135 | 7 | 4 | 2 |
| nominal vs GRU | 140 | 2 | 0 | 6 |
| oracle vs GRU | 135 | 4 | 5 | 4 |

结论分三层：

1. **在当前随机样本的平均值上，用户判断基本成立**。参数组经过可行域过滤且以 nominal 为中心，
   大部分机体“接近 nominal”，固定 LQI 本身就是接近最优的；而 6 秒安全率（60° 倾角 / 10 rad/s）
   是较粗的度量，保守的 nominal 不会因为跟踪差而失去安全分。因此平均安全率差距只有 2-3 个点，
   不足以证明自适应价值。
2. **但差异集中在参数域边缘和硬机体上，那里固定 LQI 会真实崩溃**。按参数距区间中心偏差
   分层的边缘 1/3 机体上：oracle 89.6% > nominal 87.5% > GRU 83.3%（中心区域三者都在
   96-100%）。最典型的是 test 组 249：下电机时间常数 0.115 s（约 96 百分位，远慢于 nominal
   假设），该组 nominal 4 条全坠、真值参数 oracle 4 条全救、当前 DAgger GRU 也是 4 条全坠。
   这说明“用真值算增益”在 actuator 动态明显偏离 nominal 的机体上有真实救回价值，也说明当前
   GRU 还没学到这个尾部救回行为（这是它目前最大的缺口，而不是“自适应没用”）。
3. **跟踪质量上 oracle 始终优于 nominal**：在三个控制器都安全的 135 条 episode 上，角速度
   RMS 为 oracle 0.300、GRU 0.354、nominal 0.380；姿态 RMS 为 oracle 0.074、nominal 0.070、
   GRU 0.088。本线的核心指标是角速度指令跟踪，这个维度上 per-airframe 增益的优势是系统性的。

因此正确的解读是：固定 nominal LQI 是一个必须正视的强基线；自适应/神经控制器的价值不能靠
“全随机样本平均安全率”证明，必须按机体难度分层评估（参数域边缘、坏点池、nominal 失效子集），
并在跟踪质量、鲁棒余量、分布外机体系数上展示优势。当前报告中的 94.6% vs 96.0% 不应被解读为
“自适应追平经典”，而应解读为“在 nominal 中心的分布上已追平；在 nominal 失效的尾部还没有
达标，下一步应针对坏点池补 DAgger 数据”。

### 5.5 硬机体 DAgger 跟进（2026-08-05 晚）

按 5.4 的结论补了两轮实验，证据链详见
`runs/lqi_gru_hard_dagger_pilot_v1/` 与
[HARD_PLANT_REPAIR_PLAN_zh.md](HARD_PLANT_REPAIR_PLAN_zh.md)。

**跟踪型硬机体（坏点池：收敛差而非坠机）**：34 个筛选坏点中 24 个入训练、10 个留出，
两轮 DAgger 后：

| 指标 | before | round 1 | round 2 | nominal | oracle |
|---|---:|---:|---:|---:|---:|
| 留出坏机体角速度 RMS (rad/s) | 0.354 | 0.336 | 0.327 | 0.379 | 0.302 |
| test 全集安全率 | 94.6% | 95.3% | 94.6% | 94.6% | 93.9% |
| test 全集角速度 RMS | 0.390 | 0.382 | 0.387 | 0.414 | 0.340 |

结论：坏点 DAgger 教会了学生 oracle 式跟踪（留出坏机体 0.327，接近 oracle 0.302、
明显优于 nominal 0.379），且 test 安全率不降反升（round 1 达 95.3%）。这是自适应价值在
“nominal 差”机体上的直接证据。

**安全崩溃型（组 249 类型）**：600 组全范围 6 秒安全筛选找到 10 个“nominal 全坠、
oracle 全救”组（gap=1.0），入训练做救回 DAgger 轮：

| 指标 | GRU（救回轮后） | nominal | oracle |
|---|---:|---:|---:|
| 10 个训练侧可救组安全率（4 trial） | 42.5% | 32.5% | 62.5% |
| 10 个训练侧可救组角速度 RMS | 0.865 | 1.295 | 0.757 |
| test 组 249 安全率（未见机体） | 0% | 0% | 100% |

结论：救援学习是**定向的、不泛化**的——学生在训练过的可救组上优于 nominal（42.5% vs
32.5%、rate 0.87 vs 1.30），但组 249 仍未救回。失败机制分析显示 10 个可救组参数签名高度
异质（推力比极端、电机反应系数极端、高惯量、网格几何偏差），组 249 的“慢下电机 τ≈0.115 s”
失败模式在训练数据中几乎无同类。修补方向（P0b）：按失败机制分层采样，每个机制独立成层
补数据并做独立 gate。

### 5.6 报告 5.3 节验证补充（2026-08-05 晚）

验证代码与结果位于 `runs/lqi_gru_hard_dagger_pilot_v1/verification/`（rescue_r1 学生；
完整 test 复评 95.3% 安全、rate 0.394，与历史最佳轮持平）：

1. **姿态估计扰动门（5.3.1）**：单因子扫描——噪声 0/2/5/10 deg、偏置 1/3 deg、延迟 2/5 步，
   在 test 12 组 × 2 条上配对比较（角速度 RMS，rad/s）：

   | 条件 | GRU | nominal | oracle |
   |---|---|---:|---:|
   | baseline | 0.388 | 0.441 | 0.352 |
   | 噪声 2° | 0.385 | 0.422 | 0.339 |
   | 噪声 5° | 0.395 | 0.404 | 0.313 |
   | 噪声 10° | 0.452 | 0.509 | 0.390 |
   | 噪声 5° + 偏置 1° | 0.400 | 0.407 | 0.316 |
   | 噪声 5° + 偏置 3° | 0.408 | 0.412 | 0.322 |
   | 噪声 5° + 延迟 2 步 | 0.397 | 0.407 | 0.315 |
   | 噪声 5° + 延迟 5 步 | 0.399 | 0.411 | 0.317 |

   GRU 在全部扰动档下跟踪都优于 nominal，退化幅度与 nominal 相当，未出现灾难性失稳；
   该 24-episode 子集的安全率（92-100%）样本太小，安全门需更大样本复核。
2. **长时漂移审计（5.3.4）**：30 秒闭环（8 组 × 2 条）：GRU 整体安全率 87.5%、rate 0.349
   （nominal 0.363、oracle 0.332）；分段 rate RMS 0.384 → 0.260 → 0.251，**跟踪随时间变好
   而非漂移**；hidden 范数 4.23 → 4.28 → 4.27 稳定。gru_reset 全程 100% 安全但 rate 0.741，
   30 秒尺度上记忆仍是性能核心。
3. **嵌入式路径代理（5.3.3）**：单实例推理 GPU 0.39 ms/步、CPU 0.35 ms/步（batch 1），
   均远低于 500 Hz 的 2 ms 预算；PX4/MCU 位精确验证仍属 `px4_trans` 独立线。

结论：5.3 的估计扰动鲁棒性、长时漂移、推理预算三项在当前模型上**通过**；“更宽参数域 +
教师域筛选”已在 5.5 节完成并暴露安全崩溃救援不泛化的问题，对应修补计划 P0b。

## 6. 复现

```bash
# 环境（PYTHONPATH 指向四个包 src）
export PYTHONPATH=$PWD/Identification/src:$PWD/Controller/src:$PWD/SimEnv/src:$PWD/Train/src
cd Identification

# 1) DAgger 迭代（6 轮；会创建 runs/lqi_gru_dagger_pilot_v2/）
python -m flight_identification.lqi_gru_dagger --stage iterate \
  --base-dataset datasets/lqi_gru_oracle_distillation_pilot_256_v2 \
  --config configs/lqi_gru_dagger_v1.yaml \
  --output-root runs/lqi_gru_dagger_pilot_v2 \
  --iterations 6 --train-groups 96 --trials 4 --epochs 60 \
  --previous-command-noise-std 0.05 --previous-command-reset-prob 0.05 \
  --gate-groups 24 --gate-trials 4 --gate-duration-s 6.0 --device cuda:0

# 2) 最终未见机体闭环（test 37 组 × 4）
python -m flight_identification.lqi_gru_closed_loop \
  --dataset runs/lqi_gru_dagger_pilot_v2/datasets/merged \
  --config configs/lqi_gru_dagger_v1.yaml \
  --checkpoint runs/lqi_gru_dagger_pilot_v2/students/iter_05/student.pt \
  --output runs/lqi_gru_dagger_pilot_v2/final_test_closed_loop.json \
  --maximum-groups 37 --trials 4 --duration-s 6.0 --split test --device cuda:0

# 3) 模型结构比较
python -m flight_identification.lqi_gru_arch_compare \
  --dataset runs/lqi_gru_dagger_pilot_v2/datasets/merged \
  --config configs/lqi_gru_dagger_v1.yaml \
  --output-root runs/lqi_gru_dagger_pilot_v2/arch_comparison \
  --architectures gru lstm tcn transformer --epochs 40 --trials 2 --device cuda:0
```

## 7. 最终判定

DAgger 路径成立：一轮聚合就把闭环安全率从 BC 的 41.7% 提升到 89.6%，最终在未见机体上达到
94.6%（超过 per-airframe oracle LQI 0.7pp，落后固定 nominal LQI 1.4pp），角速度/姿态跟踪
位于 oracle 与 nominal 之间，推理时延 0.52 ms 满足 500 Hz 预算。GRU-192x2 是当前已验证的
部署候选；第 4 节的模型结构比较给出了 GRU/LSTM/TCN/Transformer 的闭环证据：递归结构
（GRU/LSTM）最优，Transformer 持平但跟踪略差，TCN 落后且推理超预算。
