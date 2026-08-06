# 真机闭环日志离线辨识方案

## 使用场景

本方案不在飞行过程中运行辨识网络，也不在线更新 LQR 增益。目标流程是：

```text
完成一次或多次正常闭环飞行
  -> 导出日志
  -> 地面 GPU/工作站离线辨识
  -> 生成等效动态和候选 LQI 增益
  -> 仿真/HIL/人工审核
  -> 下一次飞行才可能使用经验证的增益
```

因此模型可以读取完整序列、使用双向网络并联合处理多次飞行，不需要满足机载推理延迟、显存或因果性约束。

## 4 输出部署结构（v4 正式实验，2026-08-05）

按最终部署结构，LQR 已改为 **4 输出**：`[lower_motor, servo_1, servo_2,
servo_3]`；上桨油门由飞手控制，训练/评估时用虚拟飞手（油门域 PID 定高）完成
闭环，下桨基准跟随飞手总距、LQR 只输出差动。数据集
`lqr_sim2real_micro_offline_logs_v4`（8192 机体 × 8 次，59.3% 收敛）在该结构
下生成并完成正式训练（MLP v7 / TCN v7 / BiGRU v7b）。

最终结果（949 组 × 8 初值 × 2 s × 2 种子）：三模型 hybrid
（TCN roll + BiGRU pitch + MLP yaw）80% 混合增益收敛率 79.1% / 78.7%，相对
固定复合标称（68.5% / 68.9%）提升 **+10.6 / +9.8 pp**（组聚类 95% CI 不含
零），安全非劣，8 s 长时审计 100% 收敛。完整报告见
[OFFLINE_4OUT_COMPOSITE_REPORT_zh.md](OFFLINE_4OUT_COMPOSITE_REPORT_zh.md)。

## 数据采集约束

训练和真机数据必须来自闭环控制下的飞行。禁止把以下内容作为辨识输入数据：

- 单独给某一个舵面施加理想阶跃；
- 开环执行器扫频；
- 与真实飞行任务不相符的逐通道 PRBS；
- 直接记录仿真真值舵角并把它作为网络输入；
- 对每个随机机体使用真值参数重新综合控制器后再采集。

允许使用真实可执行的闭环初始条件和机动：

- 大幅随机初始 roll/pitch；
- 正负三轴初始角速度，尤其是 yaw rate；
- 标称 LQR 闭环恢复到零 roll/pitch 和零 yaw rate；
- 实际任务允许的多轴姿态或 yaw-rate 指令；
- 控制器自然产生的大幅舵指令和短时饱和。

新的 `lqr_sim2real_micro_offline_logs_v2.yaml` 使用分层多轴初始条件。同一机体的八次飞行在倾角幅值、倾角方向和 `p/q/r` 幅值上做 Latin-hypercube 覆盖，但五路执行器始终只接收标称 LQR 的闭环输出。配置不会向控制器输出叠加测试信号。

正式数据已生成 8192 个机体参数组，每组八次闭环飞行。初始倾角为 5–25 deg，初始角速度范围为 `p/q <= 1.5 rad/s`、`r <= 2.0 rad/s`。环境运行 3 s 判断是否收敛，保存前 2 s、500 Hz 的完整辨识日志。未收敛试飞保留失败元数据，但不进入正常响应回归。

65536 次闭环飞行中有 33977 次在 3 s 内满足收敛条件。训练、验证和测试 split 分别保留了 4037、850、860 个至少有一次成功日志的独立机体；保留机体的成功日志数中位数为 8。正常回归仍按机体参数组切分，不会把同一机体的不同飞行泄漏到不同 split。

## 真机日志契约

离线工具使用无 pickle 的 NPZ 文件：

```text
features:      float32 [flight, time, 14]
feature_names: unicode [14]
sample_hz:     scalar
valid_mask:    optional bool [flight, time]
```

14 个通道为：

```text
attitude_q_tilt.w/x/y/z
angular_velocity_b.x/y/z
motor_speed.upper/lower
command.motor_upper/lower
command.servo_1/2/3
```

`attitude_q_tilt` 是去除不可观测 yaw 后的倾斜姿态四元数，不是完整航向姿态。`command.*` 必须是控制器实际发送给执行器的最终归一化命令，而不是姿态参考或分配前力矩。舵角不需要反馈。

当前工具要求日志采样率、单段样本数和检查点一致，避免未经验证的重采样改变 10–40 ms 执行器动态。后续接入具体真机日志格式时，应在独立适配器中完成时间同步、坐标转换、去 yaw、命令单位转换和窗口提取，再写成上述规范 NPZ。

## 离线模型

当前实现支持三种可公平比较的每次飞行编码器：

| 架构 | 采样 | 参数量 | 旧数据 8 段 NRMSE | 新数据验证 8 段 NRMSE | 新数据测试 8 段 NRMSE | 测试平均 R2 | 测试 yaw R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 展平 MLP | 100 Hz | 1,837,997 | 0.3460 | 0.3115 | 0.3082 | 0.627 | 0.452 |
| 全序列 TCN | 500 Hz | 306,861 | 0.3526 | 0.3283 | 0.3190 | 0.682 | 0.443 |
| 双向 GRU | 500 Hz | 292,141 | 0.3495 | 0.3248 | 0.3240 | 0.609 | 0.373 |

三种架构都在新分层数据上重新训练（MLP 和 TCN 使用 seed 20260825；GRU 另建
`sim2real_offline_logs_step_response_bigru_500hz_v7`）。1/2/4/8 次飞行在三种
架构上都单调改善，8 次飞行后测试完整响应 NRMSE 分别为 MLP `0.3082`、TCN
`0.3190`、GRU `0.3240`，明显优于旧数据基线。逐轴结果显示 yaw 仍是最弱轴，但
分层 yaw-rate 初始条件把测试 yaw 平均 `R2` 从旧数据的 `0.306` 提升到 MLP
`0.452`、TCN `0.443`、GRU `0.373`，说明可实机复现的分层 yaw-rate 初始条件确实
补充了辨识信息。TCN 在 roll/pitch 上最优（测试 `R2` 0.783/0.820），MLP 在 yaw
上最优（0.452），GRU 未在任何轴超过二者，因此最终 hybrid 仍按验证集逐轴 `R2`
选择为 TCN(roll/pitch) + MLP(yaw)。

同一机体的多次飞行先独立编码，再通过集合统计聚合，因此飞行顺序不影响结果。日志多于检查点训练的最大次数时，离线工具按三轴角速度 RMS 和控制指令运动量选择信息量最高的记录；这只是选择正常闭环日志，不会制造额外激励。

## 严格配对非线性审计（最终结果）

评估使用测试集全部 `860` 个未见机体参数组、每组 `8` 个初始条件（最大倾角
`15 deg`、最大初始角速度 `1 rad/s`、`2 s` 仿真），并独立复现两个评估随机种子。
所有控制器变体共享完全相同的机体、初始状态和随机数序列；置信区间按机体参数组
聚类。

| 控制器 | seed 20260826 收敛率 | seed 20260829 收敛率 | 相对复合标称（两种子） |
| --- | ---: | ---: | --- |
| 原始标称 LQI | 57.94% | 57.14% | -2.56 / -2.86 pp |
| 固定复合标称 LQI | 60.49% | 60.00% | 基准 |
| hybrid 预测增益 50% | 66.90% | 67.14% | +6.41 / +7.14 pp |
| hybrid 预测增益 80% | 69.08% | 69.55% | +8.59 / +9.55 pp |
| hybrid 预测增益 90% | 69.64% | 69.88% | +9.14 / +9.88 pp |
| hybrid 全预测增益 | 70.00% | 70.09% | +9.51 / +10.09 pp |
| 复合真值增益（oracle） | 70.19% | 70.07% | +9.69 / +10.07 pp |
| 原始真值增益（oracle） | 70.81% | 70.60% | +10.32 / +10.60 pp |

hybrid 全预测增益相对复合标称的收敛提升在两次种子下分别
`+9.51`（95% CI `[8.04, 11.03]`）和 `+10.09`（95% CI `[8.58, 11.61]`）
个百分点，置信区间均不含零；胜/败组数 858/204 与 894/200。安全率从约
`99.6%` 提高到 `99.8%`，平均饱和占比下降约 `0.9-1.0` 个百分点。全预测增益几乎
追平复合 oracle（约 `+9.7/10.1` pp），说明当前误差主要已不是辨识网络容量，而
是模型余量。50% 混合比例也能提供 `+6.4/7.1` pp 的稳健收益，适合作为飞行中限速
插值的目标上限。

产物仍标记为 `deployment_mode = research_only`、`gain_updates_enabled = false`。
这两个种子满足文档要求的“跨数据种子复现”：相对固定复合标称 LQI 的安全非劣和
收敛优势均可复现，可以作为进入 HIL 验证的候选，但还不是飞行授权。

## 胜/败分层分析

对两次 final 审计的逐组收敛 delta 按物理参数（审计内 `parameter_stratified`）、
原始日志信息量分数和主检查点输入 OOD z-score（`offline_quality_stratification`
工具）分层：

- hybrid 全预测增益在约 `39-40%` 的机体参数组上优于复合标称、约 `12%` 上劣于
  标称，其余不变。
- 与日志信息量的相关系数两次种子均为 `-0.08 ~ -0.04`（可忽略），低信息量组的
  平均收益 `+0.111/+0.133` 甚至略高于高信息量组的 `+0.095/+0.097`；
- 与输入 z-score p95 的相关系数为 `+0.03 ~ +0.08`（弱正），低/高 OOD 组的平均
  收益都在 `+0.09 ~ +0.11`。

即 hybrid 优势不依赖“日志恰好很有信息量”或“输入恰好接近训练分布”，在全部
信息量/OOD 分位都能复现正收益。旧的“物理参数模型”路线（legacy）则在
高信息量/高 OOD 组明显退化（z 相关系数约 `-0.17 ~ -0.20`），这正是复合阶跃
响应路线替代它的原因。

## 离线推理

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.offline_log_inference \
  --checkpoint Identification/runs/<run>/identifier.pt \
  --logs /path/to/closed_loop_flights.npz \
  --experiment-config Identification/configs/lqr_sim2real_micro_offline_logs_v2.yaml \
  --output /path/to/offline_identification.json \
  --secondary-checkpoint Identification/runs/sim2real_offline_logs_step_response_tcn_500hz_v6/identifier.pt \
  --secondary-output-axes roll,pitch \
  --device cuda:0 --analysis-gain-blend 0.9
```

推荐的离线组合使用 TCN 的 roll/pitch 响应和 MLP 的 yaw 响应（验证集逐轴 `R2`
确定；BiGRU 未在任何轴胜出）。输入使用一份原始 500 Hz NPZ，工具把原始序列交给
TCN，并严格按训练规则做整数 5 倍抽取后交给 100 Hz MLP。

输出包含选中的日志编号、信息量分数、输入分布 z-score、预测响应、固定滤波器系数、标称 19 状态增益、完整预测增益和 90% 分析增益。所有输出固定标记为：

```text
deployment_mode = offline_analysis_only
gain_accepted_for_flight = false
```

离线工具计算的极点半径只针对辨识模型，不是真实未知机体的稳定性证明。候选增益必须在独立随机初始条件仿真、模型不确定性包线和 HIL 中验证后，才能形成下一次飞行使用的发布产物。

## 评估结论与剩余工作

离线分析阶段的步骤已全部完成：

1. 三种架构（MLP/TCN/BiGRU）均已在新分层闭环数据上重新训练并给出 1/2/4/8 次
   飞行响应误差与逐轴 `R2`，yaw 重点检查完成。
2. 候选模型已在 `393` 组 × 8 初值上扫描 `0.0–1.0` 混合比例，并在最终 `860`
   组 × 8 初值 × 2 种子上完成严格配对非线性评估。
3. 胜/败已按物理参数、日志信息量和输入 OOD 分数分层，结论是 hybrid 优势在各
   分位均成立。
4. 相对固定复合标称 LQI 的安全非劣和收敛优势已在两个评估种子上复现，满足生成
   HIL 候选的条件。

推荐产物：TCN(roll/pitch, `sim2real_offline_logs_step_response_tcn_500hz_v6`)
+ MLP(yaw, `sim2real_offline_logs_step_response_mlp_v6`)，分析默认 `90%`
增益混合，配 `sim2real_offline_hybrid_v6/manifest.json` 和验证集校准的质量
门限（`offline_log_quality_calibration_v1.json`）。飞行前仍需完成：

1. 把分析增益注入飞行控制环时采用按时间限速的增益插值，而不是一次跳变，并验证
   切换瞬态；
2. 在飞行循环中加入辨识不确定度、持续激励度和饱和率门控，信息不足时保持上一次
   已接受增益；
3. 在传感器噪声、延迟、模型外扰动和参数缓慢漂移下重复严格配对评估；
4. 对 DARE 失败、增益非有限、局部极点半径恶化和状态超包线提供原子回退到标称
   增益；
5. 完成硬件在环和逐步放宽包线的实机 shadow 试验。
