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

| 架构 | 是否双向 | 当前参数量 | 旧数据 8 段响应 NRMSE |
| --- | --- | ---: | ---: |
| 展平 MLP | 不适用 | 1,069,997 | 0.3460 |
| 全序列 TCN | 是 | 250,941 | 0.3526 |
| 双向 GRU | 是 | 292,141 | 0.3495 |

TCN 和双向 GRU 都直接读取 500 Hz 完整日志。旧数据上它们没有超过 MLP，因此暂不以“网络更复杂”作为选择理由。逐轴结果显示时序模型改善了 pitch，但 yaw 仍受原数据激励不足限制。新分层数据将用于重新比较，不沿用旧数据结论选型。

新分层数据的第一版 100 Hz、2 s MLP 基线有 1,837,997 个参数。八次飞行时，完整响应 NRMSE 为 `0.3082`，比旧数据 MLP 的 `0.3460` 明显降低；平均标签 `R2` 从 `0.466` 提高到 `0.627`。其中 yaw 平均 `R2` 从 `0.306` 提高到 `0.452`，说明可实机复现的分层 yaw-rate 初始条件确实补充了辨识信息。

同一机体的多次飞行先独立编码，再通过集合统计聚合，因此飞行顺序不影响结果。日志多于检查点训练的最大次数时，离线工具按三轴角速度 RMS 和控制指令运动量选择信息量最高的记录；这只是选择正常闭环日志，不会制造额外激励。

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

推荐的离线组合使用 TCN 的 roll/pitch 响应和 MLP 的 yaw 响应；轴选择只使用验证集确定。输入使用一份原始 500 Hz NPZ，工具把原始序列交给 TCN，并严格按训练规则做整数 5 倍抽取后交给 100 Hz MLP。

输出包含选中的日志编号、信息量分数、输入分布 z-score、预测响应、固定滤波器系数、标称 19 状态增益、完整预测增益和 90% 分析增益。所有输出固定标记为：

```text
deployment_mode = offline_analysis_only
gain_accepted_for_flight = false
```

离线工具计算的极点半径只针对辨识模型，不是真实未知机体的稳定性证明。候选增益必须在独立随机初始条件仿真、模型不确定性包线和 HIL 中验证后，才能形成下一次飞行使用的发布产物。

## 下一步评估顺序

1. 在新分层闭环数据上重新训练 MLP、TCN 和双向 GRU。
2. 比较 1/2/4/8 次飞行的响应误差，重点检查 yaw 输出。
3. 对候选模型扫描 `0.0–1.0` 增益混合比例，而不是只比较 25/50/75%。
4. 使用完全相同的陌生机体和初始条件运行严格配对非线性评估。
5. 按物理参数、日志信息量和输入 OOD 分数分析候选增益的胜例与败例。
6. 只有相对固定复合标称 LQI 的安全非劣和收敛优势都能跨数据种子复现时，才生成 HIL 候选。
