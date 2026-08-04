# 舵效与舵机动态的控制等效辨识实验

## 结论

可以把“舵效不足”和“舵机变慢”合并辨识，但不应把它们压缩成一个静态标量。
两者在闭环中的共同作用是从舵指令到机体角加速度的动态映射。本实验直接监督该映射的短时阶跃响应，再把预测响应投影到稳定的固定滤波器基底，用于重构 LQI 模型。

在 393 个未见机体参数组、每组 8 个相同初始条件的严格配对评估中：

- 标称响应模型的归一化 RMSE 为 `0.5551`；
- 原先分别预测物理效能和时间常数再合成响应时为 `0.3827`；
- 直接预测控制等效阶跃响应后为 `0.3460`；
- 固定滤波器基底对真值响应的投影误差仅为 `0.0053`，因此当前主要误差来自辨识网络，而不是等效模型容量。

固定复合模型的收敛率为 `67.33%`。直接使用全部预测增益为 `67.14%`，没有提高；将标称增益向预测增益缓慢移动 `50%` 时为 `68.67%`，相对固定复合标称提高 `1.34` 个百分点，按机体参数组聚类自助法得到的 95% 置信区间为 `[0.22, 2.45]` 个百分点。平均控制饱和占比同时降低 `0.39` 个百分点，95% 置信区间为 `[-0.75, -0.08]` 个百分点。

这个结果支持继续研究“受限、缓慢的在线增益更新”，但还不支持实机部署。产物仍标记为 `research_only`，自动增益更新保持关闭。

## 为什么不能只辨识一个“综合效能”标量

对单个一阶舵机，忽略死区和回差时，可写为：

```text
delta_dot = (k_cmd * u - delta) / tau
alpha     = G * delta
```

其中 `u` 是舵指令，`delta` 是不可测的机械舵角，`tau` 是舵机时间常数，`G` 是舵角到角加速度的效能矩阵。单位阶跃指令产生：

```text
alpha(t) = G * k_cmd * (1 - exp(-t / tau))
```

短历史内，较小的 `G` 和较大的 `tau` 都会降低角加速度，因此分别回归二者容易产生强相关和非唯一标签。但如果只回归一个静态乘数，又会丢失相位滞后：两个执行器在稳态效能相同、时间常数不同时，需要的 LQR 增益并不相同。

因此这里合并的是动态对象 `H(t): u -> alpha(t)`，不是把所有物理量相乘成一个数。它保留了控制真正需要的瞬态幅值和快慢，同时不要求区分“究竟是舵机慢了还是舵效不足”。

## 控制模态与监督标签

三个舵面先按标称的指令到角加速度矩阵做奇异值分解，得到一个固定、正交的三模态变换。模态只定义舵指令空间的基，不是额外随机化的“耦合强度”参数。所有训练样本使用同一个标称变换，网络预测每个模态对滚转、俯仰和偏航角加速度的响应。

监督标签取单位模态阶跃后 `10/20/40/80/160 ms` 的响应快照：

```text
y[n, time, angular_axis, command_mode]
```

共 `5 x 3 x 3 = 45` 个输出。选择这些时刻是为了覆盖当前随机化中快舵机的上升段、典型舵机的主要瞬态和慢舵机的接近稳态段，同时避免直接回归高度相关的滤波器系数。标签仍完全由仿真真值生成，但输入只有 1 秒飞行历史，不包含舵角反馈或物理参数标签。

网络沿用重复试飞集合 MLP：每段 1 秒历史先独立编码，再对可用试飞做集合聚合，最后输出 45 个响应值。模型参数量为 `1,069,997`。训练时随机遮蔽部分试飞，使同一个检查点可以接收 1、2、4 或 8 段历史。

测试集结果如下：

| 历史段数 | 响应快照平均 R2 | 完整响应 NRMSE |
| ---: | ---: | ---: |
| 1 | 0.1875 | 0.4337 |
| 2 | 0.3258 | 0.3867 |
| 4 | 0.4182 | 0.3596 |
| 8 | 0.4661 | 0.3460 |

历史增加带来单调改善，说明正常闭环飞行中的多次不同初始姿态确实提供了额外辨识信息。不过 8 段历史后的平均 `R2` 仍只有约 `0.47`，不能把网络输出当成精确真值。

## 固定滤波器复合 LQI

网络响应不能直接进入代数 LQR 模型。推理时先把五个响应快照最小二乘投影到三个稳定一阶基底：

```text
phi_j(t) = 1 - exp(-t / tau_j)
tau_j in {15 ms, 40 ms, 80 ms}
H_hat(t) = sum_j C_j * phi_j(t)
```

`C_j` 是每个角加速度轴到每个舵指令模态的等效系数。固定正时间常数保证执行器子系统自身稳定，也把任意 45 维网络输出约束为可用于状态空间综合的低阶动态模型。

复合 LQI 使用 19 个状态：

```text
[roll, pitch, p, q, r,
 motor_state_1, motor_state_2,
 3 modes x 3 fixed filters,
 roll_integral, pitch_integral, yaw_rate_integral]
```

控制器没有使用机械舵角。九个滤波器状态完全由历史舵指令、已知指令死区和回差预处理、固定滤波器时间常数递推得到。网络只改变这些滤波状态到角加速度的系数，进而重新求解 LQI 增益。

## 严格配对闭环结果

评估使用测试集全部 `393` 个未见参数组，每组采样 `8` 个最大初始倾角 `15 deg`、最大初始角速度 `1 rad/s` 的初始条件，仿真 `2 s`。所有控制器变体共享相同参数、初始状态和随机数序列；置信区间按参数组聚类，避免把同一机体的 8 次试飞当成独立样本。

| 控制器 | 收敛率 | 相对复合标称 | 平均饱和占比 |
| --- | ---: | ---: | ---: |
| 原始标称 LQI | 64.03% | -3.31 pp | 8.23% |
| 固定复合标称 LQI | 67.33% | 基准 | 8.59% |
| 预测增益 100% | 67.14% | -0.19 pp | 8.37% |
| 预测增益 25% | 68.38% | +1.05 pp | 8.26% |
| 预测增益 50% | 68.67% | +1.34 pp | 8.20% |
| 预测增益 75% | 68.29% | +0.95 pp | 8.23% |
| 复合模型真值增益 | 75.16% | +7.82 pp | 8.12% |

`25%` 和 `50%` 更新相对复合标称的收敛提升置信区间不跨零；`75%` 和 `100%` 更新没有可靠提升。这与实际部署需求一致：辨识输出应通过增益变化限幅和时间插值缓慢注入，而不能一次替换全部增益。

需要注意，复合标称相对原始标称的 `+3.31` 个百分点来自控制器动态表示的变化，不是神经网络自适应收益。网络自适应的净收益必须以“固定复合标称 LQI”为基线，当前最好的净收益是 `50%` 更新的 `+1.34` 个百分点。

## 产物与复现

训练检查点与训练报告：

```text
Identification/runs/sim2real_composite_step_response_all_modes_mlp_v3/identifier.pt
Identification/runs/sim2real_composite_step_response_all_modes_mlp_v3/report.json
```

闭环配对报告：

```text
Identification/runs/sim2real_composite_step_response_all_modes_mlp_v3/control_audit_paired_large_v1.json
```

训练命令：

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.sim2real_composite_training \
  --dataset Identification/datasets/lqr_sim2real_micro_repeated8_v1 \
  --output Identification/runs/sim2real_composite_step_response_all_modes_mlp_v3 \
  --experiment-config Identification/configs/lqr_sim2real_micro_repeated8_v1.yaml \
  --adaptive-mode-indices 0,1,2 \
  --target-representation step_response \
  --device cuda:0
```

闭环评估命令：

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  /home/ryan/miniconda3/envs/rl-flight/bin/python \
  -m flight_identification.sim2real_composite_evaluation \
  --checkpoint Identification/runs/sim2real_composite_step_response_all_modes_mlp_v3/identifier.pt \
  --legacy-checkpoint Identification/runs/sim2real_micro_repeated8_mlp_v3/identifier.pt \
  --dataset Identification/datasets/lqr_sim2real_micro_repeated8_v1 \
  --experiment-config Identification/configs/lqr_sim2real_micro_repeated8_v1.yaml \
  --output Identification/runs/sim2real_composite_step_response_all_modes_mlp_v3/control_audit_paired_large_v1.json \
  --device cuda:0 --maximum-parameter-groups 393 \
  --trial-count 8 --evaluation-initial-conditions 8 \
  --maximum-initial-tilt-rad 0.2617994 \
  --maximum-initial-rate-rad-s 1.0 --duration-s 2.0
```

## 部署前仍需完成

本次只验证了离线辨识后安装固定增益，尚未验证飞行中每秒推理、连续低通更新增益的时变闭环。进入实机前至少还需要：

1. 把 `50%` 目标变化改成按时间限速的增益插值，而不是一次跳变，并验证切换瞬态。
2. 加入辨识不确定度、输入持续激励度和饱和率门控；信息不足时保持上一次已接受增益。
3. 在传感器噪声、延迟、模型外扰动和参数缓慢漂移下重复严格配对评估。
4. 对 DARE 失败、增益非有限、局部极点半径恶化和状态超包线提供原子回退到标称增益。
5. 完成硬件在环和逐步放宽包线的实机 shadow 试验。
