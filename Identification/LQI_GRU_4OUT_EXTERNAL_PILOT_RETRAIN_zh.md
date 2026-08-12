# 四输出 External-Upper DAgger-GRU 重训与评测说明

更新时间：2026-08-07

## 1. 当前状态

四输出代码、正式数据集、6 轮 DAgger、checkpoint 固定口径筛选和首轮
独立 test 闭环评测均已完成。当前模型**尚未达到部署门槛**：DAgger 已基本
解决失稳，但完整 GRU 的角速度跟踪和严格收敛性能仍明显落后 nominal LQI。

当前保留的主候选（仅供后续研究，不建议上机）为：

```text
Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/students/final/student.pt
```

额外 10 epoch 精炼模型保存在 `students/refine_10_from_final/student.pt`，但闭环
安全性下降，因此没有替换主候选。

本实验替代旧的五输出 DAgger-GRU 主线。旧数据集和 checkpoint 只能用于历史对照，不能作为最终部署模型。

唯一部署契约为：

```text
飞手 upper PWM -------------------------------+
                                                +--> 五通道执行命令
传感器、目标、upper PWM、历史命令 --> GRU --> lower + servo1/2/3
```

- 飞手始终拥有 `motor_upper`。
- nominal LQI、oracle LQI 和 GRU 均只能输出四维：`motor_lower + servo_1/2/3`。
- 环境执行前显式拼接 `[pilot_upper, controlled_4out]`。
- DAgger 中 oracle 的命令驱动观测器由实际执行的学生动作推进，不使用未执行的 oracle 反事实动作。

主配置文件：

- `configs/lqi_gru_dagger_4out_external_pilot_v1.yaml`

## 2. 外部飞手能力分布

配置不再假设外部飞手等价于理想高度 PID。每个 episode 从以下 profile 抽样：

| Profile | 训练权重 | 特性 |
|---|---:|---|
| skilled | 0.20 | 小延迟、高更新率、小偏差 |
| average | 0.35 | 中等延迟和有限修正能力 |
| poor | 0.30 | 低增益、慢更新、明显偏差、可能漏操作 |
| manual | 0.15 | 不闭环保持高度，分段改变 upper PWM |

所有特性均在 YAML 中开放，包括：

- `feedback_gain_scale`
- `reaction_delay_s`
- `update_period_s`
- `height_bias_m`
- `height_noise_std_m`
- `lapse_probability`
- `manual_throttle_range`
- `manual_hold_s`

训练使用混合 profile；正式评测既要测混合分布，也要使用 `--pilot-profiles` 分别锁定每一档能力。

## 3. 8 GiB GPU 约束

主配置采用以下保守起点：

- 参数组 `4096`；这是按当前 15 GiB 主机 RAM 和现有全量内存加载器设定的上限
- 仿真 `parallel_count: 1024`
- GRU `hidden_size: 128`、两层
- 训练 `batch_size: 8`
- `tbptt_steps: 512`
- 数据 shard 每次 256 个参数组（共 1024 个并行 episode）

训练器支持环境变量继续降低 batch：

```bash
export FLIGHT_IDENTIFICATION_LQI_GRU_BATCH_SIZE=4
```

启动前运行 `nvidia-smi`。建议实际占用保持在约 7 GiB 以下。如果生成数据时 OOM，依次将 `parallel_count` 调为 128、64；如果训练 OOM，先将 batch 调为 4，再将 `tbptt_steps` 调为 256。

## 4. 环境准备

以下命令均从仓库根目录执行：

```bash
cd /home/ryan_wsl/RL_Flight
export PYTHONPATH=Identification/src:Train/src:Controller/src:SimEnv/src
```

使用现有虚拟环境：

```bash
.venv/bin/python --version
```

不要覆盖已有结果目录。数据生成器和训练器会拒绝写入非空目录，以防误删历史实验。

## 5. 第一步：生成四输出 oracle 行为克隆数据

完整命令：

```bash
.venv/bin/python -m flight_identification.lqi_gru_experiment \
  --config Identification/configs/lqi_gru_dagger_4out_external_pilot_v1.yaml \
  --device cuda:0
```

预期数据目录：

```text
Identification/datasets/lqi_gru_dagger_4out_external_pilot_v1/
```

生成结束后必须检查 `manifest.json`：

```text
upper_rotor_owner = external_pilot
action_names = [motor_lower, servo_1, servo_2, servo_3]
oracle_lqi_gain shape = [..., 4, 13]
external_pilot_profiles = [skilled, average, poor, manual]
```

如果 oracle 完整安全轨迹比例过低，先检查各 profile 和 T/W 分层，不能直接开始训练。

## 6. 第二步：运行完整 DAgger

DAgger 的 `iterate` 阶段会自动：

1. 从行为克隆数据准备 merged dataset；
2. 训练当前学生；
3. 学生闭环采集访问状态；
4. 用四输出 oracle 标注；
5. 聚合 shard；
6. 完成最后一轮后额外训练一次 `students/final/student.pt`，确保最终模型见过最后一批 DAgger 数据。

正式运行先用基础行为克隆数据做 1 epoch warm start：

```bash
export FLIGHT_IDENTIFICATION_LQI_GRU_BATCH_SIZE=8
.venv/bin/python -m flight_identification.lqi_gru_distillation \
  --dataset Identification/datasets/lqi_gru_dagger_4out_external_pilot_v1 \
  --config Identification/configs/lqi_gru_dagger_4out_external_pilot_v1.yaml \
  --output-directory Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/students/pretrain_01 \
  --device cuda:0 \
  --epochs 1 \
  --arch gru \
  --previous-command-noise-std 0.05 \
  --previous-command-reset-prob 0.05
```

随后运行 DAgger。`--initial-checkpoint` 明确指定 warm start；
`--update-epochs` 控制每轮聚合后的更新轮数，`--epochs` 控制最后一次完整拟合。
本次为先验证闭环流程，二者均使用 1；长日程实验再逐步提高，而不是直接覆盖
这批结果。

```bash
export FLIGHT_IDENTIFICATION_LQI_GRU_BATCH_SIZE=8
.venv/bin/python -m flight_identification.lqi_gru_dagger \
  --stage iterate \
  --base-dataset Identification/datasets/lqi_gru_dagger_4out_external_pilot_v1 \
  --config Identification/configs/lqi_gru_dagger_4out_external_pilot_v1.yaml \
  --output-root Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1 \
  --device cuda:0 \
  --iterations 6 \
  --train-groups 64 \
  --trials 4 \
  --epochs 1 \
  --update-epochs 1 \
  --initial-checkpoint Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/students/pretrain_01/student.pt \
  --gate-groups 24 \
  --gate-trials 4 \
  --gate-duration-s 6.0 \
  --arch gru \
  --previous-command-noise-std 0.05 \
  --previous-command-reset-prob 0.05
```

可恢复性：

- 已存在的 iteration checkpoint 会被复用；
- 已存在的 DAgger iteration shard 会被跳过；
- 不要使用 `--overwrite`，除非明确要重建 prepared dataset；
- 最终候选为：

```text
Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/students/final/student.pt
```

## 7. 第三步：离线模仿与记忆消融

```bash
.venv/bin/python -c \
  'from flight_identification.lqi_gru_distillation import eval_main; eval_main()' \
  --dataset Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/datasets/merged \
  --checkpoint Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/students/final/student.pt \
  --output Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/evals/offline_final.json \
  --device cuda:0 \
  --batch-size 8
```

离线 RMSE 只能检查容量和记忆使用情况，不能代替闭环结论。

## 8. 第四步：统一闭环评测

### 8.1 严格定点调节

`regulation` 会把 roll/pitch/yaw 目标固定为零，并报告持续 0.5 秒满足 2°/0.1 rad/s 的严格收敛率。

```bash
.venv/bin/python -m flight_identification.lqi_gru_closed_loop \
  --dataset Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/datasets/merged \
  --config Identification/configs/lqi_gru_dagger_4out_external_pilot_v1.yaml \
  --checkpoint Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/students/final/student.pt \
  --output Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/evals/regulation_mixed.json \
  --device cuda:0 \
  --maximum-groups 256 \
  --trials 8 \
  --duration-s 3.0 \
  --task-mode regulation \
  --pilot-height-mode training_truth
```

### 8.2 动态跟踪

```bash
.venv/bin/python -m flight_identification.lqi_gru_closed_loop \
  --dataset Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/datasets/merged \
  --config Identification/configs/lqi_gru_dagger_4out_external_pilot_v1.yaml \
  --checkpoint Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/students/final/student.pt \
  --output Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/evals/tracking_mixed.json \
  --device cuda:0 \
  --maximum-groups 256 \
  --trials 8 \
  --duration-s 6.0 \
  --task-mode tracking \
  --pilot-height-mode training_truth \
  --report-segments-s 1.0 2.0 4.0 6.0
```

### 8.3 按飞手能力分层

对 `skilled average poor manual` 各执行一次，例如 poor：

```bash
.venv/bin/python -m flight_identification.lqi_gru_closed_loop \
  --dataset Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/datasets/merged \
  --config Identification/configs/lqi_gru_dagger_4out_external_pilot_v1.yaml \
  --checkpoint Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/students/final/student.pt \
  --output Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/evals/tracking_poor.json \
  --device cuda:0 \
  --maximum-groups 256 \
  --trials 8 \
  --duration-s 6.0 \
  --task-mode tracking \
  --pilot-height-mode training_truth \
  --pilot-profiles poor
```

`fixed_target` 可用于让反馈型飞手产生跨控制器一致的外生 upper 调度；`training_truth` 则用于评估真实的“飞手—飞行器—姿态控制器”耦合。两套结果必须分开报告。

## 9. 评测判据

报告必须同时展示 nominal_4out、oracle_4out、GRU 和 GRU-reset：

- 安全率与完整时域生存率；
- 严格收敛率；
- 姿态和角速度 RMS、p50、p90；
- 最终姿态/角速度误差；
- 下桨/舵机饱和率；
- 命令变化量；
- nominal 到 oracle 的 gap recovery；
- 各飞手 profile 的独立结果。

建议首轮验收门槛：

- 先确认 oracle_4out 相比 nominal_4out 有可辨识优势；若收敛率优势不足约 10 pp 且动态 RMS 优势不足约 15%，停止蒸馏并先修正 benchmark。
- GRU 的 gap recovery 目标不低于 70%。
- Core 安全率相对 nominal 的下降不超过 0.5-1 pp。
- poor/manual 飞手结果不得被混合平均数掩盖。
- 上桨 ownership 必须始终为外部透传；模型 checkpoint 的 action schema 必须严格为四维。

部署前还有一个必须完成的 estimator gate：当前 SimEnv 没有完整机载姿态估计器，训练数据使用 truth tilt 经接口代理。闭环评测应至少扫描姿态噪声、固定偏置和控制周期延迟；在真正上机前，应接入 PX4 实际输出格式做 SIL/HIL 回放。电机转速输入也只有在实机能稳定提供时才允许保留，否则需要在下一版配置中删除或换成命令驱动观测器。

## 10. 2026-08-07 正式运行记录

### 10.1 数据与资源

- RTX 4060 Laptop，显存 8188 MiB；训练实测总占用约 1.68--1.88 GiB，峰值远低于 8 GiB。
- 正式基础数据集包含 4096 个参数组、16384 条 episode；197 条生成失败。
- train/validation/test 接受 episode 数分别为 11320/2433/2434。
- 有效控制步共 48747457；全数据约 5.3 GiB。
- 每条动作严格为四维 `lower + servo_1/2/3`，oracle gain 为 `4 x 13`。
- 6 轮 DAgger 共新增 1536 条学生闭环、oracle 标注轨迹。
- 外部飞手、GRU 蒸馏和 DAgger 共 14 项相关 unittest 全部通过；25 个正式
  evaluation/gate JSON 均可解析。

### 10.2 训练

- 基础 warm start：1 epoch，validation loss `0.0003121`。
- DAgger：6 轮，每轮 1 epoch；最后在完整 merged dataset 再拟合 1 epoch。
- 额外精炼：从 final 恢复训练 10 epoch，最佳 validation loss 从
  `0.0015304` 降至 `0.0009926`，最优为精炼第 5 轮；但验证 loss 后续明显反弹。
- 精炼模型闭环安全性下降，证明离线模仿 loss 不能作为 checkpoint 的最终选择标准。

逐轮训练与门控产物位于：

```text
Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/
```

### 10.3 固定验证集 checkpoint 选择

使用 validation split、24 个参数组、每组 4 次、6 s、seed `20260850`、
`fixed_target`，对所有 checkpoint 做相同闭环比较。完整 GRU 的关键结果：

| checkpoint | 安全率 | 姿态 RMS (rad) | 角速度 RMS (rad/s) |
|---|---:|---:|---:|
| pretrain_01 | 0.6979 | 0.3409 | 3.1316 |
| iter_00 | 0.5625 | 0.4084 | 3.2148 |
| iter_01 | 0.7604 | 0.1692 | 3.1258 |
| iter_02 | 0.9896 | 0.1658 | 1.3225 |
| iter_03 | 0.9688 | 0.1846 | 1.5534 |
| iter_04 | 0.9479 | 0.1528 | 2.2339 |
| iter_05 | 0.9688 | 0.1635 | 1.1884 |
| final | 0.9896 | 0.1295 | 1.5032 |
| refine_10_from_final | 0.9688 | 0.1474 | 1.1280 |

该结果不是单调的；不能默认最后一轮最好。综合安全率与姿态误差，保留 `final`，
但其角速度误差仍不合格。

### 10.4 独立 test 主结果

动态跟踪使用 test split、64 个参数组、每组 4 次、6 s、seed `20260860`、
`training_truth` 动态飞手闭环：

| 控制器 | 安全率 | 姿态 RMS (rad) | 角速度 RMS (rad/s) |
|---|---:|---:|---:|
| oracle LQI | 0.9922 | 0.1062 | 0.5217 |
| nominal LQI | 0.9844 | 0.1282 | 0.6427 |
| GRU | 0.9883 | 0.1391 | 1.6449 |
| GRU-reset | 0.9922 | 0.1084 | 0.9389 |

在这一正确四输出口径下，nominal 与 oracle 已有明确差距；先前“几乎没有差距”
主要来自旧五输出/固定上桨口径。当前 GRU 对 nominal-to-oracle gap 的姿态、角速度
恢复均为负，不能部署。动态 tracking 中目标会变化，不使用 `converged_fraction`
作为主判据；严格收敛只看下面的 regulation。

3 s regulation 结果同样显示完整 GRU 的角速度阻尼问题：GRU/nominal/oracle 的
角速度 RMS 分别为 `1.6327/0.6779/0.5764 rad/s`，严格收敛率分别为
`0.0039/0.3320/0.5898`。

### 10.5 外部飞手分层

每个 profile 使用 16 个 test 参数组、每组 4 次、6 s、相同 seed。下表给出
完整 GRU 与 nominal 的 `安全率 / 角速度 RMS`：

| profile | GRU | nominal LQI |
|---|---:|---:|
| skilled | 1.0000 / 1.5666 | 0.9844 / 0.6491 |
| average | 0.9531 / 1.8813 | 0.9844 / 0.6002 |
| poor | 0.9531 / 2.1396 | 0.9844 / 0.6225 |
| manual | 0.9688 / 1.6131 | 1.0000 / 0.7375 |

飞手能力越差，完整 GRU 的角速度误差总体越大；开放飞手分布是必要评测维度。
GRU-reset 在所有分层中都显著优于完整 GRU，说明递归隐状态漂移/错误累积是当前
首要问题，而非单纯网络宽度不足。

### 10.6 当前结论与下一轮训练方向

1. 保留四输出、外部 upper ownership 和动态飞手闭环口径，不回退到旧五输出。
2. 暂停单纯增加行为克隆 epoch；它降低离线 loss，却降低闭环安全性。
3. 下一轮优先做 hidden-state regularization：随机截断/重置、状态范数约束、
   burn-in 后再计 loss，并让 DAgger gate 直接参与 early stopping/checkpoint 选择。
4. 增加角速度/动作增量的闭环定向采样和 loss 权重扫描，避免学生通过保守小动作
   获得表面安全率。
5. 同时训练显式 memoryless MLP/短窗口 TCN 基线；当前 GRU-reset 的优势表明，
   先证明“记忆确实有益”再扩大 recurrent 模型更合理。
6. 上机前仍需完成姿态估计噪声、偏置、延迟扫描和 PX4 SIL/HIL 回放。

正式 JSON 结果位于：

```text
Identification/outputs/lqi_gru_dagger_4out_external_pilot_v1/evals/
```
