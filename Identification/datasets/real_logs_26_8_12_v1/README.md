# 真机日志标准数据集：26.8.12（清洗后）

本目录把 `Identification/real_log/26.8.12/` 下的 19 条 ULog 清洗为最新 v4/v7
离线辨识网络可直接消费的标准数据集。数据形态与
`lqr_sim2real_micro_offline_logs_v4` 训练数据一致：2 秒、500 Hz、14 通道
闭环日志，`valid_mask` 全真。

## 文件

| 文件 | 说明 |
| --- | --- |
| `bundle.npz` | 标准 NPZ：`features[16,1000,14]`、`feature_names`、`sample_hz=500`、`valid_mask[16,1000]` 以及每个窗口的来源/时间/档位/信息量元数据 |
| `windows.json` | 16 个选中窗口的逐项指标 |
| `build_report.json` | 全量清洗报告（解析、分段、候选、拒绝原因、MLP 质量门限） |
| `windows_summary.png` | 每个窗口的姿态/角速度/指令曲线 |
| `offline_identification_v7_hybrid.json` | v7 hybrid（TCN roll + BiGRU pitch + MLP yaw）逐窗口辨识与 top-8 聚合增益 |
| `offline_identification_mlp_tcn_rollpitch.json` | 官方 CLI 组合（MLP 主 + TCN roll/pitch 副）输出 |

## 规范与单位

14 通道（与 v7 检查点 `feature_names` 完全一致）：

```text
attitude_q_tilt.w/x/y/z       去 yaw 倾斜四元数（与训练端 _yaw_free_attitude 同式）
angular_velocity_b.x/y/z      机体角速度 [rad/s]
motor_speed.upper/lower       电机转速 [rad/s]（ESC rpm × 2π/60）
command.motor_upper/lower     归一化油门 [0,1]（actuator_motors.control[0/1]）
command.servo_1/2/3           归一化舵指令 [-1,1]（actuator_servos.control[0/1/2]）
```

通道映射来自固件契约（`px4_trans/src/modules/nn_control`）：`actuator_motors`
通道 0/1 为上/下桨，`esc_status` 按 `actuator_function` 101/102 匹配上/下桨。
姿态为 Hamilton `[w,x,y,z]`、body FRD -> NED，与训练数据同约定。

## 清洗与选取规则

1. 按 MD5 去重（log_20=log_14、log_21=log_15），19 -> 17 条唯一日志。
2. 解析后统一重采样到 500 Hz：角速度/姿态线性插值，指令与 ESC 转速零阶保持；
   任一主题缺测超过 10 ms 的样本标记为无效。
3. 解锁/激活判据：上桨或下桨指令 > 0.05，短脉冲去除、短缺口回填。
4. 只保留 2 秒全窗 `valid_mask` 全真、全程激活、起始倾角 5-60°、
   峰值倾角 ≤ 60°、峰值角速度 ≤ 20 rad/s 的闭环窗口。
5. 窗口起点取“手松开”前 0.15 s（角速度或倾角开始变化的位置），并尝试多个
   偏移以避开日志缺口；同一日志内窗口互不重叠。
6. 档位：
   - A（贴近训练分布）：起始倾角 5-25°、起始角速度 ≤ 2.5 rad/s、峰值 ≤ 45°/8 rad/s；
   - B：起始倾角 ≤ 40°、起始角速度 ≤ 5 rad/s、峰值 ≤ 55°/12 rad/s；
   - C：其余通过硬条件的窗口（仍为闭环，但激励更激进）。

## 选中窗口（16 个）

| # | 日志 | 类型 | 档位 | 初始倾角° | 峰值倾角° | 起始角速度 | 峰值角速度 | 饱和占比 | 信息量 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | log_27 | 起飞回正 | A | 16.7 | 17.1 | 0.11 | 1.26 | 0.72 | 1.42 |
| 1 | log_23 | 起飞回正 | A | 23.0 | 23.5 | 0.36 | 2.78 | 0.72 | 1.69 |
| 2 | log_28 | 起飞回正 | B | 39.7 | 46.5 | 0.26 | 3.71 | 0.86 | 2.22 |
| 3 | log_24 | 起飞回正 | B | 27.3 | 27.6 | 0.15 | 2.57 | 0.47 | 2.07 |
| 4 | log_29 | 空中扰动 | A | 16.3 | 22.1 | 2.22 | 2.43 | 0.24 | 2.25 |
| 5 | log_14 | 起飞回正 | A | 17.7 | 17.8 | 0.10 | 1.90 | 0.72 | 1.91 |
| 6 | log_25 | 起飞回正 | A | 23.3 | 26.7 | 0.07 | 2.20 | 0.96 | 1.78 |
| 7 | log_14 | 空中扰动 | C | 6.9 | 21.4 | 0.73 | 13.22 | 0.38 | 2.73 |
| 8 | log_30 | 空中扰动 | A | 21.4 | 26.3 | 1.14 | 2.84 | 0.33 | 2.27 |
| 9 | log_15 | 起飞回正 | A | 16.6 | 16.7 | 0.23 | 1.54 | 0.79 | 1.21 |
| 10 | log_31 | 起飞回正 | B | 36.7 | 39.8 | 0.22 | 3.23 | 0.76 | 2.00 |
| 11 | log_22 | 空中扰动 | A | 12.3 | 22.1 | 1.40 | 3.68 | 0.78 | 2.50 |
| 12 | log_32 | 起飞回正 | C | 27.8 | 31.2 | 0.16 | 16.70 | 0.62 | 2.81 |
| 13 | log_33 | 起飞回正 | B | 37.0 | 40.4 | 0.26 | 3.44 | 0.68 | 1.97 |
| 14 | log_34 | 起飞回正 | A | 7.0 | 7.5 | 0.39 | 0.77 | 0.52 | 1.19 |
| 15 | log_34 | 起飞回正 | A | 5.1 | 12.9 | 0.23 | 1.21 | 0.63 | 0.91 |

信息量分数与离线工具一致：三轴角速度 RMS + 2×指令变化 RMS。

## 被排除的日志

| 日志 | 原因 |
| --- | --- |
| log_18 | 记录存在周期性 0.17-0.19 s 缺测（连续有效段最长仅 1.6 s），无法构成 2 秒干净窗口 |
| log_19 | 时长 0.84 s 且无油门输出 |
| log_26 | 全程无上桨油门（无飞行） |
| log_20/21 | MD5 与 log_14/15 重复 |

另有各日志的失控/接机后段（如 log_22 的 3.9 s 后、log_14 的 7.8 s 后）被
峰值倾角/角速度或无效样本规则排除。

## 质量检查结果

- 官方质量校准（MLP v7 验证集，100 Hz 视角）：16 个窗口单独评估有 5 个通过
  （log_24/29/30/31/33）；其余主要因单轴角速度覆盖偏低（真实回正比仿真
  分层激励温和）或 z-score 超限。
- top-8 聚合的角速度覆盖全部达标（x/y/z RMS 0.81/0.69/0.46 rad/s，远高于
  校准下限 0.28/0.27/0.20），指令运动量 0.91（下限 0.052）。
- 唯一未过项是输入 z-score：p95=3.10（限 3.08），max=30.0，主要由
  log_32 起飞与 log_14 空中扰动的高角速度尖峰、以及真实电机转速相对仿真
  标称偏高（0.5 油门约 1570 rad/s vs 训练均值 1072 rad/s）造成。
- v7 hybrid top-8 聚合给出的辨识模型极点半径 0.9956，与标称模型同量级，
  增益有限。产物一律为 `offline_analysis_only`，未放行飞行。

## 复现命令

```bash
# 清洗
PYTHONNOUSERSITE=1 PYTHONPATH=Identification/src \
  .venv/bin/python Identification/scripts/build_real_log_dataset.py \
  --logs Identification/real_log/26.8.12 \
  --output Identification/datasets/real_logs_26_8_12_v1 \
  --mlp-checkpoint Identification/runs/repro_20260806/sim2real_offline_logs_step_response_mlp_v7/identifier.pt \
  --quality-calibration Identification/runs/repro_20260806/sim2real_offline_hybrid_v7/offline_log_quality_calibration_v1.json

# v7 hybrid 逐窗口辨识
PYTHONNOUSERSITE=1 PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  .venv/bin/python Identification/scripts/evaluate_real_log_hybrid.py \
  --bundle Identification/datasets/real_logs_26_8_12_v1/bundle.npz \
  --output Identification/datasets/real_logs_26_8_12_v1/offline_identification_v7_hybrid.json \
  --experiment-config Identification/configs/lqr_sim2real_micro_offline_logs_v4.yaml \
  --tcn-checkpoint Identification/runs/repro_20260806/sim2real_offline_logs_step_response_tcn_500hz_v7/identifier.pt \
  --bigru-checkpoint Identification/runs/repro_20260806/sim2real_offline_logs_step_response_bigru_500hz_v7b/identifier.pt \
  --mlp-checkpoint Identification/runs/repro_20260806/sim2real_offline_logs_step_response_mlp_v7/identifier.pt
```

### 2026-08-28 强积分重合成（增大积分项）

同一 bundle 与同一组编码器，仅把 LQR 积分权重提高 4 倍后重新合成增益
（`--experiment-config` 换成 `lqr_sim2real_micro_offline_logs_v4_strong_integral.yaml`，
其 `controller_config` 指向 `lqi_sim2real_micro_coaxial_4out_strong_integral.yaml`）：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  .venv/bin/python Identification/scripts/evaluate_real_log_hybrid.py \
  --bundle Identification/datasets/real_logs_26_8_12_v1/bundle.npz \
  --output Identification/datasets/real_logs_26_8_12_v1/offline_identification_v7_hybrid.json \
  --experiment-config Identification/configs/lqr_sim2real_micro_offline_logs_v4_strong_integral.yaml \
  --tcn-checkpoint Identification/runs/repro_20260806/sim2real_offline_logs_step_response_tcn_500hz_v7/identifier.pt \
  --bigru-checkpoint Identification/runs/repro_20260806/sim2real_offline_logs_step_response_bigru_500hz_v7b/identifier.pt \
  --mlp-checkpoint Identification/runs/repro_20260806/sim2real_offline_logs_step_response_mlp_v7/identifier.pt
```

系数与质量指标不变，`analysis_gain` 的三路积分列约翻倍，辨识模型极点半径
0.9956→0.9912。本目录被 gitignore，本 README 与 JSON 均为本地产物；入库的
固件载荷为 `px4_trans/px4/src/modules/nn_control/LqiIdentifiedModel.hpp`。
