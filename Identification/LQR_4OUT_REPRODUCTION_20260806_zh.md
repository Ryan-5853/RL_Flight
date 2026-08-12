# LQR 四输出离线复合辨识复现报告

日期：2026-08-06

## 1. 最新实验判定

根据 `README.md`、`OFFLINE_LOG_IDENTIFICATION_zh.md` 和
`OFFLINE_4OUT_COMPOSITE_REPORT_zh.md`，LQR 辨识网络最新完成的正式实验是
2026-08-05 的四输出部署结构实验，而不是 README 后部的 repeated8 历史基线：

- 数据：`lqr_sim2real_micro_offline_logs_v4`，8192 个机体、每机 8 次闭环试飞；
- 上桨由虚拟飞手定高，LQR 输出为下桨与三路舵机，增益形状 `[4, 19]`；
- 三模型 hybrid：roll=TCN、pitch=BiGRU、yaw=MLP；
- 推荐使用 80% 预测增益混合；
- 状态仍为 `research_only`，不构成飞行授权。

## 2. 复现环境与 8 GB 适配

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB；
- PyTorch：2.12.1+cu126；
- Python：3.10.12；
- 数据生成保持原配置 `parallel_count=2048`，实测约 2.4 GiB；
- 训练保持 effective batch=128，通过 `--micro-batch-size` 分块反传；
- MLP/TCN/BiGRU 分别使用 micro-batch 128/32/4，避免 OOM；
- micro-batch 只分割同一 effective batch，累积后才执行一次 optimizer step。

## 3. 数据集复现

原配置和 seed `20260828` 从零生成：

| 指标 | 原报告 | 本次复现 |
| --- | ---: | ---: |
| 机体参数组 | 8192 | 8192 |
| 闭环试飞 | 65536 | 65536 |
| 成功试飞 | 38840 | 38985 |
| 成功率 | 59.27% | 59.49% |
| 进入训练/验证/测试的组 | 4412/928/949 | 4437/935/954 |

成功试飞相差 145 条（0.22 个百分点）。仿真传感器噪声没有绑定独立的显式
generator，因此即使物理采样 seed 相同也不是逐 bit 确定；参数范围、数据结构与统计
分布一致。

数据目录：`Identification/datasets/lqr_sim2real_micro_offline_logs_v4`（约 3.5 GiB）。

## 4. 模型复现结果

三个模型均使用全部 `--adaptive-mode-indices 0,1,2`。这项参数没有写在原报告的
命令中，但可由 MLP 报告参数量 1,837,997 精确反推。

| 模型 | 参数量（原/复现） | best epoch | test 8-flight mean R2（原/复现） | test NRMSE（原/复现） |
| --- | ---: | ---: | ---: | ---: |
| MLP v7, 100 Hz | 1,837,997 / 1,837,997 | 152 | 0.707 / 0.7058 | 0.2419 / 0.2438 |
| TCN v7, 500 Hz | 306,861 / 334,605 | 131 | 0.666 / 0.6981 | 0.2848 / 0.2724 |
| BiGRU v7b, 500 Hz | 292,141 / 292,141 | 231 | 0.698 / 0.6907 | 0.2485 / 0.2529 |

MLP 与 BiGRU 的模型规模和指标高度吻合。原 TCN 的通道参数没有保存在文档、Git
配置或 checkpoint 中；报告中的 306,861 不能由当前默认 `64,96,128` 通道结构
还原。本次使用当前代码默认 TCN，参数量较大，但测试指标达到或略优于原报告。

本次验证逐轴 R2：

| 模型 | roll | pitch | yaw |
| --- | ---: | ---: | ---: |
| MLP | 0.7227 | 0.7827 | 0.5627 |
| TCN | 0.7289 | 0.7962 | 0.4974 |
| BiGRU | 0.7267 | 0.7938 | 0.5039 |

roll/yaw 保持原选择；pitch 上 TCN 仅比 BiGRU 高 0.0024，而原报告是 BiGRU 更高。
为复现最新正式方案，闭环审计仍使用 TCN-roll + BiGRU-pitch + MLP-yaw。

## 5. 949 组严格配对闭环审计

每组 8 个新初值、2 s；下表均为 80% 增益混合相对固定复合标称 LQI，置信区间按
机体参数组聚类 bootstrap：

| seed | 原报告标称/80% | 本次标称/80% | 本次增益与 95% CI | 安全率 |
| --- | ---: | ---: | ---: | ---: |
| 20260826 | 68.45% / 79.06% | 67.86% / 76.71% | +8.85 pp [7.27, 10.50] | 100% / 100% |
| 20260829 | 68.91% / 78.66% | 68.62% / 77.03% | +8.40 pp [6.84, 9.96] | 100% / 100% |

核心结论复现：两个独立 seed 均有显著收敛提升，组聚类 95% CI 不含零，安全非劣，
全部局部真值模型稳定。预测响应 NRMSE 为 0.2423。

本次 seed29 的 75% blend 为 77.32%，略高于 80% 的 77.03%；差异只有 0.29 个
百分点，不改变 75–80% 保守混合区间的结论。

## 6. 8 秒长时审计

393 组、每组 4 个新初值：

| 控制器 | 安全率 | 收敛率 |
| --- | ---: | ---: |
| 固定复合标称 LQI | 100% | 99.7455% |
| 全预测增益 | 100% | 100% |
| 75% 混合 | 100% | 100% |
| 80% 混合 | 100% | 100% |

原报告标称为 99.43%，预测与 75/80% 混合为 100%；长时间不退化结论复现。

## 7. 质量校准与产物

验证集质量校准保留 740 个至少有 4 条成功日志的机体，生成三轴最小角速度 RMS、
命令运动量和特征 z-score 阈值。通过阈值只允许进入独立仿真/HIL，不代表可以飞行。

- 组合清单：`runs/repro_20260806/sim2real_offline_hybrid_v7/manifest.json`
- 两个主审计：`control_audit_hybrid_3way_final949_seed20260826/29_v1.json`
- 长时审计：`control_audit_hybrid_3way_8s_393_ic4_v1.json`
- 质量校准：`offline_log_quality_calibration_v1.json`

最终边界维持原结论：`deployment_mode=research_only`、
`gain_updates_enabled=false`、`gain_accepted_for_flight=false`。

## 8. 核心复现命令

数据生成：

```bash
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  .venv/bin/python -m flight_identification.repeated_trial_experiment \
  --config Identification/configs/lqr_sim2real_micro_offline_logs_v4.yaml
```

模型训练的公共参数为 dataset、experiment-config、device cuda:0、effective
batch-size 128、`--adaptive-mode-indices 0,1,2`。差异参数：

```text
MLP:   --architecture mlp   --downsample 5 --micro-batch-size 128
TCN:   --architecture tcn   --downsample 1 --micro-batch-size 32
BiGRU: --architecture bigru --downsample 1 --recurrent-hidden-size 64 \
       --recurrent-layers 2 --micro-batch-size 4
```

闭环评测使用 MLP 为主 checkpoint，TCN 替换 roll，BiGRU 替换 pitch，
`--maximum-parameter-groups 949 --trial-count 8 --evaluation-initial-conditions 8
--duration-s 2 --gain-blend-fractions 0.5,0.75,0.8,0.9`，分别运行 seed
20260826 与 20260829。
