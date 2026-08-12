# DAgger-GRU 实验进展与复现审计（2026-08-06）

## 结论

仓库文档记录的最新已完成模型不是基础 `lqi_gru_dagger_pilot_v2`，而是
`lqi_gru_hard_dagger_pilot_v1/students/rescue_r1/student.pt`。历史结论为：普通 test
复评安全率 95.3%、角速度 RMS 0.394 rad/s；在 10 个训练侧可救组上安全率 42.5%，
优于 nominal 的 32.5%，但未见的慢下电机 group 249 仍为 0/4，说明救援行为不泛化。

本次从当前提交可重建的最早输入开始，完整重跑了 256 组 oracle-LQI 数据、6 轮
DAgger（每轮 60 epochs、96 组 x 4 条 rollout）以及 37 组 x 4 条最终 test。核心结论
得到复现：DAgger 把 GRU 闭环安全率提升到接近固定 nominal，并把角速度跟踪误差降到
nominal 以下；清空 hidden 后明显退化。

历史 hard/rescue 轮无法被称为“精确复现”，因为其必需的候选点 JSON、
`safety_crash_screen.json`、历史 parameter manifest/hash、聚合数据和 checkpoint 均未进入仓库。
当前代码按文档 seed 重建出的基础数据也与报告统计不一致，继续合成 rescue 输入将形成新实验，
而不是复现历史实验。

## 当前文档中的最新进展

- 基础 DAgger：test 安全率 94.6%，nominal 96.0%，oracle 93.9%；rate RMS 0.402 rad/s。
- 两轮 tracking-hard DAgger：留出坏机体 rate RMS 从 0.354 降至 0.327；test 最好安全率
  95.3%。
- `rescue_r1`：训练侧可救组安全率 42.5%（nominal 32.5%，oracle 62.5%），但 group 249
  仍未救回。
- 部署前审计：姿态噪声/偏置/延迟扫描通过；30 秒 hidden 范数稳定；CPU/GPU batch-1
  推理均小于 0.4 ms。结论仍是 research candidate，不是飞行授权。

## 本次环境与 8 GB 适配

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB。
- Python：项目 `.venv`，PyTorch 2.12.1+cu126。
- 原 batch 64 在共享 GPU 时实测达到 7849 MiB，仅余 108 MiB，因此未冒险继续。
- iter 0-2 使用 batch 16，iter 3-5 在外部负载下降后使用 batch 32；完整 3000-step
  因果序列、网络结构、loss、epoch、rollout 数和 gate 协议均未缩减。
- 为避免修改正式 YAML，训练器支持环境变量
  `FLIGHT_IDENTIFICATION_LQI_GRU_BATCH_SIZE`，并把实际 batch 写入 checkpoint。

## 基础数据复现

输出：`datasets/repro_20260806/lqi_gru_oracle_distillation_pilot_256_v2/`

| 项目 | 历史报告 | 本次复现 |
|---|---:|---:|
| 参数组 / episode | 256 / 1024 | 256 / 1024 |
| 完整安全轨迹 | 911 | 946 |
| train / validation / test | 639 / 135 / 137 | 657 / 140 / 149 |
| 失败 episode | 113（由总数反推） | 78 |

另一个直接证据是：历史报告称 group 249 的下电机时间常数约 0.115 s，而当前配置与 seed
重建为 0.019712 s。因此历史 `pilot_256_v2` 至少有一个未记录的数据生成差异。

## 六轮 DAgger validation gate

每轮均为 24 个 validation 参数组 x 4 条、6 秒、无 teacher forcing。

| iter | GRU 安全率 | rate RMS | attitude RMS | reset 安全率 | oracle 安全率 |
|---:|---:|---:|---:|---:|---:|
| 0 | 73.96% | 2.1130 | 0.4243 | 85.42% | 87.50% |
| 1 | 86.46% | 0.7749 | 0.1896 | 81.25% | 87.50% |
| 2 | 86.46% | 0.5731 | 0.2030 | 84.38% | 89.58% |
| 3 | **90.63%** | 0.4805 | 0.1792 | 79.17% | 91.67% |
| 4 | 89.58% | **0.4746** | **0.1636** | 80.21% | 87.50% |
| 5 | 85.42% | 0.4814 | 0.1704 | 81.25% | 89.58% |

复现了报告的主要动力学：第一次聚合带来最大安全提升，后续迭代主要继续降低跟踪误差；
离线 validation loss 随 DAgger 分布扩展反而从 `3.22e-4` 上升到 `1.78e-2`，再次证明不能
按 teacher-forced MSE 选闭环模型。

## 最终 test（37 组 x 4 条，6 秒）

| 控制器 | 安全率 | rate RMS (rad/s) | attitude RMS (rad) |
|---|---:|---:|---:|
| oracle LQI | 98.65% | 0.4116 | 0.1078 |
| nominal LQI | 96.62% | 0.5500 | 0.1257 |
| DAgger GRU iter 5 | **92.57%** | **0.4581** | 0.1337 |
| GRU reset | 91.89% | 0.7326 | 0.2509 |

validation 最佳 iter 3 在同一 test 上为 91.89% / 0.4555 rad/s；按安全率优先的规则，本次
选择 iter 5。GRU 相对 nominal 的 rate RMS 改善约 16.7%，但安全率低 4.05 个百分点；相对
reset 的 rate RMS 改善约 37.5%，说明 recurrent hidden 确实有闭环价值。

## 产物与校验

- 最终模型：`runs/repro_20260806/lqi_gru_dagger_pilot_v2/students/iter_05/student.pt`
- 最终 test：`runs/repro_20260806/lqi_gru_dagger_pilot_v2/final_test_closed_loop.json`
- iter 3 test：`runs/repro_20260806/lqi_gru_dagger_pilot_v2/best_gate_iter03_test_closed_loop.json`
- 六个 gate：`runs/repro_20260806/lqi_gru_dagger_pilot_v2/gates/`
- 基础 dataset manifest SHA-256：
  `e8be8c859074fd3cb12be3f65503fd40842ddad25d6f127c59a06f9991135fa5`
- iter 5 checkpoint SHA-256：
  `48df1181fa508cd56b0d5c455f0b736ff78be1fdef636055c998d9224f89a46d`
- final test SHA-256：
  `f866b2207d4f590fd57b8fa4abf24952ed6cbadf9812b2fc595b8d3fbe3c1145`

核心 DAgger/蒸馏单元测试共 9 项，全部通过。基础数据约 328 MiB，DAgger 运行目录约 1.2 GiB。

## 要使 hard/rescue 可精确复现，必须补存

1. 历史基础 dataset 的 manifest、parameter payload 和内容 hash；
2. hard-point `candidates-json` 及其生成命令；
3. 600 组 `safety_crash_screen.json`；
4. hard_r1/hard_r2/rescue_r1 每轮所用聚合 shard manifest 与 parent checkpoint hash；
5. 每轮实际 batch、代码 commit、Torch/CUDA 版本。

在这些输入补齐前，最严谨的结论是：基础六轮 DAgger 的机制与主要性能趋势已复现；历史最新
`rescue_r1` 的文字结论已确认，但其数值实验因输入产物缺失而不可精确重放。
