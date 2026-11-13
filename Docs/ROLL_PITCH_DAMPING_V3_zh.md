# roll/pitch 软件抑振 v3：台架候选

针对 `log/26.11.5/log_0_UnknownDate_1.ulg` 中约5–6 Hz的真实姿态/舵机抖动，已完成低带宽参数覆盖、50 Hz PWM保持/延迟检查、日志固定输入对照和候选编译。

**当前不是可直接实飞的版本。**较大的突加静态 roll 偏置测试仍触发姿态边界，因此没有替换当前选中的 v2 payload，也没有刷机。输出的文件明确命名为 `BENCH_ONLY.px4`。约300 Hz的原始机械振动也没有被本次软件改动消除。

## 改了什么

配置：`px4_trans/configs/lqi_coaxial_collective_airframe1_yaw_balanced_v3.yaml`。

| roll/pitch 项目，模型力矩域 | v2 | v3 |
|---|---:|---:|
| 姿态误差增益 Kp，N·m/rad | 0.0814317 | **0.012** |
| 角速度反馈 Kd，N·m/(rad/s) | 0.0168644 | **0.004** |
| 姿态误差积分 Ki | 0.0997527 | **0.020** |
| 积分力矩硬上限 | ±0.0045 N·m | **±0.0081 N·m** |
| 总 roll/pitch 力矩请求上限 | ±0.009 N·m | **保持不变** |

同时减小 P、D、I 的动态增益，目标是降低实际执行器跟不上的频段中的环路增益。积分累积速度降为原来的约20%；由于日志中已有积分容量不足的表现，允许更大的**有界**静态补偿容量，保留按轴、按方向抗饱和与回退。没有无限积分，也没有提高总力矩或电机输出上限。

yaw 增益、yaw目标±0.30 rad/s及0.20秒平滑、共模优先值−0.08及−0.16～0预留范围、转速比1.15约束、两路物理电机输出0.45上限、零油门和30°停机保护均保留。没有修改机械 trim、PWM端点、PWM频率或 IMU 参数；没有给5–6 Hz直接加陷波器掩盖运动。

部署层通过新增 `attitude_response` 显式覆盖8状态矩阵中的 roll/pitch 两行，yaw 行逐字保留。这次是按执行器动态检查后的 PID/力矩域重整定，不能再称为原 LQI 重新合成。新模型 checksum 为 **0xe55d7665**；矩阵、积分容量和其它模型参数均参与 checksum。

## 为什么不是只把增益略降一点

线性筛选模型包含：

- 三阶姿态/角速度/舵机力矩动态及姿态积分；
- 标称40 Hz二阶 gyro低通近似；
- **50 Hz的PWM零阶保持**；
- 额外0、2、4、10、20 ms输送延迟；
- 假设有效控制增益为标称的0.5、1、1.5、2、3倍，舵机时间常数0.04、0.06、0.10、0.12秒。

这些是假设的动态不确定性，不是已完成的实机辨识。每一延迟组有20组增益/时间常数组合。

原 v2 在部分假设组合中不稳定；v3 在0–4 ms额外延迟的60组组合中均得到极点模小于1，4 ms组最坏约0.99115。10 ms组虽然均小于1，最坏约0.99954，已非常接近临界；20 ms额外延迟组仍有3组不稳定。因此不能宣称覆盖任意延迟/效能，也不能根据线性筛选保证实飞稳定。

在5.6 Hz处，控制器本身的频率响应幅值由约0.596降至约0.141，相差 **−12.54 dB**。这不是完整飞机闭环增益裕度，也不是实测振幅下降。

## 验证结果与未通过项

### 已通过

- 候选5组 golden vectors、转速/输出边界扫描、按轴积分限幅/回退、舵机缩放饱和回退、共模退让与参考航向检查。
- 默认 v2 的全部主机核心/保护检查及两项 payload 来源测试仍通过。回归测试改为根据实际增益构造“确实饱和”的输入，避免降低增益后原测试不再碰到限幅。
- v3 payload checksum读回一致；yaw行、转速表、共模边界和总力矩边界与v2逐项比较。
- 同一份实测输入下，19–32 s Servo1 的4–8 Hz输出RMS从约 **0.1662降到0.0422**，约为原来的 **25.4%**。这是固定输入控制器对照，不是改版后的实际飞行轨迹或实测机体振幅。
- 使用实际C++核心、三轴非线性刚体/格栅耦合、电机/舵机动态、50 Hz保持与4 ms额外输送延迟的模型检查中，名义、0.004 N·m外扰/较慢舵机、强格栅、下桨系数变化等5组声明工况通过边界及末段恢复检查。
- 固件目标 `micoair_h743-v2_nncontrol` 完整编译通过，候选已归档，原选中payload已恢复。

### 明确保留的失败结果

在模型中从初始时刻突加 **+0.006 N·m roll外扰**，v3未能在30°姿态边界内恢复，试验继续记录到约0.626秒、roll超过45°后终止。实际飞控会在30°边界先停机；测试故意未模拟这一停机来查看控制器趋势。

这说明只压低反馈带宽，不能同时保证承受较大的未配平偏置。不能把该失败项删掉后称“模型全部通过”。用户要求先修共振，因此当前交付的是**供台架检查抑振方向的候选**；持续roll偏置/实际效能与时延还需明确后，才能判断可用的自由飞行参数。

本模型没有真实桨流、机械间隙/卡滞、供电、估计器完整动态和原始IMU振动。1.15仍是模型转速边界，不是ESC实测转速闭环。

## 文件与复现

参数生成，不切换默认payload：

```sh
python3 px4_trans/tools/generate_coaxial_collective_backend.py \
  --deployment-config px4_trans/configs/lqi_coaxial_collective_airframe1_yaw_balanced_v3.yaml \
  --output-dir px4_trans/generated/airframe-1/yaw-balanced-v3 \
  --reuse-model-manifest px4_trans/generated/airframe-1/yaw-balanced-v2/lqi_coaxial_collective_manifest.json \
  --no-activate
python3 analysis/roll_pitch_damping_v3_20261105/check_candidate.py
python3 analysis/roll_pitch_damping_v3_20261105/pwm_delay_screen.py
python3 analysis/roll_pitch_damping_v3_20261105/replay_check.py
python3 analysis/roll_pitch_damping_v3_20261105/coupled_check.py
```

`coupled_metrics.json` 中 `stress_case=true` 的未通过项会被保留；脚本只对其它明确声明的台架模型工况做退出码断言，不能把脚本退出0解释成所有压力工况通过。

`build_candidate.py` 会临时选用v3进行编译、归档 `BENCH_ONLY.px4`，并在 `finally` 中恢复原选中payload。默认v2文件和旧固件归档保留，未自动刷写任何设备。

最终候选：`px4_trans/generated/firmware/roll_pitch_damping_v3_20261105/`。验证数据均在 `analysis/roll_pitch_damping_v3_20261105/`；早期试调文件不是发布参数，最终参数以v3 YAML及payload为准。
