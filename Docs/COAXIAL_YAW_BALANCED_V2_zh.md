# 2026-11-05 失控后的 v2 修复候选

本次修改针对 `log/26.11.5/log_0_UnknownDate.ulg` 中的 roll 力矩长期饱和、积分累积，以及后段固定负格栅共模妨碍正 yaw 纠偏。源码、配置和候选固件已生成，未刷机。

## 修改后的行为

### roll/pitch 按轴抗饱和

- 两轴积分各自限制为本轴最大模型力矩的 **50%**。当前 `maximum_moment=0.009 N·m`，所以积分力矩限幅各为 **±0.0045 N·m**；原有 P/D 增益和总力矩上限没有加大。
- 用“未裁剪力矩 − 经力矩裁剪及舵机向量缩放后可实现的模型力矩”作为各轴残差。只有继续推向更深饱和的积分更新被拒绝，反向回退仍允许；不再只检查舵机是否到 ±1。
- 低于约半悬停目标推力时不增加 roll/pitch 积分；零油门仍清空姿态积分。该条件基于模型目标推力，**不是接地检测器**。
- 旧积分状态过大时也先裁入边界，不能因拒绝新积分而保留越界旧状态。

### 格栅与电机共同分配 yaw

- 格栅共模优先值从 −0.16 减半为 **−0.08**。原最大幅度保持不变，允许范围仍为 **−0.16～0**，并随低推力渐入。
- 先保证 roll/pitch 差模行程，再尝试用优先共模和电机差速实现 yaw。
- 如果电机差速到达可行边界，把剩余 yaw 请求交给可用格栅共模范围：正向纠偏不足时，负共模朝零退让；负向纠偏不足时，可使用优先值到 −0.16 之间的预留量。之后重新计算受约束的电机输出。
- 选择过程是随需求分段连续的分配，没有使用“固定共模开/关”的继电器切换；仍受实际舵机剩余行程约束。

保留的边界：yaw 目标 **±0.30 rad/s**、yaw 指令一阶时间常数 **0.20 s**、基础下/上目标转速比 **1.08**、任一方向最大/最小预测转速比 **1.15**、两路物理电机命令各 **≤0.45**、yaw 积分力矩 **±0.003 N·m**。

基础悬停前馈仍约为物理输出 **0.276/0.350**。这不是固定最终输出：优先共模减小后，剩余 yaw 力矩补偿和反馈会改变电机分配。没有把两路 PWM 相等当作实际转速相等。

### yaw 速率模式下的姿态参考

新 contract 中，roll/pitch 目标以**当前实测航向**为基准构造；yaw 继续跟踪角速度指令。这样在 yaw 跟踪不足时，不再让一个持续积分却无法达到的目标航向牵动 roll/pitch 参考轴。旧 contract 的参考行为保留。

## 配置与主要文件

- 部署配置：`px4_trans/configs/lqi_coaxial_collective_airframe1_yaw_balanced_v2.yaml`。
- 新参数 `allocation_safety.roll_pitch_integral_moment_fraction=0.50`、`grid_common_preference_fraction=0.50`。
- 核心：`LqiCoaxialCollectiveCore.hpp`；参考适配器：`LqiManualReference.hpp`。
- 模型 ABI v5、collective contract=4。两项参数及行为版本进入模型 checksum，当前为 **0x30298daf**。
- 原有暂定下桨反扭矩修正比例 0.70 和命令—转速表保持原来源，没有宣称这次完成了新的实机气动辨识。

## 验证与边界

1. `run_lqi_core_check.sh` 全部通过：保护、输出映射、三套 golden vectors、原有转速/输出边界扫描，以及新增三轴共享分配测试。
2. 新增回归覆盖：正负两向、roll/pitch 两轴各持续20秒饱和；积分硬上限；力矩裁剪和舵机向量缩放时的积分回退；低推力不累积；格栅正 yaw 时回零和负 yaw 时受限预留；共模分配连续性；不可达 yaw 不再旋转 roll/pitch 参考。
3. payload 标定/manifest 两项测试通过；v2 payload 读回 checksum 一致；旧 v1 的五组 golden vectors 在兼容路径仍通过。
4. 对同一份日志输入进行近似固定输入回放：v1 估计 roll 积分力矩峰值约 0.0327 N·m，v2 约 0.00394 N·m；v2 pitch 积分也保持在 0.0045 N·m 内。27–27.5 s 的 v2 共模退到零，v1 则仍约 −0.155。**这些是固定记录输入下的算法对照，不能预测改版后真实飞行轨迹。**原 v1 源码已保存在 `analysis/crash_20261105_unknown/controller_v1_source`，防止以后代码改变污染对照。
5. 补充了调用实际 C++ 核心的非线性三轴刚体、平移、上下桨和舵机一阶动态、格栅作用点及耦合模型。名义工况、持续 roll 外扰、舵机时间常数0.10秒、格栅偏转效能增大、下桨系数比0.65/0.80等六组设定均通过声明的姿态/输出/积分边界与末段恢复检查。**0.006 N·m 持续 roll 外扰工况仍有约24.4°最大 roll 瞬态**，表明大偏置下的动态裕度并不充分；不能用“六组通过”推断实机稳定。
6. 完整 `micoair_h743-v2_nncontrol` 固件编译通过；打包镜像解压、尺寸、内嵌 profile/参数及 checksum 核对见固件目录的 `build_manifest.json`。

完整 SimEnv 源码在当前工作区不可导入，因此没有声称完成完整 SimEnv 测试。上述自建刚体模型没有真实桨流、供电、IMU 振动、机械间隙/卡滞或接地；也没有复现实机独立的上下桨命令—转速曲线。

1.15 仍是**模型稳态转速约束**，不是实测 ESC RPM 闭环。11月5日日志已显示实测比值可能达到1.30以上，这个模型误差仍需要台架标定解决。格栅共模方向、实际角度和 yaw/roll/pitch 耦合效能亦未由软件测试确认。

## 复现

```sh
python3 px4_trans/tools/generate_coaxial_collective_backend.py \
  --deployment-config px4_trans/configs/lqi_coaxial_collective_airframe1_yaw_balanced_v2.yaml \
  --output-dir px4_trans/generated/airframe-1/yaw-balanced-v2 \
  --reuse-model-manifest px4_trans/generated/airframe-1/yaw-balanced-v1/lqi_coaxial_collective_manifest.json
bash px4_trans/tests/run_lqi_core_check.sh
python3 -m unittest Identification.tests.test_coaxial_collective_firmware_payload
python3 analysis/yaw_balance_v2_20261105/replay_compare.py
python3 analysis/yaw_balance_v2_20261105/coupled_check.py
```

生成后在 `px4_trans/px4` 中执行 `make micoair_h743-v2_nncontrol`。本次构建继承工作区已有的日志和保护改动，没有自动刷写。

固定输入回放结果：`analysis/yaw_balance_v2_20261105/frozen_replay_metrics.json`；三轴模型结果：`coupled_metrics.json`。新固件应先台架检查实际共模方向与退让、roll/pitch 指令响应和 ESC 转速关系，再评估飞行条件。
