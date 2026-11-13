# 格栅共模 / 近等转速 yaw 配平候选 v1

固件：`micoair_h743-v2_nncontrol_yaw_balanced_v1.px4`。未刷机、未实飞验证。

- yaw 满杆 ±0.30 rad/s，指令一阶时间常数 0.20 s。
- 格栅共模 −0.16（模型舵角约 −3.2°），roll/pitch 行程优先。
- 基础下/上桨目标转速比 1.08，预测稳态转速最大/最小比不超过 1.15。
- 两路 `actuator_motors` 输出均不超过 0.45；yaw 积分力矩限制 ±0.003 N·m，并允许饱和回退。
- 0.66 kg 基础悬停输出约 0.276/0.350。实际输出仍随总推力和有界反馈变化。

这不是实测 RPM 闭环：不同桨负载/曲线和瞬态可能使实际 RPM 比超出预测限制。
负共模产生所需负 yaw 的方向和效能需先在实际机架台架确认。

完整设计、验证和复现命令：`Docs/COAXIAL_YAW_BALANCED_V1_zh.md`。
固件哈希、payload checksum 和源码哈希：`build_manifest.json`。
