# 2026-11-05 失控后的 v2 修复候选

固件：`micoair_h743-v2_nncontrol_yaw_balanced_v2.px4`。未刷机、未实飞验证。

- roll/pitch 按轴、按方向抗饱和；积分力矩各限 ±0.0045 N·m。
- 低目标推力不增加 roll/pitch 积分。
- 格栅共模优先值 −0.08，可按 yaw 需求在 −0.16～0 内分配；正 yaw 受限时负共模可回零，roll/pitch 行程优先。
- yaw 速率模式的 roll/pitch 参考跟随实测航向。
- 保持 yaw ±0.30 rad/s / 平滑0.20 s、预测转速比1.15、两路物理输出各≤0.45；不通过扩大差速上限纠偏。

主机测试、日志固定输入回放和六组简化三轴模型检查已完成；这些不能替代实机验证。较大持续 roll 外扰模型中仍有约24.4°瞬态。
实际转速比仍可能因标定误差超出预测值；没有新增实测 RPM 闭环。

完整说明：`Docs/COAXIAL_YAW_BALANCED_V2_zh.md`。
固件及源码哈希：`build_manifest.json`。模型 checksum：`0x30298daf`。
