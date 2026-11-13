# v3 软件抑振：仅供台架检查

`micoair_h743-v2_roll_pitch_damping_v3_BENCH_ONLY.px4` 未刷机，未替换默认 v2 payload。

- roll/pitch Kp/Kd/Ki：0.012 / 0.004 / 0.020。
- 积分仍有界、按方向抗饱和，力矩容量±0.0081 N·m；总请求仍限±0.009 N·m。
- yaw、格栅共模边界、转速比/电机输出边界及保护保持。
- 显式检查50Hz保持和舵机滞后；同一日志输入下4–8Hz Servo1输出RMS约为v2的25.4%。这不是实机振动测试。

**大静态roll偏置压力测试仍失败，不能据此直接进行自由飞行。**约300Hz原始机械振动仍需硬件处理。

详细说明：`Docs/ROLL_PITCH_DAMPING_V3_zh.md`。全部通过项、失败项及哈希见 `build_manifest.json`。
