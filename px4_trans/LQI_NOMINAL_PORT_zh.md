# Nominal LQI PX4 移植契约

当前部署目标是 `Controller/configs/lqi_sim2real_micro_coaxial_4out.yaml` 的
4×13 LQI，不是蒸馏实验中仍存在的 5 输出教师控制器。

## 执行器所有权

- 飞手的 `manual_control_setpoint.throttle` 从 `[-1,1]` 映射到 `[0,1]`，作为
  `motors[0]` 上桨命令原样输出，LQI 不修改上桨。
- 下桨的基准开度跟随同一个飞手开度，LQI 的第 0 路增量叠加到该基准后限幅到
  `[0,1]`。
- 油门空闲门控：映射后的飞手开度为 `0` 时，LQI 不计算控制输出、不更新持久
  状态，上桨/下桨/三路舵机全部输出 `0`；只有油门出现非零值后才开始输出。
- 上下桨开度差限制：下桨归一化开度被限制在飞手上桨开度 `±0.2`（20 个百分点）
  内，并同时保持在 `[0,1]`。
- LQI 的第 1～3 路直接作为三路舵机归一化命令，限幅到 `[-1,1]`。
- PX4 输出驱动负责实际 PWM 范围、方向、disarmed/failsafe 值；控制器不重复应用
  物理方向或 trim。

## 状态与参考系

13 维状态顺序固定为：

```text
[roll_error, pitch_error, p_error, q_error, r_error,
 upper_motor_error, lower_motor_error,
 servo_1, servo_2, servo_3,
 integral_roll, integral_pitch, integral_yaw_rate]
```

- 世界系为 NED，机体系为 FRD。
- 姿态为 Hamilton 四元数 `[w,x,y,z]`，表示机体 FRD 到世界 NED 的旋转 `q_wb`。
- 姿态误差严格使用 `conjugate(q_target_wb) * q_current_wb`，即与 Python
  `flight_controller.math.quaternion_rotation_error` 相同的 target-to-current
  局部旋转向量；只取 roll/pitch 两项，偏航采用角速度模式。
- 角速度和目标角速度都在机体 FRD 中，误差为 current-minus-target。
- PX4 正 pitch 杆表示向前推杆/机头下压，因此转换到 FRD Euler 目标时必须取负号；
  正 roll 杆保持正号。
- roll/pitch/yaw 指令使用部署清单中的一阶滤波；目标 yaw 从当前航向初始化并积分
  yaw-rate 指令，避免偏航过程中 roll/pitch 误差轴错位。
- `q` 和 `-q` 必须产生完全相同的控制结果，主机测试已覆盖该约束。

## 电机与舵机状态

电机目标转速使用与 Python 控制器一致的 PWM→rad/s 分段线性表。若
`esc_status` 中 Motor1、Motor2 的新鲜反馈同时可用，则按 actuator function 匹配、
对机械 RPM 取绝对值并转换为 rad/s；否则两路一起回退到命令驱动的一阶观察器。
舵机没有角度反馈，始终使用命令驱动观察器。

## 框架与权重分离

- `LqiControllerModel.hpp`：稳定的数据 ABI、维度和参考系/所有权枚举。
- `LqiControllerCore.hpp`：不包含任何具体权重的通用 500 Hz LQI 算法。
- `LqiManualReference.hpp`：PX4 摇杆到目标姿态/角速度的适配层。
- `LqiNominalModel.hpp`：生成的 nominal 数值载荷，禁止手改。
- `px4_trans/configs/lqi_nominal_deployment.yaml`：模型来源和命令尺度清单。

修改权重来源或命令尺度后，从仓库根目录重新生成并验证：

```bash
PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
  .venv/bin/python px4_trans/tools/generate_lqi_backend.py
bash px4_trans/tests/run_lqi_core_check.sh
```

生成器会同时更新 checksum、权重字节数以及 Python/C++ 黄金向量。模型描述中的
schema、NED/FRD、四元数约定、执行器所有权或观察器模式不匹配时，核心会拒绝运行。
