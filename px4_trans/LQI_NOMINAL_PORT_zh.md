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

## 辨识模型版本（lqi_identified，19 状态复合 LQI）

2026-08-12 起，固件支持第二版 LQI 权重：由真机日志离线辨识得到的 19 状态复合
LQI。原有 13 状态标称权重（`LqiNominalModel.hpp`、`lqi_golden.*`）保持不动，
通过参数 `NN_LQI_MODEL` 选择：

- `NN_LQI_MODEL=0`：标称 13 状态 LQI（默认，行为与之前完全一致）；
- `NN_LQI_MODEL=1`：真机辨识 19 状态复合 LQI（`lqi_identified`）。

该参数在 `ln` 模块启动时读取，修改后需要重启 `ln` 模块。执行器输出仍由
`NN_LQI_OUTPUT_EN` 统一门控，辨识版本默认也不放行飞行。

19 状态顺序：

```text
[roll_error, pitch_error, p_error, q_error, r_error,
 upper_motor_error, lower_motor_error,
 latent(3 基 × 3 轴), integral_roll, integral_pitch, integral_yaw_rate]
```

- 电机状态与标称版相同：ESC 反馈新鲜且两路同时可用时使用
  `|rpm| * rpm_to_rad_s - trim`，否则回退命令驱动一阶观察器。
- 9 个 latent 状态是 3 个基时间常数（15/40/80 ms）× 3 轴的一阶滤波，输入矩阵为
  `mode_transform @ diag(servo_slopes)`，由三路舵机最终（限幅后）命令驱动；
  复合辨识系数通过增益矩阵耦合到角加速度状态。
- 参考系、四元数、摇杆适配、上下桨差动限幅、饱和冻结积分等约定与标称版一致。
- 新 ABI 位于 `LqiCompositeControllerModel.hpp`，算法位于
  `LqiCompositeCore.hpp`，数值载荷由
  `Identification/datasets/real_logs_26_8_12_v1/offline_identification_v7_hybrid.json`
  经生成器写入 `LqiIdentifiedModel.hpp`（禁止手改）。

生成器现在同时输出标称和辨识两套 golden 向量；主机检查
`px4_trans/tests/run_lqi_core_check.sh` 会依次验证两个核心。SimEnv 闭环极性
检查脚本为 `Identification/scripts/polarity_check_identified_lqi.py`，结果为
`Identification/datasets/real_logs_26_8_12_v1/polarity_check_identified_lqi.json`
（四个方向初始角加速度方向正确、4 s 内回正，舵机指令对 ±roll 反对称，输出有界）。

## 2026-08-28 调参：增大积分项与打杆指令比例

针对 `lqi_identified` 真机悬停稳态误差偏大、手动打杆响应不足的问题，保持
辨识数据和编码器不变，做了两处调整：

- **积分项约翻倍**：离线辨识重新合成增益时把 LQR 积分权重提高 4 倍
  （`integral_state_scales` 由 `[0.04, 0.04, 0.25]` 减半为
  `[0.02, 0.02, 0.125]`），新的 `analysis_gain` 中三路积分列约为原来的 2 倍，
  其余列由 LQR 耦合略增。合成配置在
  `Controller/configs/lqi_sim2real_micro_coaxial_4out_strong_integral.yaml`，
  辨识实验配置在
  `Identification/configs/lqr_sim2real_micro_offline_logs_v4_strong_integral.yaml`；
  标称 13 状态 4out 配置与权重保持不变。
- **打杆指令比例提高**：`configs/lqi_nominal_deployment.yaml` 的
  `manual_reference` 由 roll/pitch 12°、yaw 1.2 rad/s 调整为 roll/pitch 18°、
  yaw 1.5 rad/s；滤波时间常数不变。该常量为标称/辨识两版共用的摇杆参考契约，
  标称增益本身未变。

重新生成后的验证：两套主机 golden 检查全部通过；SimEnv 闭环 8 s 扰动回正中
roll 残差 0.16°→0.13°、pitch 残差 0.08°→0.003°；打杆阶跃 t90 由约 0.44 s
缩短到约 0.39 s，四个方向极性检查全部正确、输出有界。辨识模型极点半径由
0.9956 改善到 0.9912。
