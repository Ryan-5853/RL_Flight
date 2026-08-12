# PX4 HIL 参考系与执行器方向核对

本文档以当前部署控制器
`Controller/configs/lqi_sim2real_micro_coaxial_4out.yaml` 和
`px4_trans/configs/lqi_nominal_deployment.yaml` 为准，说明控制器假设的
坐标系、姿态/角速度方向、电机/舵机顺序、反扭力方向和格栅偏转方向，
并给出硬件与 HIL 的核对步骤。

## 1. 控制器假设的参考系

- 世界系：NED（北、东、地）。
- 机体系：FRD（前 x、右 y、下 z）。
- 姿态四元数：Hamilton `q_wb = [w,x,y,z]`，表示“机体 FRD → 世界 NED”的旋转。
- 角速度：机体系 FRD `[p,q,r]`。
  - `p > 0`：右翼下沉（正滚转）。
  - `q > 0`：机头上仰（正俯仰）。
  - `r > 0`：机头向右转（从上方看顺时针，正偏航）。
- 姿态误差：`conjugate(q_target) * q_current`，取 roll/pitch 两项；
  偏航只使用角速度误差 `r_current - r_target`。

## 2. 执行器顺序

| 物理含义 | SimEnv control 列 | PX4 功能号 | 参数 | 说明 |
| --- | --- | --- | --- | --- |
| 上桨 | `control[:,0]` | Motor 1 | `HIL_ACT_FUNC1=101` | 飞手/总距拥有 |
| 下桨 | `control[:,1]` | Motor 2 | `HIL_ACT_FUNC2=102` | LQI 拥有 |
| 舵机 1 | `control[:,2]` | Servo 1 | `HIL_ACT_FUNC3=201` | 对应格栅 1 |
| 舵机 2 | `control[:,3]` | Servo 2 | `HIL_ACT_FUNC4=202` | 对应格栅 2 |
| 舵机 3 | `control[:,4]` | Servo 3 | `HIL_ACT_FUNC5=203` | 对应格栅 3 |

`HIL_ACT_REV` 必须为 `0`。HIL 中所有通道使用 SimEnv/控制器的归一化方向，
真实舵机/电机方向只由 QGC Actuators 和 `PWM_MAIN_REV` 处理，不能在 HIL 里
用 `HIL_ACT_REV` 纠正。

## 3. 机体与格栅几何（sim2real_micro_coaxial）

| 项目 | 数值（FRD） |
| --- | --- |
| 质量 | 0.5 kg |
| 质心 | `[0, 0, 0]` |
| 惯量 | `[7.5e-4, 7.5e-4, 1.1e-3]` kg·m² |
| 中立推力方向 | `[0, 0, -1]` |
| 直接推力中心 | `[0, 0, 0]` |
| 推力占比 | 直接 0.65，格栅 1/2/3 各 0.1167 |

格栅位置与偏转轴：

| 格栅 | 气动中心 | 偏转轴 | 正舵角产生的力矩（悬停附近） |
| --- | --- | --- | --- |
| grid_1（机尾） | `[-0.05, 0, 0.05]` | `[-1, 0, 0]` | `Mx +`，`My 0`，`Mz +` |
| grid_2（右前） | `[0.025, 0.0433, 0.05]` | `[0.5, 0.866, 0]` | `Mx -`，`My -`，`Mz +` |
| grid_3（左前） | `[0.025, -0.0433, 0.05]` | `[0.5, -0.866, 0]` | `Mx -`，`My +`，`Mz +` |

上表由 `LocalPlantModel.control_effectiveness()` 在悬停配平点计算得到：
`servo1 +0.01 → Mx≈+1.0e-4`，`servo2 +0.01 → Mx≈-1.8e-5, My≈-1.05e-4`，
`servo3 +0.01 → Mx≈-8.2e-5, My≈+6.8e-5`（具体幅值随推力变化，符号不变）。

## 4. 上下桨反扭力方向

SimEnv 动力学定义：

```text
reaction_torque = Q_upper - Q_lower
motor_reaction_moment_b = reaction_torque * airflow_axis_b
airflow_axis_b = -neutral_thrust_direction_b = [0, 0, +1]
```

因此：

- `Q_upper > Q_lower` 时，机体反扭力矩沿 `+z`（正偏航，机头向右/顺时针）。
- `Q_upper < Q_lower` 时，反扭力矩沿 `-z`（负偏航）。
- sim2real 标称两桨 `torque_coefficient` 相等（`2.30e-8`），悬停时反扭力相互抵消，
  偏航控制主要靠上下桨差速和格栅偏转。

核对硬件时：确认上桨、下桨的实际旋向，以及“上桨更快时机头向哪转”与上式一致。

## 5. LQI 输出方向（控制器期望）

LQI 输出为 `delta = -K * state`，四路输出为 `lower, servo1, servo2, servo3`，
上桨由外部总距直接给出。悬停附近期望符号：

| 误差 | lower | servo1 | servo2 | servo3 |
| --- | --- | --- | --- | --- |
| roll_error > 0 | ~0 | 负 | 正 | 正 |
| pitch_error > 0 | ~0 | 0 | 正 | 负 |
| p > 0 | ~0 | 负 | 正 | 正 |
| q > 0 | ~0 | 0 | 负 | 正 |

对照第 3 节的力矩表，这些命令产生的力矩方向与误差方向相反，即负反馈。

## 6. 检查步骤

### 6.1 静态参数检查（不上电/不装桨）

在 PX4 NSH 或 QGC 中确认：

```text
param show HIL_ACT_FUNC1
param show HIL_ACT_FUNC2
param show HIL_ACT_FUNC3
param show HIL_ACT_FUNC4
param show HIL_ACT_FUNC5
param show HIL_ACT_REV
```

期望：`101, 102, 201, 202, 203`，`HIL_ACT_REV=0`。

同时确认 WebUI 加载的 SimEnv 是 `SimEnv/configs/sim2real_micro_coaxial.yaml`。
启动 HIL 后查看 `WebUI/runtime-runs/runtime-<session-id>.jsonl` 的 `session`
记录：`actual_model.body.mass` 应为 `0.5` 左右，`motors.pwm_to_rpm_table`
应为 7 个点。

### 6.2 姿态与角速度方向（通电，不开桨）

在 WebUI/QGC 姿态页面观察：

1. 机头朝前、机体水平时，roll/pitch/yaw 应接近 0。
2. 右翼下沉 → `roll > 0`，`p > 0`。
3. 机头上仰 → `pitch > 0`，`q > 0`。
4. 机头向右转（俯视顺时针）→ `yaw > 0`，`r > 0`。

任一符号相反，先检查飞控安装方向/`SENS_BOARD_ROT`，不要改 HIL 映射。

### 6.3 电机与反扭力方向（HIL，不开桨）

1. 在 HIL 会话中让上桨命令为 0.3、下桨 0：
   - `truth.motor_speed[0]` 应增大，`truth.motor_speed[1]` 保持 0。
   - 观察 `truth.angular_velocity_b[2]`：若 `Q_upper` 更大，`r` 应向正方向变化。
2. 交换：下桨 0.3、上桨 0，`r` 应向反方向变化。
3. 用 `listener actuator_motors` 确认 PX4 的 Motor1/Motor2 分别对应上/下桨。

### 6.4 舵机方向（HIL，不开桨）

在 HIL 中分别给单个舵机 +0.2 的阶跃（或使用 QGC Actuators 测试），
观察 `truth.attitude_q_wb` / `truth.angular_velocity_b` 的响应，符号应满足：

| 通道 | 期望响应 |
| --- | --- |
| servo1 + | `p` 正（右翼下沉），`yaw` 正 |
| servo2 + | `p` 负，`q` 负，`yaw` 正 |
| servo3 + | `p` 负，`q` 正，`yaw` 正 |

若符号整体反了：检查 SimEnv 配置是否误用 `template.yaml`/`example.yaml`
（它们的格栅偏转轴正好相反），或舵机物理连杆方向是否反了。
若只是某一路反/换位：检查 `HIL_ACT_FUNC3..5` 与物理舵机、格栅的对应关系。

### 6.5 闭环一致性（HIL 日志）

用在线详细日志验证“命令力矩方向”与“实际角加速度方向”一致：

```bash
jq -r 'select(.type=="step") | [.step, .sim_time_s,
  (.truth.attitude_q_wb|join(",")),
  (.truth.angular_velocity_b|join(",")),
  (.controller.command|join(","))] | @tsv' \
  WebUI/runtime-runs/runtime-<session-id>.jsonl | head
```

在小误差阶段，`controller.command[2:5]` 的符号应满足第 5 节表格；
同时 `truth.angular_velocity_b` 的变化方向应使姿态误差减小。
若命令方向正确但误差仍增大，检查 SimEnv 是否为 sim2real 机体、`HIL_ACT_REV`
是否为 0；若都正确，则是增益/执行器带宽问题，不是方向问题。

## 7. 常见错误

- 使用 `example.yaml`/`template.yaml` 跑 HIL：格栅偏转轴与控制器相反，发散。
- `HIL_ACT_REV` 非 0：输出被取反，正反馈发散。
- `HIL_ACT_FUNC3..5` 与物理舵机顺序不对应：表现为某一轴/两轴失控。
- 上/下桨顺序颠倒：推力大小仍正确，但反扭力和差速偏航方向反了。
- 飞控安装方向错误：roll/pitch/yaw 全部或部分反号，应先修正 `SENS_BOARD_ROT`。

