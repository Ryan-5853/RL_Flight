# 两机侧拼组合体协同控制器：Demo 设计

状态：控制器待实现；组合姿态估计器已完成。

更新日期：2026-11-03

## 1. Demo 目标与范围

两架同型共轴反桨矢量推进无人机沿组合体 Y 轴刚性连接，由主机运行一套组合控制器
`C_PAIR`，使用已有组合姿态估计器反馈，一次输出两机各五路执行器命令。

首版只做悬停附近、小角度、低速的姿态与竖直加速度控制：

- 总距：复用单体竖直加速度 PI，控制组合总推力；
- Roll：左右推力差；
- Pitch：两侧格栅共同产生组合体 Pitch 力矩；
- Yaw：两侧上下桨差动，优先保持各侧推力。

不新增组合估计器、双机辨识、位置闭环、在线拓扑识别、复杂故障恢复或通用六维分配器。
链路、控制权切换和断链动作沿用现有 AIRLINK/owner 逻辑，本文不重复实现。

## 2. 固定几何与坐标

A 为主机、B 为从机；A/B/C 均为 FRD，N 为 NED。C 原点取组合体标称重心。
采用现有估计器的安装约定：

```text
r_A_C = [0, +a, 0]
r_B_C = [0, -a, 0]
R_CA = Rz(-30°)           # A 向量 -> C 向量
R_CB = Rz(+30°)           # B 向量 -> C 向量
v_C = R_Ci v_i
v_i = R_Ci^T v_C
```

A 在 C 的 +Y 侧，B 在 -Y 侧；两机推力轴均为 C 系 -Z。
避免用“左机/右机”命名，以免与 FRD 的 +Y 向右混淆。C 轴向不等于 A 轴向。

Demo 只配置 `a`、两侧质量、惯量和安装 yaw，使用与 `CEKF_*_YAW` 相同的安装角。
配置加载时检查一次数值与旋转关系即可，不引入拓扑版本、联合哈希或运行时兼容性协商。
运行中修改安装参数后，重新初始化控制器。

### 2.1 质量与惯量

先旋转各机惯量，再加平行轴项：

```text
m_pair = m_A + m_B
I_C = sum_i { R_Ci I_i R_Ci^T + m_i [(r_i^T r_i) E - r_i r_i^T] }
```

`E` 为单位阵。Demo 默认质量对称、连接件影响较小；连接件明显改变重心或惯量时再补参数。

对相同单体、绕 Z 安装角 ±θ：

```text
Ixx_C = 2 (Ixx cos²θ + Iyy sin²θ) + 2 m a²
Iyy_C = 2 (Ixx sin²θ + Iyy cos²θ)
Izz_C = 2 Izz + 2 m a²
```

沿 Y 拼接确实不给 `Iyy_C` 增加平行轴项，但安装旋转仍会混合单体 `Ixx/Iyy`。
当前基线两者相等，因此 θ=30° 时仍可近似使用 `Iyy_C=2 Iyy_unit`。

以 `m=0.66 kg`、`I_unit=diag(5.0e-4,5.0e-4,7.333333e-4) kg m²`、
`a=0.075 m` 为例：

| 参数 | 数值 |
| --- | ---: |
| `m_pair` | `1.320 kg` |
| `Ixx_C` | `8.425e-3 kg m²` |
| `Iyy_C` | `1.000e-3 kg m²` |
| `Izz_C` | `8.892e-3 kg m²` |

上述几何值是 Demo 初值，实际中心间距不同就更新 `a`。

## 3. 直接复用已有估计器接口

订阅现有 `composite_attitude_estimate`，不再新建一套带 epoch/拓扑指纹的状态协议。
实际消息定义见 [CompositeAttitudeEstimate.msg](../px4_trans/px4/msg/CompositeAttitudeEstimate.msg)。

| 已有字段 | 控制器用途 |
| --- | --- |
| `timestamp_sample`、`timestamp` | 采样去重、状态年龄 |
| `q` | C -> NED 四元数，Hamilton WXYZ |
| `angular_velocity` | C 系角速度，rad/s |
| `specific_force` | C 系虚拟 IMU 比力，m/s² |
| `valid` | 基本输入有效性 |
| `reset_counter` | 估计器重置时重置控制器内部状态 |

消息没有 `linear_acceleration_n`。控制器适配层计算：

```text
a_N = R_NC specific_force_C + [0, 0, g]
a_up = -a_N.z
```

水平静止时 `specific_force_C≈[0,0,-g]`，应得到 `a_up≈0`。
杆臂补偿由已有估计器负责；未启用时，小角度 Demo 接受其近似误差，不在控制器重复补偿。

参考输入为 `q_NC_ref`、`omega_C_ref` 和归一化总距 `c∈[0,1]`。
目标姿态必须在 C 系定义，不能直接把 A 系目标四元数作为 C 系目标。
参考角速度也统一表达在当前 C 系。

输出沿用 `AirlinkPairCommand` 的主从两组命令：

```text
A: [upper_motor, lower_motor, servo_1, servo_2, servo_3]
B: [upper_motor, lower_motor, servo_1, servo_2, servo_3]
```

电机、舵机的归一化范围和配平沿用现有五路契约。从机只执行命令，不再运行第二套组合闭环。

## 4. 控制环

目标频率 500 Hz。按新采样更新，正常 `dt=0.002 s`；重复采样不重复积分。
调度间隔明显异常时重置本次积分更新，不拿固定 2 ms 对长时间缺帧补积分。

### 4.1 总距：竖直加速度 PI

复用 `CoaxialCollectiveServoController` 的总距结构及当前 YAML 参数：

```text
a_ref = (2c - 1) * 5.0                     # m/s²；c=0.5 为零加速度
alpha = dt / (0.010 + dt)
a_filtered += alpha * (a_up - a_filtered)
e_a = a_ref - a_filtered
I_a = clamp(I_a + e_a * dt, -2.0, +2.0)
a_corrected = a_ref + 1.0 * e_a + 4.0 * I_a
T_C = m_pair * (g + a_corrected) / max(cos_tilt, 0.35)
```

总推力限制到两机可用推力范围，首轮沿用总能力的 `0.02～0.98` 比例。
`c≤0.02` 时电机置零、舵机回配平、积分清零。积分限幅值沿用现有配置；
`I_a` 是加速度误差的时间积分，不将配置字段名误当成其物理单位。

### 4.2 Roll/Pitch：姿态 LQI

复用单体四元数误差函数 `quaternion_rotation_error(current, target)`，取最短旋转的
X/Y 分量。小角度附近分别构造：

```text
x_roll  = [e_roll,  p-p_ref, integral(e_roll)]
x_pitch = [e_pitch, q-q_ref, integral(e_pitch)]
M = -K x
```

Roll 使用组合体 `Ixx_C` 重新求 LQI；Pitch 复用单体增益，输出先明确为
**单体等效力矩** `My_unit`，再转换为组合总力矩：

```text
My_C = (Iyy_C / Iyy_unit) * My_unit       # 当前约为 2 * My_unit
```

随后两侧各承担 `My_C/2`。不能把 `My_unit` 当组合总量再除以二，否则增益少一半。
该复用以两侧格栅均能达到目标力矩为前提；推力变化通过分配器的工作点映射处理。

### 4.3 Yaw：角速度 LQI

沿用现有单体控制器的偏航速率控制，不增加绝对航向外环：

```text
x_yaw = [r-r_ref, integral(r-r_ref)]
Mz_C = -K_yaw x_yaw
```

模型为 `rate_dot=Mz/Izz_C`、`integral_dot=rate_error`，使用组合体 `Izz_C` 求增益。
首版的目标四元数不用于单独锁定绝对 yaw；Yaw 输入通过 `r_ref` 生效。

不要使用原稿的 `[yaw_error, rate_error, integral(rate_error)]` 三状态模型：
小角度下 yaw 误差和速率误差积分的导数相同，会引入不可控的冗余状态。

### 4.4 参数取值

直接读取当前配置，避免在文档复制一套过期数值：
[单体控制配置](../Controller/configs/lqi_coaxial_collective_servo_airframe1_manual_v3.yaml)。
当前 LQI 尺度为：

| 轴 | 角度尺度 | 速率尺度 | 积分尺度 | 输入力矩尺度 |
| --- | ---: | ---: | ---: | ---: |
| Roll | `0.3490658 rad` | `1.4 rad/s` | `0.20 rad·s` | `0.008 N·m` |
| Pitch | `0.3490658 rad` | `1.4 rad/s` | `0.20 rad·s` | `0.008 N·m` |
| Yaw | 不使用 | `3.0 rad/s` | `0.80 rad` | `0.020 N·m` |

`input_weight_scale=0.15`。力矩尺度用于 LQI 权重，不等于执行器真实权限。
Roll/Yaw 用各自状态模型做 ZOH 离散后求离散 Riccati 方程，Pitch 沿用单体增益。
首次加载参数时计算增益，不在控制循环里反复求解。

## 5. 执行器分配

固定顺序：总距与 Roll → Pitch 格栅 → Yaw 电机差动 → 输出与残差。
复用已有 `LocalPlantModel`、电机表和格栅逆映射，不另造推力模型。

### 5.1 总距与 Roll

```text
Mx_C = a * (T_B - T_A)
T_A = T_C/2 - Mx_C/(2a)
T_B = T_C/2 + Mx_C/(2a)
```

因此正 Roll 需要 B 增推、A 减推。先限总推力，再按两侧能力限制 Roll。
相同单机范围 `[T_min,T_max]` 时：

```text
Mx_max = a * min(T_C - 2*T_min, 2*T_max - T_C)
Mx_C = clamp(Mx_C, -Mx_max, +Mx_max)
```

限幅后重新算 `T_A/T_B`，保持 `T_A+T_B=T_C`。
通过现有推力反解获取两侧电机基线；不直接把同一个 PWM 增减量当作推力差。

### 5.2 Pitch 格栅

这里的目标向量明确是**力矩向量，单位 N·m**，不是未定义的“格栅矢量”：

```text
M_A_C_target = [0, My_C/2, 0]
M_B_C_target = [0, My_C/2, 0]
M_i_local_target = R_Ci^T * M_i_C_target
```

把局部 X/Y 力矩送入单体已有三舵机分配器，按各侧 `T_i` 对应工作点计算。
±30° 安装使 C 系纯 Pitch 映射为单体局部 Roll/Pitch 混合，因此两侧舵机值通常不同。

沿用单体分配器尽量保持推力/Yaw 不变的做法。舵机饱和时，相对配平的完整三舵机
增量统一缩放；若需要两侧联合缩放，取两侧可用缩放系数的较小值。
Pitch 分配不直接修改电机基线。

模型验算使用完整力与力矩合成：

```text
F_C = sum_i R_Ci F_i
M_C = sum_i [R_Ci M_i + r_i_C × (R_Ci F_i)]
```

格栅横向力也会产生杆臂力矩，不能只旋转局部力矩后求和。
首版记录这部分残差，不增加 Roll/Pitch/Yaw 之间的迭代补偿。

### 5.3 Yaw 电机差动

默认每侧目标反扭矩为 `Mz_C/2`。在每侧既定目标推力 `T_i` 下，
使用已有模型在当前电机基线附近求上下桨命令：

```text
T_i(uU,uL) ≈ T_i_target
Mz_i(uU,uL) ≈ Mz_C/2
```

Demo 优先复用已有电机局部逆映射，限制差动幅度；不要求首版实现二维查表、
全局非线性优化或一侧饱和后向另一侧重分配。
超出能力时缩小差动，保留推力优先级，并记录实际推力误差与未达到的 Yaw 力矩。

“推力保持”是稳态模型中的近似目标；上下电机动态不同，不能承诺瞬态严格守恒。
局部线性解是否足够，由模型检查中的残差决定；误差明显时再增加固定推力的一维搜索。

## 6. Demo 保留的最小保护与日志

只保留影响控制计算的基本处理：

- 状态无效、超时或输入含 NaN/Inf 时，不发布有效控制输出，交给已有 owner 处理。
  超时阈值复用已有链路/路由约定，不另设一套健康状态机。
- 电机与舵机按现有范围限幅，分配器尽量通过整体缩放保持方向。
- 各积分器限幅；对应通道饱和且误差继续推动饱和时，撤回该步积分；反向误差允许退饱和。
- 启动、退出、零总距或 `reset_counter` 改变时清空积分并初始化滤波器。
  重新进入时使用新的状态，不额外等待稳定窗口。

不增加拓扑指纹联锁、epoch 握手、完整故障分类、恢复协商、按轴 owner 仲裁或加载审批。

最小日志：采样时间、状态年龄、姿态/角速度及目标、竖直加速度及目标、
请求/模型估计的 `[T_C,Mx,My,Mz]`、两侧目标推力、十路命令和饱和标志。
不要求每帧打印静态矩阵、配置哈希或完整内部求解过程。

## 7. 最少模型检查

实现后先用已有单体模型拼成双机模型，检查以下几项即可推进 Demo 调参：

1. 水平静止输入得到近零运动学加速度；中位总距目标为零竖直加速度。
2. 正 Roll 对应 B 增推、A 减推；未饱和时两侧目标推力之和等于总距请求。
3. 实际 ±30° 安装下，两侧格栅合成正 Pitch；验证力矩旋转和横向力杆臂项。
4. 相同小 Pitch 误差下，组合角加速度量级接近单体，排除多除一次二的错误。
5. Yaw 差动前后比较两侧模型推力，确认误差足以支撑 Demo；饱和时记录权限削减。
6. 悬停、三轴分别小阶跃、三轴小指令叠加，检查发散、长期饱和和明显交叉耦合。
7. 检查新设计的离散闭环极点，并简单覆盖零总距、估计器重置、状态失效。

不为首版增加通用 90°/180° 安装测试、结构载荷认证、完整故障注入矩阵或额外双机辨识。
这些检查是实现验证，文档本身不代表检查已经通过。

## 8. 已知近似与后续改进

- 当前参数来自单体加几何合成，未补偿气动互扰与连接件柔性。
- Pitch 增益复用适用于当前近似对称惯量和小角度工况；Roll 导致两侧推力差时，
  低推力侧格栅可能先饱和，首版接受共同降低 Pitch 权限。
- Yaw 惯量增加而反扭矩能力有限，先接受较慢的偏航响应，不追求与单体相同速度。
- 格栅耦合与电机动态先通过日志观察；只有 Demo 中出现明显问题才增加残差补偿和非线性求解。

## 9. 参考

- [组合体姿态估计器](COMPOSITE_ATTITUDE_ESTIMATOR_zh.md)
- [AIRLINK 协议](AIRLINK_PROTOCOL_V1_zh.md)
- [双机体刚性拼接可行性报告](../Identification/PAIR_ASSEMBLY_FEASIBILITY_zh.md)
- [单体控制器](../Controller/src/flight_controller/coaxial_collective.py)
- [单体执行器模型](../Controller/src/flight_controller/plant.py)
- [单体物理参数](../SimEnv/configs/sim2real_micro_coaxial_3100kv_airframe1_manual_v2.yaml)
- [单体控制参数](../Controller/configs/lqi_coaxial_collective_servo_airframe1_manual_v3.yaml)
