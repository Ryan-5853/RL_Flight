> 恢复规则更新：AIRLINK 故障后，两机各自收到物理 RC disarm 且确认撤防后可清除故障；重新握手并再次 arm 即可恢复，无需重启。详见 `Docs/GROUND_PAIR_FLIGHT_zh.md`，以下历史永久锁存描述以此为准。

# AIRLINK v1 双机串口协议

> 2026-11-03 更新：当前地面组合飞行、External Mode 3 和断链停机锁死规则以
> [地面组合飞行流程](GROUND_PAIR_FLIGHT_zh.md) 为准；本文中的旧接线边界和单体降落回退不再适用。

状态：v1 设计规范；固件协议、状态机和 PX4 集成已实现，见
`AIRLINK_FIRMWARE_zh.md`
日期：2026-10-30
物理层：全双工 UART，`6,000,000 8N1`，无硬件流控
适用拓扑：一架主机、一架从机；从机上传传感器，主机下发从机控制命令

## 1. 目标与边界

AIRLINK v1 在两条飞控之间交换 500 Hz 的传感器状态和从机执行器命令。主机在本地
执行主机的五路命令，同时把从机的五路命令通过串口发送给从机。从机不运行远端
位置环，也不把 airlink 视为飞行动作接口。

v1 固定支持：

- Bosch BMI088：陀螺仪 `±2000 °/s`、加速度计 `±24 g`；
- 两路 ESC、两电机三舵机机架；
- 一路有效电池电压源；
- 主机到从机的五路物理执行器命令；
- 从机到主机的 IMU、姿态、角速度估计、ESC 转速和电压。

协议包含显式、可靠、可拒绝的 `PAIR_ARM`/`PAIR_DISARM` 协同 arm 事务，但不包含
通用 arm/disarm、起飞、降落、飞行模式切换、位置目标、导航、分片、设备路由、
认证或加密。`PAIR_ARM` 只是请求从机执行本机 arming checks；它不能替代本机 RC、
PX4 Commander、failsafe、安全门控或执行器授权。CRC 只检测传输错误，不提供身份
认证，也不能阻止受控链路外的字节注入。

除本规范明确规定的传感器打包方式外，不修改 PX4、SimEnv、Controller 或测试
代码。更换 IMU 量程、电压量级、ESC 方向约定、输出数量或输出映射时，必须提升
协议版本或增加显式 capability；不能悄悄复用 v1 的固定比例。

## 2. 串口与字段约定

- UART 主机和从机的 TX/RX 交叉连接，共地；必须全双工；
- `6,000,000` baud、8 数据位、无校验、1 停止位、无 RTS/CTS；
- 所有多字节整数使用小端序；
- `int16`、`int32`、`uint16`、`uint32` 分别为二进制补码或有符号/无符号整数；
- 不使用编译器 packed struct 的内存布局，发送端逐字段显式序列化；
- `SENSOR_STATE` 和 `ACTUATOR_COMMAND` payload 长度固定，不允许可选字段；
- payload 最大 512 B，v1 不分片；
- 保留字段发送时置 0。接收端遇到非零的保留状态位、帧标志或事件字段时，把
  该帧视为格式错误，不能猜测其语义。

## 3. 帧结构

### 3.1 16 B 帧头


| 偏移 | 长度 | 字段             | 约定                             |
| ---: | ---: | ---------------- | -------------------------------- |
|    0 |    2 | `sof`            | 固定线上字节`A5 5A`              |
|    2 |    1 | `version`        | v1 固定`01`                      |
|    3 |    1 | `frame_type`     | `01` DATA，`02` COMMAND          |
|    4 |    1 | `flags`          | 帧标志，见 3.2                   |
|    5 |    1 | `message_id`     | 消息类型，见第 4 节              |
|    6 |    1 | `source_role`    | `01` 主机，`02` 从机             |
|    7 |    1 | `dest_role`      | 必须与`source_role` 相反         |
|    8 |    4 | `source_session` | 发送端本次启动的非零随机`uint32` |
|   12 |    2 | `sequence`       | 每个发送方向独立递增的`uint16`   |
|   14 |    2 | `payload_len`    | payload 字节数，0–512           |
|   16 |    N | `payload`        | 消息载荷                         |
| 16+N |    4 | `crc32c`         | CRC-32C/ISCSI，小端存储          |
| 20+N |    2 | `eof`            | 固定`5A A5`                      |

完整帧长为：

```text
frame_len = 16 + payload_len + 4 + 2 = 22 + payload_len
```

`source_session` 在每次端点启动时重新生成，至少一个端点重启即结束旧控制会话。
它不是安全凭据。`sequence` 每发送一个帧加一，`0xFFFF` 后回到 `0x0000`；周期
流的序号跳变只用于丢包和乱序诊断，不用于判断 10 ms 新鲜度。可靠事件重发复用
原序号。

### 3.2 帧标志


|   位 | 名称       | 说明                         |
| ---: | ---------- | ---------------------------- |
|    0 | `ACK_REQ`  | 要求接收端对可靠事件回复 ACK |
|    1 | `RETRY`    | 该帧是先前可靠事件的重发     |
|    2 | `RESPONSE` | ACK、握手 REPORT 或测试回显  |
|    3 | `URGENT`   | 紧急撤销类事件               |
| 4–7 | reserved   | 必须为 0                     |

`SENSOR_STATE`、`ACTUATOR_COMMAND` 和 `HEARTBEAT` 的 `flags` 全为 0。
`CONTROL_ENABLE`、`CONTROL_RELEASE`、`EMERGENCY_WITHDRAW`、`PAIR_ARM`、
`PAIR_DISARM` 以及 `LINK_TEST_CONTROL` 的 START/ABORT 设置 `ACK_REQ`；首次发送时
`RETRY=0`，重发时 `RETRY=1`，同一事件的所有发送复用原序号和事务 ID。`ACK` 只
设置 `RESPONSE`，不要求被 ACK，也不重发。`LINK_TEST_DATA` 主机方向 flags 为 0，
从机回显方向只设置 `RESPONSE`。
`LINK_HANDSHAKE.REQUEST` flags 为 0，`REPORT` 设置 `RESPONSE`，`CONFIRM` 设置
`ACK_REQ`。

### 3.3 CRC-32C

CRC 参数：

- 名称：CRC-32C/ISCSI（Castagnoli）；
- reflected polynomial：`0x82F63B78`；
- initial value：`0xFFFFFFFF`；
- final xor：`0xFFFFFFFF`；
- 覆盖范围：从 `version`（偏移 2）到 payload 最后一字节；
- 不覆盖 `sof`、`crc32c` 字段本身和 `eof`；
- CRC 字段按小端序写出。

参考循环：

```c
uint32_t airlink_crc32c(const uint8_t *data, size_t length)
{
    uint32_t crc = 0xFFFFFFFFu;

    for (size_t i = 0; i < length; ++i) {
        crc ^= data[i];

        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 1u) ? (crc >> 1) ^ 0x82F63B78u : crc >> 1;
        }
    }

    return crc ^ 0xFFFFFFFFu;
}
```

CRC 自检：ASCII `123456789` 的结果为 `0xE3069283`，线上小端字节为
`83 92 06 E3`。

## 4. 消息目录


| 类别    |   ID | 消息                 | 方向             | payload | 完整帧 | 周期             |
| ------- | ---: | -------------------- | ---------------- | ------: | -----: | ---------------- |
| DATA    | `01` | `SENSOR_STATE`       | 从机 → 主机     |    42 B |   64 B | 500 Hz           |
| COMMAND | `01` | `ACTUATOR_COMMAND`   | 主机 → 从机     |     6 B |   28 B | 500 Hz，仅激活时 |
| COMMAND | `02` | `HEARTBEAT`          | 双向             |     4 B |   26 B | 500 Hz，预控制/恢复 |
| COMMAND | `03` | `CONTROL_ENABLE`     | 主机 → 从机     |    12 B |   34 B | 可靠事件         |
| COMMAND | `04` | `CONTROL_RELEASE`    | 主机 → 从机     |    12 B |   34 B | 可靠事件         |
| COMMAND | `05` | `EMERGENCY_WITHDRAW` | 双向             |    12 B |   34 B | 可靠事件         |
| COMMAND | `06` | `ACK`                | 接收方 → 发起方 |    16 B |   38 B | 响应             |
| COMMAND | `07` | `PAIR_ARM`           | 主机 → 从机     |    12 B |   34 B | 可靠事件         |
| COMMAND | `08` | `PAIR_DISARM`        | 主机 → 从机     |    12 B |   34 B | 可靠事件         |
| COMMAND | `09` | `LINK_HANDSHAKE`     | 双向，主机发起   |    24 B |   46 B | 三段式握手       |
| COMMAND | `0A` | `LINK_TEST_CONTROL`  | 主机 → 从机     |    16 B |   38 B | 可靠事件         |
| DATA | `0B` | `LINK_TEST_DATA`     | 双向回环         | 32–512 B | 54–534 B | 测试期间         |

未知 DATA 消息直接丢弃。包头和 CRC 有效、但无法支持的 COMMAND 消息，如果
`ACK_REQ=1`，回复 `UNSUPPORTED`；否则丢弃。未知消息不得改变控制状态。

## 5. `SENSOR_STATE`

### 5.1 固定 42 B payload


| payload 偏移 | 长度 | 字段                           | 类型       | 含义                          |
| -----------: | ---: | ------------------------------ | ---------- | ----------------------------- |
|            0 |    2 | `status_flags`                 | `uint16`   | 有效性与安全状态              |
|            2 |    4 | `imu_sample_time_us`           | `uint32`   | IMU 采样时刻低 32 位          |
|            6 |    4 | `attitude_sample_time_us`      | `uint32`   | 姿态采样时刻低 32 位          |
|           10 |    6 | `accelerometer_b`              | `int16[3]` | 机体系加速度，`x,y,z`         |
|           16 |    6 | `gyro_b`                       | `int16[3]` | 机体系角速度，`x,y,z`         |
|           22 |    8 | `attitude_q_wxyz`              | `int16[4]` | 机体系到当地 NED 的姿态四元数 |
|           30 |    6 | `estimated_angular_velocity_b` | `int16[3]` | 机体角速度估计，`x,y,z`       |
|           36 |    4 | `esc_rpm`                      | `int16[2]` | 上桨、下桨机械 RPM            |
|           40 |    2 | `battery_voltage_mv`           | `uint16`   | 电池电压，mV                  |

payload 必须恰好为 42 B，完整帧必须恰好为 64 B。字段顺序、量程和比例是 v1
线上 ABI，不能依赖本机 C/C++ 结构体自然对齐实现。

### 5.2 数据来源与坐标

发送的是 PX4 校准、旋转、滤波后的主题值，不是 BMI088 寄存器的原始 ADC 值：

- `accelerometer_b` 来自 `sensor_accel.x/y/z`；
- `gyro_b` 来自 `sensor_gyro.x/y/z`；
- `attitude_q_wxyz` 来自 `vehicle_attitude.q`，顺序为 `w,x,y,z`；
- `estimated_angular_velocity_b` 来自 `vehicle_angular_velocity.xyz`；
- `esc_rpm[0]` 来自 `actuator_function == Motor1` 的 ESC 报告，语义为上桨；
- `esc_rpm[1]` 来自 `actuator_function == Motor2` 的 ESC 报告，语义为下桨；
- `battery_voltage_mv` 来自配置选定的 `battery_status.voltage_v`。

ESC 必须按功能号匹配，不能假设 `esc_status` 数组下标、ESC 地址或消息到达顺序
等同于上/下桨。若功能号缺失，只能使对应一路无效。多电池系统中必须在启动时
锁定一个电压源，不能每帧随机选择。

所有向量使用 PX4 FRD 机体系和 NED 世界系的既定轴语义。`imu_sample_time_us`
取本快照所选 `sensor_gyro.timestamp_sample` 的低 32 位；配对加速度计的独立
采样时刻只在本机用于新鲜度判断，不另行上线。

### 5.3 `status_flags`


|     位 | 名称                       | 为 1 的含义                              |
| -----: | -------------------------- | ---------------------------------------- |
|      0 | `ACCEL_VALID`              | 加速度计字段可用于控制                   |
|      1 | `GYRO_VALID`               | 原始角速度字段可用于控制                 |
|      2 | `ATTITUDE_VALID`           | 四元数通过有限值、分量和范数检查         |
|      3 | `ANGULAR_VELOCITY_VALID`   | 角速度估计可用于控制                     |
|      4 | `ESC_UPPER_VALID`          | 上桨 ESC RPM 有效                        |
|      5 | `ESC_LOWER_VALID`          | 下桨 ESC RPM 有效                        |
|      6 | `BATTERY_VALID`            | 电压字段有效                             |
|      7 | `SAMPLE_FRESH`             | 本帧所需核心源采样在发送时不超过 10 ms   |
|      8 | `LOCAL_RC_VALID`           | 本机 RC 输入有效                         |
|      9 | `SAFETY_GATE_ALLOW`        | 本机安全门控允许 airlink 接管            |
|     10 | `AIRLINK_CONTROL_ACTIVE`   | 本机当前处于 airlink 控制激活态          |
|     11 | `ACCEL_CLIPPED`            | 有限的加速度值曾超出协议编码范围并被饱和 |
|     12 | `GYRO_CLIPPED`             | 有限的陀螺仪值曾超出协议编码范围并被饱和 |
|     13 | `ANGULAR_VELOCITY_CLIPPED` | 有限的角速度估计曾超出编码范围并被饱和   |
| 14–15 | reserved                   | 必须为 0                                 |

`SAMPLE_FRESH` 由发送端根据完整 64 位本地单调时间和核心源采样判断；在发送
快照时，本帧所用加速度计、陀螺仪、姿态和角速度估计都没有超过 10,000 µs
采样年龄时置 1；任一核心源缺失或过期时清 0。该位描述源采样年龄，不替代接收端
按本地到达时刻执行的链路新鲜度判断。接收端只有在 `SAMPLE_FRESH=1`、字段对应
有效位为 1 且完整帧到达年龄不超过 10,000 µs 时才把该字段交给控制器；不得对
过期帧外推或继续使用上一帧字段。

`LOCAL_RC_VALID`、`SAFETY_GATE_ALLOW` 和 `AIRLINK_CONTROL_ACTIVE` 描述从机的
本机安全状态。它们不能被主机覆盖。接收端保存这些位供控制和安全监督使用，
不得把 airlink 命令理解为 arm 或安全授权。

有效位与裁剪位按字段独立维护。单路 ESC 缺失不得清除另一路有效位或数据。
非有限值、缺失源或无法安全表示的源值必须把对应数值写 0 并清除有效位；不得用
`0xFFFF`、`-32768` 等哨兵值偷偷编码无效状态。接收端必须先检查有效位，再使用
字段，不能重新解释无效字段的原始值。

### 5.4 量化、范围与饱和

取整统一为“最接近整数，恰好半格时远离零”。所有编码先检查有限值和源字段
有效性，再应用范围规则。


| 字段       | 比例/步进              | 有符号原始值 | 解码范围                        | 超出范围处理                                         |
| ---------- | ---------------------- | ------------ | ------------------------------- | ---------------------------------------------------- |
| 加速度     | `128`，`1/128 m/s²`   | `int16`      | `-256.000`–`255.9921875 m/s²` | 饱和到`-32768/32767`，置 bit 11                      |
| 陀螺仪     | `800`，`0.00125 rad/s` | `int16`      | `-40.96`–`40.95875 rad/s`      | 饱和到`-32768/32767`，置 bit 12                      |
| 角速度估计 | `800`，`0.00125 rad/s` | `int16`      | `-40.96`–`40.95875 rad/s`      | 饱和到`-32768/32767`，置 bit 13                      |
| 姿态四元数 | `32767`，`1/32767`     | `int16`      | 分量解码限制为`[-1,1]`          | 源越界或范数不合格则整组写 0，清 bit 2               |
| ESC 转速   | `1 RPM`                | `int16`      | `-32767`–`32767 rpm`           | v1 无裁剪位；越界、缺失或非有限时写 0 并清对应有效位 |
| 电压       | `1000`，`1 mV`         | `uint16`     | `0`–`65535 mV`                 | 负值、NaN、Inf、超过`65.535 V` 时写 0 并清 bit 6     |
| 采样时间   | `1 µs`                | `uint32`     | `0`–`4294967295`               | 取源`uint64 timestamp_sample` 的低 32 位             |

编码公式：

```text
accel_i      = round_half_away(a_i * 128)
gyro_i       = round_half_away(w_i * 800)
estimated_i  = round_half_away(w_est_i * 800)
quaternion_i = round_half_away(q_i * 32767)
esc_i        = round_half_away(rpm_i)
voltage_mv   = round_half_away(voltage_v * 1000)
```

`int16` 加速度、陀螺仪和角速度估计的端点由二分补码边界决定：负端是
`-32768`，正端是 `32767`。只有这三个字段有裁剪状态位。ESC 和电压没有裁剪位，
因此不能把超范围值静默饱和成一个看似有效的端点；它们必须按无效值处理。这样
不会把超范围源伪装成一个看似有效的端点；只有真实位于允许范围内的端点值才
保持有效。

### 5.5 四元数规则

发送前要求四个分量均为有限值、每个分量位于 `[-1,1]`，且：

```text
abs(norm(q) - 1) <= 0.01
```

不满足任一条件时，整组四元数写 0，并清除 `ATTITUDE_VALID`；不允许只修正一个
分量，也不允许把无效范数归一化后继续标为有效。有效四元数的发送端编码使用
`round_half_away(q_i * 32767)`，因此 `-1` 发送为 `-32767`，`1` 发送为
`32767`。

接收端将每个原始值按 `raw / 32767` 解码并限制到 `[-1,1]`；`-32768` 作为
兼容的解码别名得到 `-1`，但 v1 发送端不得为有效四元数产生该值。接收端对
`ATTITUDE_VALID=1` 的四元数再次检查范数，容差为 `0.0101`；失败时只把本地
姿态标为不可用，不猜测或替换分量。

单位四元数逐分量量化的最坏单分量误差不超过 `1/65534`，约
`1.526e-5`；量化后范数与 1 的偏差不超过 `1/32767`，约 `3.052e-5`。

`q` 和 `-q` 表示同一个姿态。接收端不得直接要求两帧四元数原始符号相同；姿态
差异应使用同向比较，例如 `min(||q_a-q_b||, ||q_a+q_b||)`，或换算成等价的最小
转角。发送端不强制把 `w` 变为正，也不因符号翻转清有效位。

### 5.6 时间戳模运算

时间戳只供同一发送时钟内的诊断和模差使用，不能把主机与从机的 32 位时间直接
比较，也不能用它们判断包到达新鲜度。接收端的新鲜度必须使用本机单调时钟和完整
帧校验通过的时刻。

同一时钟的 32 位模差为：

```text
delta_us = (uint32_t)(new_sample_us - old_sample_us)
```

若 `delta_us < 0x80000000`，它表示向前的模差；若 `delta_us >= 0x80000000`，
距离已经超过半个回绕周期，v1 诊断路径应丢弃这个差值，而不是把它解释成一个巨大
的负年龄。例如：

```text
old = 0xFFFFFF00
new = 0x00000100
delta_us = 0x00000200 = 512
```

`imu_sample_time_us` 和 `attitude_sample_time_us` 来自不同源，不能相互相减后
声称得到传感器年龄。

## 6. `ACTUATOR_COMMAND`

### 6.1 6 B LSB-first 位流

从机完整五路命令由 48 bit 组成。位号按字节内 LSB-first 编号：


| 字段          | 位范围 | 码值范围 | 控制器域      |
| ------------- | -----: | -------: | ------------- |
| `upper_motor` |   0–8 |   0–500 | `c ∈ [0,1]`  |
| `lower_motor` |  9–17 |   0–500 | `c ∈ [0,1]`  |
| `servo_1`     | 18–27 |  0–1000 | `s ∈ [-1,1]` |
| `servo_2`     | 28–37 |  0–1000 | `s ∈ [-1,1]` |
| `servo_3`     | 38–47 |  0–1000 | `s ∈ [-1,1]` |

编码：

```text
upper_code = floor(500 * upper_motor + 0.5)
lower_code = floor(500 * lower_motor + 0.5)
servo_code = floor(500 * (servo + 1.0) + 0.5)

packed = upper_code
       | (lower_code << 9)
       | (servo_1_code << 18)
       | (servo_2_code << 28)
       | (servo_3_code << 38)
```

`packed` 的 6 个字节按 `b0 = packed[7:0]` 到 `b5 = packed[47:40]` 依次上线。
解码：

```text
upper_motor = upper_code / 500.0
lower_motor = lower_code / 500.0
servo_i     = servo_i_code / 500.0 - 1.0
```

两路电机的有效码为 `0..500`，`501..511` 非法；三路舵机的有效码为 `0..1000`，
`1001..1023` 非法。任一非法码使整个 payload 无效，不能局部钳位或只执行其他
四路。输入的非有限值和超出 `[0,1]`、`[-1,1]` 的命令也必须在发送前整帧拒绝。

量化步进在控制器域为 `0.002`。对当前标准 `1000–2000 µs` 输出映射，对应的
最终 PWM 步进为 1 µs，最大量化误差为 0.5 µs。解码后的值仍经过从机自己的
`FinalMotorOutputMap`、`PWM_MAIN_REV`、端点、舵机 trim 和机械方向配置，因此它
不是绝对 PWM 脉冲，也不是舵面位置的测量。

从机收到合法命令后必须立即使用新值；协议不增加额外低通，也不在命令超时后保持
最后动作。完整指令帧为 `16 + 6 + 4 + 2 = 28 B`。

最小、中点和最大码的位流向量：

```text
全最小：最高 00 00 00 00 00 00          packed=0x000000000000
全中点：FA F4 D1 47 1F 7D               packed=0x7D1F47D1F4FA
全最大：F4 E9 A3 8F 3E FA               packed=0xFA3E8FA3E9F4
```

## 7. 心跳、可靠事件和 ACK

### 7.1 `HEARTBEAT`

payload 4 B，完整帧 26 B：


| 偏移 | 长度 | 字段           | 类型               |
| ---: | ---: | -------------- | ------------------ |
|    0 |    2 | `health_flags` | `uint16`           |
|    2 |    2 | `reserved`     | `uint16`，必须为 0 |

`health_flags`：


|    位 | 名称                     |
| ----: | ------------------------ |
|     0 | `LOCAL_READY`            |
|     1 | `LOCAL_RC_VALID`         |
|     2 | `SAFETY_GATE_ALLOW`      |
|     3 | `AIRLINK_CONTROL_ACTIVE` |
|     4 | `INBOUND_PERIODIC_FRESH` |
|     5 | `ARMED`                  |
|     6 | `AIRBORNE_LATCHED`       |
|     7 | `PAIR_MODE_SELECTED`     |
|     8 | `GS_LINK_VALID`          |
|     9 | `GS_POSITION_ACTIVE`     |
|    10 | `SINGLE_LANDING_ACTIVE`  |
|    11 | `PAIR_REATTACH_LATCHED`  |
|    12 | `LINK_TEST_ACTIVE`       |
| 13–15 | reserved，必须为 0       |

`INBOUND_PERIODIC_FRESH` 表示发送端最近能收到对端有效帧，用于在 arm 前确认
双向链路，而不是声明本端已经收到 ACK。`PAIR_ACTIVE` 中主机按
`SENSOR_STATE`、从机按 `ACTUATOR_COMMAND` 的 10 ms 窗口判断；发现/预 arm/
恢复阶段按最近 10 ms 内收到有效心跳判断。

`AIRBORNE_LATCHED` 在当前本地 arm epoch 建立后（单体本地 arm 或协同
`PAIR_ARM`），第一次出现 active controller 的归一化总距命令 `collective > 0`
时置 1，并保持到 landing 结束条件触发且 disarm 后清 0。该锁存表示“已经给出
起飞总距”，不声称 PX4 `land_detected` 已证明物理离地。空中拼接时，两机应在
进入 docking 前已经通过各自的单体 arm epoch 将该位置 1。

`PAIR_REATTACH_LATCHED` 在双机已经 armed/active 后的地面或空中组合控制、以及
空中拼接接管过程中发生意外 airlink 中断后置 1。v1 暂时不允许意外分离后的重新
拼接；该锁存持续到两端都重新上电，不能因链路、心跳、会话恢复、触地或 disarm
自动消除。尚未完成组合 arm 的预控制链路抖动不设置该锁存。

`HEARTBEAT` 在没有 500 Hz 周期数据流的发现、预 arm、恢复和降级阶段双向发送，
固定 500 Hz，与控制周期同相或错相调度均可。进入稳定 `PAIR_ACTIVE` 后，500 Hz 的 `SENSOR_STATE` 和
`ACTUATOR_COMMAND` 已足以证明各自方向存活，可以停止显式心跳。任一方向收到
通过版本、角色、会话、长度、CRC 和 EOF 检查的有效 peer 帧，都刷新该方向的
`LINK_ACTIVITY`；这不替代 10 ms 的传感器/控制数据新鲜度检查。

在预 arm 和恢复阶段，有效心跳超过 100,000 µs 视为链路失效。在
`PAIR_ACTIVE` 阶段，即使其他事件帧仍刷新 `LINK_ACTIVITY`，只要两个 500 Hz
方向中的任一成功方向数据流超过 10,000 µs，控制流也必须撤销。

### 7.2 可靠事件公共 payload

`CONTROL_ENABLE`、`CONTROL_RELEASE`、`EMERGENCY_WITHDRAW`、`PAIR_ARM` 和
`PAIR_DISARM` 均使用 12 B：


| 偏移 | 长度 | 字段                  | 类型     | 说明                             |
| ---: | ---: | --------------------- | -------- | -------------------------------- |
|    0 |    4 | `target_peer_session` | `uint32` | 必须等于对端当前启动会话         |
|    4 |    4 | `event_id`            | `uint32` | 发起端在该会话内唯一的非零事务号 |
|    8 |    1 | `reason_code`         | `uint8`  | 原因，见下表                     |
|    9 |    1 | `event_flags`         | `uint8`  | 必须为 0                         |
|   10 |    2 | `reserved`            | `uint16` | 必须为 0                         |

每条可靠事件完整帧为 `16 + 12 + 4 + 2 = 34 B`。

事件必须同时匹配帧头的 `source_session`、payload 的 `target_peer_session` 和当前
已发现的对端会话。会话不匹配时不得执行状态变化。

`reason_code`：


|      值 | 名称                 |
| ------: | -------------------- |
|       0 | `NONE`               |
|       1 | `OPERATOR_REQUEST`   |
|       2 | `PEER_STREAM_STALE`  |
|       3 | `SENSOR_STALE`       |
|       4 | `CONTROL_STALE`      |
|       5 | `HEARTBEAT_TIMEOUT`  |
|       6 | `LOCAL_RC_OR_SAFETY` |
|       7 | `SESSION_CHANGED`    |
|       8 | `INVALID_COMMAND`    |
|       9 | `EMERGENCY_TRIGGER`  |
|      10 | `OTHER`              |
| 11–255 | reserved             |

`CONTROL_ENABLE.reason_code` 使用 `NONE`。`CONTROL_RELEASE` 和
`EMERGENCY_WITHDRAW`、`PAIR_ARM` 和 `PAIR_DISARM` 使用实际原因；没有更具体
原因时用 `OTHER`。

### 7.3 `LINK_HANDSHAKE`

心跳只用于发现对端和证明基本字节链路。发现成功后，必须由主机发起三段式握手：
主机发送 `REQUEST`，从机返回 `REPORT`，主机验证后发送 `CONFIRM`，从机成功 ACK
后才允许两端进入 `AIRLINK_READY`。

payload 固定 24 B，完整帧 46 B：

| 偏移 | 长度 | 字段 | 类型 | 说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | `target_peer_session` | `uint32` | 本次握手针对的对端会话 |
| 4 | 4 | `transaction_id` | `uint32` | 主机生成的本次握手非零事务号 |
| 8 | 2 | `protocol_version` | `uint16` | v1 固定为 1 |
| 10 | 1 | `phase` | `uint8` | `REQUEST=1`、`REPORT=2`、`CONFIRM=3` |
| 11 | 1 | `reserved` | `uint8` | 必须为 0 |
| 12 | 4 | `capabilities` | `uint32` | 支持和启用的能力位 |
| 16 | 4 | `state_flags` | `uint32` | 握手时的安全与控制状态 |
| 20 | 4 | `config_crc32c` | `uint32` | 规范化控制器/安全配置摘要 |

`transaction_id` 在发送端本次启动会话内必须与所有可靠事件 `event_id` 共用一个
非零唯一编号空间，避免相同 ACK 键对应不同事务。

`capabilities` 至少定义：

| 位 | 名称 |
| ---: | --- |
| 0 | `C_SINGLE` |
| 1 | `C_PAIR` |
| 2 | `PAIR_ARM` |
| 3 | `GS_POSITION` |
| 4 | `SINGLE_LANDING` |
| 5 | `LINK_TEST` |
| 6 | `SENSOR_STATE_V1` |
| 7 | `OUTPUT_2M3S` |
| 8–31 | reserved，必须为 0 |

`state_flags` 至少定义：

| 位 | 名称 |
| ---: | --- |
| 0 | `LOCAL_READY` |
| 1 | `ARMED` |
| 2 | `AIRBORNE_LATCHED` |
| 3 | `PAIR_MODE_SELECTED` |
| 4 | `PAIR_REATTACH_LATCHED` |
| 5 | `GS_LINK_VALID` |
| 6 | `LOCAL_RC_VALID` |
| 7 | `SAFETY_GATE_ALLOW` |
| 8 | `LINK_TEST_ACTIVE` |
| 9 | `C_PAIR_ACTIVE` |
| 10–31 | reserved，必须为 0 |

握手规则：

- 两端必须先收到对端有效心跳并锁定会话，主机才发送 `REQUEST`；
- 从机只接受主机发来的、目标会话等于本地会话的 `REQUEST`，并回复独立
  `REPORT`；
- 主机验证协议版本、主从角色、必需 capability、配置摘要和从机状态；任一不满足
  时不得发送 `CONFIRM`；
- `REPORT` 必须置 `LOCAL_READY=1`，保留位全 0；armed/airborne 状态可以参与
  后续组合门控，但不单独禁止链路握手；
- `CONFIRM` 设置 `ACK_REQ`，并复用同一 `transaction_id`；
- 从机验证 `CONFIRM` 后回复成功 ACK，双方才进入 `AIRLINK_READY`；
- 会话变化、重启或链路重新建立后必须重新握手；
- 握手不改变 arm、控制器所有权或飞行模式。

主机等待 `REPORT`、从机等待 `CONFIRM` 的交换超时均为 5 ms，各最多重发 3 次；
`CONFIRM` 另使用普通可靠 ACK。相同 `(transaction_id, phase)` 重发只重发原结果，
不重复初始化状态。

`REQUEST` 参考帧（`target_peer_session=0x10203040`、
`transaction_id=0x11223344`、capability `0x000000FF`、state `0x000000C9`、
`config_crc32c=0x89ABCDEF`）：

```text
A5 5A 01 02 00 09 01 02 88 77 66 55 01 01 18 00
40 30 20 10 44 33 22 11 01 00 01 00 FF 00 00 00
C9 00 00 00 EF CD AB 89 E8 CD 6C 4D 5A A5
```

完整帧 46 B，CRC-32C 为 `0x4D6CCDE8`。

### 7.4 `PAIR_ARM` 与 `PAIR_DISARM`

两条消息都使用第 7.2 节的 12 B payload。`event_id` 同时作为本次 arm 事务的
`arm_epoch_id`；同一事务的所有重发必须复用该值。

只有主机可以向从机发送：

- `PAIR_ARM`：显式请求从机执行本机 arming checks 和 arm；
- `PAIR_DISARM`：显式请求从机 disarm；接收端先执行本地 disarm，再回 ACK。

`PAIR_ARM` 只有在以下条件全部成立时才允许被从机接受：

- 主机已在本会话完成 fresh `CONTROL_ENABLE` ACK；
- 本次事务会话、角色、长度、保留字段和 ACK 超时均合法；
- 两机 `AIRBORNE_LATCHED=0`，且 `PAIR_REATTACH_LATCHED=0`；
- 从机本机 RC、Commander、failsafe、执行器授权和配置一致性检查全部允许；
- 电气连接的 `AIRLINK_READY` 在 arm 事务期间持续有效。

从机只有在真正进入 PX4 armed 状态后才返回 `SUCCESS` ACK；仅完成参数检查时
不得返回成功。ACK 成功只表示从机已 arm，执行器输出仍由组合控制 gate 保持禁止，
直到从机收到第一条合法 `ACTUATOR_COMMAND`。

主机收到从机成功 ACK、并确认链路仍为 `AIRLINK_READY` 后才执行本机正式 arm。
ACK 超时、从机拒绝、链路失效、主机本地 arm 失败或空中状态不一致时，主机必须
保持/回到 disarm 并发送 `PAIR_DISARM`。从机在 arm 后最多等待
`PAIR_ARM_COMMIT_TIMEOUT=100 ms` 接收第一条合法组合控制帧；超时后必须自行
disarm。等待时间 `<=100,000 µs` 时仍可提交，`>100,000 µs` 必须 disarm；该窗口
内从机执行器 gate 保持关闭。主机从本机 arm 成功到发出第一帧 `C_PAIR` 命令也
受同一 100 ms 提交窗口约束，超时则本地 disarm。

`PAIR_DISARM` 是幂等请求。即使没有收到该消息，本地链路失效、安全门失败或
landing 终止条件也必须直接 disarm。重复 `PAIR_ARM` 不得产生新的 arm epoch；
仅当从机仍在同一 epoch 的 armed 状态时才可返回缓存成功，否则返回
`INVALID_STATE`，不能因重发旧事务而重新 arm。重复 `PAIR_DISARM` 在已经 disarm
时返回成功且不重复执行。

`PAIR_ARM` 参考帧（`target_peer_session=0x10203040`、
`event_id=0xA1B2C3D4`、`reason_code=OPERATOR_REQUEST`）：

```text
A5 5A 01 02 01 07 01 02 88 77 66 55 00 01 0C 00
40 30 20 10 D4 C3 B2 A1 01 00 00 00 BF 4B F8 F5
5A A5
```

该帧 header 的 `source_session=0x55667788`、`flags=ACK_REQ`、
`payload_len=12`；完整帧 34 B，CRC-32C 为 `0xF5F84BBF`。

### 7.5 `LINK_TEST_CONTROL` 与 `LINK_TEST_DATA`

链路测试只在主机本地触发，要求两机已 `AIRLINK_READY`、均 disarmed 且不处于
`PAIR_ARM_PENDING`/`PAIR_ACTIVE`。本地命令：

```text
airlink test --duration-ms 5000 --rate-hz 500 --payload-bytes 256
```

三个参数省略时使用上述默认值。该命令通过 QGC 的 MAVLink Console/NSH 调用，
且只允许在配置为 master 的端点执行；测试结束后在主机日志和控制台输出机器可
解析的一行结果。参数必须在命令解析阶段按协议范围校验；例如 `100000 ms`
必须直接拒绝，不得截断成 `uint16_t` 后以另一个时长入队。

`LINK_TEST_CONTROL` payload 固定 16 B，完整帧 38 B：

| 偏移 | 长度 | 字段 | 类型 | 说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | `target_peer_session` | `uint32` | 对端启动会话 |
| 4 | 4 | `test_id` | `uint32` | 非零测试事务号 |
| 8 | 1 | `opcode` | `uint8` | `START=1`、`ABORT=2` |
| 9 | 1 | `echo_mode` | `uint8` | v1 只支持 `EXACT=1` |
| 10 | 2 | `payload_len` | `uint16` | 测试数据 payload，32–512 B |
| 12 | 2 | `rate_hz` | `uint16` | 1–500 Hz |
| 14 | 2 | `duration_ms` | `uint16` | 1000–10000 ms |

`test_id` 同时作为 START/ABORT 的 ACK 事务 ID，并与可靠事件/握手事务共用同一
非零唯一编号空间；这两条控制消息的 ACK 去重键为
`(source_session, test_id, opcode)`，ACK 的 `detail` 回带对应 opcode。

`START` 设置 `ACK_REQ`；从机验证能力和安全状态后回复 ACK 并进入 echo 模式。
测试在 `duration_ms` 到期后自动结束；需要提前停止时，主机发送 `ABORT`，ABORT
同样需要 ACK。测试期间不得执行 arm、`CONTROL_ENABLE` 或 `PAIR_ARM`；如果收到
这些请求，优先发送 ABORT、终止测试并保持 disarm。若 ABORT 丢失或 echo 流中断，
从机在 `duration_ms + 100 ms` 后强制退出测试，主机也在同一上限后结束统计并
恢复 500 Hz 心跳。

START/ABORT 的成功 ACK 往返完成前仍使用 500 Hz 心跳。进入
`LINK_TEST_ACTIVE` 后普通心跳暂停，存活和统计由 `LINK_TEST_DATA` 负责；测试
不参与飞行控制 10 ms 新鲜度判断。

`LINK_TEST_DATA` payload 长度等于协商的 `payload_len`：

| 偏移 | 长度 | 字段 | 类型 | 说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | `test_id` | `uint32` | 必须匹配 START |
| 4 | 4 | `test_sequence` | `uint32` | 从 0 递增 |
| 8 | 4 | `host_send_time_us` | `uint32` | 主机单调时间低 32 位 |
| 12 | 2 | `pattern_len` | `uint16` | `payload_len - 16` |
| 14 | 2 | `pattern_seed` | `uint16` | 可复现填充种子 |
| 16 | N | `pattern` | `uint8[N]` | 确定性伪随机字节 |

主机发送 `LINK_TEST_DATA`；从机逐字节回显同一 payload，响应帧保持同一
`test_id`/`test_sequence` 并设置 `RESPONSE=1`。主机用本机单调时钟计算往返延迟，
不比较两端时钟。测试数据不 ACK、不重传，否则会掩盖真实丢包和延迟；测试结束至少
报告：

- 发送帧数、唯一回显数、丢包/重复/乱序数；
- CRC、长度、会话错误和重同步次数；
- RTT 的最小值、均值、p95、p99、最大值和抖动；
- 前向 offered 吞吐、回显 goodput 和双向合计 goodput；
- 最大接收间隔和测试期间链路中断次数。

测试帧通过合法长度和 CRC 后计入吞吐；任何帧都不得被解释为飞行控制命令。
主机把结果发布到本地 `airlink_test_result` 状态，并通过 QGC/MAVLink
`STATUSTEXT` 输出一行 `AIRLINK_TEST_JSON=...`；结果只描述本次测试，不写入
控制器状态。

默认 `256 B × 500 Hz` 的单方向占用
`(16 + 256 + 4 + 2) × 500 = 139.0 kB/s`，双向回显合计 278.0 kB/s；允许的
最大 `512 B × 500 Hz` 为单方向 267.0 kB/s、双向合计 534.0 kB/s。测试仅在
disarmed 状态运行；最大配置每个方向约占 6 Mbaud、8N1 链路容量的 44.5%，必须
限制持续时间并同时观察 CPU 调度和串口驱动丢字节计数。

### 7.6 `ACK`

payload 16 B，完整帧 38 B：


| 偏移 | 长度 | 字段                  | 类型     | 说明                    |
| ---: | ---: | --------------------- | -------- | ----------------------- |
|    0 |    4 | `target_peer_session` | `uint32` | 发起事件的主机/从机会话 |
|    4 |    4 | `event_id`            | `uint32` | 被确认事件的非零事务号  |
|    8 |    1 | `acked_message_id`    | `uint8`  | 被确认事件的 ID         |
|    9 |    1 | `result`              | `uint8`  | 结果                    |
|   10 |    1 | `detail`              | `uint8`  | 实现细节，0 表示无      |
|   11 |    1 | `reserved0`           | `uint8`  | 必须为 0                |
|   12 |    4 | `reserved1`           | `uint32` | 必须为 0                |

ACK 的 `source_session` 是响应方自己的启动会话；
`target_peer_session` 是原事件发送方的启动会话。ACK 帧头设置 `RESPONSE=1`，
不设置 `ACK_REQ`，即使重复事件只重新回复 ACK，也不重复执行事件。

`result`：


|      值 | 名称               | 含义                          |
| ------: | ------------------ | ----------------------------- |
|       0 | `SUCCESS`          | 事件已接受并执行/进入对应状态 |
|       1 | `UNSUPPORTED`      | 消息或参数不支持              |
|       2 | `BAD_LENGTH`       | payload 长度错误              |
|       3 | `SESSION_MISMATCH` | 会话未知、过期或不匹配        |
|       4 | `INVALID_STATE`    | 当前状态不允许该事件          |
|       5 | `REJECTED_SAFETY`  | 本机 RC/安全门控拒绝          |
|       6 | `BUSY`             | 正在处理其他可靠事件          |
|       7 | `DUPLICATE`        | 重复事件；可返回缓存结果      |
|       8 | `BAD_ARGUMENT`     | 事件字段非法                  |
|       9 | `INTERNAL_ERROR`   | 本地处理失败                  |
| 10–255 | reserved           | 保留                          |

`detail` 的值不改变 `result` 语义。重复事件应在重试窗口内返回第一次处理的结果；
如果保留值 `DUPLICATE`，仍不得再次执行状态变化。

可靠事件默认 ACK 超时为 5 ms，最多重发 3 次，即最多 4 次发送。`PAIR_ARM` 因为
要等待 PX4 真正 armed，单独使用 100 ms ACK 超时，最多重发 1 次；每次重发仍复用
原 `event_id`。`PAIR_DISARM` 的本地执行不等待 ACK，远端即使没有收到也必须在
本地失效路径中 disarm。

接收端按 `(source_role, source_session, event_id)` 去重，至少保留 60 s 或最近
64 个事务，取较长者。ACK 丢失后的重复 `CONTROL_ENABLE` 只能重新回复原来的结果，
不能创建第二份授权。

## 8. 双控制器与协同飞行生命周期

### 8.1 控制器归属

两架飞机都保留单体控制器 `C_SINGLE` 和协同控制器配置 `C_PAIR`。实际执行规则：

- airlink 未激活时，每架飞机只能由本机 `C_SINGLE` 产生执行器命令；
- airlink 激活后，主机执行 `C_PAIR`，同时生成两架飞机的五路命令；
- 从机执行 `C_FOLLOWER`：使用主机下发的五路命令，不自行运行第二套协同闭环；
- 从机上的 `C_PAIR` 副本保持加载、配置校验和待命，但不抢输出；
- 任何情况下，本机 PX4 的 RC、Commander、failsafe 和执行器授权均可否决控制器
  输出。

“链路建立”与“允许组合控制”必须分开：

| 状态 | 含义 |
| --- | --- |
| `AIRLINK_DOWN` | 未发现对端，或最近有效 peer 帧超过监督窗口 |
| `AIRLINK_DISCOVERED` | 已收到相反角色、版本合法的对端心跳并锁定会话 |
| `LINK_HANDSHAKE_PENDING` | 主机已发起 request/report/confirm 事务 |
| `AIRLINK_READY` | `LINK_HANDSHAKE` 成功；协议、状态和配置已确认；双向帧新鲜 |
| `PAIR_ENABLE_PENDING` | 已发起新的 `CONTROL_ENABLE` 事务 |
| `PAIR_CONTROL_READY` | 从机已成功 ACK，等待第一条合法控制帧 |
| `PAIR_ARM_PENDING` | 已发送 `PAIR_ARM`，主机尚未 arm，等待从机成功 ACK |
| `PAIR_ARMED_GROUND` | 两机已 armed，组合总距为 0，尚未出现起飞总距 |
| `PAIR_ACTIVE` | 从机已接收合法控制帧；传感器与控制双向 500 Hz |
| `PAIR_LOST` | 激活后任一关键方向失效；立即执行本地回退 |

`AIRLINK_READY` 本身不授予执行器控制权。尤其不能把“检测到字节流”或“收到一个
心跳”直接解释为切换到 `C_PAIR`。

### 8.2 角色配置、发现和握手

主从角色由每架飞机本地的 QGC 可见参数确定：

| `AIRLINK_ROLE` | 角色 | `source_role` |
| ---: | --- | ---: |
| 0 | disabled | 不发送 AIRLINK |
| 1 | master | `01` |
| 2 | slave | `02` |

QGC 通过标准 `PARAM_SET` 设置 `AIRLINK_ROLE`。同时提供本地命令包装：

```text
airlink role master
airlink role slave
airlink role off
```

该命令只写本机参数，不属于 AIRLINK 线上消息，因为角色必须在解析对端帧之前
确定。参数类型 `int32`、默认 0，并需持久化；只允许在 disarmed 且
`PAIR_ACTIVE`/`PAIR_ARM_PENDING` 均不存在、`LINK_TEST_ACTIVE=0` 时修改。修改后
清除旧会话并重新开始发现；两机配置成同一角色时不得握手。
任何帧的 `source_role`/`dest_role` 与本地 `AIRLINK_ROLE` 和固定一对一路由不一致
时，必须在发往状态机前丢弃。

发现与握手流程：

1. 两端按本地角色从上电开始持续尝试建立 airlink，不等待第一次 arm；
2. 未进入 `PAIR_ACTIVE` 时，以 500 Hz 双向发送 `HEARTBEAT`；
3. 收到对端角色相反、版本和会话合法的有效心跳后进入
   `AIRLINK_DISCOVERED`，但不授予控制权；
4. 主机使用新的 `transaction_id` 发送 `LINK_HANDSHAKE.REQUEST`；
5. 从机验证角色和会话后发送包含协议、能力、状态和配置摘要的 `REPORT`；
6. 主机验证协议版本、主从角色、必需 capability、配置摘要和从机状态；正常时
   发送 `CONFIRM`；
7. 从机成功 ACK `CONFIRM` 后，双方进入 `AIRLINK_READY`；
8. 只有 `AIRLINK_READY` 后的组合流程才允许发起 `CONTROL_ENABLE`。

`AIRLINK_READY` 后继续保持 500 Hz 心跳，直到组合控制已经激活且两个方向的
500 Hz 数据流均连续两帧有效。此时停发显式心跳并进入数据交换；任一侧数据流
失效后，该侧立即恢复 500 Hz `HEARTBEAT` 并按链路失效处理。

当组合模式准备 arm 时，主机必须重新完成一次带新 `event_id` 的
`CONTROL_ENABLE` ACK 往返。主机同时要求从机有效心跳、从机
`INBOUND_PERIODIC_FRESH=1`、`LOCAL_RC_VALID=1`、`SAFETY_GATE_ALLOW=1` 且两端
`PAIR_REATTACH_LATCHED=0`。

`CONTROL_ENABLE` ACK 只表示从机接受后续接管；它不启动 `C_PAIR`、不发送 actuator
命令，也不授予 arm。地面组合尚未 arm 时，先完成第 7.4 节的 `PAIR_ARM`；空中
拼接时两机必须已独立 `ARMED=1`。只有 arm 条件成立后，主机才在下一个 500 Hz
控制节拍启动 `C_PAIR` 并发送对应 `ACTUATOR_COMMAND`。

从机收到第一条合法、会话匹配且到达新鲜的控制帧后切换到 `C_FOLLOWER` 并进入
`PAIR_ACTIVE`；主机只有在 10 ms 内观察到从机 `AIRLINK_CONTROL_ACTIVE=1` 才
确认双端均已接管，否则回退。

这是控制权切换的两阶段过程：这里的 ACK 指 `CONTROL_ENABLE` ACK，只表示从机
接受接管，不表示从机已经应用命令；实际执行命令的第一帧完成 follower 切换。
链路时延使两端的切换不可能完全同时，但主机和从机必须在同一 500 Hz 节拍边界
执行各自动作，28 B 控制帧的串口线速传输时间约 47 µs，不能在该窗口内混合两套
控制器输出。

从机在 `PAIR_CONTROL_READY` 中若 10 ms 内既没有收到 `PAIR_ARM` 也没有收到第一条
合法控制帧，必须回到本机控制。`PAIR_ARM` 被接受后，等待首帧的窗口扩展为
100 ms，并按第 7.4 节保持执行器 gate 关闭。`CONTROL_ENABLE` 不是永久授权。

### 8.3 地面组合后的 arm 与怠速

地面组合上电时：

1. 两端自动发现并进入 `AIRLINK_READY`，但执行器仍由 disarm 状态禁止输出；
2. 飞手在主机选择组合飞行模式，主机置
   `HEARTBEAT.health_flags.PAIR_MODE_SELECTED=1`；
3. 主机执行第 8.2 节的 fresh `CONTROL_ENABLE` 往返；
4. 只有 airlink 双向有效、`PAIR_REATTACH_LATCHED=0`、两机都在地面、本机
   RC/安全门控允许时，主机发送带新 `event_id` 的 `PAIR_ARM`；
5. 从机重新执行本机 RC、
   Commander、failsafe 和配置检查，本机真正进入 PX4 armed 状态后才回
   `SUCCESS` ACK，但在收到第一条组合控制帧前保持执行器 gate 关闭；
6. 主机收到成功 ACK 且链路仍 Ready 后执行本机正式 arm；任一步骤失败时主机
   保持/回到 disarm 并发送 `PAIR_DISARM`；
7. 两机均确认 armed 后进入 `PAIR_ARMED_GROUND`。主机在下一个 500 Hz 节拍运行
   `C_PAIR` 的零总距怠速模式并发送对应命令，从机切换 `C_FOLLOWER`；
8. 从机在 arm 后 100 ms 内没有收到第一条合法组合控制帧时自行 disarm；该窗口
   内执行器 gate 始终关闭；
9. 两机开始 500 Hz 交换 `SENSOR_STATE` 与 `ACTUATOR_COMMAND` 后，显式心跳可以
   停止；任一有效 peer 帧刷新 `LINK_ACTIVITY`，但控制/传感器新鲜度仍按 10 ms
   独立检查；
10. 本次 `PAIR_ARM` 完成后，主机第一次令组合控制器的归一化总距命令
    `collective > 0` 时置 `AIRBORNE_LATCHED=1`；该锁存保持到触地 disarm。

“怠速”表示组合控制器已接管、总距命令为 0，舵机保持组合控制器的地面稳定命令。
它不使用 ESC 的启动 PWM，也不把低油门电机转动当成起飞。`AIRBORNE_LATCHED`
依据起飞总距指令锁存，而不是 PX4 `land_detected`；这是当前 v1 明确接受的判据。

从发送 `PAIR_ARM` 到两机 armed 且第一帧 `C_PAIR` 怠速命令生效的过渡期，主从机
输出多路选择器都必须保持电机总距为 0，并把舵机限制在预定义的 disarmed/neutral
安全位置；本机 disarm、RC 和 failsafe 仍可直接覆盖并关闭输出。只有 100 ms
提交窗口内的第一帧合法组合命令可以解除该过渡门控并进入
`PAIR_ARMED_GROUND`。

`PAIR_ARM` 是“主机 RC arm 组合体”的显式双机事务。`CONTROL_ENABLE` 成功但仍未
收到 `PAIR_ARM` 时，两机必须保持 disarm。

地面组合模式 arm 门至少要求：

- 主机选中的是组合飞行模式；
- airlink 双向 ready，且本次 arm 前 fresh `CONTROL_ENABLE` ACK 成功；
- 从机进入 `PAIR_CONTROL_READY`，主机本机安全门允许本地 arm；
- 两机 `AIRBORNE_LATCHED=0`；
- `PAIR_REATTACH_LATCHED=0`；
- 两机本机 RC、安全门、传感器和控制器配置全部满足；
- 两机对实际拼接构型、控制器版本和输出映射使用同一配置。

任一条件不满足时，主机必须拒绝 arm；从机不得由 `CONTROL_ENABLE` 自行推导 arm，
且主机本地 arm 失败时必须显式发送 `PAIR_DISARM`。

### 8.4 地面链路中断

- 未 arm 时 airlink 变为 `AIRLINK_DOWN`：两端保持 disarm；主机在组合模式下收到
  arm 请求必须拒绝；
- 已 arm、`AIRBORNE_LATCHED=0` 时 airlink 中断：主机和从机分别在本地立即执行
  disarm，停止组合控制，禁止保持最后的怠速或控制命令，并置
  `PAIR_REATTACH_LATCHED=1`；
- 地面中断后链路恢复不会自动 arm。必须重新完成 airlink discover、
  `CONTROL_ENABLE`、本机 arming checks 和飞手重新 arm；
- 若中断发生在尚未完成组合 arm 的 `AIRLINK_READY` 阶段，不设置
  `PAIR_REATTACH_LATCHED`；链路恢复后可重新执行握手，但仍不能自动 arm；
- 一旦地面空中组合控制曾进入 `PAIR_ARMED_GROUND` 后意外断链，v1 要求两端重新
  上电后才能再次拼接；仅完成地面 disarm 不能清除 `PAIR_REATTACH_LATCHED`。

“立即 disarm”按本地链路失效检测触发，不能等待对端发来 disarm 命令。

### 8.5 起飞后的组合控制

`PAIR_ACTIVE` 且 `AIRBORNE_LATCHED=1` 时：

- 主机 `C_PAIR` 是唯一协同控制输出源，从机 `C_FOLLOWER` 只执行主机五路命令；
- 控制源服从主机现有 RC/GS-LINK 仲裁和 PX4 安全门；
- 从机不因本机 RC 模式开关变化自行切换协同/单体控制；
- `SENSOR_STATE` 和 `ACTUATOR_COMMAND` 互为存活证据，但控制命令必须在 10 ms
  内到达，从机传感器也必须在 10 ms 内到达主机；
- 主机在 `PAIR_ACTIVE` 中要求从机 `SENSOR_STATE.status_flags` 的
  `AIRLINK_CONTROL_ACTIVE=1`；该位清 0、会话改变或从机本地安全状态不允许时，
  主机按 `PAIR_LOST` 处理；
- 任一有效 peer 帧可刷新 `LINK_ACTIVITY`，但不能把已过期的控制数据重新视为
  有效。

### 8.6 起飞后断开或意外分离

`PAIR_ACTIVE` 中主机或从机任一关键方向失效，均标记 `PAIR_LOST` 并置
`PAIR_REATTACH_LATCHED=1`。如果本机 `AIRBORNE_LATCHED=1`，执行：

1. 在当前 500 Hz 控制节拍内切换为本机 `C_SINGLE`；
2. 丢弃协同控制器积分状态、旧控制帧和待发送命令，不保持最后动作；
3. 若 GS-LINK 有效且单体位置闭环具有有效状态、目标和安全门，则进入
   `SINGLE_GS_POSITION`；
4. 否则进入 `SINGLE_LANDING`，总距设为中立杆量的一半。v1 控制器总距域中
   `collective_neutral=0.5`，因此下降总距初值为
   `collective_descent=0.25`；姿态保持水平，直到 GS-LINK 恢复或 landing
   终止；
5. `SINGLE_LANDING` 中 GS-LINK 恢复有效且位置闭环状态、目标和安全门重新成立时，
   可以退出降落回退并进入 `SINGLE_GS_POSITION`，接收 GS-LINK 控制；它不能因此
   恢复 airlink 协同控制；
6. landing 终止条件为任一：冲击超过阈值、组合体翻转超过阈值，或连续
   `SINGLE_LANDING` 持续超过超时上限；
7. landing 终止条件触发后立即执行本地 disarm；`AIRBORNE_LATCHED` 在 disarm 后
   清 0。

主机和从机各自执行上述判断。不能因为一侧仍能收到零星帧而让另一侧继续相信
组合控制有效；任一方向关键流超过 10 ms 即对应该端的回退。

v1 没有机械锁紧/分离传感器，因此把 `AIRLINK_DOWN` 定义为逻辑分离，即使两机
结构仍可能接触。两端都按本节的单体回退执行；该策略必须通过连接器半插入、
运行中拔线和单侧断线测试验证。

`PAIR_REATTACH_LATCHED=1` 时，空中 airlink 恢复只能进入双向心跳/状态交换，
不得自动执行 `CONTROL_ENABLE` 或重新进入 `C_PAIR`。意外分离后的默认安全路径是
独立降落。v1 暂时不允许意外分离后的重新拼接；即使两机落地、disarm 或 airlink
恢复，也不能执行新的 `PAIR_ARM`。

当前 v1 没有清除 `PAIR_REATTACH_LATCHED` 的消息。该锁存持续到两端重新上电；
仅 disarm、触地、重启其中一端或恢复链路都不能清除它。

冲击、翻转和 landing 超时的阈值是控制器/安全参数，不在 payload 中协商。它们
必须在起飞前由两机加载同一安全配置，并把触发值、原始传感器值和时间写入日志。
若 GS-LINK 恢复使 `SINGLE_LANDING` 退出，landing 超时计时清除；之后再次失去
GS-LINK 并从 `SINGLE_GS_POSITION` 进入 `SINGLE_LANDING` 时重新计时。

### 8.7 空中拼接

空中拼接必须由地面站组织，并满足以下顺序：

1. 两机均处于 `C_SINGLE`，GS-LINK 有效并执行相对位置/姿态引导；
2. 地面站选择明确的 docking 任务，两机保持同一任务 ID 和相互兼容的构型；
3. 物理接近阶段禁止 airlink 的“检测到链路即切换”；主机维护
   `PAIR_MODE_SELECTED=1` 和 `PAIR_REATTACH_LATCHED=0`；
4. 接口电气连接闭合且 airlink 达到 `AIRLINK_READY` 后，v1 即把它作为“已连接”
   的唯一判据；两机仍需报告 `ARMED=1`、`AIRBORNE_LATCHED=1`，主机完成 fresh
   `CONTROL_ENABLE` ACK；
5. 主机在 ACK 后的下一个 500 Hz 节拍切换到 `C_PAIR` 并发送第一帧对应命令；
6. 从机收到第一条合法控制帧后切换到 `C_FOLLOWER`，两机随后按主机 RC 或地面站
   选择的有效输入协同飞行。

“airlink 建立瞬间立刻切换”在实现上定义为：接口电气闭合后先达到
`AIRLINK_READY`，随即自动发起 enable，ACK 成功后由第一条合法控制帧完成接管；
不做额外的长时间保持或机械确认。

空中 enable 失败时，两机保持 `C_SINGLE` 并退出 docking 任务。若已经发生刚性
接触但 airlink 无法工作，两机无法得知连接状态，必须保持 `C_SINGLE`；这是 v1
在没有机械锁紧传感器时接受的未覆盖风险。`AIRLINK_READY` 只能证明电气连接通过
串口正常通信，不能证明结构承载能力；接口半插入、线缆先断和连接器相对运动必须
作为硬件台架专项测试。

### 8.8 控制切换约束

- 所有控制器实例以 500 Hz 运行或保持可立即调用；切换发生在固定控制节拍边界；
- pair→single 和 single→pair 都不得混合两套控制器的输出；允许的传输时延窗口
  只由同一控制帧的两端应用构成；
- 进入 `C_PAIR` 前清空或按已验证规则初始化协同积分器，禁止继承过期单体积分；
- 从机在 `PAIR_ACTIVE` 中只执行与当前主机 `source_session` 匹配的命令；
- `CONTROL_RELEASE`、`EMERGENCY_WITHDRAW` 和本地超时优先于任何切换请求；
- 从机执行器命令只能有一个所有者。不得同时允许本地单体控制器和 airlink 命令
  写同一路执行器；
- 控制器切换日志至少记录模式、时间、会话、触发原因、两机 armed/airborne 状态、
  当前控制器版本和切换前最后有效命令。

### 8.9 安全优先级

安全优先级从高到低为：

1. PX4 本机 RC、Commander、failsafe 和执行器授权；
2. 本机控制超时、心跳超时、会话失效和 airborne 回退；
3. `EMERGENCY_WITHDRAW` 和 `CONTROL_RELEASE`；
4. 正常的 airlink `CONTROL_ENABLE`、`C_PAIR` 与周期命令。

任何较低优先级事件都不能覆盖较高优先级状态。`EMERGENCY_WITHDRAW` 在会话内
具有粘滞性：执行后不会因下一次心跳或普通命令自动解除，只允许重新握手。

## 9. 周期、时序与带宽

协同控制正常激活时：


| 项目               | 参数                                  |
| ------------------ | ------------------------------------- |
| `SENSOR_STATE`     | 500 Hz，标称周期 2 ms，不 ACK、不重传 |
| `ACTUATOR_COMMAND` | 500 Hz，标称周期 2 ms，不 ACK、不重传 |

接收轮询可使用 1 ms 周期以及时处理 ACK 和入站数据；发送侧必须继续按上述
2 ms 周期节流。提高接收轮询频率不得改变周期数据帧的协议发送速率。
| `HEARTBEAT`        | 发现/预 arm/恢复阶段 500 Hz；激活后可停 |
| 传感器到达新鲜度   | 10,000 µs；超过即过期                |
| 控制到达新鲜度     | 10,000 µs；超过立即回 RC             |
| 预控制心跳超时     | 100,000 µs；超时禁止 arm 或撤销控制  |
| `PAIR_ACTIVE` 存活 | 任一有效 peer 帧刷新 activity         |
| 默认可靠事件 ACK   | 5 ms，最多 3 次重发                   |
| `PAIR_ARM` ACK     | 100 ms，最多重发 1 次                 |
| `PAIR_ARM` 提交    | 0–100 ms；超过后从机自行 disarm       |
| 半个帧的接收超时   | 5 ms 无新字节后丢弃并重新同步         |

周期流采用“最新值覆盖”。每个方向只保留一个待发送快照和一个已接收最新快照；
不能因为串口或控制器短暂落后而积压旧命令。500 Hz 是标称调度频率，实际抖动和
丢包由 10 ms 窗口处理，但不得通过重复发送同一传感器帧伪造新鲜度。

带宽按十进制 SI 单位计算：


| 流量               | 计算                | 标称带宽  |
| ------------------ | ------------------- | --------: |
| 传感器             | 64 B × 500 Hz      | 32.0 kB/s |
| 控制               | 28 B × 500 Hz      | 14.0 kB/s |
| 激活态周期流       | 传感器 + 控制       | 46.0 kB/s |
| 双向发现/恢复心跳  | 26 B × 500 Hz × 2  | 26.0 kB/s |

6 Mbaud、8N1 的有效字节率为 `6,000,000 / 10 = 600,000 B/s`。裸周期流量约占
7.67%。在 `PAIR_ACTIVE` 中停止显式心跳时，周期核心仍为 46.0 kB/s；若保留
500 Hz 诊断心跳，则上升至 72.0 kB/s，约占 12.0%。预控制阶段只有 26.0 kB/s
心跳和可靠事件，不运行 500 Hz 双机控制流。全双工链路的两个方向独立计算，不能
把上下行字节合并后假设只有一个发送方向可用。

## 10. 字节流解析

解析器按长度读取，不把串口 `read()` 边界当作帧边界。推荐状态机：

1. 在字节流中搜索 `A5 5A`；
2. 保留候选 SOF，读取完整 16 B 帧头；
3. 检查版本、类别、标志、角色、消息 ID 和 `payload_len <= 512`；
4. 按 `payload_len` 读取 payload、4 B CRC 和 2 B EOF；
5. 先验证 CRC，再验证 EOF；
6. 校验通过后才分发消息、刷新本地接收新鲜度和可能触发 ACK；
7. 任何结构、长度、CRC 或 EOF 错误都丢弃该候选帧，并从候选 SOF 后一个字节
   重新扫描，保证假帧头后的真实 SOF 仍能被找到。

payload 中允许出现任意 `A5 5A` 或 `5A A5`，包括两者跨越 payload 字节边界。
进入 payload 后必须按长度读取，绝不能扫描 SOF。任意拆包必须得到相同结果：
逐字节喂入、一次喂入整帧、以及在所有位置切分的 64 B 帧，都必须解码为同一
`SENSOR_STATE`。

坏 CRC、坏版本、错误角色、错误 EOF 和坏长度帧不得改变控制状态、不得刷新
10 ms 新鲜度、不得回复 ACK。接收端应记录 `crc_errors`、`length_errors`、
`version_errors`、`resyncs`、`buffer_overflows` 和 `partial_timeouts`，但这些
计数器不能替代安全状态机。运行时接收路径最多保留一个最大帧；读取前必须检查
剩余空间并在每次读取后立即解析，禁止先累积多帧再处理。`buffer_overflows`
表示应用层单帧缓冲被越界写入，必须保持为零；网络或 UART 丢字节应体现在
`missing`、`resyncs` 或其他协议错误中。

## 11. 验证与验收

### 11.1 字段量化和边界

协议实现至少验证以下结果：


| 测试              | 输入                             | 期望原始值/状态                       |
| ----------------- | -------------------------------- | ------------------------------------- |
| 加速度负端        | `-256.0 m/s²`                   | `-32768`，bit 0 有效，bit 11 清 0     |
| 加速度零点        | `0 m/s²`                        | `0`                                   |
| 加速度正端        | `255.9921875 m/s²`              | `32767`，bit 0 有效，bit 11 清 0      |
| 加速度负饱和      | 小于`-256.0 m/s²` 的有限值      | `-32768`，bit 11 置 1                 |
| 加速度正饱和      | 大于`255.9921875 m/s²` 的有限值 | `32767`，bit 11 置 1                  |
| 陀螺仪/估计负端   | `-40.96 rad/s`                   | `-32768`                              |
| 陀螺仪/估计正端   | `40.95875 rad/s`                 | `32767`                               |
| 陀螺仪/估计负饱和 | 小于`-40.96 rad/s`               | `-32768`，对应裁剪位                  |
| 陀螺仪/估计正饱和 | 大于`40.95875 rad/s`             | `32767`，对应裁剪位                   |
| 四元数零点        | `(0,0,0,0)` 或无效源             | 四字段全 0，bit 2 清 0                |
| 四元数端点        | `-1`、`1`                        | 发送`-32767`、`32767`；解码 `-1`、`1` |
| ESC 端点          | `-32767`、`0`、`32767 rpm`       | 原样有符号 int16                      |
| ESC 越界          | 小于`-32767` 或大于 `32767 rpm`  | 0，清对应有效位；不得伪造端点         |
| 电压端点          | `0`、`65535 mV`                  | `0x0000`、`0xFFFF`                    |
| 电压越界          | 负值、NaN、Inf 或`>65.535 V`     | `0`，清 bit 6                         |
| 时间端点          | `0`、`0xFFFFFFFF`                | 保持低 32 位模值                      |

BMI088 当前配置的边界测试：

```text
+24 g = +235.3596 m/s² -> round(235.3596 × 128) = +30126
-24 g = -235.3596 m/s² -> round(-235.3596 × 128) = -30126
+2000 °/s = +34.906585 rad/s -> round(34.906585 × 800) = +27925
-2000 °/s = -34.906585 rad/s -> round(-34.906585 × 800) = -27925
```

四个边界都必须落在 int16 可表达范围内，有限源值解码后仍通过有效位，且不得
误置 `ACCEL_CLIPPED`、`GYRO_CLIPPED` 或 `ANGULAR_VELOCITY_CLIPPED`。这些值不是
协议 int16 的端点，不能为了测试裁剪位而把它们当作超范围输入。

### 11.2 四元数测试

- 随机单位四元数逐分量量化后，原始解码分量误差 `<= 0.5/32767`；
- 量化后范数与 1 的偏差 `<= 1/32767`；
- `q` 和 `-q` 分别编码、解码后必须被姿态比较器判为同一姿态；
- 原始符号不同不改变有效位，也不造成控制回路跳变；
- `[0.5,0,0,0]`、含 NaN 的分量、含 `|q_i|>1` 的分量以及范数偏离超过 0.01 的
  四元数必须写全零并清除 `ATTITUDE_VALID`；
- 接收端对 `ATTITUDE_VALID=1` 但解码范数偏差超过 0.0101 的帧必须拒绝姿态
  字段，不能归一化一个已标记无效的源值。

### 11.3 可选传感器和电压

- 上桨有效、下桨缺失：保留 `esc_rpm[0]`，`esc_rpm[1]=0`，bit 4 为 1、bit 5
  为 0，其他有效位不受影响；
- 下桨有效、上桨缺失：对称处理，不能交换两路或清掉另一路；
- 电压 NaN、负值和 `>65.535 V`：原始字段为 0、`BATTERY_VALID=0`，不得饱和成
  `0xFFFF` 的有效测量；
- 检查非有限 IMU 值清对应有效位，不从其他轴补齐或用零作为有效样本。

### 11.4 时间、控制和解析测试

- `0xFFFFFF00 -> 0x00000100` 的模差为 `0x00000200`；
- `0xFFFFFFFF -> 0x00000000` 的模差为 1；
- 到达年龄等于 10,000 µs 时仍可接受，10,001 µs 时必须判为过期；
- 预控制心跳年龄等于 100,000 µs 时仍可接受，100,001 µs 时必须撤销；
- `SENSOR_STATE.payload_len` 必须为 42，完整帧必须为 64；41 或 43 字节 payload
  必须拒绝；
- `ACTUATOR_COMMAND.payload_len` 必须为 6，完整帧必须为 28；5 或 7 字节 payload
  必须拒绝；
- `PAIR_ARM`、`PAIR_DISARM` 等可靠事件 payload 必须为 12 B；`ACK` 必须为
  16 B；对应完整帧分别必须为 34 B 和 38 B，任何长度不符的消息不得改变
  armed/control 状态；
- `LINK_HANDSHAKE` payload 必须为 24 B、完整帧 46 B；只接受 phase
  REQUEST/REPORT/CONFIRM，未知 phase 或 reserved 非零时拒绝；
- `LINK_HANDSHAKE` 参考 `REQUEST` 帧必须解析为 46 B，CRC-32C 为
  `0x4D6CCDE8`；
- `LINK_TEST_CONTROL` payload 必须为 16 B、完整帧 38 B；
  `LINK_TEST_DATA` payload 必须为 32–512 B、完整帧 54–534 B；
- `PAIR_ARM` 参考帧必须解析为 34 B，CRC-32C 为 `0xF5F84BBF`；
- `HEARTBEAT` 完整帧必须为 26 B；其 payload 为 4 B；
- 电机码 501、舵机码 1001 必须使整个命令帧拒绝，不能执行其他四路；
- 全最小、全中点、全最大和混合命令位流必须与第 6 节向量一致；
- 坏 CRC、假 SOF、payload 内嵌 `A5 5A`/`5A A5`、错误版本、错误角色、错误
  EOF、任意字节拆包和跨帧粘包必须按第 10 节处理；
- 默认可靠事件 ACK 丢失后最多 4 次发送；`PAIR_ARM` 使用 100 ms 超时且最多重发
  1 次；重复事件不重复执行，重复 `PAIR_ARM` 不得产生第二个 arm epoch；
- `CONTROL_ENABLE` 成功后，超时、释放、紧急撤销和会话恢复测试都必须禁止自动
  重新接管。

### 11.5 协同飞行生命周期

- 地面组合、`AIRLINK_DOWN` 时下达组合模式 arm，要求主机拒绝 arm；
- 仅发送成功 `CONTROL_ENABLE`、缺少 `PAIR_ARM` 时必须保持 disarm；从机发送
  `PAIR_ARM`、旧会话或错误角色的事件必须拒绝；
- 只有 `CONTROL_ENABLE` ACK、成功 `PAIR_ARM` ACK 和本机 arm 全部完成才允许
  `PAIR_ARMED_GROUND`；任一 ACK 超时或本机 arm 失败时两端必须 disarm；
- 从机 `PAIR_ARM` 成功后必须报告真实 PX4 armed；收到第一条合法组合控制帧前
  不得开放执行器 gate；
- 从机 arm 后等待首帧的窗口在 `100,000 µs` 边界仍可提交，`100,001 µs` 必须
  disarm；
- 主机本机 arm 后未在 `100,000 µs` 内发出第一帧 `C_PAIR` 命令时，必须本地
  disarm，不能把 master armed 留在等待状态；
- 从机执行 `PAIR_ARM` 并回成功 ACK 后，主机才执行本机 arm；从机 ACK 超时、
  拒绝、链路失效或主机本地 arm 失败时，不得进入 `PAIR_ARMED_GROUND`，主机和
  已 arm 的从机都必须 disarm；
- `PAIR_DISARM` 和本地失效路径必须立即关闭执行器 gate；丢失
  `PAIR_DISARM` ACK 不得延迟本地 disarm，重复消息必须幂等；
- 地面组合已 arm、`AIRBORNE_LATCHED=0`，断开 airlink 后要求两机各自自动 disarm，
  不保持最后命令，并置 `PAIR_REATTACH_LATCHED=1`；
- `PAIR_ARM` 后总距始终为 0 时 `AIRBORNE_LATCHED=0`；首次注入
  `collective>0` 后必须置 1，并保持到 landing 终止和 disarm；
- 地面已 arm 但 `AIRBORNE_LATCHED=1` 的注入场景必须走空中回退，禁止误触发自动
  disarm；
- 空中 `PAIR_ACTIVE` 中断控制方向或传感器方向，要求对应飞机在一个控制节拍内
  进入 `C_SINGLE`，不执行旧协同命令；
- 从机仍发送新鲜传感器但把 `AIRLINK_CONTROL_ACTIVE` 清 0 时，主机必须按
  `PAIR_LOST` 回退，不能因为数据包仍到达而继续协同控制；
- 空中回退时 GS-LINK 与位置闭环有效，要求进入 `SINGLE_GS_POSITION`；GS-LINK
  失效时要求进入 `SINGLE_LANDING`，归一化总距为 0.25，并在 landing 终止后
  disarm；
- `SINGLE_LANDING` 中注入 fresh GS-LINK 位置控制，要求允许返回
  `SINGLE_GS_POSITION`，但 `PAIR_REATTACH_LATCHED` 保持为 1；
- 分别注入冲击、翻转和 landing 超时，每一条件都必须单独触发本地 disarm；
- 仅收到 ACK、事件帧或单向数据包不得维持已经过期的控制/传感器方向；
- `PAIR_ACTIVE` 停止心跳后，连续 500 Hz 传感器/控制帧必须视为链路存活；删除相应
  500 Hz 流后，即使还有其他有效帧，关键控制方向仍必须在 10 ms 后回退；
- 空中拼接必须在两机 `ARMED=1`、`AIRBORNE_LATCHED=1`、
  `CONTROL_ENABLE` ACK 后的第一条合法控制帧才切换，不能仅凭 SOF/心跳切换，
  也不在空中重发 `PAIR_ARM`；
- `CONTROL_ENABLE` ACK 成功但第一帧 `ACTUATOR_COMMAND` 丢失时，主机在 10 ms
  内不得确认 `PAIR_ACTIVE`，两端都必须保持/回到 `C_SINGLE`；
- 只允许一个方向收到心跳或周期帧时，arm 门与 `CONTROL_ENABLE` 都必须拒绝；
- `LINK_HANDSHAKE` 验证失败、配置摘要不一致、角色相同或会话改变时不得发送
  `CONFIRM`，任一端都不得进入 `AIRLINK_READY`；
- 从机成功 ACK `CONFIRM` 前不得发起 `CONTROL_ENABLE`；重复 REQUEST/REPORT/
  CONFIRM 不得产生第二个握手结果；
- 只有会话、角色、CRC、EOF 和双向新鲜度都通过的电气 `AIRLINK_READY` 才可按
  v1 的连接判据继续；裸 SOF、半插入噪声或单向字节流不得进入 `C_PAIR`；
- 意外失去 `PAIR_ACTIVE` 后设置 `PAIR_REATTACH_LATCHED=1`；空中链路恢复、会话
  恢复、心跳恢复、触地、disarm 或单端重启都不得重新进入 `C_PAIR`，只有两端
  都重新上电后才允许新任务；
- 两机控制器版本、拼接构型或输出映射不一致时，必须拒绝 arm 和 `CONTROL_ENABLE`；
- pair→single 与 single→pair 切换测试必须证明只有一个控制源拥有执行器，旧积分器
  和旧命令不可跨模式泄漏。

### 11.6 角色、握手与链路诊断

- QGC 设置 `AIRLINK_ROLE=0/1/2` 后，只有 armed=false 且无组合控制时允许应用；
  armed 期间修改必须拒绝并记录；
- 两机同为 master、同为 slave 或任一 disabled 时不得进入 `AIRLINK_READY`；
- 两端上电后应在 500 Hz 心跳下发现对方；主机 REQUEST、从机 REPORT、主机
  CONFIRM、从机 ACK 四步完成前不得发起 `CONTROL_ENABLE`；
- 在 1 s 窗口内分别统计两端心跳，排除握手/测试流量后应各为约 500 帧；
  `PAIR_ACTIVE` 后仅在两个方向连续两帧数据有效后才停止心跳；
- 从机 REPORT 中协议版本、capability 或 `config_crc32c` 不匹配时，主机不得
  CONFIRM；
- 会话重启、角色修改、链路中断后必须使用新 `transaction_id` 重新握手；
- `PAIR_ACTIVE` 中两个方向各连续两帧 500 Hz 数据有效后停发心跳；任一方向失效
  后立即恢复 500 Hz 心跳并按 10 ms 规则回退；
- 对 `airlink test` 的默认参数 `5000 ms / 500 Hz / 256 B` 执行完整回环，验证
  发送数、唯一回显数、丢包、重复、乱序、CRC 错误、接收缓冲溢出、RTT
  p50/p95/p99/max、抖动、单向 offered 吞吐、回显 goodput、双向合计 goodput、
  最大包间隔和重连次数均有输出；
- `payload_len` 为31/513、`rate_hz` 为0/501、`duration_ms` 为999/10001 时，
  START 必须回复 `BAD_ARGUMENT`，任一端不得进入 `LINK_TEST_ACTIVE`；
- `LINK_TEST_DATA` 不得 ACK 或重传；人为删除指定 sequence 后，统计必须按回显
  缺失增加丢包数，不能因重发把它掩盖；
- 链路测试期间注入 arm、`CONTROL_ENABLE`、`PAIR_ARM` 或角色修改，必须终止测试
  且保持 disarm；
- 从机丢失 `ABORT`、主机丢失回显或会话改变时，测试必须按 duration 上限结束并恢复 500 Hz
  心跳，不能遗留 `LINK_TEST_ACTIVE`；
- 链路测试只产生 RTT，不得把未同步主机/从机时钟的差值报告成单向延迟。

## 12. 完整 64 B 金样例

三个样例使用相同的从机帧头：`source_session=0x10203040`、从机角色 `02`、主机
角色 `01`、`message_id=01`、`frame_type=DATA`。每个样例的 payload 都是 42 B，
完整帧都是 64 B。`crc32c` 字段已按小端放在 payload 后、EOF 前。

### 12.1 全零 payload 金样例

字段：`status_flags=0`、两个时间为 0、六路 IMU/姿态/估计原值全 0、ESC 全 0、
电压 0。该帧用于结构解析和失败安全入口，不代表有效飞行数据。

```text
帧头：
A5 5A 01 01 00 01 02 01 40 30 20 10 00 00 2A 00
payload：
00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00
00 00
CRC-32C / EOF：
11 4E F8 0C 5A A5
```

完整十六进制：

```text
A5 5A 01 01 00 01 02 01 40 30 20 10 00 00 2A 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 11 4E F8 0C 5A A5
```

校验值：`CRC32C=0x0CF84E11`，帧长 64。

### 12.2 标称悬停金样例

字段：

- `status_flags=0x07FF`：bit 0–10 全有效，bit 11–15 全 0；
- `imu_sample_time_us=1,000,000`（`0x000F4240`）；
- `attitude_sample_time_us=1,002,000`（`0x000F4A10`）；
- 加速度 `a_b=[0,0,-1255/128] m/s²`，即约 `-9.8046875 m/s²`；
- 陀螺仪和角速度估计均为 `[0,0,0]`；
- 四元数为单位姿态 `[32767,0,0,0]`；
- `esc_rpm=[6000,6000]`；
- 电压 `14800 mV`，即 `14.8 V`。

```text
帧头：
A5 5A 01 01 00 01 02 01 40 30 20 10 2A 00 2A 00
payload：
FF 07 40 42 0F 00 10 4A 0F 00 00 00 00 00 19 FB
00 00 00 00 00 00 FF 7F 00 00 00 00 00 00 00 00
00 00 00 00 00 00 70 17 70 17 D0 39
CRC-32C / EOF：
EE 8B 82 05 5A A5
```

完整十六进制：

```text
A5 5A 01 01 00 01 02 01 40 30 20 10 2A 00 2A 00
FF 07 40 42 0F 00 10 4A 0F 00 00 00 00 00 19 FB
00 00 00 00 00 00 FF 7F 00 00 00 00 00 00 00 00
00 00 00 00 70 17 70 17 D0 39 EE 8B 82 05 5A A5
```

校验值：`CRC32C=0x05828BEE`，帧长 64。

### 12.3 协议范围边界金样例

字段：

- `status_flags=0x3FFF`：bit 0–10 有效，bit 11、12、13 分别表示三个 int16
  物理量已饱和，保留位 14–15 为 0；
- `imu_sample_time_us=0xFFFFFF00`，`attitude_sample_time_us=0x00000100`；这只是
  两个源的编码值，不能互相相减作为年龄；
- 加速度 `[-32768,0,32767]`；
- 陀螺仪 `[-32768,0,32767]`；
- 姿态四元数 `[32767,0,0,0]`，范数为 1；
- 角速度估计 `[-32768,0,32767]`；
- ESC `[-32767,32767]`；
- 电压 `65535 mV`。

```text
帧头：
A5 5A 01 01 00 01 02 01 40 30 20 10 FF FF 2A 00
payload：
FF 3F 00 FF FF FF 00 01 00 00 00 80 00 00 FF 7F
00 80 00 00 FF 7F FF 7F 00 00 00 00 00 00 00 80
00 00 FF 7F 01 80 FF 7F FF FF
CRC-32C / EOF：
D0 4E 69 E5 5A A5
```

完整十六进制：

```text
A5 5A 01 01 00 01 02 01 40 30 20 10 FF FF 2A 00
FF 3F 00 FF FF FF 00 01 00 00 00 80 00 00 FF 7F
00 80 00 00 FF 7F FF 7F 00 00 00 00 00 00 00 80
00 00 FF 7F 01 80 FF 7F FF FF D0 4E 69 E5 5A A5
```

校验值：`CRC32C=0xE5694ED0`，帧长 64。

## 13. 已确定约束与剩余待定参数

### 13.1 显式双机 arm

v1 使用 `PAIR_ARM`/`PAIR_DISARM` 可靠事件管理协同 arm，不使用
`CONTROL_ENABLE` 隐式 arm。`PAIR_ARM` 的 `SUCCESS` ACK 必须发生在从机 PX4
真正 armed 之后；任一安全检查失败、ACK 超时或主机本地 arm 失败时，两端都必须
保持或回到 disarm。

从机被 `PAIR_ARM` arm 后，直到收到第一条组合控制帧，执行器 gate 必须继续关闭。
`PAIR_DISARM` 和所有本地失效处理必须幂等且不需要等待 ACK。

### 13.2 起飞与 landing 终止判据

`AIRBORNE_LATCHED` 不依赖高度、GPS 或 PX4 `land_detected`。每次本机本地 arm
（单体或 `PAIR_ARM`）建立新的 arm epoch；在该 epoch 中第一次出现 active
controller 的归一化总距命令 `collective > 0` 时即锁存“已起飞”。地面怠速总距
为 0，ESC 启动 PWM 不计入该判据。

这一判据锁存的是“已下达起飞总距”，不是物理离地证明。若电调未起转、总距命令
误发或飞机仍在地面，地面断链后的自动 disarm 将不会触发，而是进入空中单体回退。

`SINGLE_LANDING` 的三个终止条件为：

- 撞击：加速度或其高频量超过统一配置的 impact 阈值；
- 翻转：组合体相对竖直方向的倾角超过统一配置的 flip 阈值；
- 超时：连续处于 `SINGLE_LANDING` 超过统一配置的 landing timeout。

任一条件触发都在本地立即 disarm。固件首轮实现采用以下经验默认值；它们参与
`config_crc32c`，两机必须配置为相同值，并在无桨台架和 HIL 中按机架质量、
冲击噪声、结构承载和预期下降时间重新标定：

| 固件参数 | 默认值 | 单位 | 含义 |
| --- | ---: | --- | --- |
| `AIRLINK_IMPACT` | 35.0 | m/s² | 加速度冲击判定阈值 |
| `AIRLINK_IMP_MS` | 20 | ms | 冲击触发去抖时间 |
| `AIRLINK_FLIP_DEG` | 70.0 | ° | 相对竖直方向的翻转角阈值 |
| `AIRLINK_FLIP_MS` | 150 | ms | 翻转触发去抖时间 |
| `AIRLINK_LAND_MS` | 15000 | ms | 连续 `SINGLE_LANDING` 的强制 disarm 上限 |

### 13.3 电气连接作为唯一连接状态

当前结构没有机械锁紧反馈，v1 明确把接口电气闭合后的
`AIRLINK_READY` 作为“已连接”的唯一协议判据。允许进入 `C_PAIR` 的前提为：

```text
allow_pair_takeover =
    airlink_ready
    and control_enable_acked
    and controller_config_compatible
    and both_required_arm_states_satisfied
    and pair_reattach_latched == 0
```

`AIRLINK_DOWN` 因此也作为逻辑分离。即使两机结构上仍可能接触，两端也执行第 8.6
节的单体回退。该选择只覆盖“串口电气连接能够正常收发”的条件，不能证明结构
强度或连接器完全插入；接口半插入、连接器先断、单侧链路失效和插拔抖动属于硬件
验证项。

### 13.4 空中拼接接管

空中拼接由地面站引导，两机在连接前保持 `C_SINGLE` 和 `ARMED=1`。接口电气闭合
产生 `AIRLINK_READY` 后，主机立即执行 fresh `CONTROL_ENABLE`；成功后由第一帧
合法控制命令切换到 `C_PAIR`/`C_FOLLOWER`。空中拼接不再执行 `PAIR_ARM`，因为
两机在接近前已经 arm。

仍需补充实际进入 docking 流程的相对位置、速度、姿态差阈值和稳定时间，以及
主机 RC 与 GS-LINK 同时有效时的控制源仲裁。这些是引导判据，不是连接状态本身。

### 13.5 GS-LINK 恢复与下降回退

GS-LINK 恢复且位置闭环的状态、目标和安全门重新有效时，允许从
`SINGLE_LANDING` 返回 `SINGLE_GS_POSITION` 并接收 GS-LINK 控制。该恢复只影响
单体控制器，不能清除 `PAIR_REATTACH_LATCHED`，也不能进入 `C_PAIR`。

下降总距暂定为中立杆量的一半。当前控制器总距域为 `[0,1]`、中立值为 0.5，因此：

```text
collective_descent = 0.5 * collective_neutral = 0.25
```

该值必须在落地前通过统一配置写死在两机控制器中。若以后把中立总距改为可标定
配置，下降值必须按其配置重新计算并纳入配置兼容性摘要。

固件当前在 GS-LINK 恢复后的下一个有效控制周期退出 landing；impact/flip 和
landing 超时采用 §13.2 的经验默认值。仍需通过台架/HIL 标定这些阈值，以及下降中
的目标航向、水平速度和姿态误差上限。

### 13.6 心跳与 arm 稳定窗口

心跳固定为 500 Hz，预控制超时暂定 100 ms。仍需确认 arm 前要求连续多少个双向
心跳刷新，以及是否需要比 100 ms 更短的预 arm 链路失效门。无论最终参数如何，
arm 都必须使用 fresh `CONTROL_ENABLE` ACK 和 fresh `PAIR_ARM` ACK，不能只因
最近一次心跳仍在窗口内就通过。

### 13.7 意外分离后不重新拼接

一旦 `PAIR_ARMED_GROUND`/`PAIR_ACTIVE` 因任一方向失效进入 `PAIR_LOST`，置
`PAIR_REATTACH_LATCHED=1`。v1 暂时禁止重新拼接：

- 不执行新的 `CONTROL_ENABLE`；
- 不执行新的 `PAIR_ARM`；
- 触地、disarm、GS-LINK 恢复或 airlink 恢复都不能清除该锁存；
- 只有两端都重新上电后才允许新的会话和新的组合任务。

### 13.8 控制配置一致性

两套控制器同时驻留意味着版本不匹配可能在新旧配置之间造成不可见的状态跳变。
需要补充可靠的能力/配置握手，至少包含：

- 控制器类型、参数摘要和代码/模型校验和；
- 机架 ID、拼接构型、接口方向和输出映射；
- 两电机三舵机布局和 airframe capability；
- `C_SINGLE`/`C_PAIR` 状态机和控制切换版本；
- `PAIR_ARM_COMMIT_TIMEOUT=100 ms`、`collective_descent=0.25`；
- impact、flip、landing timeout 数值以及 GS-LINK 安全参数版本。

在该握手完成前只能建立只读链路，不得 arm，也不得发送可执行的从机命令。主机与
从机能力不兼容时应拒绝 `CONTROL_ENABLE`/`PAIR_ARM` 并给出可记录的
`UNSUPPORTED` 或 `BAD_ARGUMENT` 结果。

`config_crc32c` 必须由固定字段顺序、固定端序、无 padding 的规范化配置字节流
使用与帧相同的 CRC-32C 参数计算，输入至少包含控制器类型、模型/参数摘要、
airframe ID、输出映射、总距域、landing 安全参数和 GS-LINK 能力版本；角色字段和
`source_session` 不参与计算。摘要不一致时不得确认握手。

## 14. 版本和实现边界

v1 接受以下固定约束，任意一项改变都必须提升版本或定义新的显式 capability：

- BMI088 陀螺仪 `±2000 °/s` 和加速度计 `±24 g`；
- 两路电机、三路舵机的执行器拓扑；
- 标准 `1000–2000 µs` 输出及现有控制器域映射；
- 上/下桨 `Motor1/Motor2` 功能号约定；
- 单电池电压源；
- 16 bit IMU/姿态/角速度量化比例 `128`、`800`、`32767`；
- 42 B 传感器 payload、6 B 指令 payload 和固定帧头；对应 64 B 传感器帧、28 B
  控制帧、26 B 心跳帧、34 B 可靠事件帧、38 B ACK/测试控制帧、46 B 握手帧和
  54–534 B 链路测试数据帧；
- `PAIR_ARM`/`PAIR_DISARM` 的可靠事务语义，以及电气 `AIRLINK_READY` 作为唯一
  连接判据；
- QGC `AIRLINK_ROLE` 主从配置、三段式 `LINK_HANDSHAKE` 和 500 Hz 心跳；
- `LINK_TEST_CONTROL`/`LINK_TEST_DATA` 回环诊断与结果字段；
- 归一化总距域 `[0,1]`、`collective_neutral=0.5` 和
  `collective_descent=0.25`；
- 意外 `PAIR_LOST` 后 `PAIR_REATTACH_LATCHED` 保持到两端重新上电。

`PAIR_ARM` 是请求，不承诺远端一定 arm；远端本机 RC、Commander、failsafe 和
执行器授权始终可以拒绝。v1 不承诺 airlink 的序号、心跳或 CRC 可以替代飞控的
RC 接管、失效保护和飞行授权。通信测试通过只证明协议编解码和链路行为，不构成
飞行许可。
