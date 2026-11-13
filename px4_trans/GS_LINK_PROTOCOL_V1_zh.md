# GS-Link v1 地面站串口协议设计

> 2026-11-03：地面组合适配已增加 `DATA_RC` bit2 `PairControl` 和 DATA `0x10–0x13`。定义见固件 `GsLinkDiagnostics.hpp` 及 [地面组合流程](../Docs/GROUND_PAIR_FLIGHT_zh.md)。ESP 保证逐 UDP 包一帧；基础 v1 帧格式和遥测不变。

状态：v1 通信测试已实现；飞行控制动作未启用\
日期：2026-10-20\
物理层：TELEM1 / `SERIAL1` / `/dev/ttyS0`，`4,000,000 8N1`，无硬件流控\
端点：地面站 GCS 与飞控 FC；ESP32 为透明字节桥\
适用范围：单机、单串口、无字符串的二进制字节流

## 1. 设计边界

GS-Link v1 只替换地面站到飞控的传输层，不改变既有 RC 监督状态机：

- 实机 RC 仍是唯一主手动输入和唯一解锁来源；
- 通道 8 的模式槽 4 仍是地面站子模式的物理硬门控；
- RC disarm、RC 丢失、通道 8 离开模式槽 4 或链路超时，仍立即退出；
- 飞控仍只接收四轴归一化虚拟遥控量，不接收位置目标；
- `NN_LQI_OUTPUT_EN`、Commander、failsafe 和执行器授权全部不变；
- 串口 1 由 ESP 自定义协议独占，MAVLink 不再占用该口；
- USB `/dev/ttyACM0` 的 QGC 通道独立保留。

本协议不包含字符串、文本命令、JSON、NUL 结尾字段或变长编码。所有字段均为
定宽整数或 IEEE-754 binary32；名称只出现在文档和调试工具中。

## 2. 基本约定

- 所有多字节整数和浮点字段使用小端序；
- `int16`、`int32`、`uint16`、`uint32`、`uint64` 均为二进制补码/无符号；
- `float32` 为 IEEE-754 binary32，小端序；
- 线上结构不使用编译器 padding，发送端逐字段序列化；
- 包头固定 10 字节，payload 可变长，最大 512 字节；
- 最大完整包 524 字节；v1 不做分片；
- 保留字段发送时置 0，接收时忽略；
- 保留标志位发送时置 0，接收时忽略；
- 已知消息 ID 在 v1 使用固定 payload 长度；长度不符即拒绝；
- 新功能使用新消息 ID；不兼容的布局变化提升协议版本。

## 3. 包结构

### 3.1 包头

| 偏移 | 长度 | 字段 | 说明 |
| ---: | ---: | --- | --- |
| 0 | 1 | `magic0` | 固定 `0xA5` |
| 1 | 1 | `magic1` | 固定 `0x5A` |
| 2 | 1 | `version` | v1 固定 `0x01` |
| 3 | 1 | `packet_class` | `0x01` 数据包，`0x02` 指令包 |
| 4 | 1 | `message_id` | 消息类型编号 |
| 5 | 1 | `flags` | 包标志 |
| 6 | 2 | `sequence` | 发送端递增序号，`uint16` LE |
| 8 | 2 | `payload_len` | payload 字节数，`uint16` LE，0–512 |
| 10 | N | `payload` | 消息负载 |
| 10+N | 2 | `crc16` | CRC-16/CCITT-FALSE，LE |

总长度 = `10 + payload_len + 2`。

### 3.2 包标志

| 位 | 名称 | 说明 |
| ---: | --- | --- |
| 0 | `ACK_REQ` | 发送端要求 ACK；只用于指令类消息 |
| 1 | `RETRY` | 同一序号的重传 |
| 2 | `URGENT` | 紧急消息，例如 abort |
| 3 | `RESPONSE` | 该包是对先前消息的响应 |
| 4–7 | reserved | 置 0 |

数据包通常不设置 `ACK_REQ`。ACK 包的 `RESPONSE` 置 1。

### 3.3 校验

使用 CRC-16/CCITT-FALSE：

- polynomial `0x1021`
- initial value `0xFFFF`
- no input/output reflection
- xorout `0x0000`
- 覆盖范围：从 `version`（偏移 2）到 payload 最后一个字节，即
  `packet[2 .. 9+payload_len]`；
- 不覆盖 magic，也不覆盖 CRC 本身；
- CRC 字段按小端序发送。

CRC 校验不是认证。链路必须是受控局域网；需要抗注入时在协议层另加认证/HMAC，
不得削弱 RC 门控。

## 4. 包类别与消息编号

线上只有两类包：

| `packet_class` | 名称 | 说明 |
| ---: | --- | --- |
| `0x01` | DATA | 连续数据，不要求 ACK |
| `0x02` | COMMAND | 指令、心跳、ACK 和指令类响应 |

### 4.1 数据消息

| `message_id` | 名称 | 方向 | payload 长度 |
| ---: | --- | --- | ---: |
| `0x01` | `DATA_RC` | GCS → FC | 16 |
| `0x02` | `DATA_TELEMETRY` | FC → GCS | 22 + N（N=0–18） |
| `0x03` | `DATA_NAV`（可选，预留） | FC → GCS | 待定 |
| `0x04–0x7F` | 保留数据消息 | 待定 | 待定 |
| `0x80–0xFF` | 保留高速/事件数据 | 待定 | 待定 |

未知数据消息直接丢弃并计数，不发送 ACK。

`DATA_NAV` 预留给位置/速度等导航扩展，不属于 v1 基础遥测。若地面站需要位置闭环，
必须显式启用并定义该包，或从另一条链路取得位置/速度。

### 4.2 指令消息

| `message_id` | 名称 | 方向 | payload 长度 |
| ---: | --- | --- | ---: |
| `0x01` | `CMD_PING` | 双向 | 4 |
| `0x02` | `CMD_HEARTBEAT` | 双向 | 16 |
| `0x03` | `CMD_TAKEOFF` | GCS → FC | 16 |
| `0x04` | `CMD_LAND` | GCS → FC | 12 |
| `0x05` | `CMD_ABORT` | GCS → FC | 12 |
| `0x06–0x7E` | 保留指令 | 待定 | 待定 |
| `0x7F` | `CMD_ACK` | 接收方 → 发送方 | 12 |
| `0x80–0xFF` | 保留指令事件/响应 | 待定 | 待定 |

未知指令 ID 如果包头和 CRC 有效，回复 `UNSUPPORTED`；否则丢弃。

## 5. 数据包

### 5.1 `DATA_RC`：地面站遥控数据

方向 GCS → FC，建议 50 Hz（20 ms）。该包不要求 ACK；丢失由下一帧覆盖。

| payload 偏移 | 长度 | 字段 | 类型 | 说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | `sender_time_ms` | uint32 | 发送端单调毫秒，仅诊断 |
| 4 | 2 | `roll` | int16 | `[-10000, 10000]` |
| 6 | 2 | `pitch` | int16 | `[-10000, 10000]` |
| 8 | 2 | `yaw` | int16 | `[-10000, 10000]` |
| 10 | 2 | `throttle` | int16 | `[-10000, 10000]` |
| 12 | 1 | `flags` | uint8 | 见下 |
| 13 | 1 | `link_quality` | uint8 | 0–100；255 表示未知 |
| 14 | 2 | reserved | uint16 | 置 0 |

轴向与现有控制器一致：

- `roll` 右滚为正；
- `pitch` 低头为正，`LqiManualReference` 内部反号；
- `yaw` 为偏航角速度指令；
- `throttle=-10000` 为 LQI 空闲/零集体，`+10000` 为满油门。

FC 接收后按 `value / 10000.0f` 转成 `[-1,1]`，再交给现有
`GroundStationGate`。`sender_time_ms` 只用于诊断；FC 必须用本地到达时间判断
新鲜度，不能假设两端时钟同步。

`flags`：

| 位 | 名称 | 说明 |
| ---: | --- | --- |
| 0 | `VALID` | 本帧四轴数据有效 |
| 1 | `GS_READY` | 地面站位置环已完成初始化 |
| 2–7 | reserved | 置 0 |

### 5.2 `DATA_TELEMETRY`：飞控遥测回传

方向 FC → GCS，建议 50 Hz。该包不要求 ACK。

| payload 偏移 | 长度 | 字段 | 类型 | 说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | `timestamp_ms` | uint32 | FC 本地单调毫秒 |
| 4 | 2 | `status_flags` | uint16 | 见下 |
| 6 | 2 | `roll_cdeg` | int16 | 右滚为正，0.01° |
| 8 | 2 | `pitch_cdeg` | int16 | 低头为正，0.01° |
| 10 | 2 | `yaw_cdeg` | int16 | 顺时针为正，0.01°，范围 -18000..18000 |
| 12 | 2 | `motor1_rpm` | uint16 | 上桨机械 RPM；0xFFFF 无效 |
| 14 | 2 | `motor2_rpm` | uint16 | 下桨机械 RPM；0xFFFF 无效 |
| 16 | 2 | `battery_voltage_mv` | uint16 | 电压 mV；0 表示未知 |
| 18 | 2 | `battery_current_ca` | int16 | 电流 0.01 A；-32768 表示未知 |
| 20 | 1 | `battery_remaining` | uint8 | 0–100%；255 表示未知 |
| 21 | 1 | `rc_channel_count` | uint8 | 0–18，后面紧跟 N 个 RC 通道 |
| 22 | N | `rc_channels` | int8[N] | 归一化 RC 通道，见下 |

这是基础遥测，不包含位置、速度、角速度或四元数。姿态使用欧拉角，便于地面站直接
显示和做基本姿态监视；电机转速按上桨/下桨两路返回。若后续仍需要地面站位置闭环，
应另开可选 `DATA_NAV` 数据包或由其他链路提供位置/速度，不能把缺失的位置当成零。

`rc_channel_count` 只发送当前 `rc_channels.channel_count` 个有效通道（最多 18 个），
不固定发送 18 路。通道值使用 int8 量化：

- `-127..127` 对应 `rc_channels.channels[]` 的归一化值 `-1..1`；
- 通道 3 的 throttle 按 PX4 主题约定为 `0..1`，因此线上为 `0..127`；
- `-128` 表示该通道值无效；
- 这是监视用低精度值，控制回路仍使用 `DATA_RC` 的 int16 值。

这样 8 路接收机只占 8 字节通道数据，16 路接收机才占 16 字节；不需要为本机未使用的
通道支付固定 16 位带宽。

`timestamp_ms` 是 FC 本地单调时间，不能与 GCS 时钟直接比较；GCS 判断链路新鲜度
必须使用本地接收时间。心跳中的 `session_id` 只用于区分飞控启动会话，不是安全认证。

`status_flags`：

| 位 | 名称 | 说明 |
| ---: | --- | --- |
| 0 | `ARMED` | 飞控已解锁 |
| 1 | `GS_MODE_ACTIVE` | RC 监督下的地面站模式已生效 |
| 2 | `RC_VALID` | 物理 RC 输入有效 |
| 3 | `RC_SWITCH_SELECTED` | 通道 8 位于地面站模式槽（当前为槽 4） |
| 4 | `FAILSAFE` | failsafe 生效 |
| 5 | `ATTITUDE_VALID` | 姿态角有效 |
| 6 | `MOTOR_RPM_VALID` | 两路电机 RPM 都有效 |
| 7 | `POWER_VALID` | 电压/电流至少一项有效 |
| 8 | `BATTERY_WARNING` | 电池警告或严重状态 |
| 9 | `COMMAND_BUSY` | 正在执行起飞/降落命令 |
| 10 | `TAKEOFF_ACTIVE` | 起飞流程进行中 |
| 11 | `LANDING_ACTIVE` | 降落流程进行中 |
| 12 | `LINK_OK` | 本端判断对端链路有效 |
| 13 | `RC_SWITCH_VALID` | RC 飞行模式槽有效 |
| 14 | `RC_CHANNELS_VALID` | `rc_channels` 采样有效且至少有一路 |
| 15 | reserved | 置 0 |

## 6. 指令包

### 6.1 `CMD_PING`

payload 4 字节：

| 偏移 | 长度 | 字段 | 类型 |
| ---: | ---: | --- | --- |
| 0 | 4 | `token` | uint32 |

发送端设置 `ACK_REQ`；接收端回复 `CMD_ACK`，`echo_token` 原样带回 `token`。
Ping 不改变飞控状态，不需要 session 匹配。

### 6.2 `CMD_HEARTBEAT`

payload 16 字节，双向发送，当前配置 20 Hz：

| 偏移 | 长度 | 字段 | 类型 | 说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | `uptime_ms` | uint32 | 发送端单调毫秒 |
| 4 | 4 | `session_id` | uint32 | 本端启动会话 ID |
| 8 | 2 | `status_flags` | uint16 | 与遥测状态位相同 |
| 10 | 2 | `protocol_version` | uint16 | v1 为 1 |
| 12 | 1 | `state` | uint8 | 见下 |
| 13 | 1 | `error_code` | uint8 | 最近错误，0 为无 |
| 14 | 2 | reserved | uint16 | 置 0 |

心跳不要求 ACK。FC 的 `session_id` 在每次飞控启动时变化，GCS 必须在收到
FC 心跳后记录它，并在起飞/降落/abort 指令中带回。

`state`：

| 值 | 名称 |
| ---: | --- |
| 0 | `BOOT` |
| 1 | `READY` |
| 2 | `AUTO` |
| 3 | `FAILSAFE` |
| 4 | `ERROR` |
| 5–255 | reserved |

### 6.3 `CMD_TAKEOFF`

payload 16 字节，GCS → FC：

| 偏移 | 长度 | 字段 | 类型 | 范围/说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | `target_altitude_mm` | int32 | 相对接受指令时当前位置的高度增量，正向上 |
| 4 | 2 | `max_climb_cm_s` | uint16 | 10–500 |
| 6 | 2 | `timeout_ms` | uint16 | 1000–30000 |
| 8 | 4 | `command_token` | uint32 | 本会话内唯一 |
| 12 | 4 | `session_id` | uint32 | 必须匹配 FC 心跳 |

`target_altitude_mm` 必须大于 0；实现可设置更严格的上限。ACK 的 `OK` 只表示
指令已被接受并开始执行，不表示已经到达目标高度；完成状态通过遥测的
`TAKEOFF_ACTIVE`、位置和 `COMMAND_BUSY` 判断。

### 6.4 `CMD_LAND`

payload 12 字节，GCS → FC：

| 偏移 | 长度 | 字段 | 类型 | 范围/说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 2 | `max_descent_cm_s` | uint16 | 10–200 |
| 2 | 2 | `timeout_ms` | uint16 | 1000–60000 |
| 4 | 4 | `command_token` | uint32 | 本会话内唯一 |
| 8 | 4 | `session_id` | uint32 | 必须匹配 FC 心跳 |

完成状态通过遥测的 `LANDING_ACTIVE`、着陆检测和 `COMMAND_BUSY` 判断。

### 6.5 `CMD_ABORT`

payload 12 字节，GCS → FC：

| 偏移 | 长度 | 字段 | 类型 | 说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | `reason_code` | uint32 | 1 飞手/地面站请求，2 链路原因，3 目标无效，4 其他 |
| 4 | 4 | `command_token` | uint32 | 本会话内唯一 |
| 8 | 4 | `session_id` | uint32 | 必须匹配 FC 心跳 |

Abort 只取消起飞/降落流程并回到 RC 控制，不绕过 RC 门控，也不直接写执行器。

### 6.6 `CMD_ACK`

payload 12 字节：

| 偏移 | 长度 | 字段 | 类型 | 说明 |
| ---: | ---: | --- | --- | --- |
| 0 | 1 | `acked_class` | uint8 | 被确认包的类别 |
| 1 | 1 | `acked_id` | uint8 | 被确认包的消息 ID |
| 2 | 2 | `acked_seq` | uint16 | 被确认包的序号 |
| 4 | 1 | `result` | uint8 | 见下表 |
| 5 | 1 | `detail` | uint8 | 命令相关细节，0 为无 |
| 6 | 2 | reserved | uint16 | 置 0 |
| 8 | 4 | `echo_token` | uint32 | Ping/命令 token，无则 0 |

ACK 包的 `flags.RESPONSE=1`，不设置 `ACK_REQ`。

`OK` 表示包已通过校验并且指令已被接受/开始处理；对起飞、降落和 abort 不代表
流程已经完成。完成/失败状态通过后续遥测状态位和心跳 `state` 观察。

`result`：

| 值 | 名称 | 说明 |
| ---: | --- | --- |
| 0 | `OK` | 已接受/已处理 |
| 1 | `UNSUPPORTED` | 未知或不支持的指令 |
| 2 | `BAD_LENGTH` | payload 长度不符 |
| 3 | `BAD_ARGUMENT` | 参数越界或非法 |
| 4 | `REJECTED_SAFETY` | RC/模式/失效保护不允许 |
| 5 | `BUSY` | 正在执行其他指令 |
| 6 | `TIMEOUT` | 指令执行超时 |
| 7 | `DUPLICATE` | 重复 token，未重复执行 |
| 8 | `BAD_VERSION` | 协议版本不支持 |
| 9 | `INTERNAL_ERROR` | 飞控内部错误 |
| 10 | `STALE_SESSION` | session 不匹配或已过期 |
| 11 | `LINK_STALE` | 链路/心跳不新鲜 |
| 12–255 | reserved | 保留 |

## 7. 序号、重试与去重

- 每个发送方向维护独立的 16 位 `sequence`，每发送一个非 ACK 包递增，65535 后回绕；
- 重传必须复用原 `sequence`、原 `command_token`，并设置 `RETRY`；
- `CMD_PING`、`CMD_TAKEOFF`、`CMD_LAND`、`CMD_ABORT` 设置 `ACK_REQ`；
- ACK 等待时间 20 ms，最多重试 3 次（共 4 次发送）；
- 接收端对同一 `(message_id, sequence)` 的重复包重新回 ACK，但不重复执行；
- 指令执行去重按 `(message_id, command_token)` 做，至少保留 5 秒或最近 32 条；
- 重复 token 返回 `DUPLICATE`，不得再次启动起飞、降落或 abort 动作。

数据包不重传。序号跳变只用于统计丢包，不允许用旧 RC 帧补偿。

## 8. 时序与链路监督

| 项目 | 频率/超时 |
| --- | --- |
| `DATA_RC` | 50 Hz 建议，FC 侧新鲜度超时 250 ms |
| `DATA_TELEMETRY` | 50 Hz 建议，GCS 侧新鲜度超时 250 ms |
| `CMD_HEARTBEAT` | 20 Hz 建议，双向超时 500 ms |
| 指令 ACK | 20 ms 超时，最多 3 次重试 |
| `CMD_TAKEOFF` 执行超时 | 1000–30000 ms，由指令指定 |
| `CMD_LAND` 执行超时 | 1000–60000 ms，由指令指定 |

带宽估算（含 10 字节包头和 2 字节 CRC；`DATA_RC` 28 B @ 50 Hz ≈ 1.4 kB/s，
双向心跳 28 B @ 20 Hz ≈ 1.12 kB/s）：

| 遥测配置 | 遥测包长 | 遥测带宽 | 总带宽 |
| --- | ---: | ---: | ---: |
| 无 RC 通道（N=0） | 34 B | 1.7 kB/s | 3.7 kB/s |
| 8 路 int8（N=8） | 42 B | 2.1 kB/s | 4.1 kB/s |
| 16 路 int8（N=16） | 50 B | 2.5 kB/s | 4.5 kB/s |
| 18 路 int8（N=18） | 52 B | 2.6 kB/s | 4.6 kB/s |
| 16 路 int16（对照） | 66 B | 3.3 kB/s | 5.3 kB/s |
| 18 路 int16（对照） | 70 B | 3.5 kB/s | 5.5 kB/s |

按 16 路接收机比较，int8+实际通道数比固定 int16 方案把遥测带宽从 3.3 kB/s
降到 2.5 kB/s，节约约 0.8 kB/s（遥测段约 24%，含 RC 和心跳的总链路约 15%）；
8 路接收机可降到 2.1 kB/s，节约约 1.2 kB/s。全部配置仍远低于 4 Mbaud 能力。

FC 只有在以下条件都成立时才把地面站输入视为有效：

- `DATA_RC.flags.VALID=1`；
- RC 数据在 250 ms 内；
- GCS 心跳在 500 ms 内；
- 物理 RC、模式槽 4 和 External Mode 状态仍由 `GroundStationGate` 判定通过。

任一条件失效，FC 立即回到 RC；不自动恢复。GCS 的 `DATA_RC` 只依赖新鲜
FC 遥测，不因低频心跳偶发缺失而停止；需要 `session_id` 的指令仍要求 FC
心跳新鲜。GCS 心跳在 FC 侧超过 500 ms 时仍会触发回 RC。

## 9. 解析与错误处理

建议使用非阻塞字节流状态机：

1. 搜索 `0xA5 0x5A`；
2. 读取 10 字节包头；
3. 检查版本、类别、长度；长度大于 512 时丢弃并重新同步；
4. 读取 payload 和 CRC；
5. 校验 CRC；失败则丢弃并重新同步；
6. 按类别和消息 ID 分发；未知数据丢弃，未知指令按规则回 `UNSUPPORTED`；
7. 更新链路时间戳和错误统计。

magic 可能出现在 payload 中，解析器必须按长度读取，不能在 payload 内扫描 magic。
CRC 失败时不要回 ACK，避免错误放大。

v1 接收端遇到非 `0x01` 版本直接丢弃并计数；`BAD_VERSION` 预留给未来的版本协商。
已知指令的 payload 长度不符且包头/CRC 有效时回 `BAD_LENGTH`；已知数据消息长度
不符时丢弃。

建议统计：

- `rx_packets`、`tx_packets`；
- `crc_errors`、`length_errors`、`version_errors`；
- `unknown_data`、`unknown_commands`、`duplicates`；
- `last_valid_rx_ms`、`last_valid_tx_ms`；
- `tx_retries`、`ack_timeouts`。

## 10. 与 RC 安全链路的关系

GS-Link 不能取代 RC，也不能直接写电机/舵机：

- `DATA_RC` 只提供四轴归一化虚拟遥控量；
- 模式槽只从物理 RC 的 `manual_control_switches` 读取，协议不能设置；
- 起飞/降落/abort 仅在 `GS_MODE_ACTIVE`、RC 已解锁、模式槽 4 且链路新鲜时执行；
- 任何指令被拒绝都返回 ACK，不允许静默改变飞行状态；
- RC disarm、RC 丢失、通道 8 离开模式槽 4 或链路超时，地面站指令立即取消；
- 起飞/降落指令不会自动 arm，也不会绕过 `NN_LQI_OUTPUT_EN`；
- `CMD_TAKEOFF`/`CMD_LAND` 的飞控执行逻辑尚未实现前，飞控必须返回
  `UNSUPPORTED`，不能假装执行。

## 11. ESP、GCS 和 FC 的角色

- FC：协议端点，解析 `DATA_RC`/指令，生成 `DATA_TELEMETRY`/心跳/ACK；
- GCS：协议端点，运行位置环，发送 `DATA_RC` 和指令，接收遥测；
- ESP：透明字节桥，不修改包头、payload、CRC 或序号；
- ESP 不应向 `ttyS0` 注入文本日志、调试字符串或私有帧；
- LAN 传输选用 UDP；一个 UDP datagram 承载一个 GS-Link 完整包，不把多个控制帧
  合并到一个 datagram，也不在控制路径使用 TCP；
- ESP 应按包头长度把 UART 字节组成完整帧后再发 UDP；不要按串口 read() 的任意
  片段直接发包，否则会放大包率和抖动；
- UDP 端口和单播/组播地址作为部署参数，不在包头中编码；
- 524 字节最大包远低于 1500 字节以太网 MTU，避免 IP 分片；
- 丢包、乱序和重复由 GS-Link 的序号、超时和 ACK/重试处理；
- ESP 的串口接收缓冲至少容纳 524 字节，不能把半个包交给 FC。

## 12. 参考实现细节

CRC 参考实现：

```c
static uint16_t gs_crc16(const uint8_t *data, size_t length)
{
    uint16_t crc = 0xFFFF;

    for (size_t i = 0; i < length; ++i) {
        crc ^= (uint16_t)data[i] << 8;

        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021)
                                 : (uint16_t)(crc << 1);
        }
    }

    return crc;
}
```

CRC 自检：输入 ASCII `"123456789"`（仅用于校验算法，不在协议线上发送字符串），
结果应为 `0x29B1`。

示例 `DATA_RC` 单帧：

```text
A5 5A 01 01 01 00 01 00 10 00
E8 03 00 00
00 00 00 00 00 00
F0 D8
01 64
00 00
97 B2
```

解释：

- `version=01`、`class=DATA(01)`、`message_id=DATA_RC(01)`；
- `flags=00`、`sequence=1`、`payload_len=16`；
- `sender_time_ms=1000`；
- roll/pitch/yaw 为 0；
- throttle `0xD8F0`（小端 `F0 D8`）= `-10000`，即空闲；
- `flags=01`（VALID）、`link_quality=100`；
- CRC 为 `0xB297`，线上字节 `97 B2`。

## 13. 迁移与实现状态

目标状态是 ESP 自定义协议独占 `ttyS0`：

当前 `micoair_h743-v2_nncontrol` 已实现通信-only 版本：

1. `MAV_0_CONFIG=0`，`/dev/ttyS0` 由 `gs_link` 独占；
2. `gs_link start -d /dev/ttyS0 -b 4000000` 在 `nn_control` 之前启动；
3. `DATA_RC` 转换为 `GsLinkControl`，继续经过 RC 模式槽 4 门控；`NN_GS_EN` 默认
   为 `1`，因此门控通过时允许替换 RC 四轴输入；
4. `DATA_TELEMETRY` 包含姿态、两路电机转速、电源状态和实际 RC 通道；
5. `CMD_PING`、心跳和 ACK 已实现；
6. `CMD_TAKEOFF`、`CMD_LAND`、`CMD_ABORT` 当前一律返回 `UNSUPPORTED`，
   不改变飞控状态、不写执行器；
7. USB `/dev/ttyACM0` 的 QGC MAVLink 保持不变。

通信-only 固件不会执行地面站闭环实飞；启用任何飞行动作前必须另行实现和验证
起飞/降落状态机。

## 14. 后续项（当前通信测试不启用）

- UDP 端口分配、单播/组播和 MTU 策略；
- 多机 ID/路由；
- 认证、HMAC、重放窗口；
- 遥测频率和字段裁剪；
- 起飞/降落状态机的具体高度、速度和完成判据；
- 地面站闭环实飞授权与飞行前检查；
- 是否需要事件包或日志通道；
- 是否需要大包分片（v1 明确不支持）。


## 动捕三轴姿态与时间同步扩展（2026-11-06）

新增DATA `0x03`（36字节动捕姿态）和DATA `0x04`（8字节时间同步请求/24字节回复）；旧消息不变。字段、坐标变换、50 Hz发送规则与启用参数见[MOCAP_FULL_ATTITUDE_zh.md](MOCAP_FULL_ATTITUDE_zh.md)。
