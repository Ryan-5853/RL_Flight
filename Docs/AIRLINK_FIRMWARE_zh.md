> 恢复规则更新：AIRLINK 故障后，两机各自收到物理 RC disarm 且确认撤防后可清除故障；重新握手并再次 arm 即可恢复，无需重启。详见 `Docs/GROUND_PAIR_FLIGHT_zh.md`，以下历史永久锁存描述以此为准。

# AIRLINK v1 固件实现

> 2026-11-03 更新：当前地面组合飞行、External Mode 3 和断链停机锁死规则以
> [地面组合飞行流程](GROUND_PAIR_FLIGHT_zh.md) 为准；本文中的旧接线边界和单体降落回退不再适用。

日期：2026-10-30

本文对应 `AIRLINK_PROTOCOL_V1_zh.md` 的固件侧实现。实现位于
`px4_trans/px4/src/modules/airlink/`，默认角色为 disabled，不会因刷入新固件
而自动建立主从控制。

## 1. 已实现范围

- UART `6,000,000 8N1`、全双工、无硬件流控；
- 16 B 固定帧头、显式小端序列化、CRC-32C/ISCSI、EOF 和流式重同步；
- `SENSOR_STATE` 固定 42 B 打包，包含 IMU、姿态四元数、角速度估计、双 ESC
  RPM 和单电池电压；
- `ACTUATOR_COMMAND` 6 B、48 bit LSB-first 五路执行器命令；
- `HEARTBEAT`、三段式 `LINK_HANDSHAKE`、`CONTROL_ENABLE/RELEASE`、
  `PAIR_ARM/DISARM`、`EMERGENCY_WITHDRAW` 和 ACK；
- 可靠事件原序号重发、64 项/60 s 去重缓存、结果缓存和会话隔离；
- 500 Hz 心跳、500 Hz 传感器/控制流、10 ms 数据新鲜度、100 ms 预控制超时；
- `PAIR_ARM` 100 ms ACK 窗口、100 ms 首帧提交窗口、地面断链本地 disarm；
- `PAIR_REATTACH_LATCHED` 在当前启动周期内粘滞；
- 空中 `PAIR_LOST` 后进入 `SINGLE_LANDING`：先尝试恢复 GS-LINK，否则以
  collective `0.25` 保持水平姿态下降；冲击、翻转或超时触发本地 disarm；
- `LINK_TEST_CONTROL`/`LINK_TEST_DATA` 回环测试、丢包/重复/乱序、
  RTT 分位数、吞吐和最大接收间隔统计；
- QGC `AIRLINK_ROLE`、`AIRLINK_PAIR_EN` 参数和 NSH 命令；
- PX4 Commander arming check 接入，配对模式下不能绕过 `PAIR_ARM` 直接 arm；
- `airlink_status`、`airlink_actuator_command`、`airlink_pair_command` 和
  `airlink_test_result` uORB 接口。

## 2. 固件组成

| 文件 | 作用 |
| --- | --- |
| `AirlinkProtocol.hpp` | 固定 ABI、CRC、帧解析、字段量化、全部消息编解码 |
| `AirlinkSession.hpp/.cpp` | 发现、握手、可靠事件、arm 生命周期、新鲜度和链路测试状态机 |
| `Airlink.hpp/.cpp` | UART、uORB、参数、Commander arm/disarm、控制和诊断发布 |
| `airlink_params.c` | 角色、配对模式和 SINGLE_LANDING 安全阈值 |
| `AirlinkStatus.msg` | 会话、链路状态、错误计数和本地安全门控 |
| `AirlinkActuatorCommand.msg` | 从机收到的五路控制器域命令 |
| `AirlinkPairCommand.msg` | C_PAIR 控制器向固件提供的本机/从机五路命令 |
| `AirlinkTestResult.msg` | 链路测试完整机器可读结果 |

## 3. 板级端口与参数

`micoair_h743-v2_nncontrol` 启动脚本将 AIRLINK 配置为：

```text
device = /dev/ttyS1
baud   = 6000000
format = 8N1, no flow control
```

`/dev/ttyS1` 是该板卡映射中的 UART2/GPS2，`/dev/ttyS0` 已由 GS-LINK 使用。
两端 TX/RX 必须交叉并共地。

参数默认值：

```text
AIRLINK_ROLE=0       # 0 disabled, 1 master, 2 slave
AIRLINK_PAIR_EN=0    # 0 保留单机控制，1 选择双机控制
AIRLINK_IMPACT=35    # m/s²，初始经验值
AIRLINK_IMP_MS=20    # 连续冲击确认时间
AIRLINK_FLIP_DEG=70  # 相对竖直方向，初始经验值
AIRLINK_FLIP_MS=150  # 连续翻转确认时间
AIRLINK_LAND_MS=15000 # SINGLE_LANDING 强制 disarm 时限
```

`AIRLINK_ROLE` 和 `AIRLINK_PAIR_EN` 只能在 disarmed、无 `PAIR_ACTIVE` 和无
链路测试时修改。armed/active 时 QGC 写入会被固件拒绝并恢复原值。

五个降落阈值参与握手 `config_crc32c`。两机必须使用相同数值；修改后旧会话必须
重新握手。`nn_control` 在 disarmed 阶段锁存这些阈值，armed 后修改不会影响当前
flight epoch。默认值仅用于首轮 HIL/台架，不是飞行标定结果。

## 4. 操作命令

NSH/QGC MAVLink Console：

```sh
airlink help
airlink status
airlink role master
airlink role slave
airlink role off
airlink pair on
airlink pair off
```

主从配置顺序、状态判据、主题监听、链路测试和故障排查的完整 PX4 端操作说明见
[`../px4_trans/README.md`](../px4_trans/README.md) 的“AIRLINK v1 PX4/NSH 调试指令”一节。

配对 arm 事务：

```sh
airlink arm
airlink disarm
```

QGC `MAV_CMD_COMPONENT_ARM_DISARM` 和 RC/摇杆 `action_request` 的 arm 请求也会
触发同一事务。配对模式下，Airlink 会先完成 fresh `CONTROL_ENABLE`，随后发送
`PAIR_ARM`；从机真正进入 PX4 armed 后才回成功 ACK，主机收到 ACK 后才执行本机
arm。

控制释放和紧急撤销：

```sh
airlink release
airlink emergency
```

## 5. 控制权接口

`airlink` 不直接绕过 `nn_control` 写执行器。命令所有权由以下规则执行：

- master 在 `PAIR_ARMED_GROUND/PAIR_ACTIVE` 只使用新的
  `airlink_pair_command.master_*`；
- slave 在 `PAIR_ACTIVE` 只使用 10 ms 内新鲜的
  `airlink_actuator_command`；
- 其他状态恢复 `nn_control` 的 C_SINGLE 输出；
- 两种模式不会在同一控制周期混合写入执行器；
- 本地 RC、Commander、failsafe、`NN_LQI_OUTPUT_EN` 和 airframe 配置门仍然有效。

`AirlinkPairCommand.msg` 是 C_PAIR 控制器与 AIRLINK 固件之间的稳定接口。
当前仓库尚无发布该主题的 C_PAIR 控制器实现，因此没有 C_PAIR 发布者时：

- `CONTROL_ENABLE` 不会由自动流程发起；
- `PAIR_ARM` 会被拒绝；
- 飞机保持单机控制或 disarm。

这是有意的 fail-closed 行为，不是通信层的失败。

## 6. 链路测试

### 控制待命授权与新鲜度（2026-11-04 修正）

`PAIR_CONTROL_READY` 在同一有效会话内等待飞手 ARM，不再按授权 ACK 的
10 ms 年龄退出。两端继续检查实时 RC、安全门和组合模式；任一端条件撤销时
退回 READY，周期数据断流仍走原有断链处理。PAIR_ARM 两端仍检查实时安全和
对端新鲜度。飞行中的传感器/控制流 10 ms 门槛没有放宽。

### 延迟与复位取证

两端 disarmed，在主机 PX4 控制台执行下面的测试；测试要求 `AIRLINK_READY`，
可先将两端切出组合模式以避免自动控制授权进入待命。恢复组合模式另行操作。
`rtt_max_us` 是完整往返时间，不能直接作为单向延迟，也不能简单除二推定最坏
单向延迟。`max_receive_gap_us` 是测试回包到达的最大间隔，`link_interruptions`
统计回包间隔超过 10000 us 的次数；这些是测试负载下的指标，不是实际
SENSOR_STATE 源数据年龄。

实际传感器交换可在主机运行 `airlink sensortest 10000`，结束输出
`AIRLINK_SENSOR_TEST_JSON`，现有字段只有接收、重复、坏帧数量和结束时新鲜度，
不能据此断言整个测试期间没有超时。

判定 CEKF 重置需同时保留主机控制台的 `CEKF reset: ...`、
`listener airlink_status`、`listener airlink_sensor_state` 和
`listener composite_state_estimate`（或相应 ULog）。重点比较接收时间间隔、
`status_flags` 的 SampleFresh 位、估计器 reset_counter 和会话 ID。
两机启动时钟独立，不能将主机接收时间减从机 IMU 时间当作单向延迟。
SampleFresh 还取决于从机本地 IMU、加速度、姿态、角速度全部有效且新鲜，
掉位不等于无线传输超时。


默认参数：

```sh
airlink test --duration-ms 5000 --rate-hz 500 --payload-bytes 256
airlink aborttest
```

测试仅允许 master、`AIRLINK_READY`、两端 disarmed 时执行。退出后结果同时发布到
`airlink_test_result`，并在控制台输出完整 `AIRLINK_TEST_JSON=...`。其中
`buffer_overflows` 必须为零；非零表示应用层单帧接收缓冲被错误写入，而不是可以
通过重试消除的普通网络丢包。MAVLink `STATUSTEXT` 因 PX4 50 B 限制输出关键摘要，
完整字段以 uORB 主题和控制台日志为准。

AIRLINK 工作项按 1 kHz 读取和解析接收数据，但 `SENSOR_STATE`、
`ACTUATOR_COMMAND`、心跳、会话 tick 和状态发布继续按 2 ms 节流，数据流协议
速率仍为 500 Hz。

## 7. 验证

主机协议和双端状态机测试：

```bash
bash px4_trans/tests/run_airlink_protocol_check.sh
```

覆盖：

- CRC-32C 自检和协议文档全部 64 B 金样例；
- 任意拆包、粘包、假 SOF、payload 内 SOF、坏 CRC/EOF/version/role；
- 传感器量化边界、饱和位、四元数、可选 ESC 和电压；
- 五路执行器最小/中点/最大位流及非法码；
- 三阶段握手、配置不一致、角色冲突、可靠重试；
- `PAIR_ARM`、首次控制接管、重复 arm、10 ms 流超时和重连锁存；
- `SINGLE_LANDING` 冲击、翻转确认时间和总超时边界；
- 未知 COMMAND 的 `UNSUPPORTED` ACK 和链路测试回环。

固件构建：

```bash
env \
  CCACHE_DISABLE=1 \
  MAKEFLAGS= \
  PATH="$PWD/px4_trans/px4/.venv/bin:$PATH" \
  PYTHON_EXECUTABLE="$PWD/px4_trans/px4/.venv/bin/python" \
  make -C px4_trans/px4 micoair_h743-v2_nncontrol
```

输出：

```text
px4_trans/px4/build/micoair_h743-v2_nncontrol/micoair_h743-v2_nncontrol.px4
```

## 8. 尚未构成飞行许可的边界

- 尚未在真实 MicoAir H743 上验证 6 Mbaud 误码、DMA 缓存、连接器半插入和
  运行中拔线；
- 当前仓库没有 C_PAIR 控制算法发布者；
- 冲击、翻转和 landing 超时已经提供可配置经验默认值；这些值尚未按当前机架
  质量、减振和下降速度完成台架/HIL 标定；
- 编译与主机测试通过只证明协议、状态机和固件集成可构建，不等同于 HIL、台架、
  实机或飞行许可。

## 飞行中通信健康统计（2026-11-04）

AIRLINK 持续记录 1 kHz 工作项的开始间隔、执行耗时、双向角色对应周期帧的
收发间隔和峰值、超 10 ms 次数、周期流连续超时事件、本机源年龄和对端 IMU
无效事件。由 `airlink_health`（同时记录 ULog）到两端 GS-Link DATA 0x10 node schema 2（260 B），
地面站兼容 schema 1 并将新字段直接保存在 pair JSONL。

新增 DATA 0x0C 为 8 字节本机时间 token 的请求/回显，flags 分别 0/Response。
每端最多 5 Hz 探测 RTT，100 ms 未获匹配回复计一次超时，测试负载期间暂停主动探测。
RTT 包含发送排队和双方处理调度，不代表单向无线延迟；不相减两机启动时钟。
探测帧不更新任何周期流、心跳安全计时，不可掩盖断流。新 RuntimeTiming
capability 为 bit9，必须两端升级后才能完成组合握手。

GS-Link node schema 2 前 108 B 保持兼容，其后依次为 38 个 U32 字段。
时间单位微秒，未知年龄 UINT32_MAX；health 新增峰值/计数在模块重启时清零。
`health_flags` bit0/1 为已有周期 RX/TX，bit2 为当前会话 RTT 有效（<=1秒），
bit3 周期流超时，bit4 对端 IMU 无效，bit5 正在监控周期流，bit6 本机四路源年龄新鲜，
GS-Link bit7 表示统计主题新鲜（<=10ms）。
`stream_timeouts` 在读串口前检查，连续过期只计一次，恢复后再次过期才增计。
`run_work_us` 是墙钟执行耗时而非纯 CPU 时间；`tx_interval_us` 记录成功写串口驱动，
不代表无线发射时刻。详细字段、判读规则与地面站显示见地面站
`docs/08-ground-pair-flight.md` 和 `docs/gs-link-diagnostics-schema.json`。

此改动用于取证，不改变 CEKF 重置策略或 10 ms 控制/传感器安全门槛。

## 组合估计失效双机停机（2026-11-04）

保护范围是主机组合解锁事务开始、组合已解锁及组合激活阶段。未解锁时估计未就绪
只阻止组合控制/解锁，不在上电初始化期间误触发双机停机。

保护期间，估计输出无效、采样/发布年龄超过 10 ms、任一 IMU 源缺失/过期、
退化成单源、非有限反馈/非法四元数、EKF 失去姿态有效状态或报告内部 fault，
以及 reset_counter 改变都会锁存故障。comp_attitude 在源恢复、会话变化或 CLI
请求需要重建 EKF 时，若组合仍受保护，直接发布持续无效估计，不在已解锁状态重建。
AIRLINK 另行检查估计发布年龄及收到的从机传感器有效位，故估计器线程停止运行也会被检测。

故障发生后：

- 主机立即撤销本机执行器输出并请求强制 disarm；comp_control 和 NNControl 均有
  独立估计保护，阻止复位后的高推力帧到达输出端。
- AIRLINK 清除排队中的旧控制帧和待完成 ARM 事务，优先发送带 reason=11
  （EstimatorInvalid）的可靠 EmergencyWithdraw。接收端撤销跟随命令、强制 disarm
  并锁存 pair_lost/reattach；数据恢复、GS 请求、延迟 ACK 均不能自行恢复输出。
- 若紧急包丢失或链路断开，主机停止发送有效控制，从机原有 10 ms 控制流看门狗停机。
  这是调度及通信延迟受限的双机停机，不保证两台硬件同一时刻断电。
- 两端实际 disarm 后，分别通过物理 RC OFF 确认清除本端锁存，再重新握手和 OFF→ON
  解锁。自动 disarm 不清除故障。估计器故障原因/epoch 通过原观测器诊断上报。

新增 EstimatorKill capability bit10；两机必须同步更新，避免与旧的紧急撤销语义混用。
故障锁存不会放宽任何 10 ms 新鲜度门槛。
