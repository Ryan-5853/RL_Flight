# 地面站位置闭环与 RC 子模式约定

状态日期：2026-11-03\
适用固件：`micoair_h743-v2_nncontrol`\
串口：TELEM1 / `SERIAL1` / `/dev/ttyS0`\
串口波特率：`4000000`；`8N1`、无硬件流控是当前实现默认
协议设计：[GS-Link v1](GS_LINK_PROTOCOL_V1_zh.md)
UDP 实测记录：[2026-10-20 延迟与带宽测试](GS_LINK_UDP_LATENCY_TEST_2026-10-20_zh.md)

## 0. 约定状态

本文把“已经确认”和“下一步讨论”分开记录，避免把原型协议误当成最终协议。

### 已固化

- 地面站负责位置闭环（若启用）；飞控不接收位置目标，不新增位置控制器；
- GS-Link v1 基础遥测只含姿态、两路电机转速和电源状态；位置/速度必须来自
  可选 `DATA_NAV` 或另一条链路；
- 飞控只接收四轴归一化虚拟遥控量；
- 地面站控制注册为 PX4 External Mode，但仍受 RC 手动控制监督；
- 实机 RC 是唯一主手动输入；地面站只能在 RC 监督通过后替换四轴指令；
- RC 解锁、通道 8 模式槽 4、链路新鲜度和全部既有安全门控必须同时成立；
- 通道 8 离开模式槽 4、RC disarm、RC 丢失或地面站链路超时，立即退出；
- 地面站链路中断后恢复，不自动重新接管，必须先回模式 1 再选模式 6；
- 串口固定为 4,000,000 baud；`8N1`、无硬件流控按当前实现，协议讨论时确认；
- 目标状态下，串口 1 的运行时唯一占用者是 ESP 自定义 GS-Link 协议；MAVLink
  不再占用该口；
- ESP32 不参与飞控模式判断、解锁判断或安全门控。

### 协议 v1 通信测试已实现，飞行动作未启用

- 帧格式、可变 payload、CRC-16、数据和指令包布局已固化在
  [`GS_LINK_PROTOCOL_V1_zh.md`](GS_LINK_PROTOCOL_V1_zh.md)；
- 起飞/降落的飞控执行状态机、UDP 端口/寻址、多机路由和认证仍未实现；
- 当前固件已由 `gs_link` 独占 `ttyS0`，`MAV_0_CONFIG=0`；
- `TAKEOFF`/`LAND`/`ABORT` 只返回 `UNSUPPORTED`，不会改变飞控状态或执行器；

当前代码里的 MAVLink `MANUAL_CONTROL` 只是历史过渡原型；GS-Link v1 驱动已经
替换了该口的输入适配层，RC 监督行为保持不变。

## 1. 当前控制器输入契约

`nn_control` 的控制后端只接收以下四轴归一化手动指令：

| 字段 | 范围 | 含义 |
| --- | --- | --- |
| `rc_roll` | `[-1, 1]` | 右滚为正 |
| `rc_pitch` | `[-1, 1]` | PX4 约定下低头为正；`LqiManualReference` 内部反号 |
| `rc_yaw` | `[-1, 1]` | 偏航角速度指令 |
| `rc_throttle` | `[-1, 1]` | 映射到上桨集体基准 `[0, 1]` |

后端输入还包括姿态、角速度、角加速度、加速度、NED 局部位置/速度、ESC 转速和电池
状态；输出仍是两路电机 `[0, 1]` 加三路舵机 `[-1, 1]`。固定 500 Hz（2 ms）时基和
`NN_LQI_OUTPUT_EN` 输出门控不变。

实机主手动输入只接受物理 RC：`manual_control_setpoint.data_source == SOURCE_RC`，
并且原始 `input_rc` 必须新鲜、未丢失、未 failsafe，来源不能是 MAVLink。这样
`RC_CHANNELS_OVERRIDE` 不能伪装成 RC 绕过通道 8。

HIL 保留原有 WebUI/MAVLink 飞手输入路径；这是现有仿真工作流的例外，不改变实机
RC 优先约定。

## 2. 链路

```text
地面站位置环
  │  UDP：遥测 + 地面站指令
  ▼
机载 ESP32（传输桥，不做飞控模式/安全判断）
  │  UART：4,000,000 baud（当前 8N1，无硬件流控）
  ▼
飞控 TELEM1 = SERIAL1 = /dev/ttyS0
  │  GS-Link v1（gs_link 独占）
  ▼
nn_control
  │  RC 解锁 + RC 模式槽 4 + 指令新鲜
  ▼
LQI / 执行器输出
```

飞控侧不再新增位置环。若地面站启用位置闭环，位置和速度必须由可选 `DATA_NAV`
或其他链路提供；GS-Link v1 基础遥测只提供姿态、两路电机转速和电源状态。

### 2.1 串口 1 的占用关系

这里的“串口 1”指 TELEM1 / `SERIAL1` / `/dev/ttyS0`。它不是一个可以同时被多个
独立驱动打开的共享总线；运行时和启动/刷写阶段是先后关系：

| 阶段 | 占用者 | 波特率 | 说明 |
| --- | --- | --- | --- |
| 正常运行 | ESP GS-Link v1 自定义协议 | 4,000,000 | 串口 1 的唯一运行时占用者；`MAV_0_CONFIG=0` |
| bootloader / 刷写 | bootloader USART | 115,200 | `INTERFACE_USART_CONFIG="/dev/ttyS0,115200"`，仅启动/恢复阶段 |

当前目标的其他默认串口分配如下（`rc.serial` 生成配置）：

| 端口 | 默认功能 |
| --- | --- |
| TELEM2 / `/dev/ttyS3` | 未分配（`MAV_1_CONFIG=0`） |
| TELEM3 / `/dev/ttyS4` | 未分配 |
| TELEM4 / `/dev/ttyS7` | MAVLink instance 2，115200（板级默认 `MAV_2_CONFIG=104`） |
| GPS1 / `/dev/ttyS2` | GPS（`GPS_1_CONFIG=201`） |
| GPS2 / `/dev/ttyS1` | 禁用（`GPS_2_CONFIG=0`） |
| RC / `/dev/ttyS5` | RC 输入（`RC_PORT_CONFIG=300`、`RC_SBUS_PRT_CFG=300`） |
| UART6 / `/dev/ttyS6` | 未分配 |

如果把 GPS、RC、DShot 遥测或第二个 MAVLink 实例也配置到 `101`，`rc.serial_port`
会打印 `Conflicting config for /dev/ttyS0`，不能把两个独立驱动同时挂到同一个
UART 上。当前已决定：GS-Link v1 自定义协议成为 `ttyS0` 的唯一运行时占用者，
关闭该口的 MAVLink。

关闭 TELEM1 的 MAVLink 不会关闭 USB 上的 QGC 通道。USB `/dev/ttyACM0` 由
`cdcacm_autostart` 独立启动，默认 `SYS_USB_AUTO=2`；只有禁用 USB 自动启动、
移除 MAVLink 模块或主动停止 USB MAVLink 实例时，QGC 才会失去 USB 连接。
关闭 `MAV_0` 后 MAVLink 实例号可能变化，必须用 `mavlink status` 重新确认，
不能继续假设 TELEM1 仍是 instance 0。

## 3. RC 子模式状态机

进入地面站控制必须同时满足：

1. 遥控器已解锁（`actuator_armed.armed == true`）；
2. RC 手动输入有效、新鲜，且来源是物理 RC；
3. 原始 `input_rc` 没有 `rc_lost`/`rc_failsafe`，来源不是 MAVLink；
4. `RC_MAP_FLTMODE` 已配置为通道 8；
5. 通道 8 当前对应 `manual_control_switches.mode_slot == 4`；
6. `COM_FLTMODE4` 映射到 `nn_control` 注册并可选择的 External Mode；
7. 飞控实际 `vehicle_status.nav_state` 已进入该 External Mode；
8. 本次解锁后模式 6 被重新选择，并保持 100 ms；
9. 地面站指令在 `NN_GS_TIMEOUT_MS` 内持续更新。

物理 RC 的槽 4 可能比 Commander 的 External Mode 早一个控制周期可见。门控在
`nav_state` 尚未切到 External Mode 时保持 RC，但保留本次模式 6 的选择授权；
只有 External Mode 生效后才开始 100 ms 确认计时，不需要飞手重复拨动开关。

以下任一条件成立，在下一个 2 ms 控制周期退出地面站控制：

- RC disarm；
- 通道 8 离开模式槽 4，包括进入模式 1–3、5–6；
- RC 信号丢失、failsafe、校准或通道数不满足；
- 地面站指令超时；
- `NN_GS_EN=0` 或当前传输层来源不匹配。

退出后立即回到 RC。若地面站链路恢复，即使通道 8 仍在模式 6，也不会自动重新
接管；飞手必须先回模式 1（Manual/RC），再拨到模式 6。

GS-Link `DATA_RC` 超时不会改变 PX4 External Mode 的 `can_run` 状态，也不会让
Commander 自动进入 RTL/Land/Descend。模式仍保持为 External Mode 2，但四轴输入由
`GroundStationGate` 切回物理 RC。只有物理 RC 失效、姿态/角速度无效或 Commander
自身的硬安全门控失败时，才允许进入 PX4 通用 failsafe。

Ground Station 外部模式显式发布的控制模式与 PX4 Manual 一致：manual、attitude、
rates 和 allocation 开启，position、velocity、altitude、climb-rate、auto 和 offboard
关闭。否则 PX4 外部模式默认会要求位置/爬升率控制，land detector 在静态台架无法
确认 `landed`，并会拒绝 RC disarm：`Disarming denied: not landed`。

当前映射如下：

```text
RC_MAP_FLTMODE=8
COM_FLTMODE1=0      # Manual，RC
COM_FLTMODE4=101    # External Mode 2，地面站
COM_FLTMODE6=102    # External Mode 3，组合模式
NN_GS_MODE_SLOT=4
```

`nn_control` 启动后注册名为 `Ground Station` 的 External Mode，并把模式槽 4 自动
映射到 PX4 实际分配的 External Mode 编号。PX4 通过 `COM_MODE<n>_HASH` 保证同一
模式名稳定分配到同一索引。22001 机架启动脚本预留 `COM_MODE0_HASH`，并把
`Ground Station` 的 FNV-1a 哈希写入 `COM_MODE1_HASH`，因此固定分配为
External Mode 2（`nav_state=24`）：

```text
COM_MODE0_HASH = -301203826
COM_MODE1_HASH = -696130346  # FNV-1a("Ground Station")
```

WebUI 只有在地面站遥测新鲜且飞控报告 `GS_MODE_ACTIVE` 时才显示“地面站正在控制”。
四通道命令已经发送、但飞控仍按 RC 处理时，界面显示等待飞控确认；这避免把地面站
发送状态误认为飞控实际控制源。

`NN_GS_SW_CHAN` 和 `NN_GS_SW_TH` 是旧端点门控参数，仅为参数兼容保留，不再参与
当前状态机。

## 4. 当前固件参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `NN_GS_EN` | `1` | 默认允许 RC 监督下的地面站输入；可显式设为 `0` 禁用 |
| `NN_GS_MODE_SLOT` | `4` | 地面站 RC 飞行模式槽位 |
| `NN_GS_TIMEOUT_MS` | `250` | 地面站指令最大允许间隔 |
| `RC_MAP_FLTMODE` | 部署值 `8` | 通道 8 作为六槽飞行模式开关 |
| `COM_FLTMODE1` | 部署值 `0` | 模式槽 1 = Manual/RC |
| `COM_FLTMODE4` | 部署值 `101` | 目标为模式槽 4 = External Mode 2 |
| `COM_MODE0_HASH` | `-301203826` | 预留 External Mode 1 槽 |
| `COM_MODE1_HASH` | `-696130346` | 固定 `Ground Station` 为 External Mode 2 |
| `COM_RC_IN_MODE` | 实机 `0`；HIL `2` | 实机 RC-only；HIL 保留 WebUI 飞手 |
| `SER_TEL1_BAUD` | `4000000` | TELEM1 与 ESP32 的 UART 波特率 |

当前固件使用 `MAV_0_CONFIG=0`，TELEM1 的 `/dev/ttyS0` 由 `gs_link` 独占；
`MAV_1_CONFIG` 默认是 `0`，不要把“串口 1”误认为 `MAV_1`。USB `/dev/ttyACM0`
仍由独立 MAVLink 实例提供 QGC 通道。

启动脚本使用 `param set-default NN_GS_EN 1`，因此全新参数配置默认允许 RC 监督下的
地面站输入；已经保存的显式用户值仍然优先。`NN_GS_EN` 只控制地面站输入路径，
执行器权威仍由独立的 `NN_LQI_OUTPUT_EN` 控制。

如果已有参数存储仍保存 `NN_GS_EN=0`，可执行 `param set NN_GS_EN 1`。`nn_control`
的参数更新周期为 1 s，外部 arming-check 更新周期为 300 ms；设置后应等待约
1.5 s 再确认 `nn_control status` 显示 `GS enabled: 1`。

启动脚本会按 `SYS_HITL` 设置 `COM_RC_IN_MODE`，设置 TELEM1 的 4 Mbaud，并在
`nn_control` 之前启动 `gs_link`。刷写后建议核对：

```text
param show COM_RC_IN_MODE
param show MAV_0_CONFIG
param show SER_TEL1_BAUD
param show NN_GS_EN
param show NN_GS_MODE_SLOT
param show NN_GS_TIMEOUT_MS
param show RC_MAP_FLTMODE
param show COM_FLTMODE1
param show COM_FLTMODE4
param show COM_MODE0_HASH
param show COM_MODE1_HASH
commander status
gs_link status
```

## 5. 旧原型传输层（MAVLink，已不再占用 ttyS0）

GS-Link v1 的正式帧格式和消息布局见
[`GS_LINK_PROTOCOL_V1_zh.md`](GS_LINK_PROTOCOL_V1_zh.md)。以下 MAVLink 内容只描述
旧过渡原型，当前固件不再用它占用 `ttyS0`：

| MAVLink 字段 | 范围 | 飞控含义 |
| --- | --- | --- |
| `y` | `-1000..1000` | roll，右滚为正 |
| `x` | `-1000..1000` | pitch，PX4 约定低头为正 |
| `r` | `-1000..1000` | yaw rate |
| `z` | `0..1000` | throttle，`500` 对应内部 `0` |
| `target` | `0` 或飞控 sysid | 目标系统 |

这里的 MAVLink 只用于验证链路和门控，不代表最终协议。GS-Link v1 的帧格式、
时间戳、CRC、频率和失联语义已经写入独立协议文档，当前阶段是评审和实现。

`MANUAL_CONTROL.z=0` 才是内部 `-1`（LQI 空闲/零集体）；`z=500` 是内部 `0`，
约为 50% 上桨集体，不是“零油门”。地面站初始化、待机和中断前必须发送 `z=0`；
只有确认起飞条件后才按斜坡增加 `z`。

## 6. 安全边界

以下约束保持不变：

- 实机只有 RC 能解锁/上锁；本子模式不以地面站消息作为解锁或模式切换依据；
- `actuator_armed`、lockdown、`failsafe_flags`、姿态/角速度/加速度新鲜度检查、
  输出有限性和 `NN_LQI_OUTPUT_EN` 全部继续生效；
- 地面站只能替换四轴归一化手动量，不能直接写执行器；
- RC 通道 8 的模式槽是硬门控；地面站无法自行改变飞行模式槽；
- RC 丢失时地面站输入立即失效，不会作为 RC 失效后的备用控制源；
- 地面站链路超时后必须回模式 1 再选模式 6，不能静默恢复控制。

GS-Link v1 只有 CRC，不提供链路认证或加密；能访问该 UDP 或串口链路的
人可能注入合法格式的消息。ESP32 所在局域网必须是受控网络，通道 8 仍是物理安全
开关。最终协议如需要认证、重放保护或多机隔离，应在协议层实现，不能削弱本门控。

## 7. 验证与发布门控

无桨、电机断开条件下：

1. 确认 GS-Link 已打开 TELEM1 的 4 Mbaud：

   ```text
   gs_link status
   ```

2. 校准并观察通道 8、六槽模式和 External Mode 注册：

   ```text
   listener rc_channels
   listener input_rc
   listener manual_control_setpoint
   listener manual_control_switches
   listener vehicle_status
   commander status
   ```

   `commander status` 应出现一个 `Ground Station` External Mode。切到模式 6 时
   `manual_control_switches.mode_slot` 必须为 4，`vehicle_status.nav_state` 必须为
   该 External Mode 的 nav_state；若模式槽是 6、但 nav_state 仍为 Manual，说明
   External Mode 未注册或模式槽映射不正确。

3. 观察协议解析后的 RC 输入和门控状态：

   ```text
   listener gs_link_control
   listener gs_link_status
   ```

4. 观察控制器状态：

   ```text
   nn_control status
   listener manual_control_switches
   ```

5. 按顺序验证：

   - 模式 1 中 arm，再拨模式 6：保持 100 ms 后进入；
   - arm 时已在模式 6：不能直接进入，必须先回模式 1；
   - 断开地面站：下一周期回 RC；
   - 地面站恢复但仍在模式 6：不能自动恢复；
   - 地面站恢复后先回模式 1，再选模式 6：可重新进入；
   - RC disarm：立即退出；`COM_DISARM_MAN=1` 时 manual-thrust 模式允许在台架
    直接上锁，不依赖位置估计；
   - 通道 8 从模式 6 切到模式 1–5：立即退出。

新增门控的主机测试：

```bash
bash px4_trans/tests/run_ground_station_gate_check.sh
```

只有完成无桨台架、HIL、失效注入和执行器映射验证后，才允许把
`NN_LQI_OUTPUT_EN` 置为 `1`。4 Mbaud、GS-Link v1 驱动和地面站位置环都完成验证前，
不得把本链路视为可飞行状态。

## 8. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-10-21 | 改为 RC 飞行模式槽 4；注册 Ground Station External Mode，离开槽或链路失效立即回 RC |
| 2026-10-20 | 固化 RC 监督状态机、实机 RC-only、通道 8 手势、超时退出和安全门控 |
| 2026-10-20 | 串口固定为 4,000,000 8N1；MAVLink 明确标记为待替换原型 |
| 2026-10-20 | 确定 GS-Link v1 二进制协议，ESP 独占串口 1，MAVLink 仅保留为过渡原型 |
| 2026-10-20 | 实现通信-only `gs_link`；遥测加入实际 RC 通道；飞行指令返回 UNSUPPORTED |
