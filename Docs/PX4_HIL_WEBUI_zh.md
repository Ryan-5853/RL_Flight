# PX4 LQI 与 WebUI 的 HIL 使用说明

本文说明本工程的 HIL（Hardware-in-the-loop，硬件在环）边界、协议和操作步骤。WebUI 中现有的 SimEnv、手柄、遥测和三维显示保持不变；控制律从本机 Python 控制器切换为真实飞控板上的 `nn_control` LQI。飞手输入可选浏览器手柄或飞控物理 RC 接收机，两者不同时争抢输入。

## 系统边界

```text
浏览器手柄 ──HTTP──> WebUI/SimEnv ──MAVLink 2──> PX4 nn_control
                                      ▲
物理 RC 接收机 ───────────────────┘（选 rc 时）
                                      <────────── 5 路执行器
                         │
                         └── 用执行器命令推进下一步 SimEnv
```

- SimEnv 是被控对象，固定 500 Hz。
- PX4 是控制器，`nn_control` 固定 500 Hz。
- WebUI 每步发送一份当前状态；选 `webui` 时同时发送飞手输入。只有收到已解锁的新执行器帧才推进仿真。
- 超时、断线或非法范围会让会话进入故障；旧执行器帧不会被复用。
- 默认 50 ms 收不到新执行器帧时，WebUI 会发送上锁命令并停止闭环。选 `webui` 时另有 250 ms 飞手输入超时；选 `rc` 时由 PX4 监测 RC 丢失。
- 首次建链单独提供最多 2 秒的有界预热/解锁窗口，并按 500 Hz 重发状态以启动 PX4 传感器/uORB 链；未确认解锁前 SimEnv 不会推进。
- 自动解锁且选 `webui` 时，握手期间仅向 PX4 发送“姿态杆居中、油门 -1”的安全解锁样本；收到 `SAFETY_ARMED` 后立即恢复飞手原始输入，该样本不会推进 SimEnv。选 `rc` 时不修改实体遥控器输入，飞手必须把油门拉到最低。
- 自动解锁会在有效杆量开始注入后先请求 PX4 `Manual` 模式，至少等待一个 Commander 更新周期再请求解锁。`nn_control` 不注册 PX4 External Mode；若 QGC 或上一次会话留下 `External 1`，PX4 会按设计以“Mode is not registered”拒绝解锁。
- 默认由 WebUI 通过 MAVLink `SERIAL_CONTROL` 打开 PX4 NSH，自动设置 LQI/HIL 参数、启动 `pwm_out_sim` 与 `nn_control`；500 Hz 执行器流由固件和 `MAV_CMD_SET_MESSAGE_INTERVAL` 配置，WebUI 不再重复下发 NSH 流命令；不再需要在 Windows QGC 和 WSL 之间切换。
- 建立新上位机会话时，WebUI 会先停止并重新启动 `nn_control` 与 `pwm_out_sim`，清除上一次 HIL 中断后遗留的回调状态和旧执行器时间戳；无需为此重启飞控。
- HIL 模式只启动 `pwm_out_sim`。Commander 的物理输出 lockdown 保持生效，因此不会驱动真实 PWM/DShot。
- HIL 直通 LQI 使用注入的最新姿态/角速度 uORB 数据做输入有效性判断；不会因为 EKF 的 `attitude_invalid` 或 `angular_velocity_invalid` 标志单独截断有效的直通控制输出。
- 控制器侧 MAVLink 杆量新鲜度窗口为 500 ms；WebUI 仍保留 250 ms 飞手输入超时，因此不会放宽上位机故障保护，只吸收 HIL 串口调度抖动。

## 协议 v6

全部使用 MAVLink common 标准消息，不维护私有 XML。

| 方向 | 消息 | 频率 | 契约 |
|---|---|---:|---|
| WebUI → PX4 | `HIL_SENSOR` | 500 Hz | FRD 加速度计比力（m/s²）和角速度（rad/s）；每 10 帧附带磁场与气压 |
| WebUI → PX4 | `HIL_STATE_QUATERNION` | 50 Hz | 真值姿态、NED 位置/速度，用于本轮绕过 EKF 的控制器 HIL |
| WebUI → PX4 | `MANUAL_CONTROL` | 500 Hz | 仅 `pilot_source=webui`；飞手 roll/pitch/yaw/throttle，上桨开度来自 throttle |
| WebUI → PX4 | `HEARTBEAT` | 1 Hz | GCS 链路存活 |
| PX4 → WebUI | `HIL_ACTUATOR_CONTROLS` | 500 Hz | `[上电机, 下电机, 舵机1, 舵机2, 舵机3]` |

参考系和量纲：

- 世界系：NED，`x` 北、`y` 东、`z` 下。
- 机体系：FRD，`x` 前、`y` 右、`z` 下。
- 四元数：Hamilton `q_wb=[w,x,y,z]`，将 FRD 机体系向量旋到 NED 世界系。
- 角速度：FRD，rad/s。
- 加速度：加速度计机体系非重力比力；`HIL_SENSOR` 按 MAVLink 契约使用 m/s²，`HIL_STATE_QUATERNION` 的整数字段使用 mG。
- 电机命令范围 `[0,1]`；舵机范围 `[-1,1]`。
- WebUI 手柄仍是四通道 `[-1,1]`。WebUI 的 Pitch 正方向是拉杆/抬头，而 MAVLink `x` 正方向是推杆，因此映射为 `x=-pitch`、`y=roll`、`r=yaw`、`z=(throttle+1)/2×1000`。这个负号只在协议边界出现一次。

当前 HIL 状态消息直接向 PX4 发布姿态和局部位置，因此它验证的是飞控板时序、uORB 接口、LQI、解锁/故障链和执行器映射，不包含 EKF 姿态估计误差。后续要验证完整估计器时，应切换为原始 IMU/磁力计/气压计/GPS 注入方案。

当前协议没有注入 ESC RPM，因而 LQI 在 HIL 中走已经定义的命令驱动电机状态观测器；真机 ESC RPM 反馈分支不属于本轮 HIL 覆盖范围。

## 首次准备飞控

编译并刷入本工程的专用固件：

```bash
cd /home/ryan_wsl/RL_Flight/px4_trans/px4
make micoair_h743-v2_nncontrol
```

编译产物为
`px4_trans/px4/build/micoair_h743-v2_nncontrol/micoair_h743-v2_nncontrol.px4`。
在 QGC 的“固件”页选择“高级设置→自定义固件”并指向该文件；本次
HIL_SENSOR 量纲修复位于飞控固件中，仅重启 WebUI 不会生效。
协议 v4 还将 `HIL_ACTUATOR_CONTROLS` 设为不受普通遥测带宽倍率压缩的
500 Hz 常速流，并在发送边界读取最新 `actuator_outputs_sim`；因此从
v3 升级也必须重新刷写固件。

通过 QGC MAVLink Console 执行一次：

```text
param set SYS_HITL 1
param set NN_LQI_OUTPUT_EN 1
param set COM_RC_IN_MODE 1
param set HIL_ACT_FUNC1 101
param set HIL_ACT_FUNC2 102
param set HIL_ACT_FUNC3 201
param set HIL_ACT_FUNC4 202
param set HIL_ACT_FUNC5 203
param save
reboot
```

`SYS_HITL=1` 必须在重启前保存。固件已经为 HIL 设置如下函数顺序：

```text
HIL_ACT_FUNC1=101   # 上电机
HIL_ACT_FUNC2=102   # 下电机
HIL_ACT_FUNC3=201   # 舵机 1
HIL_ACT_FUNC4=202   # 舵机 2
HIL_ACT_FUNC5=203   # 舵机 3
```

完成 HIL 后恢复真机模式：

```text
param set NN_LQI_OUTPUT_EN 0
param set SYS_HITL 0
param set COM_RC_IN_MODE 3
param save
reboot
```

## 启动 WebUI

安装上位机额外依赖（本工程 `.venv` 已安装时可跳过）：

```bash
cd /home/ryan_wsl/RL_Flight
.venv/bin/python -m pip install pymavlink==2.4.49 pyserial==3.5
```

启动现有 WebUI：

```bash
cd /home/ryan_wsl/RL_Flight/WebUI
../.venv/bin/python server.py
```

在“运行”配置页设置：

1. “控制器运行后端”选择 `px4_hil`。
2. “PX4 HIL 连接”填 `/dev/ttyACM0`（Windows 可填 `COM12`）；UDP 可填 `udp:127.0.0.1:14560`。
3. USB CDC/高速串口推荐 2,000,000 baud。500 Hz 双向协议不建议使用 115200。
4. 用浏览器手柄时将“HIL 飞手输入源”选为 `webui`；用直接接在飞控上的物理接收机时选 `rc`。WebUI 会分别设置 `COM_RC_IN_MODE=1` 或 `0`。
5. 保持“启动时自动解锁（仅 HIL）”勾选；现在默认开启，且暂停或关闭会话会发送上锁命令。
6. 保持“自动配置并启动 PX4 HIL”开启；USB 连接时“PX4 侧 MAVLink 设备”使用 `/dev/ttyACM0`。自动配置还会为这套直通状态 HIL 设置 `EKF2_EN=0`、`COM_ARM_WO_GPS=1`；真机飞行前必须恢复 `EKF2_EN=1`。
7. 点击原有“应用/开始”。

自动解锁仍受 PX4 Commander 的全部检查约束。若解锁失败，先在 QGC 查看 preflight reason；不要通过 force-arm 绕过检查。

## 故障判断

- 姿态/角速度方向、上下桨反扭力方向、格栅偏转方向的完整定义与硬件核对步骤见
  [`PX4_HIL_DIRECTION_CHECK_zh.md`](PX4_HIL_DIRECTION_CHECK_zh.md)。
- 若 auto start 一启动就出现 `Device ... is dead` 且 Windows/QGC 同时掉线，先看飞控 SD 卡根目录是否新增 `fault_*.log`。如果存在，说明飞控自身发生 HardFault，需要重新刷入包含 USB CDC 完成时序修复的固件（本次修改位于 NuttX `stm32_otgdev.c` 与 `cdcacm.c`）。
- “Preflight Fail: Crash dumps present on SD”：SD 卡根目录还有旧 `fault_*.log`，PX4 默认禁止带崩溃转储解锁。先确认本地已留档，再在 PX4 NSH 执行 `rm /fs/microsd/fault_*.log` 并重启飞控。
- “Preflight Fail: ekf2 missing data” 或 “Global position estimate required”：确认自动配置已把 `EKF2_EN` 设为 `0`、`COM_ARM_WO_GPS` 设为 `1`（这套 HIL 直接注入真值状态，不使用 EKF/GPS）；真机飞行前再恢复。
- 解锁后 `nn_control` 显示 `angular velocity missing/stale` 且执行器全零：这是 HIL 仿真 IMU 尚未被 `sensors` 模块接入的启动竞态。本工程固件已允许 HIL 解锁后继续扫描接入 IMU，WebUI 也会在首个闭环帧后给最多 2 秒沉降窗口；若仍复现，再检查 HIL_SENSOR 是否实际到达飞控（`sensors status` 中应出现 `Accel: 1310988`）。
- `pwm_out_sim status` 显示所有通道 `func: 0`、`actuator_outputs_sim publications: 0`：通常是上一个会话异常退出后飞控仍处于 armed（`Disarming denied: not landed`），而 MixingOutput 在 armed 状态下拒绝加载 `HIL_ACT_FUNC*` 映射。WebUI 现在会在建链开始时用 MAVLink force-disarm（21196）强制解锁复位，并将会话关闭/暂停也改为强制上锁；如仍出现，手动重启一次飞控即可恢复。
- `mavlink status` 显示 HIL_SENSOR 以 300 Hz+ 到达飞控，但 `sensors status` 中 `vehicle_imu` 间隔仍达几十/上百 ms：说明飞控内 IMU 管线在多次会话后退化。WebUI 现在每次建链都会重启 `sensors`，并设置 `SENS_IMU_MODE=1`、`IMU_GYRO_RATEMAX=1000`；若重启 WebUI 后仍复现，重启一次飞控让整条管线回到干净状态。
- 解锁后姿态迅速发散：先确认 WebUI 加载的是与 LQI 部署一致的 `SimEnv/configs/sim2real_micro_coaxial.yaml`（三路格栅 `deflection_axis_b` 为 `[-1,0,0]`、`[0.5,±0.866,0]`），不要混用 `template.yaml`/旧训练配置（三轴正好取反）。同时检查 `HIL_ACT_REV`：QGC 执行器校准可能在真机舵机反向时把该参数写成非 0，`MixingOutput` 会对 HIL 通道取反造成正反馈。WebUI 现在每次建链强制 `HIL_ACT_REV=0`，并在日志中打印 `param show HIL_ACT_REV`。
- 在线仿真详细日志：每次在线仿真（含 HIL）都会写入 `WebUI/runtime-runs/runtime-<session-id>.jsonl`，默认 100 Hz 记录输入、真值/传感器、参考、执行器命令、HIL 诊断、步进耗时与终止原因；会话结束或故障时追加 `summary` 事件最终存盘。频率由 `runtime.online_log_hz` 配置（`0` 关闭）。分析发散问题时先看该文件中 `truth.attitude_q_wb`、`truth.angular_velocity_b` 与 `controller.command` 的发散起始步。
- “PX4 is not in HIL mode”：确认 `SYS_HITL=1` 已保存并重启。
- “heartbeat timed out”：检查端口、权限、波特率以及该端口的 MAVLink 实例。
- WebUI 会把每次 HIL 建链、自动配置和首个执行器响应之前的操作写入
  `WebUI/runtime-runs/hil-startup-<session-id>.jsonl`，服务器终端同时打印
  `[PX4 HIL startup]` 摘要。会话创建失败时，HTTP/WebUI 错误文本会直接包含
  该文件的绝对路径。
- 每个 JSONL 事件包含 UTC 时间、单调时钟、`step`、操作说明、状态、耗时、
  串口是否仍存在、pyserial/pymavlink 健康状态及对应 NSH 命令的输出尾部。
  `failed` 事件表示故障在该操作期间**首次被观察到**；结合
  `last_completed_step` 可以确定上一项已完成操作。由于 USB 设备复位可能相对
  触发命令有短暂延迟，这比把下一次串口写失败直接归因于当前命令更准确。
- 首周期中，`first-cycle.receive` 只表示读到一帧原始 MAVLink 消息；事件会写明
  `received_message_type`、`received_mode` 和时间戳。真正可推进仿真的验收点是
  `first-cycle.actuator-boundary COMPLETED`，它要求消息为新鲜的
  `HIL_ACTUATOR_CONTROLS`，且 `mode` 包含 armed 位。等待期间每秒输出一次
  `PROGRESS`，汇总各消息类型、未解锁/陈旧执行器帧、空读和 HIL 重发次数。
- 自动解锁的 `COMMAND_ACK` 和相关 `STATUSTEXT` 会单独记录为
  `arming.feedback`。若首周期超时，`timeout-diagnostics.shell-command.NN`
  会逐条标出正在执行的 PX4 诊断命令；最终必有
  `first-cycle.startup-result FAILED`，运行线程也会在 server 终端打印
  `[runtime <session-id>] FAULTED` 及 JSONL 路径。因此终端最后一行可以直接区分
  “arm 指令已发出”“PX4 拒绝解锁”“只收到未解锁帧”和“USB 在某次写入时消失”。
- “HIL actuator timeout”：确认固件包含 `pwm_out_sim`、`HIL_ACT_FUNC1..5` 正确、`NN_LQI_OUTPUT_EN=1`，并使用足够带宽的链路。
- 启动超时时 WebUI 会继续发送安全的 HIL 传感器/杆量保活帧，同时采集 `sensors status`、`commander check/status`、`nn_control status`、执行器输出、`vehicle_status`、`health_report` 和 `failsafe_flags`；错误中的 `live timeout diagnostics` 之后才是有效的在线状态，不会再因诊断耗时产生假的 sensor stale。
- 若 `nn_control` 的 cycle 仍为 0，且 `sensors status` 显示角速度传感器未选中，先确认已刷入包含 500 Hz `HIL_SENSOR` 接收修复的本工程固件，再检查 USB MAVLink 是否正在接收该消息。
- “PX4 HIL remained disarmed”：先查看错误中的 `arming feedback`，其中会保留模式/解锁 `COMMAND_ACK` 及 PX4 健康状态文本；超时诊断还会自动附带 `vehicle_status`、`health_report` 和 `failsafe_flags`。物理 `rc` 模式必须将油门拉到最低；不要用 force-arm 绕过。
- 有执行器帧但全为零：通常是尚未解锁、LQI 输出开关为 0，或 Commander 正在执行安全抑制。运行 `nn_control status` 查看 inhibit reason。
