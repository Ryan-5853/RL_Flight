# 地面组合飞行流程

2026-11-03。本文为当前地面组合任务的运行规定，覆盖旧 AIRLINK 文档中的空中拼接、
`AIRLINK_PAIR_EN` 选模及断链后单体降落条款。线上协议仍使用当前代码的 v2 ABI；
配置摘要格式升级到 4，旧策略固件不能与新固件配对。

## 配置与启动

两机接好电气接口后上电，使用同一遥控器的两个接收端。主从角色仍需分别设置：

```sh
# 主机
param set AIRLINK_ROLE 1
# 从机
param set AIRLINK_ROLE 2
```

角色默认 disabled；不自动推选主机。`AIRLINK_PAIR_EN` 保留兼容旧参数，已不参与选模。
直控机架启动脚本启动 AIRLINK、组合估计器、组合控制器和 nn_control，启用 CEKF。
执行器输出仍需满足既有 `NN_LQI_OUTPUT_EN`、机架匹配和 PX4 安全条件。

`Composite` 通过名称 hash 固定注册为 External Mode 3，`COM_MODE2_HASH=-70735788`。
三段开关固定映射：槽1 `COM_FLTMODE1=0`（Manual），槽4 `COM_FLTMODE4=101`
（Ground Station / External Mode 2），槽6 `COM_FLTMODE6=102`（Composite / External Mode 3）。
`NN_GS_MODE_SLOT=4`。启动脚本和 nn_control 启动时均纠正旧的已保存映射，注册 Ground Station
后只维护槽4，不覆盖组合模式的槽6。
两机须使用相同的模式映射。模式注册失败或落在其他 external 索引时不启用组合参考。

## 正常运行

1. 上电未建立首个会话时，等待连接，不把缺少对端当作断链故障。
2. 接好后自动进行心跳发现、REQUEST/REPORT/CONFIRM/ACK 握手和配置校验。
   成功后进入组合待命，两机保持 disarm。
3. 两机遥控器模式切到 External Mode 3，分别报告组合模式。
   模式切换本身不解锁。当前流程要求先 disarm、再选模式、最后解锁；不支持单体已解锁后的空中接管。
4. 飞手发出 arm，主机完成 fresh CONTROL_ENABLE 和 PAIR_ARM。
   从机本机检查通过、真正 armed 后返回成功，主机随后 armed。
   Commander 在实际 arm 入口检查本次事务授权，两机不能被同一个 RC arm 请求提前独立解锁。
5. 主机 nn_control 用组合体姿态和本机 RC 生成 C 系参考，comp_control 输出两机各五路命令。
   操纵符号、最大倾角、偏航速率和滤波沿用现有单体手动参考适配器。
   disarmed 时参考总距为 0。控制器参考无效或超时则撤销输出，不回到单体输出。
6. 主机执行本机命令，从机在 External Mode 3 下只执行 AIRLINK 命令；从机不使用本机
   roll/pitch/yaw/throttle 操纵量，持续上传 IMU、姿态、转速、电压和控制状态。
   会话、CRC、范围、新鲜度、PX4 安全门和本机 armed 状态仍须有效。

## 遥控器 disarm

两机始终各自响应遥控器 disarm，不依赖对端 ACK。任一端也可通过双向 PAIR_DISARM
通知另一端停机，接收端先 disarm 再 ACK；disarm 取消尚未完成的 enable/arm 事务。
正常 disarm 不锁存断链故障。链路正常、条件满足时，可以重新发起组合解锁。

## 意外断链与输出锁死

首次成功握手后开始监督，覆盖待命、解锁过渡、地面怠速、飞行及正常 disarm 后。

- 已建链的空闲状态使用 10 ms peer 帧新鲜度；组合激活时两方向关键数据流分别使用 10 ms。
- 单向断线由本端接收超时和对端心跳接收状态/故障锁存传播共同检测；两端独立采取动作。
- 对端 session 改变、意外断链或组合关键流失效后，立即撤销组合输出、强制本地 disarm，
  锁存故障。空中也停机，不运行 GS 恢复、单体控制或 SINGLE_LANDING。
- nn_control 发布无效执行器输出；Commander 强制撤防并阻止再次解锁。
  AIRLINK 故障不再设置永久 `manual_lockdown`；独立的遥控器 kill 开关仍保持原行为。
- AIRLINK 模块状态发布在成功建链后超过 10 ms 未刷新，也触发本机故障锁存。
- 通信恢复、切模式和自动撤防不会清除故障。两机各自收到物理遥控器明确的 disarm 动作，
  并确认已撤防且 RC 有效后，清除本机故障并开启新会话；两端重新握手后，新的 arm 动作可开始新任务，
  无需重新上电。遥控器已接到两机时一次 disarm 操作即可同时确认；任一端未收到该动作则仍禁止协同恢复。
- 此规则取代旧版协议文档中“故障必须两端重新上电”的约定；飞行中的 10 ms 断链检测保持不变。

10 ms 是检测窗口，不是两个飞控及驱动完全同时停机的保证；实际输出撤销还包含本地任务和驱动调度延迟。

## 验证

主机状态机测试覆盖：首次未连接、待命断链、空中断链、两个方向的单向断线、会话变化、
恢复后保持锁存、任一端正常 disarm 和重新授权。组合控制与单体 LQI 回归和目标固件构建另行执行。
建链后的链路测试同样受 10 ms 断链监督，使用默认 500 Hz；低频回环不能作为保持链路的替代。
这些检查不能代替双接收机、真实串口及输出驱动的无桨台架验证。

## GS-Link 组合输入与双机诊断（2026-11-03）

组合模式默认仍由主机 RC 操纵。GS-Link `DATA_RC.flags` 新增 bit2 `PairControl`：
只有主机、External Mode 3、已 armed、组合激活、本地 RC 有效、组合反馈有效且
`NN_GS_EN=1` 时，新的显式请求边沿才可替换主机四轴参考。从机不采纳 GS 操纵量。
普通心跳和监控不会请求控制。释放请求或 GS 超时回 RC；超时恢复必须先 request-off，
再 request-on。RC disarm 和 AIRLINK 断链锁死始终覆盖此路径。

新增 GS-Link DATA ID 0x10/0x11/0x12/0x13，分别上报本机与 AIRLINK/ULog 状态、
组合观测器、组合控制器、本机反馈与执行器命令；payload 为 108/120/204/292 B，
50 Hz，完整帧均小于 ESP 的 512 B 限制。每个 UDP 包对应一帧，不跨包拼接。
两机独立发送自己的诊断，主机额外发送权威组合闭环状态。

直控机架设置 SDLOG_MODE=2，两端从启动到关机持续记录。关键命令、执行器、组合反馈
以及角速度/加速度的记录间隔设为 2 ms，AIRLINK 和估计器健康状态为 20 ms。
GS-Link 上报 logger 的启用状态、写入量、速率、dropouts 与 message_gaps。
高频 ULog 留在飞控 SD 卡；上位机另记录 50 Hz 双机详细诊断和 100 Hz 位置环数据。
上位机位于 `/home/ryan/TeleaiTrans_dev/Dock_ground`，其 `docs/08-ground-pair-flight.md`
说明主从地址、C 系位姿转换、字段布局、共享日志与下载。当前工作区的 WebUI 不参与该链路。


## 自锁 ARM 开关（2026-11-04 修正）

本机架的按压式开关有稳定 ON/OFF 两态，按保持状态开关处理。
机架脚本将 COM_ARM_SWISBTN 强制设为0，覆盖旧保存值；RC_MAP_ARM_SW默认5，保留已有映射。
AIRLINK启用且已撤防时也纠正旧按钮配置。组合模式下物理RC ARM只启动主从解锁事务，
Commander不再同时尝试独立ARM；正常预检、陀螺偏置检查、事务授权和断链保护仍然生效。

AIRLINK直接读取新鲜 manual_control_switches 和有效RC：

- OFF确认撤防/取消未完成事务；实际撤防后清除本机故障并更新会话。
- 即使本机原本未armed，OFF也可以确认清锁，不要求先ARM成功。
- 持续OFF不会重复清锁或不断更新会话；ON保持不变不会自动重试。
- 上电/RC恢复时如果开关在ON，必须先OFF，再ON；丢失或无效RC不能产生ARM/清锁动作。
- OFF收到后拦截迟到的解锁ACK和本地ARM；映射开关在OFF时QGC/命令行不能请求组合解锁。
- 两机均应升级；确认OFF后等待重新READY，再切ON开始新事务。

`airlink status`新增 mapped/OFF/arm edge qualified/disarm ack时间，
`listener airlink_status`仍提供operator_disarm_timestamp和session_id。
QGC拒绝文本分别显示AIRLINK fault lock (RC OFF required)或transaction not authorized。
测试：tests/run_airlink_protocol_check.sh覆盖自锁开关、重复/过期输入、RC恢复、故障清除和原有双机会话回归。

两端RC的ON采样可以不同步：从机先收到PAIR_ARM、后收到本机新鲜ON时，
在现有ARM事务超时内等待本机ON确认；OFF、事务取消或超时立即拒绝，不在OFF时解锁。
HIL保持既有MAVLink模拟输入，不要求实体ARM开关。
