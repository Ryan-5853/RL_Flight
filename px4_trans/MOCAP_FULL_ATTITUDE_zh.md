# GS-Link 动捕三轴姿态融合

本次链路为 Dock_ground VRPN → GS-Link → vehicle_visual_odometry → EKF2。
地面站按 50 Hz 调度，仅发送新鲜的新样本，不排队、不重传，不依赖 deadman 或 RC 接管。
机载 IMU 继续高频预测，控制器继续使用 EKF 输出的 vehicle_attitude。

## 坐标与目标飞机

Dock_ground 的 q_body_to_world 是机体 FLU → 世界 FLU，wxyz 顺序。
在协议边界同时变换世界和机体系：C=diag(1,-1,-1)，R_frd=C R_flu C^T，
得到 q_frd=[w,x,-y,-z]。世界 X 没有地理北约束，因此发布 POSE_FRAME_FRD。
VRPN 刚体坐标必须已经与本机 FLU 对齐；任意标记架安装偏差仍需标定，不能靠静态归零代替。

普通飞行使用原始本机观测。组合飞行若 pose_frame=master，在转换为组合控制姿态前发送；
若 pose_frame=composite，按 pair_master_yaw_deg 的逆变换还原主机姿态。
本版本只发往主机 GS-Link。副机不能复用主机四元数，需要独立刚体或已标定的副机安装外参。
仿真源不会发送动捕姿态。

## 协议增量（GS-Link v1，保留原消息布局）

均为小端，普通 GS-Link 头和 CRC 不变。

DATA 0x03，flags=0，36 字节 payload：

| 偏移 | 类型 | 字段 |
|---|---|---|
| 0 | uint32 | 地面站 heartbeat session_id |
| 4 | uint32 | 动捕 sample_sequence |
| 8 | uint64 | 映射到飞控 hrt 时钟的 sample_time_us |
| 16 | float32[4] | 本机 FRD → 本地 FRD 四元数，wxyz |
| 32 | float32 | 各向同性旋转向量方差，rad² |

每帧共 48 字节，50 Hz 时净协议流量约 2.4 kB/s，不含 UDP/Wi-Fi 开销。
只发布姿态；位置、速度、角速度及其方差均设为 NaN。

DATA 0x04 时间同步：请求 flags=0、payload=uint64 ground_send_ns（8 字节）；
回复 flags=Response(8)、payload 为 ground_send_ns、fc_receive_us、fc_transmit_us（三个 uint64，共24字节）。
地面站使用四时间戳 midpoint offset，在10秒窗口中选择最小 RTT 交换；只接受往返≤20 ms的匹配回复。
启动时10 Hz同步，建立后1 Hz保活，3秒无有效回复则停止姿态发送。飞控重启会话后重新同步。
该方案不能消除网络路径不对称；时钟误差上界约为 RTT/2。飞控容忍最多10 ms未来时间并夹到接收时刻。
源时间可信时扣除源端龄期，否则使用地面站接收时间，动捕采集到地面站这一段仍需实测。

地面站丢弃超过50 ms的源样本及重复样本。飞控拒绝超过100 ms的观测、重复/乱序序号、
非递增采样时刻、错误会话、无效四元数和方差，以及心跳过期时的观测。

## 地面站配置与可观测性

flight 下新增可选项，默认值如下（旧配置无需修改）：

```json
"mocap_attitude_enabled": true,
"mocap_attitude_noise_rad": 0.05
```

发送频率固定50 Hz；控制线程/动捕源低于50 Hz时，以可用新样本为限，不补发积压帧。
状态 JSON 的 gs_link.mocap_attitude 包含 time_sync_valid、time_sync_rtt_ms、time_sync_age_ms、tx_packets。
旧固件不回复时间同步时，地面站不会发出新的姿态观测，原 RC/遥测链路保持兼容。

## 飞控参数

升级包含本改动的固件后，在地面设置并保存：

```sh
param set EKF2_EV_CTRL 16
param set EKF2_EV_NOISE_MD 0
param set EKF2_EVA_NOISE 0.05
param set EKF2_EV_DELAY 0
param set EKF2_MAG_TYPE 5
param save
```

随后重启。EV_CTRL bit4=完整姿态，优先于bit3的旧yaw更新，不重复融合yaw。
这里的MAG_TYPE=5关闭动捕室内的磁融合；任意航向FRD观测与GNSS位置/速度融合同时活动时，完整姿态融合会拒绝启动。
本次没有改写已保存参数，也没有自动刷写飞控。参数默认仍为未启用外部融合。

0.05 rad是初始观测标准差（约2.86°），消息携带其平方；EKF2_EVA_NOISE仍是下限。
有时间同步后EV_DELAY只补偿未包含的动捕采集/滤波延迟，应通过动态日志对齐实测。
若源时间戳不可信，不能把EV_DELAY=0理解为整条观测链路零延迟。

## 融合与检查

使用局部系左乘误差 Log(q_pred q_obs^-1)，与PX4现有误差状态约定一致；
对三个姿态误差状态计算中心差分观测雅可比，进行三轴顺序Joseph协方差更新。
重用现有measurementUpdate，因此姿态修正可通过交叉协方差影响陀螺仪偏置等状态。
在地面初次启用FRD融合时对齐yaw；空中不因动捕重置或重新接入直接重置姿态。
每个样本先做三轴创新门限检查，超过30°旋转差拒绝整帧。
持续失败200 ms或数据中断400 ms后停止外部姿态融合；IMU预测继续。

```sh
gs_link status
listener vehicle_visual_odometry
listener estimator_aid_src_ev_att
listener estimator_status_flags
```

观察time_last_fuse持续推进、innovation_rejected=false、test_ratio合理；cs_ev_yaw在本模式下代表完整外部姿态拥有航向辅助权。
新增estimator_aid_src_ev_att和vehicle_visual_odometry默认日志，供检查三轴创新与掉线行为。

验证包含：EKF外部视觉11项测试（新增三轴收敛、符号等价、异常跳变拒绝、掉线及yaw互斥），
GS-Link协议/观测门控主机测试，以及Dock_ground包含真实UDP往返时间同步的5项CTest。
目标micoair_h743-v2_nncontrol固件完成编译。尚未进行实体机链路、安装外参或飞行验证。


## 2026-11-06 日志修正

此前默认主题集的新增项被定制initialize_logged_topics绕过。现已在实际列表直接订阅
vehicle_visual_odometry、estimator_aid_src_ev_att、estimator_status_flags和logger_status，
并关闭双FIFO日志、将双原始gyro降为100 Hz。200 Hz姿态/角速度记录保留。
完整频率表与独立固件见[日志减负固件说明](generated/firmware/mocap_logging_20261106/README_zh.md)。


## 2026-11-06 提频与延迟复测

地面站默认发送上限现为100 Hz，可用flight.mocap_attitude_rate_hz配置10–100 Hz，
不超过control.rate_hz。前文50 Hz为初版配置。配套建议EV_DELAY=10 ms、PREDICT_US=5000，
保存后重启；没有自动写入实体机。参见[参数与回退说明](generated/config/mocap_100hz_20261106/README_zh.md)。
