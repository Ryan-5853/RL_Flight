# 2026-11-06 动捕姿态诊断日志减负固件

固件：micoair_h743-v2_nncontrol_mocap_logging.px4。基于当前工作区构建，保留已有三轴动捕融合；本次只修改实际生效的日志主题列表，不修改控制/估计逻辑或已保存参数。已编译，未刷机。

此前定制initialize_logged_topics绕过默认主题集，导致默认列表中新增的动捕主题没有实际录入。本次直接修正定制列表，新增关键主题采用非optional订阅，以便发布器晚于logger启动时仍可记录。

| 内容 | 新频率/上限 |
|---|---:|
| vehicle_attitude（含姿态重置计数）、vehicle_angular_velocity | 200 Hz |
| actuator_motors、actuator_servos | 200 Hz |
| sensor_gyro实例0/1 | 各100 Hz，原先最高500 Hz |
| sensor_gyro_fifo实例0/1 | 停止记录；驱动和EKF采样不变 |
| vehicle_acceleration | 100 Hz |
| sensor_accel实例0/1 | 各50 Hz |
| vehicle_visual_odometry | 每个发布样本，当前源约50 Hz |
| estimator_aid_src_ev_att实例0/1 | 每次状态更新，当前源约50 Hz |
| estimator_status_flags实例0/1 | 最高50 Hz |
| estimator_status实例0/1 | 最高20 Hz，原先5 Hz |
| estimator_sensor_bias实例0/1 | 最高10 Hz，原先5 Hz |
| esc_status | 20 Hz，原先50 Hz |
| gs_link_control、gs_link_status | 最高50 Hz、10 Hz |
| logger_status | 1 Hz（包含dropouts、buffer_used_bytes等） |

保留原有RC、目标值、局部位置、电源、CPU和IMU健康状态。每个主题的实际频率受发布频率、调度和写盘影响。姿态/原始双陀螺仍支持本次低频晃动诊断；关闭FIFO后不再保留完整原生高频振动样本。

logging_budget.json按单EKF、50 Hz动捕以及其他主题达到配置上限估算数据记录约107.4 KiB/s（不含格式、参数、事件等开销）。这是配置预算，不是新固件实测值。旧日志打印约448 KiB/s；新配置目标约为其四分之一，实际写盘与dropout改善需复测。

刷入此固件并重启后，无需改变EKF或日志参数。当前分支使用固定列表，单改SDLOG_PROFILE或logger_topics.txt不会替代此配置。

核对：

```sh
logger status
listener vehicle_visual_odometry -n 1
listener estimator_aid_src_ev_att -n 1
listener estimator_status_flags -n 1
listener logger_status -n 1
```

启动输出应出现“200 Hz attitude/control, 100 Hz dual gyro, full mocap/EKF diagnostics; no FIFO”。
录制相同晃动动作，检查logger dropouts是否继续增长，并确认新ULog含以上三个动捕/估计关键主题和logger_status。
