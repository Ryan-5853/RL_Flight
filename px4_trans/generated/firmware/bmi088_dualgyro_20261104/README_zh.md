# BMI088 独占选源与双陀螺逐轴记录

适用机型：MicoAir H743-v2，22001 direct controller，2026-11-04 已核对的机上设备映射。

## 固件行为

机架启动脚本强制覆盖已保存的陀螺优先级：

| 参数 | 数值 | 含义 |
|---|---:|---|
| CAL_GYRO0_PRIO | 100 | BMI088（CAL_GYRO0_ID=6684690）唯一可选主陀螺 |
| CAL_GYRO1_PRIO | 0 | BMI270（CAL_GYRO1_ID=3604506）退出控制/估计选源 |
| CAL_GYRO2_PRIO | 0 | 排除其他陀螺来源 |
| CAL_GYRO3_PRIO | 0 | 排除其他陀螺来源 |

这里是 PX4 传感器选源优先级，不是两枚陀螺的加权平均。优先级 0 没有自动后备选源资格。没有修改复合估计器两架飞机之间的融合权重。

驱动继续发布两枚陀螺原始数据。PX4Gyroscope::update/updateFIFO 不读取 CAL_GYRO*_PRIO，因此 BMI270 不参与控制仍可诊断记录。其 VehicleIMU 积分状态可能因校准 enabled=false 而停止更新，不能将它误判为原始陀螺停止。

## 新增 ULog 记录

| topic | 实例 | 记录间隔 | 数据 |
|---|---|---:|---|
| sensor_gyro | 0、1 | 2 ms（最高 500 Hz，实际受发布率限制） | x/y/z，rad/s；device_id、温度、削顶计数 |
| sensor_gyro_fifo | 0、1 | 0（不人为限频） | 原生 FIFO x/y/z、samples、dt、scale |
| vehicle_imu_status | 0、1 | 100 ms | 可用实例的振动、削顶和方差状态 |
| sensor_selection | 0 | 100 ms | 实际使用的设备 ID |

保留原来的姿态、vehicle_angular_velocity、两枚 accel、执行器、RC、估计器状态和 bias。原始流采用非 optional 订阅，确保驱动晚于 logger 出现时仍可订阅。继续使用 SDLOG_MODE=2，从启动记到关机。

FIFO 只读取前 samples 个元素，角速度=整数样本×scale（rad/s），样本间隔 dt 的单位是微秒。原始 gyro 已经过驱动安装旋转，但未应用全部校准/机体调平处理；对比 vehicle_angular_velocity 前须处理零偏、板到机体变换与滤波。

两枚 gyro 都启用时 sensors_status_imu 的一致性指标可作旁证；本配置禁用 BMI270 选源后，该指标不再代表两枚间差异，必须从新录制的两路原始数据计算。

## 部署与核对

新 logger 配置编译在固件内。当前分支 initialize_logged_topics 使用固定 profile，单独调整 SDLOG_PROFILE 或写 logger_topics.txt 不会打开这组新订阅，需刷本次固件。

固件与 manifest 位于 `px4_trans/generated/firmware/bmi088_dualgyro_20261104/`。当前工作环境无可连接飞控串口，本次只完成构建，未刷入飞控。

刷入重启后，在 NSH 核对：

```sh
param show CAL_GYRO0_ID
param show CAL_GYRO1_ID
param show CAL_GYRO0_PRIO
param show CAL_GYRO1_PRIO
listener sensor_selection -n 1
listener estimator_status -n 1
listener sensor_gyro -i 0 -n 1
listener sensor_gyro -i 1 -n 1
listener sensor_gyro_fifo -i 0 -n 1
listener sensor_gyro_fifo -i 1 -n 1
logger status
```

实际 gyro_device_id 应为 6684690；raw 两路分别出现对应的设备 ID。实例按运行时 device_id 识别，不能只认 0/1 序号。

如果需要在旧固件先切换选源，可在已确认 disarm 且 ID 映射与上表一致时执行：

```sh
param set CAL_GYRO0_PRIO 100
param set CAL_GYRO1_PRIO 0
param set CAL_GYRO2_PRIO 0
param set CAL_GYRO3_PRIO 0
param save
reboot
```

这组命令只切换选源，不会给旧固件增加新的日志 topic。先固定台架核对电机运行时 roll/p，再检查 logger dropouts；实际 SD 带宽及两路 FIFO 完整性须用新 ULog 验证。

若以后需要恢复多陀螺自动后备，应同时修改启动脚本的强制参数设置，否则下次启动仍会恢复 BMI088 独占。
