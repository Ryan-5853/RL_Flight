# 总距混合控制器：30°停机保护与10°目标限幅

适用当前 airframe-1、NN_LQI_MODEL=2。本次保留 BMI088 独占陀螺选源和双陀螺日志。

## 限幅与保护

- 满杆 roll/pitch 目标从各 ±20°改为各 ±10°（0.1745329252 rad）；这是各轴欧拉角限制，不是合成倾角限制。yaw 角速度上限仍为 1.2 rad/s。
- 机架启动时强制设置 FD_FAIL_R=30、FD_FAIL_P=30，保留现有 Commander 确认时间（当前机上配置 0.3 秒）。
- 总距混合控制器新增独立保护：armed、单体本地输出、姿态新鲜且有效时，实际 roll 或 pitch 达到/超过 ±30°立即锁存终止。在 backend.run 之前撤销执行器输出，禁止本周期的倾斜补偿与总距 PI 再产生新的电机命令，并请求强制 disarm。
- 锁存后即使姿态恢复或切换模式，也不恢复输出，直到确认 disarm。运行时可在 nn_control status 中看到 `30 deg attitude termination (latched)`，首次触发输出错误提示。
- 独立保护只在本地总距混合控制器正常单体路径上新触发；没有改写组合控制器/AIRLINK 接管控制的失控策略。锁存后不能靠切换路径绕过。
- 无额外 0.3 秒确认等待；单次有效的新鲜姿态越过 30°就触发。这是本次避免等待标准故障标志期间继续增推力的明确行为。错误的大角度姿态估计也可能触发，因此仍须台架验证反馈质量。

## 总距反馈的坐标定义

输入 `vehicle_acceleration.xyz` 为校准、滤波后的机体 FRD 比力 f_b（单位 m/s²），不是已去掉重力的世界线加速度。q_wb 将机体旋转到 NED；R(q) 为对应旋转矩阵。

```
f_down = R31*fx + R32*fy + R33*fz
measured_up_acceleration = -(f_down + g)
```

- 静止且姿态正确时，无论倾斜多少，Rf_b=[0,0,-g]，向上加速度反馈为0。
- 向上加速2 m/s²时，反馈为+2。
- 使用三个机体分量投影，不是直接反馈 f_b.z，也不是把机体 z 除以倾斜余弦。
- 反馈再经约10 ms一阶滤波，与杆量对应的向上加速度目标做 PI；中位目标0，满量程目标约±5 m/s²。
- 最后由垂直需求计算总推力时，才另外除以 R33 的受限值以补偿倾斜。这是推力换算，不是测量值的坐标变换。
- 该环没有直接锁定高度或垂直速度。中位目标0加速度不保证立即停止已有的垂直速度；姿态估计错误也会污染投影和推力补偿。

## 生成与验证

部署配置：`px4_trans/configs/lqi_coaxial_collective_airframe1_manual_v3.yaml`；活动载荷与 manual-v3 manifest 已同步，model checksum 更新为 `0x6bdc0589`。

本工作区缺少 SimEnv 的 Python 源文件，无法从原始仿真重新构建 plant。为只改手动参考，生成器新增 `--reference-only-manifest`：先校验已有 payload 的 SHA-256、模型 float32 checksum 和 plant/controller/标定源文件哈希，保留已验证的全部增益及执行器模型，只更新参考参数并重写 golden/manifest。

复现：

```sh
python3 px4_trans/tools/generate_coaxial_collective_backend.py \
  --output-dir px4_trans/generated/airframe-1/manual-v3 \
  --reference-only-manifest px4_trans/generated/airframe-1/manual-v3/lqi_coaxial_collective_manifest.json
bash px4_trans/tests/run_lqi_core_check.sh
```

通过的检查：

- 新保护：roll/pitch 正负方向、不同 heading、四元数 q/-q 与归一化、锁存保持、切模式、disarm 清除；
- 实际参考适配器：输入超量程杆量后，roll/pitch 均为±10°；
- 实际总距核心：倾斜20°、叠加世界水平加速度0/5 m/s²时，垂直反馈仍分别正确得到0/+2 m/s²对应的滤波值；
- 现有 nominal、identified、coaxial 三套核心 golden vectors；
- 机架脚本语法、修改文件空白检查、固件完整编译。

最新 log_0_UnknownDate.ulg 的实际姿态回放中，新保护首次触发于启动后14.142536秒，原 failsafe 日志为15.079168秒，即回放判据提前约0.937秒。这是离线判据回放，不证明实机一定能在同一轨迹或物理延迟下停止；电机/转子还存在惯性。

固件与manifest位于 `px4_trans/generated/firmware/attitude30_reference10_20261104/`。本环境没有飞控串口，只完成构建，未刷入实机。

刷入重启后核对：

```sh
param show FD_FAIL_R
param show FD_FAIL_P
listener sensor_selection -n 1
nn_control status
logger status
```

应为30°/30°、BMI088 accel=6946834、gyro=6684690；日志需显示正在写入。最大目标角在编译后的模型载荷中，单改 QGC 参数不能替代刷写此载荷。
