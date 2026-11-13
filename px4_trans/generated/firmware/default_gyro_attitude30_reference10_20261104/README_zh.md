# 更换飞控后的默认陀螺选源

本版撤销 BMI088 独占诊断策略。机架启动脚本只对 CAL_GYRO0/1/2/3_PRIO 执行 param reset，恢复 PX4 参数默认值；传感器校准代码再按内部/外部传感器默认优先级处理，由传感器健康状态和选源器决定实际主陀螺。内部 gyro 默认有效优先级为50，不承诺主源一定是某枚芯片。

启动时重置优先级，是为了同时清除旧诊断固件持久化的100/0配置。未重置 CAL_GYRO*_ID、零偏、加速度计校准或其他飞控参数；不使用旧板设备ID指定新板主源。由于每次启动都恢复默认，后续若需要保留人工优先级设置，也应修改这段启动策略。

保留上一版：

- 当前总距混合控制器 roll/pitch 目标各 ±10°；
- 本地总距混合控制器实际 roll/pitch 达到±30°的锁存终止与强制 disarm 请求；
- FD_FAIL_R/P=30°；
- 双 sensor_gyro、双 FIFO、sensor_selection 和现有诊断记录。

固件：`px4_trans/generated/firmware/default_gyro_attitude30_reference10_20261104/`。当前环境没有飞控串口，生成与检查固件后未刷机。

刷入后重启，用以下命令核对新板实际选源：

```sh
param show CAL_GYRO0_PRIO
param show CAL_GYRO1_PRIO
listener sensor_selection -n 1
logger status
```

只在旧固件 NSH 中 reset 优先级不能持久撤销旧策略，因为旧固件启动时会再次强制写入 BMI088 独占值。应刷本版再重启。logger 继续依赖SD剩余空间，记录状态以机上 logger status 为准。
