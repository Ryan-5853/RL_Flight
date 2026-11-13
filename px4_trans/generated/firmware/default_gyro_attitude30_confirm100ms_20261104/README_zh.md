# 30°姿态保护，标准故障确认100 ms

本版只将 Commander 的 roll/pitch 姿态故障确认时间从当前0.3秒改为0.1秒，并在机架启动时强制设置：

```
FD_FAIL_R = 30
FD_FAIL_P = 30
FD_FAIL_R_TTRI = 0.1
FD_FAIL_P_TTRI = 0.1
```

保留默认自动陀螺选源、双原始 gyro/FIFO 日志、各轴±10°目标限制，以及本地总距控制器独立的30°锁存停机保护。独立保护不增加确认延时，不等同于Commander的100 ms确认流程。

当前无可连接飞控串口，仅生成固件，未刷入机上。刷入重启后执行以下命令，核对实际生效参数：

```sh
param show FD_FAIL_R
param show FD_FAIL_P
param show FD_FAIL_R_TTRI
param show FD_FAIL_P_TTRI
nn_control status
```

如果只在当前固件即时修改确认时间，可在disarm状态执行：

```sh
param set FD_FAIL_R_TTRI 0.1
param set FD_FAIL_P_TTRI 0.1
param save
```

这只改变标准故障检测，不能给未包含独立保护的旧固件补上30°锁存停机逻辑。
