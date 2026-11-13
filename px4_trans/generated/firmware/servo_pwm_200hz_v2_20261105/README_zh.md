# v2，仅将舵机PWM改为200 Hz

固件：`micoair_h743-v2_nncontrol_v2_servo200hz.px4`，未刷机。

基于最后一次飞行所用的v2控制参数；模型checksum保持 `0x30298daf`。
唯一执行配置变化为 `PWM_MAIN_TIM1=200`、`PWM_MAIN_TIM2=200`，启动时覆盖旧的50Hz存储值。两组对应M6/M7/M8，M5同组但未启用。

控制增益、配平、积分容量、yaw、输出限幅、PWM端点、DIS和保护均保持，未使用v3低带宽候选。
运行控制器没有额外的50Hz软件保持器；此前的50Hz保持是物理PWM周期及离线模型中的零阶保持。新PWM周期为5ms，控制计算仍为2ms。

如继续使用当前v2固件，无需为这个参数变化刷机。在未解锁状态执行并重启：

```sh
param set PWM_MAIN_TIM1 200
param set PWM_MAIN_TIM2 200
param save
reboot
```

重启后检查：

```sh
param show PWM_MAIN_TIM1
param show PWM_MAIN_TIM2
pwm_out status
```

两组参数应为200。实际脉冲周期可用示波器/逻辑分析仪确认约5ms；机械带宽仍取决于舵机及负载。哈希和检查结果见 `build_manifest.json`。
