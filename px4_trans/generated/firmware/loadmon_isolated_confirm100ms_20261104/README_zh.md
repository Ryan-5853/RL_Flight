# arm 附近 cpuload 停更：监控队列隔离与诊断

## 已确认的现象

用户记录显示 cpuload.timestamp 停在530464579 us，读取年龄增加到7.544秒，随后恢复到538616748 us。停更前后CPU约60%、RAM约62%。这证明发生了实际监控停更，而非仅QGC显示缓存；快照不能排除停更期间短时CPU/调度压力。

## 源码发现与本版处理

load_mon 原先运行于 wq:lp_default，与 ParamAutosave 及其他低优先级后台任务共享单个串行worker。参数自动保存包含持久化、校验和备份文件写入，可能因存储等待阻塞整个队列。已确认存在这种阻塞风险，但现有现场数据没有记录队列执行状态，因此不能唯一认定本次由哪一个任务或哪一项I/O引起。Commander在disarm时更新COM_FLIGHT_UUID也可触发参数自动保存，不能不经时间对齐就把所有停更归为arm本身。

本版将 load_mon 移到独立 wq:load_mon，保持与原lp_default相同的低优先级及2 Hz周期。新增约3500字节独立线程栈和相关调度开销，不提高它到控制任务优先级。参数自动保存与其他后台任务仍在原队列运行。

新增 load_mon status 诊断：

- cycle：单次Run耗时；
- run interval：工作项实际开始执行的间隔；
- publish interval：cpuload成功发布的间隔；
- idle counter unchanged：NuttX空闲计数不变、原采样逻辑提前返回的次数。

如果新固件仍停更，可据此区分：运行间隔很长（调度受阻）、运行耗时很长（Run内部阻塞）、运行正常但发布间隔长且idle counter计数上升（空闲统计不推进）。保留原cpuload计算和早退行为，不伪造新鲜时间戳，不关闭CPU/RAM检查，也不延长Commander的2秒超时。

紧凑ULog配置新增cpuload，500 ms记录间隔，便于与vehicle_control_mode和事件时间对齐。两路原始gyro/FIFO继续记录。

## 保留的飞行配置

- 默认自动陀螺选源；
- roll/pitch目标各±10°；
- 独立30°锁存终止，不增加确认等待；
- FD_FAIL_R/P=30°，FD_FAIL_R/P_TTRI=0.1秒；
- 既有执行器和控制链路。

## 部署与现场验证

本环境没有飞控连接，完成编译与镜像核对后未刷入，独立队列能否消除现场约8秒停顿仍需实机验证。

刷入重启后应看到：

```sh
load_mon status
```

输出包含 `running on dedicated wq:load_mon (2 Hz)`。

保持机体固定、无起飞测试，观察arm/disarm附近：

```sh
listener cpuload -n 20
```

正常时间戳约每0.5秒推进，不应越过2秒。随后读取：

```sh
load_mon status
work_queue status
logger status
```

保留新ULog。如果仍发生停更，应同时记录上述输出；独立队列隔离不能消除整个CPU饥饿、全局锁或底层采样统计不推进，需要按新增诊断进一步定位。
