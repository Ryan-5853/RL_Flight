# 500 Hz 单环境实时推理

## 推荐部署拓扑

目标硬件为 Intel Core i9-13980HX 与 RTX 4060 Laptop GPU。默认推荐把单环境仿真和
小型策略网络都放在 CPU 的同一个性能核上：

```text
P-core：策略推理 → 500 Hz 仿真 → 发布最新状态
其他核：遥测、日志、网络和 GUI
GPU：渲染或其他批量任务
```

原因是单环境状态很小，CPU/GPU 每 2 ms 往返的固定同步成本通常高于数值计算本身。
若策略网络足够大、CPU 推理使 P99 超过预算，则把策略和仿真一起放在
`cuda:0`，不要采用“GPU 策略 + CPU 仿真”的逐步同步组合。

无论使用哪种设备，都应插电运行并固定高性能电源模式。13980HX 同时包含 P-core 和
E-core；实时进程应固定到一个空闲 P-core，日志和渲染固定到其他核。CPU 编号因 BIOS、
操作系统和超线程设置而异，应先用 `lscpu -e=CPU,CORE,SOCKET,MAXMHZ,ONLINE` 确认，
不要照抄固定的 `taskset` 编号。

## CPU 推理

```python
import torch

from simenv import RealtimeSimulationEnvironment

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

policy = load_policy().eval().to("cpu")
policy = torch.compile(policy, fullgraph=True, mode="reduce-overhead")

with RealtimeSimulationEnvironment.create(
    "configs/example.yaml",
    device="cpu",
) as sim:
    # 同时编译策略、动力学、传感器、finite 检查和状态提交；
    # 预热产生的状态和随机计数器会在返回前完整恢复。
    sim.warmup(
        policy=policy,
        steps=10,
        observation_fields=("gyro", "accelerometer", "motor_speed"),
    )
    stats = sim.run_policy(
        policy,
        steps=30_000,
        observation_fields=("gyro", "accelerometer", "motor_speed"),
        realtime=True,
    )
    print(stats)
```

策略必须接收 `[1,N]` 张量并返回 `[1,5]` 或 `[5]`。控制顺序为上下电机及三个舵机；
电机范围是 `[0,1]`，舵机范围是 `[-1,1]`。策略输出层应显式满足这些范围，例如分别
使用 sigmoid 和 tanh，而不应依赖仿真器静默裁剪。

## GPU 推理

```python
device = torch.device("cuda:0")
policy = torch.compile(
    load_policy().eval().to(device),
    fullgraph=True,
    mode="reduce-overhead",
)

with RealtimeSimulationEnvironment.create(
    "configs/example.yaml",
    device=device,
) as sim:
    sim.warmup(policy=policy, steps=20)
    stats = sim.run_policy(
        policy,
        steps=30_000,
        realtime=True,
        synchronize_cuda=True,
    )
```

`synchronize_cuda=True` 会在每步统计完整 GPU 完成延迟，数据真实但同步本身也有成本。
关闭它只测量 CPU enqueue 时间，不能据此宣称满足 2 ms deadline。正式部署前应使用
开启同步的统计完成压力测试。

RTX 4060 Laptop 的实际性能受整机 TGP、Dynamic Boost、温度和独显直连模式影响，
不能只凭 GPU 型号判断。应在目标笔记本插电、达到稳定温度后，分别运行 CPU 与 CUDA
版本至少 60 秒，以 deadline miss 和 P99 而不是平均 Hz 选择设备。

## 日志与可视化

实时入口刻意不创建持久化日志，也不会启动日志线程。推荐让 500 Hz 线程只发布最新
状态或写入预分配 ring buffer：

```text
500 Hz 仿真/策略线程
    ├── 2 ms deadline
    └── 无锁或短临界区发布最新状态

50–100 Hz 遥测线程
    └── 降采样写日志/发送网络数据

30–60 Hz GUI/渲染线程
    └── 读取最新状态，不阻塞仿真
```

`run_policy(callback=...)` 的 callback 位于实时线程中，因此只能做常数时间、非阻塞的
内存发布；禁止在 callback 中写文件、打印、执行网络请求或等待锁。

## 验收

先测空策略的仿真上限：

```bash
PYTHONPATH=src python scripts/benchmark_realtime.py \
  configs/example.yaml --device cpu --steps 30000
```

再用 `--paced` 测量真实 500 Hz 调度：

```bash
PYTHONPATH=src python scripts/benchmark_realtime.py \
  configs/example.yaml --device cpu --steps 30000 --paced
```

最终必须换成真实策略调用 `run_policy()`，连续测试至少 30,000 步（60 秒）。建议记录：

- P50、P99 和最大 compute latency；
- deadline miss 数量与最大 lateness；
- CPU 频率、温度和是否发生功耗/热降频；
- GUI、遥测和日志同时开启时的结果。

普通 Linux、Python 和 PyTorch 提供的是软实时能力。若验收标准要求任何情况下都不得
错过 2 ms deadline，需要进一步使用实时调度、锁内存、隔离 CPU，或把最终数值内核和
策略迁移到原生实时进程。
