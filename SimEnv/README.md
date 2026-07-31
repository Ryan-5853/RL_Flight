# SimEnv

张量化飞行器仿真环境接口框架。公开接口和参数定义见
[接口文档](docs/sim_env_interface.md)。

配置可从带完整中文注释且可直接运行的
[配置模板](configs/template.yaml) 开始；[示例配置](configs/example.yaml) 也已标注主要字段、单位和约束。

## 当前状态

已实现配置校验、参数随机化与 batch 化、mask 批量重置、实例隔离、观测、控制周期调度、完整动态状态恢复和分块日志。

日志支持两种显式模式：`full` 逐物理步保存全部字段，适合动力学调试与重放；
`compact` 通过 `physics_step_stride` 和 `fields` 降采样、裁剪，适合长时间训练。
初始化和状态恢复事件在两种模式下都会强制保存。`full` 模式把 mask reset 保存为
完整批次时间线帧；`compact` 模式则把 reset 后状态按被选中的实例稀疏保存到对应
reset snapshot，避免高并行训练因少量实例 reset 而反复复制整个批次。实际记录策略
写入批次 `metadata.json`。
动力学层已实现上下桨独立时间常数的一阶响应、含上下桨转速耦合项的总推力、独立平方反扭矩及其差值、舵机死区/回差/限速、四路推力分配、格栅衰减与耦合、偏置作用点力矩和六自由度刚体积分。传感器层已实现陀螺仪、加速度计和电机转速传感器的批量采样、零偏、确定性独立噪声、整数/线性插值延迟和 mask reset。动力学、执行器、传感器和 reset 数值路径均按批量张量运行。

仿真与控制统一采用固定 500 Hz 时基。每次 `advance()` 对激活实例恰好推进一个
2 ms 步：电机使用一阶系统精确离散，舵机使用带机械限速的分段解析解，六自由度
刚体使用中点积分和四元数指数映射。兼容字段 `physics_step` 与 `control_step`
在该模型中使用同一个 500 Hz 时钟；`physics_steps_advanced` 每行只会返回 0 或 1。

## 使用

```bash
python -m pip install -e .
```

```python
import torch

from simenv import SimulationEnvironment

with SimulationEnvironment.create(
    "configs/example.yaml",
    parallel_count=1024,
    device="cuda:0",
) as env:
    observation = env.observe("truth")
    control = torch.zeros((1024, 5), device=env.device, dtype=env.dtype)
    result = env.advance(control)

    # 保存全部数值状态；新建同结构环境后可从下一物理步精确继续。
    state = env.state_dict()

    # 任务失败的槽位批量换入新配置；其他槽位保持不变。
    reset_mask = ~result.valid
    reset_result = env.reset(reset_mask, "configs/example.yaml")

with SimulationEnvironment.create(
    "configs/example.yaml", parallel_count=1024, device="cuda:0"
) as restored:
    restored.load_state_dict(state)
    next_result = restored.advance(control)
```

`reset(reset_mask, config_path)` 接收与环境同设备的 `[B]` bool 张量。新配置会按完整批量 `B` 解析和随机化，并通过 mask 一次性替换选中行的参数与状态；未选中行保持不变。选中槽位获得新 UUID，时钟、控制量、有效状态和错误码恢复初始值。新配置必须保持批次频率、设备拓扑、传感器字段和查表 shape 不变。全 false mask 是不读取配置文件的空操作。

当前 `env.dynamics_implemented` 和 `env.sensors_implemented` 均为 `True`。

## 单环境 500 Hz 实时推理

实时演示使用独立的 `RealtimeSimulationEnvironment`。它固定 `B=1`、显式关闭
持久化日志、编译动力学和传感器内核，并复用扁平观测缓冲区。首次编译较慢，必须在
控制回路开始前调用 `warmup()`；预热完成后会恢复预热前的完整数值状态。

```python
import torch

from simenv import RealtimeSimulationEnvironment

torch.set_num_threads(1)

with RealtimeSimulationEnvironment.create(
    "configs/example.yaml",
    device="cpu",
) as sim:
    sim.warmup(steps=10)

    # 实际部署时替换成推理模型；输入为复用的 [1,N] 传感器向量，
    # 输出必须是同设备、同 dtype 的 [1,5] 或 [5] 控制量。
    control = torch.zeros((1, 5), device=sim.device, dtype=sim.dtype)

    def policy(observation: torch.Tensor) -> torch.Tensor:
        return control

    stats = sim.run_policy(
        policy,
        steps=5000,
        observation_source="sensor",
        realtime=True,
    )
    print(stats)
```

对单环境，仿真和策略应放在同一设备。小型策略通常优先使用 CPU；若策略必须使用
CUDA，则仿真也应创建在同一 CUDA 设备，避免每个 2 ms 周期进行 CPU/GPU 往返。
实时入口不写磁盘；需要录像或遥测时，应由回调把降采样状态发送给独立进程，不能在
500 Hz 仿真线程内同步写文件。可用下列命令先测纯仿真/空策略的延迟上限：

```bash
PYTHONPATH=src python scripts/benchmark_realtime.py \
  configs/example.yaml --device cpu --steps 5000
```

13980HX/RTX 4060 Laptop 的 CPU/GPU 选择、策略预热、线程隔离和验收方法见
[实时推理部署说明](docs/realtime_inference.md)。

物理标称参数只写在 SimEnv 配置的 `value` 中。训练层注入的
`dynamic_randomization` 不得重复声明 `baseline` 或使用 `distribution: fixed`；它只描述
围绕环境标称值的 `stddev` 或实际采样 `range`。不需要随机化的参数不应出现在规格中。

`state_dict/load_state_dict` 覆盖 truth/执行器状态、传感器输出、延迟环形缓冲、
参数、控制量、物理/控制步、valid/error/generation、实例 seed 和所有随机子系统计数器。
checkpoint 张量可以先加载到 CPU，再恢复到同结构的 CPU/CUDA 环境。batch UUID、实例
UUID 和日志线程不影响数值演化；恢复环境保留自己的新身份，并在时间线写入 resume 事件。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
