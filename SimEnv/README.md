# SimEnv

张量化飞行器仿真环境接口框架。公开接口和参数定义见
[接口文档](docs/sim_env_interface.md)。

## 当前状态

已实现配置校验、参数随机化与 batch 化、mask 批量重置、实例隔离、观测、控制周期调度和分块日志。
动力学层已实现共轴双电机一阶响应、舵机死区/回差/限速、四路推力分配、格栅衰减与耦合、偏置作用点力矩和六自由度刚体积分。传感器层已实现陀螺仪、加速度计和电机转速传感器的批量采样、零偏、确定性独立噪声、整数/线性插值延迟和 mask reset。动力学、执行器、传感器和 reset 数值路径均按批量张量运行。

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

    # 任务失败的槽位批量换入新配置；其他槽位保持不变。
    reset_mask = ~result.valid
    reset_result = env.reset(reset_mask, "configs/example.yaml")
```

`reset(reset_mask, config_path)` 接收与环境同设备的 `[B]` bool 张量。新配置会按完整批量 `B` 解析和随机化，并通过 mask 一次性替换选中行的参数与状态；未选中行保持不变。选中槽位获得新 UUID，时钟、控制量、有效状态和错误码恢复初始值。新配置必须保持批次频率、设备拓扑、传感器字段和查表 shape 不变。全 false mask 是不读取配置文件的空操作。

当前 `env.dynamics_implemented` 和 `env.sensors_implemented` 均为 `True`。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
