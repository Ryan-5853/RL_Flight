# RL Flight Controller

`Controller` 是 SimEnv 与具体控制律之间的统一批量抽象层。PID、LQR、混合控制器和
神经网络都接收相同的 `ControllerState + ControllerReference`，并返回可直接传给
`SimulationEnvironment.advance()` 的 `[B,5]` 物理命令。

## 公共契约

```python
from flight_controller import (
    ControllerContext,
    ControllerReference,
    ControllerState,
    create_controller,
)

context = ControllerContext(
    batch_size=env.parallel_count,
    device=env.device,
    dtype=env.dtype,
    control_dt=1.0 / control_hz,
    parameters=env.parameters,
)
controller = create_controller(
    {"type": "hybrid_pid_lqr", "params": {"collective_mode": "hover"}},
    context,
)

truth = env.observe("truth").values
state = ControllerState.from_truth(truth)
output = controller.step(state, reference)
result = env.advance(output.command)
controller.reset(reset_mask)
```

`ControllerState` 保留 SimEnv 的批量维，即使 `B=1` 也使用 `[1,...]`。控制器内部
积分、混合状态、循环网络 hidden 和上一动作都由同一个 mask reset 生命周期管理。

## 内置控制器

- `pid`：高度 PID、四元数姿态 PID/PD、偏航角速度控制和加权控制分配；
- `lqr`：高度 PID 加 10 状态离散姿态/执行器 LQR；
- `hybrid_pid_lqr`：电机启动和大误差阶段使用 PID，进入配平邻域后平滑切到 LQR；
- `neural`：兼容当前 Train `forward_step` MLP/GRU，也兼容普通可调用前馈网络；
- `module:factory`：配置可以引用自定义工厂，不必修改 SimEnv 或 WebUI。

神经网络 `output_mode` 支持当前四维残差动作 `residual_4`，以及直接输出五路物理
命令的 `physical_5`。部署层还可以提供 `coaxial_differential_cyclic_3`：三维策略
动作分别表示下桨差速和两个 cyclic 自由度，模型适配器负责将其映射为五路物理命令，
控制器则统一管理三维上一动作和 reset 生命周期。更复杂的 Transformer、时序卷积或
自定义观测编码器可以实现
`FlightController`，再通过 `register_controller()` 或 `module:factory` 接入。

## PID + LQR

默认参数见 `configs/hybrid_pid_lqr_hover.yaml`。控制器会从 SimEnv 当前参数自动：

1. 求解同时满足重力平衡和上下桨反扭矩平衡的悬停点；
2. 计算 `[T,Mx,My,Mz]` 对五路 PWM 的稳态控制效果矩阵；
3. 建立包含 roll/pitch、三轴角速度、双电机转速和三舵角的 10 状态模型；
4. 按实际 `control_dt` 精确离散，并求解离散 Riccati 方程；
5. 根据电机转速、姿态误差和角速度在 PID 与 LQR 间带滞环切换。

`collective_mode: hover` 使用高度 PID 保持 reset 时的位置；`manual` 则把外部
collective command 解释为上桨 PWM，并自动计算反扭矩平衡的下桨基准。

## 测试

```bash
PYTHONPATH=src:../SimEnv/src \
python -m unittest discover -s tests -v
```
