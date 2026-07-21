# RL Flight Train

训练层采用 TorchRL 作为学习框架，以设备绑定的 `TensorDict` 贯穿环境适配、GRU
rollout、GAE 和 recurrent PPO 更新。reward、动作变换和姿态数学均为批量
`torch.Tensor` 运算；训练热路径不执行 CPU/NumPy 往返。

完整设计见 [TRAINING_FRAMEWORK.md](TRAINING_FRAMEWORK.md)。

## 当前实现

- SimEnv `create / observe / advance / masked reset` 适配；
- 21 维观测和 5 维标准动作/仿真命令变换；
- TorchRL `GRUModule`、`ProbabilisticActor`、`ValueOperator`；
- GPU 驻留的 `[B,T]` rollout；
- TorchRL `GAE`、`ClipPPOLoss` 和 recurrent sequence minibatch；
- JSON/YAML 配置、运行 manifest、标量记录和原子 checkpoint；
- reward、四元数、动作变换和 PPO 张量闭环测试。

SimEnv 当前动力学和传感器 kernel 仍是占位实现。示例配置仅用于接口冒烟，不能用于
判断控制器学习效果。

## V100 运行环境

Tesla V100 的计算能力为 `sm_70`。本项目固定使用 PyTorch 2.12.1 的 cu126 wheel；
不要安装 cu130，因为 CUDA 13 已移除 Volta 支持。驱动 535.288.01 满足 CUDA 12.x
的兼容要求，无需为本项目升级驱动，也无需另装 Conda `cudatoolkit`。

在全新 Conda 环境中安装：

```bash
conda create -n rl-flight python=3.10 pip -y
conda activate rl-flight

# 防止 ~/.local 中现有的 torch 2.13.0+cu130 混入 Conda 环境。
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate rl-flight

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install -e ../SimEnv -e .
```

`requirements.txt` 和 `pyproject.toml` 精确锁定 Python 依赖版本；cu126 wheel 的来源由
第一条 `pip install torch` 命令锁定。不要随后执行不带版本和索引的
`pip install torch`。

验证环境和 V100 架构支持：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("wheel CUDA:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("architectures:", torch.cuda.get_arch_list())

assert torch.__version__.startswith("2.12.1")
assert torch.version.cuda == "12.6"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 0)
assert "sm_70" in torch.cuda.get_arch_list()

x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
loss = (x @ x).square().mean()
loss.backward()
torch.cuda.synchronize()
print("CUDA forward/backward: OK")
PY
```

## 使用

环境安装完成后运行测试：

```bash
python -m unittest discover -s tests -v
```

运行 CPU 接口冒烟：

```bash
flight-train run --config configs/experiments/gru_ppo_smoke.json
```

正式训练配置应使用 `cuda:0`，并将
`run.allow_unimplemented_simulator` 设为 `false`。如果 CUDA 或真实仿真 kernel 不可用，
运行器会在采样前失败。
