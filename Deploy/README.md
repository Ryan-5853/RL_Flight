# RL Flight Deploy

`Deploy` 将训练框架的完整 checkpoint 转换成小型、不可变、可校验的策略
bundle，并提供一个不依赖 TorchRL/TensorDict 的低延迟推理组件。

当前内置转换器支持本工程 Flight Train 的任意深度和宽度 MLP actor，包括：

- SAC 的 `loc + scale` 输出头：部署时只保留确定性 `loc`，再执行 `tanh`；
- PPO 的独立 `loc_network`；
- 任意 MLP 隐藏层数量和规模，结构从权重 shape 推导，不写死 `256×256`；
- 单帧或历史拼接输入，输入宽度从 actor 权重和 checkpoint 控制契约交叉确定。

`angular_acceleration_allocated_inner_loop_21d_v2` checkpoint 会由更高优先级的
`flight-train-angular-acceleration-cascade-v1` 专用适配器处理。除了确定性 MLP，
bundle contract 还会固化姿态 PID、角加速度估计、21 维字段表、61 帧历史、500 Hz
时基以及共轴差速/cyclic 的 3→5 动作分配。WebUI 可以据此重建完整级联控制器，
不需要读取 Train 配置或训练 checkpoint。专用 contract 同时携带规范化 SimEnv
兼容指纹，用于拒绝动力学、执行器或传感器语义不匹配的 WebUI session。

核心框架并不把 MLP 当作唯一网络。新的训练框架、Transformer、CNN、GRU 或其他
有状态网络通过 `CheckpointAdapter` 注册；bundle schema 已预留显式 state
input/output。首版 TorchScript 后端只接受无状态策略，遇到尚未支持的有状态网络会
拒绝导出，不会把它错误地当作 MLP。

## 安装与测试

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

也可以不安装，直接使用：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m flight_deploy.cli adapters
```

## 使用

查看 checkpoint 类型和转换器：

```bash
flight-deploy adapters
flight-deploy inspect /path/to/checkpoints/best_total_evaluation.pt
```

生成 bundle：

```bash
flight-deploy export \
  /path/to/checkpoints/best_total_evaluation.pt \
  artifacts/attitude_v4_best_total
```

角加速度级联模型使用同一个命令，适配器会根据 checkpoint 契约自动选择。也可以用
`--adapter flight-train-angular-acceleration-cascade-v1` 强制要求专用契约；checkpoint
不匹配时导出会直接失败。

产物结构：

```text
artifacts/attitude_v4_best_total/
├── manifest.json
├── model.pt2
├── model.ts
├── golden_vectors.pt
└── weights.pt
```

`model.pt2` 是当前 PyTorch ExportedProgram，`model.ts` 是兼容本地低延迟执行的
冻结 TorchScript 模型；`weights.pt` 是不依赖训练 checkpoint 的 canonical state
dict，用于未来重新生成其他后端。manifest 固化：

- checkpoint SHA-256、run、step 和算法；
- 通用输入、输出及显式状态 ABI；
- 实际网络层宽、activation 和输出变换；
- 训练控制契约、历史模式、归一化、action trim/scale；
- 每个产物的 SHA-256 和大小。
- 固定随机输入和期望输出；`verify` 会实际执行模型，而不只检查文件摘要。

验证及基准：

```bash
flight-deploy verify artifacts/attitude_v4_best_total
flight-deploy benchmark artifacts/attitude_v4_best_total \
  --device cpu --steps 30000 --warmup-steps 200

flight-deploy benchmark artifacts/attitude_v4_best_total \
  --backend torch_export --device cpu --steps 30000
```

Python 推理：

```python
import torch
from flight_deploy import PolicyRuntime

runtime = PolicyRuntime.load("artifacts/attitude_v4_best_total")
observation = torch.zeros(1, runtime.input_dim)
policy_action = runtime.infer(observation)
```

通用 runtime 接受已经预处理的 `[B,input_dim]` 张量。飞行控制特有的 21 维特征、
历史缓冲和物理动作变换保留在 bundle contract 中，并由外围组件组合。这样同一个
runtime 可以承载完全不同的网络和任务，而不会把某架飞行器的传感器语义写死在模型
执行器中。

## 扩展 checkpoint 和网络类型

实现 `flight_deploy.adapters.base.CheckpointAdapter` 的三个方法：

```python
class MyAdapter(CheckpointAdapter):
    name = "my-framework-transformer-v1"

    def probe(self, checkpoint): ...
    def describe(self, checkpoint): ...
    def convert(self, checkpoint, *, source_path): ...
```

`convert()` 返回 `ConvertedPolicy`：

- `module`：优化前的确定性 `torch.nn.Module`；
- `input_dim/output_dim`：数据平面 ABI；
- `architecture`：后端选择和资源估算所需的结构描述；
- `contract`：预处理、后处理、单位和时基；
- `state_inputs/state_outputs`：GRU、LSTM、Transformer cache 等显式状态。

然后把 adapter 注册到 `AdapterRegistry`。未来可再为同一个 canonical policy
增加 ONNX、ExecuTorch、TensorRT、CMSIS-NN 或自定义 C 后端，不需要修改 checkpoint
读取逻辑和上层推理组件。

独立 Python 包还可以声明 entry point，运行时会自动发现：

```toml
[project.entry-points."flight_deploy.adapters"]
my_transformer = "my_package.deploy:MyAdapter"
```

## 当前边界

- checkpoint 是 Python pickle 容器，只应转换可信训练产物；bundle runtime 不读取
  checkpoint。
- 大型 SAC checkpoint 使用 mmap 加载，避免将数 GiB replay tensor 全量复制到
  常驻内存，但首次扫描 SHA-256 仍需顺序读取整个文件。
- 当前未安装 ONNX/ONNX Runtime，因此首版不伪造 ONNX 产物；后端依赖和等价验证补齐
  后再启用。
- GRU schema 已预留，但需要实现 stateful TorchScript/ExportedProgram backend 后才
  允许生成可运行 bundle。
