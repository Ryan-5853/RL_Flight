# 强化学习控制器训练框架

## 1. 文档目的

本文定义 `Train` 层的模块边界、公开接口、配置结构、运行目录和复现契约。实现者应能在不修改训练主流程的前提下替换仿真环境、任务、网络、强化学习算法、评估方案或记录后端。

训练层的唯一入口是：

```bash
python -m flight_train run --config configs/experiments/gru_ppo.yaml
```

除命令行明确允许的运行时覆盖项外，所有会影响实验结果的内容均来自配置文件并写入运行目录。框架不通过 Python 全局变量、隐式默认值或手工修改源码定义实验。

本文基于以下外部契约：

- 控制目标：以循环神经网络替代传统飞控的 controller + allocator，输入目标姿态和飞行器观测，输出两路电机及三路舵机归一化命令。
- 仿真接口：`SimulationEnvironment.create / observe / advance`；批量维始终为第一维，控制周期为一次 `advance`，单实例故障不得影响其他实例。
- 默认时基：物理仿真 5 kHz，控制与网络推理 500 Hz，每次动作保持 10 个物理步。
- 当前策略输入：姿态四元数 4、角速度 3、加速度 3、电机转速 2、目标姿态 4、上次动作 5，共 21 维。
- 当前策略输出：左/右电机 `[0,1]`，三个舵机 `[-1,1]`，共 5 维。

> 21 维输入是首版实验约定，不是写死在模型中的常量。最终维度必须由观测配置和环境元数据推导并在启动时校验。

## 2. 设计原则

1. **配置即实验定义**：模型、任务、奖励、随机化、算法、种子、训练预算和记录策略全部配置化。
2. **依赖方向单一**：运行器依赖抽象协议；任务不导入算法；模型不调用环境；环境适配器不计算奖励。
3. **批量和掩码优先**：所有逐步接口保留 `[B,...]`，用 `active / valid / terminated / truncated` 掩码表达实例状态，不按实例写 Python 循环。
4. **循环状态显式**：RNN hidden state 是训练状态的一部分，初始化、重置、截断、回放和 checkpoint 均有明确语义。
5. **训练与部署同源**：归一化、特征顺序、动作变换和网络前向路径只有一个实现，导出时不得重写一份近似逻辑。
6. **复现优先于便利**：每次运行固化解析后的完整配置、代码身份、依赖、实际随机参数、随机数状态和产物校验和。
7. **完整记录且可控开销**：训练级指标、控制级轨迹和仿真层物理级日志分层保存；任何降采样或关闭都必须显式配置并记录，不允许静默丢数据。
8. **快速失败**：配置、shape、单位、时基、设备、动作范围或恢复点不兼容时，在开始采样前报错。

## 3. 职责边界

```text
CLI / ConfigLoader
        |
        v
   ExperimentRunner ---------------- ArtifactStore / EventRecorder
     |       |       |       |                    |
     v       v       v       v                    v
  EnvAdapter Task  Policy Algorithm         run directory
     |               |       |
     v               v       v
 SimulationEnv    torch.nn  TorchRL primitives
```

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `ConfigLoader` | 组合配置、schema 校验、插值、计算配置指纹 | 创建环境、设置奖励 |
| `EnvAdapter` | 将仿真接口转换为统一的控制步接口、维护上次动作、整理 truth/sensor 字段 | 目标生成、奖励、网络 hidden state |
| `Task` | 目标命令、初始条件、观测特征、奖励、终止与课程学习 | 推进物理仿真、优化网络 |
| `Policy` | 特征到动作分布/动作的映射、RNN 状态转移、部署前向接口 | 奖励、采样预算、写日志 |
| `Algorithm` | loss、优势/目标值、优化器、更新调度和算法状态 | 直接依赖具体仿真类 |
| `Collector` | 按策略与任务采集 TensorDict、维护 episode/RNN 边界 | 定义 loss |
| `Replay/TrajectoryStore` | 序列化存储与抽样，保存 burn-in 和有效掩码 | 解释奖励语义 |
| `Evaluator` | 确定性评估、固定场景集、验收指标 | 改变训练状态 |
| `Recorder` | 指标、事件、轨迹、配置和系统信息持久化 | 决定训练逻辑 |
| `CheckpointManager` | 原子保存、恢复、保留策略、完整性校验 | 猜测不兼容状态如何迁移 |

依赖方向应为 `runner -> protocols <- implementations`。`flight_train.core` 只能包含协议和通用数据类型，不得导入 `sim_env` 或 TorchRL 的具体算法类。

## 4. 建议目录

```text
Train/
├── TRAINING_FRAMEWORK.md
├── pyproject.toml
├── configs/
│   ├── schema/                    # 配置 schema 及版本迁移
│   ├── environment/               # 仿真配置引用和训练侧适配配置
│   ├── task/                      # 姿态跟踪、悬停、课程学习
│   ├── model/                     # MLP、GRU、RMA
│   ├── algorithm/                 # PPO、SAC、TD3
│   ├── logging/                   # 完整记录策略
│   └── experiments/               # 可直接运行的完整实验
├── src/flight_train/
│   ├── cli.py
│   ├── config.py
│   ├── core/                      # Protocol、StepBatch、错误类型
│   ├── envs/                      # SimulationEnvironment 适配器
│   ├── tasks/                     # 目标、特征、奖励、终止
│   ├── models/                    # 策略/价值网络和导出前向
│   ├── algorithms/                # TorchRL 算法封装
│   ├── collectors/
│   ├── replay/
│   ├── evaluation/
│   ├── recording/
│   └── runner.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── reproducibility/
└── runs/                          # 生成物，不纳入版本控制
```

## 5. 最小公开接口

接口以 Python `Protocol` 和 TorchRL `TensorDict` schema 表达。仿真层和任务层只生产通用控制步字段，不依赖某个算法的专用 loss。

### 5.1 控制步数据

控制步使用 `TensorDict(batch_size=[B], device=device)`，必需 key 为：

```text
observation       [B, observation_dim]
reward            [B, 1]
terminated        [B, 1] bool
truncated         [B, 1] bool
done              [B, 1] bool
valid             [B, 1] bool
is_init           [B, 1] bool
episode_id        [B] int64
episode_step      [B] int64
info/*            [B, ...]
```

collector 将其组织为 TorchRL 规范的 `[B,T]` transition TensorDict：当前状态字段位于根，转移结果位于 `("next", ...)`。TensorDict 的 `device` 必须等于环境执行设备。

必须区分：

- `terminated`：状态确实终止，不 bootstrap。
- `truncated`：例如达到 30 s 上限，价值目标通常允许 bootstrap。
- `valid=False`：仿真数值失败或非法控制；冻结该实例，记录错误，随后由适配器按配置重置。该步不得作为正常训练样本。

### 5.2 环境适配器

```python
class BatchedControlEnv(Protocol):
    @property
    def spec(self) -> EnvSpec: ...

    def reset(
        self,
        mask: torch.Tensor | None = None,
        *,
        scenario: TensorDict | None = None,
    ) -> TensorDictBase: ...

    def step(
        self,
        action: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> TensorDictBase: ...

    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...
```

`EnvSpec` 至少包含 observation/action 的字段名、shape、dtype、device、上下界、单位、字段顺序、`physics_hz`、`control_hz` 和 `parallel_count`。

适配器每个控制步严格执行：

1. 校验动作 `[B,5]`、有限性和范围；策略标准动作先经唯一的 `ActionTransform` 转为仿真命令。
2. 调用 `SimulationEnvironment.advance(action, active_mask)` 一次。
3. 按任务配置分别读取 `truth`、`sensor`；不得通过 `info` 偷渡未声明真值给 student。
4. 由 `Task` 计算下一观测、奖励和结束标志。
5. 保存实际送入环境的 `previous_action`，用于下一步 21 维观测。

当前仿真实现已提供 `reset(reset_mask, config_path)`，适配器应直接使用该 GPU bool mask 独立重置完成、截断或失效的实例。重置属于稀疏 episode 边界；控制步热路径中的观测、动作、reward、RNN state 和 rollout 不得为判断单实例状态而转到 CPU。

### 5.3 任务

```python
class Task(Protocol):
    def reset(self, env_view: TensorDict, mask: torch.Tensor) -> TensorDict: ...
    def build_observation(self, env_view: TensorDict) -> TensorDict: ...
    def transition(
        self,
        previous: TensorDict,
        action: torch.Tensor,
        current: TensorDict,
    ) -> TaskTransition: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

首个任务 `attitude_tracking`：初始姿态随机、单个 episode 内目标姿态固定、时长默认 30 s（15000 个控制步）。目标四元数必须归一化；姿态误差建议使用四元数最短弧或等价的旋转向量，奖励中不能直接对 `q` 和 `-q` 给出不同结果。

### 5.4 循环策略

```python
class RecurrentPolicy(Protocol):
    def initial_state(self, batch_shape: torch.Size, device: torch.device) -> TensorDict: ...

    def forward_step(
        self,
        observation: TensorDict,
        recurrent_state: TensorDict,
        reset_mask: torch.Tensor,
        *,
        deterministic: bool,
    ) -> PolicyStep: ...

    def forward_sequence(
        self,
        sequence: TensorDict,       # [B,T,...]
        initial_state: TensorDict,
        reset_mask: torch.Tensor,   # [B,T,1]
    ) -> PolicySequence: ...
```

`PolicyStep` 必须同时返回动作分布所需参数、标准动作、实际命令和下一 recurrent state。训练时保存的是生成该动作前后的 hidden state 或足以通过 burn-in 精确重建它的数据。

RNN 规则：

- `reset_mask = terminated | truncated | ~valid | explicit_reset`；在新 episode 第一条观测送入网络前清零 hidden state。
- PPO rollout 可以按 `sequence_length` 截断 BPTT，但截断边界不等于 episode 边界，不能清零 hidden state。
- SAC/TD3 的 replay 以连续序列抽样，包含 `burn_in + learn_length`；burn-in 只重建 hidden state，不计算优化 loss。
- padding 样本必须有 `loss_mask=False`，不得影响归一化统计、优势估计或 loss。
- hidden state、reset mask、序列起点和 episode id 均进入 checkpoint/轨迹记录。

### 5.5 算法

```python
class Algorithm(Protocol):
    def act(self, batch: TensorDict, recurrent_state: TensorDict, *, explore: bool) -> PolicyStep: ...
    def update(self, data: TensorDict, global_step: int) -> Mapping[str, torch.Tensor]: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

学习层使用 TorchRL：以 `TensorDict` 作为采样与训练的数据总线，循环网络使用 `GRUModule`，优势估计与 PPO 目标分别使用 `GAE` 和 `ClipPPOLoss`。环境动力学和任务 reward 保持纯张量 kernel，不依赖具体算法。首版先完成 PPO + GRU，因为 on-policy 序列边界更容易验证；SAC/TD3 必须等序列 replay、burn-in 和 recurrent target network 测试完善后再启用。

训练热路径中的 TensorDict 必须绑定运行设备；rollout、优势、value target、loss 和权重不得为了 minibatch 整理或统计计算下 GPU。只有 checkpoint、完整轨迹持久化和低频标量指标记录可以在显式 I/O 边界执行设备到主机传输。

## 6. 特征和动作契约

### 6.1 默认 student 观测

| 顺序 | 字段 | shape | 来源 |
| --- | --- | --- | --- |
| 0 | `attitude_q_wb` | 4 | sensor/EKF 接口 |
| 1 | `angular_velocity_b` | 3 | sensor |
| 2 | `acceleration_b` | 3 | sensor |
| 3 | `motor_rpm` | 2 | sensor 或显式配置的可用反馈 |
| 4 | `target_attitude_q_wb` | 4 | task command |
| 5 | `previous_action` | 5 | adapter |

配置必须逐字段声明来源、变换、单位和拼接顺序。`teacher` 可以额外观察 truth 或域参数，但要使用不同的 observation profile；student 配置引用 teacher-only 字段时应启动失败。

建议同时提供下列派生特征选项：目标相对姿态、角速度误差、动作变化量。启用任何派生特征都会改变 observation schema hash，因而不能无提示加载旧 checkpoint。

### 6.2 归一化

- 四元数先规范化，并统一符号（例如令 `w >= 0`）或改用连续的相对旋转表示。
- 有物理边界的特征优先使用配置中的固定 scale；在线 running statistics 必须保存在 checkpoint 中并冻结用于评估/导出。
- 统计量只使用 `valid & loss_mask` 样本，禁止混入 padding、冻结实例或 teacher-only 字段。
- 归一化 epsilon、裁剪范围和更新阶段均是配置的一部分。

### 6.3 动作变换

策略统一输出标准空间 `[-1,1]^5`：

```text
motor_pwm = (standard_action[0:2] + 1) / 2
servo_pwm = standard_action[2:5]
```

变换后再次 clamp 并记录 clamp 比例。探索噪声在标准空间定义；实际环境命令、未裁剪标准动作和分布参数均写入训练轨迹。部署导出包必须包含同一个动作变换。

## 7. 配置体系

### 7.1 规则

- 顶层必须有 `schema_version` 和唯一 `experiment.name`。
- 配置采用 YAML；支持显式 `defaults` 组合，但最终先解析成一棵完整、无插值的配置再运行。
- `${...}` 插值只能引用配置树或框架列出的环境变量白名单。运行目录中保存插值后的值；秘密值只能保存名称和摘要，不保存明文。
- 未识别字段报错，不能忽略；类型、范围、互斥项和跨字段约束由版本化 schema 校验。
- CLI 默认只允许覆盖 `run.device`、`run.output_root` 和 `resume.from`。改变学习率、seed 等实验语义必须生成一份新的已解析配置并形成新的 fingerprint。
- 仿真配置作为独立文件传入仿真层；训练配置记录其内容副本及 SHA-256，而不是只记录易失路径。

### 7.2 完整实验示例

```yaml
schema_version: 1

experiment:
  name: attitude_gru_ppo_v1
  description: "GRU policy for randomized attitude tracking"
  tags: [attitude, gru, ppo, student]

seed:
  base: 20260721
  deterministic_algorithms: true
  cudnn_benchmark: false

run:
  output_root: runs
  device: cuda:0
  dtype: float32
  parallel_count: 4096
  total_control_steps: 1000000000
  fail_on_nonfinite_loss: true
  resume:
    from: null
    mode: exact              # exact | weights_only

environment:
  factory: sim_env.environment:SimulationEnvironment
  config_path: ../SimEnv/configs/identified_vehicle.yaml
  observation_source: sensor
  truth_fields_for_diagnostics:
    - attitude_q_wb
    - angular_velocity_b
  invalid_instance: reset
  action_transform: normalized_5d_v1

task:
  name: attitude_tracking
  episode_duration_s: 30.0
  command:
    type: fixed_per_episode
    attitude_sampling: uniform_so3_bounded
    max_tilt_rad: 0.35
    max_yaw_rad: 3.141592653589793
  initial_state:
    attitude_sampling: uniform_so3_bounded
    max_tilt_rad: 0.35
  observation:
    profile: student_v1
    fields:
      - {name: attitude_q_wb, source: sensor, transform: quaternion_canonical}
      - {name: angular_velocity_b, source: sensor, scale: [5.0, 5.0, 5.0]}
      - {name: acceleration_b, source: sensor, scale: [20.0, 20.0, 20.0]}
      - {name: motor_rpm, source: sensor, scale: [1800.0, 1800.0]}
      - {name: target_attitude_q_wb, source: task, transform: quaternion_canonical}
      - {name: previous_action, source: adapter, scale: 1.0}
  reward:
    terms:
      attitude: {type: quaternion_geodesic, weight: 4.0}
      angular_rate: {type: squared_norm, weight: 0.1}
      action_rate: {type: squared_norm, weight: 0.01}
      action_saturation: {type: saturation, weight: 0.02}
    alive_bonus: 0.1
  termination:
    max_tilt_rad: 1.3
    max_angular_rate_rad_s: 20.0

model:
  type: gru_actor_critic
  encoder: {hidden_sizes: [128, 128], activation: silu}
  recurrent: {type: gru, hidden_size: 128, num_layers: 1}
  actor_head: {hidden_sizes: [128], distribution: tanh_normal}
  critic_head: {hidden_sizes: [128]}
  initialization: orthogonal

algorithm:
  name: recurrent_ppo
  gamma: 0.995
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coefficient: 0.001
  value_coefficient: 0.5
  max_grad_norm: 1.0
  epochs_per_rollout: 5
  minibatches: 8
  sequence_length: 128
  optimizer:
    type: adam
    learning_rate: 0.0003
    eps: 0.00001
  scheduler: {type: linear, final_factor: 0.1}

collector:
  control_steps_per_rollout: 128
  reset_hidden_on_episode: true

evaluation:
  every_control_steps: 1000000
  deterministic: true
  separate_environment: true
  scenario_set: configs/evaluation/attitude_v1.yaml
  seeds: [10001, 10002, 10003]
  episodes_per_scenario: 20
  export_best_by: attitude_error_rms

checkpoint:
  every_control_steps: 5000000
  keep_last: 5
  keep_best: 3
  save_replay: false
  atomic_write: true

recording:
  mode: complete
  metrics_flush_seconds: 10
  control_trajectory:
    enabled: true
    format: zarr
    chunk_steps: 4096
    compression: zstd
  simulator_physics_log:
    enabled: true
    overflow_policy: block
  system_metrics: true
  tensorboard: true
```

上述数值是结构示例，不代表已经调优。奖励权重、随机化范围和 PPO 参数应通过独立实验验证。

### 7.3 启动前跨字段校验

至少校验：

1. `physics_hz / control_hz` 为正整数，任务秒数可无歧义转换为控制步数。
2. 环境 action spec 为两电机加三舵机，顺序、范围与动作变换一致。
3. 观测字段在所选 source 中存在，拼接后维度与模型输入一致。
4. student 未使用 truth/域参数；teacher-only 字段带明确标签。
5. `collector.control_steps_per_rollout`、PPO `sequence_length` 和 minibatch 可整除或有显式 padding 策略。
6. off-policy 算法配置了 `burn_in`、`learn_length`、序列 replay 和 target network 更新。
7. 仿真、任务、模型和算法的 dtype/device 相容。
8. 评估 seed 与训练 seed 空间隔离，评估环境不共享训练 RNG 或归一化更新状态。

## 8. 种子与确定性

`seed.base` 通过稳定的命名派生函数生成子种子，禁止依赖模块创建顺序：

```text
derive(base, "sim.train")
derive(base, "sim.eval")
derive(base, "task.command.train")
derive(base, "task.initial_state.train")
derive(base, "policy.init")
derive(base, "policy.exploration")
derive(base, "collector")
derive(base, "replay.sampling")
derive(base, "dataloader")
```

派生算法及版本写入 manifest。框架初始化并保存 Python、NumPy（若使用）、PyTorch CPU、所有 CUDA generator、任务和 replay generator 的状态。仿真环境仍以其配置中的 seed 保证实例级随机流隔离；训练层将派生出的 `sim.train` seed 注入固化后的仿真配置副本。

复现分两个等级：

- **重复运行复现**：相同代码、锁定依赖、硬件/驱动、配置与 seed 得到相同结果。启用 PyTorch deterministic algorithms；遇到非确定算子直接失败。
- **精确续训复现**：checkpoint 后的动作、采样索引和 loss 与未中断运行一致。除模型/优化器外，还要求环境可序列化动态状态、所有 RNG、collector 游标、RNN hidden、归一化统计、scheduler、AMP scaler、任务课程状态和 replay 内容均恢复。

若仿真层不能提供动态 `state_dict`，框架必须在 manifest 中将 `exact_resume_supported=false`。此时 `weights_only` 可以从新 episode 继续，但不得称为精确续训。

## 9. 完整记录规范

### 9.1 运行目录

每次启动创建不可复用的目录：

```text
runs/<experiment-name>/<UTC timestamp>_<config-fingerprint>_<run-id>/
├── manifest.json
├── status.json
├── config/
│   ├── requested.yaml             # 用户输入
│   ├── resolved.yaml              # 完整解析结果
│   ├── schema.json
│   └── simulator.yaml             # 实际交给仿真器的内容
├── source/
│   ├── git.json                   # repo、commit、branch、dirty、diff hash
│   ├── working_tree.patch         # 工作树非空时保存 diff
│   └── entrypoint.txt
├── environment/
│   ├── python_packages.txt
│   ├── system.json
│   ├── pytorch.json
│   └── hardware.json
├── metrics/
│   ├── scalars.jsonl
│   └── tensorboard/
├── events/events.jsonl
├── trajectories/control/          # 控制级 TensorDict，含 RNN/reset/mask
├── simulator/                     # 仿真器 logs/<batch_id> 原始物理级日志
├── evaluation/<evaluation-id>/
├── checkpoints/
│   ├── step_<N>.pt
│   └── index.json
├── exports/
└── checksums.sha256
```

路径中的 timestamp 只用于区分运行，不参与随机数。`config-fingerprint` 是对规范化 resolved config、仿真配置内容和 schema 版本计算的 SHA-256 短摘要；`run-id` 使用 UUID。

### 9.2 `manifest.json`

manifest 至少包含：

- `run_id`、父运行 `parent_run_id`、创建/结束 UTC 时间、状态和退出原因；
- 配置 fingerprint、schema 版本、所有输入文件的路径与内容 SHA-256；
- 代码仓库 commit、dirty 状态、patch SHA-256；无 Git 时保存源码树摘要并标注限制；
- Python、PyTorch、TorchRL、CUDA/cuDNN、驱动、操作系统、CPU/GPU 型号；
- device、dtype、并行数、时基、训练总步数定义；
- 所有派生 seed 名称和值、确定性开关；
- observation/action spec 及其 schema hash；
- 仿真 `batch_id`、`instance_ids`、实际随机化参数文件位置；
- checkpoint 的能力声明，如 `exact_resume_supported` 和缺失状态；
- 记录器格式版本、压缩方式、flush/overflow 策略和任何显式降采样。

### 9.3 三层时间线

| 层级 | 时间分辨率 | 必需内容 |
| --- | --- | --- |
| 物理层 | 默认 5 kHz | 由仿真接口规定的控制、执行器、真值、传感器、力/矩、valid/error |
| 控制层 | 默认 500 Hz | 原始/归一化观测、目标、动作分布、标准动作、实际动作、reward 分项、done、RNN 状态引用 |
| 训练层 | 每次 update/eval | loss、梯度/参数范数、学习率、吞吐量、样本数、评估结果、checkpoint 事件 |

每条标量事件同时带 `global_control_steps`、`environment_steps`、`update_steps`、wall-clock UTC 和 monotonic time，禁止只用含义模糊的 `step`。

“完全记录”意味着配置声明为 `recording.mode=complete` 时，三个层级均可追溯且记录器溢出必须阻塞或使运行失败。由于完整 5 kHz 日志体积很大，启动时要估算磁盘占用并打印/记录预计值；磁盘空间不足应在训练前失败。允许另建 `compact` 模式，但它必须显式列出降采样字段，且不能用于宣称逐步可重放。

### 9.4 checkpoint 内容

精确恢复 checkpoint 至少包含：

- actor、critic、target network 及全部 buffers；
- optimizer、scheduler、AMP scaler；
- 全部 RNG states；
- collector 的当前观测、episode id/step、RNN hidden、reset mask；
- observation normalization 和 reward normalization 状态；
- algorithm counters、课程学习和自适应参数；
- 仿真动态 state、任务 state；
- off-policy replay 数据、写指针、优先级和抽样 RNG（若配置精确恢复）；
- resolved config fingerprint、observation/action schema hash 和代码身份。

checkpoint 先写同目录临时文件，`fsync` 后原子重命名，再更新 `index.json`。加载时校验摘要和 schema；`exact` 模式任何状态缺失均失败，`weights_only` 模式明确列出跳过项并建立新的 `run_id` 和 `parent_run_id`。

## 10. 训练生命周期

```text
load + resolve config
        -> validate schema/cross constraints
        -> allocate run directory and write initial manifest
        -> seed all named RNG streams
        -> create simulator + adapter + task
        -> derive and validate specs
        -> create policy + algorithm + collector
        -> optional exact restore
        -> collect -> update -> record -> evaluate -> checkpoint
        -> final evaluation + final checkpoint
        -> flush logs, checksums, terminal status
```

运行器必须用 `try/finally` 关闭环境并 flush 记录器。SIGINT/SIGTERM 到达时，在安全边界保存 `interrupt` checkpoint；若无法保存，仍在 `status.json` 和事件日志中记录原因。未捕获异常保存 traceback、最后成功 flush 的步数及运行状态 `failed`，不得把失败运行标成完成。

训练计数以实际推进的有效控制步为准。额外同时记录请求推进步、无效步、重置次数和仿真错误码分布，避免并行实例冻结时训练预算被悄悄缩短。

## 11. 评估与模型选择

评估使用独立环境与 RNG，不更新归一化统计，不使用探索噪声，并固定版本化 scenario set。至少报告：

- 姿态 geodesic error 的 RMS、P95、最大值和稳态误差；
- 角速度 RMS/峰值、稳定时间、超调量；
- 动作饱和率、动作变化率、电机差动与舵机使用量；
- episode 成功率、终止原因、仿真 invalid/error 分布；
- 按域随机参数区间分桶的性能，而不只报告全局均值；
- RNN 冷启动前 256 ms 和稳态阶段分别统计；
- 单步推理延迟和显存占用（训练侧），后续另测 MCU 延迟。

`best` checkpoint 只能由配置中的单一主指标和明确的同分规则选择。评估结果保存场景、seed、checkpoint SHA-256 和完整逐 episode 指标，确保能够离线重算汇总值。

建议维护三套固定评估：标称参数、训练分布内随机化、训练分布外压力测试。测试集 seed 不用于调参采样，变更 scenario set 必须提升版本。

## 12. Teacher、蒸馏与部署衔接

Teacher、student 和蒸馏属于不同 run，不在一个运行目录中覆盖产物：

1. teacher 配置使用 `observation.profile=teacher_*`，允许 truth 和实际域参数；
2. 数据生成 run 记录 teacher checkpoint SHA-256、仿真配置、噪声注入和每条序列的 episode/RNN 边界；
3. distillation run 将数据集 manifest 作为输入，监督目标明确是动作、分布参数、价值还是 hidden 表征；
4. 量化/导出 run 引用 student checkpoint，保存归一化、特征 schema、动作变换、量化校准集及 golden vectors；
5. MCU 实现对 golden vectors 做逐项一致性测试，并记录最大误差、P99/最差推理时间和内存布局。

导出接口只允许调用 `Policy.forward_step` 的部署子集，输入输出顺序由 schema 生成。GRU 初始 hidden state 的形状、数值和是否注入慢网络辨识状态必须进入导出元数据。

## 13. 错误处理

- **实例级仿真错误**：记录 `instance_id/error_code`，样本 `valid=False`，清除对应 RNN state，并按配置重置；不停止其他实例。
- **批次级设备/日志错误**：停止采样、尝试保存 emergency checkpoint、运行标记失败。
- **非有限 loss/梯度**：保存诊断 checkpoint 和触发 batch 的引用，停止运行；默认不自动跳过。
- **记录器落后**：`complete` 模式阻塞或失败，绝不静默丢弃。
- **配置/checkpoint 不兼容**：在创建 optimizer 或推进环境前失败，输出字段级差异。

## 14. 验收测试

框架完成首版实现时至少通过：

1. **接口 shape**：`B=1` 和 `B>1` 下观测、动作、reward、mask 与 RNN state shape 正确。
2. **动作边界**：标准空间端点精确映射到电机/舵机范围，NaN/Inf 和越界有明确结果。
3. **四元数等价**：`q` 与 `-q` 的姿态误差和奖励一致。
4. **RNN reset**：只重置指定实例；BPTT 截断不清 hidden；episode 边界不泄漏 hidden。
5. **失效隔离**：一个实例 invalid 不改变其他实例轨迹、RNG 或 loss mask。
6. **确定性短跑**：同设备、同配置运行两次固定小训练，初始权重、动作、奖励、loss 和 checkpoint 摘要一致。
7. **精确恢复**：连续运行 N+M 步与 N 步保存后恢复 M 步逐张量一致；不支持时测试必须明确 skip 原因且 manifest 声明 false。
8. **记录完整性**：任取一个 `instance_id`，能关联实际随机参数、物理日志、控制轨迹、episode、训练 update 和 checkpoint。
9. **配置拒绝**：未知字段、truth 泄漏、维度不符、非法时基、错误序列参数均在采样前失败。
10. **评估隔离**：插入评估不会改变后续训练动作、RNG 或优化结果。
11. **checkpoint 损坏**：摘要不符时拒绝加载，原子写入中断不破坏上一个有效 checkpoint。
12. **最小学习冒烟测试**：小批量、短 episode 的 PPO + GRU 能完成采样、更新、评估、保存和恢复，且 loss 有限。

## 15. 分阶段实现顺序

### 阶段 A：可验证闭环

- 配置加载/schema、运行目录、manifest 和 seed 管理；
- 仿真适配器、姿态跟踪任务、21 维 student 观测和动作变换；
- MLP/GRU policy、recurrent PPO、同步批次 episode；
- 控制级/物理级完整记录、确定性评估和 checkpoint；
- shape、失效隔离、RNN reset、短跑复现测试。

### 阶段 B：规模化训练

- masked 异步 reset 的吞吐与随机流隔离压力测试；
- 高吞吐异步记录、磁盘容量预检、性能 profiling；
- curriculum、域参数分桶评估、训练分布外测试；
- 精确环境 state 恢复及中断续训。

### 阶段 C：算法与架构扩展

- recurrent SAC/TD3 序列 replay；
- RMA/慢参数辨识网络；
- teacher 数据生成、合并蒸馏；
- INT8 量化、导出、golden vector 与 MCU 性能验收。

## 16. 首版完成定义

首版不是“能够启动训练”，而是同时满足：

- 一份完整 YAML 可以从空运行目录启动 GRU + PPO 姿态跟踪训练；
- 训练主循环不直接引用具体仿真字段、奖励实现或 GRU 类；
- 相同配置与环境得到可验证的确定性短跑结果；
- 任一 checkpoint 都能追溯到完整配置、源码状态、依赖、seed、实际仿真参数和数据时间线；
- RNN 状态、实例 invalid、episode 结束和 BPTT 截断语义均有自动化测试；
- 评估独立、结果可重算，best 模型选择无人工步骤；
- 对不能保证的能力（尤其仿真状态精确恢复）在 manifest 中明确声明，而不是隐式承诺。
