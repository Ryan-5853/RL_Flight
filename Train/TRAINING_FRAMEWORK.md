# 强化学习控制器训练框架

## 1. 文档目的

本文定义 `Train` 层的模块边界、公开接口、配置结构、运行目录和复现契约。实现者应能在不修改训练主流程的前提下替换仿真环境、任务、网络、强化学习算法、评估方案或记录后端。

训练层的唯一入口是：

```bash
python -m flight_train run --config configs/experiments/gru_ppo.yaml
```

除命令行明确允许的运行时覆盖项外，所有会影响实验结果的内容均来自配置文件并写入运行目录。框架不通过 Python 全局变量、隐式默认值或手工修改源码定义实验。

本文基于以下外部契约：

- 控制目标：以循环神经网络替代传统飞控的姿态 controller + allocator。虚拟/真实飞手直接拥有上桨油门；策略输入飞手姿态命令和飞行器观测，输出下桨电机及三路舵机归一化命令。
- 仿真接口：`SimulationEnvironment.create / observe / advance`；批量维始终为第一维，控制周期为一次 `advance`，单实例故障不得影响其他实例。
- 固定时基：物理仿真、控制与网络推理统一为 500 Hz；一次 `advance` 对应一个
  2 ms 仿真/控制步，不存在隐藏物理子步。
- 当前单帧策略输入：目标相对当前姿态四元数 4、角速度 3、加速度 3、电机转速 2、舵机实际角 3、目标偏航角速度 1、当前上桨油门 1、上次策略动作 4，共 21 维；MLP 可等间隔堆叠整帧，也可拼接当前完整帧、连续动作历史和稀疏物理响应历史。
- 当前策略输出：下桨电机及三个舵机，共 4 维标准动作；SimEnv 仍接收由飞手油门和策略动作合成的 5 维执行器命令。

四维标准动作统一解释为残差：

```text
physical_policy_command = trim_command + residual_scale * policy_action
```

`policy_action=0` 必须对应标称悬停配平点。配平值和残差尺度属于控制契约、实验
fingerprint 和部署接口，不能隐藏在环境源码中；改变任一数值都必须拒绝旧
checkpoint 精确续训。

> 21 维输入是首版实验约定，不是写死在模型中的常量。最终维度必须由观测配置和环境元数据推导并在启动时校验。

## 2. 设计原则

1. **配置即实验定义**：训练主脚本类型、模型、任务、奖励计算器、两级随机化、算法、种子、训练预算和记录策略全部配置化。
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
     |       |       |       |       |                    |
     v       v       v       v       v                    v
  EnvAdapter Task CommandSource RewardCalculator Policy Algorithm    run directory
     |                             |       |
     v                             v       v
 SimulationEnv                  torch.nn  TorchRL primitives
```

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `ConfigLoader` | 组合配置、schema 校验、插值、组件工厂解析、计算配置指纹 | 创建环境、执行随机采样 |
| `EnvAdapter` | 将仿真接口转换为统一的控制步接口、维护上次动作、发布奖励上下文 | 目标生成、解释奖励语义、网络 hidden state |
| `CommandSource` | 模拟/接收飞手油门和三轴摇杆，生成目标姿态并维护命令状态 | 学习高度控制、计算奖励、分配策略动作 |
| `Task` | 终止条件和奖励上下文，调用注入的奖励计算器 | 生成飞手命令、推进物理仿真、硬编码奖励公式、优化网络 |
| `RewardCalculator` | 从环境发布的任意张量字段批量计算总奖励、分项和诊断量 | 推进环境、读取 SimEnv 私有成员、执行 CPU/NumPy 运算 |
| `StaticRandomizer` | 在训练层采样飞行器静态参数并通过显式 TensorDict 交给环境 | 生成逐物理步噪声或动态环境行为 |
| `SimulationEnv` | 按传入的动态随机化规格生成建模误差、噪声和环境随机过程 | 决定训练分布、采样训练侧静态参数 |
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
│   ├── commands/                  # VirtualPilot、真实遥控器命令源接口
│   ├── tasks/                     # 终止、奖励上下文
│   ├── rewards/                   # 可注入、全张量奖励计算器
│   ├── randomization/             # 训练侧静态参数采样器
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

### 5.1 配置输入与组件注入

CLI 只接收一个实验配置路径。ConfigLoader 将其解析为不可变的
`ResolvedExperimentConfig`，校验完全部字段、分布和组件类型后，再解析
`experiment.entrypoint.type` 指定的训练主脚本。入口必须满足统一协议：

```python
class TrainingEntrypoint(Protocol):
    def __call__(
        self,
        config: ResolvedExperimentConfig,
        components: ComponentRegistry,
    ) -> RunResult: ...

class StaticRandomizer(Protocol):
    def sample(
        self,
        mask: torch.Tensor,           # [B] bool, device=device
        episode_id: torch.Tensor,     # [B] int64, device=device
    ) -> TensorDictBase: ...          # [B]，训练层静态参数

    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

`ComponentRegistry` 只允许从配置 schema 白名单解析环境、命令源、任务、奖励计算器、模型、
算法和记录器，禁止任意执行配置中的 shell 或脚本文本。标准训练入口的组装顺序为：

```python
entrypoint = registry.build(config.experiment.entrypoint)
static_randomizer = registry.build_static_randomizer(config.randomization.static)
reward_calculator = registry.build(config.reward.calculator)
command_source = registry.build(config.command_source)
simulator = registry.build_simulator(
    config.environment,
    dynamic_randomization=config.randomization.dynamic,
)
task = registry.build_task(config.task, reward_calculator=reward_calculator)
components = ComponentSet(
    static_randomizer=static_randomizer,
    reward_calculator=reward_calculator,
    command_source=command_source,
    simulator=simulator,
    task=task,
)
return entrypoint(config, components)
```

静态随机化由训练层的设备绑定 `torch.Generator` 直接生成 `[B,...]` TensorDict，
在 reset 时与 mask 一起传给环境；动态随机化配置及其 seed 不在训练层采样，原样
编译为仿真配置交给 SimEnv。组件构建完成后不得依赖 Python 全局随机状态。

### 5.2 控制步数据

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

### 5.3 环境适配器

```python
class BatchedControlEnv(Protocol):
    @property
    def spec(self) -> EnvSpec: ...

    def reset(
        self,
        mask: torch.Tensor | None = None,
        *,
        scenario: TensorDict | None = None,
        static_parameters: TensorDictBase | None = None,
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

1. VirtualPilot 根据高度真值、增量式 PI 和摇杆状态生成当前上桨油门与目标姿态。
2. 校验策略动作 `[B,4]`；把上桨油门、下桨电机和三舵机合成为 `[B,5]` SimEnv 命令。
3. 调用 `SimulationEnvironment.advance(command)` 一次。
4. 按任务配置分别读取 `truth`、`sensor`；不得通过 `info` 偷渡未声明真值给 student。
5. 由 `Task` 计算结束标志，生成奖励上下文后调用注入的 `RewardCalculator`。
6. VirtualPilot 生成下一周期命令，保存 `previous_policy_action`，构造下一帧 21 维基础观测，再按控制契约生成策略历史输入。

当前仿真实现已提供 `reset(reset_mask, config_path)`，适配器应直接使用该 GPU bool mask 独立重置完成、截断或失效的实例。重置属于稀疏 episode 边界；控制步热路径中的观测、动作、reward、RNN state 和 rollout 不得为判断单实例状态而转到 CPU。

`static_parameters` 是训练层为本次 reset 采样的 `[B,...]` 飞行器静态参数；
SimEnv 只能把 mask 对应行应用到相应实例。动态随机化规格在环境创建时注入，
其逐步 RNG 和过程状态由 SimEnv 自己维护并纳入 SimEnv 的 state/记录接口。

#### 5.3.1 虚拟飞手与命令所有权

VirtualPilot 是训练环境中的外部 `CommandSource`，部署时由真实遥控器输入替换。它允许
读取“向上为正”的高度真值，使用增量式 PI
`Δu=Kp*(e-e_prev)+Ki*e*dt` 跟踪默认 `height_m=0`，并使用独立上升/下降斜率限制
直接输出上桨 PWM。高度误差和控制输出均限幅，不使用迟滞档位。该高度真值不进入策略
观测、不产生高度奖励，该辅助控制器也不是要部署的姿态控制器。

roll/pitch 摇杆分别表示目标姿态角，二维目标在归一化圆盘内有界采样；yaw 摇杆表示
目标角速度并积分为航向。每个实例独立持有随机目标和保持计时器，实际摇杆使用精确
离散的一阶惯性环节：`u_next = u + (1-exp(-dt/tau))*(u_target-u)`。所有状态均为设备上
的 `[B,...]` 张量，masked reset 只重置相应实例，并进入 checkpoint。

### 5.4 任务与奖励计算器

任务不再内置机械性的奖励参数模板，也不拥有目标命令 RNG。VirtualPilot 负责目标
命令；Task 只负责终止条件和奖励上下文的组织，具体奖励由运行时注入的张量化对象完成。

```python
class RewardCalculator(Protocol):
    """可插拔的批量奖励函数；不得依赖具体算法或 CPU 状态。"""

    def __call__(self, context: TensorDictBase) -> RewardOutput: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

@dataclass(frozen=True)
class RewardOutput:
    reward: torch.Tensor               # [B, 1]，与 context 同设备
    terms: TensorDictBase              # [B, ...]，分项与诊断量
    valid: torch.Tensor | None = None  # [B, 1]，奖励自身发现非法值时使用
```

`context` 是环境与任务公开的张量视图，而不是固定字段模板。它可以包含
`observation`、`next_observation`、truth/sensor 观测、目标、动作、上一动作、
角速度、姿态误差、执行器状态、终止标志、静态域参数和仿真诊断字段；字段的
来源、单位、shape、是否允许 student 使用必须在配置中声明。RewardCalculator
可以读取任意已声明字段，但只能返回批量张量，不得调用 `.cpu()`、`.numpy()`、
`.item()` 或按实例写 Python 循环。

奖励上下文的权限独立于策略观测权限：奖励计算器允许读取 SimEnv 公开的 truth、
实际域参数和诊断值来形成训练信号，但这些字段不会因此进入 student observation。
“可以观测环境的任何值”指任意公开、已声明、可张量化的环境字段，不包括 SimEnv
私有 Python 成员；这样既保留奖励表达能力，也能做 schema 校验和离线重算。

奖励计算器的类型由 `reward.calculator.type` 解析，构造参数由
`reward.calculator.params` 传入；运行器不通过 `if/else` 解释奖励公式。计算器的
类路径/版本、参数、输入 context schema hash 和输出分项都写入 manifest。训练、
评估和离线重算必须使用同一计算器版本；更换计算器必须产生新的实验 fingerprint。

```yaml
reward:
  calculator:
    type: flight_train.rewards.attitude:AttitudeRewardCalculator
    params:
      primary_metric: attitude_geodesic_rad
      termination_penalty: 0.0
  context:
    fields:
      - {name: attitude_geodesic_rad, source: task, dtype: float32, shape: [1], unit: rad}
      - {name: angular_velocity_b, source: sensor, dtype: float32, shape: [3], unit: rad/s}
      - {name: action, source: policy, dtype: float32, shape: [4], unit: normalized}
      - {name: static_domain, source: train_randomizer, visibility: reward_only}
```

任务通过 `reward_context` 返回 context，训练目标只消费唯一的 `reward` 字段。
这使得更换奖励实现不需要修改环境适配器、collector 或 PPO；奖励上下文中禁止
出现未声明的 CPU 对象或 Python 标量。

```python
class Task(Protocol):
    def transition(
        self,
        attitude: torch.Tensor,
        angular_velocity: torch.Tensor,
        target_attitude: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        environment_context: TensorDictBase,
    ) -> TaskTransition: ...
```

首个任务 `attitude_tracking`：时长默认 30 s（15000 个控制步），跟踪 VirtualPilot 持续更新的目标姿态。目标四元数必须归一化；姿态误差使用四元数最短弧。目标命令的采样、滤波和 RNG 只归 VirtualPilot 所有，Task 只计算终止与 reward context，避免双重命令源。

自稳训练采用成功率驱动的 episode 课程。默认从 2 s 纯悬停开始，在累计至少
`B × evaluation_episodes_per_env` 个完整 episode 后计算
`truncated / (truncated + terminated)`；成功率连续达到配置门槛才进入下一阶段。
阶段、累计成功/失败数、连续达标数和当前命令比例必须进入 checkpoint。终止成本按
当前阶段剩余时长增加，避免策略通过提前失稳、立即 reset 来减少姿态误差成本。

### 5.5 MLP 与循环策略

在进入循环策略训练前，先使用无状态 `mlp_actor_critic` 建立 PPO 和 SAC 基线。两者严格使用
自稳契约生成的观测和 4 维标准动作，actor 与 critic 均采用
`256×256×128` SiLU MLP。MLP 不创建或伪造 recurrent state。PPO 将每个有效控制步
作为独立 minibatch 样本；SAC 将 transition 写入设备上的 TensorDict replay，并使用
actor、双 Q、自动温度和软目标网络。SAC 方差必须通过
`model.policy_distribution` 声明逐动作初值、下界和上界；网络输出层从零均值及指定
方差开始。标称 teacher 默认固定小方差，并在 replay warm-up 后先执行配置数量的
critic-only 更新；actor 解冻后，每次也必须先完成 critic 更新并重新前向，禁止
actor 使用尚未训练或当前更新前的 Q 梯度。这避免 500 Hz 飞控的稳定配平初始化
被随机 critic 或熵膨胀立即破坏。warm-up 与 critic-only 阶段仍统计完整 episode
成功率，但必须禁止课程晋级；actor 首次实际更新后才启用连续达标计数。
500 Hz 下无状态 SAC 不得只依赖一步 TD 让姿态结果逐次反传。配置通过
`algorithm.n_step_return` 声明多步回报长度；实现必须跨 collector rollout
保留 `n-1` 步原始上下文，只把拥有完整未来窗口或窗口内遇到真实 episode 结束的
起点写入 replay。replay transition 必须携带 `steps_to_next_obs`，使目标折扣为
`gamma ** steps_to_next_obs`；`next.observation`、`terminated`、`truncated`、
`done` 和 `valid` 必须共同指向多步终点，禁止观测已移动而终止标志仍停留在一步后。
上下文属于 exact checkpoint 状态。
两种算法共用环境、奖励、动作变换、记录、固定评测和 exact checkpoint。

对于带明显执行器低通、速率限制、死区或回差的 MLP 实验，推荐
`multirate_actuator` 历史模式：当前 21 维完整帧保留即时状态，最近每一个控制步的
4 维策略动作连续保存以避免高频控制混叠，而角速度、加速度、电机实际转速和舵机
实际角这 11 维物理响应可按较低频率保存以扩展时间覆盖。当前实验采用 32 步连续
动作与 15 帧、间隔 4 步的物理响应，共 314 维，在 500 Hz 下覆盖最长 120 ms。

示例配置为 `configs/experiments/mlp_ppo_smoke.json`，其中
`model.type=mlp_actor_critic`、`algorithm.name=ppo` 且 `sequence_length=1`。
SAC 冒烟配置为 `configs/experiments/mlp_sac_smoke.json`，正式标称配置为
`configs/experiments/mlp_sac_nominal_baseline_1.yaml`。

MLP 与后续 GRU 共用固定输入尺度：姿态/目标四元数、当前油门和上一策略动作不变，角速度除以任务
终止阈值，加速度除以 `9.80665 m/s²`，电机转速除以 `1800 rad/s`。尺度必须进入导出
元数据和 MCU golden vectors，禁止部署端根据运行数据重新估计归一化统计量。

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

### 5.6 算法

```python
class Algorithm(Protocol):
    def act(self, batch: TensorDict, recurrent_state: TensorDict, *, explore: bool) -> PolicyStep: ...
    def update(self, data: TensorDict, global_step: int) -> Mapping[str, torch.Tensor]: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

学习层使用 TorchRL：以 `TensorDict` 作为采样与训练的数据总线，循环网络使用
`GRUModule`，PPO 使用 `GAE + ClipPPOLoss`；无状态 MLP-SAC 使用 `SACLoss`、
`TensorDictReplayBuffer`、双 Q 和 `SoftUpdate`。环境动力学和
`RewardCalculator` 保持纯张量 kernel，不依赖具体算法。当前 SAC 只支持 MLP；
recurrent SAC/TD3 仍需序列 replay、burn-in 和 recurrent target network 测试后启用。

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
| 5 | `upper_throttle_command` | 1 | VirtualPilot，`[0,1]` 映射到 `[-1,1]` |
| 6 | `previous_policy_action` | 4 | adapter |

配置必须逐字段声明来源、变换、单位和拼接顺序。`teacher` 可以额外观察 truth 或域参数，但要使用不同的 observation profile；student 配置引用 teacher-only 字段时应启动失败。

建议同时提供下列派生特征选项：目标相对姿态、角速度误差、动作变化量。启用任何派生特征都会改变 observation schema hash，因而不能无提示加载旧 checkpoint。

### 6.2 归一化

- 四元数先规范化，并统一符号（例如令 `w >= 0`）或改用连续的相对旋转表示。
- 有物理边界的特征优先使用配置中的固定 scale；在线 running statistics 必须保存在 checkpoint 中并冻结用于评估/导出。
- 统计量只使用 `valid & loss_mask` 样本，禁止混入 padding、冻结实例或 teacher-only 字段。
- 归一化 epsilon、裁剪范围和更新阶段均是配置的一部分。

### 6.3 动作变换

策略统一输出标准空间 `[-1,1]^4`，VirtualPilot 另行输出上桨 PWM：

```text
upper_motor_pwm = virtual_pilot.upper_throttle
lower_motor_pwm = (policy_action[0] + 1) / 2
servo_pwm = policy_action[1:4]
simulator_command = [upper_motor_pwm, lower_motor_pwm, servo_pwm[0:3]]
```

PPO 的 log-prob、探索噪声、动作变化与饱和惩罚只覆盖 4 维策略动作，不能包含策略不可控的上桨油门。合成后的 5 维实际环境命令、VirtualPilot 状态、未裁剪标准动作和分布参数均写入训练轨迹。部署导出包必须包含同一个下桨/舵机动作变换。

## 7. 配置体系

### 7.1 规则

- 本节新增的统一输入端契约使用 `schema_version: 2`。它要求训练入口、两级随机化和
  RewardCalculator，不能把旧版固定 `task.reward` 权重模板静默解释为 v2；迁移必须显式进行。
- 顶层必须有 `schema_version` 和唯一 `experiment.name`。
- 配置采用 YAML；支持显式 `defaults` 组合，但最终先解析成一棵完整、无插值的配置再运行。
- `${...}` 插值只能引用配置树或框架列出的环境变量白名单。运行目录中保存插值后的值；秘密值只能保存名称和摘要，不保存明文。
- 未识别字段报错，不能忽略；类型、范围、互斥项和跨字段约束由版本化 schema 校验。
- CLI 默认只允许覆盖 `run.device`、`run.output_root` 和 `resume.from`。改变学习率、seed 等实验语义必须生成一份新的已解析配置并形成新的 fingerprint。
- 仿真配置作为独立文件传入仿真层；训练配置记录其内容副本及 SHA-256，而不是只记录易失路径。

训练层输入文件必须同时声明以下四类内容：

1. `experiment.entrypoint`: 训练主脚本/编排器类型，例如 `flight_train.runner:run_experiment`；
   主脚本类型是配置语义的一部分，不能由文件名或源码默认推断。
2. `randomization.static`: 训练层主动采样的飞行器静态参数，例如质量、质心、
   惯量、电机/舵机常数、气动几何和控制分配相关参数。物理标称值只来自 SimEnv；
   训练节点只声明相对/绝对范围、分布、单位、约束和 `seed_stream`。采样结果以
   `[B,...]` TensorDict 传给仿真环境，在一个 episode 内保持不变。
3. `randomization.dynamic`: 仿真层采样的动态参数和随机过程，例如传感器噪声、
   bias 漂移、延迟、动力学建模误差、执行器扰动和环境随机行为。每个字段声明
   scale/stddev、分布、时间相关性、作用域和 `seed_stream`；分布中心由 SimEnv
   对应参数的标称值给出。训练层只把已解析规格和 seed 交给
   SimEnv，不在训练层重复采样动态噪声。
4. `reward.calculator`: 可导入的奖励计算器类型、版本、参数和 context schema。

两级随机化必须使用不同的命名随机流：至少包含
`randomization.static.seed` 与 `randomization.dynamic.seed`，并按 instance/episode
稳定派生子种子。静态样本和动态实际样本都必须记录，不能只记录分布配置。

SimEnv 配置是物理标称值的唯一事实来源；训练 resolved 配置只拥有随机化规则。
关闭某参数的随机化时必须删除该参数节点，不能使用 `distribution: fixed` 复制标称值。
Train 在环境创建后按字段路径绑定 SimEnv 已解析参数；未知字段、重复 `baseline` 或
static/dynamic 所有权冲突均在首次采样前失败。

#### 7.1.1 两级随机化边界

```text
训练配置
  ├─ static relative/absolute range + uniform distribution + seed
  │       └─ Train StaticRandomizer -> [B, parameter_dim] static_domain
  │                                      (episode 内固定)
  └─ dynamic stddev/range + normal/process distribution + seed
          └─ SimEnv DynamicRandomizer -> noise/model_error/environment_behavior
                                          (控制步/物理步按配置演化)
```

训练层静态随机化改变“这架飞行器是谁”；仿真层动态随机化改变“这次飞行过程中
观测和动力学如何偏离基线”。两者不可合并成一个无来源的参数字典，否则无法判断
随机性由哪一层消费、如何重放，也无法隔离 student 可见信息。

静态字段推荐格式：

```yaml
randomization:
  static:
    seed: 31001
    scope: episode
    parameters:
      body.mass:
        distribution: uniform
        mode: relative
        range: [-0.10, 0.10]
        unit: ratio
        constraints: [positive]
        seed_stream: static.body.mass
      body.inertia_diagonal_b:
        distribution: uniform
        mode: relative
        range: [-0.10, 0.10]
        unit: ratio
        seed_stream: static.body.inertia
  dynamic:
    seed: 41001
    scope: simulator
    parameters:
      sensors.gyro.noise.stddev:
        stddev: [0.0002, 0.0002, 0.0002]
        distribution: normal
        unit: rad/s
        seed_stream: dynamic.sensor.gyro.noise
```

物理标称值从 SimEnv 同路径参数读取；训练配置禁止再写 `baseline`。`range` 表示
均匀采样区间：`mode: relative` 时是相对偏差，`mode: absolute` 时是实际物理值；
`stddev` 表示围绕 SimEnv 标称值的正态扰动尺度。向量、矩阵和
曲线参数必须逐元素声明 shape、单位和约束；协方差、相关时间、截断范围等过程
参数也属于配置。实际采样值写入每个 episode 的随机化记录，并带
`instance_id`、`episode_id`、两级 seed 和 schema hash。

#### 7.1.2 参数与分布通用 schema

需要随机化的物理参数只能在 `static` 或 `dynamic` 中出现一次；关闭随机化时删除节点。
标称值始终保留在 SimEnv 配置，随机化参数节点包含：

| 字段 | 含义 |
| --- | --- |
| `distribution` | `uniform`、`normal`、`truncated_normal` 或注册的过程类型；不支持 `fixed` |
| `range` | `uniform/log_uniform` 的闭区间；同时也是有界分布的真实 support |
| `stddev` | 围绕 SimEnv 标称值的正态扰动尺度 |
| `valid_range` | 独立于采样分布的物理合法范围；普通 normal 超界必须按声明的 `reject/clamp/fail` 处理 |
| `mode` | static 使用 `absolute`（直接采样实际值）或 `relative`（标称值乘 `1+偏差`） |
| `unit/shape/dtype` | 物理单位和张量 schema；标量也必须能广播到 `[B,1]` |
| `scope` / `temporal_process` | `run`、`instance`、`episode`、`control_step`、`physics_step` 或具体随机过程 |
| `seed_stream` | 稳定命名的子随机流；不得使用字段遍历顺序作为随机流身份 |
| `constraints` | 正值、矩阵正定、曲线单调、四元数归一化等跨元素约束 |

配置解析器必须把 `normal + valid_range` 编译成确定的截断/拒绝策略，不能依赖实现
默认值。相关参数必须通过协方差矩阵或显式 `correlation_group` 表达；左右电机等
需要共享或镜像的参数必须声明 `sharing: common/mirrored/independent`。启动时训练层
向 SimEnv 查询可写参数 schema，路径、shape、单位或 ownership 不一致立即失败。

### 7.2 完整实验示例

```yaml
schema_version: 2

experiment:
  name: attitude_gru_ppo_v1
  description: "GRU policy for randomized attitude tracking"
  tags: [attitude, gru, ppo, student]
  entrypoint:
    type: flight_train.runner:run_experiment
    version: train-framework-v1

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

environment:
  factory: sim_env.environment:SimulationEnvironment
  config_path: ../SimEnv/configs/identified_vehicle.yaml
  observation_source: sensor
  truth_fields_for_diagnostics:
    - attitude_q_wb
    - angular_velocity_b
  invalid_instance: reset
  action_transform: self_stabilize_4d_to_simenv_5d_v1

command_source:
  type: flight_train.commands:VirtualPilotCommandSource
  version: 2
  seed: 51001
  params:
    throttle:
      minimum: 0.20
      maximum: 0.85
      spool: {duration_s: 1.0, target_range: [0.30, 0.40]}
      height_controller:
        observation_source: truth
        target_m: 0.0
        initial_throttle_range: [0.48, 0.60]
        proportional_gain: 0.08
        integral_gain: 0.04
        error_limit_m: 5.0
      slew_rate: {rise_per_s: 0.50, fall_per_s: 0.35}
    sticks:
      roll: {mode: angle, limit_rad: 0.35, time_constant_s: 0.20}
      pitch: {mode: angle, limit_rad: 0.35, time_constant_s: 0.20}
      yaw: {mode: rate, limit_rad_s: 1.50, time_constant_s: 0.30}
      target_sampling:
        distribution: centered
        center_exponent: 2.0
        hold_duration_s: {range: [1.0, 4.0]}
      reset: {filtered_stick: zero, initial_target_scale: 0.25}

control_contract:
  version: self_stabilize_v1
  observation_profile: attitude_self_stabilize_21d_v3
  action_transform:
    type: residual_around_trim
    trim_command: [0.5539, 0.0, 0.0, 0.0]
    residual_scale: [0.30, 1.0, 1.0, 1.0]
  policy_action: {fields: [lower_motor, servo_1, servo_2, servo_3]}
  external_action: {fields: [upper_motor]}
  simulator_command: {fields: [upper_motor, lower_motor, servo_1, servo_2, servo_3]}

randomization:
  static:
    seed: 31001
    scope: episode
    parameters:
      body.mass:
        distribution: uniform
        mode: relative
        range: [-0.10, 0.10]
        unit: ratio
        seed_stream: static.body.mass
      body.inertia_diagonal_b:
        distribution: uniform
        mode: relative
        range: [-0.10, 0.10]
        unit: ratio
        seed_stream: static.body.inertia
  dynamic:
    seed: 41001
    scope: simulator
    parameters:
      sensors.gyro.noise.stddev:
        stddev: [0.0002, 0.0002, 0.0002]
        distribution: normal
        unit: rad/s
        seed_stream: dynamic.sensor.gyro.noise

task:
  episode_duration_s: 30.0
  episode_curriculum:
    durations_s: [2.0, 5.0, 10.0, 30.0]
    target_scales: [0.0, 0.15, 0.40, 1.0]
    success_fraction: 0.80
    evaluation_episodes_per_env: 1.0
    consecutive_passes: 2
  termination:
    max_tilt_rad: 1.3
    max_angular_rate_rad_s: 20.0

reward:
  calculator:
    type: flight_train.rewards.attitude:AttitudeRewardCalculator
    version: 3
    params:
      roll_pitch_weight: 2.0
      tilt_weight: 2.0
      yaw_rate_weight: 0.05
      angular_rate_weight: 0.05
      survival_progress_weight: 5.0
      tilt_barrier_weight: 10.0
      barrier_start_fraction: 0.30
      termination_penalty: 500.0
      early_termination_penalty: 2500.0
  context:
    fields:
      - {name: attitude_geodesic_rad, source: task, dtype: float32, shape: [1], unit: rad}
      - {name: angular_velocity_b, source: sensor, dtype: float32, shape: [3], unit: rad/s}
      - {name: action, source: policy, dtype: float32, shape: [4], unit: normalized}
      - {name: static_domain, source: train_randomizer, visibility: reward_only}

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
  actor_max_grad_norm: 1.0
  critic_max_grad_norm: 5.0
  epochs_per_rollout: 5
  minibatches: 8
  sequence_length: 128
  optimizer:
    type: adam
    actor_learning_rate: 0.0003
    critic_learning_rate: 0.0001
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
  resume:
    from: null
    mode: exact
  # 估算出的新 checkpoint 大小之外仍须保留的空间；不足时在安全边界退出。
  minimum_free_space_bytes: 2147483648
  # 按全部并行实例累计控制步计数，在完整 update 边界保存；null 表示仅最终保存。
  interval_control_steps: 5000000
  keep_last: 5

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

上述数值是结构示例，不代表已经调优。奖励计算器参数、随机化范围和 PPO 参数应通过独立实验验证。

### 7.3 启动前跨字段校验

至少校验：

1. `physics_hz` 与 `control_hz` 均为不可随机化的 500 Hz，任务秒数可无歧义转换为控制步数。
2. 策略 action spec 固定为下桨电机加三舵机 4 维；VirtualPilot 上桨油门与其合成后的 SimEnv command 为 5 维，所有权、顺序和范围一致。
3. 观测字段在所选 source 中存在，拼接后维度与模型输入一致。
4. student 未使用 truth/域参数；teacher-only 字段带明确标签。
5. `collector.control_steps_per_rollout`、PPO `sequence_length` 和 minibatch 可整除或有显式 padding 策略。
6. off-policy 算法配置了 `burn_in`、`learn_length`、序列 replay 和 target network 更新。
7. 仿真、任务、模型和算法的 dtype/device 相容。
8. 评估 seed 与训练 seed 空间隔离，评估环境不共享训练 RNG 或归一化更新状态。
9. `experiment.entrypoint.type`、环境、任务、奖励计算器和算法类型均在组件白名单中，构造参数与其版本化 schema 一致。
10. 每个随机化参数只属于 static 或 dynamic 一层；标称值仅来自 SimEnv，Train 节点不含 baseline/fixed，分布参数、shape、单位、截断和约束完整。
11. static sampler 输出 `[B,...]` 且保持在环境设备；dynamic 规格中的字段被 SimEnv 声明支持，两个 seed/stream 命名空间无冲突。
12. RewardCalculator 声明的 context 字段全部可由环境/任务提供，输出严格为有限的 `[B,1]` 同设备张量；student observation 权限与 reward context 权限分别校验。
13. VirtualPilot 三轴模式、时间常数、保持时间、油门范围和增量式高度 PI 参数合法；command seed 与 static/dynamic seed 隔离。

## 8. 种子与确定性

`seed.base` 通过稳定的命名派生函数生成子种子，禁止依赖模块创建顺序：

```text
derive(base, "sim.train")
derive(base, "sim.eval")
derive(base, "randomization.static.train")
derive(base, "randomization.dynamic.train")
derive(base, "command_source.virtual_pilot.train")
derive(base, "task.initial_state.train")
derive(base, "policy.init")
derive(base, "policy.exploration")
derive(base, "collector")
derive(base, "replay.sampling")
derive(base, "dataloader")
```

派生算法及版本写入 manifest。`randomization.static.seed` 和
`randomization.dynamic.seed` 在 resolved config 中必须是明确整数：可以直接指定，
也可以由上述命名流派生，但不能在运行时隐式取全局 seed。框架初始化并保存 Python、
NumPy（若使用）、PyTorch CPU、所有 CUDA generator、静态随机化、VirtualPilot 和 replay
generator 的状态。动态随机化 RNG 及过程状态由仿真层保存；训练层将解析后的动态
seed 与随机化规格注入固化的仿真配置副本。

复现分两个等级：

- **重复运行复现**：相同代码、锁定依赖、硬件/驱动、配置与 seed 得到相同结果。启用 PyTorch deterministic algorithms；遇到非确定算子直接失败。
- **精确续训复现**：checkpoint 后的动作、采样索引和 loss 与未中断运行一致。除模型/优化器外，还要求环境可序列化动态状态、所有 RNG、collector 游标、RNN hidden、归一化统计、scheduler、AMP scaler、任务课程状态和 replay 内容均恢复。

当前 SimEnv 已提供完整动态 `state_dict/load_state_dict`，包括 truth/执行器状态、传感器
输出与延迟环形缓冲、随机计数器、当前参数、控制量、物理/控制步和有效性状态。Train
只在完整采集/算法 update 安全边界保存；SAC 还保存双 Q/目标 Q、alpha、三个优化器、
replay 内容和采样状态。恢复后 manifest 标记 `exact_resume_supported=true`。
恢复兼容摘要只忽略总训练预算、输出目录和 resume 来源；其他实验语义不一致立即拒绝。

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
├── randomization/
│   ├── static_schema.json          # 训练层静态参数 schema/分布/seed
│   ├── static_samples/             # instance/episode 对应的实际静态样本
│   ├── dynamic_schema.json         # 交给 SimEnv 的动态规格/分布/seed
│   └── dynamic_samples/            # SimEnv 返回的实际模型误差/过程摘要或日志索引
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
- 训练主入口、RewardCalculator、StaticRandomizer、VirtualPilot 的完全限定类型、版本和构造参数；
- observation/action spec 及其 schema hash；
- static/dynamic 随机化 schema hash、owner、scope、seed，以及仿真 `batch_id`、`instance_ids`、实际样本文件位置；
- checkpoint 的能力声明，如 `exact_resume_supported` 和缺失状态；
- 记录器格式版本、压缩方式、flush/overflow 策略和任何显式降采样。

### 9.3 三层时间线

| 层级 | 时间分辨率 | 必需内容 |
| --- | --- | --- |
| 仿真层 | 固定 500 Hz | 由仿真接口规定的控制、执行器、真值、传感器、力/矩、valid/error |
| 控制层 | 固定 500 Hz，与仿真步一一对齐 | 原始/归一化观测、目标、动作分布、标准动作、实际动作、reward 分项、done、RNN 状态引用 |
| 训练层 | 每次 update/eval | loss、梯度/参数范数、学习率、吞吐量、样本数、评估结果、checkpoint 事件 |

每条标量事件同时带 `global_control_steps`、`environment_steps`、`update_steps`、wall-clock UTC 和 monotonic time，禁止只用含义模糊的 `step`。

“完全记录”意味着配置声明为 `recording.mode=complete` 时，三个层级均可追溯且记录器溢出必须阻塞或使运行失败。完整 500 Hz 仿真日志在高并行训练下仍可能很大，启动时要估算磁盘占用并打印/记录预计值；磁盘空间不足应在训练前失败。允许另建 `compact` 模式，但它必须显式列出降采样字段，且不能用于宣称逐步可重放。

训练更新指标必须区分“数值有效”和“任务存活”：`valid_fraction` 只描述 SimEnv/reward
张量是否有效，正常姿态终止并立即 reset 后仍可能保持为 1。每次 update 至少同时记录
terminated/truncated/done/invalid 比例、reset 数、已结束 episode 平均长度/生存时间、
姿态误差 mean/P95、角速度、高度误差、动作 RMS/峰值/饱和率，防止反复坠毁的策略被
误判为正常训练。

### 9.4 checkpoint 内容

精确恢复 checkpoint 至少包含：

- actor、critic、target network 及全部 buffers；
- optimizer、scheduler、AMP scaler；
- 全部 RNG states；
- collector 的当前观测、episode id/step、RNN hidden、reset mask；
- observation normalization 和 reward normalization 状态；
- RewardCalculator 的可变状态、StaticRandomizer generator/state、VirtualPilot RNG/摇杆/油门状态和当前静态参数；
- algorithm counters、课程学习和自适应参数；
- 仿真动态 state、动态随机过程/RNG state、任务 state；
- off-policy replay 数据、写指针、优先级和抽样 RNG（若配置精确恢复）；
- resolved config fingerprint、observation/action schema hash 和代码身份。

checkpoint 先写同目录临时文件，`fsync` 后原子重命名，再更新 `index.json`。加载时校验摘要和 schema；`exact` 模式任何状态缺失均失败，`weights_only` 模式明确列出跳过项并建立新的 `run_id` 和 `parent_run_id`。

## 10. 训练生命周期

```text
load + resolve config
        -> validate schema/cross constraints
        -> allocate run directory and write initial manifest
        -> resolve entrypoint/component factories
        -> seed static/dynamic and all named RNG streams
        -> create StaticRandomizer + RewardCalculator + VirtualPilot
        -> create simulator(dynamic spec) + adapter + task(reward object)
        -> derive and validate specs
        -> create policy + algorithm + collector
        -> optional exact restore
        -> collect -> update -> record -> evaluate -> checkpoint
        -> final evaluation + final checkpoint
        -> flush logs, checksums, terminal status
```

运行器必须用 `try/finally` 关闭环境并 flush 记录器。SIGINT/SIGTERM 到达时，在安全边界保存 `interrupt` checkpoint；若无法保存，仍在 `status.json` 和事件日志中记录原因。checkpoint 写入前按去重后的 tensor storage 估算新文件大小，并在其之外保留 `minimum_free_space_bytes`；SimEnv 日志也维护独立余量。任一保护触发时不再尝试写新的 interrupt checkpoint，保留最后一份已校验 checkpoint，并把运行记录为 `interrupted`、原因为 `insufficient_disk_space`。未捕获异常保存 traceback、最后成功 flush 的步数及运行状态 `failed`，不得把失败运行标成完成。

长 rollout 期间必须提供实时可见进度，不能只在 update 完成后静默写文件。交互终端按
时间节流原位显示 collect 局部步数、全局样本比例、rollout 序号、已用时间和 ETA；进入
PPO update、保存 checkpoint、正常结束或安全中断时显式切换阶段。每轮完成行同时报告
吞吐量、reward、姿态 P95、终止率和动作饱和率。非 TTY/重定向日志只保留低频阶段与
每轮汇总，进度走 stderr，stdout 保留机器可解析的最终 run 目录。

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

当前标称固定评测由 `scripts/evaluate_fixed.py` / `flight-train evaluate` 实现，并使用
`configs/evaluation/fixed_attitude_v1.yaml`。科目固定为定点悬停、固定姿态匀速平移参考、
roll/pitch 正余弦圆轨迹参考。每个科目必须创建独立 SimEnv，使用确定性策略动作，并记录：

- 首次终止前的生存时间和生存比例；
- 姿态测地角 RMSE、三维位置 RMSE、三维速度 RMSE；
- 4 维标准策略动作的 RMS、相邻动作变化 RMS 和 `|action|>=0.95` 饱和率；
- 静态目标的持续窗口稳定时间，或周期目标 roll/pitch 波形的最优相位滞后；
- 每个原始指标和分项分数的 mean、P50、P95、逐实例值。

固定评测禁用 Train 层 domain randomization，以 SimEnv 配置标称参数作为基准。轨迹参考
不改变控制器接口：网络仍只接收目标姿态，高度仍由外部增量式 PI 控制，位置/速度参考
只用于评分。这样轨迹误差会如实显示缺少位置/速度外环的影响，而不会暗中给自稳网络
增加训练时不存在的观测或控制权限。

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
7. **精确恢复**：连续运行 N+M 步与 N 步保存后新建 SimEnv/模型恢复 M 步，仿真真值、传感器延迟历史、VirtualPilot、collector/RNN、RNG、动作、reward、loss、优化器和最终 checkpoint 摘要逐张量一致；manifest 声明 true 并记录父 run。
8. **记录完整性**：任取一个 `instance_id`，能关联实际随机参数、物理日志、控制轨迹、episode、训练 update 和 checkpoint。
9. **配置拒绝**：未知字段、truth 泄漏、维度不符、非法时基、错误序列参数均在采样前失败。
10. **评估隔离**：插入评估不会改变后续训练动作、RNG 或优化结果。
11. **checkpoint 损坏**：摘要不符时拒绝加载，原子写入中断不破坏上一个有效 checkpoint。
12. **最小学习冒烟测试**：小批量、短 episode 的 PPO + GRU 能完成采样、更新、评估、保存和恢复，且 loss 有限。
13. **两级随机化隔离**：相同 static seed 得到逐实例相同静态参数；改变 dynamic seed 不改变静态样本，改变 static seed 不改变 SimEnv 动态随机流。
14. **静态参数保持**：仅 reset mask 对应实例重采样，未重置实例参数不变；episode 内所有物理步使用同一静态样本。
15. **动态过程归属**：训练层不生成逐步噪声；SimEnv 能按 dynamic seed 重放噪声/建模误差，并返回实际参数或可追溯日志索引。
16. **奖励计算器替换**：用两个 RewardCalculator 对同一 context 计算可得到不同奖励，但环境推进、观测、rollout schema 和算法代码均无需修改。
17. **奖励张量边界**：RewardCalculator 在 `B=1/B>1`、CPU/CUDA 下输出 `[B,1]` 同设备有限张量，热路径不发生主机同步或逐实例循环。
18. **虚拟飞手所有权**：任意改变 4 维策略动作都不能改变上桨 PWM；roll/pitch 目标满足圆盘限幅，yaw 以有界角速度积分且正确 wrap。
19. **飞手动态与隔离**：一阶惯性和增量式高度 PI 符合解析式，油门 slew 符号正确；masked reset 只重置相应实例，全部状态留在运行设备并进入 checkpoint。

## 15. 分阶段实现顺序

### 阶段 A：可验证闭环

- 配置加载/schema、运行目录、manifest 和 seed 管理；
- 训练入口/组件注册、两级随机化 schema、StaticRandomizer 和 SimEnv 动态规格注入；
- 仿真适配器、姿态跟踪任务、可注入 RewardCalculator、21 维 student 观测和动作变换；
- MLP/GRU policy、recurrent PPO、MLP-SAC GPU replay、同步批次 episode；
- 控制级/物理级完整记录、确定性评估和 checkpoint；
- shape、失效隔离、RNN reset、短跑复现测试。

### 阶段 B：规模化训练

- masked 异步 reset 的吞吐与随机流隔离压力测试；
- 高吞吐异步记录、磁盘容量预检、性能 profiling；
- curriculum、域参数分桶评估、训练分布外测试；
- 精确环境 state 恢复及中断续训。

### 阶段 C：算法与架构扩展

- 在已完成 MLP-SAC 的基础上实现 recurrent SAC/TD3 序列 replay；
- RMA/慢参数辨识网络；
- teacher 数据生成、合并蒸馏；
- INT8 量化、导出、golden vector 与 MCU 性能验收。

## 16. 首版完成定义

首版不是“能够启动训练”，而是同时满足：

- 一份完整 YAML 可以从空运行目录启动 GRU + PPO 姿态跟踪训练；
- YAML 明确给出训练主脚本、静态/动态随机化范围或尺度、独立 seed、奖励计算器、学习超参数和记录策略；物理标称值只在 SimEnv 配置中出现；
- 训练主循环不直接引用具体仿真字段、奖励实现、随机化字段或 GRU 类；
- 相同配置与环境得到可验证的确定性短跑结果；
- 任一 checkpoint 都能追溯到完整配置、源码状态、依赖、seed、实际仿真参数和数据时间线；
- RNN 状态、实例 invalid、episode 结束和 BPTT 截断语义均有自动化测试；
- 评估独立、结果可重算，best 模型选择无人工步骤；
- 对不能保证的能力（尤其仿真状态精确恢复）在 manifest 中明确声明，而不是隐式承诺。
