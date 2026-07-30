# 飞行器仿真环境接口

仿真环境只负责四类操作：根据配置批量创建相互隔离的虚拟环境、按 mask 重置指定实例、返回指定来源的观测、在控制量保持不变时推进物理仿真。环境不包含控制器或训练逻辑。所有物理计算均使用张量完成，批量维固定为第一维 `B`。

## 1. 参考系与单位

| 量 | 定义 |
| --- | --- |
| 世界系 | NED：x 北、y 东、z 地 |
| 机体系 | FRD：x 前、y 右、z 下；所有机体坐标均相对机体原点 |
| 四元数 | Hamilton `[w,x,y,z]`；`q_wb` 将机体系向量旋转到世界系 |
| 位置、速度、加速度 | m、m/s、m/s^2 |
| 质量、惯量 | kg、kg m^2 |
| 角度、角速度 | rad、rad/s |
| 转速 | rad/s |
| 力、力矩 | N、N m |
| 时间、频率 | s、Hz |
| PWM 命令 | 归一化无量纲值；电机 `[0,1]`，舵机 `[-1,1]` |

默认物理仿真频率为 5 kHz，控制输入频率为 500 Hz。两个控制时刻之间使用零阶保持。

`position_n` 和 `velocity_n` 描述质心在 NED 世界系中的位置和速度。机体原点用于定义机体几何；`center_of_mass_b`、直接推力中心和三个格栅气动中心均相对该机体原点给出。平动方程在质心处积分，转动力矩必须先用各作用点减去 `center_of_mass_b` 得到力臂。

### 1.1 批量张量约定

- 标量参数和状态的 shape 为 `[B]`，三维向量为 `[B,3]`，四元数为 `[B,4]`。
- 两台电机和三个舵机的状态分别为 `[B,2,...]` 和 `[B,3,...]`。
- 长度为 `K` 的查表为 `[B,K,2]`，格栅耦合矩阵为 `[B,3,3]`。
- 所有浮点张量使用环境创建时指定的 `dtype` 和 `device`；索引、状态码和掩码分别使用 `torch.int64`、`torch.int32` 和 `torch.bool`。
- 批量维只表示并行实例，任何物理公式都不得在维度 0 上求和、平均或归一化。
- 一个批次内的实例具有相同模型拓扑、表格长度、物理频率和控制频率；参数值、初始状态、噪声、零偏和延迟可以不同。

## 2. 可随机化量

配置中的每个数值、向量和查表节点都可写成一个 `Randomizable` 字段：

```yaml
mass:
  value: 2.40
  randomization:
    distribution: normal   # none | normal | uniform
    mode: relative         # relative | absolute
    mean: 0.0
    stddev: 0.05
    clip: [-0.20, 0.20]
```

- `value` 是标称值。
- `normal` 使用 `mean`、`stddev`，`uniform` 使用 `min`、`max`。
- `relative` 表示 `sampled = value * (1 + offset)`；`absolute` 表示 `sampled = value + offset`。
- `clip` 限制随机偏移量，而不是最终值。
- 向量或表格可以使用标量随机化参数广播到所有元素，也可以提供同形数组逐元素指定。
- 省略 `randomization` 或使用 `distribution: none` 时，实例值等于标称值。

参数随机化只在 `create` 时为每个实例独立采样一次，并在该实例的整个生命周期内保持不变。标量、向量、表格和矩阵采样后分别存为 `[B]`、`[B,...]`、`[B,K,2]` 和 `[B,...,...]`，即使某个参数未启用随机化，也必须在逻辑上广播到批量维。运行时的电机噪声和传感器噪声按各自采样频率持续生成。噪声模型中的 `stddev`、`bias`、`delay` 等参数本身仍可使用上述字段进行实例级随机化。

`physics_hz`、`control_hz`、设备数量、传感器类型、传感器插值策略和表格长度属于批次结构，不做实例级随机化。传感器 `sample_hz` 可以按实例随机化，但随机化后的每个值都必须整除 `physics_hz`。需要不同结构或不同时基时，应创建另一个批次；这样可避免按实例分支破坏张量化执行。

## 3. 创建环境 `create`

```python
@classmethod
def create(
    cls,
    config_path: str | Path,
    parallel_count: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    *,
    dynamic_randomization: Mapping[str, Mapping[str, Any]] | None = None,
    dynamic_seed: int | None = None,
) -> "SimulationEnvironment": ...
```

`parallel_count` 即批量大小 `B`，必须为正整数。`create` 读取配置、校验参数、为 `B` 个实例独立采样参数并初始化状态。返回的环境具有一个 `batch_id`，同时为每个实例生成一个 UUID，保存在 `instance_ids[B]` 中。创建后立即按 `logging` 配置记录物理时间线。

环境配置中的 `value` 是物理标称值的唯一来源。外部 `dynamic_randomization` 只能按
公开参数路径声明 `distribution`、`stddev/range`、合法范围和 `seed_stream`，禁止包含
重复的 `baseline` 或 `distribution: fixed`；不随机化时删除对应规格。正态分布默认
围绕环境已解析参数采样，uniform 使用显式实际范围。未知字段在创建时失败。

所有实际参数可通过只读属性查看：

```python
env.parallel_count   # int，值为 B
env.batch_shape      # torch.Size([B])
env.batch_id         # str，仅用于日志和追踪
env.instance_ids     # tuple[str, ...]，长度 B
env.parameters       # Mapping[str, Tensor]，所有 Tensor 第一维均为 B
```

### 3.1 顶层配置

```yaml
schema_version: 1
seed: 20260721

parallel:
  # 数量、device 和 dtype 由 create 参数传入，配置文件不重复指定
  independent_rng: true  # 必须为 true；false 为非法配置

timing:
  physics_hz: {value: 5000}
  control_hz: {value: 500}

initial_state:
  position_n: {value: [0.0, 0.0, 0.0]}
  velocity_n: {value: [0.0, 0.0, 0.0]}
  attitude_q_wb: {value: [1.0, 0.0, 0.0, 0.0]}
  angular_velocity_b: {value: [0.0, 0.0, 0.0]}

body: {}
motors: []
servos: []
aerodynamics: {}
sensors: {}
logging: {}
```

### 3.2 机体参数

```yaml
body:
  mass: {value: 2.40}                       # kg，必须 > 0
  center_of_mass_b: {value: [0, 0, 0.08]}  # m，机体系
  inertia_diagonal_b:                       # [Ixx,Iyy,Izz]，kg m^2
    value: [0.030, 0.028, 0.012]
```

只考虑机体系三轴转动惯量，不考虑惯量积：

```text
I_b = diag(Ixx, Iyy, Izz)
```

三个惯量必须为正。`center_of_mass_b` 是质心相对机体原点的位置，不得默认为机体原点。

### 3.3 推力电机参数

`motors` 必须包含两个电机，顺序固定为 `upper`、`lower`，分别对应上桨和下桨。两个电机独立配置，下列每个量均可附带 `randomization`：

```yaml
motors:
  - name: upper
    pwm_deadzone: {value: 0.08}
    pwm_to_rpm_table:
      value: [[0.00, 0.0], [0.08, 0.0], [0.50, 900.0], [1.00, 1800.0]]
    time_constant: {value: 0.030}       # s
    torque_coefficient: {value: 9.8765432e-8} # N m/(rad/s)^2
    noise:
      distribution: normal
      stddev: {value: 5.0}       # rad/s
  - name: lower
    # 其余字段同 upper；上下桨可以使用不同参数
    time_constant: {value: 0.050}
    torque_coefficient: {value: 1.10e-7}
```

字段语义：

- `pwm_deadzone`：`pwm <= pwm_deadzone` 时目标转速为 0。
- `pwm_to_rpm_table`：`[pwm, rad/s]` 分段线性查表；横轴必须严格递增，区间外钳位到端点。
- `time_constant`：该桨从转速指令到实际转速的一阶惯性时间常数；上、下桨分别配置，可以不同。加速和减速使用该桨的同一个时间常数。
- `torque_coefficient`：该桨的反扭矩平方系数，必须非负；上下桨可以不同。
- `noise`：叠加到用于推力和反扭矩计算的有效转速，每个物理步重采样；不改变电机内部转速状态。

每个桨的转速状态按下式独立更新，其中 `tau_i` 取该桨自己的 `time_constant`：

```text
d(rpm_i)/dt = (rpm_target_i - rpm_i) / tau_i
```

每个物理步内 PWM 目标保持常数，因此实现使用该一阶方程的精确离散响应：

```text
rpm_next_i = rpm_i + (1 - exp(-dt / tau_i)) * (rpm_target_i - rpm_i)
rpm_effective_i = max(rpm_next_i + noise_i, 0)
```

`rpm_effective` 只用于推力和反扭矩计算，内部 `motor_speed` 状态仍为 `rpm_next`；真值中的 `effective_motor_speed` 可用于核对噪声实际作用后的转速。两台共轴反桨电机均以产生 `+z_b` 向下气流为正转速方向。各桨反扭矩幅值和机体反扭矩向量定义为：

```text
Q_upper = torque_coefficient_upper * rpm_effective_upper^2
Q_lower = torque_coefficient_lower * rpm_effective_lower^2
M_motor_b = (-neutral_thrust_direction_b) * (Q_upper - Q_lower)
```

扭矩模型不含上下桨耦合项。只有两桨的反扭矩幅值相等时才完全抵消；系数不同意味着即使转速相同也可能存在净反扭矩。

### 3.4 舵机参数

`servos` 必须包含三个舵机，顺序固定为 `servo_1`、`servo_2`、`servo_3`：

```yaml
servos:
  - name: servo_1
    pwm_angle_table:
      value: [[-1.0, -0.35], [0.0, 0.0], [1.0, 0.35]] # [PWM, rad]
    tau: {value: 0.020}          # s
    max_speed: {value: 8.0}      # rad/s
    backlash: {value: 0.010}     # rad
    deadzone: {value: 0.015}     # 归一化 PWM
```

字段语义：

- `pwm_angle_table`：`[归一化 PWM, rad]` 分段线性查表，横轴严格递增，区间外钳位。
- `deadzone`：新 PWM 与上次有效 PWM 的差小于该值时，目标角不更新。
- `backlash`：运动反向后，累计目标角变化未超过该机械间隙时，实际输出角不响应。
- `tau`：越过死区和回差后，目标角到实际角的一阶惯性时间常数。
- `max_speed`：对一阶惯性计算得到的舵角变化率进行限幅。

舵角正方向遵循机体系右手定则；每个格栅的偏转轴由气动配置给出。

实现为每个舵机维护 `servo_effective_pwm`、`servo_command_angle`、`servo_target_angle`、`servo_motion_direction` 和 `servo_backlash_remaining`。PWM 变化未越过死区时保持有效 PWM；运动方向反转时先由后续命令角变化消耗 `backlash`，剩余角变化才传递到目标角。实际舵角使用 `(target-angle)/tau` 得到角速度，再由 `max_speed` 限幅并按物理步积分。

### 3.5 格栅气动参数

总推力分为直接推力和三个格栅分配推力四部分。四个标称比例之和必须为 1：

```yaml
aerodynamics:
  # [k1, k2, k3]；转速按 rad/s
  thrust_coefficients: {value: [4.0e-6, 4.0e-6, 2.0e-6]}
  neutral_thrust_direction_b: {value: [0.0, 0.0, -1.0]} # 单位向量
  direct_thrust_center_b: {value: [0.0, 0.0, 0.20]}      # m
  thrust_partition:
    direct: {value: 0.40}
    grid_1: {value: 0.20}
    grid_2: {value: 0.20}
    grid_3: {value: 0.20}

  grids:
    - name: grid_1
      aerodynamic_center_b: {value: [0.10, 0.00, 0.25]} # m
      deflection_axis_b: {value: [1.0, 0.0, 0.0]}        # 单位向量
      self_attenuation_curve:
        value: [[0.0, 1.0], [0.35, 0.85]] # [|舵角| rad, 推力保留比例]
      vector_deflection:
        gain: {value: 1.0}
        offset: {value: 0.0}               # rad

  coupling_attenuation:
    value: [[0.0, 0.10, 0.10],
            [0.10, 0.0, 0.10],
            [0.10, 0.10, 0.0]]
```

总推力不再视为两个单桨推力的线性叠加，而由上下桨有效转速的耦合二次式直接计算：

```text
T_total = k1 * rpm_effective_upper^2
        + k2 * rpm_effective_lower^2
        + k3 * rpm_effective_upper * rpm_effective_lower
```

`thrust_coefficients` 依次保存 `[k1,k2,k3]`。`k1`、`k2` 必须非负，并要求 `k3 + 2*sqrt(k1*k2) >= 0`，从而保证任意非负上下桨转速下总推力非负。若转速单位为 rad/s，则三个系数的单位均为 `N/(rad/s)^2`。总推力随后进入四路推力分配；两桨反扭矩不参与该分配。对格栅 `i`：

1. 由舵机机械角 `delta_i` 查表得到自身推力保留比例 `eta_i`，范围为 `[0,1]`。
2. 其他格栅造成的耦合衰减按下式计算：

```text
eta_effective_i = clip(eta_i - sum(coupling[i,j] * (1 - eta_j), j != i), 0, 1)
```

3. 机械偏转到推力矢量偏转采用一阶线性特性：

```text
vector_angle_i = gain_i * delta_i + offset_i
```

4. 该格栅分配到的推力大小为：

```text
T_i = T_total * partition_i * eta_effective_i
```

推力方向由 `neutral_thrust_direction_b` 绕 `deflection_axis_b` 旋转 `vector_angle_i` 得到。格栅力矩使用气动中心相对质心的位置计算：

```text
M_grid_i = (aerodynamic_center_b_i - center_of_mass_b) x F_grid_i
```

三个 `deflection_axis_b` 必须位于垂直于中立推力方向的平面内，并且两两夹角为 120°。正舵角按该轴的右手定则旋转机体受力方向。三个气动作用点可以使用实测坐标，不要求构成严格正三角形；它们相对质心的横向偏置和高度差都会进入上述叉积，因此能表达质心位于推力作用区域上方的倒立摆式构型。

直接推力为 `T_direct = T_total * partition_direct`，方向为 `neutral_thrust_direction_b`，作用点为 `direct_thrust_center_b`，不受格栅自身或耦合衰减影响。`coupling_attenuation` 为 3x3 矩阵，对角线必须为 0，非对角元素必须位于 `[0,1]`；若物理结构要求互易，可在配置校验中要求矩阵对称。

### 3.6 传感器参数

每个传感器至少配置噪声、零偏、延迟和采样频率：

```yaml
sensors:
  gyro:
    sample_hz: {value: 5000}       # 默认等于 physics_hz
    noise:
      distribution: normal
      stddev: {value: [0.002, 0.002, 0.002]} # rad/s
    bias: {value: [0.0, 0.0, 0.0]}             # rad/s
    delay: {value: 0.001}                       # s
```

- 首版支持的传感器字段和理想输入固定如下：

| 字段 | shape | 单位 | 加噪前理想值 |
| --- | --- | --- | --- |
| `gyro` | `[B,3]` | rad/s | `angular_velocity_b`，机体系角速度 |
| `accelerometer` | `[B,3]` | m/s^2 | `force_b / body.mass`，质心处机体系非重力比力；与同一物理步的力严格对齐 |
| `motor_speed` | `[B,2]` | rad/s | 两台电机的实际 `motor_speed` |

- 未声明的传感器名称必须在创建时失败，不能猜测其真值来源或单位。
- 省略 `sample_hz` 时使用 `physics_hz`。
- `noise` 在每个传感器采样时刻重新采样。
- `bias` 在创建时确定，此后保持不变；其 `value` 或随机化结果就是本实例零偏。
- `delay` 表示从真值采样时刻到表观值可见时刻的固定延迟。
- `sample_hz` 必须能整除 `physics_hz`；未到采样时刻时保持上一次表观值。
- 整数物理步延迟直接读取对应历史真值。非整数物理步延迟必须配置 `interpolation: linear`，在相邻两个物理步真值间线性插值；未指定时创建失败，不得隐式取整。
- 创建和 reset 时，历史缓冲区使用当时的初始真值填充，初始表观值为 `ideal + bias`；初始化本身不消耗运行时噪声样本。第一次正时间采样才叠加 counter 0 对应的噪声。

## 4. 按 mask 批量重置 `reset`

```python
def reset(
    self,
    reset_mask: torch.Tensor,  # [B], bool，与环境同设备
    config_path: str | Path,
) -> ResetResult: ...
```

`reset` 将 `reset_mask=True` 的所有槽位替换为由新配置生成的候选环境。候选参数和初始状态必须与执行以下独立创建所得的相同索引逐位一致：

```python
replacement = SimulationEnvironment.create(
    config_path,
    parallel_count=env.parallel_count,
    device=env.device,
    dtype=env.dtype,
)
```

实际实现不得创建并长期保留第二个环境对象，而应按以下顺序执行：

1. 校验 `reset_mask` 的 shape、dtype 和 device。全 false mask 直接返回，不读取或校验 `config_path`，不写重置日志。
2. 以完整 `B` 读取、随机化并校验新配置，形成所有槽位的候选参数和初始状态。
3. 检查新配置与原批次的张量结构兼容；失败时不修改原批次。
4. 对所有参数、动力学状态、执行器状态、传感器缓存和延迟队列，以广播后的 mask 执行 `torch.where` 或等价的 masked copy。
5. 以 mask 清零 `physics_step`、`control_step`、当前控制量和随机数计数器，将 `valid` 设为 true、`error_code` 设为 0。
6. 仅为选中槽位生成新 UUID，并将 `generation[B]` 在对应位置加 1。
7. 写入一次带完整 mask 的重置配置、候选参数快照和 UUID 映射事件。

未选中实例的参数、状态、时钟、UUID、generation、随机数状态和日志语义必须逐位不变。数值重置路径不得按实例执行 Python 循环；允许按固定字段集合循环，并允许在完成数值更新后遍历选中索引生成 UUID、序列化日志等非数值控制面元数据。

### 4.1 结构兼容性

为了维持固定 shape 的批量张量，新配置必须满足：

- `physics_hz` 和 `control_hz` 与原批次相同；
- 电机、舵机和格栅数量及顺序相同；
- 传感器字段集合相同；
- 每张查表的节点数相同；
- 所有参数、真值状态和传感器状态的非批量维 shape 相同。

质量、质心、惯量、查表数值、执行器参数、气动参数、传感器参数、初始状态、随机种子和随机化范围均可改变。新配置中的日志目录和分块设置不替换批次级日志设置；新配置本身仍会作为重置快照保存在原批次日志中。结构不兼容时抛出 `ConfigurationError`，原槽位保持不变。

### 4.2 返回结构

```python
@dataclass(frozen=True)
class ResetResult:
    batch_id: str
    reset_mask: torch.Tensor                 # [B], bool
    previous_instance_ids: tuple[str, ...]  # 长度 B
    instance_ids: tuple[str, ...]           # 长度 B
    generation: torch.Tensor                # [B], int64
```

返回结果覆盖完整批次，便于调用方继续保留张量化控制流。批次槽位索引在整个批次生命周期内不变；UUID 标识该槽位中一次具体的环境生命周期。初始创建时所有槽位的 `generation=0`，每次成功重置后仅在 `reset_mask=True` 的位置加 1。

## 5. 获取观测 `observe`

```python
def observe(
    self,
    source: Literal["truth", "sensor"],
    fields: tuple[str, ...] | None = None,
) -> Observation: ...
```

`observe` 一次返回全部 `B` 个实例的当前观测，不推进仿真或重新采样噪声。同一时刻重复调用返回相同结果。

- `truth`：位置、速度、姿态、角速度、加速度、电机实际转速、舵机实际角度、力和力矩等内部真值。
- `sensor`：配置启用的传感器经过噪声、零偏和延迟后的表观值。
- `fields=None` 返回该来源的全部字段；请求不存在的字段必须报错。

```python
@dataclass(frozen=True)
class Observation:
    batch_id: str
    instance_ids: tuple[str, ...]
    physics_step: torch.Tensor  # [B], int64
    control_step: torch.Tensor  # [B], int64
    sim_time_s: torch.Tensor    # [B]
    source: Literal["truth", "sensor"]
    values: Mapping[str, torch.Tensor] # 每个值的第一维均为 B
    valid: torch.Tensor                # [B], bool
```

例如，姿态真值为 `[B,4]`，两台电机转速为 `[B,2]`，三个舵机角为 `[B,3]`。接口不提供“去掉批量维”的特殊情况；即使 `B=1`，仍返回 `[1,...]`。

## 6. 推进时间步 `advance`

```python
def advance(
    self,
    control: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> AdvanceResult: ...

# control.shape == [B,5]
# control[:,0:2]：upper/lower motor PWM，范围 [0,1]
# control[:,2:5]：servo_1/2/3 PWM，范围 [-1,1]
# active_mask.shape == [B]；省略时全部为 True
```

`control` 的 dtype、device 和批量大小必须与环境一致。环境逐实例检查越界和非有限控制量：合法且 `active_mask=True` 的实例推进；未激活实例保持状态和时钟不变；输入非法或已经数值失败的实例标记为无效并冻结，不得阻止其他实例推进。控制量在整个控制周期内保持不变。

环境依次更新电机、舵机、气动力/力矩、六自由度状态和传感器，并将每个物理子步交给日志采样器。`full` 模式逐步保存；`compact` 模式按显式 stride/fields 保存。`physics_hz=5000`、`control_hz=500` 时，每个激活实例一次调用准确推进 10 个物理步和 2 ms。

```python
@dataclass(frozen=True)
class AdvanceResult:
    batch_id: str
    instance_ids: tuple[str, ...]
    physics_step: torch.Tensor           # [B], int64
    control_step: torch.Tensor           # [B], int64
    sim_time_s: torch.Tensor             # [B]
    physics_steps_advanced: torch.Tensor # [B], int64；未激活实例为 0
    valid: torch.Tensor                  # [B], bool
    error_code: torch.Tensor             # [B], int32；0 表示正常
```

返回值只报告时间推进结果；观测必须通过 `observe` 显式读取。

### 6.1 完整动态状态与精确恢复

```python
state = env.state_dict()

restored = SimulationEnvironment.create(
    same_config,
    parallel_count=env.parallel_count,
    device=env.device,
    dtype=env.dtype,
    dynamic_randomization=same_dynamic_spec,
    dynamic_seed=same_dynamic_seed,
)
restored.load_state_dict(state)
```

`state_dict` 的 schema version 1 包含所有会影响未来数值轨迹的状态：

- 当前完整参数，包括 episode 静态参数和创建时动态随机化参数；
- 六自由度 truth、上下桨实际/有效转速、舵机死区/回差/方向状态及气动力矩中间量；
- 当前传感器表观输出和每种传感器的延迟环形历史缓冲；
- 当前保持的 5 维控制量、物理步、控制步、valid、error code 和 generation；
- 每个实例的随机 seed，以及电机和各传感器独立随机流的 sample counter；
- batch size、dtype、时基、配置 SHA-256、动态参数字段和 dynamic seed 兼容元数据。

`load_state_dict` 在任何写入前严格校验 schema、配置摘要、batch、dtype、时基、字段、
shape、有限性、传感器历史容量和动态随机化身份。checkpoint 可以通过 `map_location=cpu`
加载；通过校验后张量会复制到目标环境 device。字段缺失或不兼容时必须失败，不能使用
默认初始状态填补。

`batch_id`、`instance_ids` 和日志线程不参与物理数值演化，不从旧状态覆盖到新环境。
恢复后的环境保留新日志身份，并追加 `event_code=3` 的 resume 边界。验收要求连续运行
N+M 步与运行 N 步、保存、创建新环境、恢复后运行 M 步的 truth、传感器、随机计数器
和下一状态逐张量完全一致。

## 7. 日志与校验

创建批次时建立 `logs/<batch_id>/`，保存 `instance_ids`、原始配置、shape 为 `[B,...]` 的实际参数和 `[time,B,...]` 分块时间线。`logging.mode=full` 时默认逐物理步保存全部控制、真值、传感器和状态字段，可用于逐步复盘；`compact` 时由 `physics_step_stride` 和 `fields` 显式声明降采样与字段裁剪，不能宣称物理级逐步可重放。初始化和恢复事件不受 stride 影响，始终写入共享时间线；`full` 模式的重置事件也写入共享时间线，`compact` 模式的重置后状态则按选中实例稀疏写入对应 reset snapshot 的 `post_reset_timeline`，避免为少量重置复制完整批次。实际模式、stride、字段清单写入 `metadata.json`。`event_code=-1` 表示该日志行对该槽位无事件，0 表示物理调度步，1 表示初始状态，2 表示重置事件，3 表示从完整动态状态恢复。

每次非空重置还必须追加一条 UUID 映射事件，并保存 `reset_mask[B]`、该次重置的配置快照和参数。`full` 模式保存完整 `[B,...]` 候选参数；`compact` 模式保存 `parameter_instance_indices` 以及 mask 选中行，避免高并行训练反复复制未重置槽位。通过 `instance_index + generation` 可以将共享时间线无歧义地映射到对应 UUID。共享存储只是写入优化，不改变实例的逻辑隔离。

创建时必须校验：`parallel.independent_rng` 为 true；质量和三轴惯量为正；四元数已归一化；所有表格横轴严格递增；上下桨时间常数为正；反扭矩系数非负；推力系数组合对非负转速不产生负推力；推力分配比例之和为 1；所有气动作用点和质心采用同一机体系；推力方向与格栅轴均为单位向量；衰减比例合法；传感器频率可由物理时间步调度；`physics_hz / control_hz` 为正整数。

相同配置、seed、`parallel_count` 和控制序列必须产生相同的实例参数、状态和传感器序列。检测到 `NaN`、`Inf` 或物理约束错误时，只冻结对应实例并记录具体字段；除非日志设备或执行设备发生批次级故障，否则不得停止其他实例。

## 8. 全张量化实现方法

### 8.1 数据组织

环境采用 Structure of Arrays，而不是创建 `B` 个 Python 飞行器对象。例如：

```python
mass                    # [B]
center_of_mass_b        # [B,3]
inertia_diagonal_b      # [B,3]
motor_rpm               # [B,2]
servo_angle             # [B,3]
position_n              # [B,3]
attitude_q_wb           # [B,4]
grid_aero_center_b      # [B,3,3]
coupling_attenuation    # [B,3,3]
```

所有数值物理参数必须是带批量维的张量；UUID、字段名和单位等非数值元数据可以保留在 CPU。未随机化的参数可由标称值 `expand` 到 `[B,...]`，但对外 shape 和随机化参数完全相同。运行期间参数只读，避免共享 view 导致实例间修改串扰。

这种布局可行的原因是：同一批次的每个实例执行相同的动力学方程，差异只体现在张量元素的参数值上。因此一次逐元素运算、叉积或批量矩阵运算即可同时更新全部实例，不需要按实例调用 Python 对象。

### 8.2 动力学与执行器

每个物理子步依次执行：电机状态、舵机状态、四路推力与力矩、刚体加速度、速度/角速度、位置/四元数、传感器。控制量在整个控制周期内零阶保持。所有查表均使用端点钳位的批量分段线性插值。

四路机体系力及总力矩为：

```text
F_direct_b = T_total * partition_direct * neutral_direction_b
F_grid_i_b = T_total * partition_i * eta_effective_i * rotated_direction_i_b

M_direct_b = (direct_center_b - center_of_mass_b) x F_direct_b
M_grid_i_b = (grid_center_i_b - center_of_mass_b) x F_grid_i_b

F_b = F_direct_b + sum(F_grid_i_b, i=1..3)
M_b = M_direct_b + sum(M_grid_i_b, i=1..3) + M_motor_b
```

当前接口没有定义机体阻力、风场、地面接触或其他外力参数，因此实现不会添加未配置的经验阻尼。世界系平动方程和机体系转动方程为：

```python
force_n = rotate_body_to_world(attitude_q_wb, force_b)
linear_acceleration_n = force_n / mass[:, None] + gravity_n

i_omega = inertia_diagonal_b * angular_velocity_b
angular_acceleration_b = (
    moment_b - torch.linalg.cross(angular_velocity_b, i_omega)
) / inertia_diagonal_b
```

其中 NED 重力为 `gravity_n=[0,0,9.80665] m/s^2`。平动和角速度使用半隐式 Euler：先更新速度，再用新速度更新位置；Hamilton 四元数满足 `q_dot=0.5*q⊗[0,omega_b]`，每步更新后重新归一化。四元数积分使用更新后的机体系角速度。

上下桨各自的时间常数、PWM 死区、舵机死区、回差、限速和实例有效状态都使用 `[B,...]` 张量或布尔掩码实现，不能使用基于单个实例值的 Python `if`。耦合推力二次式和两个独立反扭矩平方项同样在整个批次上直接计算。

分段曲线保持统一节点数 `K`，使用批量 `torch.searchsorted`、`torch.gather` 和线性插值计算。表格横轴和值均保留批量维 `[B,K]`；不允许在推进过程中调用 NumPy/SciPy 插值器或遍历实例。

格栅力为 `[B,3,3]`，分别表示 B 个实例、3 个格栅和 3 个机体系分量。耦合衰减使用 `[B,3,3]` 矩阵运算，力矩通过批量叉积计算，最后只在格栅维求和：

```python
grid_moment_b = torch.linalg.cross(
    grid_aero_center_b - center_of_mass_b[:, None, :],
    grid_force_b,
).sum(dim=1)  # 只对三个格栅求和，绝不对 B 求和
```

时间线中的动力学分量至少包括 `motor_speed[B,2]`、`effective_motor_speed[B,2]`、`total_thrust[B]`、`motor_torque[B,2]`、`direct_force_b[B,3]`、`grid_force_b[B,3,3]`、`direct_moment_b[B,3]`、`grid_moment_b[B,3,3]`、`motor_reaction_moment_b[B,3]`、`force_b[B,3]` 和 `moment_b[B,3]`。`motor_torque` 保存上、下桨各自的非负反扭矩幅值，最终机体轴向反扭矩取二者之差；`total_thrust` 是含 `k3` 耦合项的整体推力，不提供可线性求和的单桨推力字段。

物理子步在时间上存在严格前后依赖，因此允许对固定的 `substeps = physics_hz // control_hz` 做循环；循环体内部必须完全张量化，不得包含 `for instance in range(B)`。固定子步循环可由 `torch.compile` 捕获并融合；是否启用自动微分由调用方决定，不影响接口。

### 8.3 传感器与延迟

每种传感器使用批量真值历史环形缓冲区 `[B,L,...]`，其中 `L=ceil(max(delay * physics_hz))+1`，在创建时固定。写入位置由每个实例自己的 `physics_step[B] % L` 得到，整数延迟和线性插值所需的两个历史位置均通过批量索引读取，不维护共享的标量写指针。

不同实例的传感器采样率使用 `[B]` 到期掩码更新；未到采样时刻的实例保留上次输出。模型拓扑和传感器种类必须相同，但采样率、延迟、噪声和零偏可以按实例不同。每种传感器的插值策略是批次结构，不能按实例分支。reset 配置的延迟不得超过创建时已分配的历史容量；在容量以内可以通过 mask 改变选中实例的延迟，选中行的整段历史用新初始真值填充，未选中行逐位不变。

### 8.4 随机数隔离

不得在推进函数中使用 Python 全局随机数，也不能让某个实例是否激活改变其他实例随后获得的随机数。推荐使用无状态、计数器式随机数：

```text
random_key = hash(base_seed, instance_index, subsystem_id, sample_counter)
```

其中 `subsystem_id` 区分参数随机化、电机 1/2 和各传感器，`sample_counter` 为 `[B]` 独立计数。传感器层用这些张量 key 直接生成完整 `[B,...]` 标准正态噪声，再用采样到期掩码提交并只递增到期实例的 counter。这样暂停、冻结、reset 或修改实例 `i` 不会改变实例 `j` 的随机序列；增加批量大小也不会改变原有索引实例的序列。

若首版只使用单个 `torch.Generator`，也必须每个物理步为所有 B 个实例生成固定 shape 的噪声，不能根据 `active_mask` 改变随机数消耗量。但该方案只保证固定批次下可复现，不满足严格的实例随机流隔离，因此不作为最终实现。

### 8.5 故障隔离

环境维护 `valid[B]` 和 `error_code[B]`。每个物理子步计算候选新状态后，先逐实例检查有限性和物理边界，再通过掩码提交：

```python
commit = active_mask & valid & candidate_is_finite
state = torch.where(commit[..., None], candidate_state, state)
valid = valid & candidate_is_finite
```

实际实现应按各张量维数正确扩展掩码。任何跨实例统计量都只能用于诊断，不能反馈到动力学、传感器或随机数状态。这样一个实例的异常值、暂停、参数和控制输入不会改变其他实例。

### 8.6 张量化日志

每个物理步直接写入设备上的预分配日志块 `[chunk_steps,B,...]`。块满后使用 pinned memory 和非阻塞复制交给后台写线程，推进路径中禁止逐实例写文件、调用 `.item()` 或每步执行同步 `.cpu()`。日志块带有 `active_mask`、`valid` 和 `instance_index`，因此暂停实例和失败实例仍可被准确解释。

若日志写入速度低于仿真速度，环境必须按配置选择阻塞或报告溢出错误，不得静默丢弃记录。日志缓冲区大小属于批次结构，在创建时完成分配。`logging.minimum_free_space_bytes` 指定日志写入后必须保留的磁盘余量（默认 512 MiB）；每个 tensor 文件写入前按实际 storage 大小预检，并通过临时文件、`fsync` 和原子替换发布。余量不足时抛出 `InsufficientDiskSpaceError`，不发布半截 timeline，调用方应在最近的训练安全边界停止。

### 8.7 验收标准

1. 任意数值参数的实际值都具有第一维 `B`，观测和控制分别保持 `[B,...]` 与 `[B,5]`。
2. 核心推进代码中不存在按实例 Python 循环，也不存在 CPU/NumPy 往返。
3. 将实例 `i` 的参数、控制或 `active_mask` 改变后，其他实例在相同随机 key 下的状态和观测逐位不变。
4. 单个实例产生非法输入或数值发散时，仅该实例冻结并返回非零 `error_code`。
5. `B=1` 与较大批次中索引 0 的结果一致；扩大批量不会改变已有实例的随机序列。
6. CPU 与 CUDA 均能执行相同接口；同一设备和 dtype 下重复运行结果确定。
7. 日志可以按 `instance_index` 完整还原任意实例的参数、输入、状态和观测时间线。
8. 使用任意 `reset_mask[B]` 重置后，选中行与相同配置独立创建的 `B` 批环境中同索引行一致，未选中行逐位不变；数值更新路径不存在按实例循环。
9. 结构不兼容的重置在写入前失败；成功的非空重置会更新选中行的 UUID 和 generation，并保存一次带 mask 的配置快照和事件日志；全 false mask 是空操作。
