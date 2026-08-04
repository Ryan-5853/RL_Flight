# RL Flight WebUI

轻量化飞行仿真可视化控制台。浏览器负责读取 Windows 电脑上的手柄，远程
Python 服务负责在 CPU 上串联统一 Controller、单实例 SimEnv 和低频遥测。
Controller 可以选择 PID、LQR、PID+LQR 混合控制器，或由部署层加载的神经网络
推理包。实时服务不从 Train 构造模型。

## 本地预览

需要使用服务器本地配置和 CPU 运行时时，请使用内置服务启动：

```bash
python server.py --host 0.0.0.0 --port 8080
```

然后访问 `http://localhost:8080`。

默认加载器直接使用仓库 `Deploy` 提供的 `flight_deploy.PolicyRuntime`，只读取已经
导出的完整性校验 bundle，不读取训练 checkpoint。推理包固定放在仓库相对目录
`WebUI/artifacts/`；服务启动后自动递归发现包含 `manifest.json` 的 bundle，页面只需
从“推理包路径”下拉框选择：

```bash
python server.py --host 0.0.0.0 --port 8080
```

需要替换后端时可增加
`--inference-loader my_deployment.loader:load_package`。加载函数签名为
`(path: Path, device: torch.device, dtype: torch.dtype) -> RealtimeInferencePackage`。
推理包必须声明固定的 21 维基础观测、`residual_4`、`physical_5` 或
`coaxial_differential_cyclic_3` 输出契约，并实现
`infer/reset/warmup/close/describe`。默认 `flight_deploy` 适配器还会读取 manifest
中的 observation history、归一化和 action transform；例如 61 帧 uniform MLP 会在
适配层组成 `21×61=1281` 维运行时输入。PID、LQR 和混合控制器不读取推理包，它们从
独立的控制器辨识模型求配平、控制分配矩阵和 LQR 增益；该模型可手动设置，也可在
会话创建时从 SimEnv 实际参数复制快照。普通静态文件服务器只能预览
页面，无法使用配置浏览和 CPU 运行时接口。

角加速度级联 bundle 使用 `angular_acceleration_cascade_v1` contract。WebUI 加载后会
自动在神经网络外构造 manifest 指定的姿态 PID、角加速度后向差分、61 帧历史和
共轴差速/cyclic 分配；上层仍选择 `controller.type: neural`，无需在测试 YAML 中重复
PID 增益。bundle 要求的 `control_hz` 与 SimEnv 不一致时会拒绝启动。
专用 bundle 还记录排除 seed、初始状态和日志后的 SimEnv 兼容指纹；机体、执行器、
气动或传感器契约不匹配时同样拒绝启动。训练使用的环境可从服务器配置根
`train_environment` 导入。

带有 `simulator_compatibility.configuration` 的专用部署包会在 WebUI 中被选中时
自动回填其训练 SimEnv。手工导入仍可用于检查或调整初始状态；运行时只对有效
单实例动力学做兼容校验，不把日志、随机种子、初始状态或零延迟插值等表示差异
误判为动力学不兼容。

交互仿真默认使用 SimEnv 的 `RealtimeSimulationEnvironment`：固定单环境 CPU
执行、编译动力学和传感器内核，并关闭 500 Hz 热路径中的持久化日志。上传配置中的
`logging.directory` 不会被交互会话采用，session status 中
`persistent_logging=false`、`log_directory=null`。`--runtime-log-root` 仍作为服务端
管理的兼容目录参数保留，但默认实时会话不会在其中创建时间线文件。

## 当前能力

- Canvas 三维 NED 坐标、位置轨迹、带数值的自适应参考网格、飞行器姿态与交互视角；
- 仿真启动、暂停、单步与重置；
- 手柄输入、CPU 策略推理、CPU 单实例 SimEnv 推进和真实仿真遥测链路；
- 基于正式虚拟飞手配置的离线完整 rollout、可视化视频合成与本地导出；
- 统一 `ControllerState / ControllerReference / [B,5] command` 控制器抽象；
- 自动悬停配平、PID、离散 LQR 和 PID→LQR 平滑接管；
- 依据 SimEnv 与 Train 模板字段生成单环境测试配置；
- 参数修改状态、恢复默认值、YAML 导出和启动前自动生成；
- 左右侧边栏独立折叠，中央三维视图自动扩展；
- Windows 浏览器本机 Gamepad API 读取、热插拔检测与多设备选择；
- 引导式九动作校准：油门最大/最小、偏航左右、俯仰上下、横滚左右及回中；
- 自动识别前四轴的 Roll、Pitch、Yaw、Throttle 映射和正反方向；
- 手动轴重映射、逐通道反向、输入死区和标准化控制帧接口。

页面尚未启动 CPU 会话时会保留少量视觉演示动画；会话启动后，飞行器姿态、
电机、舵机、三个格栅力矩以及合力/合力矩均由后端真值遥测覆盖。

## 实时运行链路

```text
Windows Gamepad API（未连接时使用零指令虚拟输入）
        │  标准化 roll / pitch / yaw / throttle（约 60 Hz）
        ▼
同源 HTTP control API
        │  只更新“最新指令”，不驱动仿真时钟
        ▼
服务器独立控制线程（RealtimeSimulationEnvironment，目标 500 Hz）
        │
        ├─ CPU 摇杆滤波、目标姿态与油门斜率限制
        ├─ PID / LQR / PID+LQR：读取独立辨识模型并直接输出 5 维命令
        ├─ Neural：CPU 21 维观测和 MLP/GRU 推理
        └─ CPU SimEnv.advance([上桨, 下桨, 舵机1, 舵机2, 舵机3])
        │
        ▼
持久 telemetry stream（默认 60 Hz）→ Canvas 可视化
```

控制热路径中的状态、观测、循环网络 hidden、动作和仿真状态都保留在 CPU，不再
创建 CUDA stream、pinned memory 或设备间复制。会话创建时先编译并预热 SimEnv
动力学、传感器、有限性检查和状态提交内核；预热结束后恢复完整初始数值状态，再
启动控制线程。实时入口不创建磁盘日志线程，浏览器所需真值只在遥测边界降采样打包。
神经控制器由部署加载器在会话创建时载入一次；WebUI 不导入 Train 的模型构造器。
推理包下拉框会标记 `DIRECT`、`CASCADE` 或 `CUSTOM`，并记住用户明确选择的包；首次
打开优先选择直接姿态控制包，不再因为文件修改时间变化而静默切换控制架构。

CPU 神经热路径会清零无物理意义的次正规浮点数。角加速度级联包把姿态 PID、差分估计
和特征构造融合为编译内核；神经 Hover 高度环采用与传统控制器相同参数和公式的 B=1
专用实现，保留高度闭环但不再每步调度完整传统控制器植物模型。每次创建新会话都会
清空延迟统计窗口，避免旧控制器的 p95/max 污染新会话。

服务端只允许存在一个交互式仿真会话。浏览器暂停时保留该会话和仿真状态；
配置未改变时再次启动会继续原会话，配置改变后会关闭旧会话并按新配置创建。
手柄数据超过 `runtime.command_timeout_ms` 未更新时，服务端会自动暂停并报告
`controller_input_timeout`，避免失联后继续使用旧指令。

当前网络边界采用标准库 HTTP：手柄的 `requestAnimationFrame` 输入事件会立即上传，
最多保留两个 in-flight 最新值请求；遥测使用持久 chunked NDJSON stream，并由后端
条件变量在新快照发布时立即唤醒，不再做 10 ms 轮询。长轮询仅作为浏览器或代理不支持
streaming body 时的兼容回退。网络请求频率不会决定 500 Hz 的 CPU 控制循环。

## 离线 rollout 视频导出

顶部“导出视频”会使用右侧当前选择的环境、控制器、部署推理包和虚拟飞手
参数，生成一段完整 episode 视频。该链路不是对远程实时画面录屏：

```text
当前 SimEnv / Controller / VirtualPilotCommandSource v2 配置
        │
        ▼
服务器后台离线 rollout（每个 500 Hz 控制周期都执行）
        │
        ├─ 虚拟飞手按 seed、目标保持时间和 centered 分布采样打杆
        ├─ 高度增量 PI、摇杆一阶滤波和油门斜率限制
        ├─ 当前 PID / LQR / Hybrid / Neural 控制器闭环推进
        └─ 按所选 24 / 30 / 60 FPS 保留紧凑可视化快照
        │
        ▼
一次性下载 rollout 数据 → 浏览器本地 Canvas + MediaRecorder 合成视频
```

因此 SSH、公网 RTT 和浏览器遥测频率都不会进入控制闭环。服务器按 CPU 可达到
的最快速度完成采样；采样完成后，浏览器按视频时长在本机完成编码。导出画面为
16:9，中央是飞行器位置轨迹、带 N/E 刻度数值的自适应网格、姿态与力/力矩
可视化，右侧是 Roll、Pitch、Yaw 姿态误差随时间曲线，底部包含虚拟飞手左右
摇杆、四通道数值、五通道控制器输出以及 NED 位置、相对起点位移、高度和角速度。
浏览器支持 H.264/MP4 时优先导出 MP4，否则自动回退到 VP9/VP8 WebM。每次视频
合成前会清空交互画面的旧轨迹，并用本次 rollout 帧从起点重新累积，确保导出与
实时画面采用相同的坐标映射和缩放规则。

导出时可以选择同时保留原始 rollout JSON。若仿真提前触发 `tilt_limit`、
`angular_rate_limit` 或 `simenv_invalid`，采样会在终止状态停止并把终止原因
写入数据和视频，而不会先自动 reset。正常达到 `task.episode_duration_s` 时以
`episode_timeout` 完成。

虚拟飞手参数与 Train 当前 schema 保持一致，使用
`VirtualPilotCommandSource version: "2"`。高度控制配置位于
`command_source.params.throttle.height_controller`，包含目标高度、初始油门
范围、增量 PI 增益和误差限幅。

### 运行配置

右侧“统一控制器”可以选择：

- `hybrid_pid_lqr`：默认；启动/大误差由 PID 捕获，配平附近平滑切换到 LQR；
- `pid`：高度和姿态全部使用传统 PID/PD；
- `lqr`：高度使用 PID，姿态与执行器动态使用离散 LQR；
- `neural`：调用服务器部署层提供的 `RealtimeInferencePackage`。

所有控制器的实时 `collective_mode: hover` 都使用统一的
`controller.params.pid.altitude.{kp,ki,kd}` 高度 PID，并把 reset 时的位置作为高度目标。
神经网络控制器复用同一高度 PID 和推力模型生成上电机 collective，部署包须使用
保留外部 collective 通道的 `residual_4` 或 `coaxial_differential_cyclic_3`；策略继续
计算其余通道。`physical_5` 直接拥有两个电机，因此不能与 hover 高度 PID 叠加。
`manual` 将手柄油门映射为上桨 PWM；传统
控制器同时计算反扭矩平衡的下桨基准。`command_source.params.throttle.height_controller`
仅供训练和离线 VirtualPilot rollout 使用。没有手柄时页面会发送零姿态虚拟
输入，因此默认混合控制器可直接启动并观察自稳。页面默认初态故意设置为约
`roll=8° / pitch=-6°` 并带有小角速度；点击“启动仿真”即可看到 PID 捕获和 LQR
接管，不需要先修改参数。

`controller.params.flight_mode` 可选择统一参考模式：

- `attitude`：roll/pitch 手柄直接生成目标姿态；
- `position`：公共位置外环根据 N/E 位置、速度误差生成目标水平加速度，再结合当前
  偏航转换成目标 roll/pitch；传统和神经网络姿态控制器消费完全相同的姿态参考。

位置外环通过 `controller.params.position` 配置 `kp`、`kd`、最大水平加速度和最大
倾角，并要求 `collective_mode: hover` 负责垂直位置。切换到 `position` 并应用配置后，
可在三维视口中单击设置水平目标点；点击会在
当前目标高度平面反投影，因此只改变 N/E，D（高度）保持不变。拖拽仍用于旋转视角，
俯视和透视模式均可设置目标。Reset 会把目标点恢复到重置位置。

传统控制器的物理模型位于 `controller.params.model_parameters`，与 SimEnv 实际动力学
参数完全分离。页面默认使用 `source: manual`，因此质量、惯量、重心、执行器标定与
时间常数、推力/反扭矩系数和气动分配几何都可以作为辨识值单独修改。点击
“从 SimEnv 复制到辨识模型”会把参数面板中的当前 SimEnv 配置复制过来，并保持
`manual`，便于只改一项制造已知失配；选择 `source: synchronized` 则在创建会话时
直接复制该仿真实例经过随机化后的实际参数。两种方式得到的都是控制器私有快照，
后续 SimEnv reset/参数重采样不会悄悄改变已经计算好的 PID 增益、Hover 配平、控制
分配矩阵或 LQR 增益。状态栏会显示 `MANUAL`/`SYNC SNAPSHOT`、失配参数数量，并在
悬停提示中同时给出实际质量和控制器模型质量。

右侧“推理与观测”中包含以下运行时字段：

- `runtime.checkpoint_path`：仅神经网络控制器需要；服务器允许根内的部署推理包路径；
- `runtime.compile_kernels`：默认启用 SimEnv 实时动力学与传感器内核编译；
- `runtime.warmup_steps`：启动控制线程前执行并完整回滚的预热步数，默认 3；
- `runtime.spin_us`：绝对 2 ms deadline 前的短自旋窗口，默认 200 μs；
- `runtime.execution_hz`：墙钟控制循环目标，默认与 SimEnv 时基一致为 500 Hz；
- `runtime.telemetry_hz`：CPU 真值序列化并推送前端的最高频率；
- `runtime.command_timeout_ms`：最近一个有效手柄帧的最大允许年龄。
- `runtime.cpu_threads`：PyTorch CPU 算子的 intra-op 线程数；单环境默认使用 1。

`run.device` 必须是 `cpu`，服务端不会在交互运行时占用 CUDA。
推理包 metadata 必须与固定 21 维基础实时观测和控制器输出模式一致；Deploy manifest
声明的历史长度、字段归一化、配平和 residual scale 也会被逐项验证，否则会话创建直接
失败，避免只匹配张量 shape、但语义错误的控制器进入闭环。

左侧 `CTRL RATE` 显示后端按完成控制步实测的频率，延迟显示为对应的平均控制
周期；它们不再使用前端模拟的 GPU 负载或固定延迟。默认墙钟执行目标现在与 SimEnv
时基一致，为 500 Hz。实际能否达到目标取决于控制器、CPU 和操作系统调度；
`loop_overruns` 和 `real_time_factor` 会如实反映完整 WebUI 闭环，而不是只反映纯
SimEnv benchmark。单环境默认使用一个 PyTorch intra-op 线程。

session status 还会返回：

- `simulation_backend: simenv-realtime-single-v1`；
- `simulation_compiled`：实时内核是否已安装编译包装；
- `simulation_warmup_s`：创建会话时的首次编译和预热耗时；
- `persistent_logging: false`：确认实时热路径没有磁盘日志。

可在目标机器上先绕过浏览器，测量完整“控制器 + SimEnv + 60 Hz 遥测打包”链路：

```bash
PYTHONPATH=../SimEnv/src:../Controller/src:../Deploy/src:. \
python benchmark_runtime.py \
  ../SimEnv/configs/example.yaml \
  /path/to/controller_test.yaml \
  --seconds 60
```

神经控制器额外传入 `--inference-package /path/to/bundle`。该脚本与训练进程无关，
不会创建训练环境或读取训练 checkpoint。

### Runtime API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/runtime/capabilities` | CPU、PyTorch、逻辑核心数、推理加载器状态及允许根 |
| `GET` | `/api/runtime/checkpoints` | 旧版候选文件枚举接口；部署包也可直接填写允许根内路径 |
| `POST` | `/api/runtime/sessions` | 上传两份 YAML 和服务器推理包路径，创建唯一会话 |
| `POST` | `/api/runtime/sessions/{id}/control` | 更新带单调序号的四通道手柄指令 |
| `POST` | `/api/runtime/sessions/{id}/start` | 在收到新鲜控制帧后启动/继续 CPU 循环 |
| `POST` | `/api/runtime/sessions/{id}/pause` | 暂停并保留当前环境与 GRU 状态 |
| `POST` | `/api/runtime/sessions/{id}/step` | 在暂停状态推进一个控制周期并立即发布遥测 |
| `POST` | `/api/runtime/sessions/{id}/reset` | 重置单环境、目标、上一动作及循环状态 |
| `GET` | `/api/runtime/sessions/{id}/status` | 查询频率、步数、episode、超时和循环超限计数 |
| `GET` | `/api/runtime/sessions/{id}/telemetry` | 读取新遥测；支持 `after` 和 `timeout` 长轮询参数 |
| `DELETE` | `/api/runtime/sessions/{id}` | 关闭环境、日志线程并释放 CPU 会话资源 |
| `POST` | `/api/runtime/rollouts` | 创建完整虚拟飞手离线 rollout 作业 |
| `GET` | `/api/runtime/rollouts/{id}` | 查询采样进度、状态和失败原因 |
| `GET` | `/api/runtime/rollouts/{id}/data` | 采样完成后一次性读取视频帧数据 |
| `DELETE` | `/api/runtime/rollouts/{id}` | 取消正在执行的离线 rollout |

创建会话的请求体如下：

```json
{
  "simenv_yaml": "schema_version: 1\n...",
  "test_yaml": "schema_version: 2\n...",
  "checkpoint_path": null
}
```

字段名 `checkpoint_path` 为兼容已有页面配置暂时保留，其语义已经变为部署推理包路径。
传统控制器创建会话时可以省略该字段。遥测在原有 `truth/runtime` 外增加
`controller.command`、控制器诊断量和实际参考指令，例如：

```json
{
  "controller": {
    "type": "hybrid_pid_lqr",
    "command": [[0.73, 0.69, 0.01, -0.02, 0.01]],
    "diagnostics": {"controller.lqr_blend": [1.0]}
  },
  "reference": {
    "target_attitude_q_wb": [[1, 0, 0, 0]]
  }
}
```

控制帧只包含校准后的四个逻辑通道，额外摇杆轴和按键不会进入推理接口：

```json
{
  "sequence": 1024,
  "channels": {
    "roll": 0.12,
    "pitch": -0.08,
    "yaw": 0.03,
    "throttle": 0.46
  }
}
```

## 手柄访问要求

手柄由打开页面的 Windows 浏览器直接读取。使用远程服务器部署页面时必须启用 HTTPS；开发时可使用 Windows 本机的 `localhost`。通过普通远程 HTTP 地址打开页面时，浏览器可能禁止 Gamepad API。

连接手柄后需要按下任意按键让浏览器激活设备。校准向导默认只处理前四个轴，不要求校准额外轴、方向键或按钮。校准数据保存在当前浏览器的 `localStorage` 中，不会上传到服务器。

三个姿态通道采用航空器右手系约定：右杆向右输出正 Roll；右杆向前/屏幕上方
推动时输出负 Pitch，命令机头下俯；左杆向右输出正 Yaw。手柄配置从
`rl-flight.gamepad.v4` 开始统一采用该逻辑语义，硬件原始轴的正反方向只由默认
映射或校准结果处理。校准向导会根据实际采集的左右、前后端点重新判断每根轴的
极性，因此使用其他手柄或非标准轴布局时，重新运行一次校准即可。

也可以通过以下接口读取最新标准化控制帧：

```js
const frame = window.RLFlightGamepad.getFrame();
// frame.channels: { roll, pitch, yaw, throttle }
// frame.axes: 所有经过校准的标准化轴
// frame.buttons: 按钮值与按下状态
```

也可以监听每一帧输入事件：

```js
window.addEventListener('rlflightcontrollerframe', (event) => {
  const frame = event.detail;
});
```

## 配置生成

右侧参数来自 `SimEnv/configs/template.yaml` 和测试相关的 Train 配置字段。纯训练字段（并行环境数、训练预算、PPO optimizer、collector）不进入单环境控制器测试页面。

点击“生成配置”或“启动仿真”会生成两份内存中的 YAML：

- `simenv_webui_<timestamp>.yaml`：单实例物理环境配置；
- `controller_test_<timestamp>.yaml`：推理、命令映射、任务、安全终止和模型配置。

后端接入层可以监听启动事件并上传这两份配置：

```js
window.addEventListener('rlflightsimulationstart', (event) => {
  const { simenv, test } = event.detail.configuration;
  // simenv.filename / simenv.yaml / simenv.config
  // test.filename / test.yaml / test.config
});
```

点击配置区底部的下载按钮可以把两份 YAML 导出到 Windows 本地，用于人工检查。

## 飞行器视觉与力矩遥测

中央模型使用直径与高度约 `2:1` 的简化圆筒，包含上下两层共轴反桨和三组互成约 120° 的气动格栅。质心、直接推力中心、格栅气动中心及偏转轴由当前 SimEnv 配置更新。

SimEnv 使用右手 FRD/NED 坐标，Canvas 使用右手 FLU/NWU 视觉坐标。所有机体几何、
机体系向量和世界位置在投影前统一执行 `x_visual=x`、`y_visual=-y`、
`z_visual=-z`：配置中较小的 z 位于上部、较大的 z 位于下部，因此共轴电机在上，
`z≈0.25 m` 的三组格栅在下。该转换同样应用于质心、气动中心、力/力矩箭头、
位置轨迹和目标位置。对应欧拉角使用 `roll_visual=roll_FRD`、
`pitch_visual=-pitch_FRD`、`yaw_visual=-yaw_NED`，确保 Roll/Yaw 方向、低头姿态、
真实推力方向及导出视频中的虚拟打杆语义一致。

每组格栅的细横线表示 `deflection_axis_b`，三条栅片沿导流方向绘制。导流方向使用与 SimEnv 动力学相同的 Rodrigues 公式，将 `neutral_thrust_direction_b` 绕 `deflection_axis_b` 旋转当前 `servo_angle`；因此导流方向始终垂直于格栅旋转轴，并按右手定则随舵角偏转。

后端可以直接调用浏览器接口：

```js
window.RLFlightAircraft.updateTelemetry({
  truth: {
    attitude_q_wb: [1, 0, 0, 0],
    motor_speed: [1200, 1150],
    servo_angle: [0.02, -0.01, 0.03],
    grid_moment_b: [
      [0.01, 0.02, 0.00],
      [-0.01, 0.01, 0.00],
      [0.00, -0.02, 0.01]
    ],
    moment_b: [0.00, 0.01, 0.01]
  }
});
```

或者由网络层派发事件：

```js
window.dispatchEvent(new CustomEvent('rlflightsimulationtelemetry', {
  detail: telemetry
}));
```

接口同时接受单环境批次形式，例如 `attitude_q_wb: [[w,x,y,z]]` 和 `grid_moment_b: [[[…],[…],[…]]]`。三个局部箭头使用 `grid_moment_b`，合成箭头使用包含直接推力偏置力矩、三个格栅力矩和电机反扭矩的 `moment_b`。

## 从服务器本地目录导入

参数栏标题右侧的 `⌁` 按钮打开服务器配置浏览器。它通过同源 API 浏览远程服务器目录，不使用 Windows 文件选择器。

默认开放的服务器目录：

```text
/home/ryan/RL_Flight/Controller/configs
/home/ryan/RL_Flight/SimEnv/configs
/home/ryan/RL_Flight/Train/configs/experiments
```

只显示 `.yaml`、`.yml` 和 `.json` 文件。服务端会解析配置并返回结构化数据，前端按字段路径填入参数面板；文件中未被当前测试 UI 使用的字段会被安全忽略。

导入采用“模板默认值 + 文件覆盖”策略：对应配置域的所有字段先获得当前 WebUI 模板默认值，再由服务器文件中的有效字段覆盖。文件缺失字段、`null`、空字符串、空数组或不受支持的枚举不会造成空白输入框；导入结果会显示文件覆盖项数和默认补齐项数。

可以在启动时显式追加其他服务器配置根：

```bash
python server.py --host 0.0.0.0 --port 8080 \
  --config-root identified=/srv/rl-flight/identified-configs
```

服务器拒绝绝对路径、`..` 路径穿越、配置根外的符号链接目标、非 YAML/JSON 文件以及超过 2 MiB 的文件。
