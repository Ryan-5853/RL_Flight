# RL Flight WebUI

轻量化飞行仿真可视化控制台。浏览器负责读取 Windows 电脑上的手柄，远程
Python 服务负责在 CPU 上串联统一 Controller、单实例 SimEnv 和低频遥测。
Controller 可以选择 PID、LQR、PID+LQR 混合控制器或已有 MLP/GRU checkpoint。

## 本地预览

需要使用服务器本地配置、checkpoint 和 CPU 运行时时，请使用内置服务启动：

```bash
python server.py --host 0.0.0.0 --port 8080
```

然后访问 `http://localhost:8080`。

默认 checkpoint 根目录是 `/home/ryan/RL_Flight/Train/runs`。也可以显式开放
一个或多个服务器目录：

```bash
python server.py --host 0.0.0.0 --port 8080 \
  --checkpoint-root /srv/rl-flight/runs \
  --checkpoint-root /srv/rl-flight/releases \
  --runtime-log-root /srv/rl-flight/webui-runtime-logs
```

选择 `controller.type: neural` 时，checkpoint 必须是根目录内的 `.pt` 文件，并且
必须带有训练框架生成的同名 `.pt.sha256` 摘要文件。PID、LQR 和混合控制器不读取
checkpoint，它们从本次 SimEnv 参数自动求配平、控制分配矩阵和 LQR 增益。普通静态
文件服务器只能预览页面，无法使用配置浏览和 CPU 运行时接口。

交互仿真的 SimEnv 日志始终写入后端管理的 `--runtime-log-root`，默认是本目录
下的 `runtime-runs/`。上传配置中的 `logging.directory` 不会被直接采用，因为
其相对路径已失去原文件目录语义，绝对路径也不应成为浏览器可控制的服务器写入
目标。实际日志目录会在 session status 的 `log_directory` 字段中返回。

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
服务器独立控制线程（SimEnv control_hz，默认 500 Hz）
        │
        ├─ CPU 摇杆滤波、目标姿态与油门斜率限制
        ├─ PID / LQR / PID+LQR：读取 SimEnv 参数并直接输出 5 维命令
        ├─ Neural：CPU 21 维观测和 MLP/GRU 推理
        └─ CPU SimEnv.advance([上桨, 下桨, 舵机1, 舵机2, 舵机3])
        │
        ▼
低频 telemetry API（默认 30 Hz）→ Canvas 可视化
```

控制热路径中的状态、观测、循环网络 hidden、动作、仿真状态和遥测都保留在
CPU，不再创建 CUDA stream、pinned memory 或设备间复制。SimEnv 仍会按照
`logging` 配置异步归档时间线。checkpoint 在会话创建时从服务器磁盘读取一次，
校验 SHA-256 后把 Actor 权重载入 CPU。

服务端只允许存在一个交互式仿真会话。浏览器暂停时保留该会话和仿真状态；
配置未改变时再次启动会继续原会话，配置改变后会关闭旧会话并按新配置创建。
手柄数据超过 `runtime.command_timeout_ms` 未更新时，服务端会自动暂停并报告
`controller_input_timeout`，避免失联后继续使用旧指令。

当前网络边界采用标准库 HTTP：手柄上传使用最新值覆盖，遥测使用最长 1 秒的
长轮询。它不会让网络请求频率决定 500 Hz 的 CPU 控制循环，后续如需跨公网
降低请求开销，可在保持同一会话对象的前提下替换为 WebSocket。

## 离线 rollout 视频导出

顶部“导出视频”会使用右侧当前选择的环境、控制器、checkpoint 和虚拟飞手
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
- `neural`：保持现有 MLP/GRU checkpoint 推理链路。

`collective_mode: hover` 会把 reset 时的位置作为高度目标；`manual` 将手柄油门映射为
上桨 PWM，同时由控制器计算反扭矩平衡的下桨基准。没有手柄时页面会发送零姿态虚拟
输入，因此默认混合控制器可直接启动并观察自稳。页面默认初态故意设置为约
`roll=8° / pitch=-6°` 并带有小角速度；点击“启动仿真”即可看到 PID 捕获和 LQR
接管，不需要先修改参数。

右侧“推理与观测”中包含四个运行时字段：

- `runtime.checkpoint_path`：仅神经网络控制器需要；服务器 checkpoint 根目录内的相对路径，也接受允许
  根目录内的绝对路径；页面会从后端枚举带有效摘要文件的 checkpoint 作为候选；
- `runtime.telemetry_hz`：CPU 真值序列化并推送前端的最高频率；
- `runtime.command_timeout_ms`：最近一个有效手柄帧的最大允许年龄。
- `runtime.cpu_threads`：PyTorch CPU 算子的 intra-op 线程数；单环境默认使用 1。

`run.device` 必须是 `cpu`，服务端不会在交互运行时占用 CUDA。
`model` 的类型和层宽必须与 checkpoint 完全一致，否则创建会话时会直接返回
权重形状错误，避免用不匹配的控制器进入闭环。

左侧 `CTRL RATE` 显示后端按完成控制步实测的频率，延迟显示为对应的平均控制
周期；它们不再使用前端模拟的 GPU 负载或固定延迟。当前 V100 与 CPU 的同配置
短测结果分别约为 10.7 Hz 和 26 Hz。CPU 全链路约快 2.4 倍，但仍未达到配置的
500 Hz，因此 `loop_overruns` 会如实增长。1/2/4/8 个 intra-op 线程的短测结果
接近，单线程略优，所以默认 `runtime.cpu_threads: 1`。

### Runtime API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/runtime/capabilities` | CPU、PyTorch、逻辑核心数及允许的 checkpoint 根 |
| `GET` | `/api/runtime/checkpoints` | 枚举带 `.sha256` 的可加载 checkpoint |
| `POST` | `/api/runtime/sessions` | 上传两份 YAML 和服务器 checkpoint 路径，创建唯一会话 |
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

传统控制器创建会话时可以省略 `checkpoint_path`。遥测在原有 `truth/runtime` 外增加
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

俯仰通道采用航空器右手系约定：右杆向前/屏幕上方推动时输出负 Pitch，命令机头
下俯；向后/屏幕下方拉动时输出正 Pitch，命令机头上仰。手柄配置从
`rl-flight.gamepad.v2` 开始采用该约定，不会继续加载旧版向导中将“向前推杆”
保存为正 Pitch 的方向；升级后如使用非标准轴布局，请重新运行一次校准向导。

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

SimEnv 的 FRD/NED 坐标使用 `+Z` 向下，而 Canvas 内部使用图形学常见的 `+Z` 向上空间。所有机体几何和机体系向量在投影前统一执行 `z_visual = -z_FRD`：配置中较小的 z 位于上部、较大的 z 位于下部，因此共轴电机在上，`z≈0.25 m` 的三组格栅在下。该转换同样应用于质心、气动中心和力矩箭头。由于 Z 镜像同时改变绕 X/Y 轴的旋转手性，Canvas 姿态使用 `roll_visual=-roll_FRD`、`pitch_visual=-pitch_FRD`、`yaw_visual=yaw_NED`，确保模型的低头、横滚和真实推力方向一致。

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
