# RL Flight Train

训练层采用 TorchRL 作为学习框架，以设备绑定的 `TensorDict` 贯穿环境适配、GRU
rollout、PPO 更新以及 MLP-SAC 的 GPU 经验回放。reward、动作变换和姿态数学均为批量
`torch.Tensor` 运算；训练热路径不执行 CPU/NumPy 往返。

完整设计见 [TRAINING_FRAMEWORK.md](TRAINING_FRAMEWORK.md)。

训练配置的输入契约、训练主脚本选择、训练层静态参数随机化、仿真层动态/噪声随机化、
独立 seed 流，以及可注入的张量化 `RewardCalculator` 见文档的
[配置输入与组件注入](TRAINING_FRAMEWORK.md#51-配置输入与组件注入)、
[两级随机化边界](TRAINING_FRAMEWORK.md#711-两级随机化边界) 和
[任务与奖励计算器](TRAINING_FRAMEWORK.md#54-任务与奖励计算器) 章节。

## 当前实现

- SimEnv `create / observe / advance / masked reset` 适配；
- 自稳模式 VirtualPilot：高度真值增量式 PI 油门、三轴随机摇杆、一阶惯性和 masked reset；
- 21 维观测、4 维策略动作，以及“外部上桨油门＋策略动作”到 5 维 SimEnv 命令的合成；
- TorchRL `GRUModule`、`ProbabilisticActor`、`ValueOperator`；
- GPU 驻留的 `[B,T]` rollout；
- TorchRL `GAE`、`ClipPPOLoss`、`SACLoss`、双 Q 目标网络和 GPU replay；
- JSON/YAML 配置、运行 manifest、标量记录和原子 checkpoint；
- reward、四元数、动作变换、随机化隔离、记录完整性和 PPO 张量闭环测试；
- 带 SHA-256 校验的原子 checkpoint、独立评估 RNG，以及 Train–SimEnv 日志关联。

SimEnv 当前已实现执行器、推力矢量气动、六自由度刚体和传感器张量 kernel，并有
CPU 数值测试。示例配置及短跑验收仍仅用于验证接口和数值闭环，不代表奖励、随机化
范围或 PPO/SAC 参数已经调优，也不能替代 SIL/HIL 与实机前的安全验证。

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

完整中文注释的训练入口模版位于：

```text
configs/experiments/training_entry_template.yaml
```

建议先复制为新的实验文件再修改，模版本身由自动化测试保证可以被当前 v2 解析器
直接加载。注释同时标明了当前 runner 已消费的字段和尚未启用的预留节，避免把仅归档
的配置误认为已经执行。

环境安装完成后运行测试：

```bash
python -m unittest discover -s tests -v
```

第 14 节验收标准可单独运行：

```bash
PYTHONPATH=src:../SimEnv/src python -m unittest tests.test_acceptance -v
```

SimEnv 已提供版本化完整动态 `state_dict/load_state_dict`，第 7 项会验证连续运行
`N+M` 与运行 `N` 步后新建环境精确恢复 `M` 步的模型、优化器、仿真、传感器、
VirtualPilot、collector、RNG、动作、reward 和 loss 逐张量一致。运行 manifest 声明
`exact_resume_supported: true`。CUDA 项应在 V100 环境运行同一命令完成硬件验收。

从已有 checkpoint 精确续训到配置中的绝对总步数：

```bash
flight-train run --config configs/experiments/my_experiment.yaml \
  --device cuda:0 \
  --output-root /path/to/new-runs \
  --resume-from /path/to/parent/checkpoints/step_N.pt
```

续训会创建新的 run/log 身份，并在 manifest 记录 `parent_run_id` 和 checkpoint 路径。
允许修改 `run.total_control_steps`、`run.output_root` 和整个 `checkpoint` 调度段；模型、PPO、
并行数、device/dtype、时基、随机化、VirtualPilot、reward 或控制契约不一致会在采样前拒绝。

`checkpoint.interval_control_steps` 按全部并行环境累计控制步计数，并在完整 rollout/PPO
update 边界保存；`null` 表示只保存最终版本。`checkpoint.keep_last` 控制最近版本保留数，
每份 `.pt` 均带 SHA-256 sidecar，并由 `checkpoints/index.json` 记录类型（`periodic`、
`final` 或 `interrupt`）。第一次收到 SIGINT/SIGTERM 后训练会完成当前安全边界、保存
`interrupt` checkpoint 并将运行标记为 `interrupted`；第二次信号会立即退出。

正式 MLP 配置对应的 SimEnv 日志默认采用显式 `compact` 模式：5000 Hz 物理时间线
每 10 步保存一次（500 Hz），并只保存配置列出的诊断字段。初始化、reset 和 resume
边界始终保存；需要逐物理步复盘时将环境配置改为 `mode: full`、
`physics_step_stride: 1`，并移除 `fields` 裁剪。

运行 CPU 接口冒烟：

```bash
flight-train run --config configs/experiments/gru_ppo_smoke.json
```

先运行目标文档要求的无状态 MLP PPO 基线（21 维观测，`256×256×128`，4 维策略动作）：

```bash
flight-train run --config configs/experiments/mlp_ppo_smoke.json
```

当前 `mlp_nominal_baseline_1.yaml` 是关闭 Train 层静态/动态随机化的长期标称基线：
从当前模型约 `0.565` 的悬停上桨工作点附近快速 spool，roll/pitch 限制为
`0.10 rad`、yaw rate 限制为 `0.35 rad/s`。episode 从纯悬停 2 秒开始；
当前阶段成功率连续两次达到 80% 后，依次解锁 5、10、30 秒，并把打杆比例扩展为
0%、15%、40%、100%。课程阶段与累计结果属于 checkpoint 的精确续训状态。
训练使用 `B=256、T=256`，完整 on-policy batch 为 65,536；在 33,554,432
累计控制步内执行 512 次策略刷新。PPO 使用
`gamma=0.9995 / gae_lambda=0.9985`、4 个 epoch 和 4 个 minibatch，
并分离 actor/critic 优化器，以改善此前 500 Hz 下信用分配过短、策略刷新太少和
critic 梯度挤压 actor 的问题。该配置仍保持固定标称动力学，确认
存活率和固定评测持续改善后，再复制为新实验并逐步加入参数随机化。

MLP-SAC 标称基线使用独立配置，不覆盖 PPO 实验：

```bash
flight-train run \
  --config configs/experiments/mlp_sac_nominal_baseline_1.yaml
```

该配置使用 `B=256、T=128`，稳态下每次向 GPU replay 新增 32,768 条
transition；replay 容量为 4,000,000，warm-up 为 262,144，每次采集后执行
16 次 4096 batch 更新，样本更新比仍约为 2。critic 使用 64-step
（0.128 秒）折扣回报并从第 64 步目标 Q bootstrap；跨 rollout 的 63 步原始
上下文会进入 exact checkpoint，采集边界不会伪造 episode 截断。
标称 teacher 的四路标准差固定为 `[0.05, 0.08, 0.08, 0.08]`，不允许
actor 通过膨胀方差获得熵收益；自动温度限制在 `1e-5～1e-4`。零策略动作以动力学平衡点
`lower_motor=0.53523` 为中心。TorchRL `SACLoss` 维护双 Q 和软目标网络。
warm-up 后先执行 1024 次 critic-only 更新，actor 保持初始稳定策略不变；
随后每轮先更新 critic，再重新前向并以 `3e-5` 学习率更新 actor。
warm-up 和 critic-only 期间课程锁定在阶段 0，但仍记录成功率；actor 解冻后
连续成功计数才开始生效，避免固定策略被提前推入更难课程并污染 critic replay。
课程拆成更细的 7 个阶段，固定每控制步生存奖励，不再让 reward 随课程 episode
进度改变。当前先用 8,388,608 条 transition 做短验证，并每 2,097,152 条执行
固定评测和 exact checkpoint；checkpoint 包含完整 replay，文件显著大于 PPO
checkpoint。

每次 update 的 `metrics.jsonl` 除算法 loss 外还记录：

- `terminated_fraction`、`truncated_fraction`、`done_fraction`、`invalid_fraction`；
- `episode_reset_count`、已结束 episode 的平均长度和平均生存秒数；
- 姿态误差 mean/P95、角速度 mean、高度绝对误差 mean；
- 策略动作 RMS、绝对峰值和 `|action|>=0.95` 饱和率。
- `curriculum_stage`、当前 episode 时长、命令比例、最近成功率和是否晋级；
- `reward.attitude/tilt/yaw_rate/risk/survival/termination` 等分项。

固定评测按悬停生存时间选择最佳策略，独立保存在
`checkpoints/best_fixed_evaluation.pt`；对应控制步、评测总分、悬停生存时间和摘要
写入 `checkpoints/best_fixed_evaluation.json`。该文件不参与周期 checkpoint 的
`keep_last` 清理。

`valid_fraction=1` 只表示数值/接口有效，不能代表没有坠毁；判断是否学会稳定控制必须结合
终止率、生存时间、姿态误差和动作统计。

训练命令默认在 stderr 显示实时进度。交互终端使用单行进度条并约每 2 秒刷新采样进度；
PPO/SAC 阶段同时显示已完成/总 update；SAC 汇总额外显示 replay size、alpha
以及当前 rollout 上四路 actor 标准差和 `warmup/critic-only/actor` 阶段。
重定向到文件或作业系统时不输出逐采样临时行，只保留阶段切换、每轮汇总、checkpoint
和最终状态，避免日志刷屏。每轮汇总包含完成比例、累计控制步、已用时间、ETA、吞吐量、
平均奖励、姿态误差 P95、终止率和动作饱和率。例如：

```text
[======>.................] 25.00% | 262,144/1,048,576 | 已用 00:03:10 | ETA 00:09:30
| 第 4/16 轮完成 | reward=0.421 | 姿态P95=3.17° | 终止=0.42%
| 动作饱和=4.81% | 1,380 sample/s
checkpoint 已保存 | 类型=periodic | 控制步=262,144
```

进度写入 stderr，CLI 正常完成后仍在 stdout 最后一行打印 run 目录，便于 shell 脚本捕获。

MLP 和 GRU 使用同一个 TorchRL `GAE + ClipPPOLoss` 调度路径。MLP 的
`algorithm.sequence_length` 必须为 `1`；GRU 则以完整时间块作为 minibatch 单位，
保留 episode reset 与 BPTT 边界。V100 训练可以使用只影响 resolved run 配置和
fingerprint 的白名单覆盖，不需要修改 smoke 配置：

```bash
flight-train run --config configs/experiments/mlp_ppo_smoke.json \
  --device cuda:0 --output-root /tmp/rl-flight-v100-runs
```

自稳控制契约中，上桨油门由 VirtualPilot 的高度真值增量式 PI 直接控制，默认跟踪
`height_m=0`；策略只输出下桨电机和三个舵机。
部署时真实遥控器替换 VirtualPilot。roll/pitch 摇杆映射为目标角，yaw 摇杆映射为目标
角速度并积分成航向；训练时的高度控制器允许读取真值，但不属于
神经网络高度闭环，也不加入高度奖励。

策略的 4 维输出是配平点附近的归一化残差，不是执行器绝对命令。当前 SAC 标称
配置在 `upper_motor=0.565` 时使用动力学静态平衡值
`trim=[lower_motor=0.53523, servo_1=0, servo_2=0, servo_3=0]`；动作零点映射到
该配平值，下桨残差尺度为 `0.30`，三个舵面残差尺度为 `1.0`。SAC MLP 输出层
严格从零均值开始，因此新策略从配平点附近探索。PPO 基线与通用入口模版中保留的
旧实验值 `0.5539` 不会被本次 SAC 配置修改。

21 维输入的部署尺度固定为：目标相对当前姿态四元数 4、归一化角速度 3、
归一化加速度 3、归一化电机转速 2、舵机实际角 3、归一化偏航角速度指令 1、
上桨油门 1、上一策略动作 4。角速度除以任务终止阈值，加速度除以
`9.80665 m/s²`，电机转速除以 `1800 rad/s`，舵角除以 `π/2`。该版本 profile 为
`attitude_self_stabilize_21d_v3`；v2 checkpoint 的输入语义不同，不能续训或评测。

部署/确定性评估使用 `ActorCritic.forward_step(observation)`：MLP 接收 `[B,21]` 并
返回 `[B,4]`，不创建 hidden state；GRU 使用同一入口并额外接收/返回 recurrent state。
训练 collector 使用随机动作，评估 collector 使用分布 mode，二者不会混淆。

固定三科评测使用版本化配置
[`configs/evaluation/fixed_attitude_v1.yaml`](configs/evaluation/fixed_attitude_v1.yaml)：

```bash
flight-train evaluate \
  --config configs/experiments/mlp_nominal_baseline_1.yaml \
  --checkpoint /path/to/run/checkpoints/step_N.pt \
  --suite configs/evaluation/fixed_attitude_v1.yaml \
  --device cuda:0 \
  --output-root /path/to/evaluation-runs
```

也可以直接运行 [`scripts/evaluate_fixed.py`](scripts/evaluate_fixed.py)，参数完全相同。
评测使用 checkpoint 的确定性策略动作，并为每个科目创建全新的 SimEnv，防止动力学、
传感器历史、GRU hidden 或高度 PI 状态跨科目泄漏。固定基准评测禁用 Train 层静态和
动态 domain randomization；SimEnv 配置中的标称参数与确定性传感器随机流仍然生效。

三个科目为：

- `hover`：水平姿态、固定初始位置和零速度参考；
- `constant_translation`：固定倾斜姿态，同时跟踪匀速直线位置/速度参考；
- `circle`：`roll=A·sin(ωt)`、`pitch=A·cos(ωt)`，同时跟踪圆形位置和切向速度参考。

当前网络是姿态自稳内环，没有位置/速度外环，因此报告会把姿态 RMSE、位置 RMSE 和
速度 RMSE 分开列出。直线/圆轨迹得分低时，可以据此判断是姿态内环没有跟上，还是
姿态已跟上但缺少外环造成轨迹偏差。

每科报告生存时间、姿态/位置/速度跟踪误差、策略动作 RMS/峰值、动作变化 RMS、动作饱和率
和响应时间。平移科目的响应时间是姿态误差进入阈值并持续指定窗口的稳定时间；悬停从
本次最大姿态误差起算恢复时间，避免起始水平状态虚假得到零响应时间；圆轨迹的响应时间
是 roll/pitch 基频波形的最优相位滞后。每项同时保存 mean、P50、P95 和
逐实例数值。默认科目总分为生存 35%、跟踪 35%、动作 15%、响应 15%，三个科目等权
形成最终总分。阈值和权重均在 suite YAML 中显式配置。

评测目录包含 `report.json` 以及每科一个 `trajectory_<name>.pt`。轨迹保存实际/目标位置、
速度、姿态、目标欧拉角、动作、时间和存活 mask，可以离线重算评分；报告同时记录
checkpoint 路径、SHA-256、训练控制步、suite seed、device 和并行实例数。

正式训练配置应使用 `cuda:0`，并将
`run.allow_unimplemented_simulator` 设为 `false`。如果 CUDA 或真实仿真 kernel 不可用，
运行器会在采样前失败。
