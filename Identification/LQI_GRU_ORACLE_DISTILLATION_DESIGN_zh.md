# 每机体真值 LQI 到因果 GRU 的端到端蒸馏实验 v1

## 1. 要回答的问题

这个实验不以“GRU 能回归机体参数”为最终目标，而是直接检验：在机体参数固定但未知、参数组跨越 `sim2real_micro` 全范围时，一个只接收可部署因果观测的 GRU，是否能通过一段闭环历史形成足够的隐状态，并复现该机体专属 LQI 从观测到五轴物理执行器命令的控制规律。

必须区分三个逐级结论：

1. `有表示能力`：留出参数组上的教师命令误差低，且比无记忆模型低。
2. `确实在利用历史辨识`：清空/打乱历史会显著变差，GRU 隐状态能线性解码部分控制相关参数。
3. `能作为控制器`：学生在闭环非线性仿真中安全、收敛和跟踪接近 oracle LQI，并显著优于无记忆学生和固定 nominal LQI。

只有第 3 条通过，才可以说 GRU “吃下了”这组控制规律。离线行为克隆分数本身不够。

## 2. 教师、学生和信息隔离

### Oracle 教师

每个固定随机参数点独立执行以下过程：

1. 从 `sim2real_micro` 可行域采样完整机体、气动、电机和舵机参数。
2. 使用该点真实参数计算真实 hover trim 和局部连续模型。
3. 以 500 Hz 离散化并独立求解 13 状态 LQI 的 DARE，得到该点自己的 `K_i in R^(5x13)`。
4. 教师可读取真值角速度、真值电机转速、真值舵角和真实参数；输出完整五轴物理命令 `[upper_motor, lower_motor, servo_1, servo_2, servo_3]`。
5. 不安全或仿真无效的教师 episode 不进入蒸馏集；它们只应进入拒绝/覆盖率审计。

因此这里没有 nominal gain 共享，也没有四维 residual action 或“上电机由虚拟飞手拥有”的接口歧义。虚拟飞手只发布目标姿态、目标 yaw rate 和 collective demand，最终五轴命令全部来自 LQI。

### GRU 学生

学生输入为 25 维因果观测：

- tilt 姿态估计四元数 4 维；
- gyro 3 维、accelerometer 3 维、motor speed 2 维；
- 目标 tilt 四元数 4 维、目标角速度 3 维、collective command 1 维；
- 上一拍五轴物理命令 5 维。

明确禁止输入：参数标签、LQI gain、真值舵角、真值角加速度、真值 force/moment。参数标签和 oracle gain 在 shard 中带有 `audit_only` 后缀，只供分组、覆盖率和隐藏状态探针使用；加载器不会把它们拼入网络输入。

SimEnv 当前没有姿态估计器模型，所以 v1 把仿真姿态通过 `attitude_q_tilt_estimate` 接口提供给学生。这不等于参数真值泄漏，但会高估现实中的状态估计质量。进入 sim-to-real 前必须增加姿态估计噪声、延迟和偏置实验。

## 3. 数据覆盖设计

正式配置使用 8,192 个参数组，每组 4 条、每条 6 秒，共 32,768 条 episode、约 54.6 小时物理飞行时间。仿真、教师、数据和学生契约均保持 500 Hz，每条序列 3,000 帧；不通过降采样改变执行器零阶保持语义。参数组按 70/15/15 切分，任何一个机体参数点只属于 train、validation 或 test 之一。

每组 4 条轨迹采用分层多轴初始状态，并叠加虚拟飞手的 roll/pitch 角度、yaw rate 和 collective 分段命令。正式生成前建议先执行 oracle screening：

- 每个点至少 4 条初始位姿轨迹和 2 条持续命令轨迹；
- 完整 6 秒安全率应至少为 95%；
- 轨迹中必须同时包含非饱和辨识段和接近控制边界的困难段；
- 按姿态误差、角速度、五轴命令移动量和饱和率分层采样，避免大量 hover 帧淹没动态帧；
- 报告原始随机点覆盖、可求解 DARE 覆盖、安全教师覆盖和最终数据覆盖，不能只报告最终留下的数据。

如果全范围中有大量点连 oracle LQI 都无法产生优质轨迹，应把问题定义为“局部 LQI 专家族的适用域不足”，不能让 GRU 为失败教师背锅，也不能静默缩窄随机范围。

## 4. 网络和训练

基线学生为 2 层单向 GRU，hidden size 192，后接 `192-96-5` MLP。电机输出使用 sigmoid，舵机输出使用 tanh，天然满足物理命令范围。损失由两部分构成：

`L = weighted_MSE(u_student, u_oracle) + 0.2 * weighted_MSE(Delta u_student, Delta u_oracle)`

电机轴权重高于舵机轴，避免三个舵机数量优势掩盖 collective/yaw motor 错误。所有 normalization 只由训练参数组拟合。模型在 episode 开始时清空 hidden；同一 episode 内严格按时间因果推进，绝不使用双向 GRU、未来帧或跨 episode hidden。

为了判断容量而不是只调一个网络，正式初验至少训练以下四个相同预算基线：

1. `GRU-192x2`：主模型。
2. `MLP-current`：仅当前帧，无记忆下界。
3. `GRU-reset`：使用同一 GRU 权重，但每帧清空 hidden；验证提升是否来自历史。
4. `GRU-384x2`：容量上界；若 192 失败而 384 显著成功，说明瓶颈更可能是容量而不是信息不可辨识。

可再加入“显式参数辅助头”作为训练期 regularizer，但它不能作为部署输入；主结论必须由没有参数监督也能训练的纯动作蒸馏模型给出。

## 5. 三阶段验证和门槛

### Gate A：离线留出机体模仿

在未见参数组上报告每一轴 RMSE/NRMSE、P50/P90/P99 序列误差，以及 `0-0.5 s`、`0.5-2 s`、`>2 s` 三个时间段误差。

初步通过条件：

- 五轴 test NRMSE 均小于 0.20，P99 没有单轴灾难性长尾；
- `GRU-reset / GRU` 的 RMSE 比值至少 1.25；
- `>2 s` 的误差比前 0.5 秒至少低 20%，说明在线历史确实在改善控制选择；
- 最终 hidden 对控制相关有效参数的线性探针 median R2 至少 0.5。探针只作机理证据，不替代控制结果。

### Gate B：反事实隐辨识

构造“瞬时观测近似相同、机体参数不同”的配对窗口。比较完整历史、历史置换、错误机体历史和每帧清 hidden 四种输入。若完整历史不能选择接近各自 oracle 的不同动作，说明学生主要在拟合瞬时反馈律，没有隐性辨识机体。

初步通过条件：完整历史动作误差相对错误历史至少降低 25%，而时间置换显著破坏性能。

### Gate C：配对非线性闭环

在完全未见参数组、相同初始状态和相同飞手命令下，配对运行：oracle LQI、GRU、MLP-current、GRU-reset、固定 nominal LQI。必须让学生自己的上一拍命令回灌输入，禁止 teacher forcing。

报告安全率、6 秒生存率、姿态/角速度跟踪 RMS、五轴饱和率、命令移动量和相对 oracle 的 regret，并分别统计全体、参数范围边缘 10% 和 oracle 可控子集。

容量初验的通过条件：

- GRU 安全率不低于 oracle 2 个百分点以上；
- 姿态/角速度 RMS 不超过 oracle 的 1.15 倍；
- 相对 MLP-current 和 GRU-reset，在至少 80% 的参数分层箱中有优势；
- P99 饱和和动作跳变没有比 oracle 恶化 25% 以上；
- 8 秒延长审计不出现后期 hidden 漂移。

任何一个条件失败，都只能得出“当前信息、数据或网络不足”，不能直接归因于 GRU 理论上无能力。

## 6. 当前实现与初步冒烟结论

实现入口：

- `flight-identification-lqi-gru-data`：逐参数点设计 oracle LQI 并生成严格分组数据；
- `flight-identification-lqi-gru-train`：训练因果五轴 GRU；
- `flight-identification-lqi-gru-eval`：执行 Gate A 的离线指标、hidden-reset 消融和参数探针。

已完成 256 参数组、4 条/组、6 秒、500 Hz 的 GPU pilot，并完成总计 80 轮训练、Gate A 和 37 个留出参数组乘 4 个新初值的 6 秒 Gate C。离线五轴 NRMSE 为 `0.063-0.086`，reset-hidden 消融恶化 `13.74x`；但无 teacher forcing 闭环中 GRU 安全率只有 `34.46%`，oracle LQI 为 `86.49%`，fixed nominal LQI 为 `95.27%`。因此当前结论是：GRU 能拟合且明显利用历史，但纯行为克隆没有形成可靠闭环控制器，也没有可线性解码的完整原始参数表示。详细证据见 `runs/lqi_gru_oracle_distillation_pilot_256_gru192x2_v3_resume50/CONCLUSION_zh.md`。

## 7. 推荐执行顺序

当前 256 组 pilot 已明确未通过 Gate C，因此不要直接扩到 8,192 组。下一步应先做学生闭环 DAgger、动作历史 scheduled sampling、hidden 长时漂移约束，以及 oracle 优于 nominal 的教师域筛选；仍需固定 dataset manifest/hash，并让所有基线使用同一参数组切分。
