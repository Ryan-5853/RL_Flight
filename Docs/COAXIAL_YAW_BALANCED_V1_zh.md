# 保守格栅共模与近等转速配平候选 v1

这是根据 `log/26.11.4/log_0_UnknownDate_1.ulg` 制作的**待台架验证候选**。已修改源码和生成 payload，未刷入飞控。目标是约束电机差速，并让小幅格栅共模承担持续反扭矩补偿；当补偿不足时，接受 yaw 跟踪不足，不继续扩大目标转速差。

## 最终取值

| 项目 | 候选值 | 含义 |
|---|---:|---|
| 满杆 yaw 角速度 | ±0.30 rad/s | 约 ±17.2°/s；适配器和核心均有限幅 |
| yaw 指令一阶时间常数 | 0.20 s | 平滑杆量变化；不是独立角加速度硬限幅 |
| 格栅共模 | **−0.16** | 归一化舵机命令，模型舵角约 −0.056 rad/−3.2° |
| 基础下桨/上桨目标转速比 | **1.08** | 基础配平只让下桨快约 8% |
| 任一桨相对另一桨的最大目标转速比 | **1.15** | 两个方向均约束，含 yaw 积分要求 |
| 两路电机输出上限 | **0.45** | `actuator_motors` 域；内部上限 0.90，最终仍乘 0.5 |
| yaw 积分力矩上限 | ±0.003 N·m | 按当前积分增益换算限积分状态，并允许饱和时回退 |
| 下桨反扭矩系数修正比例 | 0.70 | 来自日志的暂定模型假设，非完成实测辨识 |

按当前 0.66 kg、既有推力和转速表，基础悬停点约为：

- 上桨输出 **0.2763**，目标约 **9205 RPM**；
- 下桨输出 **0.3499**，目标约 **9942 RPM**。

这是基础前馈点，不是锁死的固定输出。共同升降推力、yaw 修正和有界积分仍会改变输出。配置中还根据暂定下桨力矩模型及实际加入的格栅共模，补偿基础点的剩余 yaw 力矩，所以控制器刚输出的电机值也可能与基础点略有不同。

初步的共模 −0.12 在简化模型 `kQ下/kQ上=0.60` 时不足以覆盖反扭矩损失。因此最终选 −0.16，同时将基础转速比从 1.10 降为 1.08；不是简单地把电机差速上限调大。

## 实现行为

配置：`px4_trans/configs/lqi_coaxial_collective_airframe1_yaw_balanced_v1.yaml`。

1. 根据总推力计算 1.08 转速比的基础工作点，并在电机绝对上限处联合降低基础转速，保持配平比值。
2. 三舵机先分配 roll/pitch，再从剩余共同可用行程加入不超过 0.16 的负共模；低推力按目标推力渐入，零油门五路输出均为零。
3. 计算基础电机反扭矩与格栅共模的合成 yaw 前馈，余下 yaw 请求交给电机差速。
4. 差速修正作为一个向量按比例缩小，直到两路输出和预测转速比均可行；不靠独立裁剪下桨来实现配平。
5. yaw 积分始终有界；在 yaw 分配或力矩饱和时禁止朝更深饱和的方向积分，允许反向回退。roll/pitch 积分与 yaw 电机限幅解耦。

共模符号来自当前几何：格栅径向偏转轴、FRD 下推力方向以及现有舵角映射决定**负共模产生负 yaw 力矩**。这不是给三个 `NN_SV_TRIM` 参数各减 0.16：本候选中的共模与控制分配、可用舵机行程和力矩前馈相连。现有机械/输出映射和 `NN_SV_TRIM` 应维持已核对的状态。

1.15 限制作用于**命令—转速表计算的稳态转速**，并非 ESC 实测转速闭环。两套实物电机曲线不一致、负载变化或瞬态动态不同，都可能让实际转速比超出该数值；必须通过下一次 ESC 转速记录验证。这里没有声称实现硬件转速硬限幅。

## 验证

- 全部 `run_lqi_core_check.sh` 检查通过，包括原有保护、输出映射和三套核心 golden vectors。
- 新增 `yaw_balance_envelope_check.cpp`：全总距/偏航误差扫描、持续 20 秒不可能跟踪误差、积分限幅与饱和回退、共模对 roll/pitch 行程退让、核心直接输入的 yaw 限幅、零油门和无效 profile 拒绝。
- 两项 payload 标定来源/manifest 一致性测试通过。工作区 `SimEnv/src/simenv` 只有缓存，没有可直接导入的源码，因此这两项测试直接检查 YAML 标定表；没有宣称跑过完整 SimEnv 仿真。
- `analysis/yaw_balance_candidate_20261105/verify.py` 编译并直接调用实际 C++ 核心，使用含电机/舵机一阶动态、格栅衰减的简化非线性 yaw/垂向模型，在假定系数比 0.60、0.65、0.70、0.75、0.80 下检查 ±0.15 rad/s 指令及归零恢复。五个假设工况末尾均恢复近零 yaw 角速度，输出和预测转速比约束始终成立；系数比 0.60 时已接近能力边界。
- 简化模型不包含完整六自由度、真实桨流、旋翼加减速反作用、供电、安装误差和接地；它用于比较参数与检查边界，不能当作实飞验证。格栅的实际方向、效能及全行程余量仍需台架确认。
- 完整目标 `micoair_h743-v2_nncontrol` 编译通过。固件与源文件哈希见候选目录 `build_manifest.json`。

在台架上先确认负共模确实产生所需 yaw 力矩，同时检查差模 roll/pitch 的方向和行程；候选不会为了追 yaw 自动扩大共模或突破转速比边界。过弱的实际格栅效能可能使航向无法保持，过小的推力余量也可能导致掉高。

## 生成与回退

生成候选并用于当前构建：

```sh
python3 px4_trans/tools/generate_coaxial_collective_backend.py \
  --deployment-config px4_trans/configs/lqi_coaxial_collective_airframe1_yaw_balanced_v1.yaml \
  --output-dir px4_trans/generated/airframe-1/yaw-balanced-v1 \
  --reuse-model-manifest px4_trans/generated/airframe-1/manual-v3/lqi_coaxial_collective_manifest.json
bash px4_trans/tests/run_lqi_core_check.sh
python3 -m unittest Identification.tests.test_coaxial_collective_firmware_payload
```

该路径保留经过哈希检查的原有力矩域 LQI 增益和标定表，显式叠加新的部署分配约束；不是重新辨识整个模型。模型 ABI 为 v4、collective contract 为 3，profile 参数进入 checksum。旧配平 payload 仍保留在 `manual-v3` 目录；需要回退时从该已验证 manifest 重新生成并构建，不要仅把最终输出缩放改回去。
