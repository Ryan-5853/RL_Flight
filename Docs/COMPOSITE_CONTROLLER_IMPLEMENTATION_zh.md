# 组合体控制器实现与验证

> 2026-11-03 更新：当前地面组合飞行、External Mode 3 和断链停机锁死规则以
> [地面组合飞行流程](GROUND_PAIR_FLIGHT_zh.md) 为准；本文中的旧接线边界和单体降落回退不再适用。

2026-11-03。实现入口为 PX4 `comp_control`，对应设计依据为
[组合体协同控制器](COMPOSITE_COOPERATIVE_CONTROLLER_zh.md)。原设计文档的已有修改予以保留。

## 接口与运行边界

```text
composite_control_reference（未来模式选择器显式使能、C 系目标）
                     ↓
comp_attitude → composite_attitude_estimate → comp_control
                                                ↓
                                     airlink_pair_command
                                      ↙                 ↘
                            现有主机输出路由        现有 AIRLINK 从机执行
```

- 固件配置 `micoair_h743-v2_nncontrol` 编入模块，但没有新增自动启动或自动接管。
  `comp_control start` 仅启动模块；未收到新鲜且 `enabled=true` 的参考消息时不产生有效控制输出。
- 输入姿态 `q` 为 C→NED，角速度和 `specific_force` 都是 C 系；使用同一条估计消息保证反馈对应。
  使用 `a_N = R_NC f_C + g_N` 计算运动学竖直加速度。
- `CompositeControlReference` 明确提供 `q`、当前 C 系表达的参考角速度、`collective∈[0,1]`。
  中位总距是零竖直加速度。此接口不是遥控器切换方案。
- 按新状态采样运行，调度周期 2 ms；重复采样不重复积分。
  首帧、估计器重置、退出和零总距清空积分；滤波器从当前测量初始化。
  输入无效、超出既有 10 ms 新鲜度约定、非有限数或异常 dt 时撤销有效输出。
- 主从输出顺序完全一致：`[upper_motor, lower_motor, servo_1, servo_2, servo_3]`。
  电机使用现有控制器域 `[0,1]`，舵机 `[-1,1]`；沿用下游最终电机映射，不在组合控制器重复缩放。
- 没有修改单体控制器、主从输出路由或 AIRLINK 会话状态机。
  连接、断链、重连以及遥控器模式切换都不在此次实现范围内。
  联调时由现有 owner 决定是否应用组合命令，`enabled` 本身不改变 owner。

## 控制和分配

`PairControlCore.hpp` 独立维护五个持久状态：竖直加速度滤波、竖直加速度积分、
Roll/Pitch 角度误差积分及 Yaw 速率误差积分。竖直 PI、推力反解、局部电机雅可比表及三舵机
逆映射复用单体模型参数与算法，单体代码不变。

Roll/Yaw 增益由组合惯量通过 ZOH 和离散 Riccati 方程离线生成；Pitch 使用单体增益乘
`Iyy_pair/Iyy_unit`，再由两侧各承担一半。控制循环不运行 Riccati 求解。

分配顺序：

1. 限制总推力及 Roll 权限，`T_A=T/2-Mx/(2a)`、`T_B=T/2+Mx/(2a)`。
2. 两侧分别计算 `R_Ci^T [0,My/2,0]`，送入单体三舵机逆映射。
   两侧使用自己的推力工作点；饱和时完整舵机向量共同缩放。
3. 按每侧 `Mz/2` 解局部 `[dT=0,dMz]` 电机逆映射。
   电机触限时缩小整个双电机增量，保持推力优先的局部线性方向。
4. 通道饱和且积分继续推动饱和时撤回该轴本步积分，反向误差允许退出饱和。

几何配置为 [composite_controller.yaml](../px4_trans/configs/composite_controller.yaml)。
修改几何或物理模型后执行 `python3 px4_trans/tools/generate_pair_gains.py` 并重新构建、重启模块。
生成器旋转单体惯量并加入平行轴项；当前 Demo 使用相同单体模型，要求组合惯量为对角阵。
启动时检查两个安装角与 `CEKF_MSTR_YAW/CEKF_SLV_YAW` 一致。
更换单体模型还须同步重新生成现有单体 payload，不能只修改组合几何配置。

## 诊断

`composite_control_status` 记录采样时间、状态年龄、姿态/角速度及目标、竖直加速度及目标、
请求和模型合成的 `[T,Mx,My,Mz]`、两侧目标推力、舵机及 Yaw 缩放系数、通道饱和标志、两侧模型实际推力及力/力矩残差。
十路执行器命令在 `airlink_pair_command` 中。
这三个控制相关 topic 已加入默认 ULog 主题列表，无发布时不产生命令数据。

`PairWrench.hpp` 使用同一物理基线计算完整稳态模型残差：
`F_C=ΣR_Ci F_i`，`M_C=Σ(R_Ci M_i+r_i×R_Ci F_i)`，包含格栅衰减、横向力和杆臂项。
该诊断模型专用于当前基线的 -Z 推力、零局部重心和直接推力力臂；生成器检查这些约束。
它是稳态模型估计，不能表示电机、舵机动态或真实气动互扰。

## 可复现验证

在仓库根目录执行：

```bash
python3 px4_trans/tools/generate_pair_gains.py
bash px4_trans/tests/run_pair_control_check.sh
python3 px4_trans/tests/check_pair_model.py
bash px4_trans/tests/run_lqi_core_check.sh
bash px4_trans/tests/run_airlink_protocol_check.sh
make -C px4_trans/px4 micoair_h743-v2_nncontrol -j4
```

Python 检查依赖 NumPy、SciPy（增益生成）、PyYAML，无须 SimEnv Python 源码或 Torch。

已通过的检查：

- 悬停输入零运动学加速度，中位总距请求 `12.94478 N`。
- 正 Roll 时 B 增推、A 减推，两侧推力之和保持不变。
- ±30° 下两侧舵机输出不同，完整合成 Pitch 力矩方向正确；非对称格栅指令的杆臂项非零。
- Pitch 增益对应的组合角加速度与单体一致，排除多除一次二。
- 0.1 rad/s 小 Yaw 参考下，每侧模型推力变化约 `0.001231 N`。
- 全部执行器饱和约束、零总距、非有限输入和异常 dt。
- 悬停、Roll/Pitch/Yaw 分别小阶跃、叠加小指令各 6 秒的非线性稳态力矩闭环，
  包含刚体惯性耦合；未出现发散或末段长期执行器饱和。
- 新设计离散闭环最大极点模 `0.996514`；原有三套 LQI golden-vector 回归通过。
- 目标 PX4 固件编译通过。

这些是主机模型和编译验证。尚未进行包含电机/舵机时滞的双机全状态仿真、台架或真机飞行验证；
完整竖直运动闭环与气动交叉耦合仍需联调验证。独立参考 topic 和 owner 的模式切换接线尚待确定。
