"""完全张量化的执行器、推力矢量气动与六自由度刚体动力学。

本模块统一采用接口文档中的坐标系约定：世界系为 NED，机体系为 FRD，
Hamilton 四元数 ``q_wb`` 将机体系向量旋转到世界系。所有状态和参数的第一维
都是批量维 ``B``；计算过程中不遍历单个仿真实例，也不发生 CPU/NumPy 往返。

动力学内核为整批实例计算候选状态。实例是否激活、候选状态是否有限，以及最终
是否提交，由环境层使用逐实例 mask 决定，因此一个实例失败不会影响其他实例。
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping

import torch


class TensorDynamicsKernel:
    """批量计算执行器响应、四路推力和六自由度刚体状态。"""

    implemented = True
    _GRAVITY_N = (0.0, 0.0, 9.80665)

    def __init__(self, parallel_count: int, device: torch.device, dtype: torch.dtype) -> None:
        """缓存批量索引和电机随机子系统标识，不保存跨实例共享的动力学状态。"""
        self._parallel_count = parallel_count
        self._device = device
        self._dtype = dtype
        self._batch_index = torch.arange(
            parallel_count, dtype=torch.int64, device=device
        )
        self._gravity_n = torch.tensor(
            self._GRAVITY_N, dtype=dtype, device=device
        )
        self._motor_index = torch.arange(
            2, dtype=torch.int64, device=device
        )[None, :]
        self._derived_parameters: dict[str, torch.Tensor] = {}
        digest = hashlib.blake2b(b"motors", digest_size=8).digest()
        self._motor_subsystem_id = int.from_bytes(digest, "little") & ((1 << 62) - 1)

    def refresh_parameters(
        self,
        parameters: Mapping[str, torch.Tensor],
        physics_dt: float,
        mask: torch.Tensor | None = None,
    ) -> None:
        """缓存生命周期内不变、但可能在 masked reset 时变化的派生参数。"""

        candidates = {
            "motor_response": 1.0
            - torch.exp(-physics_dt / parameters["motors.time_constant"]),
            "direct_arm_b": (
                parameters["aerodynamics.direct_thrust_center_b"]
                - parameters["body.center_of_mass_b"]
            ),
            "grid_arm_b": (
                parameters["aerodynamics.grids.aerodynamic_center_b"]
                - parameters["body.center_of_mass_b"][:, None, :]
            ),
            "airflow_axis_b": -parameters[
                "aerodynamics.neutral_thrust_direction_b"
            ],
            "motor_table_x": parameters[
                "motors.pwm_to_rpm_table"
            ][..., 0].contiguous(),
            "motor_table_y": parameters[
                "motors.pwm_to_rpm_table"
            ][..., 1].contiguous(),
            "servo_table_x": parameters[
                "servos.pwm_angle_table"
            ][..., 0].contiguous(),
            "servo_table_y": parameters[
                "servos.pwm_angle_table"
            ][..., 1].contiguous(),
            "attenuation_table_x": parameters[
                "aerodynamics.grids.self_attenuation_curve"
            ][..., 0].contiguous(),
            "attenuation_table_y": parameters[
                "aerodynamics.grids.self_attenuation_curve"
            ][..., 1].contiguous(),
        }
        if mask is None or not self._derived_parameters:
            self._derived_parameters = candidates
            return
        for name, candidate in candidates.items():
            current = self._derived_parameters[name]
            expanded = mask.reshape(
                self._parallel_count, *([1] * (current.ndim - 1))
            )
            current.copy_(torch.where(expanded, candidate, current))

    def step(
        self,
        state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        control: torch.Tensor,
        active_mask: torch.Tensor,
        physics_dt: float,
        instance_seeds: torch.Tensor,
        motor_noise_counters: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        """推进一个物理时间步并返回完整候选状态。

        ``control`` 的 shape 为 ``[B,5]``，前两列是归一化电机 PWM，后三列是
        归一化舵机 PWM。``active_mask`` 由环境层用于决定候选状态是否提交；内核
        始终执行同形状批量运算，以保持张量路径和实例随机流稳定。
        """
        next_state = dict(state)

        # 执行器先响应当前保持不变的控制信号，再由实际转速和实际舵角计算气动力。
        motor_speed, effective_motor_speed, total_thrust, motor_torque = self._update_motors(
            state,
            parameters,
            control[:, :2],
            physics_dt,
            instance_seeds,
            motor_noise_counters,
        )
        (
            servo_angle,
            servo_effective_pwm,
            servo_command_angle,
            servo_target_angle,
            servo_motion_direction,
            servo_backlash_remaining,
        ) = self._update_servos(
            state, parameters, control[:, 2:], physics_dt
        )

        force_components = self._forces_and_moments(
            parameters, total_thrust, motor_torque, servo_angle
        )
        force_b = force_components["force_b"]
        moment_b = force_components["moment_b"]
        linear_acceleration_n, angular_acceleration_b = self._accelerations(
            state, parameters, force_b, moment_b
        )

        # 半隐式 Euler：先更新线速度和角速度，再用新速度推进位置和姿态。
        velocity_n = state["velocity_n"] + linear_acceleration_n * physics_dt
        position_n = state["position_n"] + velocity_n * physics_dt
        angular_velocity_b = (
            state["angular_velocity_b"] + angular_acceleration_b * physics_dt
        )
        attitude_q_wb = self._integrate_quaternion(
            state["attitude_q_wb"], angular_velocity_b, physics_dt
        )

        next_state.update(
            {
                "position_n": position_n,
                "velocity_n": velocity_n,
                "attitude_q_wb": attitude_q_wb,
                "angular_velocity_b": angular_velocity_b,
                "linear_acceleration_n": linear_acceleration_n,
                "angular_acceleration_b": angular_acceleration_b,
                "motor_speed": motor_speed,
                "effective_motor_speed": effective_motor_speed,
                "total_thrust": total_thrust,
                "motor_torque": motor_torque,
                "servo_angle": servo_angle,
                "servo_effective_pwm": servo_effective_pwm,
                "servo_command_angle": servo_command_angle,
                "servo_target_angle": servo_target_angle,
                "servo_motion_direction": servo_motion_direction,
                "servo_backlash_remaining": servo_backlash_remaining,
                **force_components,
            }
        )
        return next_state

    def _update_motors(
        self,
        state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        motor_pwm: torch.Tensor,
        physics_dt: float,
        instance_seeds: torch.Tensor,
        noise_counters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """更新上下桨转速，并计算耦合总推力和各桨反扭矩幅值。"""
        target_speed = self._lookup(
            motor_pwm,
            self._derived_parameters["motor_table_x"],
            self._derived_parameters["motor_table_y"],
        )
        # 死区内强制目标转速为零；PWM 恰好位于死区边界时也视为停转命令。
        target_speed = torch.where(
            motor_pwm <= parameters["motors.pwm_deadzone"],
            torch.zeros_like(target_speed),
            target_speed,
        )
        current_speed = state["motor_speed"]
        # 控制量在物理步内恒定，因此直接使用一阶惯性环节的精确离散解。
        response = self._derived_parameters["motor_response"]
        motor_speed = current_speed + response * (target_speed - current_speed)

        # 噪声只作用于曲线查表，不反馈到电机内部转速状态，且有效转速不得为负。
        noise = self._motor_noise(instance_seeds, noise_counters)
        effective_speed = torch.clamp_min(
            motor_speed + noise * parameters["motors.noise.stddev"], 0.0
        )
        upper_speed, lower_speed = effective_speed.unbind(dim=1)
        k1, k2, k3 = parameters["aerodynamics.thrust_coefficients"].unbind(dim=1)
        total_thrust = (
            k1 * upper_speed.square()
            + k2 * lower_speed.square()
            + k3 * upper_speed * lower_speed
        )
        motor_torque = parameters["motors.torque_coefficient"] * effective_speed.square()
        return motor_speed, effective_speed, total_thrust, motor_torque

    def _update_servos(
        self,
        state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        servo_pwm: torch.Tensor,
        physics_dt: float,
    ) -> tuple[torch.Tensor, ...]:
        """更新三个舵机的死区、反向回差、一阶惯性和速度限制状态。"""
        previous_pwm = state["servo_effective_pwm"]
        # PWM 变化不足以跨越死区时保持上一次有效命令，避免舵机在死区内抖动。
        outside_deadzone = (
            torch.abs(servo_pwm - previous_pwm) >= parameters["servos.deadzone"]
        )
        effective_pwm = torch.where(outside_deadzone, servo_pwm, previous_pwm)
        command_angle = self._lookup(
            effective_pwm,
            self._derived_parameters["servo_table_x"],
            self._derived_parameters["servo_table_y"],
        )

        command_delta = command_angle - state["servo_command_angle"]
        command_direction = torch.sign(command_delta)
        previous_direction = state["servo_motion_direction"]
        # 只有非零运动方向真正反转时才重新装载一段完整机械回差。
        reversing = (
            (command_direction != 0)
            & (previous_direction != 0)
            & (command_direction != previous_direction)
        )
        backlash_remaining = torch.where(
            reversing,
            parameters["servos.backlash"],
            state["servo_backlash_remaining"],
        )
        backlash_consumed = torch.minimum(
            torch.abs(command_delta), backlash_remaining
        )
        backlash_remaining = backlash_remaining - backlash_consumed
        # 命令变化先消耗回差，只有超出剩余间隙的部分才能传到机械目标角。
        transmitted_delta = command_direction * (
            torch.abs(command_delta) - backlash_consumed
        )
        target_angle = state["servo_target_angle"] + transmitted_delta
        motion_direction = torch.where(
            command_direction != 0, command_direction, previous_direction
        )

        # 舵机一阶响应得到的角速度还要受到实际最大机械速度约束。
        angle_rate = torch.clamp(
            (target_angle - state["servo_angle"]) / parameters["servos.tau"],
            min=-parameters["servos.max_speed"],
            max=parameters["servos.max_speed"],
        )
        servo_angle = state["servo_angle"] + angle_rate * physics_dt
        return (
            servo_angle,
            effective_pwm,
            command_angle,
            target_angle,
            motion_direction,
            backlash_remaining,
        )

    def _forces_and_moments(
        self,
        parameters: Mapping[str, torch.Tensor],
        total_thrust: torch.Tensor,
        motor_torque: torch.Tensor,
        servo_angle: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """计算直接推力、三个格栅矢量推力及其相对质心的力矩。"""
        partition = parameters["aerodynamics.thrust_partition"]
        neutral_direction = parameters[
            "aerodynamics.neutral_thrust_direction_b"
        ]

        # 与舵角无关的直接推力作用在独立气动中心，偏置同样会产生力矩。
        direct_force_b = (
            total_thrust * partition[:, 0]
        )[:, None] * neutral_direction
        direct_arm_b = self._derived_parameters["direct_arm_b"]
        direct_moment_b = torch.linalg.cross(direct_arm_b, direct_force_b)

        # 自身衰减由绝对舵角决定，耦合矩阵的第 i 行表示其他格栅对格栅 i 的损失。
        attenuation = self._lookup(
            torch.abs(servo_angle),
            self._derived_parameters["attenuation_table_x"],
            self._derived_parameters["attenuation_table_y"],
        )
        coupling_loss = torch.bmm(
            parameters["aerodynamics.coupling_attenuation"],
            (1.0 - attenuation).unsqueeze(-1),
        ).squeeze(-1)
        effective_attenuation = torch.clamp(
            attenuation - coupling_loss, min=0.0, max=1.0
        )
        # 机械舵角通过一阶线性特性映射为推力矢量偏转角。
        vector_angle = (
            parameters["aerodynamics.grids.vector_deflection.gain"]
            * servo_angle
            + parameters["aerodynamics.grids.vector_deflection.offset"]
        )
        # 正偏转遵循各 deflection_axis_b 的右手定则；中立时受力方向保持不变。
        grid_direction_b = self._rotate_about_axis(
            neutral_direction[:, None, :].expand(-1, 3, -1),
            parameters["aerodynamics.grids.deflection_axis_b"],
            vector_angle,
        )
        grid_thrust = (
            total_thrust[:, None]
            * partition[:, 1:]
            * effective_attenuation
        )
        grid_force_b = grid_thrust[:, :, None] * grid_direction_b
        # 三个作用点分别保留，先逐格栅计算 r×F，再只在格栅维求和。
        grid_arm_b = self._derived_parameters["grid_arm_b"]
        grid_moment_b = torch.linalg.cross(grid_arm_b, grid_force_b)

        # 上下桨旋向相反；不设扭矩耦合项，机体反扭矩为两桨反扭矩之差。
        airflow_axis_b = self._derived_parameters["airflow_axis_b"]
        reaction_torque = motor_torque[:, 0] - motor_torque[:, 1]
        motor_reaction_moment_b = reaction_torque[:, None] * airflow_axis_b
        force_b = direct_force_b + grid_force_b.sum(dim=1)
        moment_b = (
            direct_moment_b
            + grid_moment_b.sum(dim=1)
            + motor_reaction_moment_b
        )
        return {
            "direct_force_b": direct_force_b,
            "grid_force_b": grid_force_b,
            "direct_moment_b": direct_moment_b,
            "grid_moment_b": grid_moment_b,
            "motor_reaction_moment_b": motor_reaction_moment_b,
            "grid_effective_attenuation": effective_attenuation,
            "force_b": force_b,
            "moment_b": moment_b,
        }

    def _accelerations(
        self,
        state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        force_b: torch.Tensor,
        moment_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """由总外力和总外力矩计算质心线加速度与机体系角加速度。"""
        # q_wb 将机体系合力旋转到 NED；NED 的重力方向是 +z_n。
        force_n = self._rotate_body_to_world(state["attitude_q_wb"], force_b)
        linear_acceleration_n = (
            force_n / parameters["body.mass"][:, None] + self._gravity_n
        )
        # 接口不考虑惯量积，因此 Iω 可由三轴主惯量逐元素相乘得到。
        angular_velocity = state["angular_velocity_b"]
        inertia = parameters["body.inertia_diagonal_b"]
        angular_momentum = inertia * angular_velocity
        # 机体系欧拉方程：I·ω_dot = M - ω×(Iω)。
        gyroscopic = torch.linalg.cross(angular_velocity, angular_momentum)
        angular_acceleration_b = (moment_b - gyroscopic) / inertia
        return linear_acceleration_n, angular_acceleration_b

    @staticmethod
    def _lookup(
        value: torch.Tensor,
        x_axis: torch.Tensor,
        y_axis: torch.Tensor,
    ) -> torch.Tensor:
        """对批量分段线性表进行端点钳位插值，保留所有前导维。"""
        clamped = torch.maximum(torch.minimum(value, x_axis[..., -1]), x_axis[..., 0])
        upper = torch.searchsorted(x_axis, clamped.unsqueeze(-1)).squeeze(-1)
        upper = torch.clamp(upper, min=1, max=x_axis.shape[-1] - 1)
        lower = upper - 1
        x0 = torch.gather(x_axis, -1, lower.unsqueeze(-1)).squeeze(-1)
        x1 = torch.gather(x_axis, -1, upper.unsqueeze(-1)).squeeze(-1)
        y0 = torch.gather(y_axis, -1, lower.unsqueeze(-1)).squeeze(-1)
        y1 = torch.gather(y_axis, -1, upper.unsqueeze(-1)).squeeze(-1)
        return y0 + (clamped - x0) * (y1 - y0) / (x1 - x0)

    @staticmethod
    def _rotate_about_axis(
        vector: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        """使用 Rodrigues 公式将向量绕单位轴按右手定则旋转。"""
        cosine = torch.cos(angle)[..., None]
        sine = torch.sin(angle)[..., None]
        projection = (axis * vector).sum(dim=-1, keepdim=True)
        return (
            vector * cosine
            + torch.linalg.cross(axis, vector) * sine
            + axis * projection * (1.0 - cosine)
        )

    @staticmethod
    def _rotate_body_to_world(q_wb: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
        """用 Hamilton 四元数 ``q_wb`` 将 ``[B,3]`` 向量从机体系旋转到世界系。"""
        q_vector = q_wb[:, 1:]
        cross = torch.linalg.cross(q_vector, vector_b)
        return vector_b + 2.0 * q_wb[:, :1] * cross + 2.0 * torch.linalg.cross(
            q_vector, cross
        )

    @staticmethod
    def _integrate_quaternion(
        q_wb: torch.Tensor, angular_velocity_b: torch.Tensor, physics_dt: float
    ) -> torch.Tensor:
        """按 ``q_dot = 0.5*q⊗[0,ω_b]`` 积分姿态并逐实例归一化。"""
        scalar = q_wb[:, :1]
        vector = q_wb[:, 1:]
        q_dot_scalar = -(vector * angular_velocity_b).sum(dim=1, keepdim=True)
        q_dot_vector = (
            scalar * angular_velocity_b
            + torch.linalg.cross(vector, angular_velocity_b)
        )
        candidate = q_wb + 0.5 * physics_dt * torch.cat(
            (q_dot_scalar, q_dot_vector), dim=1
        )
        return candidate / torch.linalg.vector_norm(
            candidate, dim=1, keepdim=True
        ).clamp_min(torch.finfo(candidate.dtype).tiny)

    def _motor_noise(
        self, seeds: torch.Tensor, counters: torch.Tensor
    ) -> torch.Tensor:
        """由实例、子系统、电机编号和计数器生成独立标准正态噪声。"""
        # 每个 [实例, 电机] 元素拥有独立 key，暂停某行不会消耗其他行的随机序列。
        key = (
            seeds[:, None]
            ^ ((counters + 1) * 6364136223846793005)
            ^ ((self._batch_index[:, None] + 1) * 1442695040888963407)
            ^ self._motor_subsystem_id
            ^ ((self._motor_index + 1) * 3202034522624059733)
        )
        u1 = self._uniform_from_key(key ^ 2862933555777941757)
        u2 = self._uniform_from_key(key ^ 3037000493)
        return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)

    def _uniform_from_key(self, key: torch.Tensor) -> torch.Tensor:
        """将 int64 key 混合为开区间 ``(0,1)`` 上的确定性均匀数。"""
        # 采用纯张量整数混合，溢出按 int64 二进制补码自然回绕。
        key = key - 7046029254386353131
        key = (key ^ self._logical_right_shift(key, 30)) * -4658895280553007687
        key = (key ^ self._logical_right_shift(key, 27)) * -7723592293110705685
        key = key ^ self._logical_right_shift(key, 31)
        mantissa = torch.bitwise_and(key, (1 << 53) - 1).to(self._dtype)
        return (mantissa + 1.0) / float((1 << 53) + 1)

    @staticmethod
    def _logical_right_shift(value: torch.Tensor, bits: int) -> torch.Tensor:
        """在 PyTorch 有符号 int64 张量上实现逻辑右移。"""
        mask = (1 << (64 - bits)) - 1
        return torch.bitwise_and(torch.bitwise_right_shift(value, bits), mask)
