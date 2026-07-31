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
            "motor_midpoint_response": 1.0
            - torch.exp(
                -0.5 * physics_dt / parameters["motors.time_constant"]
            ),
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
        (
            motor_speed,
            effective_motor_speed,
            total_thrust,
            motor_torque,
            midpoint_total_thrust,
            midpoint_motor_torque,
        ) = self._update_motors(
            state,
            parameters,
            control[:, :2],
            instance_seeds,
            motor_noise_counters,
        )
        (
            servo_angle,
            midpoint_servo_angle,
            servo_effective_pwm,
            servo_command_angle,
            servo_target_angle,
            servo_motion_direction,
            servo_backlash_remaining,
        ) = self._update_servos(
            state, parameters, control[:, 2:], physics_dt
        )

        # 执行器在 2 ms 宏步内按解析模型推进；周期中点的力和力矩用于二阶刚体积分，
        # 周期末的分量则作为可观测真值和日志值，保证所有公开字段位于同一时间戳。
        midpoint_force_components = self._forces_and_moments(
            parameters,
            midpoint_total_thrust,
            midpoint_motor_torque,
            midpoint_servo_angle,
        )
        force_components = self._forces_and_moments(
            parameters, total_thrust, motor_torque, servo_angle
        )

        midpoint_moment_b = midpoint_force_components["moment_b"]
        initial_angular_acceleration = self._angular_acceleration(
            state["angular_velocity_b"],
            parameters["body.inertia_diagonal_b"],
            midpoint_moment_b,
        )
        midpoint_angular_velocity_b = (
            state["angular_velocity_b"]
            + 0.5 * initial_angular_acceleration * physics_dt
        )
        midpoint_angular_acceleration = self._angular_acceleration(
            midpoint_angular_velocity_b,
            parameters["body.inertia_diagonal_b"],
            midpoint_moment_b,
        )
        angular_velocity_b = (
            state["angular_velocity_b"]
            + midpoint_angular_acceleration * physics_dt
        )
        midpoint_attitude_q_wb = self._integrate_quaternion(
            state["attitude_q_wb"],
            midpoint_angular_velocity_b,
            0.5 * physics_dt,
        )
        attitude_q_wb = self._integrate_quaternion(
            state["attitude_q_wb"],
            midpoint_angular_velocity_b,
            physics_dt,
        )

        midpoint_linear_acceleration_n = self._linear_acceleration(
            midpoint_attitude_q_wb,
            parameters["body.mass"],
            midpoint_force_components["force_b"],
        )
        velocity_n = (
            state["velocity_n"] + midpoint_linear_acceleration_n * physics_dt
        )
        position_n = (
            state["position_n"]
            + state["velocity_n"] * physics_dt
            + 0.5 * midpoint_linear_acceleration_n * physics_dt * physics_dt
        )

        # 加速度真值使用周期末状态和周期末气动力，供日志与传感器在同一时间戳读取。
        linear_acceleration_n = self._linear_acceleration(
            attitude_q_wb,
            parameters["body.mass"],
            force_components["force_b"],
        )
        angular_acceleration_b = self._angular_acceleration(
            angular_velocity_b,
            parameters["body.inertia_diagonal_b"],
            force_components["moment_b"],
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
        instance_seeds: torch.Tensor,
        noise_counters: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """解析推进上下桨，并计算周期中点和周期末推力/反扭矩。"""
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
        midpoint_speed = current_speed + self._derived_parameters[
            "motor_midpoint_response"
        ] * (target_speed - current_speed)

        # 每个 2 ms 仿真步采样一次运行时转速扰动，在本步内保持；扰动不反馈
        # 到电机内部状态。周期末有效转速仍是公开真值字段的定义。
        noise = self._motor_noise(instance_seeds, noise_counters)
        speed_noise = noise * parameters["motors.noise.stddev"]
        effective_speed = torch.clamp_min(
            motor_speed + speed_noise, 0.0
        )
        midpoint_effective_speed = torch.clamp_min(
            midpoint_speed + speed_noise, 0.0
        )
        total_thrust, motor_torque = self._motor_forces(
            effective_speed, parameters
        )
        midpoint_total_thrust, midpoint_motor_torque = self._motor_forces(
            midpoint_effective_speed, parameters
        )
        return (
            motor_speed,
            effective_speed,
            total_thrust,
            motor_torque,
            midpoint_total_thrust,
            midpoint_motor_torque,
        )

    @staticmethod
    def _motor_forces(
        effective_speed: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        upper_speed, lower_speed = effective_speed.unbind(dim=1)
        k1, k2, k3 = parameters[
            "aerodynamics.thrust_coefficients"
        ].unbind(dim=1)
        total_thrust = (
            k1 * upper_speed.square()
            + k2 * lower_speed.square()
            + k3 * upper_speed * lower_speed
        )
        motor_torque = (
            parameters["motors.torque_coefficient"]
            * effective_speed.square()
        )
        return total_thrust, motor_torque

    def _update_servos(
        self,
        state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        servo_pwm: torch.Tensor,
        physics_dt: float,
    ) -> tuple[torch.Tensor, ...]:
        """处理离散机械事件，并解析推进带限速的一阶舵机。"""
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

        midpoint_servo_angle = self._servo_response(
            state["servo_angle"],
            target_angle,
            parameters["servos.tau"],
            parameters["servos.max_speed"],
            0.5 * physics_dt,
        )
        servo_angle = self._servo_response(
            state["servo_angle"],
            target_angle,
            parameters["servos.tau"],
            parameters["servos.max_speed"],
            physics_dt,
        )
        return (
            servo_angle,
            midpoint_servo_angle,
            effective_pwm,
            command_angle,
            target_angle,
            motion_direction,
            backlash_remaining,
        )

    @staticmethod
    def _servo_response(
        initial_angle: torch.Tensor,
        target_angle: torch.Tensor,
        tau: torch.Tensor,
        max_speed: torch.Tensor,
        duration: float,
    ) -> torch.Tensor:
        """精确求解 ``angle_dot=clip((target-angle)/tau, ±max_speed)``。"""
        error = target_angle - initial_angle
        direction = torch.sign(error)
        exponential_boundary = tau * max_speed
        linear_time = torch.clamp_min(
            (torch.abs(error) - exponential_boundary) / max_speed,
            0.0,
        )
        duration_tensor = torch.full_like(initial_angle, duration)
        linear_duration = torch.minimum(linear_time, duration_tensor)
        angle_after_linear = (
            initial_angle + direction * max_speed * linear_duration
        )
        exponential_duration = duration_tensor - linear_duration
        return target_angle + (
            angle_after_linear - target_angle
        ) * torch.exp(-exponential_duration / tau)

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

    def _linear_acceleration(
        self,
        attitude_q_wb: torch.Tensor,
        mass: torch.Tensor,
        force_b: torch.Tensor,
    ) -> torch.Tensor:
        """由同一时刻的姿态和机体系合力计算 NED 线加速度。"""
        force_n = self._rotate_body_to_world(attitude_q_wb, force_b)
        return force_n / mass[:, None] + self._gravity_n

    @staticmethod
    def _angular_acceleration(
        angular_velocity: torch.Tensor,
        inertia: torch.Tensor,
        moment_b: torch.Tensor,
    ) -> torch.Tensor:
        """由同一时刻的角速度和外力矩计算机体系角加速度。"""
        angular_momentum = inertia * angular_velocity
        gyroscopic = torch.linalg.cross(angular_velocity, angular_momentum)
        return (moment_b - gyroscopic) / inertia

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
        """用恒定机体系角速度的四元数指数映射推进姿态。"""
        angular_speed = torch.linalg.vector_norm(
            angular_velocity_b, dim=1, keepdim=True
        )
        half_angle = 0.5 * physics_dt * angular_speed
        half_dt = torch.full_like(angular_speed, 0.5 * physics_dt)
        vector_scale = torch.where(
            angular_speed > torch.finfo(q_wb.dtype).eps,
            torch.sin(half_angle) / angular_speed.clamp_min(
                torch.finfo(q_wb.dtype).tiny
            ),
            half_dt,
        )
        delta_scalar = torch.cos(half_angle)
        delta_vector = angular_velocity_b * vector_scale

        scalar = q_wb[:, :1]
        vector = q_wb[:, 1:]
        candidate_scalar = (
            scalar * delta_scalar
            - (vector * delta_vector).sum(dim=1, keepdim=True)
        )
        candidate_vector = (
            scalar * delta_vector
            + delta_scalar * vector
            + torch.linalg.cross(vector, delta_vector)
        )
        candidate = torch.cat((candidate_scalar, candidate_vector), dim=1)
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
