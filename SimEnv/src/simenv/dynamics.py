from __future__ import annotations

import hashlib
import math
from typing import Mapping

import torch


class TensorDynamicsKernel:
    """Tensorized actuators, thrust-vector aerodynamics, and 6-DoF rigid body."""

    implemented = True
    _GRAVITY_N = (0.0, 0.0, 9.80665)

    def __init__(self, parallel_count: int, device: torch.device, dtype: torch.dtype) -> None:
        self._parallel_count = parallel_count
        self._device = device
        self._dtype = dtype
        self._batch_index = torch.arange(
            parallel_count, dtype=torch.int64, device=device
        )
        digest = hashlib.blake2b(b"motors", digest_size=8).digest()
        self._motor_subsystem_id = int.from_bytes(digest, "little") & ((1 << 62) - 1)

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
        next_state = dict(state)

        motor_speed, motor_thrust, motor_torque = self._update_motors(
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
            parameters, motor_thrust, motor_torque, servo_angle
        )
        force_b = force_components["force_b"]
        moment_b = force_components["moment_b"]
        linear_acceleration_n, angular_acceleration_b = self._accelerations(
            state, parameters, force_b, moment_b
        )

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
                "motor_thrust": motor_thrust,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_speed = self._lookup(
            motor_pwm, parameters["motors.pwm_to_rpm_table"]
        )
        target_speed = torch.where(
            motor_pwm <= parameters["motors.pwm_deadzone"],
            torch.zeros_like(target_speed),
            target_speed,
        )
        current_speed = state["motor_speed"]
        tau = torch.where(
            target_speed >= current_speed,
            parameters["motors.tau_up"],
            parameters["motors.tau_down"],
        )
        response = 1.0 - torch.exp(-physics_dt / tau)
        motor_speed = current_speed + response * (target_speed - current_speed)

        noise = self._motor_noise(instance_seeds, noise_counters)
        effective_speed = torch.clamp_min(
            motor_speed + noise * parameters["motors.noise.stddev"], 0.0
        )
        motor_thrust = self._lookup(
            effective_speed, parameters["motors.thrust_curve"]
        )
        motor_torque = self._lookup(
            effective_speed, parameters["motors.torque_curve"]
        ) * parameters["motors.torque_sign"]
        return motor_speed, motor_thrust, motor_torque

    def _update_servos(
        self,
        state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        servo_pwm: torch.Tensor,
        physics_dt: float,
    ) -> tuple[torch.Tensor, ...]:
        previous_pwm = state["servo_effective_pwm"]
        outside_deadzone = (
            torch.abs(servo_pwm - previous_pwm) >= parameters["servos.deadzone"]
        )
        effective_pwm = torch.where(outside_deadzone, servo_pwm, previous_pwm)
        command_angle = self._lookup(
            effective_pwm, parameters["servos.pwm_angle_table"]
        )

        command_delta = command_angle - state["servo_command_angle"]
        command_direction = torch.sign(command_delta)
        previous_direction = state["servo_motion_direction"]
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
        transmitted_delta = command_direction * (
            torch.abs(command_delta) - backlash_consumed
        )
        target_angle = state["servo_target_angle"] + transmitted_delta
        motion_direction = torch.where(
            command_direction != 0, command_direction, previous_direction
        )

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
        motor_thrust: torch.Tensor,
        motor_torque: torch.Tensor,
        servo_angle: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        total_thrust = motor_thrust.sum(dim=1)
        partition = parameters["aerodynamics.thrust_partition"]
        neutral_direction = parameters[
            "aerodynamics.neutral_thrust_direction_b"
        ]

        direct_force_b = (
            total_thrust * partition[:, 0]
        )[:, None] * neutral_direction
        direct_arm_b = (
            parameters["aerodynamics.direct_thrust_center_b"]
            - parameters["body.center_of_mass_b"]
        )
        direct_moment_b = torch.linalg.cross(direct_arm_b, direct_force_b)

        attenuation = self._lookup(
            torch.abs(servo_angle),
            parameters["aerodynamics.grids.self_attenuation_curve"],
        )
        coupling_loss = torch.bmm(
            parameters["aerodynamics.coupling_attenuation"],
            (1.0 - attenuation).unsqueeze(-1),
        ).squeeze(-1)
        effective_attenuation = torch.clamp(
            attenuation - coupling_loss, min=0.0, max=1.0
        )
        vector_angle = (
            parameters["aerodynamics.grids.vector_deflection.gain"]
            * servo_angle
            + parameters["aerodynamics.grids.vector_deflection.offset"]
        )
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
        grid_arm_b = (
            parameters["aerodynamics.grids.aerodynamic_center_b"]
            - parameters["body.center_of_mass_b"][:, None, :]
        )
        grid_moment_b = torch.linalg.cross(grid_arm_b, grid_force_b)

        airflow_axis_b = -neutral_direction
        motor_reaction_moment_b = motor_torque.sum(dim=1)[:, None] * airflow_axis_b
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
        force_n = self._rotate_body_to_world(state["attitude_q_wb"], force_b)
        gravity_n = torch.as_tensor(
            self._GRAVITY_N, dtype=self._dtype, device=self._device
        )
        linear_acceleration_n = (
            force_n / parameters["body.mass"][:, None] + gravity_n
        )
        inertia = parameters["body.inertia_diagonal_b"]
        angular_velocity = state["angular_velocity_b"]
        angular_momentum = inertia * angular_velocity
        gyroscopic = torch.linalg.cross(angular_velocity, angular_momentum)
        angular_acceleration_b = (moment_b - gyroscopic) / inertia
        return linear_acceleration_n, angular_acceleration_b

    @staticmethod
    def _lookup(value: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        x_axis = table[..., 0].contiguous()
        y_axis = table[..., 1]
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
        q_vector = q_wb[:, 1:]
        cross = torch.linalg.cross(q_vector, vector_b)
        return vector_b + 2.0 * q_wb[:, :1] * cross + 2.0 * torch.linalg.cross(
            q_vector, cross
        )

    @staticmethod
    def _integrate_quaternion(
        q_wb: torch.Tensor, angular_velocity_b: torch.Tensor, physics_dt: float
    ) -> torch.Tensor:
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
        motor_index = torch.arange(2, dtype=torch.int64, device=self._device)[None, :]
        key = (
            seeds[:, None]
            ^ ((counters + 1) * 6364136223846793005)
            ^ ((self._batch_index[:, None] + 1) * 1442695040888963407)
            ^ self._motor_subsystem_id
            ^ ((motor_index + 1) * 3202034522624059733)
        )
        u1 = self._uniform_from_key(key ^ 2862933555777941757)
        u2 = self._uniform_from_key(key ^ 3037000493)
        return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)

    def _uniform_from_key(self, key: torch.Tensor) -> torch.Tensor:
        key = key - 7046029254386353131
        key = (key ^ self._logical_right_shift(key, 30)) * -4658895280553007687
        key = (key ^ self._logical_right_shift(key, 27)) * -7723592293110705685
        key = key ^ self._logical_right_shift(key, 31)
        mantissa = torch.bitwise_and(key, (1 << 53) - 1).to(self._dtype)
        return (mantissa + 1.0) / float((1 << 53) + 1)

    @staticmethod
    def _logical_right_shift(value: torch.Tensor, bits: int) -> torch.Tensor:
        mask = (1 << (64 - bits)) - 1
        return torch.bitwise_and(torch.bitwise_right_shift(value, bits), mask)
