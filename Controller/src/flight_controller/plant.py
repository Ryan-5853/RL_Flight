from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .math import inverse_lookup, lookup


@dataclass(frozen=True)
class HoverTrim:
    command: torch.Tensor
    motor_speed: torch.Tensor
    thrust: torch.Tensor


class LocalPlantModel:
    """Nominal steady wrench and actuator model derived from SimEnv parameters."""

    gravity = 9.80665

    def __init__(self, parameters: Mapping[str, torch.Tensor]) -> None:
        self.p = parameters

    def hover_trim(self) -> HoverTrim:
        mass = self.p["body.mass"]
        coefficients = self.p["aerodynamics.thrust_coefficients"]
        torque = self.p["motors.torque_coefficient"]
        ratio = torch.sqrt(
            torque[:, 0].clamp_min(1e-16) / torque[:, 1].clamp_min(1e-16)
        )
        denominator = (
            coefficients[:, 0]
            + coefficients[:, 1] * ratio.square()
            + coefficients[:, 2] * ratio
        )
        upper = torch.sqrt(mass * self.gravity / denominator.clamp_min(1e-16))
        lower = ratio * upper
        speed = torch.stack((upper, lower), dim=1)
        maximum_speed = self.p["motors.pwm_to_rpm_table"][..., -1, 1]
        if bool((speed > maximum_speed * (1.0 + 1e-6)).any().item()):
            raise ValueError(
                "configured motors cannot produce a torque-balanced hover at full PWM"
            )
        motor_pwm = inverse_lookup(speed, self.p["motors.pwm_to_rpm_table"])
        command = torch.cat((motor_pwm, torch.zeros_like(speed[:, :1]).expand(-1, 3)), dim=1)
        return HoverTrim(command=command, motor_speed=speed, thrust=mass * self.gravity)

    def balanced_motor_target(self, desired_thrust: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.p["aerodynamics.thrust_coefficients"]
        torque = self.p["motors.torque_coefficient"]
        ratio = torch.sqrt(
            torque[:, 0].clamp_min(1e-16) / torque[:, 1].clamp_min(1e-16)
        )
        denominator = (
            coefficients[:, 0]
            + coefficients[:, 1] * ratio.square()
            + coefficients[:, 2] * ratio
        )
        upper = torch.sqrt(desired_thrust.clamp_min(0.0) / denominator.clamp_min(1e-16))
        lower = ratio * upper
        speed = torch.stack((upper, lower), dim=1)
        pwm = inverse_lookup(speed, self.p["motors.pwm_to_rpm_table"]).clamp(0.0, 1.0)
        return speed, pwm

    def thrust_from_upper_pwm(self, upper_pwm: torch.Tensor) -> torch.Tensor:
        table = self.p["motors.pwm_to_rpm_table"]
        upper_speed = lookup(upper_pwm, table[:, 0])
        torque = self.p["motors.torque_coefficient"]
        ratio = torch.sqrt(
            torque[:, 0].clamp_min(1e-16) / torque[:, 1].clamp_min(1e-16)
        )
        lower_speed = ratio * upper_speed
        coefficients = self.p["aerodynamics.thrust_coefficients"]
        return (
            coefficients[:, 0] * upper_speed.square()
            + coefficients[:, 1] * lower_speed.square()
            + coefficients[:, 2] * upper_speed * lower_speed
        )

    def steady_wrench(self, command: torch.Tensor) -> torch.Tensor:
        motor_speed = lookup(command[:, :2], self.p["motors.pwm_to_rpm_table"])
        servo_angle = lookup(command[:, 2:], self.p["servos.pwm_angle_table"])
        return self.wrench_from_actuators(motor_speed, servo_angle)

    def wrench_from_actuators(
        self,
        motor_speed: torch.Tensor,
        servo_angle: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = self.p["aerodynamics.thrust_coefficients"]
        total_thrust = (
            coefficients[:, 0] * motor_speed[:, 0].square()
            + coefficients[:, 1] * motor_speed[:, 1].square()
            + coefficients[:, 2] * motor_speed[:, 0] * motor_speed[:, 1]
        )
        neutral = self.p["aerodynamics.neutral_thrust_direction_b"]
        partition = self.p["aerodynamics.thrust_partition"]
        direct_force = total_thrust[:, None] * partition[:, :1] * neutral
        direct_arm = (
            self.p["aerodynamics.direct_thrust_center_b"]
            - self.p["body.center_of_mass_b"]
        )
        direct_moment = torch.linalg.cross(direct_arm, direct_force)

        attenuation = lookup(
            servo_angle.abs(),
            self.p["aerodynamics.grids.self_attenuation_curve"],
        )
        coupling_loss = torch.bmm(
            self.p["aerodynamics.coupling_attenuation"],
            (1.0 - attenuation).unsqueeze(-1),
        ).squeeze(-1)
        effective = (attenuation - coupling_loss).clamp(0.0, 1.0)
        vector_angle = (
            self.p["aerodynamics.grids.vector_deflection.gain"] * servo_angle
            + self.p["aerodynamics.grids.vector_deflection.offset"]
        )
        axes = self.p["aerodynamics.grids.deflection_axis_b"]
        base = neutral[:, None, :].expand(-1, 3, -1)
        cosine = torch.cos(vector_angle)[..., None]
        sine = torch.sin(vector_angle)[..., None]
        projection = (axes * base).sum(dim=-1, keepdim=True)
        direction = (
            base * cosine
            + torch.linalg.cross(axes, base) * sine
            + axes * projection * (1.0 - cosine)
        )
        grid_thrust = total_thrust[:, None] * partition[:, 1:] * effective
        grid_force = grid_thrust[..., None] * direction
        grid_arm = (
            self.p["aerodynamics.grids.aerodynamic_center_b"]
            - self.p["body.center_of_mass_b"][:, None, :]
        )
        grid_moment = torch.linalg.cross(grid_arm, grid_force).sum(dim=1)
        motor_torque = self.p["motors.torque_coefficient"] * motor_speed.square()
        reaction = (motor_torque[:, 0] - motor_torque[:, 1])[:, None] * (-neutral)
        moment = direct_moment + grid_moment + reaction
        return torch.cat((total_thrust[:, None], moment), dim=1)

    def control_effectiveness(self, trim_command: torch.Tensor) -> torch.Tensor:
        """Return a per-instance steady ``d[T,Mx,My,Mz]/du`` matrix."""

        columns = []
        step = trim_command.new_tensor([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
        for index in range(5):
            offset = torch.zeros_like(trim_command)
            offset[:, index] = step[index]
            plus = self.steady_wrench((trim_command + offset).clamp(
                trim_command.new_tensor([0, 0, -1, -1, -1]),
                trim_command.new_tensor([1, 1, 1, 1, 1]),
            ))
            minus = self.steady_wrench((trim_command - offset).clamp(
                trim_command.new_tensor([0, 0, -1, -1, -1]),
                trim_command.new_tensor([1, 1, 1, 1, 1]),
            ))
            columns.append((plus - minus) / (2.0 * step[index]))
        return torch.stack(columns, dim=2)
