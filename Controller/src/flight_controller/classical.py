from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
from scipy.linalg import expm, solve_discrete_are

from .allocation import WeightedControlAllocator
from .base import FlightController
from .math import lookup, quaternion_rotation_error, tilt_cosine
from .pilot import VirtualPilotHeightController
from .plant import LocalPlantModel
from .types import ControllerContext, ControllerOutput, ControllerReference, ControllerState


def _node(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"controller.{name} must be a mapping")
    return value


def _number(config: Mapping[str, Any], name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"controller parameter {name} must be finite")
    return value


def _vector(
    config: Mapping[str, Any],
    name: str,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    raw = config.get(name, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != len(default):
        raise ValueError(f"controller parameter {name} must have length {len(default)}")
    values = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"controller parameter {name} must contain finite values")
    return values


class ClassicalControllerBase(FlightController):
    """Shared trim, altitude loop, allocation and LQR model."""

    def __init__(
        self,
        context: ControllerContext,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(context, config)
        config = self.config
        self.collective_mode = str(config.get("collective_mode", "hover"))
        if self.collective_mode not in {"hover", "manual", "external_upper"}:
            raise ValueError(
                "collective_mode must be hover, manual or external_upper"
            )
        self.upper_external = self.collective_mode == "external_upper"
        self.plant = LocalPlantModel(context.parameters)
        self.trim = self.plant.hover_trim()
        effectiveness = self.plant.control_effectiveness(self.trim.command)
        allocation = _node(config, "allocation")
        weights = torch.tensor(
            _vector(allocation, "input_weights", (1.5, 1.5, 1.0, 1.0, 1.0)),
            device=context.device,
            dtype=context.dtype,
        ).expand(context.batch_size, -1)
        self.allocator = WeightedControlAllocator(
            effectiveness,
            weights,
            _number(allocation, "damping", 1e-5),
        )

        pid = _node(config, "pid")
        altitude = _node(pid, "altitude")
        attitude = _node(pid, "attitude")
        self.height_kp = _number(altitude, "kp", 4.0)
        self.height_ki = _number(altitude, "ki", 0.8)
        self.height_kd = _number(altitude, "kd", 3.6)
        self.height_integral_limit = _number(altitude, "integral_limit", 2.0)
        self.minimum_thrust_fraction = _number(
            altitude, "minimum_thrust_fraction", 0.20
        )
        self.maximum_thrust_fraction = _number(
            altitude, "maximum_thrust_fraction", 0.96
        )
        if self.upper_external:
            virtual_pilot = _node(config, "virtual_pilot")
            self.pilot = VirtualPilotHeightController(
                virtual_pilot,
                context.batch_size,
                context.device,
                context.dtype,
                context.control_dt,
            )
        inertia = context.parameters["body.inertia_diagonal_b"]
        rp_frequency = _number(attitude, "roll_pitch_natural_frequency_rad_s", 6.0)
        yaw_frequency = _number(attitude, "yaw_rate_bandwidth_rad_s", 5.0)
        damping = _number(attitude, "damping_ratio", 0.85)
        self.attitude_kp = torch.stack(
            (
                inertia[:, 0] * rp_frequency**2,
                inertia[:, 1] * rp_frequency**2,
                torch.zeros_like(inertia[:, 2]),
            ),
            dim=1,
        )
        self.attitude_kd = torch.stack(
            (
                2.0 * damping * inertia[:, 0] * rp_frequency,
                2.0 * damping * inertia[:, 1] * rp_frequency,
                inertia[:, 2] * yaw_frequency,
            ),
            dim=1,
        )
        integral_fraction = _vector(
            attitude, "integral_fraction", (0.08, 0.08, 0.04)
        )
        self.attitude_ki = self.attitude_kp.new_tensor(integral_fraction)[None] * torch.where(
            self.attitude_kp > 0,
            self.attitude_kp,
            self.attitude_kd,
        )
        self.attitude_integral_limit = self.attitude_kp.new_tensor(
            _vector(attitude, "integral_limit", (0.35, 0.35, 0.8))
        )
        self.max_moment = self.attitude_kp.new_tensor(
            _vector(attitude, "maximum_moment_n_m", (0.35, 0.35, 0.45))
        )

        self.height_integral = torch.zeros(
            context.batch_size, device=context.device, dtype=context.dtype
        )
        self.attitude_integral = torch.zeros(
            context.batch_size, 3, device=context.device, dtype=context.dtype
        )
        self.last_command = self.trim.command.clone()
        if self.upper_external:
            self.last_command[:, 0] = self.pilot.upper_throttle[:, 0]
        self._lqr_gain: torch.Tensor | None = None
        self._lqr_integral_enabled = False
        self._lqr_state_size = 10
        self._lqr_poles = np.asarray([], dtype=np.complex128)
        if self.controller_type != "pid":
            self._lqr_gain, self._lqr_poles = self._design_lqr(
                _node(config, "lqr")
            )

    def reset(self, reset_mask: torch.Tensor) -> None:
        mask = self._active_mask(reset_mask)
        self.height_integral.masked_fill_(mask, 0.0)
        self.attitude_integral.masked_fill_(mask[:, None], 0.0)
        if self.upper_external:
            self.pilot.reset(mask)
        self.last_command.copy_(
            torch.where(mask[:, None], self.trim.command, self.last_command)
        )
        if self.upper_external:
            self.last_command[:, 0] = torch.where(
                mask,
                self.pilot.upper_throttle[:, 0],
                self.last_command[:, 0],
            )

    def describe(self) -> dict[str, Any]:
        result = super().describe()
        result.update(
            {
                "collective_mode": self.collective_mode,
                "trim_command": self.trim.command[0].detach().cpu().tolist(),
                "trim_motor_speed_rad_s": self.trim.motor_speed[0].detach().cpu().tolist(),
            }
        )
        if self.upper_external:
            result["upper_motor_external"] = True
        if self._lqr_poles.size:
            result["lqr_closed_loop_pole_radius"] = float(
                np.max(np.abs(self._lqr_poles))
            )
        return result

    def schedule_lqr_gain(self, gain: torch.Tensor) -> None:
        """Install one shared gain or one gain per parallel vehicle."""
        if self._lqr_gain is None:
            raise RuntimeError("LQR gain scheduling requires an LQR controller")
        input_count = 4 if self.upper_external else 5
        expected_shared = (input_count, self._lqr_state_size)
        expected_batched = (
            self.context.batch_size,
            input_count,
            self._lqr_state_size,
        )
        if tuple(gain.shape) not in {expected_shared, expected_batched}:
            raise ValueError(
                f"LQR gain must have shape ({input_count}, {self._lqr_state_size}) or "
                f"{expected_batched}, received {tuple(gain.shape)}"
            )
        scheduled = gain.to(device=self.context.device, dtype=self.context.dtype)
        if not bool(torch.isfinite(scheduled).all().item()):
            raise ValueError("LQR gain must contain only finite values")
        self._lqr_gain = scheduled.detach().clone()

    def _desired_actuator_equilibrium(
        self,
        state: ControllerState,
        reference: ControllerReference,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.collective_mode == "external_upper":
            return self._virtual_pilot_equilibrium(state, reference, active)
        if self.collective_mode == "manual":
            desired_thrust = self.plant.thrust_from_upper_pwm(
                reference.collective_command[:, 0].clamp(0.0, 1.0)
            )
        else:
            height = -state.position_n[:, 2]
            height_target = -reference.target_position_n[:, 2]
            vertical_speed = -state.velocity_n[:, 2]
            vertical_speed_target = -reference.target_velocity_n[:, 2]
            error = height_target - height
            speed_error = vertical_speed_target - vertical_speed
            candidate_integral = (
                self.height_integral + error * self.context.control_dt
            ).clamp(-self.height_integral_limit, self.height_integral_limit)
            self.height_integral.copy_(
                torch.where(active, candidate_integral, self.height_integral)
            )
            acceleration_up = (
                self.height_kp * error
                + self.height_ki * self.height_integral
                + self.height_kd * speed_error
            )
            cosine = tilt_cosine(state.attitude_q_wb).clamp_min(0.35)
            desired_thrust = (
                self.context.parameters["body.mass"]
                * (self.plant.gravity + acceleration_up)
                / cosine
            )
            max_thrust = self.plant.thrust_from_upper_pwm(
                torch.ones_like(desired_thrust)
            )
            desired_thrust = desired_thrust.clamp_min(
                self.minimum_thrust_fraction * self.trim.thrust
            )
            desired_thrust = torch.minimum(
                desired_thrust,
                self.maximum_thrust_fraction * max_thrust,
            )
        target_speed, motor_pwm = self.plant.balanced_motor_target(desired_thrust)
        base = torch.cat(
            (
                motor_pwm,
                torch.zeros(
                    self.context.batch_size,
                    3,
                    device=self.context.device,
                    dtype=self.context.dtype,
                ),
            ),
            dim=1,
        )
        return desired_thrust, target_speed, base

    def _virtual_pilot_equilibrium(
        self,
        state: ControllerState,
        reference: ControllerReference,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Virtual pilot owns the upper rotor; the LQR owns the other four.

        Height truth is used only by the pilot; the LQR state never receives
        altitude.
        """
        height = -state.position_n[:, 2]
        vertical_speed = -state.velocity_n[:, 2]
        upper_pwm = self.pilot.step(height, vertical_speed, active)
        upper_pwm_1d = upper_pwm.squeeze(-1)
        desired_thrust = self.plant.thrust_from_upper_pwm(upper_pwm_1d)
        upper_target_speed = lookup(
            upper_pwm_1d,
            self.context.parameters["motors.pwm_to_rpm_table"][:, 0],
        )[:, None]
        # The pilot throttle is the coaxial collective: the lower rotor base
        # follows the upper so the whole T/W envelope can hover, and the LQR
        # trims the lower differential plus the three servos.
        target_speed = torch.cat((upper_target_speed, upper_target_speed), dim=1)
        base = torch.cat(
            (
                upper_pwm,
                upper_pwm,
                torch.zeros_like(upper_pwm).expand(-1, 3),
            ),
            dim=1,
        )
        return desired_thrust, target_speed, base

    def _errors(
        self,
        state: ControllerState,
        reference: ControllerReference,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attitude_error = quaternion_rotation_error(
            state.attitude_q_wb, reference.target_attitude_q_wb
        )
        # WebUI/self-stabilize uses yaw-rate mode; yaw angle remains diagnostic only.
        attitude_error = torch.cat(
            (attitude_error[:, :2], torch.zeros_like(attitude_error[:, 2:])),
            dim=1,
        )
        rate_error = (
            state.angular_velocity_b - reference.target_angular_velocity_b
        )
        return attitude_error, rate_error

    def _pid_command(
        self,
        state: ControllerState,
        reference: ControllerReference,
        base: torch.Tensor,
        attitude_error: torch.Tensor,
        rate_error: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        integral_error = torch.cat(
            (attitude_error[:, :2], rate_error[:, 2:3]), dim=1
        )
        candidate = (
            self.attitude_integral + integral_error * self.context.control_dt
        ).clamp(-self.attitude_integral_limit, self.attitude_integral_limit)
        self.attitude_integral.copy_(
            torch.where(active[:, None], candidate, self.attitude_integral)
        )
        desired_moment = -(
            self.attitude_kp * attitude_error
            + self.attitude_kd * rate_error
            + self.attitude_ki * self.attitude_integral
        )
        desired_moment = desired_moment.clamp(-self.max_moment, self.max_moment)
        wrench_delta = torch.cat(
            (torch.zeros_like(desired_moment[:, :1]), desired_moment), dim=1
        )
        command, achieved = self.allocator.allocate(base, wrench_delta)
        # Back-calculation keeps the integral from accumulating against saturation.
        moment_residual = desired_moment - achieved[:, 1:]
        safe_ki = self.attitude_ki.clamp_min(1e-6)
        corrected = (
            self.attitude_integral
            + 0.25 * moment_residual / safe_ki * self.context.control_dt
        ).clamp(-self.attitude_integral_limit, self.attitude_integral_limit)
        self.attitude_integral.copy_(
            torch.where(active[:, None], corrected, self.attitude_integral)
        )
        return command, {
            "controller.desired_moment_b": desired_moment,
            "controller.achieved_wrench_delta": achieved,
        }

    def _lqr_command(
        self,
        state: ControllerState,
        base: torch.Tensor,
        target_motor_speed: torch.Tensor,
        attitude_error: torch.Tensor,
        rate_error: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state_parts = (
            attitude_error[:, :2],
            rate_error,
            state.motor_speed - target_motor_speed,
            state.servo_angle,
        )
        previous_integral = None
        if self._lqr_integral_enabled:
            previous_integral = self.attitude_integral.clone()
            integral_error = torch.cat(
                (attitude_error[:, :2], rate_error[:, 2:3]), dim=1
            )
            candidate = (
                self.attitude_integral
                + integral_error * self.context.control_dt
            ).clamp(
                -self.attitude_integral_limit,
                self.attitude_integral_limit,
            )
            self.attitude_integral.copy_(
                torch.where(active[:, None], candidate, self.attitude_integral)
            )
            lqr_state = torch.cat((*state_parts, self.attitude_integral), dim=1)
        else:
            lqr_state = torch.cat(state_parts, dim=1)
        if self._lqr_gain is None:
            raise RuntimeError("LQR gain is unavailable for this controller")
        if self._lqr_gain.ndim == 2:
            delta = -(lqr_state @ self._lqr_gain.transpose(0, 1))
        else:
            delta = -torch.bmm(
                self._lqr_gain, lqr_state.unsqueeze(-1)
            ).squeeze(-1)
        if self.upper_external:
            lower = delta.new_tensor([0.0, -1.0, -1.0, -1.0])
            upper = delta.new_tensor([1.0, 1.0, 1.0, 1.0])
            unconstrained = base[:, 1:] + delta
            constrained = unconstrained.clamp(lower, upper)
            command = torch.cat((base[:, :1], constrained), dim=1)
            saturated = (
                (unconstrained < lower) | (unconstrained > upper)
            ).any(dim=1)
        else:
            lower = delta.new_tensor([0.0, 0.0, -1.0, -1.0, -1.0])
            upper = delta.new_tensor([1.0, 1.0, 1.0, 1.0, 1.0])
            unconstrained_command = base + delta
            command = unconstrained_command.clamp(lower, upper)
            saturated = (
                (unconstrained_command < lower)
                | (unconstrained_command > upper)
            ).any(dim=1)
        if self._lqr_integral_enabled:
            if previous_integral is None:
                raise RuntimeError("LQI integral snapshot is unavailable")
            self.attitude_integral.copy_(
                torch.where(
                    (active & saturated)[:, None],
                    previous_integral,
                    self.attitude_integral,
                )
            )
        return command, {
            "controller.lqr_state": lqr_state,
            "controller.lqr_delta_command": delta,
            "controller.lqr_saturated": saturated,
        }

    def _common_step(
        self,
        state: ControllerState,
        reference: ControllerReference,
        active_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        active = self._active_mask(active_mask)
        thrust, target_speed, base = self._desired_actuator_equilibrium(
            state, reference, active
        )
        attitude_error, rate_error = self._errors(state, reference)
        return active, thrust, target_speed, base, attitude_error, rate_error

    def _finish(
        self,
        command: torch.Tensor,
        active: torch.Tensor,
        diagnostics: dict[str, torch.Tensor],
    ) -> ControllerOutput:
        command = torch.where(active[:, None], command, self.last_command)
        self.last_command.copy_(command)
        return ControllerOutput.create(command, diagnostics)

    def _design_lqr(
        self,
        config: Mapping[str, Any],
    ) -> tuple[torch.Tensor, np.ndarray]:
        # Use the first instance as the nominal scheduled model. Per-instance
        # gain scheduling can be added without changing the controller interface.
        p = {name: value[:1].to(torch.float64) for name, value in self.context.parameters.items()}
        plant = LocalPlantModel(p)
        trim = plant.hover_trim()
        inertia = p["body.inertia_diagonal_b"][0]
        x0 = torch.cat(
            (
                torch.zeros(5, dtype=torch.float64, device=self.context.device),
                trim.motor_speed[0],
                torch.zeros(3, dtype=torch.float64, device=self.context.device),
            )
        )
        input_count = 4 if self.upper_external else 5
        u0 = trim.command[0][1:] if self.upper_external else trim.command[0]

        def continuous(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            rates = x[2:5]
            motor_speed = x[5:7]
            servo_angle = x[7:10]
            moment = plant.wrench_from_actuators(
                motor_speed[None], servo_angle[None]
            )[0, 1:]
            angular_acceleration = moment / inertia
            if self.upper_external:
                motor_target = lookup(
                    torch.cat((trim.command[0, :1], u[:1]))[None],
                    p["motors.pwm_to_rpm_table"],
                )[0]
                servo_target = lookup(
                    u[1:][None], p["servos.pwm_angle_table"]
                )[0]
            else:
                motor_target = lookup(
                    u[:2][None], p["motors.pwm_to_rpm_table"]
                )[0]
                servo_target = lookup(
                    u[2:][None], p["servos.pwm_angle_table"]
                )[0]
            motor_dot = (
                motor_target - motor_speed
            ) / p["motors.time_constant"][0]
            servo_dot = (servo_target - servo_angle) / p["servos.tau"][0]
            return torch.cat((rates[:2], angular_acceleration, motor_dot, servo_dot))

        with torch.enable_grad():
            a = torch.autograd.functional.jacobian(
                lambda value: continuous(value, u0), x0
            ).detach().cpu().numpy()
            b = torch.autograd.functional.jacobian(
                lambda value: continuous(x0, value), u0
            ).detach().cpu().numpy()
        integral_scales_raw = config.get("integral_state_scales")
        if integral_scales_raw is not None:
            integral_scales = np.asarray(
                _vector(
                    config,
                    "integral_state_scales",
                    (0.20, 0.20, 0.50),
                ),
                dtype=np.float64,
            )
            if np.any(integral_scales <= 0.0):
                raise ValueError("lqr integral_state_scales must be positive")
            integral_output = np.zeros((3, 10), dtype=np.float64)
            integral_output[0, 0] = 1.0
            integral_output[1, 1] = 1.0
            integral_output[2, 4] = 1.0
            a = np.block(
                [
                    [a, np.zeros((10, 3), dtype=np.float64)],
                    [integral_output, np.zeros((3, 3), dtype=np.float64)],
                ]
            )
            b = np.vstack(
                (b, np.zeros((3, input_count), dtype=np.float64))
            )
            self._lqr_integral_enabled = True
            self._lqr_state_size = 13
        else:
            integral_scales = np.asarray([], dtype=np.float64)
        augmented = np.block(
            [
                [a, b],
                [
                    np.zeros(
                        (input_count, a.shape[0] + input_count),
                        dtype=np.float64,
                    )
                ],
            ]
        )
        discrete = expm(augmented * self.context.control_dt)
        state_count = a.shape[0]
        ad = discrete[:state_count, :state_count]
        bd = discrete[:state_count, state_count:]
        state_scales = np.asarray(
            _vector(
                config,
                "state_scales",
                (
                    math.radians(10),
                    math.radians(10),
                    2.0,
                    2.0,
                    1.5,
                    250.0,
                    250.0,
                    0.15,
                    0.15,
                    0.15,
                ),
            ),
            dtype=np.float64,
        )
        if self._lqr_integral_enabled:
            state_scales = np.concatenate((state_scales, integral_scales))
        default_input_scales = (
            (0.18, 0.45, 0.45, 0.45)
            if self.upper_external
            else (0.18, 0.18, 0.45, 0.45, 0.45)
        )
        input_scales = np.asarray(
            _vector(config, "input_scales", default_input_scales),
            dtype=np.float64,
        )
        q = np.diag(1.0 / state_scales**2)
        r = (
            _number(config, "input_weight_scale", 0.25)
            * np.diag(1.0 / input_scales**2)
        )
        solution = solve_discrete_are(ad, bd, q, r)
        gain = np.linalg.solve(r + bd.T @ solution @ bd, bd.T @ solution @ ad)
        poles = np.linalg.eigvals(ad - bd @ gain)
        return (
            torch.tensor(
                gain,
                device=self.context.device,
                dtype=self.context.dtype,
            ),
            poles,
        )


class PIDController(ClassicalControllerBase):
    controller_type = "pid"

    def step(
        self,
        state: ControllerState,
        reference: ControllerReference,
        active_mask: torch.Tensor | None = None,
    ) -> ControllerOutput:
        active, thrust, target_speed, base, attitude_error, rate_error = self._common_step(
            state, reference, active_mask
        )
        command, diagnostics = self._pid_command(
            state, reference, base, attitude_error, rate_error, active
        )
        diagnostics.update(
            {
                "controller.mode_code": torch.zeros_like(thrust),
                "controller.lqr_blend": torch.zeros_like(thrust),
                "controller.desired_thrust": thrust,
                "controller.target_motor_speed": target_speed,
                "controller.attitude_error": attitude_error,
                "controller.rate_error": rate_error,
                "controller.command": command,
            }
        )
        return self._finish(command, active, diagnostics)


class LQRController(ClassicalControllerBase):
    controller_type = "lqr"

    def step(
        self,
        state: ControllerState,
        reference: ControllerReference,
        active_mask: torch.Tensor | None = None,
    ) -> ControllerOutput:
        active, thrust, target_speed, base, attitude_error, rate_error = self._common_step(
            state, reference, active_mask
        )
        command, diagnostics = self._lqr_command(
            state, base, target_speed, attitude_error, rate_error, active
        )
        diagnostics.update(
            {
                "controller.mode_code": torch.ones_like(thrust),
                "controller.lqr_blend": torch.ones_like(thrust),
                "controller.desired_thrust": thrust,
                "controller.target_motor_speed": target_speed,
                "controller.attitude_error": attitude_error,
                "controller.rate_error": rate_error,
                "controller.command": command,
            }
        )
        return self._finish(command, active, diagnostics)


class HybridPIDLQRController(ClassicalControllerBase):
    controller_type = "hybrid_pid_lqr"

    def __init__(
        self,
        context: ControllerContext,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(context, config)
        hybrid = _node(self.config, "hybrid")
        self.enter_speed_fraction = _number(
            hybrid, "lqr_enter_motor_speed_fraction", 0.78
        )
        self.exit_speed_fraction = _number(
            hybrid, "lqr_exit_motor_speed_fraction", 0.65
        )
        self.enter_attitude_error = _number(
            hybrid, "lqr_enter_attitude_error_rad", math.radians(12)
        )
        self.exit_attitude_error = _number(
            hybrid, "lqr_exit_attitude_error_rad", math.radians(20)
        )
        self.enter_rate = _number(hybrid, "lqr_enter_rate_rad_s", 2.0)
        self.exit_rate = _number(hybrid, "lqr_exit_rate_rad_s", 3.5)
        transition_s = _number(hybrid, "transition_s", 0.25)
        self.blend_step = self.context.control_dt / max(transition_s, self.context.control_dt)
        self.lqr_blend = torch.zeros(
            context.batch_size, device=context.device, dtype=context.dtype
        )
        self._lqr_enabled = torch.zeros(
            context.batch_size, device=context.device, dtype=torch.bool
        )

    def reset(self, reset_mask: torch.Tensor) -> None:
        super().reset(reset_mask)
        mask = self._active_mask(reset_mask)
        self.lqr_blend.masked_fill_(mask, 0.0)
        self._lqr_enabled.masked_fill_(mask, False)

    def step(
        self,
        state: ControllerState,
        reference: ControllerReference,
        active_mask: torch.Tensor | None = None,
    ) -> ControllerOutput:
        active, thrust, target_speed, base, attitude_error, rate_error = self._common_step(
            state, reference, active_mask
        )
        pid_command, pid_diagnostics = self._pid_command(
            state, reference, base, attitude_error, rate_error, active
        )
        lqr_command, lqr_diagnostics = self._lqr_command(
            state, base, target_speed, attitude_error, rate_error, active
        )
        speed_fraction = (
            state.motor_speed / target_speed.clamp_min(1.0)
        ).amin(dim=1)
        attitude_norm = torch.linalg.vector_norm(attitude_error[:, :2], dim=1)
        rate_norm = torch.linalg.vector_norm(rate_error, dim=1)
        enter = (
            (speed_fraction >= self.enter_speed_fraction)
            & (attitude_norm <= self.enter_attitude_error)
            & (rate_norm <= self.enter_rate)
        )
        leave = (
            (speed_fraction < self.exit_speed_fraction)
            | (attitude_norm > self.exit_attitude_error)
            | (rate_norm > self.exit_rate)
        )
        enabled = torch.where(leave, False, torch.where(enter, True, self._lqr_enabled))
        self._lqr_enabled.copy_(torch.where(active, enabled, self._lqr_enabled))
        direction = torch.where(
            self._lqr_enabled,
            torch.ones_like(self.lqr_blend),
            -torch.ones_like(self.lqr_blend),
        )
        blend = (self.lqr_blend + direction * self.blend_step).clamp(0.0, 1.0)
        self.lqr_blend.copy_(torch.where(active, blend, self.lqr_blend))
        command = torch.lerp(pid_command, lqr_command, self.lqr_blend[:, None])
        diagnostics = {
            **pid_diagnostics,
            **lqr_diagnostics,
            "controller.mode_code": 2.0 + self.lqr_blend,
            "controller.lqr_blend": self.lqr_blend.clone(),
            "controller.desired_thrust": thrust,
            "controller.target_motor_speed": target_speed,
            "controller.motor_speed_fraction": speed_fraction,
            "controller.attitude_error": attitude_error,
            "controller.rate_error": rate_error,
            "controller.command": command,
        }
        return self._finish(command, active, diagnostics)
