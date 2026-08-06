"""Plan A deployment-design experiment: external upper collective + 4-output LQI.

Contract
--------
This module designs and audits a deployable variant of the Identification LQI
under the PX4 ``residual_4``-style ownership boundary:

* Channel 0 (upper rotor) is owned by an external pilot. In simulation the
  pilot is a batched height PID using truth height/vertical speed (the same
  semantics as the WebUI shared altitude PID and the PX4 pilot/hover-observer
  boundary). The LQI never writes the upper channel.
* The LQI owns the lower rotor and the three servos only (4 outputs) and
  applies a 4-row gain around the nominal hover trim.
* The 13-state observer keeps the upper/lower/servo states command-driven
  (no ESC telemetry or servo-angle feedback), matching the identification
  gain-synthesis model in ``control_evaluation._discrete_model``.

The module is fully isolated: it does not register new controller types in the
shared factory, does not modify SimEnv, and adds its own CLI, config and report.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from .config import IdentificationExperimentConfig, load_experiment_config
from .control_evaluation import _discrete_model, _lqr_gain, _nominal_actuator_model
from .experiment import (
    CommandDrivenServoObserver,
    _apply_effectiveness_labels,
    _expand_mapping,
    _sample_parameter_labels,
    _set_initial_observation_state,
    _sim2real_lqr_targets,
    _yaw_free_attitude,
)
from .gain_training import _lqr_weights
from .repeated_trial_evaluation import _nonlinear_summary, _sample_near_equilibrium
from .sim2real_control_evaluation import (
    _cluster_bootstrap_mean_ci,
    _model_from_target,
    _paired_binary_summary,
)


def _lqr_weights_external(
    controller_config: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Q, R5, R4) where R4 drops the externally-owned upper channel."""

    q, r5 = _lqr_weights(controller_config)
    r4 = r5[1:, 1:]
    return q, r5, r4


def synthesize_external_gain(
    simulator_config: Path,
    controller_config: Path,
) -> dict[str, Any]:
    """Synthesize the 5-input and 4-input nominal LQI gains for the audit."""

    from flight_controller.plant import LocalPlantModel
    from simenv.config import load_and_materialize

    materialized = load_and_materialize(
        simulator_config, 1, torch.device("cpu"), torch.float64
    )
    trim = LocalPlantModel(materialized.parameters).hover_trim()
    trim_command = trim.command[0]
    effectiveness, command_slopes, tau = _nominal_actuator_model(
        simulator_config
    )
    a, b = _discrete_model(
        effectiveness, command_slopes, tau, 1.0 / 500.0, integral=True
    )
    q, r5, r4 = _lqr_weights_external(controller_config)
    gain_5 = _lqr_gain(a, b, q, r5)
    b_ls = b[:, 1:]
    solution = solve_discrete_are(a, b_ls, q, r4)
    gain_4 = np.linalg.solve(
        r4 + b_ls.T @ solution @ b_ls, b_ls.T @ solution @ a
    )
    gain_4_dropped = gain_5[1:, :]
    return {
        "a": a,
        "b": b,
        "b_lower_servo": b_ls,
        "effectiveness": effectiveness,
        "command_slopes": command_slopes,
        "time_constants": tau,
        "q": q,
        "r5": r5,
        "r4": r4,
        "gain_5": gain_5,
        "gain_4": gain_4,
        "gain_4_dropped": gain_4_dropped,
        "trim_upper_pwm": float(trim_command[0].item()),
        "trim_lower_pwm": float(trim_command[1].item()),
        "trim_motor_speed_rad_s": trim.motor_speed[0].detach().numpy().copy(),
        "pole_radius_5": float(np.max(np.abs(np.linalg.eigvals(a - b @ gain_5)))),
        "pole_radius_4": float(
            np.max(np.abs(np.linalg.eigvals(a - b_ls @ gain_4)))
        ),
        "pole_radius_4_dropped": float(
            np.max(np.abs(np.linalg.eigvals(a - b_ls @ gain_4_dropped)))
        ),
    }


def _oracle_external_gains(
    targets: np.ndarray,
    nominal_servo_slopes: np.ndarray,
    q: np.ndarray,
    r4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-airframe 4-input oracle gains from true LQR-effective labels."""

    gains = []
    valid = np.ones(len(targets), dtype=bool)
    for index, target in enumerate(targets):
        try:
            a, b, _ = _model_from_target(target, nominal_servo_slopes)
            b_ls = b[:, 1:]
            solution = solve_discrete_are(a, b_ls, q, r4)
            gains.append(
                np.linalg.solve(
                    r4 + b_ls.T @ solution @ b_ls,
                    b_ls.T @ solution @ a,
                )
            )
        except (np.linalg.LinAlgError, ValueError):
            valid[index] = False
            gains.append(np.zeros((4, 13), dtype=np.float64))
    return np.stack(gains), valid


class ExternalHeightPilot:
    """Batched external collective pilot: a height PID acting as the pilot.

    It owns only the upper-rotor channel. The thrust model is the nominal
    SimEnv model, so the pilot is systematically wrong on randomized airframes
    exactly like a human/observer acting on the nominal calibration.
    """

    def __init__(
        self,
        context: Any,
        controller_config_path: Path,
    ) -> None:
        from flight_controller import load_controller_config
        from flight_controller.plant import LocalPlantModel

        config = load_controller_config(controller_config_path)["params"]
        altitude = config["pid"]["altitude"]
        self.kp = float(altitude["kp"])
        self.ki = float(altitude["ki"])
        self.kd = float(altitude["kd"])
        self.integral_limit = float(altitude["integral_limit"])
        self.minimum_thrust_fraction = float(altitude["minimum_thrust_fraction"])
        self.maximum_thrust_fraction = float(altitude["maximum_thrust_fraction"])
        self.dt = context.control_dt
        self.plant = LocalPlantModel(context.parameters)
        self.trim = self.plant.hover_trim()
        self.integral = torch.zeros(
            context.batch_size,
            device=context.device,
            dtype=context.dtype,
        )

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.integral.masked_fill_(reset_mask, 0.0)

    def step(
        self,
        state: Any,
        reference: Any,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from flight_controller.math import tilt_cosine

        height = -state.position_n[:, 2]
        height_target = -reference.target_position_n[:, 2]
        vertical_speed = -state.velocity_n[:, 2]
        vertical_speed_target = -reference.target_velocity_n[:, 2]
        error = height_target - height
        speed_error = vertical_speed_target - vertical_speed
        candidate = (self.integral + error * self.dt).clamp(
            -self.integral_limit, self.integral_limit
        )
        self.integral.copy_(
            torch.where(active, candidate, self.integral)
        )
        acceleration_up = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * speed_error
        )
        cosine = tilt_cosine(state.attitude_q_wb).clamp_min(0.35)
        desired_thrust = (
            self.plant.p["body.mass"]
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
        _, pwm = self.plant.balanced_motor_target(desired_thrust)
        upper = pwm[:, :1].clamp(0.0, 1.0)
        return upper, {
            "pilot.desired_thrust": desired_thrust,
            "pilot.height_error": error,
            "pilot.height_integral": self.integral,
        }


class ExternalCollectiveLQIController:
    """4-output LQI around nominal trim with an externally-owned upper rotor.

    State (13): [roll_error, pitch_error, p, q, r, upper_filter,
    lower_filter, servo_1_angle, servo_2_angle, servo_3_angle,
    integral_roll, integral_pitch, integral_yaw_rate].

    All five actuator states are command-driven observers. The upper filter
    state is driven by the external pilot command; the lower and servo states
    are driven by the LQI's own final clamped commands. No ESC telemetry or
    servo-angle feedback is used.
    """

    def __init__(
        self,
        context: Any,
        controller_config_path: Path,
        simulator_config: Path,
        gain: np.ndarray,
        observer_mode: str,
    ) -> None:
        from flight_controller import load_controller_config
        from flight_controller.plant import LocalPlantModel

        if gain.shape not in {(4, 13), (context.batch_size, 4, 13)}:
            raise ValueError(
                "external LQI gain must be (4, 13) or "
                f"({context.batch_size}, 4, 13), got {gain.shape}"
            )
        if observer_mode not in {"linear", "nonlinear"}:
            raise ValueError("observer_mode must be linear or nonlinear")
        config = load_controller_config(controller_config_path)["params"]
        pid = config["pid"]["attitude"]
        self.control_dt = context.control_dt
        self.plant = LocalPlantModel(context.parameters)
        self.trim = self.plant.hover_trim()
        self.upper_trim = float(self.trim.command[0, 0].item())
        self.lower_trim = float(self.trim.command[0, 1].item())
        self.integral_limit = torch.as_tensor(
            pid["integral_limit"],
            device=context.device,
            dtype=context.dtype,
        )
        self.gain = torch.as_tensor(
            gain, device=context.device, dtype=context.dtype
        )
        _, command_slopes, time_constants = _nominal_actuator_model(
            simulator_config
        )
        self.slopes = torch.as_tensor(
            command_slopes, device=context.device, dtype=context.dtype
        )
        self.decay = torch.as_tensor(
            np.exp(-self.control_dt / time_constants),
            device=context.device,
            dtype=context.dtype,
        )
        self.observer_mode = observer_mode
        self.actuator_state = torch.zeros(
            context.batch_size,
            5,
            device=context.device,
            dtype=context.dtype,
        )
        self.integral = torch.zeros(
            context.batch_size,
            3,
            device=context.device,
            dtype=context.dtype,
        )
        self.last_command = torch.zeros(
            context.batch_size,
            4,
            device=context.device,
            dtype=context.dtype,
        )
        self.servo_observer = self._new_servo_observer()

    def _new_servo_observer(self) -> CommandDrivenServoObserver | None:
        if self.observer_mode != "nonlinear":
            return None
        p = self.plant.p
        return CommandDrivenServoObserver(
            p["servos.pwm_angle_table"],
            p["servos.tau"],
            p["servos.max_speed"],
            p["servos.backlash"],
            p["servos.deadzone"],
            self.control_dt,
        )

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.actuator_state.masked_fill_(reset_mask[:, None], 0.0)
        self.integral.masked_fill_(reset_mask[:, None], 0.0)
        self.last_command.masked_fill_(reset_mask[:, None], 0.0)
        if self.servo_observer is not None:
            self.servo_observer = self._new_servo_observer()

    def step(
        self,
        state: Any,
        reference: Any,
        upper_pwm: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        from flight_controller.math import quaternion_rotation_error

        attitude_error = quaternion_rotation_error(
            state.attitude_q_wb, reference.target_attitude_q_wb
        )
        attitude_error = torch.cat(
            (attitude_error[:, :2], torch.zeros_like(attitude_error[:, 2:])),
            dim=1,
        )
        rate_error = (
            state.angular_velocity_b - reference.target_angular_velocity_b
        )
        servo_state = (
            self.servo_observer.angle
            if self.servo_observer is not None
            else self.actuator_state[:, 2:]
        )
        lqr_state = torch.cat(
            (
                attitude_error[:, :2],
                rate_error,
                self.actuator_state[:, :2],
                servo_state,
                self.integral,
            ),
            dim=1,
        )
        if self.gain.ndim == 2:
            delta = -(lqr_state @ self.gain.transpose(0, 1))
        else:
            delta = -torch.bmm(
                self.gain, lqr_state.unsqueeze(-1)
            ).squeeze(-1)

        lower_unconstrained = self.lower_trim + delta[:, 0:1]
        servo_unconstrained = delta[:, 1:]
        lower = lower_unconstrained.clamp(0.0, 1.0)
        servos = servo_unconstrained.clamp(-1.0, 1.0)
        saturated = (
            (lower_unconstrained < 0.0).any(dim=1)
            | (lower_unconstrained > 1.0).any(dim=1)
            | (servo_unconstrained < -1.0).any(dim=1)
            | (servo_unconstrained > 1.0).any(dim=1)
        )

        integral_error = torch.cat(
            (attitude_error[:, :2], rate_error[:, 2:3]), dim=1
        )
        previous_integral = self.integral.clone()
        candidate = (
            self.integral + integral_error * self.control_dt
        ).clamp(-self.integral_limit, self.integral_limit)
        self.integral.copy_(
            torch.where(
                (active & ~saturated)[:, None],
                candidate,
                self.integral,
            )
        )
        self.integral.copy_(
            torch.where(
                (active & saturated)[:, None],
                previous_integral,
                self.integral,
            )
        )

        upper_clamped = upper_pwm.clamp(0.0, 1.0)
        upper_dev = upper_clamped - self.upper_trim
        lower_dev = lower - self.lower_trim
        command = torch.cat((lower, servos), dim=1)
        if self.servo_observer is not None:
            self.servo_observer.advance(servos)
        command_dev = torch.cat((upper_dev, lower_dev, servos), dim=1)
        self.actuator_state.copy_(
            self.decay[None] * self.actuator_state
            + self.slopes[None] * (1.0 - self.decay[None]) * command_dev
        )
        command = torch.where(
            active[:, None], command, self.last_command
        )
        self.last_command.copy_(command)
        return command


@torch.no_grad()
def _simulate_external4(
    experiment: IdentificationExperimentConfig,
    physical_labels: torch.Tensor,
    attitude: torch.Tensor,
    angular_velocity: torch.Tensor,
    gain: np.ndarray,
    observer_mode: str,
    duration_s: float,
) -> dict[str, torch.Tensor]:
    from flight_controller import (
        ControllerContext,
        ControllerReference,
        ControllerState,
    )
    from flight_controller.math import quaternion_rotation_error, tilt_cosine
    from flight_controller.plant import LocalPlantModel
    from simenv import SimulationEnvironment
    from simenv.config import load_and_materialize

    device = torch.device(experiment.device)
    dtype = torch.float32 if experiment.dtype == "float32" else torch.float64
    count = len(physical_labels)
    nominal_materialized = load_and_materialize(
        experiment.simulator_config, 1, device, dtype
    )
    nominal_parameters = _expand_mapping(
        nominal_materialized.parameters, count
    )
    actual_parameters = _apply_effectiveness_labels(
        nominal_parameters,
        physical_labels.to(device),
        experiment.parameterization,
    )
    initial_state = _expand_mapping(
        nominal_materialized.initial_state, count
    )
    initial_state["attitude_q_wb"].copy_(attitude.to(device))
    initial_state["angular_velocity_b"].copy_(angular_velocity.to(device))
    materialized = replace(
        nominal_materialized,
        parameters=actual_parameters,
        initial_state=initial_state,
        sensor_state=_expand_mapping(
            nominal_materialized.sensor_state, count
        ),
    )
    environment = SimulationEnvironment(
        materialized, count, device, dtype, logging_enabled=False
    )
    try:
        context = ControllerContext(
            batch_size=count,
            device=device,
            dtype=dtype,
            control_dt=1.0 / experiment.control_hz,
            parameters=nominal_parameters,
        )
        controller = ExternalCollectiveLQIController(
            context,
            experiment.controller_config,
            experiment.simulator_config,
            gain,
            observer_mode,
        )
        pilot = ExternalHeightPilot(context, experiment.controller_config)
        _set_initial_observation_state(
            environment,
            LocalPlantModel(actual_parameters).hover_trim().motor_speed,
            initial_state["angular_velocity_b"],
        )
        zeros = torch.zeros(count, 3, device=device, dtype=dtype)
        identity = torch.zeros(count, 4, device=device, dtype=dtype)
        identity[:, 0] = 1.0
        reference = ControllerReference(
            target_position_n=zeros,
            target_velocity_n=zeros.clone(),
            target_attitude_q_wb=identity,
            target_angular_velocity_b=zeros.clone(),
            collective_command=torch.zeros(count, 1, device=device, dtype=dtype),
        )
        active = torch.ones(count, device=device, dtype=torch.bool)
        settled_count = torch.zeros(count, device=device, dtype=torch.int64)
        saturation_steps = torch.zeros_like(settled_count)
        saturation_steps_lqi = torch.zeros_like(settled_count)
        final_attitude_error = torch.full(
            (count,), torch.inf, device=device, dtype=dtype
        )
        final_rate = torch.full_like(final_attitude_error, torch.inf)
        total_steps = round(duration_s * experiment.control_hz)
        for _ in range(total_steps):
            truth = environment.observe("truth").values
            if experiment.measurement_mode == "simulated_sensors":
                sensors = environment.observe("sensor").values
                measured_rate = sensors["gyro"]
                measured_motor = sensors["motor_speed"]
            else:
                measured_rate = truth["angular_velocity_b"]
                measured_motor = truth["motor_speed"]
            tilt_attitude = _yaw_free_attitude(truth["attitude_q_wb"])
            tilt = torch.acos(tilt_cosine(truth["attitude_q_wb"]))
            rate_norm = truth["angular_velocity_b"].norm(dim=1)
            finite = (
                torch.isfinite(tilt_attitude).all(dim=1)
                & torch.isfinite(measured_rate).all(dim=1)
                & torch.isfinite(measured_motor).all(dim=1)
            )
            active &= (
                finite
                & (tilt <= experiment.convergence.safety_tilt_rad)
                & (rate_norm <= experiment.convergence.safety_angular_rate_rad_s)
            )
            state = ControllerState(
                position_n=torch.nan_to_num(truth["position_n"]),
                velocity_n=torch.nan_to_num(truth["velocity_n"]),
                attitude_q_wb=torch.nan_to_num(tilt_attitude),
                angular_velocity_b=torch.nan_to_num(measured_rate),
                linear_acceleration_n=torch.nan_to_num(
                    truth["linear_acceleration_n"]
                ),
                motor_speed=torch.nan_to_num(measured_motor),
                servo_angle=torch.zeros(count, 3, device=device, dtype=dtype),
            )
            upper, _ = pilot.step(state, reference, active)
            output = controller.step(state, reference, upper, active)
            command = torch.cat((upper.clamp(0.0, 1.0), output), dim=1)
            attitude_error = quaternion_rotation_error(
                torch.nan_to_num(tilt_attitude), identity
            )[:, :2].norm(dim=1)
            settled = active & (
                attitude_error
                <= experiment.convergence.maximum_roll_pitch_error_rad
            ) & (
                rate_norm <= experiment.convergence.maximum_angular_rate_rad_s
            )
            settled_count = torch.where(
                settled, settled_count + 1, torch.zeros_like(settled_count)
            )
            final_attitude_error = torch.where(
                active, attitude_error, final_attitude_error
            )
            final_rate = torch.where(active, rate_norm, final_rate)
            saturated = active & (
                (command[:, :2] <= 1e-6).any(dim=1)
                | (command[:, :2] >= 1.0 - 1e-6).any(dim=1)
                | (command[:, 2:].abs() >= 1.0 - 1e-6).any(dim=1)
            )
            saturated_lqi = active & (
                (command[:, 1:2] <= 1e-6).any(dim=1)
                | (command[:, 1:2] >= 1.0 - 1e-6).any(dim=1)
                | (command[:, 2:].abs() >= 1.0 - 1e-6).any(dim=1)
            )
            saturation_steps += saturated.to(torch.int64)
            saturation_steps_lqi += saturated_lqi.to(torch.int64)
            result = environment.advance(command, active)
            active &= result.valid
        hold_steps = round(
            experiment.convergence.hold_s * experiment.control_hz
        )
        return {
            "safe": active.cpu(),
            "converged": (active & (settled_count >= hold_steps)).cpu(),
            "final_attitude_error": final_attitude_error.cpu(),
            "final_rate": final_rate.cpu(),
            "saturation_fraction": (
                saturation_steps.to(dtype) / total_steps
            ).cpu(),
            "saturation_fraction_lqi": (
                saturation_steps_lqi.to(dtype) / total_steps
            ).cpu(),
        }
    finally:
        environment.close()


def _load_test_observations(
    group_count: int,
    repeats: int,
    seed: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return _sample_near_equilibrium(
        group_count * repeats,
        math.radians(15.0),
        1.0,
        generator,
        dtype,
    )


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    experiment = load_experiment_config(args.experiment_config)
    experiment = replace(experiment, device=args.device)
    synth = synthesize_external_gain(
        experiment.simulator_config, experiment.controller_config
    )
    q, r5, r4 = _lqr_weights_external(experiment.controller_config)

    # Sample fresh feasible airframes from the same conditional distribution
    # used by the repeated8 dataset (unbiased common ground; no dependence on
    # trajectories generated under the old 5-output controller).
    dtype = torch.float32 if experiment.dtype == "float32" else torch.float64
    from simenv.config import load_and_materialize

    nominal_materialized = load_and_materialize(
        experiment.simulator_config, 1, torch.device("cpu"), dtype
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    labels, _ = _sample_parameter_labels(
        args.parameter_groups,
        nominal_materialized.parameters,
        experiment,
        generator,
        dtype,
    )
    parameters_cpu = _apply_effectiveness_labels(
        _expand_mapping(nominal_materialized.parameters, args.parameter_groups),
        labels,
        "sim2real_micro",
    )
    targets = _sim2real_lqr_targets(parameters_cpu).numpy()
    oracle_gain_4, oracle_valid = _oracle_external_gains(
        targets,
        synth["command_slopes"][2:],
        q,
        r4,
    )
    oracle_gain_4[~oracle_valid] = synth["gain_4"]

    repeats = args.evaluation_initial_conditions
    physical = labels.repeat_interleave(repeats, dim=0)
    attitude, angular_velocity = _load_test_observations(
        args.parameter_groups,
        repeats,
        args.seed + 1,
        dtype,
    )

    gains: dict[str, np.ndarray] = {
        "nominal5": np.repeat(
            synth["gain_5"][None], args.parameter_groups * repeats, axis=0
        ),
        "external4_linear": np.repeat(
            synth["gain_4"][None], args.parameter_groups, axis=0
        ),
        "external4_nonlinear": np.repeat(
            synth["gain_4"][None], args.parameter_groups, axis=0
        ),
        "external4_dropped_nonlinear": np.repeat(
            synth["gain_4_dropped"][None], args.parameter_groups, axis=0
        ),
        "external4_oracle_nonlinear": oracle_gain_4,
    }
    variant_names = tuple(
        name.strip() for name in args.variants.split(",") if name.strip()
    )
    unknown_variants = set(variant_names) - set(gains)
    if not variant_names or "nominal5" not in variant_names or unknown_variants:
        raise ValueError(
            "variants must include nominal5 and be a subset of "
            f"{tuple(gains)}; unknown={sorted(unknown_variants)}"
        )

    variant_results: dict[str, dict[str, torch.Tensor]] = {}
    for name in variant_names:
        if name == "nominal5":
            variant_results[name] = _simulate_gain_nominal(
                experiment,
                physical,
                attitude,
                angular_velocity,
                gains[name],
                duration_s=args.duration_s,
            )
        else:
            observer_mode = (
                "linear"
                if name.endswith("_linear")
                else "nonlinear"
            )
            gain = np.repeat(gains[name], repeats, axis=0)
            variant_results[name] = _simulate_external4(
                experiment,
                physical,
                attitude,
                angular_velocity,
                gain,
                observer_mode,
                args.duration_s,
            )

    nonlinear = {
        name: _nonlinear_summary(result)
        for name, result in variant_results.items()
    }
    nominal_result = variant_results["nominal5"]
    paired_vs_nominal = {}
    for name, result in variant_results.items():
        if name == "nominal5":
            continue
        mutually_safe = result["safe"] & nominal_result["safe"]
        paired_vs_nominal[name] = {
            "safety": _paired_binary_summary(
                result["safe"],
                nominal_result["safe"],
                args.parameter_groups,
                repeats,
                args.seed + 1000,
            ),
            "convergence": _paired_binary_summary(
                result["converged"],
                nominal_result["converged"],
                args.parameter_groups,
                repeats,
                args.seed + 2000,
            ),
            "saturation_fraction_delta_lqi": (
                float(
                    (
                        result["saturation_fraction_lqi"]
                        - nominal_result["saturation_fraction"]
                    ).mean()
                )
                if "saturation_fraction_lqi" in result
                else None
            ),
            "saturation_fraction_delta_group_cluster_bootstrap_95_ci": (
                _cluster_bootstrap_mean_ci(
                    result["saturation_fraction_lqi"]
                    - nominal_result["saturation_fraction"],
                    args.parameter_groups,
                    repeats,
                    args.seed + 3000,
                )
                if "saturation_fraction_lqi" in result
                else None
            ),
        }

    local_radius = {
        "nominal5": synth["pole_radius_5"],
        "external4_linear": synth["pole_radius_4"],
        "external4_nonlinear": synth["pole_radius_4"],
        "external4_dropped_nonlinear": synth["pole_radius_4_dropped"],
    }
    oracle_radii = []
    for index in range(args.parameter_groups):
        if not oracle_valid[index]:
            oracle_radii.append(np.inf)
            continue
        a, b, _ = _model_from_target(
            targets[index], synth["command_slopes"][2:]
        )
        oracle_radii.append(
            float(
                np.max(
                    np.abs(
                        np.linalg.eigvals(
                            a - b[:, 1:] @ oracle_gain_4[index]
                        )
                    )
                )
            )
        )
    if "external4_oracle_nonlinear" in variant_names:
        local_radius["external4_oracle_nonlinear"] = {
            "stable_fraction": float(
                np.mean(np.asarray(oracle_radii) < 1.0)
            ),
            "pole_radius_p95": float(
                np.quantile(oracle_radii, 0.95)
            ),
        }

    report = {
        "schema_version": 1,
        "experiment": "external_collective_lqi_design_v1",
        "contract": {
            "upper_rotor_owner": "external_pilot_height_pid",
            "controller_outputs": "lower_motor_plus_three_servos",
            "actuator_normalization": {
                "motors": "[0,1]",
                "servos": "[-1,1]",
            },
            "frame": {
                "world": "NED",
                "body": "FRD",
                "quaternion": "Hamilton wxyz body-to-world",
            },
            "observer": {
                "motor_speed_feedback": False,
                "servo_angle_feedback": False,
                "upper_state_source": "external command filter",
                "lower_state_source": "command filter",
                "servo_state_source": (
                    "per-variant linear or nonlinear command filter"
                ),
            },
            "timing_hz": experiment.control_hz,
            "integral_freeze_on_saturation": True,
        },
        "deployment_decision": "design_only",
        "gain_updates_enabled": False,
        "parameter_groups": args.parameter_groups,
        "evaluation_initial_conditions_per_group": repeats,
        "duration_s": args.duration_s,
        "maximum_initial_tilt_rad": math.radians(15.0),
        "maximum_initial_rate_rad_s": 1.0,
        "measurement_mode": experiment.measurement_mode,
        "parameterization": experiment.parameterization,
        "nominal_pole_radius_5": synth["pole_radius_5"],
        "nominal_pole_radius_4": synth["pole_radius_4"],
        "nominal_pole_radius_4_dropped": synth["pole_radius_4_dropped"],
        "max_abs_gain_delta_4_vs_dropped": float(
            np.max(np.abs(synth["gain_4"] - synth["gain_4_dropped"]))
        ),
        "oracle_gain_synthesis_valid_fraction": float(
            oracle_valid.mean()
        ),
        "nonlinear": nonlinear,
        "paired_vs_nominal5": paired_vs_nominal,
        "local_linear": local_radius,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def _simulate_gain_nominal(
    experiment: IdentificationExperimentConfig,
    physical_labels: torch.Tensor,
    attitude: torch.Tensor,
    angular_velocity: torch.Tensor,
    gain: np.ndarray,
    duration_s: float,
) -> dict[str, torch.Tensor]:
    from .repeated_trial_evaluation import _simulate_gain

    from .control_evaluation import _nominal_actuator_model

    _, _, time_constants = _nominal_actuator_model(
        experiment.simulator_config
    )
    return _simulate_gain(
        experiment,
        physical_labels,
        attitude,
        angular_velocity,
        gain,
        np.repeat(time_constants[2:][None, :], gain.shape[0], axis=0),
        np.ones(gain.shape[:1], dtype=bool),
        duration_s,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Design audit for the deployment variant with an external upper "
            "collective and a 4-output LQI (lower motor + 3 servos)."
        )
    )
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parameter-groups", type=int, default=128)
    parser.add_argument("--evaluation-initial-conditions", type=int, default=8)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument(
        "--variants",
        default=(
            "nominal5,external4_linear,external4_nonlinear,"
            "external4_dropped_nonlinear,external4_oracle_nonlinear"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260805)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
