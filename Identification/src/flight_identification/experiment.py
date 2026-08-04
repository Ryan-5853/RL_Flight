from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from .config import IdentificationExperimentConfig


FEATURE_NAMES = (
    "attitude_q_tilt.w",
    "attitude_q_tilt.x",
    "attitude_q_tilt.y",
    "attitude_q_tilt.z",
    "angular_velocity_b.x",
    "angular_velocity_b.y",
    "angular_velocity_b.z",
    "motor_speed.upper",
    "motor_speed.lower",
    "command.motor_upper",
    "command.motor_lower",
    "command.servo_1",
    "command.servo_2",
    "command.servo_3",
)

LABEL_NAMES = (
    "log10.inertia_x_scale",
    "log10.inertia_y_scale",
    "log10.inertia_z_scale",
    "log10.collective_effectiveness_scale",
    "log10.motor_reaction_effectiveness_scale",
    "log10.motor_tau_upper_scale",
    "log10.motor_tau_lower_scale",
    "log10.grid_1_effectiveness_scale",
    "log10.grid_2_effectiveness_scale",
    "log10.grid_3_effectiveness_scale",
    "log10.servo_1_tau_scale",
    "log10.servo_2_tau_scale",
    "log10.servo_3_tau_scale",
)

EMPIRICAL_LABEL_NAMES = (
    "log10.inertia_x_scale",
    "log10.inertia_y_scale",
    "log10.inertia_z_scale",
    "log10.thrust_to_weight_scale",
    "log10.motor_reaction_effectiveness_scale",
    "log10.motor_tau_upper_scale",
    "log10.motor_tau_lower_scale",
    "log10.grid_1_effectiveness_scale",
    "log10.grid_2_effectiveness_scale",
    "log10.grid_3_effectiveness_scale",
    "log10.servo_1_tau_scale",
    "log10.servo_2_tau_scale",
    "log10.servo_3_tau_scale",
)

SIM2REAL_LABEL_NAMES = (
    "mass_kg",
    "inertia_x_kg_m2",
    "inertia_y_kg_m2",
    "inertia_z_kg_m2",
    "maximum_thrust_to_weight",
    "motor_reaction_common_scale",
    "motor_reaction_upper_lower_ratio",
    "motor_tau_upper_s",
    "motor_tau_lower_s",
    "direct_center_x_m",
    "direct_center_y_m",
    "direct_center_z_m",
    "direct_thrust_fraction",
    "grid_1_thrust_fraction",
    "grid_2_thrust_fraction",
    "grid_3_thrust_fraction",
    "grid_1_center_x_m",
    "grid_1_center_y_m",
    "grid_1_center_z_m",
    "grid_2_center_x_m",
    "grid_2_center_y_m",
    "grid_2_center_z_m",
    "grid_3_center_x_m",
    "grid_3_center_y_m",
    "grid_3_center_z_m",
    "grid_1_vector_gain",
    "grid_2_vector_gain",
    "grid_3_vector_gain",
    "servo_1_tau_s",
    "servo_2_tau_s",
    "servo_3_tau_s",
    "servo_1_max_speed_rad_s",
    "servo_2_max_speed_rad_s",
    "servo_3_max_speed_rad_s",
    "grid_coupling_attenuation",
)

SIM2REAL_TARGET_NAMES = tuple(
    f"angular_acceleration_effectiveness.{axis}.{actuator}"
    for axis in ("roll", "pitch", "yaw")
    for actuator in (
        "motor_upper",
        "motor_lower",
        "servo_1",
        "servo_2",
        "servo_3",
    )
) + (
    "trim_angular_acceleration_bias.roll",
    "trim_angular_acceleration_bias.pitch",
    "trim_angular_acceleration_bias.yaw",
    "log10.motor_tau_upper_s",
    "log10.motor_tau_lower_s",
    "log10.servo_tau_1_s",
    "log10.servo_tau_2_s",
    "log10.servo_tau_3_s",
    "log10.command_slope.motor_upper",
    "log10.command_slope.motor_lower",
    "log10.maximum_thrust_to_weight",
)

SIM2REAL_IDENTIFICATION_TARGET_NAMES = tuple(
    name
    for name in SIM2REAL_TARGET_NAMES
    if name
    not in {
        "angular_acceleration_effectiveness.pitch.servo_1",
        "trim_angular_acceleration_bias.yaw",
    }
)

FAILURE_NAMES = {
    0: "converged",
    1: "convergence_timeout",
    2: "tilt_limit",
    3: "angular_rate_limit",
    4: "simenv_invalid",
}


def _expand_mapping(
    values: Mapping[str, torch.Tensor], batch_size: int
) -> dict[str, torch.Tensor]:
    return {
        name: value[:1].expand(batch_size, *value.shape[1:]).clone()
        for name, value in values.items()
    }


def _sample_group_labels(
    group_count: int,
    nominal_inertia: torch.Tensor,
    scale_range: tuple[float, float],
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    low, high = scale_range
    labels = low + torch.rand(
        group_count, len(LABEL_NAMES), generator=generator, dtype=dtype
    ) * (high - low)
    inertia = nominal_inertia[None] * torch.pow(10.0, labels[:, :3])
    valid = 2.0 * inertia.amax(dim=1) <= inertia.sum(dim=1)
    attempts = 0
    while not bool(valid.all().item()):
        count = int((~valid).sum().item())
        labels[~valid, :3] = low + torch.rand(
            count, 3, generator=generator, dtype=dtype
        ) * (high - low)
        inertia = nominal_inertia[None] * torch.pow(10.0, labels[:, :3])
        valid = 2.0 * inertia.amax(dim=1) <= inertia.sum(dim=1)
        attempts += 1
        if attempts >= 1000:
            raise RuntimeError("could not sample physically valid principal inertias")
    return labels


def _maximum_thrust_to_weight(
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    torque = parameters["motors.torque_coefficient"]
    speed_ratio = torch.sqrt(
        torque[:, 0].clamp_min(1e-16) / torque[:, 1].clamp_min(1e-16)
    )
    maximum_speed = parameters["motors.pwm_to_rpm_table"][..., -1, 1]
    upper_speed = torch.minimum(
        maximum_speed[:, 0], maximum_speed[:, 1] / speed_ratio
    )
    lower_speed = speed_ratio * upper_speed
    k1, k2, k3 = parameters["aerodynamics.thrust_coefficients"].unbind(dim=1)
    maximum_thrust = (
        k1 * upper_speed.square()
        + k2 * lower_speed.square()
        + k3 * upper_speed * lower_speed
    )
    weight = parameters["body.mass"] * 9.80665
    return maximum_thrust / weight


def parameter_label_names(parameterization: str) -> tuple[str, ...]:
    if parameterization == "legacy_collective":
        return LABEL_NAMES
    if parameterization == "empirical_core":
        return EMPIRICAL_LABEL_NAMES
    if parameterization == "sim2real_micro":
        return SIM2REAL_LABEL_NAMES
    raise ValueError(f"unsupported parameterization: {parameterization}")


def _parameter_label_ranges(
    config: IdentificationExperimentConfig,
    nominal: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    low, high = config.log10_effectiveness_range
    label_names = parameter_label_names(config.parameterization)
    ranges = torch.tensor(
        [[low, high]] * len(label_names),
        device="cpu",
        dtype=torch.float64,
    )
    if config.parameterization == "legacy_collective":
        return ranges
    if config.empirical_ranges is None:
        raise ValueError("empirical_core requires empirical parameter ranges")
    ranges[0:3] = torch.log10(
        torch.tensor(config.empirical_ranges.inertia_scale)
    )
    nominal_twr = float(_maximum_thrust_to_weight(nominal).cpu()[0])
    ranges[3] = torch.log10(
        torch.tensor(config.empirical_ranges.thrust_to_weight)
        / nominal_twr
    )
    ranges[4] = torch.log10(
        torch.tensor(config.empirical_ranges.motor_reaction_scale)
    )
    motor_tau = nominal["motors.time_constant"][0].detach().cpu().to(torch.float64)
    for index in range(2):
        ranges[5 + index] = torch.log10(
            torch.tensor(config.empirical_ranges.motor_time_constant_s)
            / motor_tau[index]
        )
    servo_tau = nominal["servos.tau"][0].detach().cpu().to(torch.float64)
    for index in range(3):
        ranges[7 + index] = torch.log10(
            torch.tensor(config.empirical_ranges.grid_effectiveness_scale)
        )
        ranges[10 + index] = torch.log10(
            torch.tensor(config.empirical_ranges.servo_time_constant_s)
            / servo_tau[index]
        )
    return ranges


def _sample_parameter_labels(
    group_count: int,
    nominal: Mapping[str, torch.Tensor],
    config: IdentificationExperimentConfig,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.parameterization == "sim2real_micro":
        return _sample_sim2real_labels(
            group_count, nominal, config, generator, dtype
        )
    ranges = _parameter_label_ranges(config, nominal)
    if config.parameterization == "legacy_collective":
        labels = _sample_group_labels(
            group_count,
            nominal["body.inertia_diagonal_b"][0].detach().cpu(),
            config.log10_effectiveness_range,
            generator,
            dtype,
        )
        return labels, ranges

    ranges_for_dtype = ranges.to(dtype)
    labels = ranges_for_dtype[:, 0] + torch.rand(
        group_count,
        len(EMPIRICAL_LABEL_NAMES),
        generator=generator,
        dtype=dtype,
    ) * (ranges_for_dtype[:, 1] - ranges_for_dtype[:, 0])
    nominal_inertia = nominal["body.inertia_diagonal_b"][0].detach().cpu().to(dtype)
    inertia = nominal_inertia[None] * torch.pow(10.0, labels[:, :3])
    valid = 2.0 * inertia.amax(dim=1) <= inertia.sum(dim=1)
    attempts = 0
    while not bool(valid.all().item()):
        count = int((~valid).sum())
        labels[~valid, :3] = ranges_for_dtype[:3, 0] + torch.rand(
            count, 3, generator=generator, dtype=dtype
        ) * (ranges_for_dtype[:3, 1] - ranges_for_dtype[:3, 0])
        inertia = nominal_inertia[None] * torch.pow(10.0, labels[:, :3])
        valid = 2.0 * inertia.amax(dim=1) <= inertia.sum(dim=1)
        attempts += 1
        if attempts >= 1000:
            raise RuntimeError("could not sample physically valid principal inertias")
    return labels, ranges


def _sample_sim2real_labels(
    group_count: int,
    nominal: Mapping[str, torch.Tensor],
    config: IdentificationExperimentConfig,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    feasibility = config.sim2real_feasibility
    if feasibility is None:
        raise ValueError("sim2real_micro requires feasibility constraints")
    accepted: list[torch.Tensor] = []
    accepted_count = 0
    candidate_batches = 0
    while accepted_count < group_count:
        candidate_count = min(4096, max(256, 8 * (group_count - accepted_count)))
        candidates = _draw_sim2real_candidate_labels(
            candidate_count, config, generator, dtype
        )
        device = nominal["body.mass"].device
        expanded_nominal = {
            name: value.expand(candidate_count, *value.shape[1:]).clone()
            for name, value in nominal.items()
        }
        candidate_parameters = _apply_effectiveness_labels(
            expanded_nominal,
            candidates.to(device),
            "sim2real_micro",
        )
        targets = _sim2real_lqr_targets(candidate_parameters)
        servo_effectiveness = targets[:, :15].reshape(-1, 3, 5)[:, :, 2:]
        bias = targets[:, 15:18]
        trim_angle = -torch.linalg.lstsq(
            servo_effectiveness, bias.unsqueeze(-1)
        ).solution.squeeze(-1)
        residual = torch.bmm(
            servo_effectiveness, trim_angle.unsqueeze(-1)
        ).squeeze(-1) + bias
        valid = (
            torch.isfinite(targets).all(dim=1)
            & (trim_angle.abs().amax(dim=1)
               <= feasibility.maximum_linear_trim_servo_angle_rad)
            & (residual.norm(dim=1)
               <= feasibility.maximum_linear_trim_residual_rad_s2)
        )
        selected = candidates[valid.cpu()]
        accepted.append(selected[: group_count - accepted_count])
        accepted_count += min(len(selected), group_count - accepted_count)
        candidate_batches += 1
        if candidate_batches >= 1000:
            raise RuntimeError(
                "could not sample enough linearly trimmable sim2real vehicles"
            )
    labels = torch.cat(accepted, dim=0)
    label_ranges = torch.stack(
        (labels.amin(dim=0), labels.amax(dim=0)), dim=1
    ).to(torch.float64)
    return labels, label_ranges


def _draw_sim2real_candidate_labels(
    group_count: int,
    config: IdentificationExperimentConfig,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    ranges = config.sim2real_ranges
    if ranges is None:
        raise ValueError("sim2real_micro requires sim2real_ranges")

    def uniform(bounds: tuple[float, float], count: int = group_count) -> torch.Tensor:
        low, high = bounds
        return low + torch.rand(count, generator=generator, dtype=dtype) * (high - low)

    def log_uniform(
        bounds: tuple[float, float], count: int = group_count
    ) -> torch.Tensor:
        low, high = (math.log10(value) for value in bounds)
        return torch.pow(10.0, uniform((low, high), count))

    labels = torch.empty(
        group_count, len(SIM2REAL_LABEL_NAMES), dtype=dtype
    )
    labels[:, 0] = uniform(ranges.mass_kg)
    labels[:, 1] = labels[:, 0] * log_uniform(
        ranges.inertia_xy_radius_of_gyration_m
    ).square()
    labels[:, 2] = labels[:, 0] * log_uniform(
        ranges.inertia_xy_radius_of_gyration_m
    ).square()
    labels[:, 3] = labels[:, 0] * log_uniform(
        ranges.inertia_z_radius_of_gyration_m
    ).square()
    inertia = labels[:, 1:4]
    valid = 2.0 * inertia.amax(dim=1) <= inertia.sum(dim=1)
    attempts = 0
    while not bool(valid.all().item()):
        count = int((~valid).sum().item())
        mass = labels[~valid, 0]
        labels[~valid, 1] = mass * log_uniform(
            ranges.inertia_xy_radius_of_gyration_m, count
        ).square()
        labels[~valid, 2] = mass * log_uniform(
            ranges.inertia_xy_radius_of_gyration_m, count
        ).square()
        labels[~valid, 3] = mass * log_uniform(
            ranges.inertia_z_radius_of_gyration_m, count
        ).square()
        inertia = labels[:, 1:4]
        valid = 2.0 * inertia.amax(dim=1) <= inertia.sum(dim=1)
        attempts += 1
        if attempts >= 1000:
            raise RuntimeError("could not sample valid sim2real principal inertias")
    labels[:, 4] = uniform(ranges.thrust_to_weight)
    labels[:, 5] = log_uniform(ranges.motor_reaction_scale)
    labels[:, 6] = log_uniform(ranges.motor_reaction_ratio)
    labels[:, 7] = log_uniform(ranges.motor_time_constant_s)
    labels[:, 8] = log_uniform(ranges.motor_time_constant_s)

    radius_low, radius_high = ranges.direct_center_xy_radius_m
    radius = torch.sqrt(
        uniform((radius_low**2, radius_high**2))
    )
    angle = uniform((0.0, 2.0 * math.pi))
    labels[:, 9] = radius * torch.cos(angle)
    labels[:, 10] = radius * torch.sin(angle)
    labels[:, 11] = uniform(ranges.direct_center_z_m)
    labels[:, 12] = uniform(ranges.direct_thrust_fraction)
    remaining = 1.0 - labels[:, 12]
    partition_weights = 0.7 + 0.6 * torch.rand(
        group_count, 3, generator=generator, dtype=dtype
    )
    labels[:, 13:16] = (
        remaining[:, None]
        * partition_weights
        / partition_weights.sum(dim=1, keepdim=True)
    )

    base_angles = labels.new_tensor(
        [math.pi, math.pi / 3.0, -math.pi / 3.0]
    )
    grid_radius = uniform(ranges.grid_radius_m)[:, None].expand(-1, 3)
    grid_angle = base_angles[None] + uniform(
        ranges.grid_azimuth_error_rad, group_count * 3
    ).reshape(group_count, 3)
    grid_z = uniform(ranges.grid_center_z_m, group_count * 3).reshape(
        group_count, 3
    )
    grid_centers = torch.stack(
        (
            grid_radius * torch.cos(grid_angle),
            grid_radius * torch.sin(grid_angle),
            grid_z,
        ),
        dim=2,
    )
    labels[:, 16:25] = grid_centers.reshape(group_count, 9)
    for index in range(3):
        labels[:, 25 + index] = log_uniform(ranges.grid_effectiveness_scale)
        labels[:, 28 + index] = log_uniform(ranges.servo_time_constant_s)
        labels[:, 31 + index] = uniform(ranges.servo_max_speed_rad_s)
    labels[:, 34] = uniform(ranges.coupling_attenuation)

    return labels


def _apply_effectiveness_labels(
    nominal: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    parameterization: str = "legacy_collective",
) -> dict[str, torch.Tensor]:
    expected = len(parameter_label_names(parameterization))
    if labels.ndim != 2 or labels.shape[1] != expected:
        raise ValueError(
            f"{parameterization} labels must have shape [batch, {expected}]"
        )
    parameters = {name: value.clone() for name, value in nominal.items()}
    if parameterization == "sim2real_micro":
        parameters["body.mass"].copy_(labels[:, 0])
        parameters["body.inertia_diagonal_b"].copy_(labels[:, 1:4])
        common = labels[:, 5]
        ratio_root = torch.sqrt(labels[:, 6])
        parameters["motors.torque_coefficient"][:, 0].mul_(
            common * ratio_root
        )
        parameters["motors.torque_coefficient"][:, 1].mul_(
            common / ratio_root
        )
        current_twr = _maximum_thrust_to_weight(parameters)
        parameters["aerodynamics.thrust_coefficients"].mul_(
            (labels[:, 4] / current_twr)[:, None]
        )
        parameters["motors.time_constant"].copy_(labels[:, 7:9])
        parameters["aerodynamics.direct_thrust_center_b"].copy_(
            labels[:, 9:12]
        )
        parameters["aerodynamics.thrust_partition"].copy_(labels[:, 12:16])
        parameters["aerodynamics.grids.aerodynamic_center_b"].copy_(
            labels[:, 16:25].reshape(-1, 3, 3)
        )
        parameters["aerodynamics.grids.vector_deflection.gain"].copy_(
            labels[:, 25:28]
        )
        parameters["servos.tau"].copy_(labels[:, 28:31])
        parameters["servos.max_speed"].copy_(labels[:, 31:34])
        coupling = parameters["aerodynamics.coupling_attenuation"]
        coupling.copy_(labels[:, 34, None, None].expand_as(coupling))
        diagonal = torch.arange(3, device=coupling.device)
        coupling[:, diagonal, diagonal] = 0.0
        return parameters
    scales = torch.pow(10.0, labels)
    parameters["body.inertia_diagonal_b"].mul_(scales[:, :3])

    collective = scales[:, 3]
    if parameterization == "legacy_collective":
        parameters["body.mass"].mul_(collective)
    parameters["aerodynamics.thrust_coefficients"].mul_(collective[:, None])
    parameters["motors.torque_coefficient"].mul_(scales[:, 4:5])
    parameters["motors.time_constant"].mul_(scales[:, 5:7])
    parameters["aerodynamics.grids.vector_deflection.gain"].mul_(scales[:, 7:10])
    parameters["servos.tau"].mul_(scales[:, 10:13])
    return parameters


def _sim2real_lqr_targets(
    parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return the local model and observer quantities used by sim2real LQI."""
    from flight_controller.math import lookup
    from flight_controller.plant import LocalPlantModel

    plant = LocalPlantModel(parameters)
    trim = plant.hover_trim()
    batch_size = trim.command.shape[0]
    dtype = trim.command.dtype
    device = trim.command.device
    actuator_state = torch.cat(
        (
            trim.motor_speed,
            torch.zeros(batch_size, 3, device=device, dtype=dtype),
        ),
        dim=1,
    )
    inertia = parameters["body.inertia_diagonal_b"]

    def acceleration(value: torch.Tensor) -> torch.Tensor:
        return plant.wrench_from_actuators(value[:, :2], value[:, 2:])[:, 1:] / inertia

    columns = []
    state_steps = actuator_state.new_tensor([0.1, 0.1, 1e-5, 1e-5, 1e-5])
    for index in range(5):
        offset = torch.zeros_like(actuator_state)
        offset[:, index] = state_steps[index]
        columns.append(
            (acceleration(actuator_state + offset) - acceleration(actuator_state - offset))
            / (2.0 * state_steps[index])
        )
    effectiveness = torch.stack(columns, dim=2)
    bias = acceleration(actuator_state)

    def actuator_target(command: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                lookup(command[:, :2], parameters["motors.pwm_to_rpm_table"]),
                lookup(command[:, 2:], parameters["servos.pwm_angle_table"]),
            ),
            dim=1,
        )

    command_columns = []
    command_step = trim.command.new_tensor(1e-5)
    for index in range(5):
        offset = torch.zeros_like(trim.command)
        offset[:, index] = command_step
        command_columns.append(
            (
                actuator_target(trim.command + offset)
                - actuator_target(trim.command - offset)
            )[:, index]
            / (2.0 * command_step)
        )
    command_slopes = torch.stack(command_columns, dim=1)
    time_constants = torch.cat(
        (parameters["motors.time_constant"], parameters["servos.tau"]), dim=1
    )
    return torch.cat(
        (
            effectiveness.reshape(batch_size, 15),
            bias,
            torch.log10(time_constants),
            torch.log10(command_slopes[:, :2].clamp_min(1e-12)),
            torch.log10(_maximum_thrust_to_weight(parameters))[:, None],
        ),
        dim=1,
    )


def _sample_initial_state(
    batch_size: int,
    config: IdentificationExperimentConfig,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = config.initial_state.minimum_tilt_rad
    high = config.initial_state.maximum_tilt_rad
    if config.initial_state.sampling_design == "stratified_multiaxis":
        trials = config.initial_conditions_per_group
        if batch_size % trials:
            raise ValueError(
                "stratified_multiaxis sampling requires complete parameter groups"
            )
        group_count = batch_size // trials

        def latin_hypercube() -> torch.Tensor:
            scores = torch.rand(
                group_count, trials, generator=generator, dtype=dtype
            )
            permutation = scores.argsort(dim=1).argsort(dim=1).to(dtype)
            jitter = torch.rand(
                group_count, trials, generator=generator, dtype=dtype
            )
            return ((permutation + jitter) / trials).reshape(-1)

        angle = low + latin_hypercube() * (high - low)
        direction_offset = torch.rand(
            group_count, 1, generator=generator, dtype=dtype
        )
        direction = (
            torch.arange(trials, dtype=dtype)[None, :] / trials
            + direction_offset
        ).remainder(1.0).reshape(-1) * (2.0 * math.pi)
        maximum_rate = torch.tensor(
            config.initial_state.maximum_angular_rate_rad_s, dtype=dtype
        )
        angular_velocity = torch.stack(
            tuple((2.0 * latin_hypercube() - 1.0) * value for value in maximum_rate),
            dim=1,
        )
    else:
        angle = low + torch.rand(
            batch_size, generator=generator, dtype=dtype
        ) * (high - low)
        direction = torch.rand(
            batch_size, generator=generator, dtype=dtype
        ) * (2.0 * math.pi)
        maximum_rate = torch.tensor(
            config.initial_state.maximum_angular_rate_rad_s, dtype=dtype
        )
        angular_velocity = (
            2.0 * torch.rand(batch_size, 3, generator=generator, dtype=dtype) - 1.0
        ) * maximum_rate
    half_sine = torch.sin(0.5 * angle)
    attitude = torch.stack(
        (
            torch.cos(0.5 * angle),
            half_sine * torch.cos(direction),
            half_sine * torch.sin(direction),
            torch.zeros_like(angle),
        ),
        dim=1,
    )
    return attitude, angular_velocity


def _yaw_free_attitude(attitude_q_wb: torch.Tensor) -> torch.Tensor:
    """Return the observable tilt quaternion with yaw fixed to zero."""

    w, x, y, z = attitude_q_wb.unbind(dim=1)
    roll = torch.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x.square() + y.square()),
    )
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    cosine_roll = torch.cos(half_roll)
    sine_roll = torch.sin(half_roll)
    cosine_pitch = torch.cos(half_pitch)
    sine_pitch = torch.sin(half_pitch)
    return torch.stack(
        (
            cosine_roll * cosine_pitch,
            sine_roll * cosine_pitch,
            cosine_roll * sine_pitch,
            -sine_roll * sine_pitch,
        ),
        dim=1,
    )


def _group_split_assignments(
    group_count: int,
    fractions: tuple[float, float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    permutation = torch.randperm(group_count, generator=generator)
    train_count = round(group_count * fractions[0])
    validation_count = round(group_count * fractions[1])
    if group_count >= 3:
        train_count = min(max(train_count, 1), group_count - 2)
        validation_count = min(max(validation_count, 1), group_count - train_count - 1)
    else:
        train_count = max(1, min(train_count, group_count))
        validation_count = max(0, min(validation_count, group_count - train_count))
    assignment = torch.full((group_count,), 2, dtype=torch.int64)
    assignment[permutation[:train_count]] = 0
    assignment[permutation[train_count : train_count + validation_count]] = 1
    return assignment


class FirstOrderServoObserver:
    """Command-driven servo state used when no servo-angle feedback exists."""

    def __init__(
        self,
        pwm_angle_table: torch.Tensor,
        time_constant: torch.Tensor,
        dt: float,
    ) -> None:
        self.table = pwm_angle_table
        self.response = 1.0 - torch.exp(-dt / time_constant)
        self.angle = torch.zeros_like(time_constant)

    def advance(self, command: torch.Tensor) -> None:
        from flight_controller.math import lookup

        target = lookup(command, self.table)
        self.angle.add_(self.response * (target - self.angle))


class CommandDrivenServoObserver:
    """Command-only observer including deadzone, backlash and rate limiting."""

    def __init__(
        self,
        pwm_angle_table: torch.Tensor,
        time_constant: torch.Tensor,
        maximum_speed: torch.Tensor,
        backlash: torch.Tensor,
        deadzone: torch.Tensor,
        dt: float,
    ) -> None:
        self.table = pwm_angle_table
        self.time_constant = time_constant
        self.maximum_speed = maximum_speed
        self.backlash = backlash
        self.deadzone = deadzone
        self.dt = dt
        self.angle = torch.zeros_like(time_constant)
        self.effective_pwm = torch.zeros_like(time_constant)
        self.command_angle = torch.zeros_like(time_constant)
        self.target_angle = torch.zeros_like(time_constant)
        self.motion_direction = torch.zeros_like(time_constant)
        self.backlash_remaining = torch.zeros_like(time_constant)

    def advance(self, command: torch.Tensor) -> None:
        from flight_controller.math import lookup

        outside_deadzone = (
            torch.abs(command - self.effective_pwm) >= self.deadzone
        )
        effective_pwm = torch.where(
            outside_deadzone, command, self.effective_pwm
        )
        command_angle = lookup(effective_pwm, self.table)
        command_delta = command_angle - self.command_angle
        command_direction = torch.sign(command_delta)
        reversing = (
            (command_direction != 0)
            & (self.motion_direction != 0)
            & (command_direction != self.motion_direction)
        )
        backlash_remaining = torch.where(
            reversing, self.backlash, self.backlash_remaining
        )
        backlash_consumed = torch.minimum(
            torch.abs(command_delta), backlash_remaining
        )
        transmitted_delta = command_direction * (
            torch.abs(command_delta) - backlash_consumed
        )
        self.target_angle.add_(transmitted_delta)
        self.backlash_remaining.copy_(
            backlash_remaining - backlash_consumed
        )
        self.motion_direction.copy_(
            torch.where(
                command_direction != 0,
                command_direction,
                self.motion_direction,
            )
        )
        self.effective_pwm.copy_(effective_pwm)
        self.command_angle.copy_(command_angle)

        error = self.target_angle - self.angle
        direction = torch.sign(error)
        exponential_boundary = self.time_constant * self.maximum_speed
        linear_time = torch.clamp_min(
            (torch.abs(error) - exponential_boundary) / self.maximum_speed,
            0.0,
        )
        duration = torch.full_like(self.angle, self.dt)
        linear_duration = torch.minimum(linear_time, duration)
        angle_after_linear = (
            self.angle + direction * self.maximum_speed * linear_duration
        )
        exponential_duration = duration - linear_duration
        self.angle.copy_(
            self.target_angle
            + (angle_after_linear - self.target_angle)
            * torch.exp(-exponential_duration / self.time_constant)
        )


def _set_initial_observation_state(
    environment: Any,
    motor_speed: torch.Tensor,
    angular_velocity: torch.Tensor,
) -> None:
    state = environment.state_dict()
    state["truth"]["motor_speed"].copy_(motor_speed)
    state["truth"]["effective_motor_speed"].copy_(motor_speed)
    state["sensors"]["motor_speed"].copy_(motor_speed)
    state["sensor_kernel"]["history"]["motor_speed"].copy_(
        motor_speed[:, None, :]
    )
    state["sensors"]["gyro"].copy_(angular_velocity)
    state["sensor_kernel"]["history"]["gyro"].copy_(
        angular_velocity[:, None, :]
    )
    environment.load_state_dict(state)


def _window_information(features: torch.Tensor) -> torch.Tensor:
    rate_rms = torch.sqrt(features[..., 4:7].square().mean(dim=(1, 2)))
    command = features[..., 9:14]
    if command.shape[1] <= 1:
        movement = torch.zeros_like(rate_rms)
    else:
        movement = torch.sqrt(
            (command[:, 1:] - command[:, :-1]).square().mean(dim=(1, 2))
        )
    return rate_rms + movement


def _save_window_shards(
    output_directory: Path,
    shard_index: int,
    trajectories: torch.Tensor,
    observer_history: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    episode_ids: torch.Tensor,
    successful: torch.Tensor,
    split_assignment: torch.Tensor,
    window_steps: int,
    stride_steps: int,
    maximum_start_step: int | None,
    relative_root: str = "",
    metadata: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, int]:
    split_names = ("train", "validation", "test")
    counts: dict[str, int] = {name: 0 for name in split_names}
    last_start = trajectories.shape[1] - window_steps
    if maximum_start_step is not None:
        last_start = min(last_start, maximum_start_step)
    starts = range(0, last_start + 1, stride_steps)
    for split_index, split_name in enumerate(split_names):
        episode_mask = successful & (split_assignment[group_ids] == split_index)
        if not bool(episode_mask.any().item()):
            continue
        chunks = []
        observer_initial = []
        starts_out = []
        for start in starts:
            chunks.append(trajectories[episode_mask, start : start + window_steps])
            observer_initial.append(observer_history[episode_mask, start])
            starts_out.append(
                torch.full(
                    (int(episode_mask.sum().item()),), start, dtype=torch.int64
                )
            )
        features = torch.cat(chunks, dim=0).contiguous()
        repeat_count = len(chunks)
        payload = {
            "schema_version": 1,
            "features": features,
            "labels": labels[episode_mask].repeat(repeat_count, 1).contiguous(),
            "group_id": group_ids[episode_mask].repeat(repeat_count).contiguous(),
            "episode_id": episode_ids[episode_mask].repeat(repeat_count).contiguous(),
            "window_start_step": torch.cat(starts_out),
            "controller_servo_observer_initial": torch.cat(observer_initial, dim=0),
            "information_score": _window_information(features),
            "feature_names": FEATURE_NAMES,
            "label_names": LABEL_NAMES,
        }
        if metadata is not None:
            for name, value in metadata.items():
                selected = value[episode_mask]
                repeats = (repeat_count,) + (1,) * (selected.ndim - 1)
                payload[name] = selected.repeat(repeats).contiguous()
        split_directory = output_directory / relative_root / split_name
        split_directory.mkdir(parents=True, exist_ok=True)
        torch.save(payload, split_directory / f"shard_{shard_index:06d}.pt")
        counts[split_name] = int(features.shape[0])
    return counts


def _save_failures(
    output_directory: Path,
    shard_index: int,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    episode_ids: torch.Tensor,
    failure_code: torch.Tensor,
    max_tilt: torch.Tensor,
    max_rate: torch.Tensor,
    saturation_fraction: torch.Tensor,
) -> int:
    failed = failure_code != 0
    if not bool(failed.any().item()):
        return 0
    directory = output_directory / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "labels": labels[failed].contiguous(),
            "group_id": group_ids[failed].contiguous(),
            "episode_id": episode_ids[failed].contiguous(),
            "failure_code": failure_code[failed].contiguous(),
            "maximum_tilt_rad": max_tilt[failed].contiguous(),
            "maximum_angular_rate_rad_s": max_rate[failed].contiguous(),
            "saturation_fraction": saturation_fraction[failed].contiguous(),
            "failure_names": FAILURE_NAMES,
            "label_names": LABEL_NAMES,
        },
        directory / f"shard_{shard_index:06d}.pt",
    )
    return int(failed.sum().item())


@torch.no_grad()
def generate_dataset(config: IdentificationExperimentConfig) -> Mapping[str, Any]:
    from flight_controller import (
        ControllerContext,
        ControllerReference,
        ControllerState,
        create_controller,
        load_controller_config,
    )
    from flight_controller.math import quaternion_rotation_error, tilt_cosine
    from simenv import SimulationEnvironment
    from simenv.config import load_and_materialize

    if config.parameterization != "legacy_collective":
        raise ValueError(
            "empirical_core is defined for repeated-trial generation; use "
            "flight_identification.repeated_trial_experiment"
        )

    output = config.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.source_path, output / "experiment.yaml")

    device = torch.device(config.device)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    nominal_materialized = load_and_materialize(
        config.simulator_config, 1, device, dtype
    )
    controller_config = dict(load_controller_config(config.controller_config))
    if str(controller_config.get("type")) != "lqr":
        raise ValueError("identification experiment requires controller type lqr")

    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(config.seed)
    split_generator = torch.Generator(device="cpu")
    split_generator.manual_seed(config.seed + 1)
    split_assignment = _group_split_assignments(
        config.parameter_groups,
        config.window.split_fractions,
        split_generator,
    )
    labels_by_group = _sample_group_labels(
        config.parameter_groups,
        nominal_materialized.parameters["body.inertia_diagonal_b"][0].cpu(),
        config.log10_effectiveness_range,
        sampling_generator,
        dtype,
    )
    torch.save(
        {
            "schema_version": 1,
            "labels": labels_by_group,
            "group_id": torch.arange(config.parameter_groups, dtype=torch.int64),
            "split_assignment": split_assignment,
            "label_names": LABEL_NAMES,
        },
        output / "parameter_groups.pt",
    )

    groups_per_batch = config.parallel_count // config.initial_conditions_per_group
    totals = {
        "episodes": 0,
        "converged_episodes": 0,
        "failed_episodes": 0,
        "train_windows": 0,
        "validation_windows": 0,
        "test_windows": 0,
        "adaptation_train_windows": 0,
        "adaptation_validation_windows": 0,
        "adaptation_test_windows": 0,
    }
    nominal_pole_radius: float | None = None
    hold_steps = round(config.convergence.hold_s * config.control_hz)

    for shard_index, group_start in enumerate(
        range(0, config.parameter_groups, groups_per_batch)
    ):
        group_stop = min(group_start + groups_per_batch, config.parameter_groups)
        group_count = group_stop - group_start
        batch_size = group_count * config.initial_conditions_per_group
        group_ids = torch.arange(group_start, group_stop, dtype=torch.int64).repeat_interleave(
            config.initial_conditions_per_group
        )
        condition_ids = torch.arange(
            config.initial_conditions_per_group, dtype=torch.int64
        ).repeat(group_count)
        episode_ids = group_ids * config.initial_conditions_per_group + condition_ids
        episode_labels = labels_by_group[group_start:group_stop].repeat_interleave(
            config.initial_conditions_per_group, dim=0
        )

        nominal_parameters = _expand_mapping(
            nominal_materialized.parameters, batch_size
        )
        actual_parameters = _apply_effectiveness_labels(
            nominal_parameters, episode_labels.to(device)
        )
        initial_state = _expand_mapping(
            nominal_materialized.initial_state, batch_size
        )
        attitude, angular_velocity = _sample_initial_state(
            batch_size, config, sampling_generator, dtype
        )
        initial_state["attitude_q_wb"].copy_(attitude.to(device))
        initial_state["angular_velocity_b"].copy_(angular_velocity.to(device))
        materialized = replace(
            nominal_materialized,
            parameters=actual_parameters,
            initial_state=initial_state,
            sensor_state=_expand_mapping(nominal_materialized.sensor_state, batch_size),
        )
        environment = SimulationEnvironment(
            materialized,
            batch_size,
            device,
            dtype,
            logging_enabled=False,
        )
        try:
            controller = create_controller(
                controller_config,
                ControllerContext(
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                    control_dt=1.0 / config.control_hz,
                    parameters=nominal_parameters,
                ),
            )
            if nominal_pole_radius is None:
                nominal_pole_radius = float(max(abs(controller._lqr_poles)))
            _set_initial_observation_state(
                environment,
                controller.trim.motor_speed,
                initial_state["angular_velocity_b"],
            )
            servo_observer = FirstOrderServoObserver(
                nominal_parameters["servos.pwm_angle_table"],
                nominal_parameters["servos.tau"],
                1.0 / config.control_hz,
            )
            zeros_3 = torch.zeros(batch_size, 3, device=device, dtype=dtype)
            identity = torch.zeros(batch_size, 4, device=device, dtype=dtype)
            identity[:, 0] = 1.0
            reference = ControllerReference(
                target_position_n=zeros_3,
                target_velocity_n=zeros_3.clone(),
                target_attitude_q_wb=identity,
                target_angular_velocity_b=zeros_3.clone(),
                collective_command=torch.zeros(batch_size, 1, device=device, dtype=dtype),
            )
            trajectories = torch.empty(
                batch_size,
                config.episode_steps,
                len(FEATURE_NAMES),
                device=device,
                dtype=dtype,
            )
            observer_history = torch.empty(
                batch_size, config.episode_steps, 3, device=device, dtype=dtype
            )
            settled_count = torch.zeros(batch_size, device=device, dtype=torch.int64)
            max_tilt = torch.zeros(batch_size, device=device, dtype=dtype)
            max_rate = torch.zeros_like(max_tilt)
            saturation_steps = torch.zeros_like(settled_count)
            invalid = torch.zeros(batch_size, device=device, dtype=torch.bool)
            tilt_failed = torch.zeros_like(invalid)
            rate_failed = torch.zeros_like(invalid)
            first_unsafe_step = torch.full(
                (batch_size,),
                config.episode_steps + 1,
                device=device,
                dtype=torch.int64,
            )

            for step in range(config.episode_steps):
                truth = environment.observe("truth").values
                if config.measurement_mode == "simulated_sensors":
                    sensors = environment.observe("sensor").values
                    measured_rate = sensors["gyro"]
                    measured_motor_speed = sensors["motor_speed"]
                else:
                    measured_rate = truth["angular_velocity_b"]
                    measured_motor_speed = truth["motor_speed"]
                tilt_attitude = _yaw_free_attitude(truth["attitude_q_wb"])
                controller_state = ControllerState(
                    position_n=truth["position_n"],
                    velocity_n=truth["velocity_n"],
                    attitude_q_wb=tilt_attitude,
                    angular_velocity_b=measured_rate,
                    linear_acceleration_n=truth["linear_acceleration_n"],
                    motor_speed=measured_motor_speed,
                    servo_angle=servo_observer.angle,
                )
                controller_output = controller.step(controller_state, reference)
                feature = torch.cat(
                    (
                        tilt_attitude,
                        measured_rate,
                        measured_motor_speed,
                        controller_output.command,
                    ),
                    dim=1,
                )
                trajectories[:, step].copy_(feature)
                observer_history[:, step].copy_(servo_observer.angle)

                attitude_error = quaternion_rotation_error(
                    tilt_attitude, identity
                )[:, :2].norm(dim=1)
                angular_rate = truth["angular_velocity_b"].norm(dim=1)
                tilt = torch.acos(tilt_cosine(truth["attitude_q_wb"]))
                settled = (
                    (attitude_error <= config.convergence.maximum_roll_pitch_error_rad)
                    & (angular_rate <= config.convergence.maximum_angular_rate_rad_s)
                )
                settled_count = torch.where(
                    settled, settled_count + 1, torch.zeros_like(settled_count)
                )
                max_tilt = torch.maximum(max_tilt, tilt)
                max_rate = torch.maximum(max_rate, angular_rate)
                tilt_failed |= tilt > config.convergence.safety_tilt_rad
                rate_failed |= angular_rate > config.convergence.safety_angular_rate_rad_s
                command = controller_output.command
                saturated = (
                    (command[:, :2] <= 1e-6).any(dim=1)
                    | (command[:, :2] >= 1.0 - 1e-6).any(dim=1)
                    | (command[:, 2:].abs() >= 1.0 - 1e-6).any(dim=1)
                )
                saturation_steps += saturated.to(torch.int64)
                servo_observer.advance(command[:, 2:])
                result = environment.advance(command)
                invalid |= ~result.valid
                unsafe = invalid | tilt_failed | rate_failed
                first_unsafe_step = torch.where(
                    unsafe & (first_unsafe_step > config.episode_steps),
                    torch.full_like(first_unsafe_step, step + 1),
                    first_unsafe_step,
                )

            successful = (
                (settled_count >= hold_steps)
                & ~invalid
                & ~tilt_failed
                & ~rate_failed
            )
            failure_code = torch.ones(batch_size, device=device, dtype=torch.int64)
            failure_code = torch.where(tilt_failed, 2, failure_code)
            failure_code = torch.where(rate_failed, 3, failure_code)
            failure_code = torch.where(invalid, 4, failure_code)
            failure_code = torch.where(successful, 0, failure_code)
            saturation_fraction = saturation_steps.to(dtype) / config.episode_steps
            trajectories_cpu = trajectories.cpu()
            observer_history_cpu = observer_history.cpu()

            shard_counts = _save_window_shards(
                output,
                shard_index,
                trajectories_cpu,
                observer_history_cpu,
                episode_labels,
                group_ids,
                episode_ids,
                successful.cpu(),
                split_assignment,
                config.window_steps,
                config.stride_steps,
                (
                    None
                    if config.window.maximum_start_s is None
                    else round(config.window.maximum_start_s * config.control_hz)
                ),
            )
            adaptation_eligible = first_unsafe_step > config.window_steps
            adaptation_counts = _save_window_shards(
                output,
                shard_index,
                trajectories_cpu,
                observer_history_cpu,
                episode_labels,
                group_ids,
                episode_ids,
                adaptation_eligible.cpu(),
                split_assignment,
                config.window_steps,
                config.stride_steps,
                0,
                relative_root="adaptation",
                metadata={
                    "final_success": successful.cpu(),
                    "final_failure_code": failure_code.cpu(),
                    "first_unsafe_step": first_unsafe_step.cpu(),
                    "saturation_fraction": saturation_fraction.cpu(),
                },
            )
            failed_count = _save_failures(
                output,
                shard_index,
                episode_labels,
                group_ids,
                episode_ids,
                failure_code.cpu(),
                max_tilt.cpu(),
                max_rate.cpu(),
                saturation_fraction.cpu(),
            )
            totals["episodes"] += batch_size
            totals["converged_episodes"] += int(successful.sum().item())
            totals["failed_episodes"] += failed_count
            for name, count in shard_counts.items():
                totals[f"{name}_windows"] += count
            for name, count in adaptation_counts.items():
                totals[f"adaptation_{name}_windows"] += count
        finally:
            environment.close()

    manifest = {
        "schema_version": 1,
        "experiment": "closed_loop_lqr_identification_v1",
        "controller_semantics": {
            "roll_pitch_target_rad": [0.0, 0.0],
            "yaw_target": "angular_rate_zero",
            "yaw_angle_feedback": False,
            "attitude_source": "yaw_free_tilt_quaternion",
            "servo_angle_feedback": False,
            "servo_state_source": "nominal_first_order_command_observer",
            "gain_source": "fixed_nominal_environment",
            "measurement_mode": config.measurement_mode,
            "attitude_estimate_mode": "truth_tilt_proxy",
            "nominal_closed_loop_pole_radius": nominal_pole_radius,
        },
        "control_hz": config.control_hz,
        "feature_names": FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "label_encoding": "base-10 logarithm of multiplicative nominal scale",
        "label_range": list(config.log10_effectiveness_range),
        "failure_names": FAILURE_NAMES,
        "split_semantics": "parameter group assigned before episode windowing",
        "adaptation_semantics": (
            "first window remains safety-valid; final convergence is metadata, "
            "not an inclusion criterion"
        ),
        "window_steps": config.window_steps,
        "stride_steps": config.stride_steps,
        "maximum_window_start_s": config.window.maximum_start_s,
        "totals": totals,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
