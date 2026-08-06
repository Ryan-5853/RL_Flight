from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import load_experiment_config
from .control_evaluation import _lqr_gain, _nominal_actuator_model
from .experiment import (
    _apply_effectiveness_labels,
    _expand_mapping,
    _sim2real_lqr_targets,
)
from .repeated_trial_training import _retain_converged_trials, load_repeated_split
from .sim2real_composite import (
    DEFAULT_BASIS_TIME_CONSTANTS_S,
    DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
    composite_discrete_model,
    composite_lqr_weights,
    fit_coefficients_from_step_response,
    fit_composite_coefficients,
    merge_adaptive_composite_coefficients,
    servo_mode_transform,
)
from .sim2real_composite_evaluation import _predict_composite


SCHEMES = (
    "pid",
    "nominal_lqr",
    "oracle_lqr",
    "offline_lqr",
    "gru_lqr",
    "e2e_nn",
    "aa_nn",
)


SINGLE_DIMS = (
    "mass",
    "inertia",
    "servo_eff",
    "motor_eff",
    "servo_tau",
    "motor_tau",
    "sensor_noise",
    "sensor_delay",
)


SINGLE_STRENGTHS: Mapping[str, Sequence[float]] = {
    "mass": (0.7, 1.3, 2.0),
    "inertia": (0.7, 1.3, 2.0),
    "servo_eff": (0.7, 1.3, 2.0),
    "motor_eff": (0.7, 1.3, 2.0),
    "servo_tau": (0.7, 1.3, 2.0),
    "motor_tau": (0.7, 1.3, 2.0),
    "sensor_noise": (1.0, 3.0, 10.0),
    "sensor_delay": (0.002, 0.005, 0.010),
}


CROSS_CELLS: Sequence[Mapping[str, Any]] = (
    {
        "name": "mass_x_servo_eff",
        "fields": ("mass", "servo_eff"),
        "strengths": ((0.7, 0.7), (1.5, 1.5)),
    },
    {
        "name": "mass_x_servo_tau",
        "fields": ("mass", "servo_tau"),
        "strengths": ((0.7, 0.7), (1.5, 1.5)),
    },
    {
        "name": "servo_eff_x_noise",
        "fields": ("servo_eff", "sensor_noise"),
        "strengths": ((0.7, 3.0), (1.5, 10.0)),
    },
    {
        "name": "motor_eff_x_motor_tau",
        "fields": ("motor_eff", "motor_tau"),
        "strengths": ((0.7, 0.7), (1.5, 1.5)),
    },
    {
        "name": "mass_x_inertia_x_servo_tau",
        "fields": ("mass", "inertia", "servo_tau"),
        "strengths": ((0.7, 0.7, 0.7), (1.5, 1.5, 1.5)),
    },
)


def error_cells() -> list[dict[str, Any]]:
    cells = [{"name": "baseline", "fields": (), "strengths": ((),)}]
    for dim in SINGLE_DIMS:
        for strength in SINGLE_STRENGTHS[dim]:
            cells.append(
                {
                    "name": f"{dim}_x{strength}",
                    "fields": (dim,),
                    "strengths": ((strength,),),
                }
            )
    for cross in CROSS_CELLS:
        for strength in cross["strengths"]:
            cells.append(
                {
                    "name": f"{cross['name']}_x{'_'.join(str(round(v, 3)) for v in strength)}",
                    "fields": tuple(cross["fields"]),
                    "strengths": (strength,),
                }
            )
    return cells


def apply_error(
    parameters: Mapping[str, torch.Tensor],
    fields: Sequence[str],
    strengths: Sequence[float],
) -> None:
    for field, strength in zip(fields, strengths):
        if field == "mass":
            parameters["body.mass"].mul_(strength)
        elif field == "inertia":
            parameters["body.inertia_diagonal_b"].mul_(strength)
        elif field == "servo_eff":
            parameters["aerodynamics.grids.vector_deflection.gain"].mul_(
                strength
            )
        elif field == "motor_eff":
            parameters["aerodynamics.thrust_coefficients"].mul_(strength)
        elif field == "servo_tau":
            parameters["servos.tau"].mul_(strength)
        elif field == "motor_tau":
            parameters["motors.time_constant"].mul_(strength)
        elif field == "sensor_noise":
            parameters["sensors.gyro.noise.stddev"].mul_(strength)
        elif field == "sensor_delay":
            parameters["sensors.gyro.delay"].add_(strength)
        else:
            raise ValueError(f"unknown error dimension: {field}")


def _slice_split(
    split: Mapping[str, Any], count: int
) -> dict[str, Any]:
    output = dict(split)
    for name, value in tuple(output.items()):
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == len(
            split["features"]
        ):
            output[name] = value[:count]
    return output


@torch.no_grad()
def predict_step_response(
    checkpoint_path: str,
    dataset: Path,
    count: int,
    device: torch.device,
    batch_size: int,
    trial_count: int,
    split_group_id: torch.Tensor | None = None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("artifact_type") != "sim2real_composite_servo_identifier":
        raise ValueError("expected a composite servo identifier checkpoint")
    split = load_repeated_split(
        dataset, "test", int(checkpoint["downsample"])
    )
    _retain_converged_trials(split)
    split = _slice_split(split, count)
    if split_group_id is not None and not torch.equal(
        split["group_id"], split_group_id
    ):
        raise ValueError("checkpoint split group ordering differs")
    prediction = _predict_composite(
        checkpoint,
        split,
        device,
        trial_count,
        batch_size,
    )
    prediction = torch.maximum(
        torch.minimum(prediction, checkpoint["label_max"]),
        checkpoint["label_min"],
    )
    return prediction.numpy(), checkpoint


def merge_axis_predictions(
    predictions: Mapping[str, np.ndarray],
    label_names: Sequence[str],
) -> np.ndarray:
    merged = predictions["mlp"].copy()
    for axis, source in (("roll", "tcn"), ("pitch", "gru"), ("yaw", "mlp")):
        columns = [
            index
            for index, name in enumerate(label_names)
            if f".{axis}." in name
        ]
        merged[:, columns] = predictions[source][:, columns]
    return merged


def synthesize_gains(
    coefficients: np.ndarray,
    nominal_effectiveness: np.ndarray,
    command_slopes: np.ndarray,
    nominal_tau: np.ndarray,
    mode_transform: np.ndarray,
    servo_slopes: np.ndarray,
    basis_tau: Sequence[float],
    q_composite: np.ndarray,
    r_composite: np.ndarray,
) -> np.ndarray:
    gains = []
    for current in coefficients:
        a, b = composite_discrete_model(
            nominal_effectiveness[:, :2],
            command_slopes[:2],
            nominal_tau[:2],
            current,
            mode_transform,
            servo_slopes,
            basis_tau,
            upper_external=True,
        )
        gains.append(_lqr_gain(a, b, q_composite, r_composite))
    return np.stack(gains)


def _lqr_weights_from_controller_config(
    controller_config_path: str, basis_count: int
) -> tuple[np.ndarray, np.ndarray]:
    from flight_controller import load_controller_config

    lqr_config = load_controller_config(controller_config_path)["params"]["lqr"]
    return composite_lqr_weights(
        np.asarray(lqr_config["state_scales"], dtype=np.float64),
        np.asarray(lqr_config["integral_state_scales"], dtype=np.float64),
        np.asarray(lqr_config["input_scales"], dtype=np.float64),
        float(lqr_config["input_weight_scale"]),
        basis_count,
    )


def build_base_vehicles(
    args: argparse.Namespace,
) -> tuple[Mapping[str, Any], Mapping[str, Any], torch.Tensor, torch.Tensor]:
    experiment = load_experiment_config(args.experiment_config)
    device = torch.device(args.device)
    dtype = torch.float32 if experiment.dtype == "float32" else torch.float64
    from simenv.config import load_and_materialize

    nominal_materialized = load_and_materialize(
        experiment.simulator_config, 1, device, dtype
    )
    split = load_repeated_split(Path(args.dataset), "test", 1)
    _retain_converged_trials(split)
    split = _slice_split(split, args.n_groups)
    nominal_parameters = _expand_mapping(
        nominal_materialized.parameters, args.n_groups
    )
    labels = split["labels"].to(device)
    actual_parameters = _apply_effectiveness_labels(
        nominal_parameters, labels, experiment.parameterization
    )
    return actual_parameters, split, device, dtype


@torch.no_grad()
def simulate_cell(
    args: argparse.Namespace,
    cell: Mapping[str, Any],
    scheme: str,
    actual_parameters: Mapping[str, torch.Tensor],
    attitude: torch.Tensor,
    angular_velocity: torch.Tensor,
    gains: Mapping[str, np.ndarray],
    oracle_coefficients: np.ndarray,
    nominal_effectiveness: np.ndarray,
    command_slopes: np.ndarray,
    nominal_tau: np.ndarray,
    mode_transform: np.ndarray,
    servo_slopes: np.ndarray,
    basis_tau: Sequence[float],
    q_composite: np.ndarray,
    r_composite: np.ndarray,
) -> dict[str, Any]:
    experiment = load_experiment_config(args.experiment_config)
    device = torch.device(args.device)
    dtype = torch.float32 if experiment.dtype == "float32" else torch.float64
    count = len(attitude)
    from dataclasses import replace

    from flight_controller import (
        ControllerContext,
        ControllerReference,
        ControllerState,
        create_controller,
        load_controller_config,
    )
    from flight_controller.math import quaternion_rotation_error, tilt_cosine
    from flight_controller.pilot import VirtualPilotHeightController
    from simenv import SimulationEnvironment
    from simenv.config import load_and_materialize

    nominal_materialized = load_and_materialize(
        experiment.simulator_config, 1, device, dtype
    )
    nominal_parameters = _expand_mapping(
        nominal_materialized.parameters, count
    )
    initial_state = _expand_mapping(nominal_materialized.initial_state, count)
    initial_state["attitude_q_wb"].copy_(attitude.to(device))
    initial_state["angular_velocity_b"].copy_(angular_velocity.to(device))
    materialized = replace(
        nominal_materialized,
        parameters=actual_parameters,
        initial_state=initial_state,
        sensor_state=_expand_mapping(nominal_materialized.sensor_state, count),
    )
    environment = SimulationEnvironment(
        materialized, count, device, dtype, logging_enabled=False
    )
    total_steps = round(
        args.duration_s * experiment.control_hz
    )
    identity = torch.zeros(count, 4, device=device, dtype=dtype)
    identity[:, 0] = 1.0
    zeros3 = torch.zeros(count, 3, device=device, dtype=dtype)
    pilot_config = load_controller_config(args.controller_config)["params"][
        "virtual_pilot"
    ]
    pilot = VirtualPilotHeightController(
        pilot_config, count, device, dtype, 1.0 / experiment.control_hz
    )
    from .experiment import FirstOrderServoObserver
    from .sim2real_composite_evaluation import _FixedServoFilterBank

    servo_observer = FirstOrderServoObserver(
        nominal_parameters["servos.pwm_angle_table"],
        nominal_parameters["servos.tau"],
        1.0 / experiment.control_hz,
    )
    servo_filter = _FixedServoFilterBank(
        count,
        mode_transform,
        nominal_parameters["servos.pwm_angle_table"],
        nominal_parameters["servos.backlash"],
        nominal_parameters["servos.deadzone"],
        basis_tau,
        1.0 / experiment.control_hz,
        device,
        dtype,
    )

    def run_batch(
        controller: Any,
        reference_collective: bool,
        active: torch.Tensor,
        use_filter: bool,
    ) -> dict[str, torch.Tensor]:
        settled_count = torch.zeros(count, device=device, dtype=torch.int64)
        saturation_steps = torch.zeros(count, device=device, dtype=torch.int64)
        cum_rp = torch.zeros(count, device=device, dtype=dtype)
        cum_rate = torch.zeros(count, device=device, dtype=dtype)
        cum_yaw = torch.zeros(count, device=device, dtype=dtype)
        peak_tilt = torch.zeros(count, device=device, dtype=dtype)
        peak_rate = torch.zeros(count, device=device, dtype=dtype)
        height_min = torch.full(
            (count,), torch.inf, device=device, dtype=dtype
        )
        height_max = torch.full(
            (count,), -torch.inf, device=device, dtype=dtype
        )
        final_attitude_error = torch.full(
            (count,), torch.inf, device=device, dtype=dtype
        )
        final_rate = torch.full_like(final_attitude_error, torch.inf)
        for _ in range(total_steps):
            truth = environment.observe("truth").values
            sensors = environment.observe("sensor").values
            measured_rate = sensors["gyro"]
            measured_motor = sensors["motor_speed"]
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
            safe_tilt = torch.nan_to_num(tilt_attitude)
            safe_rate = torch.nan_to_num(measured_rate)
            safe_motor = torch.nan_to_num(measured_motor)
            height = -truth["position_n"][:, 2]
            height_min = torch.minimum(height_min, height)
            height_max = torch.maximum(height_max, height)
            reference = ControllerReference(
                target_position_n=zeros3.clone(),
                target_velocity_n=zeros3.clone(),
                target_attitude_q_wb=identity,
                target_angular_velocity_b=zeros3.clone(),
                collective_command=(
                    pilot.step(
                        height,
                        -truth["velocity_n"][:, 2],
                        active,
                    )
                    if reference_collective
                    else torch.zeros(count, 1, device=device, dtype=dtype)
                ),
            )
            state = ControllerState(
                position_n=torch.nan_to_num(truth["position_n"]),
                velocity_n=torch.nan_to_num(truth["velocity_n"]),
                attitude_q_wb=safe_tilt,
                angular_velocity_b=safe_rate,
                linear_acceleration_n=torch.nan_to_num(
                    truth["linear_acceleration_n"]
                ),
                motor_speed=safe_motor,
                servo_angle=(
                    servo_filter.state.flatten(start_dim=1)
                    if use_filter
                    else servo_observer.angle
                ),
            )
            output = controller.step(state, reference, active)
            attitude_error = quaternion_rotation_error(safe_tilt, identity)[
                :, :2
            ]
            rp_error = attitude_error.abs().sum(dim=1)
            yaw_rate = safe_rate[:, 2].abs()
            cum_rp += rp_error * (1.0 / experiment.control_hz)
            cum_rate += rate_norm * (1.0 / experiment.control_hz)
            cum_yaw += yaw_rate * (1.0 / experiment.control_hz)
            peak_tilt = torch.maximum(peak_tilt, tilt)
            peak_rate = torch.maximum(peak_rate, rate_norm)
            settled = active & (
                rp_error
                <= experiment.convergence.maximum_roll_pitch_error_rad
            ) & (
                rate_norm <= experiment.convergence.maximum_angular_rate_rad_s
            )
            settled_count = torch.where(
                settled, settled_count + 1, torch.zeros_like(settled_count)
            )
            final_attitude_error = torch.where(
                active, rp_error, final_attitude_error
            )
            final_rate = torch.where(active, rate_norm, final_rate)
            command = output.command
            saturated = active & (
                (command[:, :2] <= 1e-6).any(dim=1)
                | (command[:, :2] >= 1.0 - 1e-6).any(dim=1)
                | (command[:, 2:].abs() >= 1.0 - 1e-6).any(dim=1)
            )
            saturation_steps += saturated.to(torch.int64)
            if use_filter:
                servo_filter.advance(command[:, 2:])
            else:
                servo_observer.advance(command[:, 2:])
            result = environment.advance(command, active)
            active &= result.valid
        hold_steps = round(
            experiment.convergence.hold_s * experiment.control_hz
        )
        return {
            "safe": active.cpu(),
            "converged": (active & (settled_count >= hold_steps)).cpu(),
            "cum_roll_pitch_error_rad_s": cum_rp.cpu(),
            "cum_rate_error_rad_s": cum_rate.cpu(),
            "cum_yaw_rate_error_rad_s": cum_yaw.cpu(),
            "peak_tilt_rad": peak_tilt.cpu(),
            "peak_rate_rad_s": peak_rate.cpu(),
            "final_attitude_error_rad": final_attitude_error.cpu(),
            "final_rate_rad_s": final_rate.cpu(),
            "saturation_fraction": (
                saturation_steps.to(dtype) / total_steps
            ).cpu(),
            "height_min_m": height_min.cpu(),
            "height_max_m": height_max.cpu(),
        }

    try:
        if scheme == "pid":
            controller = create_controller(
                load_controller_config(args.pid_config),
                ControllerContext(
                    batch_size=count,
                    device=device,
                    dtype=dtype,
                    control_dt=1.0 / experiment.control_hz,
                    parameters=nominal_parameters,
                ),
            )
            metrics = run_batch(
                controller,
                reference_collective=False,
                active=torch.ones(count, device=device, dtype=torch.bool),
                use_filter=False,
            )
        elif scheme in ("nominal_lqr", "oracle_lqr", "offline_lqr", "gru_lqr"):
            controller = create_controller(
                load_controller_config(args.controller_config),
                ControllerContext(
                    batch_size=count,
                    device=device,
                    dtype=dtype,
                    control_dt=1.0 / experiment.control_hz,
                    parameters=nominal_parameters,
                ),
            )
            if scheme == "oracle_lqr":
                current_gains = synthesize_gains(
                    oracle_coefficients,
                    nominal_effectiveness,
                    command_slopes,
                    nominal_tau,
                    mode_transform,
                    servo_slopes,
                    basis_tau,
                    q_composite,
                    r_composite,
                )
            else:
                current_gains = gains[scheme]
            current_gains = np.repeat(
                current_gains, args.initial_conditions_per_group, axis=0
            )
            controller._lqr_state_size = current_gains.shape[2]
            controller.schedule_lqr_gain(
                torch.as_tensor(
                    current_gains, device=device, dtype=dtype
                )
            )
            metrics = run_batch(
                controller,
                reference_collective=False,
                active=torch.ones(count, device=device, dtype=torch.bool),
                use_filter=True,
            )
        elif scheme in ("e2e_nn", "aa_nn"):
            from inference_package import (
                InferenceModelAdapter,
                load_flight_deploy_package,
            )

            bundle = (
                args.e2e_bundle if scheme == "e2e_nn" else args.aa_bundle
            )
            output_mode = (
                "residual_4" if scheme == "e2e_nn" else "coaxial_differential_cyclic_3"
            )
            package = load_flight_deploy_package(
                Path(bundle), device, torch.float32
            )
            adapter = InferenceModelAdapter(package)
            collected = {key: [] for key in (
                "safe", "converged", "cum_roll_pitch_error_rad_s",
                "cum_rate_error_rad_s", "cum_yaw_rate_error_rad_s",
                "peak_tilt_rad", "peak_rate_rad_s",
                "final_attitude_error_rad", "final_rate_rad_s",
                "saturation_fraction", "height_min_m", "height_max_m",
            )}
            for index in range(count):
                package.reset()
                one_params = {
                    name: value[index : index + 1].clone()
                    for name, value in actual_parameters.items()
                }
                controller = create_controller(
                    {
                        "type": "neural",
                        "params": {"output_mode": output_mode},
                    },
                    ControllerContext(
                        batch_size=1,
                        device=device,
                        dtype=torch.float32,
                        control_dt=1.0 / experiment.control_hz,
                        parameters={
                            name: value[index : index + 1].clone()
                            for name, value in nominal_parameters.items()
                        },
                    ),
                    neural_model=adapter,
                )
                controller.reset(
                    torch.ones(1, device=device, dtype=torch.bool)
                )
                episode_observer = FirstOrderServoObserver(
                    nominal_parameters["servos.pwm_angle_table"][
                        index : index + 1
                    ],
                    nominal_parameters["servos.tau"][index : index + 1],
                    1.0 / experiment.control_hz,
                )
                episode_pilot = VirtualPilotHeightController(
                    pilot_config,
                    1,
                    device,
                    torch.float32,
                    1.0 / experiment.control_hz,
                )
                episode_pilot.reset(
                    torch.ones(1, device=device, dtype=torch.bool)
                )
                episode_initial = _expand_mapping(
                    nominal_materialized.initial_state, 1
                )
                episode_initial["attitude_q_wb"].copy_(
                    attitude[index : index + 1].to(device)
                )
                episode_initial["angular_velocity_b"].copy_(
                    angular_velocity[index : index + 1].to(device)
                )
                env = SimulationEnvironment(
                    replace(
                        nominal_materialized,
                        parameters=one_params,
                        initial_state=episode_initial,
                        sensor_state=nominal_materialized.sensor_state,
                    ),
                    1,
                    device,
                    torch.float32,
                    logging_enabled=False,
                )
                episode = _simulate_neural_episode(
                    args,
                    experiment,
                    env,
                    controller,
                    episode_pilot,
                    episode_observer,
                    device,
                    dtype,
                )
                for key in collected:
                    collected[key].append(episode[key])
            metrics = {
                key: (
                    torch.stack(values).squeeze(-1).cpu()
                    if values
                    else torch.tensor([])
                )
                for key, values in collected.items()
            }
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        return {
            key: value.numpy().tolist()
            for key, value in metrics.items()
        }
    finally:
        environment.close()


def _simulate_neural_episode(
    args: argparse.Namespace,
    experiment: Any,
    env: Any,
    controller: Any,
    pilot: Any,
    servo_observer: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    from dataclasses import replace as dataclass_replace

    from flight_controller import ControllerReference, ControllerState
    from flight_controller.math import quaternion_rotation_error, tilt_cosine

    total_steps = round(args.duration_s * experiment.control_hz)
    settled_count = torch.zeros(1, device=device, dtype=torch.int64)
    saturation_steps = torch.zeros(1, device=device, dtype=torch.int64)
    cum_rp = torch.zeros(1, device=device, dtype=torch.float32)
    cum_rate = torch.zeros(1, device=device, dtype=torch.float32)
    cum_yaw = torch.zeros(1, device=device, dtype=torch.float32)
    peak_tilt = torch.zeros(1, device=device, dtype=torch.float32)
    peak_rate = torch.zeros(1, device=device, dtype=torch.float32)
    height_min = torch.full((1,), torch.inf, device=device, dtype=torch.float32)
    height_max = torch.full((1,), -torch.inf, device=device, dtype=torch.float32)
    active = torch.ones(1, device=device, dtype=torch.bool)
    final_attitude_error = torch.full(
        (1,), torch.inf, device=device, dtype=torch.float32
    )
    final_rate = torch.full_like(final_attitude_error, torch.inf)
    identity = torch.zeros(1, 4, device=device, dtype=torch.float32)
    identity[:, 0] = 1.0
    zeros3 = torch.zeros(1, 3, device=device, dtype=torch.float32)
    base_reference = ControllerReference(
        target_position_n=zeros3,
        target_velocity_n=zeros3,
        target_attitude_q_wb=identity,
        target_angular_velocity_b=zeros3,
        collective_command=torch.zeros(
            1, 1, device=device, dtype=torch.float32
        ),
    )
    for _ in range(total_steps):
        truth = env.observe("truth").values
        sensors = env.observe("sensor").values
        tilt_attitude = _yaw_free_attitude(truth["attitude_q_wb"])
        tilt = torch.acos(tilt_cosine(truth["attitude_q_wb"]))
        rate_norm = truth["angular_velocity_b"].norm(dim=1)
        finite = (
            torch.isfinite(tilt_attitude).all(dim=1)
            & torch.isfinite(sensors["gyro"]).all(dim=1)
            & torch.isfinite(sensors["motor_speed"]).all(dim=1)
        )
        active &= (
            finite
            & (tilt <= experiment.convergence.safety_tilt_rad)
            & (rate_norm <= experiment.convergence.safety_angular_rate_rad_s)
        )
        height = -truth["position_n"][:, 2]
        height_min = torch.minimum(height_min, height)
        height_max = torch.maximum(height_max, height)
        reference = dataclass_replace(
            base_reference,
            collective_command=pilot.step(
                height, -truth["velocity_n"][:, 2], active
            ),
        )
        state = ControllerState(
            position_n=torch.nan_to_num(truth["position_n"]),
            velocity_n=torch.nan_to_num(truth["velocity_n"]),
            attitude_q_wb=torch.nan_to_num(tilt_attitude),
            angular_velocity_b=torch.nan_to_num(sensors["gyro"]),
            linear_acceleration_n=torch.nan_to_num(
                truth["linear_acceleration_n"]
            ),
            motor_speed=torch.nan_to_num(sensors["motor_speed"]),
            servo_angle=servo_observer.angle,
        )
        output = controller.step(
            state, reference, active
        )
        attitude_error = quaternion_rotation_error(
            torch.nan_to_num(tilt_attitude), identity
        )[:, :2]
        rp_error = attitude_error.abs().sum(dim=1)
        cum_rp += rp_error * (1.0 / experiment.control_hz)
        cum_rate += rate_norm * (1.0 / experiment.control_hz)
        cum_yaw += sensors["gyro"][:, 2].abs() * (1.0 / experiment.control_hz)
        peak_tilt = torch.maximum(peak_tilt, tilt)
        peak_rate = torch.maximum(peak_rate, rate_norm)
        settled = active & (
            rp_error <= experiment.convergence.maximum_roll_pitch_error_rad
        ) & (
            rate_norm <= experiment.convergence.maximum_angular_rate_rad_s
        )
        settled_count = torch.where(
            settled, settled_count + 1, torch.zeros_like(settled_count)
        )
        final_attitude_error = torch.where(
            active, rp_error, final_attitude_error
        )
        final_rate = torch.where(active, rate_norm, final_rate)
        command = output.command
        saturated = active & (
            (command[:, :2] <= 1e-6).any(dim=1)
            | (command[:, :2] >= 1.0 - 1e-6).any(dim=1)
            | (command[:, 2:].abs() >= 1.0 - 1e-6).any(dim=1)
        )
        saturation_steps += saturated.to(torch.int64)
        servo_observer.advance(command[:, 2:])
        env.advance(command, active)
    hold_steps = round(experiment.convergence.hold_s * experiment.control_hz)
    return {
        "safe": active.cpu(),
        "converged": (active & (settled_count >= hold_steps)).cpu(),
        "cum_roll_pitch_error_rad_s": cum_rp.cpu(),
        "cum_rate_error_rad_s": cum_rate.cpu(),
        "cum_yaw_rate_error_rad_s": cum_yaw.cpu(),
        "peak_tilt_rad": peak_tilt.cpu(),
        "peak_rate_rad_s": peak_rate.cpu(),
        "final_attitude_error_rad": final_attitude_error.cpu(),
        "final_rate_rad_s": final_rate.cpu(),
        "saturation_fraction": (saturation_steps.to(torch.float32) / total_steps).cpu(),
        "height_min_m": height_min.cpu(),
        "height_max_m": height_max.cpu(),
    }


def _yaw_free_attitude(quaternion: torch.Tensor) -> torch.Tensor:
    from .experiment import _yaw_free_attitude as _inner

    return _inner(quaternion)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from .experiment import _sample_initial_state
    from .repeated_trial_evaluation import _sample_near_equilibrium

    experiment = load_experiment_config(args.experiment_config)
    device = torch.device(args.device)
    dtype = torch.float32 if experiment.dtype == "float32" else torch.float64
    actual_parameters, split, _, _ = build_base_vehicles(args)
    count = args.n_groups
    nominal_effectiveness, command_slopes, nominal_tau = _nominal_actuator_model(
        experiment.simulator_config
    )
    mode_transform = servo_mode_transform(
        nominal_effectiveness, command_slopes[2:]
    )
    basis_tau = tuple(DEFAULT_BASIS_TIME_CONSTANTS_S)
    servo_slopes = command_slopes[2:]
    from simenv.config import load_and_materialize

    nominal_materialized = load_and_materialize(
        experiment.simulator_config, 1, torch.device("cpu"), torch.float64
    )
    nominal_target = _sim2real_lqr_targets(nominal_materialized.parameters)
    nominal_coefficients = fit_composite_coefficients(
        nominal_target, mode_transform, servo_slopes, basis_tau
    ).numpy()[0]
    q_composite, r_composite = _lqr_weights_from_controller_config(
        args.controller_config, len(basis_tau)
    )
    split_group_id = split["group_id"]
    mlp_prediction, mlp_checkpoint = predict_step_response(
        args.mlp_checkpoint,
        Path(args.dataset),
        count,
        device,
        args.batch_size,
        8,
        split_group_id,
    )
    tcn_prediction, _ = predict_step_response(
        args.tcn_checkpoint,
        Path(args.dataset),
        count,
        device,
        args.batch_size,
        8,
        split_group_id,
    )
    gru_prediction, _ = predict_step_response(
        args.gru_checkpoint,
        Path(args.dataset),
        count,
        device,
        args.batch_size,
        8,
        split_group_id,
    )
    mode_indices = tuple(
        int(value) for value in mlp_checkpoint["adaptive_mode_indices"]
    )
    snapshot_times = tuple(
        float(value)
        for value in mlp_checkpoint.get(
            "response_snapshot_times_s", DEFAULT_RESPONSE_SNAPSHOT_TIMES_S
        )
    )
    merged_prediction = merge_axis_predictions(
        {
            "mlp": mlp_prediction,
            "tcn": tcn_prediction,
            "gru": gru_prediction,
        },
        tuple(mlp_checkpoint["label_names"]),
    )

    def coefficients_from_prediction(prediction: np.ndarray) -> np.ndarray:
        fitted = fit_coefficients_from_step_response(
            torch.as_tensor(prediction).reshape(
                count, len(snapshot_times), 3, len(mode_indices)
            ),
            snapshot_times,
            float(np.mean(servo_slopes)),
            basis_tau,
        ).numpy().reshape(count, -1)
        return merge_adaptive_composite_coefficients(
            fitted, nominal_coefficients, mode_indices
        )

    offline_coefficients = coefficients_from_prediction(merged_prediction)
    gru_coefficients = coefficients_from_prediction(gru_prediction)
    nominal_gain = synthesize_gains(
        np.broadcast_to(
            nominal_coefficients, (count, *nominal_coefficients.shape)
        ),
        nominal_effectiveness,
        command_slopes,
        nominal_tau,
        mode_transform,
        servo_slopes,
        basis_tau,
        q_composite,
        r_composite,
    )
    offline_gain_full = synthesize_gains(
        offline_coefficients,
        nominal_effectiveness,
        command_slopes,
        nominal_tau,
        mode_transform,
        servo_slopes,
        basis_tau,
        q_composite,
        r_composite,
    )
    # Deployment recommendation: 80% blend of the three-way hybrid gain.
    offline_gain = nominal_gain + 0.8 * (
        offline_gain_full - nominal_gain
    )
    gru_gain = synthesize_gains(
        gru_coefficients,
        nominal_effectiveness,
        command_slopes,
        nominal_tau,
        mode_transform,
        servo_slopes,
        basis_tau,
        q_composite,
        r_composite,
    )
    gains = {
        "nominal_lqr": nominal_gain,
        "offline_lqr": offline_gain,
        "gru_lqr": gru_gain,
    }
    # oracle gains are recomputed per error cell from the perturbed true targets
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    repeats = args.initial_conditions_per_group
    physical = split["labels"].repeat_interleave(repeats, dim=0).to(device)
    attitude, angular_velocity = _sample_near_equilibrium(
        len(physical),
        args.maximum_initial_tilt_rad,
        args.maximum_initial_rate_rad_s,
        generator,
        torch.float32,
    )
    cells = error_cells()
    if args.cell_names:
        wanted = set(args.cell_names.split(","))
        cells = [cell for cell in cells if cell["name"] in wanted]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"cells": [cell["name"] for cell in cells], "schemes": list(SCHEMES)}
    for cell in cells:
        perturbed = {
            name: value.clone()
            for name, value in actual_parameters.items()
        }
        perturbed = {
            name: value.repeat_interleave(repeats, dim=0)
            for name, value in perturbed.items()
        }
        if cell["fields"]:
            apply_error(perturbed, cell["fields"], cell["strengths"][0])
        group_params = {
            name: value[::repeats] for name, value in perturbed.items()
        }
        true_coefficients = np.zeros(
            (count, *nominal_coefficients.shape)
        )
        infeasible = 0
        for group_index in range(count):
            one = {
                name: value[group_index : group_index + 1]
                for name, value in group_params.items()
            }
            try:
                target = _sim2real_lqr_targets(one).cpu()
                true_coefficients[group_index] = fit_composite_coefficients(
                    target,
                    mode_transform,
                    servo_slopes,
                    basis_tau,
                ).numpy()[0]
            except ValueError:
                # Physically infeasible hover under the perturbation: no
                # truth gain exists, fall back to the nominal coefficients.
                true_coefficients[group_index] = nominal_coefficients
                infeasible += 1
        if infeasible:
            print(cell["name"], "infeasible oracle groups:", infeasible)
        for scheme in SCHEMES:
            if args.schemes and scheme not in args.schemes.split(","):
                continue
            target = output_dir / f"{cell['name']}__{scheme}.json"
            if target.exists() and not args.force:
                continue
            metrics = simulate_cell(
                args,
                cell,
                scheme,
                perturbed,
                attitude,
                angular_velocity,
                gains,
                true_coefficients,
                nominal_effectiveness,
                command_slopes,
                nominal_tau,
                mode_transform,
                servo_slopes,
                basis_tau,
                q_composite,
                r_composite,
            )
            target.write_text(
                json.dumps(
                    {
                        "cell": cell["name"],
                        "scheme": scheme,
                        "metrics": metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(cell["name"], scheme, "done")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robustness benchmark across controller schemes"
    )
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--controller-config", required=True)
    parser.add_argument("--pid-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mlp-checkpoint", required=True)
    parser.add_argument("--tcn-checkpoint", required=True)
    parser.add_argument("--gru-checkpoint", required=True)
    parser.add_argument("--e2e-bundle", required=True)
    parser.add_argument("--aa-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-groups", type=int, default=16)
    parser.add_argument("--initial-conditions-per-group", type=int, default=2)
    parser.add_argument("--maximum-initial-tilt-rad", type=float, default=0.2617994)
    parser.add_argument("--maximum-initial-rate-rad-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--schemes")
    parser.add_argument("--cell-names")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
