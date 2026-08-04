from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import IdentificationExperimentConfig, load_experiment_config
from .control_evaluation import (
    _closed_loop_result,
    _command_slopes_for_thrust_scale,
    _discrete_model,
    _lqr_gain,
    _nominal_actuator_model,
    _scaled_actuator_model,
    _summary,
)
from .experiment import (
    CommandDrivenServoObserver,
    _apply_effectiveness_labels,
    _expand_mapping,
    _set_initial_observation_state,
    _yaw_free_attitude,
)
from .gain_training import _lqr_weights
from .repeated_trial_training import (
    TrialSetIdentifier,
    load_repeated_split,
    prepare_histories,
)
from .training import effective_lqr_labels


def _predict_effective(
    checkpoint: Mapping[str, Any],
    split: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    trial_count: int,
) -> torch.Tensor:
    histories = prepare_histories(split, checkpoint["normalization"])
    model = TrialSetIdentifier(
        int(checkpoint["history_steps"]),
        int(checkpoint["feature_count"]),
        len(checkpoint["label_names"]),
        tuple(checkpoint["trial_hidden_sizes"]),
        tuple(checkpoint["head_hidden_sizes"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(histories), batch_size):
            value = histories[start : start + batch_size].to(device)
            trial_mask = torch.zeros(
                len(value), value.shape[1], device=device, dtype=torch.bool
            )
            trial_mask[:, :trial_count] = True
            normalized = model(value, trial_mask).cpu()
            predictions.append(
                normalized * checkpoint["normalization"]["label_std"]
                + checkpoint["normalization"]["label_mean"]
            )
    return torch.cat(predictions)


def _gain_variants(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    nominal_effectiveness: np.ndarray,
    nominal_command_slopes: np.ndarray,
    predicted_command_slopes: np.ndarray,
    true_command_slopes: np.ndarray,
    nominal_tau: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, list[float]]]:
    nominal_a, nominal_b = _discrete_model(
        nominal_effectiveness, nominal_command_slopes, nominal_tau, 1.0 / 500.0
    )
    nominal_gain = _lqr_gain(nominal_a, nominal_b, q, r)
    names = (
        "nominal",
        "nominal_nonlinear_observer",
        "predicted_blend_50",
        "predicted",
        "predicted_first_order_observer",
        "predicted_nominal_observer",
        "predicted_nominal_servo_tau",
        "predicted_nominal_all_tau",
        "oracle_authority_nominal_servo_tau",
        "oracle_first_order_observer",
        "oracle_nominal_observer",
        "oracle",
    )
    gains = {name: [] for name in names}
    invalid = {name: 0 for name in names}
    radii = {name: [] for name in names}
    initial_covariance = np.linalg.inv(q)
    for sample_index, (prediction, true_label) in enumerate(
        zip(predictions, true_labels)
    ):
        clipped = prediction.copy()
        clipped[:9] = np.clip(clipped[:9], -6.0, 6.0)
        clipped[9:] = np.clip(clipped[9:], -2.0, 2.0)
        predicted_core = clipped[:14]
        true_core = true_label[:14]
        model_labels = {
            "predicted": predicted_core,
            "predicted_nominal_servo_tau": np.concatenate((predicted_core[:11], np.zeros(3))),
            "predicted_nominal_all_tau": np.concatenate((predicted_core[:9], np.zeros(5))),
            "oracle_authority_nominal_servo_tau": np.concatenate(
                (true_core[:11], np.zeros(3))
            ),
            "oracle": true_core,
        }
        current = {"nominal": nominal_gain}
        for name, label in model_labels.items():
            try:
                effectiveness, tau = _scaled_actuator_model(
                    label, nominal_effectiveness, nominal_tau
                )
                slopes = (
                    true_command_slopes[sample_index]
                    if name.startswith("oracle")
                    else predicted_command_slopes[sample_index]
                )
                a, b = _discrete_model(effectiveness, slopes, tau, 1.0 / 500.0)
                gain = _lqr_gain(a, b, q, r)
                if not np.isfinite(gain).all():
                    raise ValueError("non-finite gain")
                current[name] = gain
            except (np.linalg.LinAlgError, ValueError):
                current[name] = nominal_gain
                invalid[name] += 1
        current["predicted_blend_50"] = nominal_gain + 0.5 * (
            current["predicted"] - nominal_gain
        )
        current["predicted_nominal_observer"] = current["predicted"]
        current["nominal_nonlinear_observer"] = nominal_gain
        current["predicted_first_order_observer"] = current["predicted"]
        current["oracle_nominal_observer"] = current["oracle"]
        current["oracle_first_order_observer"] = current["oracle"]
        true_effectiveness, true_tau = _scaled_actuator_model(
            true_core, nominal_effectiveness, nominal_tau
        )
        true_a, true_b = _discrete_model(
            true_effectiveness,
            true_command_slopes[sample_index],
            true_tau,
            1.0 / 500.0,
        )
        for name in names:
            gains[name].append(current[name])
            radius, _ = _closed_loop_result(
                true_a, true_b, current[name], q, r, initial_covariance
            )
            radii[name].append(radius)
    return (
        {name: np.stack(values) for name, values in gains.items()},
        invalid,
        radii,
    )


def _sample_near_equilibrium(
    count: int,
    maximum_tilt_rad: float,
    maximum_rate_rad_s: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    angle = torch.rand(count, generator=generator, dtype=dtype) * maximum_tilt_rad
    direction = torch.rand(count, generator=generator, dtype=dtype) * (2.0 * math.pi)
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
    rate = (
        2.0 * torch.rand(count, 3, generator=generator, dtype=dtype) - 1.0
    ) * maximum_rate_rad_s
    return attitude, rate


@torch.no_grad()
def _simulate_gain(
    experiment: IdentificationExperimentConfig,
    physical_labels: torch.Tensor,
    attitude: torch.Tensor,
    angular_velocity: torch.Tensor,
    gain: np.ndarray,
    observer_servo_tau: np.ndarray,
    nonlinear_servo_observer: np.ndarray,
    duration_s: float,
) -> dict[str, torch.Tensor]:
    from flight_controller import (
        ControllerContext,
        ControllerReference,
        ControllerState,
        create_controller,
        load_controller_config,
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
    nominal_parameters = _expand_mapping(nominal_materialized.parameters, count)
    actual_parameters = _apply_effectiveness_labels(
        nominal_parameters,
        physical_labels.to(device),
        experiment.parameterization,
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
    try:
        controller = create_controller(
            load_controller_config(experiment.controller_config),
            ControllerContext(
                batch_size=count,
                device=device,
                dtype=dtype,
                control_dt=1.0 / experiment.control_hz,
                parameters=nominal_parameters,
            ),
        )
        controller.schedule_lqr_gain(
            torch.as_tensor(gain, device=device, dtype=dtype)
        )
        _set_initial_observation_state(
            environment,
            (
                LocalPlantModel(actual_parameters).hover_trim().motor_speed
                if experiment.parameterization == "sim2real_micro"
                else controller.trim.motor_speed
            ),
            initial_state["angular_velocity_b"],
        )
        observer_tau_tensor = torch.as_tensor(
            observer_servo_tau, device=device, dtype=dtype
        )
        nonlinear_mask = torch.as_tensor(
            nonlinear_servo_observer, device=device, dtype=torch.bool
        )[:, None]
        servo_observer = CommandDrivenServoObserver(
            nominal_parameters["servos.pwm_angle_table"],
            observer_tau_tensor,
            torch.where(
                nonlinear_mask,
                nominal_parameters["servos.max_speed"],
                torch.full_like(observer_tau_tensor, 1e12),
            ),
            torch.where(
                nonlinear_mask,
                nominal_parameters["servos.backlash"],
                torch.zeros_like(observer_tau_tensor),
            ),
            torch.where(
                nonlinear_mask,
                nominal_parameters["servos.deadzone"],
                torch.zeros_like(observer_tau_tensor),
            ),
            1.0 / experiment.control_hz,
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
            safe_tilt = torch.nan_to_num(tilt_attitude)
            safe_rate = torch.nan_to_num(measured_rate)
            safe_motor = torch.nan_to_num(measured_motor)
            state = ControllerState(
                position_n=torch.nan_to_num(truth["position_n"]),
                velocity_n=torch.nan_to_num(truth["velocity_n"]),
                attitude_q_wb=safe_tilt,
                angular_velocity_b=safe_rate,
                linear_acceleration_n=torch.nan_to_num(
                    truth["linear_acceleration_n"]
                ),
                motor_speed=safe_motor,
                servo_angle=servo_observer.angle,
            )
            output = controller.step(state, reference, active)
            attitude_error = quaternion_rotation_error(
                safe_tilt, identity
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
            command = output.command
            saturated = active & (
                (command[:, :2] <= 1e-6).any(dim=1)
                | (command[:, :2] >= 1.0 - 1e-6).any(dim=1)
                | (command[:, 2:].abs() >= 1.0 - 1e-6).any(dim=1)
            )
            saturation_steps += saturated.to(torch.int64)
            servo_observer.advance(command[:, 2:])
            result = environment.advance(command, active)
            active &= result.valid
        hold_steps = round(experiment.convergence.hold_s * experiment.control_hz)
        return {
            "safe": active.cpu(),
            "converged": (active & (settled_count >= hold_steps)).cpu(),
            "final_attitude_error": final_attitude_error.cpu(),
            "final_rate": final_rate.cpu(),
            "saturation_fraction": (
                saturation_steps.to(dtype) / total_steps
            ).cpu(),
        }
    finally:
        environment.close()


def _nonlinear_summary(result: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    safe = result["safe"]
    finite_attitude = result["final_attitude_error"][safe]
    finite_rate = result["final_rate"][safe]
    return {
        "safe_fraction": float(safe.to(torch.float32).mean()),
        "converged_fraction": float(result["converged"].to(torch.float32).mean()),
        "converged_fraction_given_safe": float(
            result["converged"][safe].to(torch.float32).mean()
        )
        if bool(safe.any())
        else 0.0,
        "final_attitude_error_rad_median": float(finite_attitude.median())
        if len(finite_attitude)
        else None,
        "final_rate_rad_s_median": float(finite_rate.median())
        if len(finite_rate)
        else None,
        "saturation_fraction_mean": float(result["saturation_fraction"].mean()),
    }


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_type") != "repeated_trial_lqr_effective_identifier":
        raise ValueError("expected a repeated-trial identifier checkpoint")
    dataset = Path(args.dataset).expanduser().resolve()
    split = load_repeated_split(dataset, "test", int(checkpoint["downsample"]))
    physical_labels = split["labels"].clone()
    parameterization = str(split["parameterization"])
    checkpoint_parameterization = str(
        checkpoint.get("parameterization", "legacy_collective")
    )
    if parameterization != checkpoint_parameterization:
        raise ValueError("checkpoint and dataset parameterization differ")
    true_effective = effective_lqr_labels(physical_labels, parameterization)
    trial_count = min(args.trial_count, int(checkpoint["trials_per_group"]))
    predictions = _predict_effective(
        checkpoint,
        split,
        torch.device(args.device),
        args.batch_size,
        trial_count,
    )
    experiment = load_experiment_config(args.experiment_config)
    nominal_effectiveness, command_slopes, nominal_tau = _nominal_actuator_model(
        experiment.simulator_config
    )
    if parameterization == "empirical_core":
        predicted_thrust_scale = np.power(10.0, predictions.numpy()[:, 14])
        true_thrust_scale = np.power(10.0, true_effective.numpy()[:, 14])
        predicted_command_slopes = _command_slopes_for_thrust_scale(
            experiment.simulator_config, predicted_thrust_scale
        )
        true_command_slopes = _command_slopes_for_thrust_scale(
            experiment.simulator_config, true_thrust_scale
        )
    else:
        predicted_command_slopes = np.broadcast_to(
            command_slopes, (len(predictions), 5)
        )
        true_command_slopes = predicted_command_slopes
    q, r = _lqr_weights(experiment.controller_config)
    gains, invalid, radii = _gain_variants(
        predictions.numpy(),
        true_effective.numpy(),
        nominal_effectiveness,
        command_slopes,
        predicted_command_slopes,
        true_command_slopes,
        nominal_tau,
        q,
        r,
    )
    local_stability = {
        name: float(np.mean(np.asarray(values) < 1.0))
        for name, values in radii.items()
    }
    local_radius = {name: _summary(values) for name, values in radii.items()}

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    repeats = args.evaluation_initial_conditions
    repeated_physical = physical_labels.repeat_interleave(repeats, dim=0)
    attitude, angular_velocity = _sample_near_equilibrium(
        len(repeated_physical),
        args.maximum_initial_tilt_rad,
        args.maximum_initial_rate_rad_s,
        generator,
        torch.float32,
    )
    available_nonlinear_names = (
        "nominal",
        "nominal_nonlinear_observer",
        "predicted",
        "predicted_first_order_observer",
        "predicted_nominal_observer",
        "predicted_nominal_servo_tau",
        "predicted_nominal_all_tau",
        "oracle_authority_nominal_servo_tau",
        "oracle_first_order_observer",
        "oracle_nominal_observer",
        "oracle",
    )
    nonlinear_names = tuple(
        name.strip() for name in args.variants.split(",") if name.strip()
    )
    unknown_variants = set(nonlinear_names) - set(available_nonlinear_names)
    if not nonlinear_names or unknown_variants:
        raise ValueError(
            "variants must be a non-empty comma-separated subset of "
            f"{available_nonlinear_names}; unknown={sorted(unknown_variants)}"
        )
    clipped_predictions = predictions.numpy().copy()
    clipped_predictions[:, :9] = np.clip(clipped_predictions[:, :9], -6.0, 6.0)
    clipped_predictions[:, 9:] = np.clip(clipped_predictions[:, 9:], -2.0, 2.0)
    nominal_servo_tau = np.broadcast_to(
        nominal_tau[2:], (len(physical_labels), 3)
    ).copy()
    predicted_servo_tau = nominal_servo_tau * np.power(
        10.0, clipped_predictions[:, 11:14]
    )
    oracle_servo_tau = nominal_servo_tau * np.power(
        10.0, true_effective.numpy()[:, 11:14]
    )
    observer_tau = {
        "nominal": nominal_servo_tau,
        "nominal_nonlinear_observer": nominal_servo_tau,
        "predicted": predicted_servo_tau,
        "predicted_first_order_observer": predicted_servo_tau,
        "predicted_nominal_observer": nominal_servo_tau,
        "predicted_nominal_servo_tau": nominal_servo_tau,
        "predicted_nominal_all_tau": nominal_servo_tau,
        "oracle_authority_nominal_servo_tau": nominal_servo_tau,
        "oracle_first_order_observer": oracle_servo_tau,
        "oracle_nominal_observer": nominal_servo_tau,
        "oracle": oracle_servo_tau,
    }
    nonlinear_observer = {
        "nominal": False,
        "nominal_nonlinear_observer": True,
        "predicted": True,
        "predicted_first_order_observer": False,
        "predicted_nominal_observer": False,
        "predicted_nominal_servo_tau": True,
        "predicted_nominal_all_tau": True,
        "oracle_authority_nominal_servo_tau": True,
        "oracle_first_order_observer": False,
        "oracle_nominal_observer": False,
        "oracle": True,
    }
    evaluation_count = len(repeated_physical)
    combined = _simulate_gain(
        experiment,
        repeated_physical.repeat((len(nonlinear_names), 1)),
        attitude.repeat((len(nonlinear_names), 1)),
        angular_velocity.repeat((len(nonlinear_names), 1)),
        np.concatenate(
            [np.repeat(gains[name], repeats, axis=0) for name in nonlinear_names]
        ),
        np.concatenate(
            [
                np.repeat(observer_tau[name], repeats, axis=0)
                for name in nonlinear_names
            ]
        ),
        np.concatenate(
            [
                np.full(
                    evaluation_count,
                    nonlinear_observer[name],
                    dtype=np.bool_,
                )
                for name in nonlinear_names
            ]
        ),
        args.duration_s,
    )
    nonlinear = {
        name: _nonlinear_summary(
            {
                key: value[index * evaluation_count : (index + 1) * evaluation_count]
                for key, value in combined.items()
            }
        )
        for index, name in enumerate(nonlinear_names)
    }
    report = {
        "schema_version": 1,
        "semantics": (
            "Identification uses repeated destructive trials. Evaluation uses new "
            "held-out initial conditions, with the identified gain "
            "installed before the first control step."
        ),
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset),
        "parameterization": parameterization,
        "test_parameter_groups": len(physical_labels),
        "identification_trial_count": trial_count,
        "evaluation_initial_conditions_per_group": repeats,
        "maximum_initial_tilt_rad": args.maximum_initial_tilt_rad,
        "maximum_initial_rate_rad_s": args.maximum_initial_rate_rad_s,
        "duration_s": args.duration_s,
        "nonlinear_variants": nonlinear_names,
        "servo_observer_semantics": {
            "nominal": "current first-order command observer",
            "predicted_or_oracle": (
                "command-driven observer with known deadzone, backlash and rate "
                "limit plus the corresponding identified time constant"
            ),
        },
        "invalid_gain_count": invalid,
        "local_linear_stable_fraction": local_stability,
        "local_linear_pole_radius": local_radius,
        "nonlinear": nonlinear,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate repeated-trial identification near equilibrium"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trial-count", type=int, default=8)
    parser.add_argument("--evaluation-initial-conditions", type=int, default=2)
    parser.add_argument("--maximum-initial-tilt-rad", type=float, default=0.0523599)
    parser.add_argument("--maximum-initial-rate-rad-s", type=float, default=0.1)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument(
        "--variants",
        default=(
            "nominal,predicted,predicted_first_order_observer,"
            "predicted_nominal_observer,"
            "predicted_nominal_servo_tau,predicted_nominal_all_tau,"
            "oracle_authority_nominal_servo_tau,oracle_first_order_observer,"
            "oracle_nominal_observer,oracle"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--experiment-config",
        default="Identification/configs/lqr_repeated_trials_wide_v1.yaml",
    )
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
