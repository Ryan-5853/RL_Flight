from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import IdentificationExperimentConfig, load_experiment_config
from .control_evaluation import _closed_loop_result, _lqr_gain, _nominal_actuator_model
from .experiment import (
    _apply_effectiveness_labels,
    _expand_mapping,
    _set_initial_observation_state,
    _yaw_free_attitude,
)
from .gain_training import _lqr_weights
from .offline_models import build_offline_identifier
from .repeated_trial_deployment import RepeatedTrialLQRScheduler
from .repeated_trial_evaluation import (
    _nonlinear_summary,
    _sample_near_equilibrium,
    _simulate_gain,
)
from .repeated_trial_training import (
    _fixed_trial_mask,
    _retain_converged_trials,
    load_repeated_split,
    prepare_histories,
)
from .sim2real_composite import (
    DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
    composite_discrete_model,
    composite_lqr_weights,
    fit_composite_coefficients,
    fit_coefficients_from_step_response,
    merge_adaptive_composite_coefficients,
    reconstruct_composite_step_response,
    true_servo_mode_step_response,
)
from .sim2real_control_evaluation import (
    _cluster_bootstrap_mean_ci,
    _model_from_target,
    _paired_binary_summary,
)


@torch.no_grad()
def _predict_composite(
    checkpoint: Mapping[str, Any],
    split: Mapping[str, Any],
    device: torch.device,
    trial_count: int,
    batch_size: int,
) -> torch.Tensor:
    histories = prepare_histories(split, checkpoint["normalization"])
    model = build_offline_identifier(
        str(checkpoint.get("architecture", "mlp")),
        int(checkpoint["history_steps"]),
        int(checkpoint["feature_count"]),
        len(checkpoint["label_names"]),
        tuple(checkpoint["trial_hidden_sizes"]),
        tuple(checkpoint["head_hidden_sizes"]),
        tuple(checkpoint.get("temporal_channels", (64, 96, 128))),
        int(checkpoint.get("recurrent_hidden_size", 96)),
        int(checkpoint.get("recurrent_layers", 2)),
        int(checkpoint.get("trial_embedding_size", 128)),
        float(checkpoint.get("temporal_dropout", 0.05)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    output = []
    for start in range(0, len(histories), batch_size):
        value = histories[start : start + batch_size].to(device)
        mask = _fixed_trial_mask(
            split["trial_mask"][start : start + batch_size].to(device), trial_count
        )
        output.append(model(value, mask).cpu())
    normalized = torch.cat(output)
    return (
        normalized * checkpoint["normalization"]["label_std"]
        + checkpoint["normalization"]["label_mean"]
    )


def _response_nrmse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (predicted - target).square().mean().sqrt()
        / target.square().mean().sqrt().clamp_min(1e-12)
    )


def _parameter_stratified_convergence(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    physical_labels: torch.Tensor,
    label_names: Sequence[str],
    group_count: int,
    repeats: int,
) -> Mapping[str, Any]:
    group_delta = (
        candidate["converged"].to(torch.float32)
        - reference["converged"].to(torch.float32)
    ).reshape(group_count, repeats).mean(dim=1)
    labels = physical_labels.reshape(group_count, repeats, -1)[:, 0].to(torch.float32)
    centered_delta = group_delta - group_delta.mean()
    entries = []
    for index, name in enumerate(label_names):
        value = labels[:, index]
        centered = value - value.mean()
        denominator = torch.sqrt(
            centered.square().sum() * centered_delta.square().sum()
        ).clamp_min(1e-12)
        correlation = float((centered * centered_delta).sum() / denominator)
        lower = value <= torch.quantile(value, 0.25)
        upper = value >= torch.quantile(value, 0.75)
        entries.append(
            {
                "name": name,
                "pearson_correlation_with_group_convergence_delta": correlation,
                "lowest_quartile_mean_delta": float(group_delta[lower].mean()),
                "highest_quartile_mean_delta": float(group_delta[upper].mean()),
            }
        )
    entries.sort(
        key=lambda item: abs(
            item["pearson_correlation_with_group_convergence_delta"]
        ),
        reverse=True,
    )
    return {
        "improved_group_fraction": float((group_delta > 0).to(torch.float32).mean()),
        "degraded_group_fraction": float((group_delta < 0).to(torch.float32).mean()),
        "unchanged_group_fraction": float((group_delta == 0).to(torch.float32).mean()),
        "top_parameter_associations": entries[:10],
    }


def _legacy_predicted_targets(
    scheduler: RepeatedTrialLQRScheduler,
    synthesis: Any,
    nominal_target: torch.Tensor,
    full_names: tuple[str, ...],
) -> torch.Tensor:
    prediction = synthesis.effective_log10.to(torch.float64).numpy()
    clipped = np.clip(
        prediction,
        scheduler.checkpoint["label_min"].numpy(),
        scheduler.checkpoint["label_max"].numpy(),
    )
    output = nominal_target.repeat(len(clipped), 1).clone()
    for source_index, name in enumerate(scheduler.label_names):
        if name in full_names:
            output[:, full_names.index(name)] = torch.from_numpy(
                clipped[:, source_index]
            )
    return output


def _composite_actual_radius(
    physical_a: np.ndarray,
    physical_b: np.ndarray,
    gain: np.ndarray,
    mode_transform: np.ndarray,
    servo_slopes: np.ndarray,
    basis_tau: Sequence[float],
    dt: float,
) -> float:
    latent_count = 3 * len(basis_tau)
    combined_a = np.zeros((13 + latent_count, 13 + latent_count), dtype=np.float64)
    combined_b = np.zeros((13 + latent_count, 5), dtype=np.float64)
    combined_a[:13, :13] = physical_a
    combined_b[:13] = physical_b
    servo_input = mode_transform @ np.diag(servo_slopes)
    for basis_index, tau in enumerate(basis_tau):
        start = 13 + 3 * basis_index
        decay = np.exp(-dt / tau)
        combined_a[start : start + 3, start : start + 3] = np.eye(3) * decay
        combined_b[start : start + 3, 2:] = (1.0 - decay) * servo_input
    selection = np.zeros((19, 13 + latent_count), dtype=np.float64)
    selection[:7, :7] = np.eye(7)
    selection[7:16, 13:] = np.eye(latent_count)
    selection[16:19, 10:13] = np.eye(3)
    closed = combined_a - combined_b @ gain @ selection
    return float(np.max(np.abs(np.linalg.eigvals(closed))))


class _FixedServoFilterBank:
    def __init__(
        self,
        count: int,
        mode_transform: np.ndarray,
        pwm_angle_table: torch.Tensor,
        backlash: torch.Tensor,
        deadzone: torch.Tensor,
        basis_tau: Sequence[float],
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.state = torch.zeros(
            count, len(basis_tau), 3, device=device, dtype=dtype
        )
        self.transform = torch.as_tensor(
            mode_transform, device=device, dtype=dtype
        )
        self.table = pwm_angle_table
        self.backlash = backlash
        self.deadzone = deadzone
        self.effective_pwm = torch.zeros(count, 3, device=device, dtype=dtype)
        self.command_angle = torch.zeros_like(self.effective_pwm)
        self.target_angle = torch.zeros_like(self.effective_pwm)
        self.motion_direction = torch.zeros_like(self.effective_pwm)
        self.backlash_remaining = torch.zeros_like(self.effective_pwm)
        self.response = 1.0 - torch.exp(
            -dt / torch.as_tensor(basis_tau, device=device, dtype=dtype)
        )

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
        direction = torch.sign(command_delta)
        reversing = (
            (direction != 0)
            & (self.motion_direction != 0)
            & (direction != self.motion_direction)
        )
        backlash_remaining = torch.where(
            reversing, self.backlash, self.backlash_remaining
        )
        consumed = torch.minimum(torch.abs(command_delta), backlash_remaining)
        self.target_angle.add_(
            direction * (torch.abs(command_delta) - consumed)
        )
        self.backlash_remaining.copy_(backlash_remaining - consumed)
        self.motion_direction.copy_(
            torch.where(direction != 0, direction, self.motion_direction)
        )
        self.effective_pwm.copy_(effective_pwm)
        self.command_angle.copy_(command_angle)
        target = self.target_angle @ self.transform.T
        self.state.add_(
            self.response[None, :, None] * (target[:, None, :] - self.state)
        )


@torch.no_grad()
def _simulate_composite_gain(
    experiment: IdentificationExperimentConfig,
    physical_labels: torch.Tensor,
    attitude: torch.Tensor,
    angular_velocity: torch.Tensor,
    gain: np.ndarray,
    mode_transform: np.ndarray,
    servo_slopes: np.ndarray,
    basis_tau: Sequence[float],
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
        # Isolated prototype: the standard controller accepts the 19-state gain
        # once its scheduled state contract is set to the command-filter state.
        controller._lqr_state_size = gain.shape[2]
        controller.schedule_lqr_gain(
            torch.as_tensor(gain, device=device, dtype=dtype)
        )
        _set_initial_observation_state(
            environment,
            LocalPlantModel(actual_parameters).hover_trim().motor_speed,
            initial_state["angular_velocity_b"],
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
            state = ControllerState(
                position_n=torch.nan_to_num(truth["position_n"]),
                velocity_n=torch.nan_to_num(truth["velocity_n"]),
                attitude_q_wb=safe_tilt,
                angular_velocity_b=safe_rate,
                linear_acceleration_n=torch.nan_to_num(
                    truth["linear_acceleration_n"]
                ),
                motor_speed=safe_motor,
                servo_angle=servo_filter.state.flatten(start_dim=1),
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
            servo_filter.advance(command[:, 2:])
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


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_type") != "sim2real_composite_servo_identifier":
        raise ValueError("expected a sim2real composite servo checkpoint")
    dataset = Path(args.dataset).expanduser().resolve()
    experiment = load_experiment_config(args.experiment_config)
    split = load_repeated_split(dataset, "test", int(checkpoint["downsample"]))
    _retain_converged_trials(split)
    count = min(args.maximum_parameter_groups, len(split["features"]))
    original_count = len(split["features"])
    for name, value in tuple(split.items()):
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == original_count:
            split[name] = value[:count]
    prediction = _predict_composite(
        checkpoint,
        split,
        torch.device(args.device),
        args.trial_count,
        args.batch_size,
    )
    prediction = torch.maximum(
        torch.minimum(prediction, checkpoint["label_max"]),
        checkpoint["label_min"],
    )
    secondary_checkpoint_path = None
    secondary_output_axes: tuple[str, ...] = ()
    if args.secondary_checkpoint:
        secondary_checkpoint_path = Path(args.secondary_checkpoint).expanduser().resolve()
        secondary = torch.load(
            secondary_checkpoint_path, map_location="cpu", weights_only=False
        )
        if secondary.get("artifact_type") != "sim2real_composite_servo_identifier":
            raise ValueError("secondary checkpoint must be a composite identifier")
        for name in (
            "label_names",
            "adaptive_mode_indices",
            "basis_time_constants_s",
            "response_snapshot_times_s",
        ):
            if tuple(secondary[name]) != tuple(checkpoint[name]):
                raise ValueError(f"secondary checkpoint differs in {name}")
        secondary_split = load_repeated_split(
            dataset, "test", int(secondary["downsample"])
        )
        _retain_converged_trials(secondary_split)
        secondary_original_count = len(secondary_split["features"])
        for name, value in tuple(secondary_split.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.ndim
                and len(value) == secondary_original_count
            ):
                secondary_split[name] = value[:count]
        if not torch.equal(split["group_id"], secondary_split["group_id"]):
            raise ValueError("primary and secondary checkpoint group ordering differs")
        secondary_prediction = _predict_composite(
            secondary,
            secondary_split,
            torch.device(args.device),
            args.trial_count,
            args.batch_size,
        )
        secondary_prediction = torch.maximum(
            torch.minimum(secondary_prediction, secondary["label_max"]),
            secondary["label_min"],
        )
        secondary_output_axes = tuple(
            value.strip() for value in args.secondary_output_axes.split(",") if value.strip()
        )
        if not secondary_output_axes or any(
            value not in {"roll", "pitch", "yaw"}
            for value in secondary_output_axes
        ):
            raise ValueError("secondary-output-axes must select roll, pitch, or yaw")
        selected_columns = [
            index
            for index, name in enumerate(checkpoint["label_names"])
            if any(f".{axis}." in name for axis in secondary_output_axes)
        ]
        prediction[:, selected_columns] = secondary_prediction[:, selected_columns]
    basis_tau = tuple(float(value) for value in checkpoint["basis_time_constants_s"])
    mode_indices = tuple(int(value) for value in checkpoint["adaptive_mode_indices"])
    mode_transform = checkpoint["mode_transform"].numpy()
    servo_slopes = checkpoint["servo_command_slopes"].numpy()
    nominal_coefficients = checkpoint["nominal_composite_coefficients"].numpy()
    true_coefficients = fit_composite_coefficients(
        split["targets"], mode_transform, servo_slopes, basis_tau
    ).numpy()
    target_representation = str(
        checkpoint.get("target_representation", "coefficients")
    )
    if target_representation == "coefficients":
        adaptive_prediction = prediction.numpy()
    else:
        snapshot_times = tuple(
            float(value)
            for value in checkpoint.get(
                "response_snapshot_times_s", DEFAULT_RESPONSE_SNAPSHOT_TIMES_S
            )
        )
        response_shape = (
            count,
            len(snapshot_times),
            3,
            len(mode_indices),
        )
        adaptive_prediction = fit_coefficients_from_step_response(
            prediction.reshape(response_shape),
            snapshot_times,
            float(np.mean(servo_slopes)),
            basis_tau,
        ).numpy().reshape(count, -1)
    predicted_full = merge_adaptive_composite_coefficients(
        adaptive_prediction, nominal_coefficients, mode_indices
    )
    oracle_selected = merge_adaptive_composite_coefficients(
        true_coefficients[..., list(mode_indices)].reshape(count, -1),
        nominal_coefficients,
        mode_indices,
    )

    true_response = true_servo_mode_step_response(
        split["targets"], mode_transform, servo_slopes
    )[..., list(mode_indices)]
    composite_response = reconstruct_composite_step_response(
        torch.from_numpy(predicted_full)
    )[..., list(mode_indices)]
    fixed_basis_oracle_response = reconstruct_composite_step_response(
        torch.from_numpy(true_coefficients)
    )[..., list(mode_indices)]
    nominal_response = reconstruct_composite_step_response(
        torch.from_numpy(nominal_coefficients[None])
    )[..., list(mode_indices)].expand_as(true_response)

    raw_split = load_repeated_split(dataset, "test", 1)
    _retain_converged_trials(raw_split)
    raw_count = len(raw_split["features"])
    for name, value in tuple(raw_split.items()):
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == raw_count:
            raw_split[name] = value[:count]
    legacy_scheduler = RepeatedTrialLQRScheduler(
        args.legacy_checkpoint,
        experiment.simulator_config,
        experiment.controller_config,
        args.device,
    )
    legacy_raw_steps = (
        legacy_scheduler.history_steps * legacy_scheduler.downsample
    )
    if raw_split["features"].shape[2] < legacy_raw_steps:
        raise ValueError(
            "evaluation histories are shorter than the legacy checkpoint contract"
        )
    legacy_synthesis = legacy_scheduler.synthesize(
        raw_split["features"][:, :, :legacy_raw_steps],
        raw_split["valid_mask"][:, :, :legacy_raw_steps],
        _fixed_trial_mask(raw_split["trial_mask"], args.trial_count),
    )
    from simenv.config import load_and_materialize
    from .experiment import _sim2real_lqr_targets

    nominal_materialized = load_and_materialize(
        experiment.simulator_config, 1, torch.device("cpu"), torch.float64
    )
    nominal_target = _sim2real_lqr_targets(nominal_materialized.parameters)
    legacy_targets = _legacy_predicted_targets(
        legacy_scheduler,
        legacy_synthesis,
        nominal_target,
        tuple(split["target_names"]),
    )
    legacy_response = true_servo_mode_step_response(
        legacy_targets, mode_transform, servo_slopes
    )[..., list(mode_indices)]
    legacy_coefficients = fit_composite_coefficients(
        legacy_targets, mode_transform, servo_slopes, basis_tau
    ).numpy()

    nominal_effectiveness, command_slopes, nominal_tau = _nominal_actuator_model(
        experiment.simulator_config
    )
    from flight_controller import load_controller_config

    lqr_config = load_controller_config(experiment.controller_config)["params"]["lqr"]
    q_composite, r_composite = composite_lqr_weights(
        np.asarray(lqr_config["state_scales"], dtype=np.float64),
        np.asarray(lqr_config["integral_state_scales"], dtype=np.float64),
        np.asarray(lqr_config["input_scales"], dtype=np.float64),
        float(lqr_config["input_weight_scale"]),
        len(basis_tau),
    )
    coefficient_variants = {
        "composite_nominal": np.broadcast_to(
            nominal_coefficients, (count, *nominal_coefficients.shape)
        ),
        "composite_predicted": predicted_full,
        "composite_legacy_physical_model": legacy_coefficients,
        "composite_oracle_all_modes": true_coefficients,
    }
    if set(mode_indices) != {0, 1, 2}:
        coefficient_variants["composite_oracle_selected_modes"] = oracle_selected
    composite_gains = {name: [] for name in coefficient_variants}
    for name, coefficients in coefficient_variants.items():
        for current in coefficients:
            a, b = composite_discrete_model(
                nominal_effectiveness[:, :2],
                command_slopes[:2],
                nominal_tau[:2],
                current,
                mode_transform,
                servo_slopes,
                basis_tau,
            )
            composite_gains[name].append(_lqr_gain(a, b, q_composite, r_composite))
        composite_gains[name] = np.stack(composite_gains[name])
    nominal_composite_gain = composite_gains["composite_nominal"]
    predicted_composite_gain = composite_gains["composite_predicted"]
    blend_fractions = tuple(
        float(value) for value in args.gain_blend_fractions.split(",")
    )
    if (
        not blend_fractions
        or any(not 0.0 < value < 1.0 for value in blend_fractions)
        or len(set(blend_fractions)) != len(blend_fractions)
        or len({round(100 * value) for value in blend_fractions})
        != len(blend_fractions)
    ):
        raise ValueError(
            "gain-blend-fractions must be unique percent-resolvable values in (0, 1)"
        )
    for fraction in blend_fractions:
        percentage = round(100 * fraction)
        composite_gains[f"composite_predicted_gain_blend_{percentage}"] = (
            nominal_composite_gain
            + fraction * (predicted_composite_gain - nominal_composite_gain)
        )

    q_original, r_original = _lqr_weights(experiment.controller_config)
    original_nominal_gain = legacy_synthesis.nominal_gain.numpy()
    original_oracle_gains = []
    true_models = []
    oracle_servo_tau = []
    for target in split["targets"].numpy():
        a, b, tau = _model_from_target(target, command_slopes[2:])
        true_models.append((a, b))
        original_oracle_gains.append(_lqr_gain(a, b, q_original, r_original))
        oracle_servo_tau.append(tau[2:])
    original_oracle_gains = np.stack(original_oracle_gains)
    oracle_servo_tau = np.stack(oracle_servo_tau)

    local_radius = {
        "original_nominal": [],
        "legacy_predicted": [],
        "original_oracle": [],
        **{name: [] for name in composite_gains},
    }
    initial_covariance = np.linalg.inv(q_original)
    for index, (physical_a, physical_b) in enumerate(true_models):
        for name, gain in (
            ("original_nominal", original_nominal_gain),
            ("legacy_predicted", legacy_synthesis.predicted_gain.numpy()[index]),
            ("original_oracle", original_oracle_gains[index]),
        ):
            radius, _ = _closed_loop_result(
                physical_a, physical_b, gain, q_original, r_original, initial_covariance
            )
            local_radius[name].append(radius)
        for name, gains in composite_gains.items():
            local_radius[name].append(
                _composite_actual_radius(
                    physical_a,
                    physical_b,
                    gains[index],
                    mode_transform,
                    servo_slopes,
                    basis_tau,
                    1.0 / experiment.control_hz,
                )
            )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    repeats = args.evaluation_initial_conditions
    physical = raw_split["labels"].repeat_interleave(repeats, dim=0)
    attitude, angular_velocity = _sample_near_equilibrium(
        len(physical),
        args.maximum_initial_tilt_rad,
        args.maximum_initial_rate_rad_s,
        generator,
        torch.float32,
    )
    evaluation_count = len(physical)
    results = {
        "original_nominal": _simulate_gain(
            experiment,
            physical,
            attitude,
            angular_velocity,
            np.repeat(original_nominal_gain[None], evaluation_count, axis=0),
            np.repeat(nominal_tau[None, 2:], evaluation_count, axis=0),
            np.ones(evaluation_count, dtype=np.bool_),
            args.duration_s,
        ),
        "legacy_predicted": _simulate_gain(
            experiment,
            physical,
            attitude,
            angular_velocity,
            np.repeat(legacy_synthesis.predicted_gain.numpy(), repeats, axis=0),
            np.repeat(legacy_synthesis.servo_time_constant_s.numpy(), repeats, axis=0),
            np.ones(evaluation_count, dtype=np.bool_),
            args.duration_s,
        ),
        "original_oracle": _simulate_gain(
            experiment,
            physical,
            attitude,
            angular_velocity,
            np.repeat(original_oracle_gains, repeats, axis=0),
            np.repeat(oracle_servo_tau, repeats, axis=0),
            np.ones(evaluation_count, dtype=np.bool_),
            args.duration_s,
        ),
    }
    for name, gains in composite_gains.items():
        results[name] = _simulate_composite_gain(
            experiment,
            physical,
            attitude,
            angular_velocity,
            np.repeat(gains, repeats, axis=0),
            mode_transform,
            servo_slopes,
            basis_tau,
            args.duration_s,
        )
    nonlinear = {name: _nonlinear_summary(value) for name, value in results.items()}
    parameter_stratified = {
        name: _parameter_stratified_convergence(
            result,
            results["composite_nominal"],
            physical,
            tuple(raw_split["label_names"]),
            count,
            repeats,
        )
        for name, result in results.items()
        if name != "composite_nominal"
    }

    def paired_against(reference_name: str) -> dict[str, Any]:
        reference_result = results[reference_name]
        paired = {}
        for name, result in results.items():
            if name == reference_name:
                continue
            paired[name] = {
                "safety": _paired_binary_summary(
                    result["safe"],
                    reference_result["safe"],
                    count,
                    repeats,
                    args.seed + 1000,
                ),
                "convergence": _paired_binary_summary(
                    result["converged"],
                    reference_result["converged"],
                    count,
                    repeats,
                    args.seed + 2000,
                ),
                "mean_saturation_fraction_delta": float(
                    (
                        result["saturation_fraction"]
                        - reference_result["saturation_fraction"]
                    ).mean()
                ),
                "mean_saturation_fraction_delta_group_cluster_bootstrap_95_ci": (
                    _cluster_bootstrap_mean_ci(
                        result["saturation_fraction"]
                        - reference_result["saturation_fraction"],
                        count,
                        repeats,
                        args.seed + 3000,
                    )
                ),
            }
        return paired

    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "legacy_checkpoint": str(Path(args.legacy_checkpoint).resolve()),
        "secondary_checkpoint": (
            None if secondary_checkpoint_path is None else str(secondary_checkpoint_path)
        ),
        "secondary_output_axes": secondary_output_axes,
        "test_parameter_groups": count,
        "trial_count": args.trial_count,
        "evaluation_initial_conditions_per_group": repeats,
        "duration_s": args.duration_s,
        "maximum_initial_tilt_rad": args.maximum_initial_tilt_rad,
        "maximum_initial_rate_rad_s": args.maximum_initial_rate_rad_s,
        "composite_contract": {
            "basis_time_constants_s": basis_tau,
            "adaptive_mode_indices": mode_indices,
            "target_representation": target_representation,
            "legacy_history_steps_used": legacy_raw_steps,
            "mode_transform": mode_transform.tolist(),
            "state_count": 19,
            "note": (
                "Modes outside adaptive_mode_indices remain nominal. The controller "
                "observes nine stable command-filter states with known deadzone and "
                "backlash preprocessing rather than mechanical servo angle."
            ),
        },
        "response_normalized_rmse": {
            "nominal": _response_nrmse(nominal_response, true_response),
            "legacy_predicted_physical_model": _response_nrmse(
                legacy_response, true_response
            ),
            "composite_predicted": _response_nrmse(
                composite_response, true_response
            ),
            "fixed_basis_oracle": _response_nrmse(
                fixed_basis_oracle_response, true_response
            ),
        },
        "local_true_plant_stable_fraction": {
            name: float(np.mean(np.asarray(values) < 1.0))
            for name, values in local_radius.items()
        },
        "local_true_plant_pole_radius_p95": {
            name: float(np.quantile(values, 0.95))
            for name, values in local_radius.items()
        },
        "nonlinear": nonlinear,
        "paired_vs_original_nominal": paired_against("original_nominal"),
        "paired_vs_composite_nominal": paired_against("composite_nominal"),
        "parameter_stratified_vs_composite_nominal": parameter_stratified,
        "deployment_mode": "research_only",
        "gain_updates_enabled": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-filter composite servo identification"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--secondary-checkpoint")
    parser.add_argument("--secondary-output-axes", default="roll,pitch")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--maximum-parameter-groups", type=int, default=393)
    parser.add_argument("--trial-count", type=int, default=8)
    parser.add_argument("--evaluation-initial-conditions", type=int, default=8)
    parser.add_argument("--maximum-initial-tilt-rad", type=float, default=0.2617994)
    parser.add_argument("--maximum-initial-rate-rad-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--gain-blend-fractions", default="0.25,0.5,0.75")
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
