from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from .config import IdentificationExperimentConfig, load_experiment_config
from .experiment import (
    FAILURE_NAMES,
    FEATURE_NAMES,
    SIM2REAL_TARGET_NAMES,
    FirstOrderServoObserver,
    _apply_effectiveness_labels,
    _expand_mapping,
    _group_split_assignments,
    _sample_parameter_labels,
    _sim2real_lqr_targets,
    _sample_initial_state,
    _set_initial_observation_state,
    _yaw_free_attitude,
    parameter_label_names,
)


def _save_group_shard(
    output: Path,
    shard_index: int,
    split_assignment: torch.Tensor,
    group_ids: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    failure_step: torch.Tensor,
    failure_code: torch.Tensor,
    saturation_fraction: torch.Tensor,
    label_names: tuple[str, ...],
    parameterization: str,
    targets: torch.Tensor | None = None,
) -> dict[str, int]:
    counts = {}
    for split_index, split_name in enumerate(("train", "validation", "test")):
        selected = split_assignment[group_ids] == split_index
        count = int(selected.sum())
        counts[split_name] = count
        if count == 0:
            continue
        directory = output / split_name
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
                "schema_version": 1,
                "features": features[selected].contiguous(),
                "valid_mask": valid_mask[selected].contiguous(),
                "labels": labels[selected].contiguous(),
                "group_id": group_ids[selected].contiguous(),
                "failure_step": failure_step[selected].contiguous(),
                "failure_code": failure_code[selected].contiguous(),
                "saturation_fraction": saturation_fraction[selected].contiguous(),
                "feature_names": FEATURE_NAMES,
                "label_names": label_names,
                "parameterization": parameterization,
                "failure_names": FAILURE_NAMES,
            }
        if targets is not None:
            payload["targets"] = targets[selected].contiguous()
            payload["target_names"] = SIM2REAL_TARGET_NAMES
        torch.save(payload, directory / f"shard_{shard_index:06d}.pt")
    return counts


@torch.no_grad()
def generate_repeated_trial_dataset(
    config: IdentificationExperimentConfig,
) -> Mapping[str, Any]:
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
        raise ValueError("repeated-trial experiment requires an LQR controller")

    sampling_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    split_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    split_assignment = _group_split_assignments(
        config.parameter_groups, config.window.split_fractions, split_generator
    )
    labels_by_group, label_ranges = _sample_parameter_labels(
        config.parameter_groups,
        nominal_materialized.parameters,
        config,
        sampling_generator,
        dtype,
    )
    label_names = parameter_label_names(config.parameterization)
    torch.save(
        {
            "schema_version": 1,
            "labels": labels_by_group,
            "group_id": torch.arange(config.parameter_groups, dtype=torch.int64),
            "split_assignment": split_assignment,
            "label_names": label_names,
            "label_ranges": label_ranges,
            "parameterization": config.parameterization,
        },
        output / "parameter_groups.pt",
    )
    trials = config.initial_conditions_per_group
    groups_per_batch = config.parallel_count // trials
    totals = {
        "groups": 0,
        "trials": 0,
        "failed_trials": 0,
        "timeout_trials": 0,
        "crashed_trials": 0,
        "valid_recorded_steps": 0,
        "train_groups": 0,
        "validation_groups": 0,
        "test_groups": 0,
    }
    hold_steps = round(config.convergence.hold_s * config.control_hz)
    nominal_pole_radius = None
    for shard_index, group_start in enumerate(
        range(0, config.parameter_groups, groups_per_batch)
    ):
        group_stop = min(group_start + groups_per_batch, config.parameter_groups)
        group_count = group_stop - group_start
        batch_size = group_count * trials
        group_ids = torch.arange(group_start, group_stop, dtype=torch.int64)
        episode_labels = labels_by_group[group_start:group_stop].repeat_interleave(
            trials, dim=0
        )
        nominal_parameters = _expand_mapping(
            nominal_materialized.parameters, batch_size
        )
        actual_parameters = _apply_effectiveness_labels(
            nominal_parameters, episode_labels.to(device), config.parameterization
        )
        group_targets = (
            _sim2real_lqr_targets(
                {
                    name: value[::trials]
                    for name, value in actual_parameters.items()
                }
            ).cpu()
            if config.parameterization == "sim2real_micro"
            else None
        )
        initial_state = _expand_mapping(nominal_materialized.initial_state, batch_size)
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
            materialized, batch_size, device, dtype, logging_enabled=False
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
            initial_motor_speed = (
                LocalPlantModel(actual_parameters).hover_trim().motor_speed
                if config.parameterization == "sim2real_micro"
                else controller.trim.motor_speed
            )
            _set_initial_observation_state(
                environment,
                initial_motor_speed,
                initial_state["angular_velocity_b"],
            )
            servo_observer = FirstOrderServoObserver(
                nominal_parameters["servos.pwm_angle_table"],
                nominal_parameters["servos.tau"],
                1.0 / config.control_hz,
            )
            zeros = torch.zeros(batch_size, 3, device=device, dtype=dtype)
            identity = torch.zeros(batch_size, 4, device=device, dtype=dtype)
            identity[:, 0] = 1.0
            reference = ControllerReference(
                target_position_n=zeros,
                target_velocity_n=zeros.clone(),
                target_attitude_q_wb=identity,
                target_angular_velocity_b=zeros.clone(),
                collective_command=torch.zeros(
                    batch_size, 1, device=device, dtype=dtype
                ),
            )
            features = torch.zeros(
                batch_size,
                config.episode_steps,
                len(FEATURE_NAMES),
                device=device,
                dtype=dtype,
            )
            valid_mask = torch.zeros(
                batch_size,
                config.episode_steps,
                device=device,
                dtype=torch.bool,
            )
            active = torch.ones(batch_size, device=device, dtype=torch.bool)
            failure_step = torch.full(
                (batch_size,),
                config.episode_steps,
                device=device,
                dtype=torch.int64,
            )
            failure_code = torch.zeros(
                batch_size, device=device, dtype=torch.int64
            )
            saturation_steps = torch.zeros_like(failure_code)
            settled_count = torch.zeros_like(failure_code)
            for step in range(config.episode_steps):
                truth = environment.observe("truth").values
                if config.measurement_mode == "simulated_sensors":
                    sensors = environment.observe("sensor").values
                    measured_rate = sensors["gyro"]
                    measured_motor = sensors["motor_speed"]
                else:
                    measured_rate = truth["angular_velocity_b"]
                    measured_motor = truth["motor_speed"]
                tilt_attitude = _yaw_free_attitude(truth["attitude_q_wb"])
                tilt = torch.acos(tilt_cosine(truth["attitude_q_wb"]))
                angular_rate = truth["angular_velocity_b"].norm(dim=1)
                finite = (
                    torch.isfinite(tilt_attitude).all(dim=1)
                    & torch.isfinite(measured_rate).all(dim=1)
                    & torch.isfinite(measured_motor).all(dim=1)
                )
                tilt_failure = tilt > config.convergence.safety_tilt_rad
                rate_failure = (
                    angular_rate > config.convergence.safety_angular_rate_rad_s
                )
                newly_failed = active & (~finite | tilt_failure | rate_failure)
                failure_code = torch.where(
                    newly_failed & tilt_failure,
                    torch.full_like(failure_code, 2),
                    failure_code,
                )
                failure_code = torch.where(
                    newly_failed & rate_failure,
                    torch.full_like(failure_code, 3),
                    failure_code,
                )
                failure_code = torch.where(
                    newly_failed & ~finite,
                    torch.full_like(failure_code, 4),
                    failure_code,
                )
                failure_step = torch.where(
                    newly_failed, torch.full_like(failure_step, step), failure_step
                )
                active &= ~newly_failed
                safe_rate = torch.nan_to_num(measured_rate)
                safe_motor = torch.nan_to_num(measured_motor)
                safe_tilt = torch.nan_to_num(tilt_attitude)
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
                output_value = controller.step(state, reference, active)
                current_feature = torch.cat(
                    (safe_tilt, safe_rate, safe_motor, output_value.command), dim=1
                )
                features[:, step].copy_(
                    torch.where(
                        active[:, None],
                        current_feature,
                        torch.zeros_like(current_feature),
                    )
                )
                valid_mask[:, step].copy_(active)
                attitude_error = quaternion_rotation_error(
                    safe_tilt, identity
                )[:, :2].norm(dim=1)
                settled = active & (
                    attitude_error
                    <= config.convergence.maximum_roll_pitch_error_rad
                ) & (
                    angular_rate
                    <= config.convergence.maximum_angular_rate_rad_s
                )
                settled_count = torch.where(
                    settled, settled_count + 1, torch.zeros_like(settled_count)
                )
                command = output_value.command
                saturated = active & (
                    (command[:, :2] <= 1e-6).any(dim=1)
                    | (command[:, :2] >= 1.0 - 1e-6).any(dim=1)
                    | (command[:, 2:].abs() >= 1.0 - 1e-6).any(dim=1)
                )
                saturation_steps += saturated.to(torch.int64)
                servo_observer.advance(command[:, 2:])
                result = environment.advance(command, active)
                simulator_failure = active & ~result.valid
                failure_code = torch.where(
                    simulator_failure,
                    torch.full_like(failure_code, 4),
                    failure_code,
                )
                failure_step = torch.where(
                    simulator_failure,
                    torch.full_like(failure_step, step + 1),
                    failure_step,
                )
                active &= result.valid
            completed = failure_code == 0
            converged = completed & (settled_count >= hold_steps)
            failure_code = torch.where(
                completed & ~converged, torch.ones_like(failure_code), failure_code
            )
            saturation_fraction = saturation_steps.to(dtype) / valid_mask.sum(
                dim=1
            ).clamp_min(1)
            grouped_shape = (group_count, trials)
            counts = _save_group_shard(
                output,
                shard_index,
                split_assignment,
                group_ids,
                labels_by_group[group_start:group_stop],
                features[:, : config.window_steps].cpu().reshape(
                    group_count, trials, config.window_steps, len(FEATURE_NAMES)
                ),
                valid_mask[:, : config.window_steps].cpu().reshape(
                    group_count, trials, config.window_steps
                ),
                failure_step.cpu().reshape(grouped_shape),
                failure_code.cpu().reshape(grouped_shape),
                saturation_fraction.cpu().reshape(grouped_shape),
                label_names,
                config.parameterization,
                group_targets,
            )
            totals["groups"] += group_count
            totals["trials"] += batch_size
            totals["failed_trials"] += int((failure_code != 0).sum())
            totals["timeout_trials"] += int((failure_code == 1).sum())
            totals["crashed_trials"] += int((failure_code >= 2).sum())
            totals["valid_recorded_steps"] += int(valid_mask.sum())
            for split_name, count in counts.items():
                totals[f"{split_name}_groups"] += count
        finally:
            environment.close()
    manifest = {
        "schema_version": 1,
        "experiment": "repeated_destructive_trial_identification_v1",
        "parameter_semantics": "one fixed parameter label shared by all trials in a group",
        "trial_semantics": (
            "Each trial starts from a configured closed-loop-compatible attitude/rate. "
            "Unsafe or invalid trials retain their pre-failure prefix and are zero-padded."
        ),
        "initial_state_sampling_design": config.initial_state.sampling_design,
        "controller_semantics": {
            "gain_source": "fixed_nominal_for_all_identification_trials",
            "collective_mode": str(
                controller_config.get("params", {}).get(
                    "collective_mode", "hover"
                )
            ),
            "upper_motor_external": (
                str(
                    controller_config.get("params", {}).get(
                        "collective_mode"
                    )
                )
                == "external_upper"
            ),
            "roll_pitch_target_rad": [0.0, 0.0],
            "yaw_target": "angular_rate_zero",
            "yaw_angle_feedback": False,
            "servo_angle_feedback": False,
            "measurement_mode": config.measurement_mode,
            "nominal_closed_loop_pole_radius": nominal_pole_radius,
        },
        "control_hz": config.control_hz,
        "trials_per_group": trials,
        "trial_steps": config.window_steps,
        "audit_episode_steps": config.episode_steps,
        "feature_names": FEATURE_NAMES,
        "mask_semantics": "true through the last pre-failure recorded state",
        "parameterization": config.parameterization,
        "label_names": label_names,
        "target_names": (
            SIM2REAL_TARGET_NAMES
            if config.parameterization == "sim2real_micro"
            else None
        ),
        "label_ranges": label_ranges.tolist(),
        "legacy_log10_range": (
            list(config.log10_effectiveness_range)
            if config.parameterization == "legacy_collective"
            else None
        ),
        "empirical_ranges": (
            None
            if config.empirical_ranges is None
            else {
                "thrust_to_weight": list(config.empirical_ranges.thrust_to_weight),
                "inertia_scale": list(config.empirical_ranges.inertia_scale),
                "motor_reaction_scale": list(config.empirical_ranges.motor_reaction_scale),
                "motor_time_constant_s": list(config.empirical_ranges.motor_time_constant_s),
                "grid_effectiveness_scale": list(config.empirical_ranges.grid_effectiveness_scale),
                "servo_time_constant_s": list(config.empirical_ranges.servo_time_constant_s),
            }
        ),
        "split_semantics": "disjoint fixed-airframe parameter groups",
        "totals": totals,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate repeated destructive identification trials per airframe"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    parser.add_argument("--parameter-groups", type=int)
    parser.add_argument("--trials-per-group", type=int)
    parser.add_argument("--parallel-count", type=int)
    parser.add_argument("--episode-duration-s", type=float)
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    updates = {}
    for argument, field in (
        (args.parameter_groups, "parameter_groups"),
        (args.trials_per_group, "initial_conditions_per_group"),
        (args.parallel_count, "parallel_count"),
        (args.episode_duration_s, "episode_duration_s"),
        (args.device, "device"),
    ):
        if argument is not None:
            updates[field] = argument
    if args.output_directory is not None:
        updates["output_directory"] = Path(args.output_directory).expanduser().resolve()
    if updates:
        config = replace(config, **updates)
    if config.parallel_count % config.initial_conditions_per_group:
        raise ValueError("parallel_count must be divisible by trials_per_group")
    print(json.dumps(generate_repeated_trial_dataset(config), indent=2))


if __name__ == "__main__":
    main()
