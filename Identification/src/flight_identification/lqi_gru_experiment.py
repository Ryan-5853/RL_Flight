from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch
import yaml

from .config import IdentificationExperimentConfig, load_experiment_config
from .experiment import (
    _apply_effectiveness_labels,
    _expand_mapping,
    _group_split_assignments,
    _sample_initial_state,
    _sample_parameter_labels,
    _set_initial_observation_state,
    _yaw_free_attitude,
    parameter_label_names,
)


OBSERVATION_NAMES = (
    "attitude_q_tilt_estimate.w",
    "attitude_q_tilt_estimate.x",
    "attitude_q_tilt_estimate.y",
    "attitude_q_tilt_estimate.z",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "accelerometer.x",
    "accelerometer.y",
    "accelerometer.z",
    "motor_speed.upper",
    "motor_speed.lower",
    "target_attitude_q_tilt.w",
    "target_attitude_q_tilt.x",
    "target_attitude_q_tilt.y",
    "target_attitude_q_tilt.z",
    "target_angular_velocity_b.x",
    "target_angular_velocity_b.y",
    "target_angular_velocity_b.z",
    "collective_command",
    "previous_command.motor_upper",
    "previous_command.motor_lower",
    "previous_command.servo_1",
    "previous_command.servo_2",
    "previous_command.servo_3",
)

ACTION_NAMES = (
    "command.motor_upper",
    "command.motor_lower",
    "command.servo_1",
    "command.servo_2",
    "command.servo_3",
)

EXTERNAL_UPPER_ACTION_NAMES = (
    "command.motor_lower",
    "command.servo_1",
    "command.servo_2",
    "command.servo_3",
)


def _action_names(controller_config: Mapping[str, Any]) -> tuple[str, ...]:
    mode = str(_node(controller_config, "params").get("collective_mode", "hover"))
    return EXTERNAL_UPPER_ACTION_NAMES if mode == "external_upper" else ACTION_NAMES


def _raw_config(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("experiment root must be a mapping")
    return raw


def _node(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = root.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _oracle_gains(
    controller_config: Mapping[str, Any],
    parameters: Mapping[str, torch.Tensor],
    control_dt: float,
) -> torch.Tensor:
    """Design exactly one standard LQI for every fixed-airframe parameter row."""

    from flight_controller import ControllerContext, create_controller

    count = next(iter(parameters.values())).shape[0]
    gains = []
    for index in range(count):
        point = {name: value[index : index + 1] for name, value in parameters.items()}
        controller = create_controller(
            controller_config,
            ControllerContext(
                batch_size=1,
                device=next(iter(point.values())).device,
                dtype=next(iter(point.values())).dtype,
                control_dt=control_dt,
                parameters=point,
            ),
        )
        input_count = 4 if controller.upper_external else 5
        if (
            not controller._lqr_integral_enabled
            or controller._lqr_gain.shape != (input_count, 13)
        ):
            raise RuntimeError(
                f"oracle controller must synthesize a {input_count}x13 LQI gain"
            )
        gains.append(controller._lqr_gain)
    return torch.stack(gains, dim=0)


def _save_shard(
    output: Path,
    shard_index: int,
    split_assignment: torch.Tensor,
    group_ids: torch.Tensor,
    observations: torch.Tensor,
    teacher_actions: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: torch.Tensor,
    lqi_gains: torch.Tensor,
    label_names: tuple[str, ...],
    action_names: tuple[str, ...] = ACTION_NAMES,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_index, split_name in enumerate(("train", "validation", "test")):
        # Failed/unsafe oracle prefixes are useful for an audit but are not
        # "perfect control" supervision. V1 only distils complete safe flights.
        selected = (split_assignment[group_ids] == split_index) & valid_mask.all(dim=1)
        counts[split_name] = int(selected.sum())
        if not bool(selected.any()):
            continue
        directory = output / split_name
        directory.mkdir(parents=True, exist_ok=True)
        # Labels and gains are audit-only. The student loader deliberately does
        # not concatenate either tensor into observations.
        torch.save(
            {
                "schema_version": 1,
                "observations": observations[selected].contiguous(),
                "teacher_actions": teacher_actions[selected].contiguous(),
                "valid_mask": valid_mask[selected].contiguous(),
                "parameter_labels_audit_only": labels[selected].contiguous(),
                "oracle_lqi_gain_audit_only": lqi_gains[selected].contiguous(),
                "group_id": group_ids[selected].contiguous(),
                "observation_names": OBSERVATION_NAMES,
                "action_names": action_names,
                "label_names": label_names,
                "leakage_contract": "audit tensors are never student inputs",
            },
            directory / f"shard_{shard_index:06d}.pt",
        )
    return counts


@torch.no_grad()
def generate_lqi_gru_dataset(
    config: IdentificationExperimentConfig,
) -> Mapping[str, Any]:
    from flight_controller import (
        ControllerContext,
        ControllerReference,
        ControllerState,
        compose_external_upper_command,
        create_controller,
        load_controller_config,
    )
    from flight_train.config import _virtual_pilot
    from simenv import SimulationEnvironment
    from simenv.config import load_and_materialize

    raw = _raw_config(config.source_path)
    distillation = _node(raw, "distillation")
    if int(distillation.get("schema_version", -1)) != 1:
        raise ValueError("distillation.schema_version must equal 1")
    stride = int(distillation.get("dataset_stride_steps", 1))
    shard_groups = int(distillation.get("shard_parameter_groups", 64))
    if stride <= 0 or shard_groups <= 0:
        raise ValueError("dataset stride and shard group count must be positive")
    if config.episode_steps % stride:
        raise ValueError("episode steps must be divisible by dataset stride")
    if config.parameterization != "sim2real_micro":
        raise ValueError("LQI-GRU v1 requires sim2real_micro parameterization")
    if config.parallel_count < config.initial_conditions_per_group:
        raise ValueError("parallel_count must fit one complete parameter group")

    output = config.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.source_path, output / "experiment.yaml")

    device = torch.device(config.device)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    nominal = load_and_materialize(config.simulator_config, 1, device, dtype)
    controller_config = dict(load_controller_config(config.controller_config))
    if str(controller_config.get("type")) != "lqr":
        raise ValueError("oracle controller must be LQR/LQI")
    collective_mode = str(
        _node(controller_config, "params").get("collective_mode", "hover")
    )
    if collective_mode not in {"manual", "external_upper"}:
        raise ValueError(
            "oracle distillation requires manual or external_upper collective mode"
        )
    external_upper = collective_mode == "external_upper"
    action_names = _action_names(controller_config)
    pilot_config = _virtual_pilot(_node(raw, "command_source"))
    # The first recurrent input must not reveal the airframe-specific oracle
    # trim. Use one public nominal command for every parameter group. Later
    # previous-command inputs are ordinary causal actuator-command history.
    nominal_initial_controller = create_controller(
        controller_config,
        ControllerContext(
            batch_size=1,
            device=device,
            dtype=dtype,
            control_dt=1.0 / config.control_hz,
            parameters=nominal.parameters,
        ),
    )
    fixed_initial_previous_command = nominal_initial_controller.trim.command[0].clone()

    sample_rng = torch.Generator(device="cpu").manual_seed(config.seed)
    split_rng = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    labels_by_group, label_ranges = _sample_parameter_labels(
        config.parameter_groups,
        nominal.parameters,
        config,
        sample_rng,
        dtype,
    )
    splits = _group_split_assignments(
        config.parameter_groups, config.window.split_fractions, split_rng
    )
    label_names = parameter_label_names(config.parameterization)
    torch.save(
        {
            "schema_version": 1,
            "labels_audit_only": labels_by_group,
            "split_assignment": splits,
            "label_names": label_names,
            "label_ranges": label_ranges,
        },
        output / "parameter_groups_audit_only.pt",
    )

    episodes = config.initial_conditions_per_group
    max_groups_per_batch = config.parallel_count // episodes
    groups_per_batch = min(shard_groups, max_groups_per_batch)
    stored_steps = config.episode_steps // stride
    totals = {"groups": 0, "episodes": 0, "valid_steps": 0, "failed_episodes": 0}
    totals.update({f"{name}_episodes": 0 for name in ("train", "validation", "test")})

    for shard_index, group_start in enumerate(
        range(0, config.parameter_groups, groups_per_batch)
    ):
        group_stop = min(group_start + groups_per_batch, config.parameter_groups)
        group_count = group_stop - group_start
        batch_size = group_count * episodes
        group_ids = torch.arange(group_start, group_stop, dtype=torch.int64).repeat_interleave(episodes)
        episode_labels = labels_by_group[group_start:group_stop].repeat_interleave(episodes, dim=0)
        nominal_parameters = _expand_mapping(nominal.parameters, batch_size)
        actual_parameters = _apply_effectiveness_labels(
            nominal_parameters, episode_labels.to(device), config.parameterization
        )
        group_parameters = {name: value[::episodes] for name, value in actual_parameters.items()}
        group_gains = _oracle_gains(
            controller_config, group_parameters, 1.0 / config.control_hz
        )
        episode_gains = group_gains.repeat_interleave(episodes, dim=0)

        initial_state = _expand_mapping(nominal.initial_state, batch_size)
        attitude, angular_rate = _sample_initial_state(batch_size, config, sample_rng, dtype)
        initial_state["attitude_q_wb"].copy_(attitude.to(device))
        initial_state["angular_velocity_b"].copy_(angular_rate.to(device))
        materialized = replace(
            nominal,
            parameters=actual_parameters,
            initial_state=initial_state,
            sensor_state=_expand_mapping(nominal.sensor_state, batch_size),
        )
        environment = SimulationEnvironment(
            materialized, batch_size, device, dtype, logging_enabled=False
        )
        try:
            oracle_context = ControllerContext(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                control_dt=1.0 / config.control_hz,
                parameters=actual_parameters,
            )
            if external_upper:
                from .external_collective_lqi import ExternalCollectiveLQIController

                oracle = ExternalCollectiveLQIController(
                    oracle_context,
                    config.controller_config,
                    config.simulator_config,
                    episode_gains,
                    "nonlinear",
                )
            else:
                oracle = create_controller(controller_config, oracle_context)
                oracle.schedule_lqr_gain(episode_gains)
            _set_initial_observation_state(
                environment, oracle.trim.motor_speed, initial_state["angular_velocity_b"]
            )
            from .external_pilot import make_external_pilot

            pilot = make_external_pilot(
                pilot_config, raw, batch_size, device, dtype, config.control_hz
            )
            pilot.generator.manual_seed(pilot_config.seed + group_start)
            active = torch.ones(batch_size, device=device, dtype=torch.bool)
            pilot.reset(active)
            previous_command = fixed_initial_previous_command[None].expand(
                batch_size, -1
            ).clone()
            observations = torch.zeros(
                batch_size, stored_steps, len(OBSERVATION_NAMES), device=device, dtype=dtype
            )
            teacher_actions = torch.zeros(
                batch_size, stored_steps, len(action_names), device=device, dtype=dtype
            )
            valid_mask = torch.zeros(batch_size, stored_steps, device=device, dtype=torch.bool)
            write_index = 0
            for step in range(config.episode_steps):
                truth = environment.observe("truth").values
                sensors = environment.observe("sensor").values
                pilot.step(-truth["position_n"][:, 2:3], active)
                command = pilot.snapshot()
                attitude_estimate = _yaw_free_attitude(truth["attitude_q_wb"])
                target_attitude = _yaw_free_attitude(command.target_attitude_q_wb)
                target_rate = torch.cat(
                    (
                        torch.zeros(batch_size, 2, device=device, dtype=dtype),
                        pilot.desired_yaw_rate,
                    ),
                    dim=1,
                )
                reference = ControllerReference(
                    target_position_n=torch.zeros(batch_size, 3, device=device, dtype=dtype),
                    target_velocity_n=torch.zeros(batch_size, 3, device=device, dtype=dtype),
                    target_attitude_q_wb=target_attitude,
                    target_angular_velocity_b=target_rate,
                    collective_command=command.upper_throttle,
                )
                state = ControllerState(
                    position_n=truth["position_n"],
                    velocity_n=truth["velocity_n"],
                    attitude_q_wb=attitude_estimate,
                    angular_velocity_b=truth["angular_velocity_b"],
                    linear_acceleration_n=truth["linear_acceleration_n"],
                    motor_speed=truth["motor_speed"],
                    servo_angle=truth["servo_angle"],
                )
                oracle_action = (
                    oracle.step(state, reference, command.upper_throttle, active)
                    if external_upper
                    else oracle.step(state, reference, active).command
                )
                applied_command = (
                    compose_external_upper_command(
                        command.upper_throttle, oracle_action
                    )
                    if external_upper
                    else oracle_action
                )
                if step % stride == 0:
                    student_observation = torch.cat(
                        (
                            attitude_estimate,
                            sensors["gyro"],
                            sensors["accelerometer"],
                            sensors["motor_speed"],
                            target_attitude,
                            target_rate,
                            command.upper_throttle,
                            previous_command,
                        ),
                        dim=1,
                    )
                    observations[:, write_index].copy_(
                        torch.where(active[:, None], student_observation, torch.zeros_like(student_observation))
                    )
                    teacher_actions[:, write_index].copy_(oracle_action)
                    valid_mask[:, write_index].copy_(active)
                    write_index += 1
                previous_command = applied_command
                result = environment.advance(applied_command, active)
                tilt = torch.acos(
                    (2.0 * truth["attitude_q_wb"][:, 0].square()
                     + 2.0 * truth["attitude_q_wb"][:, 3].square() - 1.0).clamp(-1.0, 1.0)
                )
                active &= result.valid
                active &= torch.isfinite(tilt)
                active &= tilt <= config.convergence.safety_tilt_rad
                active &= truth["angular_velocity_b"].norm(dim=1) <= config.convergence.safety_angular_rate_rad_s
            counts = _save_shard(
                output,
                shard_index,
                splits,
                group_ids,
                observations.cpu(),
                teacher_actions.cpu(),
                valid_mask.cpu(),
                episode_labels,
                episode_gains.cpu(),
                label_names,
                action_names,
            )
            totals["groups"] += group_count
            totals["episodes"] += batch_size
            totals["valid_steps"] += int(valid_mask.sum())
            totals["failed_episodes"] += int((~active).sum())
            for name, count in counts.items():
                totals[f"{name}_episodes"] += count
        finally:
            environment.close()

    manifest = {
        "schema_version": 1,
        "experiment": "oracle_per_airframe_lqi_to_causal_gru_distillation_v1",
        "control_hz": config.control_hz,
        "stored_hz": config.control_hz // stride,
        "episode_steps": stored_steps,
        "observation_names": OBSERVATION_NAMES,
        "action_names": action_names,
        "split_semantics": "disjoint fixed-airframe parameter groups",
        "expert_contract": (
            "one truth-parameter 4-output LQI; upper rotor is external"
            if external_upper
            else "one truth-parameter 5-output LQI and truth state per random airframe"
        ),
        "upper_rotor_owner": "external_pilot" if external_upper else "student_teacher",
        "external_pilot_profiles": list(pilot.profile_names),
        "student_contract": "causal observations only; no parameters, gains, or truth servo state",
        "initial_previous_command": fixed_initial_previous_command.detach().cpu().tolist(),
        "initial_previous_command_semantics": "one fixed nominal command shared by every airframe",
        "attitude_estimate_note": (
            "SimEnv has no estimator model; truth attitude is routed through the declared "
            "attitude-estimator interface. It is observable state, not a parameter input."
        ),
        "audit_only_tensors": ["parameter_labels_audit_only", "oracle_lqi_gain_audit_only"],
        "totals": totals,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def build_data_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate per-airframe oracle LQI distillation data")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    parser.add_argument("--parameter-groups", type=int)
    parser.add_argument("--episodes-per-group", type=int)
    parser.add_argument("--parallel-count", type=int)
    parser.add_argument("--episode-duration-s", type=float)
    parser.add_argument("--device")
    return parser


def data_main() -> None:
    args = build_data_parser().parse_args()
    config = load_experiment_config(args.config)
    replacements = {}
    for argument, field in (
        (args.output_directory, "output_directory"),
        (args.parameter_groups, "parameter_groups"),
        (args.episodes_per_group, "initial_conditions_per_group"),
        (args.parallel_count, "parallel_count"),
        (args.episode_duration_s, "episode_duration_s"),
        (args.device, "device"),
    ):
        if argument is not None:
            replacements[field] = Path(argument).resolve() if field == "output_directory" else argument
    manifest = generate_lqi_gru_dataset(replace(config, **replacements))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    data_main()
