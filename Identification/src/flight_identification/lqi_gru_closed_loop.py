from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from .config import load_experiment_config
from .experiment import (
    _apply_effectiveness_labels,
    _expand_mapping,
    _sample_initial_state,
    _set_initial_observation_state,
    _yaw_free_attitude,
)
from .lqi_gru_distillation import StudentStepState, load_student, step_student
from .lqi_gru_experiment import OBSERVATION_NAMES, _oracle_gains


def _node(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = root.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _accepted_group_ids(dataset: Path, split: str) -> torch.Tensor:
    values = []
    for path in sorted((dataset / split).glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        values.append(shard["group_id"])
    if not values:
        raise FileNotFoundError(f"no accepted {split} shards in {dataset}")
    return torch.unique(torch.cat(values), sorted=True)


def _summary(value: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(value.mean()),
        "p50": float(torch.quantile(value, 0.50)),
        "p90": float(torch.quantile(value, 0.90)),
        "p99": float(torch.quantile(value, 0.99)),
    }


def _perturbed_yaw_free_attitude(
    attitude_q_wb: torch.Tensor,
    noise_deg: float,
    bias_deg: float,
    latency_steps: int,
    buffer: list[torch.Tensor] | None,
    episode_bias: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Add roll/pitch estimation noise, per-episode bias, and latency."""

    w, x, y, z = attitude_q_wb.unbind(dim=1)
    roll = torch.atan2(
        2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square())
    )
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    if noise_deg > 0.0:
        noise = torch.randn_like(roll[:, None].expand(-1, 2)) * torch.deg2rad(
            torch.tensor(noise_deg, device=roll.device)
        )
    else:
        noise = torch.zeros_like(roll[:, None].expand(-1, 2))
    roll = roll + episode_bias[:, 0] + noise[:, 0]
    pitch = pitch + episode_bias[:, 1] + noise[:, 1]
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    estimate = torch.stack(
        (
            torch.cos(half_roll) * torch.cos(half_pitch),
            torch.sin(half_roll) * torch.cos(half_pitch),
            torch.cos(half_roll) * torch.sin(half_pitch),
            -torch.sin(half_roll) * torch.sin(half_pitch),
        ),
        dim=1,
    )
    if buffer is None:
        buffer = []
    buffer.append(estimate)
    if latency_steps > 0:
        estimate = buffer[max(0, len(buffer) - 1 - latency_steps)]
    return estimate, buffer


@torch.no_grad()
def _simulate(
    mode: str,
    config: Any,
    raw: Mapping[str, Any],
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    trials: int,
    duration_s: float,
    seed: int,
    checkpoint_path: Path,
    device: torch.device,
    pilot_height_mode: str,
    attitude_noise_deg: float = 0.0,
    attitude_bias_deg: float = 0.0,
    attitude_latency_steps: int = 0,
    report_segments_s: Sequence[float] = (),
) -> Mapping[str, torch.Tensor]:
    from flight_controller import (
        ControllerContext,
        ControllerReference,
        ControllerState,
        create_controller,
        load_controller_config,
    )
    from flight_controller.math import quaternion_rotation_error, tilt_cosine
    from flight_train.commands import VirtualPilotCommandSource
    from flight_train.config import _virtual_pilot
    from simenv import SimulationEnvironment
    from simenv.config import load_and_materialize

    # Every controller variant must receive identical materialization, sensor
    # noise and dynamics RNG state. Explicit pilot/initial-state generators do
    # not cover global Torch RNG consumed while SimEnv is materialized.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    dtype = torch.float32
    nominal = load_and_materialize(config.simulator_config, 1, device, dtype)
    controller_config = dict(load_controller_config(config.controller_config))
    group_count = len(labels)
    batch_size = group_count * trials
    episode_labels = labels.repeat_interleave(trials, dim=0).to(device)
    nominal_parameters = _expand_mapping(nominal.parameters, batch_size)
    actual_parameters = _apply_effectiveness_labels(
        nominal_parameters, episode_labels, config.parameterization
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample_config = replace(config, initial_conditions_per_group=trials)
    attitude, angular_rate = _sample_initial_state(
        batch_size, sample_config, generator, dtype
    )
    initial_state = _expand_mapping(nominal.initial_state, batch_size)
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
        actual_controller = create_controller(
            controller_config,
            ControllerContext(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                control_dt=1.0 / config.control_hz,
                parameters=actual_parameters,
            ),
        )
        initial_motor_speed = actual_controller.trim.motor_speed
        controller = None
        if mode == "oracle_lqi":
            controller = actual_controller
            group_parameters = {
                name: value[::trials] for name, value in actual_parameters.items()
            }
            gains = _oracle_gains(
                controller_config, group_parameters, 1.0 / config.control_hz
            ).repeat_interleave(trials, dim=0)
            controller.schedule_lqr_gain(gains)
        elif mode == "nominal_lqi":
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
        elif mode not in {"gru", "gru_reset"}:
            raise ValueError(f"unknown closed-loop mode {mode!r}")
        _set_initial_observation_state(
            environment, initial_motor_speed, initial_state["angular_velocity_b"]
        )

        pilot_config = _virtual_pilot(_node(raw, "command_source"))
        pilot = VirtualPilotCommandSource(
            pilot_config, batch_size, device, dtype, config.control_hz
        )
        pilot.generator.manual_seed(seed + 1)
        active = torch.ones(batch_size, device=device, dtype=torch.bool)
        pilot.reset(active)
        episode_bias = torch.randn(
            batch_size, 2, device=device, dtype=dtype
        ) * torch.deg2rad(torch.tensor(attitude_bias_deg, device=device, dtype=dtype))
        attitude_buffer: list[torch.Tensor] | None = None

        student = checkpoint = mean = std = None
        student_observation_indices = None
        if mode in {"gru", "gru_reset"}:
            student, checkpoint = load_student(checkpoint_path, device)
            student.eval()
            mean = checkpoint["normalization"]["mean"].to(device)
            std = checkpoint["normalization"]["std"].to(device)
            full_name_to_index = {
                name: index for index, name in enumerate(OBSERVATION_NAMES)
            }
            checkpoint_names = tuple(checkpoint["observation_names"])
            missing = [
                name for name in checkpoint_names if name not in full_name_to_index
            ]
            if missing:
                raise ValueError(
                    f"checkpoint requests unknown observations: {missing}"
                )
            student_observation_indices = torch.tensor(
                [full_name_to_index[name] for name in checkpoint_names],
                device=device,
                dtype=torch.int64,
            )
            initial_previous = checkpoint.get("initial_previous_command")
            if initial_previous is None:
                # Checkpoints produced by v1 store this contract in the dataset
                # manifest rather than duplicating it in model metadata.
                nominal_controller = create_controller(
                    controller_config,
                    ControllerContext(
                        batch_size=1,
                        device=device,
                        dtype=dtype,
                        control_dt=1.0 / config.control_hz,
                        parameters=nominal.parameters,
                    ),
                )
            initial_previous = nominal_controller.trim.command[0]
            previous_command = torch.as_tensor(
                initial_previous, device=device, dtype=dtype
            ).reshape(1, 5).expand(batch_size, -1).clone()
        else:
            previous_command = actual_controller.trim.command.clone()
        student_state = StudentStepState()

        steps = round(duration_s * config.control_hz)
        segment_boundaries = sorted(report_segments_s)
        segment_count = len(segment_boundaries)
        segment_attitude = torch.zeros(
            segment_count, batch_size, device=device, dtype=dtype
        )
        segment_rate = torch.zeros_like(segment_attitude)
        segment_movement = torch.zeros_like(segment_attitude)
        segment_survival = torch.zeros(
            segment_count, batch_size, device=device, dtype=torch.int64
        )
        segment_count_steps = torch.zeros_like(segment_attitude)
        survival_steps = torch.zeros(batch_size, device=device, dtype=torch.int64)
        capture_steps = {
            min(round(boundary * config.control_hz), steps) - 1
            for boundary in segment_boundaries
        }
        hidden_norm_snapshots: list[float | None] = []
        attitude_squared = torch.zeros(batch_size, device=device, dtype=dtype)
        rate_squared = torch.zeros_like(attitude_squared)
        saturation_steps = torch.zeros_like(attitude_squared)
        movement_sum = torch.zeros_like(attitude_squared)
        last_applied = previous_command.clone()
        accumulated_steps = torch.zeros_like(attitude_squared)

        for step in range(steps):
            truth = environment.observe("truth").values
            sensors = environment.observe("sensor").values
            if pilot_height_mode == "training_truth":
                pilot_height = -truth["position_n"][:, 2:3]
            elif pilot_height_mode == "fixed_target":
                # Identical external throttle schedule across variants: plant
                # divergence cannot alter the virtual height controller.
                pilot_height = torch.zeros(
                    batch_size, 1, device=device, dtype=dtype
                )
            else:
                raise ValueError(
                    "pilot_height_mode must be training_truth or fixed_target"
                )
            pilot.step(pilot_height, active)
            snapshot = pilot.snapshot()
            attitude_estimate, attitude_buffer = _perturbed_yaw_free_attitude(
                truth["attitude_q_wb"],
                attitude_noise_deg,
                attitude_bias_deg,
                attitude_latency_steps,
                attitude_buffer,
                episode_bias,
            )
            target_attitude = _yaw_free_attitude(snapshot.target_attitude_q_wb)
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
                collective_command=snapshot.upper_throttle,
            )
            if controller is not None:
                state = ControllerState(
                    position_n=truth["position_n"],
                    velocity_n=truth["velocity_n"],
                    attitude_q_wb=attitude_estimate,
                    angular_velocity_b=truth["angular_velocity_b"],
                    linear_acceleration_n=truth["linear_acceleration_n"],
                    motor_speed=truth["motor_speed"],
                    servo_angle=truth["servo_angle"],
                )
                command = controller.step(state, reference, active).command
            else:
                if (
                    student is None
                    or mean is None
                    or std is None
                    or student_observation_indices is None
                ):
                    raise RuntimeError("student model was not initialized")
                observation = torch.cat(
                    (
                        attitude_estimate,
                        sensors["gyro"],
                        sensors["accelerometer"],
                        sensors["motor_speed"],
                        target_attitude,
                        target_rate,
                        snapshot.upper_throttle,
                        previous_command,
                    ),
                    dim=1,
                )
                observation = observation.index_select(
                    1, student_observation_indices
                )
                normalized = (observation - mean) / std
                command, student_state = step_student(
                    student,
                    checkpoint,
                    normalized,
                    student_state,
                    full_context=mode == "gru",
                )
                if step in capture_steps and student_state.hidden is not None:
                    hidden = student_state.hidden
                    if isinstance(hidden, (tuple, list)):
                        hidden = hidden[0]
                    hidden_norm_snapshots.append(
                        float(hidden.norm(dim=-1).mean().cpu())
                    )
                elif step in capture_steps:
                    hidden_norm_snapshots.append(None)
                previous_command = command

            attitude_error = quaternion_rotation_error(
                attitude_estimate, target_attitude
            )[:, :2]
            rate_error = truth["angular_velocity_b"] - target_rate
            active_float = active.to(dtype)
            attitude_squared += attitude_error.square().sum(1) * active_float
            rate_squared += rate_error.square().sum(1) * active_float
            saturated = (
                (command[:, :2] <= 1e-6).any(1)
                | (command[:, :2] >= 1.0 - 1e-6).any(1)
                | (command[:, 2:].abs() >= 1.0 - 1e-6).any(1)
            )
            saturation_steps += saturated.to(dtype) * active_float
            movement_sum += (command - last_applied).square().mean(1).sqrt() * active_float
            accumulated_steps += active_float
            if segment_count:
                segment_index = min(
                    segment_count - 1,
                    sum(1 for boundary in segment_boundaries if boundary * config.control_hz <= step),
                )
                segment_attitude[segment_index] += (
                    attitude_error.square().sum(1) * active_float
                )
                segment_rate[segment_index] += (
                    rate_error.square().sum(1) * active_float
                )
                segment_movement[segment_index] += (
                    (command - last_applied).square().mean(1).sqrt() * active_float
                )
                segment_survival[segment_index] += active.to(torch.int64)
                segment_count_steps[segment_index] += active_float
            last_applied = command
            result = environment.advance(command, active)
            next_truth = environment.observe("truth").values
            safe = result.valid
            safe &= torch.isfinite(next_truth["attitude_q_wb"]).all(1)
            safe &= torch.acos(tilt_cosine(next_truth["attitude_q_wb"])) <= config.convergence.safety_tilt_rad
            safe &= next_truth["angular_velocity_b"].norm(dim=1) <= config.convergence.safety_angular_rate_rad_s
            active &= safe
            survival_steps += active.to(torch.int64)

        denominator = accumulated_steps.clamp_min(1.0)
        result = {
            "group_id": group_ids.repeat_interleave(trials),
            "safe": active.cpu(),
            "survival_fraction": (survival_steps.to(dtype) / steps).cpu(),
            "attitude_tracking_rms_rad": torch.sqrt(attitude_squared / denominator).cpu(),
            "rate_tracking_rms_rad_s": torch.sqrt(rate_squared / denominator).cpu(),
            "saturation_fraction": (saturation_steps / denominator).cpu(),
            "command_movement_mean": (movement_sum / denominator).cpu(),
        }
        if report_segments_s:
            segment_report = []
            for index, boundary in enumerate(segment_boundaries):
                denominator_segment = segment_count_steps[index].clamp_min(1)
                segment_report.append(
                    {
                        "boundary_s": boundary,
                        "attitude_rms_rad": torch.sqrt(
                            segment_attitude[index] / denominator_segment
                        ).mean()
                        .cpu()
                        .item(),
                        "rate_rms_rad_s": torch.sqrt(
                            segment_rate[index] / denominator_segment
                        )
                        .mean()
                        .cpu()
                        .item(),
                        "movement_mean": (
                            segment_movement[index] / denominator_segment
                        )
                        .mean()
                        .cpu()
                        .item(),
                        "survival_fraction": float(
                            (
                                segment_survival[index].to(dtype)
                                / denominator_segment
                            )
                            .mean()
                            .cpu()
                        ),
                    }
                )
            result["segments"] = segment_report
            result["hidden_norms"] = hidden_norm_snapshots
        return result
    finally:
        environment.close()


def evaluate_closed_loop(
    dataset: Path,
    config_path: Path,
    checkpoint: Path,
    output: Path,
    device: torch.device,
    maximum_groups: int,
    trials: int,
    duration_s: float,
    seed: int,
    pilot_height_mode: str,
    split: str = "test",
    variants: Sequence[str] = ("oracle_lqi", "nominal_lqi", "gru", "gru_reset"),
    attitude_noise_deg: float = 0.0,
    attitude_bias_deg: float = 0.0,
    attitude_latency_steps: int = 0,
    report_segments_s: Sequence[float] = (),
) -> Mapping[str, Any]:
    config = load_experiment_config(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    group_payload = torch.load(
        dataset / "parameter_groups_audit_only.pt", map_location="cpu", weights_only=False
    )
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split {split!r}")
    accepted = _accepted_group_ids(dataset, split)[:maximum_groups]
    labels = group_payload["labels_audit_only"][accepted]
    variant_results = {}
    for mode in ("oracle_lqi", "nominal_lqi", "gru", "gru_reset"):
        if mode not in variants:
            continue
        variant_results[mode] = _simulate(
            mode, config, raw, labels, accepted, trials, duration_s, seed,
            checkpoint, device, pilot_height_mode,
            attitude_noise_deg, attitude_bias_deg, attitude_latency_steps,
            report_segments_s,
        )
    summaries = {}
    for mode, values in variant_results.items():
        summaries[mode] = {
            "safe_fraction": float(values["safe"].to(torch.float32).mean()),
            "survival_fraction": _summary(values["survival_fraction"]),
            "attitude_tracking_rms_rad": _summary(values["attitude_tracking_rms_rad"]),
            "rate_tracking_rms_rad_s": _summary(values["rate_tracking_rms_rad_s"]),
            "saturation_fraction": _summary(values["saturation_fraction"]),
            "command_movement_mean": _summary(values["command_movement_mean"]),
        }
    oracle = summaries["oracle_lqi"]
    gru = summaries["gru"]
    oracle_safe_mask = variant_results["oracle_lqi"]["safe"]
    paired_subsets = {}
    for mode, values in variant_results.items():
        selected_count = int(oracle_safe_mask.sum())
        paired_subsets[mode] = {
            "episodes": selected_count,
            "safe_fraction_given_oracle_safe": (
                float(values["safe"][oracle_safe_mask].to(torch.float32).mean())
                if selected_count
                else None
            ),
            "attitude_tracking_rms_rad_mean": (
                float(values["attitude_tracking_rms_rad"][oracle_safe_mask].mean())
                if selected_count
                else None
            ),
            "rate_tracking_rms_rad_s_mean": (
                float(values["rate_tracking_rms_rad_s"][oracle_safe_mask].mean())
                if selected_count
                else None
            ),
        }
    report = {
        "schema_version": 1,
        "comparison": "paired_unseen_airframe_closed_loop",
        "split": split,
        "student_arch": str(
            torch.load(checkpoint, map_location="cpu", weights_only=False)
            .get("model", {})
            .get("arch", "gru")
        ),
        "parameter_groups": len(accepted),
        "trials_per_group": trials,
        "duration_s": duration_s,
        "teacher_forcing": False,
        "global_rng_reseeded_per_variant": True,
        "pilot_height_mode": pilot_height_mode,
        "attitude_perturbation": {
            "noise_deg": attitude_noise_deg,
            "bias_deg": attitude_bias_deg,
            "latency_steps": attitude_latency_steps,
        },
        "report_segments_s": list(report_segments_s),
        "exogenous_pilot_schedule_identical": pilot_height_mode == "fixed_target",
        "variants": summaries,
        "segments": (
            {
                mode: variant_results[mode].get("segments")
                for mode in variant_results
            }
            if report_segments_s
            else None
        ),
        "hidden_norms": (
            {
                mode: variant_results[mode].get("hidden_norms")
                for mode in variant_results
            }
            if report_segments_s
            else None
        ),
        "oracle_safe_paired_subset": paired_subsets,
        "gru_vs_oracle": {
            "safe_fraction_delta": gru["safe_fraction"] - oracle["safe_fraction"],
            "attitude_rms_ratio": gru["attitude_tracking_rms_rad"]["mean"] / max(oracle["attitude_tracking_rms_rad"]["mean"], 1e-12),
            "rate_rms_ratio": gru["rate_tracking_rms_rad_s"]["mean"] / max(oracle["rate_tracking_rms_rad_s"]["mean"], 1e-12),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired nonlinear closed-loop LQI-GRU audit")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-groups", type=int, default=32)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--pilot-height-mode",
        choices=("training_truth", "fixed_target"),
        default="fixed_target",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=("oracle_lqi", "nominal_lqi", "gru", "gru_reset"),
    )
    parser.add_argument("--attitude-noise-deg", type=float, default=0.0)
    parser.add_argument("--attitude-bias-deg", type=float, default=0.0)
    parser.add_argument("--attitude-latency-steps", type=int, default=0)
    parser.add_argument("--report-segments-s", type=float, nargs="+", default=())
    args = parser.parse_args()
    report = evaluate_closed_loop(
        Path(args.dataset).resolve(), Path(args.config).resolve(),
        Path(args.checkpoint).resolve(), Path(args.output).resolve(),
        torch.device(args.device), args.maximum_groups, args.trials,
        args.duration_s, args.seed, args.pilot_height_mode, args.split,
        tuple(args.variants),
        args.attitude_noise_deg, args.attitude_bias_deg,
        args.attitude_latency_steps, tuple(args.report_segments_s),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
