from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import load_experiment_config
from .deployment import AdaptiveLQRScheduler
from .experiment import (
    FEATURE_NAMES,
    FirstOrderServoObserver,
    _apply_effectiveness_labels,
    _expand_mapping,
    _sample_initial_state,
    _set_initial_observation_state,
    _yaw_free_attitude,
)


@torch.no_grad()
def _simulate(
    experiment: Any,
    physical_labels: torch.Tensor,
    initial_attitude: torch.Tensor,
    initial_rate: torch.Tensor,
    duration_s: float,
    interpolation_s: float,
    scheduler: AdaptiveLQRScheduler | None,
) -> dict[str, torch.Tensor]:
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

    device = torch.device(experiment.device)
    dtype = torch.float32 if experiment.dtype == "float32" else torch.float64
    count = len(physical_labels)
    nominal_materialized = load_and_materialize(
        experiment.simulator_config, 1, device, dtype
    )
    nominal_parameters = _expand_mapping(nominal_materialized.parameters, count)
    actual_parameters = _apply_effectiveness_labels(
        nominal_parameters, physical_labels.to(device)
    )
    initial_state = _expand_mapping(nominal_materialized.initial_state, count)
    initial_state["attitude_q_wb"].copy_(initial_attitude.to(device))
    initial_state["angular_velocity_b"].copy_(initial_rate.to(device))
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
        _set_initial_observation_state(
            environment, controller.trim.motor_speed, initial_state["angular_velocity_b"]
        )
        servo_observer = FirstOrderServoObserver(
            nominal_parameters["servos.pwm_angle_table"],
            nominal_parameters["servos.tau"],
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
        history_steps = experiment.control_hz
        history = torch.empty(
            count, history_steps, len(FEATURE_NAMES), device=device, dtype=dtype
        )
        total_steps = round(duration_s * experiment.control_hz)
        interpolation_steps = max(1, round(interpolation_s * experiment.control_hz))
        settled_count = torch.zeros(count, device=device, dtype=torch.int64)
        first_unsafe_step = torch.full(
            (count,), total_steps + 1, device=device, dtype=torch.int64
        )
        saturation_steps = torch.zeros_like(settled_count)
        invalid = torch.zeros(count, device=device, dtype=torch.bool)
        target_gain = None
        accepted = torch.zeros(count, device=device, dtype=torch.bool)
        gate_candidate = torch.zeros_like(accepted)
        stability_probability = torch.zeros(count, device=device, dtype=dtype)
        synthesis_valid = torch.zeros(count, device=device, dtype=torch.bool)
        for step in range(total_steps):
            if step == history_steps and scheduler is not None:
                scheduled = scheduler.schedule(history)
                prefix_safe = first_unsafe_step > history_steps
                prefix_saturation = saturation_steps.to(dtype) / history_steps
                finite_history = torch.isfinite(history).all(dim=(1, 2))
                gate_candidate = scheduled.gate_candidate
                accepted = (
                    scheduled.accepted
                    & prefix_safe
                    & finite_history
                    & (prefix_saturation <= 0.25)
                )
                nominal = scheduled.nominal_gain[None].expand(count, -1, -1)
                target_gain = torch.where(
                    accepted[:, None, None], scheduled.target_gain, nominal
                )
                stability_probability = scheduled.stability_probability
                synthesis_valid = scheduled.synthesis_valid
            if target_gain is not None:
                alpha = min(1.0, (step - history_steps + 1) / interpolation_steps)
                nominal = torch.as_tensor(
                    scheduler.nominal_gain_numpy,
                    device=device,
                    dtype=dtype,
                )[None]
                controller.schedule_lqr_gain(
                    nominal + alpha * (target_gain - nominal)
                )
            truth = environment.observe("truth").values
            if experiment.measurement_mode == "simulated_sensors":
                sensors = environment.observe("sensor").values
                measured_rate = sensors["gyro"]
                measured_motor = sensors["motor_speed"]
            else:
                measured_rate = truth["angular_velocity_b"]
                measured_motor = truth["motor_speed"]
            tilt_attitude = _yaw_free_attitude(truth["attitude_q_wb"])
            state = ControllerState(
                position_n=truth["position_n"],
                velocity_n=truth["velocity_n"],
                attitude_q_wb=tilt_attitude,
                angular_velocity_b=measured_rate,
                linear_acceleration_n=truth["linear_acceleration_n"],
                motor_speed=measured_motor,
                servo_angle=servo_observer.angle,
            )
            output = controller.step(state, reference)
            if step < history_steps:
                history[:, step].copy_(
                    torch.cat(
                        (
                            tilt_attitude,
                            measured_rate,
                            measured_motor,
                            output.command,
                        ),
                        dim=1,
                    )
                )
            attitude_error = quaternion_rotation_error(
                tilt_attitude, identity
            )[:, :2].norm(dim=1)
            angular_rate = truth["angular_velocity_b"].norm(dim=1)
            tilt = torch.acos(tilt_cosine(truth["attitude_q_wb"]))
            settled = (
                (attitude_error <= experiment.convergence.maximum_roll_pitch_error_rad)
                & (angular_rate <= experiment.convergence.maximum_angular_rate_rad_s)
            )
            settled_count = torch.where(
                settled, settled_count + 1, torch.zeros_like(settled_count)
            )
            unsafe = (
                (tilt > experiment.convergence.safety_tilt_rad)
                | (angular_rate > experiment.convergence.safety_angular_rate_rad_s)
                | invalid
            )
            first_unsafe_step = torch.where(
                unsafe & (first_unsafe_step > total_steps),
                torch.full_like(first_unsafe_step, step + 1),
                first_unsafe_step,
            )
            command = output.command
            saturated = (
                (command[:, :2] <= 1e-6).any(dim=1)
                | (command[:, :2] >= 1.0 - 1e-6).any(dim=1)
                | (command[:, 2:].abs() >= 1.0 - 1e-6).any(dim=1)
            )
            saturation_steps += saturated.to(torch.int64)
            servo_observer.advance(command[:, 2:])
            result = environment.advance(command)
            invalid |= ~result.valid
        return {
            "history": history.cpu(),
            "safe": (first_unsafe_step > total_steps).cpu(),
            "safe_at_identification": (first_unsafe_step > history_steps).cpu(),
            "converged": (
                settled_count
                >= round(experiment.convergence.hold_s * experiment.control_hz)
            ).cpu(),
            "saturation_fraction": (
                saturation_steps.to(dtype) / total_steps
            ).cpu(),
            "gate_candidate": gate_candidate.cpu(),
            "accepted": accepted.cpu(),
            "stability_probability": stability_probability.cpu(),
            "synthesis_valid": synthesis_valid.cpu(),
        }
    finally:
        environment.close()


def _fraction(value: torch.Tensor) -> float:
    return float(value.to(torch.float32).mean()) if value.numel() else 0.0


def evaluate_nonlinear(args: argparse.Namespace) -> Mapping[str, Any]:
    experiment = load_experiment_config(args.experiment_config)
    groups = torch.load(
        Path(args.parameter_groups).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    test_mask = groups["split_assignment"] == 2
    labels = groups["labels"][test_mask]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    permutation = torch.randperm(len(labels), generator=generator)
    labels = labels[permutation[: min(args.samples, len(labels))]]
    attitude, rate = _sample_initial_state(
        len(labels), experiment, generator, torch.float32
    )
    scheduler = AdaptiveLQRScheduler(
        args.deployment_artifact,
        experiment.simulator_config,
        experiment.controller_config,
        experiment.device,
    )
    adaptive = _simulate(
        experiment,
        labels,
        attitude,
        rate,
        args.duration_s,
        args.interpolation_s,
        scheduler,
    )
    nominal = _simulate(
        experiment,
        labels,
        attitude,
        rate,
        args.duration_s,
        args.interpolation_s,
        None,
    )
    accepted = adaptive["accepted"]
    prefix_difference = (adaptive["history"] - nominal["history"]).abs().max()
    report = {
        "schema_version": 1,
        "semantics": (
            "Held-out parameter groups with newly sampled initial conditions. "
            "Both runs share the same initial state and deterministic sensor stream. "
            "Identification occurs after one nominal-LQR second and accepted gains "
            "are linearly interpolated."
        ),
        "sample_count": len(labels),
        "duration_s": args.duration_s,
        "interpolation_s": args.interpolation_s,
        "artifact_gate_enabled": scheduler.gate_enabled,
        "prefix_max_absolute_difference": float(prefix_difference),
        "gate_candidate_count": int(adaptive["gate_candidate"].sum()),
        "synthesis_invalid_count": int((~adaptive["synthesis_valid"]).sum()),
        "accepted_count": int(accepted.sum()),
        "nominal": {
            "safe_fraction": _fraction(nominal["safe"]),
            "converged_fraction": _fraction(nominal["converged"]),
        },
        "adaptive": {
            "safe_fraction": _fraction(adaptive["safe"]),
            "converged_fraction": _fraction(adaptive["converged"]),
        },
        "accepted_subset": {
            "nominal_safe_fraction": _fraction(nominal["safe"][accepted]),
            "adaptive_safe_fraction": _fraction(adaptive["safe"][accepted]),
            "nominal_converged_fraction": _fraction(nominal["converged"][accepted]),
            "adaptive_converged_fraction": _fraction(adaptive["converged"][accepted]),
            "rescued_safety_count": int(
                (accepted & ~nominal["safe"] & adaptive["safe"]).sum()
            ),
            "harmed_safety_count": int(
                (accepted & nominal["safe"] & ~adaptive["safe"]).sum()
            ),
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nonlinear adaptive-LQR audit")
    parser.add_argument("--deployment-artifact", required=True)
    parser.add_argument("--parameter-groups", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--experiment-config",
        default="Identification/configs/lqr_wide_adaptation_v2.yaml",
    )
    parser.add_argument("--samples", type=int, default=3490)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--interpolation-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def main() -> None:
    evaluate_nonlinear(build_parser().parse_args())


if __name__ == "__main__":
    main()
