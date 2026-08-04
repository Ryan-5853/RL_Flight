from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import expm, solve_discrete_are, solve_discrete_lyapunov
import torch

from .training import (
    _aggregate_by_group,
    _normalized_tensors,
    _predict,
    apply_target_mode,
    create_identifier_model,
    lqr_latent_to_effective,
    load_split,
)


def _nominal_actuator_model(
    simulator_config: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from flight_controller.math import lookup
    from flight_controller.plant import LocalPlantModel
    from simenv.config import load_and_materialize

    materialized = load_and_materialize(
        simulator_config, 1, torch.device("cpu"), torch.float64
    )
    parameters = materialized.parameters
    plant = LocalPlantModel(parameters)
    trim = plant.hover_trim()
    inertia = parameters["body.inertia_diagonal_b"][0]
    actuator_state = torch.cat(
        (trim.motor_speed[0], torch.zeros(3, dtype=torch.float64))
    )

    def angular_acceleration(value: torch.Tensor) -> torch.Tensor:
        moment = plant.wrench_from_actuators(
            value[:2][None], value[2:][None]
        )[0, 1:]
        return moment / inertia

    effectiveness = torch.autograd.functional.jacobian(
        angular_acceleration, actuator_state
    )
    trim_command = trim.command[0]

    def actuator_target(command: torch.Tensor) -> torch.Tensor:
        motor = lookup(
            command[:2][None], parameters["motors.pwm_to_rpm_table"]
        )[0]
        servo = lookup(
            command[2:][None], parameters["servos.pwm_angle_table"]
        )[0]
        return torch.cat((motor, servo))

    target_jacobian = torch.autograd.functional.jacobian(
        actuator_target, trim_command
    )
    command_slopes = target_jacobian.diag()
    time_constants = torch.cat(
        (parameters["motors.time_constant"][0], parameters["servos.tau"][0])
    )
    return (
        effectiveness.detach().numpy(),
        command_slopes.detach().numpy(),
        time_constants.detach().numpy(),
    )


def _command_slopes_for_thrust_scale(
    simulator_config: Path,
    thrust_scales: np.ndarray,
) -> np.ndarray:
    """Evaluate the command-to-actuator slopes at each scaled hover trim."""
    from flight_controller.plant import LocalPlantModel
    from simenv.config import load_and_materialize

    scales = np.asarray(thrust_scales, dtype=np.float64)
    if scales.ndim != 1 or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("thrust scales must be a finite positive vector")
    materialized = load_and_materialize(
        simulator_config, 1, torch.device("cpu"), torch.float64
    )
    parameters = materialized.parameters
    trim_speed = LocalPlantModel(parameters).hover_trim().motor_speed[0].numpy()
    target_speed = trim_speed[None, :] / np.sqrt(scales[:, None])
    table = parameters["motors.pwm_to_rpm_table"][0].numpy()
    motor_slopes = np.empty_like(target_speed)
    for motor_index in range(2):
        x = table[motor_index, :, 0]
        y = table[motor_index, :, 1]
        segment = np.searchsorted(y, target_speed[:, motor_index], side="right") - 1
        segment = np.clip(segment, 0, len(y) - 2)
        motor_slopes[:, motor_index] = (
            (y[segment + 1] - y[segment]) / (x[segment + 1] - x[segment])
        )
    _, nominal_slopes, _ = _nominal_actuator_model(simulator_config)
    return np.concatenate(
        (
            motor_slopes,
            np.broadcast_to(nominal_slopes[None, 2:], (len(scales), 3)),
        ),
        axis=1,
    )


def _scaled_actuator_model(
    effective_labels: np.ndarray,
    nominal_effectiveness: np.ndarray,
    nominal_time_constants: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if effective_labels.shape != (14,):
        raise ValueError("effective label vector must have length 14")
    effectiveness = nominal_effectiveness.copy()
    scales = np.power(10.0, effective_labels)
    entries = (
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 2),
        (2, 3),
        (2, 4),
    )
    for index, (row, column) in enumerate(entries):
        effectiveness[row, column] *= scales[index]
    # The upper/lower motor yaw columns share one reaction-torque scale.
    effectiveness[2, 1] *= scales[5]
    return effectiveness, nominal_time_constants * scales[9:14]


def _discrete_model(
    effectiveness: np.ndarray,
    command_slopes: np.ndarray,
    time_constants: np.ndarray,
    dt: float,
    integral: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if integral:
        continuous_a = np.zeros((13, 13), dtype=np.float64)
        continuous_b = np.zeros((13, 5), dtype=np.float64)
        continuous_a[0, 2] = 1.0
        continuous_a[1, 3] = 1.0
        continuous_a[2:5, 5:10] = effectiveness
        continuous_a[5:10, 5:10] = np.diag(-1.0 / time_constants)
        continuous_b[5:10] = np.diag(command_slopes / time_constants)
        continuous_a[10, 0] = 1.0
        continuous_a[11, 1] = 1.0
        continuous_a[12, 4] = 1.0
        augmented = np.block(
            [
                [continuous_a, continuous_b],
                [np.zeros((5, 18), dtype=np.float64)],
            ]
        )
        discrete = expm(augmented * dt)
        return discrete[:13, :13], discrete[:13, 13:]
    # Exact zero-order-hold discretization for the rigid-body integrator chain
    # driven by five independent first-order actuator states.
    decay = np.exp(-dt / time_constants)
    actuator_state_integral = time_constants * (1.0 - decay)
    actuator_input_integral = command_slopes * (
        dt - actuator_state_integral
    )
    actuator_state_double_integral = (
        dt * time_constants
        - time_constants**2 * (1.0 - decay)
    )
    actuator_input_double_integral = command_slopes * (
        0.5 * dt * dt - actuator_state_double_integral
    )

    discrete_a = np.eye(10, dtype=np.float64)
    discrete_b = np.zeros((10, 5), dtype=np.float64)
    discrete_a[0, 2] = dt
    discrete_a[1, 3] = dt
    discrete_a[0:2, 5:10] = (
        effectiveness[0:2] * actuator_state_double_integral[None]
    )
    discrete_a[2:5, 5:10] = (
        effectiveness * actuator_state_integral[None]
    )
    discrete_a[5:10, 5:10] = np.diag(decay)
    discrete_b[0:2] = (
        effectiveness[0:2] * actuator_input_double_integral[None]
    )
    discrete_b[2:5] = effectiveness * actuator_input_integral[None]
    discrete_b[5:10] = np.diag(command_slopes * (1.0 - decay))
    return discrete_a, discrete_b


def _lqr_gain(
    a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray
) -> np.ndarray:
    try:
        solution = solve_discrete_are(a, b, q, r)
    except (np.linalg.LinAlgError, ValueError):
        solution = solve_discrete_are(a, b, q, r, balanced=False)
    return np.linalg.solve(r + b.T @ solution @ b, b.T @ solution @ a)


def _closed_loop_result(
    a: np.ndarray,
    b: np.ndarray,
    gain: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    initial_covariance: np.ndarray,
) -> tuple[float, float]:
    closed_loop = a - b @ gain
    radius = float(np.max(np.abs(np.linalg.eigvals(closed_loop))))
    if radius >= 1.0:
        return radius, float("inf")
    stage_cost = q + gain.T @ r @ gain
    value = solve_discrete_lyapunov(closed_loop.T, stage_cost)
    cost = float(np.trace(value @ initial_covariance))
    return radius, cost


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"mean": float("inf"), "median": float("inf"), "p90": float("inf")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
    }


def evaluate_control_value(args: argparse.Namespace) -> Mapping[str, Any]:
    from flight_controller import load_controller_config

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target_mode = checkpoint.get("target_mode")
    if target_mode not in {"lqr_effective", "lqr_latent"}:
        raise ValueError("control evaluation requires an LQR model checkpoint")
    split = load_split(
        dataset_path,
        args.split,
        downsample=int(checkpoint["downsample"]),
        minimum_start_step=round(args.minimum_start_s * 500),
        maximum_start_step=round(args.maximum_start_s * 500),
        minimum_information=None,
    )
    apply_target_mode(split, target_mode)
    normalized_features, _ = _normalized_tensors(
        split, checkpoint["normalization"]
    )
    model = create_identifier_model(
        str(checkpoint.get("architecture", "mlp")),
        int(checkpoint["history_steps"]),
        int(checkpoint["feature_count"]),
        len(checkpoint["label_names"]),
        tuple(checkpoint["hidden_sizes"]),
        0.0,
        int(checkpoint.get("tcn_channels", 128)),
    )
    model.load_state_dict(checkpoint["model_state"])
    normalized_prediction = _predict(
        model, normalized_features, torch.device("cpu"), args.batch_size
    )
    predictions = (
        normalized_prediction * checkpoint["normalization"]["label_std"]
        + checkpoint["normalization"]["label_mean"]
    )
    if args.aggregation == "group":
        evaluation_predictions, evaluation_labels = _aggregate_by_group(
            predictions, split["labels"], split["group_id"]
        )
    else:
        evaluation_predictions, evaluation_labels = predictions, split["labels"]
    if target_mode == "lqr_latent":
        evaluation_predictions = lqr_latent_to_effective(evaluation_predictions)
        evaluation_labels = lqr_latent_to_effective(evaluation_labels)

    simulator_config = Path(args.simulator_config).expanduser().resolve()
    controller_config = load_controller_config(
        Path(args.controller_config).expanduser().resolve()
    )
    lqr_config = controller_config["params"]["lqr"]
    state_scales = np.asarray(lqr_config["state_scales"], dtype=np.float64)
    input_scales = np.asarray(lqr_config["input_scales"], dtype=np.float64)
    q = np.diag(1.0 / state_scales**2)
    r = float(lqr_config["input_weight_scale"]) * np.diag(
        1.0 / input_scales**2
    )
    initial_covariance = np.diag(state_scales**2)
    nominal_effectiveness, command_slopes, nominal_tau = _nominal_actuator_model(
        simulator_config
    )
    nominal_a, nominal_b = _discrete_model(
        nominal_effectiveness, command_slopes, nominal_tau, 1.0 / 500.0
    )
    nominal_gain = _lqr_gain(nominal_a, nominal_b, q, r)

    variant_names = (
        "nominal",
        "predicted_blend_25",
        "predicted_blend_50",
        "predicted_blend_75",
        "predicted",
        "predicted_nominal_servo_tau",
        "predicted_nominal_all_tau",
        "oracle",
    )
    radii: dict[str, list[float]] = {name: [] for name in variant_names}
    costs: dict[str, list[float]] = {name: [] for name in radii}
    normalized_costs: dict[str, list[float]] = {name: [] for name in radii}
    stable_nominal_normalized_costs: dict[str, list[float]] = {
        name: [] for name in radii
    }
    gain_errors = []
    for prediction, label in zip(
        evaluation_predictions.numpy(), evaluation_labels.numpy()
    ):
        prediction = prediction.copy()
        prediction[:9] = np.clip(prediction[:9], -6.0, 6.0)
        prediction[9:14] = np.clip(prediction[9:14], -2.0, 2.0)
        true_effectiveness, true_tau = _scaled_actuator_model(
            label, nominal_effectiveness, nominal_tau
        )
        predicted_effectiveness, predicted_tau = _scaled_actuator_model(
            prediction, nominal_effectiveness, nominal_tau
        )
        nominal_servo_tau = predicted_tau.copy()
        nominal_servo_tau[2:] = nominal_tau[2:]
        true_a, true_b = _discrete_model(
            true_effectiveness, command_slopes, true_tau, 1.0 / 500.0
        )
        predicted_a, predicted_b = _discrete_model(
            predicted_effectiveness,
            command_slopes,
            predicted_tau,
            1.0 / 500.0,
        )
        predicted_nominal_servo_a, predicted_nominal_servo_b = _discrete_model(
            predicted_effectiveness,
            command_slopes,
            nominal_servo_tau,
            1.0 / 500.0,
        )
        predicted_nominal_all_tau_a, predicted_nominal_all_tau_b = _discrete_model(
            predicted_effectiveness,
            command_slopes,
            nominal_tau,
            1.0 / 500.0,
        )
        oracle_gain = _lqr_gain(true_a, true_b, q, r)
        predicted_gain = _lqr_gain(predicted_a, predicted_b, q, r)
        predicted_nominal_servo_gain = _lqr_gain(
            predicted_nominal_servo_a, predicted_nominal_servo_b, q, r
        )
        predicted_nominal_all_tau_gain = _lqr_gain(
            predicted_nominal_all_tau_a, predicted_nominal_all_tau_b, q, r
        )
        gains = {
            "nominal": nominal_gain,
            "predicted_blend_25": nominal_gain
            + 0.25 * (predicted_gain - nominal_gain),
            "predicted_blend_50": nominal_gain
            + 0.50 * (predicted_gain - nominal_gain),
            "predicted_blend_75": nominal_gain
            + 0.75 * (predicted_gain - nominal_gain),
            "predicted": predicted_gain,
            "predicted_nominal_servo_tau": predicted_nominal_servo_gain,
            "predicted_nominal_all_tau": predicted_nominal_all_tau_gain,
            "oracle": oracle_gain,
        }
        instance_costs = {}
        for name, gain in gains.items():
            radius, cost = _closed_loop_result(
                true_a, true_b, gain, q, r, initial_covariance
            )
            radii[name].append(radius)
            costs[name].append(cost)
            instance_costs[name] = cost
        for name in gains:
            normalized_costs[name].append(
                instance_costs[name] / instance_costs["nominal"]
            )
            if np.isfinite(instance_costs["nominal"]):
                stable_nominal_normalized_costs[name].append(
                    instance_costs[name] / instance_costs["nominal"]
                )
        gain_errors.append(
            np.linalg.norm(predicted_gain - oracle_gain)
            / np.linalg.norm(oracle_gain)
        )

    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "split": args.split,
        "aggregation": args.aggregation,
        "evaluation_count": len(evaluation_labels),
        "unique_group_count": int(split["group_id"].unique().numel()),
        "semantics": (
            "Each gain is evaluated on the true local discrete plant. Cost is "
            "the infinite-horizon LQR cost for covariance diag(state_scales^2). "
            "Predictions are clipped to the exact experiment support before DARE."
        ),
        "stable_fraction": {
            name: float(np.mean(np.asarray(values) < 1.0))
            for name, values in radii.items()
        },
        "pole_radius": {name: _summary(values) for name, values in radii.items()},
        "absolute_cost": {name: _summary(values) for name, values in costs.items()},
        "cost_relative_to_nominal": {
            name: _summary(values) for name, values in normalized_costs.items()
        },
        "cost_relative_to_nominal_on_nominal_stable": {
            name: _summary(values)
            for name, values in stable_nominal_normalized_costs.items()
        },
        "predicted_gain_relative_error": _summary(gain_errors),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate identified LQR models on true local plants"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--minimum-start-s", type=float, default=0.0)
    parser.add_argument("--maximum-start-s", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--aggregation", choices=("window", "group"), default="group")
    parser.add_argument("--simulator-config", default="SimEnv/configs/example.yaml")
    parser.add_argument(
        "--controller-config",
        default="Controller/configs/lqr_identification_nominal.yaml",
    )
    return parser


def main() -> None:
    evaluate_control_value(build_parser().parse_args())


if __name__ == "__main__":
    main()
