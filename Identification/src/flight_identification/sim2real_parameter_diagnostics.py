from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import load_experiment_config
from .control_evaluation import _closed_loop_result, _discrete_model, _lqr_gain
from .gain_training import _lqr_weights
from .repeated_trial_deployment import RepeatedTrialLQRScheduler
from .repeated_trial_training import (
    _fixed_trial_mask,
    _retain_converged_trials,
    load_repeated_split,
)


ACTUATORS = ("motor_upper", "motor_lower", "servo_1", "servo_2", "servo_3")
AXES = ("roll", "pitch", "yaw")


def _effectiveness_name(axis: str, actuator: str) -> str:
    return f"angular_acceleration_effectiveness.{axis}.{actuator}"


BLOCKS = {
    "motor_effectiveness": {
        _effectiveness_name(axis, actuator)
        for axis in AXES
        for actuator in ACTUATORS[:2]
    },
    "servo_effectiveness": {
        _effectiveness_name(axis, actuator)
        for axis in AXES
        for actuator in ACTUATORS[2:]
    },
    "motor_time_constants": {
        "log10.motor_tau_upper_s",
        "log10.motor_tau_lower_s",
    },
    "servo_time_constants": {
        "log10.servo_tau_1_s",
        "log10.servo_tau_2_s",
        "log10.servo_tau_3_s",
    },
    "motor_command_slopes": {
        "log10.command_slope.motor_upper",
        "log10.command_slope.motor_lower",
    },
    "returned_but_unused": {
        "trim_angular_acceleration_bias.roll",
        "trim_angular_acceleration_bias.pitch",
        "log10.maximum_thrust_to_weight",
    },
}
GAIN_INPUT_NAMES = set().union(
    BLOCKS["motor_effectiveness"],
    BLOCKS["servo_effectiveness"],
    BLOCKS["motor_time_constants"],
    BLOCKS["servo_time_constants"],
    BLOCKS["motor_command_slopes"],
)


def _model_from_values(
    values: Mapping[str, float],
    nominal_effectiveness: np.ndarray,
    nominal_tau: np.ndarray,
    nominal_slopes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    effectiveness = nominal_effectiveness.copy()
    for row, axis in enumerate(AXES):
        for column, actuator in enumerate(ACTUATORS):
            name = _effectiveness_name(axis, actuator)
            if name in values:
                effectiveness[row, column] = values[name]
    tau_names = (
        "log10.motor_tau_upper_s",
        "log10.motor_tau_lower_s",
        "log10.servo_tau_1_s",
        "log10.servo_tau_2_s",
        "log10.servo_tau_3_s",
    )
    tau = nominal_tau.copy()
    for index, name in enumerate(tau_names):
        if name in values:
            tau[index] = 10.0 ** values[name]
    slopes = nominal_slopes.copy()
    for index, name in enumerate(
        ("log10.command_slope.motor_upper", "log10.command_slope.motor_lower")
    ):
        if name in values:
            slopes[index] = 10.0 ** values[name]
    return _discrete_model(
        effectiveness, slopes, tau, 1.0 / 500.0, integral=True
    )


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
    }


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    experiment = load_experiment_config(args.experiment_config)
    split = load_repeated_split(Path(args.dataset).expanduser().resolve(), "test", 1)
    _retain_converged_trials(split)
    scheduler = RepeatedTrialLQRScheduler(
        args.checkpoint,
        experiment.simulator_config,
        experiment.controller_config,
        args.device,
    )
    trial_mask = _fixed_trial_mask(split["trial_mask"], args.trial_count)
    synthesis = scheduler.synthesize(
        split["features"], split["valid_mask"], trial_mask
    )
    prediction = synthesis.effective_log10.to(torch.float64).numpy()
    label_min = scheduler.checkpoint["label_min"].numpy()
    label_max = scheduler.checkpoint["label_max"].numpy()
    clipped_prediction = np.clip(prediction, label_min, label_max)
    prediction_names = tuple(scheduler.label_names)

    full_names = tuple(split["target_names"])
    true_full = split["targets"].to(torch.float64).numpy()
    true_identification = true_full[
        :, [full_names.index(name) for name in prediction_names]
    ]
    squared_error = (prediction - true_identification) ** 2
    target_variance = np.sum(
        (true_identification - true_identification.mean(axis=0)) ** 2, axis=0
    )
    r2 = 1.0 - squared_error.sum(axis=0) / np.maximum(target_variance, 1e-12)

    per_target = []
    for index, name in enumerate(prediction_names):
        block = next(block for block, names in BLOCKS.items() if name in names)
        per_target.append(
            {
                "name": name,
                "block": block,
                "consumed_by_gain_synthesis": name in GAIN_INPUT_NAMES,
                "r2": float(r2[index]),
                "normalized_rmse": float(
                    np.sqrt(squared_error[:, index].mean())
                    / max(np.std(true_identification[:, index], ddof=1), 1e-12)
                ),
                "true_standard_deviation": float(
                    np.std(true_identification[:, index], ddof=1)
                ),
                "true_p05": float(np.quantile(true_identification[:, index], 0.05)),
                "true_p95": float(np.quantile(true_identification[:, index], 0.95)),
            }
        )
    block_identification = {
        block: {
            "target_count": len(names & set(prediction_names)),
            "mean_r2": float(
                np.mean(
                    [
                        item["r2"]
                        for item in per_target
                        if item["block"] == block
                    ]
                )
            ),
            "consumed_by_gain_synthesis": bool(names & GAIN_INPUT_NAMES),
        }
        for block, names in BLOCKS.items()
    }

    nominal_effectiveness = scheduler.nominal_effectiveness
    nominal_tau = scheduler.nominal_tau
    nominal_slopes = scheduler.command_slopes
    q, r = _lqr_weights(experiment.controller_config)
    initial_covariance = np.linalg.inv(q)
    nominal_a, nominal_b = _model_from_values(
        {}, nominal_effectiveness, nominal_tau, nominal_slopes
    )
    nominal_gain = _lqr_gain(nominal_a, nominal_b, q, r)

    true_values = [dict(zip(full_names, row)) for row in true_full]
    predicted_values = [dict(zip(prediction_names, row)) for row in clipped_prediction]
    variants = {
        "nominal": ("nominal", set()),
        "predicted_full": ("predicted", GAIN_INPUT_NAMES),
        "oracle_full": ("oracle", GAIN_INPUT_NAMES),
    }
    for block in (
        "motor_effectiveness",
        "servo_effectiveness",
        "motor_time_constants",
        "servo_time_constants",
        "motor_command_slopes",
    ):
        variants[f"predicted_{block}_only"] = ("predicted", BLOCKS[block])
        variants[f"oracle_{block}_only"] = ("oracle", BLOCKS[block])
    variants["oracle_all_effectiveness"] = (
        "oracle",
        BLOCKS["motor_effectiveness"] | BLOCKS["servo_effectiveness"],
    )
    variants["oracle_all_time_constants"] = (
        "oracle",
        BLOCKS["motor_time_constants"] | BLOCKS["servo_time_constants"],
    )

    variant_gains: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    radii: dict[str, list[float]] = {name: [] for name in variants}
    costs: dict[str, list[float]] = {name: [] for name in variants}
    oracle_costs = []
    nominal_costs = []
    true_models = []
    for sample_index, true_value in enumerate(true_values):
        true_a, true_b = _model_from_values(
            true_value, nominal_effectiveness, nominal_tau, nominal_slopes
        )
        true_models.append((true_a, true_b))
        oracle_gain = _lqr_gain(true_a, true_b, q, r)
        _, oracle_cost = _closed_loop_result(
            true_a, true_b, oracle_gain, q, r, initial_covariance
        )
        _, nominal_cost = _closed_loop_result(
            true_a, true_b, nominal_gain, q, r, initial_covariance
        )
        oracle_costs.append(oracle_cost)
        nominal_costs.append(nominal_cost)
        for variant, (source, selected_names) in variants.items():
            if source == "nominal":
                gain = nominal_gain
            else:
                source_values = (
                    true_value if source == "oracle" else predicted_values[sample_index]
                )
                selected = {
                    name: source_values[name]
                    for name in selected_names
                    if name in source_values
                }
                a, b = _model_from_values(
                    selected, nominal_effectiveness, nominal_tau, nominal_slopes
                )
                gain = _lqr_gain(a, b, q, r)
            radius, cost = _closed_loop_result(
                true_a, true_b, gain, q, r, initial_covariance
            )
            variant_gains[variant].append(gain)
            radii[variant].append(radius)
            costs[variant].append(cost)

    nominal_cost_array = np.asarray(nominal_costs)
    oracle_cost_array = np.asarray(oracle_costs)
    local_control = {}
    for variant in variants:
        cost_array = np.asarray(costs[variant])
        gain_array = np.stack(variant_gains[variant])
        oracle_gain_array = np.stack(variant_gains["oracle_full"])
        local_control[variant] = {
            "stable_fraction": float(np.mean(np.asarray(radii[variant]) < 1.0)),
            "pole_radius": _quantiles(radii[variant]),
            "cost_ratio_to_nominal": _quantiles(cost_array / nominal_cost_array),
            "oracle_gap_closed_fraction_mean": float(
                np.mean(
                    (nominal_cost_array - cost_array)
                    / np.maximum(nominal_cost_array - oracle_cost_array, 1e-12)
                )
            ),
            "relative_gain_delta_from_nominal": _quantiles(
                np.linalg.norm(gain_array - nominal_gain, axis=(1, 2))
                / np.linalg.norm(nominal_gain)
            ),
            "relative_gain_error_to_oracle": _quantiles(
                np.linalg.norm(gain_array - oracle_gain_array, axis=(1, 2))
                / np.maximum(np.linalg.norm(oracle_gain_array, axis=(1, 2)), 1e-12)
            ),
        }

    per_target_control_sensitivity = []
    for item in per_target:
        name = item["name"]
        target_costs = []
        target_gains = []
        if name in GAIN_INPUT_NAMES:
            for true_value, (true_a, true_b) in zip(true_values, true_models):
                model_a, model_b = _model_from_values(
                    {name: true_value[name]},
                    nominal_effectiveness,
                    nominal_tau,
                    nominal_slopes,
                )
                gain = _lqr_gain(model_a, model_b, q, r)
                _, cost = _closed_loop_result(
                    true_a, true_b, gain, q, r, initial_covariance
                )
                target_gains.append(gain)
                target_costs.append(cost)
            target_cost_array = np.asarray(target_costs)
            target_gain_array = np.stack(target_gains)
            cost_ratio = _quantiles(target_cost_array / nominal_cost_array)
            gap_closed = float(
                np.mean(
                    (nominal_cost_array - target_cost_array)
                    / np.maximum(nominal_cost_array - oracle_cost_array, 1e-12)
                )
            )
            gain_delta = _quantiles(
                np.linalg.norm(target_gain_array - nominal_gain, axis=(1, 2))
                / np.linalg.norm(nominal_gain)
            )
        else:
            cost_ratio = _quantiles(np.ones_like(nominal_cost_array))
            gap_closed = 0.0
            gain_delta = _quantiles(np.zeros_like(nominal_cost_array))
        per_target_control_sensitivity.append(
            {
                "name": name,
                "block": item["block"],
                "r2": item["r2"],
                "consumed_by_gain_synthesis": name in GAIN_INPUT_NAMES,
                "oracle_only_cost_ratio_to_nominal": cost_ratio,
                "oracle_gap_closed_fraction_mean": gap_closed,
                "relative_gain_delta_from_nominal": gain_delta,
                "interpretation_limit": (
                    "One-target-at-a-time local sensitivity; correlated and "
                    "combined parameter effects are represented by block ablations."
                ),
            }
        )

    true_effectiveness = true_full[:, :15].reshape(-1, 3, 5)
    true_slopes = np.concatenate(
        (
            10.0 ** true_full[:, 23:25],
            np.broadcast_to(nominal_slopes[None, 2:], (len(true_full), 3)),
        ),
        axis=1,
    )
    dc_authority = true_effectiveness * true_slopes[:, None, :]
    authority = {
        "motor_frobenius_norm": _quantiles(
            np.linalg.norm(dc_authority[:, :, :2], axis=(1, 2))
        ),
        "servo_frobenius_norm": _quantiles(
            np.linalg.norm(dc_authority[:, :, 2:], axis=(1, 2))
        ),
        "per_axis_motor_norm_median": {
            axis: float(np.median(np.linalg.norm(dc_authority[:, row, :2], axis=1)))
            for row, axis in enumerate(AXES)
        },
        "per_axis_servo_norm_median": {
            axis: float(np.median(np.linalg.norm(dc_authority[:, row, 2:], axis=1)))
            for row, axis in enumerate(AXES)
        },
    }

    report = {
        "schema_version": 2,
        "test_parameter_groups": len(true_full),
        "trial_count_cap": args.trial_count,
        "parameter_consumption": {
            "gain_inputs": sorted(GAIN_INPUT_NAMES),
            "returned_but_unused_by_current_controller": sorted(
                BLOCKS["returned_but_unused"]
            ),
            "note": (
                "Trim bias and thrust-to-weight are returned by the scheduler but "
                "are not applied as feedforward, trim, or collective adaptation."
            ),
        },
        "block_identification": block_identification,
        "per_target_identification": per_target,
        "per_target_local_control_sensitivity": per_target_control_sensitivity,
        "command_to_angular_acceleration_authority": authority,
        "local_control_ablation": local_control,
        "nominal_to_oracle_cost_ratio": _quantiles(
            nominal_cost_array / oracle_cost_array
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parameter identifiability and LQI sensitivity diagnostics"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trial-count", type=int, default=8)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
