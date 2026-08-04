from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import binomtest

from .config import load_experiment_config
from .control_evaluation import _closed_loop_result, _discrete_model, _lqr_gain
from .gain_training import _lqr_weights
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
)
from .sim2real_parameter_diagnostics import (
    BLOCKS,
    _model_from_values,
)


def _model_from_target(
    target: np.ndarray, nominal_servo_slopes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    effectiveness = target[:15].reshape(3, 5)
    tau = np.power(10.0, target[18:23])
    slopes = np.concatenate(
        (np.power(10.0, target[23:25]), nominal_servo_slopes)
    )
    a, b = _discrete_model(
        effectiveness, slopes, tau, 1.0 / 500.0, integral=True
    )
    return a, b, tau


def _cluster_bootstrap_mean_ci(
    difference: torch.Tensor,
    group_count: int,
    repeats: int,
    seed: int,
    bootstrap_count: int = 5000,
) -> list[float]:
    per_group = (
        difference.to(torch.float64).reshape(group_count, repeats).mean(dim=1).numpy()
    )
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_count, dtype=np.float64)
    for start in range(0, bootstrap_count, 500):
        stop = min(start + 500, bootstrap_count)
        indices = generator.integers(0, group_count, size=(stop - start, group_count))
        bootstrap[start:stop] = per_group[indices].mean(axis=1)
    return [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]


def _paired_binary_summary(
    candidate: torch.Tensor,
    nominal: torch.Tensor,
    group_count: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    wins = int((candidate & ~nominal).sum())
    losses = int((~candidate & nominal).sum())
    discordant = wins + losses
    return {
        "candidate_fraction": float(candidate.to(torch.float32).mean()),
        "nominal_fraction": float(nominal.to(torch.float32).mean()),
        "fraction_delta": float(
            (candidate.to(torch.float32) - nominal.to(torch.float32)).mean()
        ),
        "group_cluster_bootstrap_95_ci": _cluster_bootstrap_mean_ci(
            candidate.to(torch.float32) - nominal.to(torch.float32),
            group_count,
            repeats,
            seed,
        ),
        "wins": wins,
        "losses": losses,
        "discordant_exact_binomial_p_value": (
            float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
        ),
    }


def _synthesize_selected_gains(
    source_rows: list[Mapping[str, float]],
    selected_names: set[str],
    scheduler: RepeatedTrialLQRScheduler,
    q: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    gains = []
    for values in source_rows:
        selected = {
            name: values[name] for name in selected_names if name in values
        }
        a, b = _model_from_values(
            selected,
            scheduler.nominal_effectiveness,
            scheduler.nominal_tau,
            scheduler.command_slopes,
        )
        gains.append(_lqr_gain(a, b, q, r))
    return np.stack(gains)


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    experiment = load_experiment_config(args.experiment_config)
    split = load_repeated_split(dataset, "test", 1)
    unfiltered_test_groups = len(split["features"])
    _retain_converged_trials(split)
    retained_test_groups = len(split["features"])
    available_groups = retained_test_groups
    count = min(args.maximum_parameter_groups, available_groups)
    for name, value in tuple(split.items()):
        if (
            isinstance(value, torch.Tensor)
            and value.ndim
            and len(value) == available_groups
        ):
            split[name] = value[:count]
    selected_trials = _fixed_trial_mask(split["trial_mask"], args.trial_count)

    scheduler = RepeatedTrialLQRScheduler(
        checkpoint,
        experiment.simulator_config,
        experiment.controller_config,
        args.device,
    )
    synthesis = scheduler.synthesize(
        split["features"], split["valid_mask"], selected_trials
    )
    q, r = _lqr_weights(experiment.controller_config)
    targets = split["targets"].numpy()
    target_names = tuple(split["target_names"])
    nominal_gain = synthesis.nominal_gain.numpy()
    predicted_gain = synthesis.predicted_gain.numpy()
    oracle_gain = []
    true_models = []
    oracle_servo_tau = []
    for target in targets:
        a, b, tau = _model_from_target(target, scheduler.command_slopes[2:])
        oracle_gain.append(_lqr_gain(a, b, q, r))
        true_models.append((a, b))
        oracle_servo_tau.append(tau[2:])
    oracle_gain_array = np.stack(oracle_gain)
    oracle_servo_tau_array = np.stack(oracle_servo_tau)

    prediction_names = tuple(scheduler.label_names)
    label_min = scheduler.checkpoint["label_min"].numpy()
    label_max = scheduler.checkpoint["label_max"].numpy()
    clipped_prediction = np.clip(
        synthesis.effective_log10.to(torch.float64).numpy(), label_min, label_max
    )
    predicted_values = [
        dict(zip(prediction_names, row)) for row in clipped_prediction
    ]
    true_values = [dict(zip(target_names, row)) for row in targets]
    all_effectiveness = BLOCKS["motor_effectiveness"] | BLOCKS["servo_effectiveness"]
    predicted_effectiveness_gain = _synthesize_selected_gains(
        predicted_values, all_effectiveness, scheduler, q, r
    )
    oracle_effectiveness_gain = _synthesize_selected_gains(
        true_values, all_effectiveness, scheduler, q, r
    )
    oracle_motor_effectiveness_gain = _synthesize_selected_gains(
        true_values, BLOCKS["motor_effectiveness"], scheduler, q, r
    )
    oracle_servo_effectiveness_gain = _synthesize_selected_gains(
        true_values, BLOCKS["servo_effectiveness"], scheduler, q, r
    )
    observer_tau = {
        "nominal": np.broadcast_to(scheduler.nominal_tau[2:], (count, 3)),
        "predicted": synthesis.servo_time_constant_s.numpy(),
        "predicted_gain_nominal_observer": np.broadcast_to(
            scheduler.nominal_tau[2:], (count, 3)
        ),
        "nominal_gain_oracle_observer": oracle_servo_tau_array,
        "oracle_gain_nominal_observer": np.broadcast_to(
            scheduler.nominal_tau[2:], (count, 3)
        ),
        "predicted_effectiveness_only": np.broadcast_to(
            scheduler.nominal_tau[2:], (count, 3)
        ),
        "oracle_effectiveness_only": np.broadcast_to(
            scheduler.nominal_tau[2:], (count, 3)
        ),
        "oracle_motor_effectiveness_only": np.broadcast_to(
            scheduler.nominal_tau[2:], (count, 3)
        ),
        "oracle_servo_effectiveness_only": np.broadcast_to(
            scheduler.nominal_tau[2:], (count, 3)
        ),
        "oracle": oracle_servo_tau_array,
    }
    gains = {
        "nominal": np.broadcast_to(nominal_gain, (count, *nominal_gain.shape)),
        "predicted": predicted_gain,
        "predicted_gain_nominal_observer": predicted_gain,
        "nominal_gain_oracle_observer": np.broadcast_to(
            nominal_gain, (count, *nominal_gain.shape)
        ),
        "oracle_gain_nominal_observer": oracle_gain_array,
        "predicted_effectiveness_only": predicted_effectiveness_gain,
        "oracle_effectiveness_only": oracle_effectiveness_gain,
        "oracle_motor_effectiveness_only": oracle_motor_effectiveness_gain,
        "oracle_servo_effectiveness_only": oracle_servo_effectiveness_gain,
        "oracle": oracle_gain_array,
    }

    variant_names = tuple(
        name.strip() for name in args.variants.split(",") if name.strip()
    )
    unknown_variants = set(variant_names) - set(gains)
    if not variant_names or "nominal" not in variant_names or unknown_variants:
        raise ValueError(
            "variants must include nominal and be a comma-separated subset of "
            f"{tuple(gains)}; unknown={sorted(unknown_variants)}"
        )
    gains = {name: gains[name] for name in variant_names}
    observer_tau = {name: observer_tau[name] for name in variant_names}

    initial_covariance = np.linalg.inv(q)
    local_radius = {name: [] for name in gains}
    for index, (a, b) in enumerate(true_models):
        for name, gain in gains.items():
            radius, _ = _closed_loop_result(
                a, b, gain[index], q, r, initial_covariance
            )
            local_radius[name].append(radius)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    repeats = args.evaluation_initial_conditions
    physical = split["labels"].repeat_interleave(repeats, dim=0)
    attitude, angular_velocity = _sample_near_equilibrium(
        len(physical),
        args.maximum_initial_tilt_rad,
        args.maximum_initial_rate_rad_s,
        generator,
        torch.float32,
    )
    evaluation_count = len(physical)
    variant_results = {}
    for name in variant_names:
        variant_results[name] = _simulate_gain(
            experiment,
            physical,
            attitude,
            angular_velocity,
            np.repeat(gains[name], repeats, axis=0),
            np.repeat(observer_tau[name], repeats, axis=0),
            np.ones(evaluation_count, dtype=np.bool_),
            args.duration_s,
        )
    nonlinear = {
        name: _nonlinear_summary(result)
        for name, result in variant_results.items()
    }
    nominal_result = variant_results["nominal"]
    paired_vs_nominal = {}
    for name, result in variant_results.items():
        if name == "nominal":
            continue
        mutually_safe = result["safe"] & nominal_result["safe"]
        paired_vs_nominal[name] = {
            "safety": _paired_binary_summary(
                result["safe"],
                nominal_result["safe"],
                count,
                repeats,
                args.seed + 1000,
            ),
            "convergence": _paired_binary_summary(
                result["converged"],
                nominal_result["converged"],
                count,
                repeats,
                args.seed + 2000,
            ),
            "median_final_attitude_error_delta_rad_given_mutual_safety": float(
                (
                    result["final_attitude_error"][mutually_safe]
                    - nominal_result["final_attitude_error"][mutually_safe]
                ).median()
            ),
            "median_final_rate_delta_rad_s_given_mutual_safety": float(
                (
                    result["final_rate"][mutually_safe]
                    - nominal_result["final_rate"][mutually_safe]
                ).median()
            ),
            "mean_saturation_fraction_delta": float(
                (
                    result["saturation_fraction"]
                    - nominal_result["saturation_fraction"]
                ).mean()
            ),
            "mean_saturation_fraction_delta_group_cluster_bootstrap_95_ci": (
                _cluster_bootstrap_mean_ci(
                    result["saturation_fraction"]
                    - nominal_result["saturation_fraction"],
                    count,
                    repeats,
                    args.seed + 3000,
                )
            ),
        }
    report = {
        "schema_version": 2,
        "deployment_mode": "shadow_only",
        "gain_updates_enabled": False,
        "deployment_decision": (
            "Candidate gains remain observational because predicted adaptation "
            "did not improve paired nonlinear convergence and increased saturation."
        ),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "unfiltered_test_parameter_groups": unfiltered_test_groups,
        "retained_test_parameter_groups": retained_test_groups,
        "retained_test_parameter_group_fraction": (
            retained_test_groups / unfiltered_test_groups
        ),
        "test_parameter_groups": count,
        "identification_trial_count_cap": args.trial_count,
        "evaluation_initial_conditions_per_group": repeats,
        "duration_s": args.duration_s,
        "maximum_initial_tilt_rad": args.maximum_initial_tilt_rad,
        "maximum_initial_rate_rad_s": args.maximum_initial_rate_rad_s,
        "nonlinear_variants": variant_names,
        "paired_noise_semantics": (
            "Each variant rebuilds SimEnv with the same seed, instance count, "
            "sample ordering, initial state, and noise counters. Sensor and motor "
            "noise are therefore paired by parameter group and initial condition."
        ),
        "synthesis_valid_fraction": float(
            synthesis.synthesis_valid.to(torch.float32).mean()
        ),
        "local_linear_stable_fraction": {
            name: float(np.mean(np.asarray(values) < 1.0))
            for name, values in local_radius.items()
        },
        "local_linear_pole_radius_p95": {
            name: float(np.quantile(values, 0.95))
            for name, values in local_radius.items()
        },
        "nonlinear": nonlinear,
        "paired_vs_nominal": paired_vs_nominal,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired nonlinear audit for the sim2real repeated-trial LQI"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-parameter-groups", type=int, default=256)
    parser.add_argument("--trial-count", type=int, default=8)
    parser.add_argument("--evaluation-initial-conditions", type=int, default=2)
    parser.add_argument("--maximum-initial-tilt-rad", type=float, default=0.2617994)
    parser.add_argument("--maximum-initial-rate-rad-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument(
        "--variants",
        default=(
            "nominal,predicted,predicted_gain_nominal_observer,"
            "nominal_gain_oracle_observer,oracle_gain_nominal_observer,"
            "predicted_effectiveness_only,oracle_effectiveness_only,"
            "oracle_motor_effectiveness_only,oracle_servo_effectiveness_only,oracle"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260824)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
