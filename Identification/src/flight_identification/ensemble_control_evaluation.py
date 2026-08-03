from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .control_evaluation import (
    _discrete_model,
    _lqr_gain,
    _nominal_actuator_model,
    _scaled_actuator_model,
)
from .ensemble import (
    _aggregate_member_predictions,
    calibrated_std,
    predict_checkpoint_members,
)
from .training import apply_target_mode, load_split


EFFECTIVE_LOWER = np.asarray([-6.0] * 9 + [-2.0] * 5, dtype=np.float64)
EFFECTIVE_UPPER = np.asarray([6.0] * 9 + [2.0] * 5, dtype=np.float64)


def _pole_radius(a: np.ndarray, b: np.ndarray, gain: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(a - b @ gain))))


def _wilson_lower(successes: int, count: int, z: float = 1.96) -> float:
    if count == 0:
        return 0.0
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = proportion + z * z / (2.0 * count)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    )
    return (center - spread) / denominator


def _load_window_predictions(
    artifact: Mapping[str, Any],
    dataset: Path,
    split_name: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = load_split(
        dataset,
        split_name,
        downsample=int(artifact["downsample"]),
        minimum_start_step=0,
        maximum_start_step=0,
        minimum_information=None,
    )
    apply_target_mode(split, str(artifact["target_mode"]))
    member_windows = predict_checkpoint_members(
        artifact["members"], split["features"], device, batch_size
    )
    uncertainty = calibrated_std(member_windows, artifact["calibration"])
    return (
        member_windows.mean(dim=0).numpy(),
        uncertainty.numpy(),
        split["labels"].numpy(),
        split["group_id"].numpy(),
    )


def _uncertainty_plants(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    multiplier: float,
) -> list[np.ndarray]:
    center = np.clip(mean, EFFECTIVE_LOWER, EFFECTIVE_UPPER)
    if multiplier == 0.0:
        return [center]
    plants = [center]
    for index in range(len(center)):
        lower = center.copy()
        upper = center.copy()
        lower[index] = max(
            EFFECTIVE_LOWER[index],
            center[index] - multiplier * standard_deviation[index],
        )
        upper[index] = min(
            EFFECTIVE_UPPER[index],
            center[index] + multiplier * standard_deviation[index],
        )
        plants.extend((lower, upper))
    return plants


def _evaluate_split(
    means: np.ndarray,
    standard_deviations: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    uncertainty_multipliers: Sequence[float],
    blends: Sequence[float],
    nominal_effectiveness: np.ndarray,
    command_slopes: np.ndarray,
    nominal_tau: np.ndarray,
    nominal_gain: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> dict[str, Any]:
    count = len(labels)
    nominal_stable = np.zeros(count, dtype=bool)
    true_stable = {
        blend: np.zeros(count, dtype=bool) for blend in blends
    }
    robust_radius = {
        (multiplier, blend): np.full(count, np.inf, dtype=np.float64)
        for multiplier in uncertainty_multipliers
        for blend in blends
    }
    gain_relative_step = np.zeros(count, dtype=np.float64)
    for sample_index, (mean, standard_deviation, label) in enumerate(
        zip(means, standard_deviations, labels)
    ):
        clipped_mean = np.clip(mean, EFFECTIVE_LOWER, EFFECTIVE_UPPER)
        true_effectiveness, true_tau = _scaled_actuator_model(
            label, nominal_effectiveness, nominal_tau
        )
        true_a, true_b = _discrete_model(
            true_effectiveness, command_slopes, true_tau, 1.0 / 500.0
        )
        nominal_stable[sample_index] = (
            _pole_radius(true_a, true_b, nominal_gain) < 1.0
        )
        predicted_effectiveness, predicted_tau = _scaled_actuator_model(
            clipped_mean, nominal_effectiveness, nominal_tau
        )
        predicted_a, predicted_b = _discrete_model(
            predicted_effectiveness,
            command_slopes,
            predicted_tau,
            1.0 / 500.0,
        )
        predicted_gain = _lqr_gain(predicted_a, predicted_b, q, r)
        gain_relative_step[sample_index] = np.linalg.norm(
            predicted_gain - nominal_gain
        ) / np.linalg.norm(nominal_gain)
        gains = {
            blend: nominal_gain + blend * (predicted_gain - nominal_gain)
            for blend in blends
        }
        for blend, gain in gains.items():
            true_stable[blend][sample_index] = (
                _pole_radius(true_a, true_b, gain) < 1.0
            )
        for multiplier in uncertainty_multipliers:
            worst = {blend: 0.0 for blend in blends}
            uncertain_models = []
            for uncertain_label in _uncertainty_plants(
                clipped_mean, standard_deviation, multiplier
            ):
                effectiveness, tau = _scaled_actuator_model(
                    uncertain_label, nominal_effectiveness, nominal_tau
                )
                uncertain_a, uncertain_b = _discrete_model(
                    effectiveness, command_slopes, tau, 1.0 / 500.0
                )
                uncertain_models.append((uncertain_a, uncertain_b))
            uncertain_as = np.stack([model[0] for model in uncertain_models])
            uncertain_bs = np.stack([model[1] for model in uncertain_models])
            for blend, gain in gains.items():
                closed_loop = uncertain_as - uncertain_bs @ gain
                worst[blend] = float(
                    np.max(np.abs(np.linalg.eigvals(closed_loop)))
                )
            for blend, radius in worst.items():
                robust_radius[(multiplier, blend)][sample_index] = radius
    return {
        "means": means,
        "standard_deviations": standard_deviations,
        "group_ids": group_ids,
        "nominal_stable": nominal_stable,
        "true_stable": true_stable,
        "robust_radius": robust_radius,
        "gain_relative_step": gain_relative_step,
    }


def _classifier_features(
    result: Mapping[str, Any],
    blend: float,
    multipliers: Sequence[float],
) -> np.ndarray:
    mean = np.clip(result["means"], EFFECTIVE_LOWER, EFFECTIVE_UPPER)
    standard_deviation = result["standard_deviations"]
    robust = np.stack(
        [result["robust_radius"][(multiplier, blend)] for multiplier in multipliers],
        axis=1,
    )
    return np.concatenate(
        (
            mean,
            standard_deviation,
            np.abs(mean),
            robust,
            result["gain_relative_step"][:, None],
        ),
        axis=1,
    ).astype(np.float32)


def _fit_risk_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    seed: int,
    calibration_fraction: float,
    epochs: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    unique_groups = np.unique(group_ids)
    permutation = generator.permutation(unique_groups)
    calibration_group_count = round(len(unique_groups) * calibration_fraction)
    calibration_groups = permutation[:calibration_group_count]
    calibration_mask = np.isin(group_ids, calibration_groups)
    calibration_indices = np.flatnonzero(calibration_mask)
    fit_indices = np.flatnonzero(~calibration_mask)
    fit_features = torch.from_numpy(features[fit_indices])
    fit_labels = torch.from_numpy(labels[fit_indices].astype(np.float32))[:, None]
    feature_mean = fit_features.mean(dim=0)
    feature_std = fit_features.std(dim=0).clamp_min(1e-5)
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(features.shape[1], 32),
        nn.SiLU(),
        nn.Linear(32, 16),
        nn.SiLU(),
        nn.Linear(16, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_function = nn.BCEWithLogitsLoss()
    normalized_fit = (fit_features - feature_mean) / feature_std
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(normalized_fit), fit_labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model((torch.from_numpy(features) - feature_mean) / feature_std)
        )[:, 0].numpy()
    checkpoint = {
        "input_count": features.shape[1],
        "hidden_sizes": (32, 16),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "model_state": model.state_dict(),
        "fit_count": len(fit_indices),
        "calibration_count": len(calibration_indices),
    }
    return checkpoint, probabilities, fit_indices, calibration_indices


def _classifier_probabilities(
    checkpoint: Mapping[str, Any], features: np.ndarray
) -> np.ndarray:
    model = nn.Sequential(
        nn.Linear(int(checkpoint["input_count"]), 32),
        nn.SiLU(),
        nn.Linear(32, 16),
        nn.SiLU(),
        nn.Linear(16, 1),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        value = torch.from_numpy(features)
        normalized = (
            value - checkpoint["feature_mean"]
        ) / checkpoint["feature_std"]
        return torch.sigmoid(model(normalized))[:, 0].numpy()


def _select_probability_threshold(
    probabilities: np.ndarray,
    stable: np.ndarray,
    minimum_accept_count: int,
    desired_precision: float,
) -> dict[str, Any]:
    order = np.argsort(-probabilities)
    ordered_stable = stable[order].astype(np.int64)
    cumulative_success = np.cumsum(ordered_stable)
    candidates = []
    for count in range(minimum_accept_count, len(order) + 1):
        successes = int(cumulative_success[count - 1])
        precision = successes / count
        threshold = float(probabilities[order[count - 1]])
        candidates.append(
            {
                "probability_threshold": threshold,
                "accepted_count": count,
                "coverage": count / len(order),
                "stable_fraction_when_accepted": precision,
                "stable_wilson_lower_95": _wilson_lower(successes, count),
                "target_met": precision >= desired_precision,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["target_met"]]
    pool = eligible if eligible else candidates
    return max(
        pool,
        key=lambda candidate: (
            candidate["coverage"]
            if eligible
            else candidate["stable_fraction_when_accepted"],
            candidate["stable_wilson_lower_95"],
        ),
    )


def _mask_report(
    result: Mapping[str, Any],
    blend: float,
    accepted: np.ndarray,
    population: np.ndarray | None = None,
) -> dict[str, Any]:
    if population is None:
        population = np.ones(len(accepted), dtype=bool)
    accepted = accepted & population
    updated_stable = result["true_stable"][blend]
    nominal_stable = result["nominal_stable"]
    policy_stable = np.where(accepted, updated_stable, nominal_stable)
    accepted_count = int(np.sum(accepted))
    successes = int(np.sum(accepted & updated_stable))
    population_count = int(np.sum(population))
    return {
        "count": population_count,
        "accepted_count": accepted_count,
        "coverage": accepted_count / population_count,
        "stable_fraction_when_accepted": successes / accepted_count
        if accepted_count
        else None,
        "stable_wilson_lower_95": _wilson_lower(successes, accepted_count),
        "nominal_stable_fraction": float(np.mean(nominal_stable[population])),
        "ungated_updated_stable_fraction": float(
            np.mean(updated_stable[population])
        ),
        "gated_policy_stable_fraction": float(np.mean(policy_stable[population])),
        "rescued_nominally_unstable_count": int(
            np.sum(accepted & ~nominal_stable & updated_stable)
        ),
        "harmed_nominally_stable_count": int(
            np.sum(accepted & nominal_stable & ~updated_stable)
        ),
    }


def _fit_learned_gate(
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    blends: Sequence[float],
    multipliers: Sequence[float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    for blend_index, blend in enumerate(blends):
        validation_features = _classifier_features(validation, blend, multipliers)
        checkpoint, probabilities, fit_indices, calibration_indices = (
            _fit_risk_classifier(
                validation_features,
                validation["true_stable"][blend],
                validation["group_ids"],
                args.seed + blend_index,
                args.gate_calibration_fraction,
                args.gate_epochs,
            )
        )
        threshold = _select_probability_threshold(
            probabilities[calibration_indices],
            validation["true_stable"][blend][calibration_indices],
            args.minimum_calibration_accept_count,
            args.desired_precision,
        )
        candidates.append(
            {
                "gain_blend": float(blend),
                "checkpoint": checkpoint,
                "threshold": threshold,
                "fit": _mask_report(
                    validation,
                    blend,
                    probabilities >= threshold["probability_threshold"],
                    np.isin(np.arange(len(probabilities)), fit_indices),
                ),
                "calibration": _mask_report(
                    validation,
                    blend,
                    probabilities >= threshold["probability_threshold"],
                    np.isin(np.arange(len(probabilities)), calibration_indices),
                ),
            }
        )
    eligible = [
        candidate for candidate in candidates if candidate["threshold"]["target_met"]
    ]
    pool = eligible if eligible else candidates
    selected = max(
        pool,
        key=lambda candidate: (
            candidate["threshold"]["coverage"]
            if eligible
            else candidate["threshold"]["stable_fraction_when_accepted"],
            candidate["threshold"]["stable_wilson_lower_95"],
        ),
    )
    blend = selected["gain_blend"]
    test_features = _classifier_features(test, blend, multipliers)
    test_probabilities = _classifier_probabilities(
        selected["checkpoint"], test_features
    )
    test_accepted = (
        test_probabilities >= selected["threshold"]["probability_threshold"]
    )
    report = {
        "gain_blend": blend,
        "probability_threshold": selected["threshold"]["probability_threshold"],
        "validation_fit": selected["fit"],
        "validation_calibration": selected["calibration"],
        "calibration_target_met": selected["threshold"]["target_met"],
        "test": _mask_report(test, blend, test_accepted),
    }
    checkpoint = dict(selected["checkpoint"])
    checkpoint.update(
        {
            "feature_semantics": (
                "clipped_mean, calibrated_std, absolute_clipped_mean, "
                "robust_radius_per_uncertainty_multiplier, gain_relative_step"
            ),
            "uncertainty_multipliers": tuple(multipliers),
            "gain_blend": blend,
            "probability_threshold": selected["threshold"]["probability_threshold"],
            "desired_precision": args.desired_precision,
            "calibration_target_met": selected["threshold"]["target_met"],
            "heldout_test_target_met": (
                report["test"]["stable_fraction_when_accepted"] is not None
                and report["test"]["stable_fraction_when_accepted"]
                >= args.desired_precision
            ),
        }
    )
    checkpoint["enabled"] = bool(
        checkpoint["calibration_target_met"]
        and checkpoint["heldout_test_target_met"]
    )
    return checkpoint, report


def _select_gate(
    result: Mapping[str, Any],
    multipliers: Sequence[float],
    blends: Sequence[float],
    desired_precision: float,
    minimum_accept_count: int,
    maximum_robust_radius: float,
) -> dict[str, Any]:
    candidates = []
    for multiplier in multipliers:
        for blend in blends:
            radii = result["robust_radius"][(multiplier, blend)]
            stable = result["true_stable"][blend]
            non_degraded = ~(result["nominal_stable"] & ~stable)
            quantiles = np.linspace(0.02, 1.0, 50)
            thresholds = np.unique(np.quantile(radii, quantiles))
            for threshold in thresholds:
                threshold = min(float(threshold), maximum_robust_radius)
                accepted = radii <= threshold
                accepted_count = int(np.sum(accepted))
                if accepted_count < minimum_accept_count:
                    continue
                successes = int(np.sum(non_degraded & accepted))
                precision = successes / accepted_count
                rescued = int(
                    np.sum(accepted & ~result["nominal_stable"] & stable)
                )
                candidates.append(
                    {
                        "uncertainty_multiplier": float(multiplier),
                        "gain_blend": float(blend),
                        "robust_radius_threshold": threshold,
                        "accepted_count": accepted_count,
                        "coverage": accepted_count / len(stable),
                        "non_degradation_fraction_when_accepted": precision,
                        "non_degradation_wilson_lower_95": _wilson_lower(
                            successes, accepted_count
                        ),
                        "rescued_count": rescued,
                        "target_met": precision >= desired_precision,
                    }
                )
    if not candidates:
        raise RuntimeError("no gate candidate accepts the requested minimum count")
    eligible = [candidate for candidate in candidates if candidate["target_met"]]
    pool = eligible if eligible else candidates
    return max(
        pool,
        key=lambda candidate: (
            candidate["coverage"]
            if eligible
            else candidate["non_degradation_fraction_when_accepted"],
            candidate["rescued_count"],
            candidate["non_degradation_wilson_lower_95"],
            -candidate["uncertainty_multiplier"],
        ),
    )


def _gate_report(
    result: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    blend = float(gate["gain_blend"])
    multiplier = float(gate["uncertainty_multiplier"])
    threshold = float(gate["robust_radius_threshold"])
    accepted = result["robust_radius"][(multiplier, blend)] <= threshold
    updated_stable = result["true_stable"][blend]
    nominal_stable = result["nominal_stable"]
    policy_stable = np.where(accepted, updated_stable, nominal_stable)
    accepted_count = int(np.sum(accepted))
    accepted_successes = int(np.sum(accepted & updated_stable))
    non_degraded = ~(nominal_stable & ~updated_stable)
    non_degraded_count = int(np.sum(accepted & non_degraded))
    return {
        "count": len(accepted),
        "accepted_count": accepted_count,
        "coverage": float(np.mean(accepted)),
        "stable_fraction_when_accepted": (
            accepted_successes / accepted_count if accepted_count else None
        ),
        "stable_wilson_lower_95": _wilson_lower(
            accepted_successes, accepted_count
        ),
        "non_degradation_fraction_when_accepted": (
            non_degraded_count / accepted_count if accepted_count else None
        ),
        "non_degradation_wilson_lower_95": _wilson_lower(
            non_degraded_count, accepted_count
        ),
        "nominal_stable_fraction": float(np.mean(nominal_stable)),
        "ungated_updated_stable_fraction": float(np.mean(updated_stable)),
        "gated_policy_stable_fraction": float(np.mean(policy_stable)),
        "rescued_nominally_unstable_count": int(
            np.sum(accepted & ~nominal_stable & updated_stable)
        ),
        "harmed_nominally_stable_count": int(
            np.sum(accepted & nominal_stable & ~updated_stable)
        ),
        "accepted_robust_radius": {
            "median": float(
                np.median(result["robust_radius"][(multiplier, blend)][accepted])
            )
            if accepted_count
            else None,
            "maximum": float(
                np.max(result["robust_radius"][(multiplier, blend)][accepted])
            )
            if accepted_count
            else None,
        },
    }


def evaluate_ensemble_control(args: argparse.Namespace) -> Mapping[str, Any]:
    from flight_controller import load_controller_config

    ensemble_path = Path(args.ensemble).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    output_directory = Path(args.output).expanduser().resolve()
    artifact = torch.load(ensemble_path, map_location="cpu", weights_only=False)
    if artifact.get("artifact_type") != "flight_identification_ensemble":
        raise ValueError("expected a flight identification ensemble artifact")
    if artifact.get("target_mode") != "lqr_effective":
        raise ValueError("deployment gate currently requires lqr_effective targets")
    device = torch.device(args.device)
    predictions = {
        split: _load_window_predictions(
            artifact, dataset, split, device, args.batch_size
        )
        for split in ("validation", "test")
    }

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
    nominal_effectiveness, command_slopes, nominal_tau = _nominal_actuator_model(
        Path(args.simulator_config).expanduser().resolve()
    )
    nominal_a, nominal_b = _discrete_model(
        nominal_effectiveness, command_slopes, nominal_tau, 1.0 / 500.0
    )
    nominal_gain = _lqr_gain(nominal_a, nominal_b, q, r)
    multipliers = tuple(float(value) for value in args.uncertainty_multipliers.split(","))
    blends = tuple(float(value) for value in args.gain_blends.split(","))
    results = {}
    for split, (mean, standard_deviation, labels, group_ids) in predictions.items():
        results[split] = _evaluate_split(
            mean,
            standard_deviation,
            labels,
            group_ids,
            multipliers,
            blends,
            nominal_effectiveness,
            command_slopes,
            nominal_tau,
            nominal_gain,
            q,
            r,
        )
    gate = _select_gate(
        results["validation"],
        multipliers,
        blends,
        args.desired_precision,
        args.minimum_accept_count,
        args.maximum_robust_radius,
    )
    learned_gate, learned_gate_report = _fit_learned_gate(
        results["validation"], results["test"], blends, multipliers, args
    )
    report = {
        "schema_version": 1,
        "semantics": (
            "The gate and gain blend are selected only on validation groups. "
            "The test split is held out until this final local-linear audit. "
            "Uncertainty plants independently perturb every effective label by "
            "the calibrated standard-deviation multiplier and clip to support."
        ),
        "ensemble": str(ensemble_path),
        "dataset": str(dataset),
        "selected_physical_gate": gate,
        "physical_gate_validation": _gate_report(results["validation"], gate),
        "physical_gate_test": _gate_report(results["test"], gate),
        "learned_gate": learned_gate_report,
        "ungated_stability": {
            split: {
                "nominal": float(np.mean(result["nominal_stable"])),
                **{
                    f"blend_{blend:g}": float(np.mean(result["true_stable"][blend]))
                    for blend in blends
                },
            }
            for split, result in results.items()
        },
    }
    deployment_artifact = dict(artifact)
    deployment_artifact["artifact_type"] = "flight_identification_deployment"
    physical_test = report["physical_gate_test"]
    physical_gate = {
        "gate_type": "physical_robust_radius",
        "uncertainty_multiplier": gate["uncertainty_multiplier"],
        "uncertainty_multipliers": (gate["uncertainty_multiplier"],),
        "gain_blend": gate["gain_blend"],
        "robust_radius_threshold": gate["robust_radius_threshold"],
        "effective_lower": torch.from_numpy(EFFECTIVE_LOWER.copy()),
        "effective_upper": torch.from_numpy(EFFECTIVE_UPPER.copy()),
        "desired_validation_precision": args.desired_precision,
        "validation_target_met": gate["target_met"],
        "heldout_test_target_met": (
            physical_test["non_degradation_fraction_when_accepted"] is not None
            and physical_test["non_degradation_fraction_when_accepted"]
            >= args.desired_precision
        ),
    }
    physical_gate["enabled"] = bool(
        physical_gate["validation_target_met"]
        and physical_gate["heldout_test_target_met"]
    )
    deployment_artifact["physical_gain_gate"] = physical_gate
    learned_gate["gate_type"] = "learned_stability_classifier"
    deployment_artifact["learned_gain_gate"] = learned_gate
    deployment_artifact["gain_gate"] = (
        physical_gate if physical_gate["enabled"] else learned_gate
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(deployment_artifact, output_directory / "deployment.pt")
    (output_directory / "control_gate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate and audit a robust LQR gain-update gate"
    )
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--uncertainty-multipliers", default="0,0.5,1.0,1.645")
    parser.add_argument("--gain-blends", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--desired-precision", type=float, default=0.99)
    parser.add_argument("--minimum-accept-count", type=int, default=100)
    parser.add_argument("--minimum-calibration-accept-count", type=int, default=50)
    parser.add_argument("--maximum-robust-radius", type=float, default=0.9999)
    parser.add_argument("--gate-calibration-fraction", type=float, default=0.4)
    parser.add_argument("--gate-epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--simulator-config", default="SimEnv/configs/example.yaml")
    parser.add_argument(
        "--controller-config",
        default="Controller/configs/lqr_identification_nominal.yaml",
    )
    return parser


def main() -> None:
    evaluate_ensemble_control(build_parser().parse_args())


if __name__ == "__main__":
    main()
