from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

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


def _load_group_predictions(
    artifact: Mapping[str, Any],
    dataset: Path,
    split_name: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    member_groups, labels = _aggregate_member_predictions(
        member_windows, split["labels"], split["group_id"]
    )
    uncertainty = calibrated_std(member_groups, artifact["calibration"])
    return (
        member_groups.mean(dim=0).numpy(),
        uncertainty.numpy(),
        labels.numpy(),
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
            for uncertain_label in _uncertainty_plants(
                clipped_mean, standard_deviation, multiplier
            ):
                effectiveness, tau = _scaled_actuator_model(
                    uncertain_label, nominal_effectiveness, nominal_tau
                )
                uncertain_a, uncertain_b = _discrete_model(
                    effectiveness, command_slopes, tau, 1.0 / 500.0
                )
                for blend, gain in gains.items():
                    worst[blend] = max(
                        worst[blend], _pole_radius(uncertain_a, uncertain_b, gain)
                    )
            for blend, radius in worst.items():
                robust_radius[(multiplier, blend)][sample_index] = radius
    return {
        "nominal_stable": nominal_stable,
        "true_stable": true_stable,
        "robust_radius": robust_radius,
        "gain_relative_step": gain_relative_step,
    }


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
            quantiles = np.linspace(0.02, 1.0, 50)
            thresholds = np.unique(np.quantile(radii, quantiles))
            for threshold in thresholds:
                threshold = min(float(threshold), maximum_robust_radius)
                accepted = radii <= threshold
                accepted_count = int(np.sum(accepted))
                if accepted_count < minimum_accept_count:
                    continue
                successes = int(np.sum(stable & accepted))
                precision = successes / accepted_count
                candidates.append(
                    {
                        "uncertainty_multiplier": float(multiplier),
                        "gain_blend": float(blend),
                        "robust_radius_threshold": threshold,
                        "accepted_count": accepted_count,
                        "coverage": accepted_count / len(stable),
                        "stable_fraction_when_accepted": precision,
                        "stable_wilson_lower_95": _wilson_lower(
                            successes, accepted_count
                        ),
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
            candidate["coverage"] if eligible else candidate["stable_fraction_when_accepted"],
            candidate["stable_wilson_lower_95"],
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
        split: _load_group_predictions(
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
    for split, (mean, standard_deviation, labels) in predictions.items():
        results[split] = _evaluate_split(
            mean,
            standard_deviation,
            labels,
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
        "selected_gate": gate,
        "validation": _gate_report(results["validation"], gate),
        "test": _gate_report(results["test"], gate),
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
    deployment_artifact["gain_gate"] = {
        "uncertainty_multiplier": gate["uncertainty_multiplier"],
        "gain_blend": gate["gain_blend"],
        "robust_radius_threshold": gate["robust_radius_threshold"],
        "effective_lower": torch.from_numpy(EFFECTIVE_LOWER.copy()),
        "effective_upper": torch.from_numpy(EFFECTIVE_UPPER.copy()),
        "desired_validation_precision": args.desired_precision,
        "validation_target_met": gate["target_met"],
    }
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
    parser.add_argument("--maximum-robust-radius", type=float, default=0.9999)
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
