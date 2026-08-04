from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import load_experiment_config
from .control_evaluation import (
    _closed_loop_result,
    _discrete_model,
    _lqr_gain,
    _nominal_actuator_model,
)
from .experiment import _apply_effectiveness_labels, _sim2real_lqr_targets
from .gain_training import _lqr_weights
from .repeated_trial_evaluation import _sample_near_equilibrium, _simulate_gain


def _group_fraction(value: torch.Tensor, group_count: int, repeats: int) -> torch.Tensor:
    return value.to(torch.float32).reshape(group_count, repeats).mean(dim=1)


def _group_mean(value: torch.Tensor, group_count: int, repeats: int) -> torch.Tensor:
    return value.to(torch.float64).reshape(group_count, repeats).mean(dim=1)


def _rank_bad_points(
    *,
    group_ids: torch.Tensor,
    labels: torch.Tensor,
    label_names: Sequence[str],
    label_ranges: torch.Tensor,
    nominal_result: Mapping[str, torch.Tensor],
    oracle_result: Mapping[str, torch.Tensor],
    nominal_local_radius: Sequence[float],
    oracle_local_radius: Sequence[float],
    repeats: int,
    maximum_points: int,
    maximum_nominal_converged_fraction: float,
    minimum_oracle_converged_fraction: float,
    minimum_oracle_safe_fraction: float,
    minimum_convergence_gain: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    group_count = len(group_ids)
    nominal_converged = _group_fraction(
        nominal_result["converged"], group_count, repeats
    )
    oracle_converged = _group_fraction(
        oracle_result["converged"], group_count, repeats
    )
    nominal_safe = _group_fraction(nominal_result["safe"], group_count, repeats)
    oracle_safe = _group_fraction(oracle_result["safe"], group_count, repeats)
    convergence_gain = oracle_converged - nominal_converged
    eligible = (
        (nominal_converged <= maximum_nominal_converged_fraction)
        & (oracle_converged >= minimum_oracle_converged_fraction)
        & (oracle_safe >= minimum_oracle_safe_fraction)
        & (convergence_gain >= minimum_convergence_gain)
    )

    # Rank by paired nonlinear evidence first. Diversity is used only to avoid
    # returning several nearly identical physical parameter combinations.
    base_score = (
        2.0 * convergence_gain
        + 0.5 * oracle_safe
        - 0.25 * nominal_safe
        - 0.25 * nominal_converged
    )
    candidate_indices = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    if not candidate_indices:
        candidate_indices = list(range(group_count))
    candidate_indices.sort(key=lambda index: (-float(base_score[index]), int(group_ids[index])))

    span = (label_ranges[:, 1] - label_ranges[:, 0]).clamp_min(1e-12)
    normalized = (labels.to(torch.float64) - label_ranges[:, 0]) / span
    selected: list[int] = []
    pool = candidate_indices[: max(maximum_points * 32, maximum_points)]
    while pool and len(selected) < maximum_points:
        if not selected:
            chosen = pool[0]
        else:
            distances = torch.cdist(
                normalized[pool], normalized[selected], p=2
            ).amin(dim=1) / math.sqrt(normalized.shape[1])
            best_base = max(float(base_score[index]) for index in pool)
            worst_base = min(float(base_score[index]) for index in pool)
            scale = max(best_base - worst_base, 1e-12)
            utility = torch.tensor(
                [(float(base_score[index]) - worst_base) / scale for index in pool],
                dtype=torch.float64,
            ) + 0.35 * distances
            chosen = pool[int(torch.argmax(utility))]
        selected.append(chosen)
        pool.remove(chosen)

    nominal_attitude = _group_mean(
        nominal_result["final_attitude_error"], group_count, repeats
    )
    oracle_attitude = _group_mean(
        oracle_result["final_attitude_error"], group_count, repeats
    )
    nominal_rate = _group_mean(nominal_result["final_rate"], group_count, repeats)
    oracle_rate = _group_mean(oracle_result["final_rate"], group_count, repeats)
    nominal_saturation = _group_mean(
        nominal_result["saturation_fraction"], group_count, repeats
    )
    oracle_saturation = _group_mean(
        oracle_result["saturation_fraction"], group_count, repeats
    )

    rows = []
    for rank, index in enumerate(selected, start=1):
        rows.append(
            {
                "rank": rank,
                "group_id": int(group_ids[index]),
                "passed_selection_thresholds": bool(eligible[index]),
                "selection_score": float(base_score[index]),
                "parameters": {
                    name: float(value)
                    for name, value in zip(label_names, labels[index], strict=True)
                },
                "nominal_lqi": {
                    "safe_fraction": float(nominal_safe[index]),
                    "converged_fraction": float(nominal_converged[index]),
                    "final_attitude_error_rad_mean": float(nominal_attitude[index]),
                    "final_rate_rad_s_mean": float(nominal_rate[index]),
                    "saturation_fraction_mean": float(nominal_saturation[index]),
                    "local_pole_radius": float(nominal_local_radius[index]),
                },
                "oracle_lqi": {
                    "safe_fraction": float(oracle_safe[index]),
                    "converged_fraction": float(oracle_converged[index]),
                    "final_attitude_error_rad_mean": float(oracle_attitude[index]),
                    "final_rate_rad_s_mean": float(oracle_rate[index]),
                    "saturation_fraction_mean": float(oracle_saturation[index]),
                    "local_pole_radius": float(oracle_local_radius[index]),
                },
                "paired_convergence_fraction_gain": float(convergence_gain[index]),
            }
        )
    return rows, {
        "evaluated_parameter_groups": group_count,
        "eligible_parameter_groups": int(eligible.sum()),
        "returned_parameter_groups": len(rows),
    }


def screen(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset = Path(args.dataset).expanduser().resolve()
    payload = torch.load(
        dataset / "parameter_groups.pt", map_location="cpu", weights_only=False
    )
    if payload.get("parameterization") != "sim2real_micro":
        raise ValueError("bad-point screening requires sim2real_micro parameter groups")
    test = payload["split_assignment"] == 2
    labels = payload["labels"][test].to(torch.float64)
    group_ids = payload["group_id"][test]
    if args.group_ids:
        requested = torch.tensor(args.group_ids, dtype=group_ids.dtype)
        selected = (group_ids[:, None] == requested[None, :]).any(dim=1)
        missing = sorted(set(args.group_ids) - set(group_ids[selected].tolist()))
        if missing:
            raise ValueError(f"requested group IDs are not in the test split: {missing}")
        labels = labels[selected]
        group_ids = group_ids[selected]
    available = len(labels)
    count = min(args.maximum_parameter_groups, available)
    labels = labels[:count]
    group_ids = group_ids[:count]
    label_names = tuple(payload["label_names"])
    label_ranges = payload["label_ranges"].to(torch.float64)

    experiment = load_experiment_config(args.experiment_config)
    if experiment.parameterization != "sim2real_micro":
        raise ValueError("experiment config must use sim2real_micro parameterization")
    device = torch.device(experiment.device)
    dtype = torch.float32 if experiment.dtype == "float32" else torch.float64
    from simenv.config import load_and_materialize

    nominal_materialized = load_and_materialize(
        experiment.simulator_config, 1, device, dtype
    )
    expanded_nominal = {
        name: value.expand(count, *value.shape[1:]).clone()
        for name, value in nominal_materialized.parameters.items()
    }
    actual_parameters = _apply_effectiveness_labels(
        expanded_nominal, labels.to(device=device, dtype=dtype), "sim2real_micro"
    )
    targets = _sim2real_lqr_targets(actual_parameters).to(torch.float64).cpu().numpy()

    nominal_effectiveness, command_slopes, nominal_tau = _nominal_actuator_model(
        experiment.simulator_config
    )
    q, r = _lqr_weights(experiment.controller_config)
    nominal_a, nominal_b = _discrete_model(
        nominal_effectiveness,
        command_slopes,
        nominal_tau,
        1.0 / experiment.control_hz,
        integral=True,
    )
    nominal_gain = _lqr_gain(nominal_a, nominal_b, q, r)
    oracle_gain = []
    nominal_radius = []
    oracle_radius = []
    initial_covariance = np.linalg.inv(q)
    for target in targets:
        effectiveness = target[:15].reshape(3, 5)
        tau = np.power(10.0, target[18:23])
        a, b = _discrete_model(
            effectiveness,
            command_slopes,
            tau,
            1.0 / experiment.control_hz,
            integral=True,
        )
        gain = _lqr_gain(a, b, q, r)
        oracle_gain.append(gain)
        nominal_radius.append(
            _closed_loop_result(a, b, nominal_gain, q, r, initial_covariance)[0]
        )
        oracle_radius.append(
            _closed_loop_result(a, b, gain, q, r, initial_covariance)[0]
        )
    oracle_gain_array = np.stack(oracle_gain)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    repeats = args.evaluation_initial_conditions
    repeated_labels = labels.to(torch.float32).repeat_interleave(repeats, dim=0)
    attitude, angular_velocity = _sample_near_equilibrium(
        len(repeated_labels),
        args.maximum_initial_tilt_rad,
        args.maximum_initial_rate_rad_s,
        generator,
        torch.float32,
    )
    nominal_result = _simulate_gain(
        experiment,
        repeated_labels,
        attitude,
        angular_velocity,
        np.broadcast_to(nominal_gain, (len(repeated_labels), *nominal_gain.shape)),
        np.broadcast_to(nominal_tau[2:], (len(repeated_labels), 3)),
        np.ones(len(repeated_labels), dtype=np.bool_),
        args.duration_s,
    )
    oracle_result = _simulate_gain(
        experiment,
        repeated_labels,
        attitude,
        angular_velocity,
        np.repeat(oracle_gain_array, repeats, axis=0),
        np.repeat(np.power(10.0, targets[:, 20:23]), repeats, axis=0),
        np.ones(len(repeated_labels), dtype=np.bool_),
        args.duration_s,
    )
    selected, counts = _rank_bad_points(
        group_ids=group_ids,
        labels=labels,
        label_names=label_names,
        label_ranges=label_ranges,
        nominal_result=nominal_result,
        oracle_result=oracle_result,
        nominal_local_radius=nominal_radius,
        oracle_local_radius=oracle_radius,
        repeats=repeats,
        maximum_points=args.maximum_points,
        maximum_nominal_converged_fraction=args.maximum_nominal_converged_fraction,
        minimum_oracle_converged_fraction=args.minimum_oracle_converged_fraction,
        minimum_oracle_safe_fraction=args.minimum_oracle_safe_fraction,
        minimum_convergence_gain=args.minimum_convergence_gain,
    )
    report = {
        "schema_version": 1,
        "purpose": "Select feasible in-range plants where nominal LQI underperforms oracle LQI.",
        "dataset": str(dataset),
        "experiment_config": str(Path(args.experiment_config).expanduser().resolve()),
        "split": "test",
        "requested_group_ids": list(args.group_ids or ()),
        "paired_noise": True,
        "seed": args.seed,
        "duration_s": args.duration_s,
        "evaluation_initial_conditions_per_group": repeats,
        "maximum_initial_tilt_rad": args.maximum_initial_tilt_rad,
        "maximum_initial_rate_rad_s": args.maximum_initial_rate_rad_s,
        "selection_thresholds": {
            "maximum_nominal_converged_fraction": args.maximum_nominal_converged_fraction,
            "minimum_oracle_converged_fraction": args.minimum_oracle_converged_fraction,
            "minimum_oracle_safe_fraction": args.minimum_oracle_safe_fraction,
            "minimum_convergence_gain": args.minimum_convergence_gain,
        },
        "counts": counts,
        "selected_points": selected,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen in-range sim2real plants for nominal-LQI bad points"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-parameter-groups", type=int, default=4096)
    parser.add_argument("--group-ids", type=int, nargs="+")
    parser.add_argument("--evaluation-initial-conditions", type=int, default=16)
    parser.add_argument("--maximum-points", type=int, default=3)
    parser.add_argument("--maximum-initial-tilt-rad", type=float, default=0.2617994)
    parser.add_argument("--maximum-initial-rate-rad-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--maximum-nominal-converged-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-oracle-converged-fraction", type=float, default=0.75)
    parser.add_argument("--minimum-oracle-safe-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-convergence-gain", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser


def main() -> None:
    screen(build_parser().parse_args())


if __name__ == "__main__":
    main()
