#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flight_train.angular_acceleration_teacher import (
    IncrementalAngularAccelerationTeacher,
    normalized_action_effectiveness,
)
from flight_train.config import load_experiment_config, override_run_config
from flight_train.evaluation import (
    _create_packed_evaluation_environment,
    _run_packed_scenarios,
    load_fixed_evaluation_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune an INDI teacher on the fixed direct-acceleration suite"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gains",
        nargs="+",
        type=float,
        default=(0.01, 0.02, 0.04, 0.08, 0.12, 0.16),
    )
    args = parser.parse_args()

    config = override_run_config(
        load_experiment_config(args.config), device=args.device
    )
    suite = load_fixed_evaluation_suite(args.suite)
    device = torch.device(args.device)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    effectiveness: torch.Tensor | None = None
    for gain in args.gains:
        env = _create_packed_evaluation_environment(config, suite, device)
        try:
            if effectiveness is None:
                effectiveness = normalized_action_effectiveness(
                    env.simulator.parameters, config.control_contract
                )
            teacher = IncrementalAngularAccelerationTeacher.create(
                env.simulator.parameters,
                config.control_contract,
                config.task,
                (gain, gain, gain),
            )
            results = _run_packed_scenarios(env, teacher, suite)
        finally:
            env.close()
        scenario_metrics = {
            name: metrics for name, (metrics, _trajectory) in results.items()
        }
        minimum_survival_fraction = min(
            float(value["metrics"]["survival_time_s"]["mean"])
            / float(value["duration_s"])
            for value in scenario_metrics.values()
        )
        summary = {
            "gain": gain,
            "total_score": sum(
                float(value["total_score"])
                for value in scenario_metrics.values()
            )
            / len(scenario_metrics),
            "minimum_survival_fraction": minimum_survival_fraction,
            "maximum_action_peak": max(
                float(value["metrics"]["action_peak_abs"]["mean"])
                for value in scenario_metrics.values()
            ),
        }
        first_metrics = next(iter(scenario_metrics.values()))["metrics"]
        if "angular_acceleration_normalized_vector_rmse" in first_metrics:
            summary["mean_normalized_rmse"] = sum(
                float(
                    value["metrics"][
                        "angular_acceleration_normalized_vector_rmse"
                    ]["mean"]
                )
                for value in scenario_metrics.values()
            ) / len(scenario_metrics)
        else:
            summary["mean_roll_pitch_rmse_deg"] = sum(
                float(value["metrics"]["roll_pitch_rmse_deg"]["mean"])
                for value in scenario_metrics.values()
            ) / len(scenario_metrics)
            summary["mean_yaw_rate_rmse_rad_s"] = sum(
                float(value["metrics"]["yaw_rate_rmse_rad_s"]["mean"])
                for value in scenario_metrics.values()
            ) / len(scenario_metrics)
        rows.append(summary)
        print(json.dumps(summary))
        partial_report = {
            "schema_version": 1,
            "status": "running",
            "experiment_config": str(Path(args.config).resolve()),
            "suite_config": str(Path(args.suite).resolve()),
            "normalized_action_effectiveness_rad_s2": (
                effectiveness.cpu().tolist()
            ),
            "condition_number": float(torch.linalg.cond(effectiveness).cpu()),
            "results": rows,
        }
        output.write_text(
            json.dumps(partial_report, indent=2), encoding="utf-8"
        )

    if effectiveness is None:
        raise RuntimeError("no teacher gain was evaluated")
    report = {
        "schema_version": 1,
        "experiment_config": str(Path(args.config).resolve()),
        "suite_config": str(Path(args.suite).resolve()),
        "normalized_action_effectiveness_rad_s2": effectiveness.cpu().tolist(),
        "condition_number": float(torch.linalg.cond(effectiveness).cpu()),
        "results": rows,
        "best_by_total_score": max(rows, key=lambda value: value["total_score"]),
        "status": "completed",
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
