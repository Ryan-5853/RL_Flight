#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _common_score(report: dict[str, Any]) -> float:
    weights = report["score_weights"]
    common_weight = weights["survival"] + weights["tracking"] + weights["response"]
    return sum(
        (
            weights["survival"] * value["scores"]["survival"]["mean"]
            + weights["tracking"] * value["scores"]["tracking"]["mean"]
            + weights["response"] * value["scores"]["response"]["mean"]
        )
        / common_weight
        for value in report["scenarios"].values()
    ) / len(report["scenarios"])


def _summary(report: dict[str, Any], *, common_score: float) -> dict[str, Any]:
    scenarios = report["scenarios"]
    survival_fractions = [
        value["metrics"]["survival_time_s"]["mean"] / value["duration_s"]
        for value in scenarios.values()
    ]
    hover = scenarios["hover"]["metrics"]
    return {
        "total_score": float(report["total_score"]),
        "common_score_excluding_action": common_score,
        "minimum_survival_fraction": min(survival_fractions),
        "mean_survival_fraction": sum(survival_fractions) / len(survival_fractions),
        "hover_roll_pitch_rmse_deg": hover["roll_pitch_rmse_deg"]["mean"],
        "hover_yaw_rate_rmse_rad_s": hover["yaw_rate_rmse_rad_s"]["mean"],
        "hover_action_rms": hover["action_rms"]["mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("GROUP_ID", "NEURAL_REPORT", "LQI_REPORT"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for group_id, neural_path_raw, lqi_path_raw in args.pair:
        neural_path = Path(neural_path_raw).expanduser().resolve()
        lqi_path = Path(lqi_path_raw).expanduser().resolve()
        neural = json.loads(neural_path.read_text(encoding="utf-8"))
        lqi = json.loads(lqi_path.read_text(encoding="utf-8"))
        neural_scenarios = set(neural["scenarios"])
        lqi_scenarios = set(lqi["scenarios"])
        if neural_scenarios != lqi_scenarios:
            raise ValueError(
                f"group {group_id}: scenario mismatch: "
                f"neural={sorted(neural_scenarios)}, lqi={sorted(lqi_scenarios)}"
            )
        neural_common = _common_score(neural)
        if "common_score_excluding_action" in lqi:
            lqi_common = float(lqi["common_score_excluding_action"])
        else:
            lqi_common = _common_score(lqi)
        neural_summary = _summary(neural, common_score=neural_common)
        lqi_summary = _summary(lqi, common_score=lqi_common)
        scenario_deltas = {
            name: {
                "total_score_delta_neural_minus_lqi": (
                    neural["scenarios"][name]["total_score"]
                    - lqi["scenarios"][name]["total_score"]
                ),
                "survival_fraction_delta_neural_minus_lqi": (
                    neural["scenarios"][name]["metrics"]["survival_time_s"]["mean"]
                    / neural["scenarios"][name]["duration_s"]
                    - lqi["scenarios"][name]["metrics"]["survival_time_s"]["mean"]
                    / lqi["scenarios"][name]["duration_s"]
                ),
            }
            for name in neural["scenarios"]
        }
        rows.append(
            {
                "group_id": int(group_id),
                "neural_report": str(neural_path),
                "lqi_report": str(lqi_path),
                "neural": neural_summary,
                "nominal_lqi": lqi_summary,
                "total_score_delta_neural_minus_lqi": (
                    neural_summary["total_score"] - lqi_summary["total_score"]
                ),
                "common_score_delta_neural_minus_lqi": (
                    neural_common - lqi_common
                ),
                "winner_by_common_score": (
                    "neural" if neural_common > lqi_common else "nominal_lqi"
                ),
                "scenarios": scenario_deltas,
            }
        )
    report = {
        "schema_version": 1,
        "comparison": "point_specific_neural_cascade_vs_nominal_lqi",
        "suite": "fixed_small_command_tracking_v1",
        "score_note": (
            "common_score_excluding_action compares survival, tracking, and response; "
            "the raw action score is heterogeneous because neural uses a 3D residual "
            "action while LQI uses a normalized 5D physical command"
        ),
        "points": rows,
        "mean_common_score_delta_neural_minus_lqi": sum(
            value["common_score_delta_neural_minus_lqi"] for value in rows
        )
        / len(rows),
        "neural_wins": sum(
            value["winner_by_common_score"] == "neural" for value in rows
        ),
        "nominal_lqi_wins": sum(
            value["winner_by_common_score"] == "nominal_lqi" for value in rows
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
