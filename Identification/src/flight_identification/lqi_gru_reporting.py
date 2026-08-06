from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .lqi_gru_distillation import evaluate


def collect_offline_metrics(
    dataset: Path,
    output_root: Path,
    device: torch.device,
) -> Mapping[str, Any]:
    """Gate A offline metrics for every DAgger-iteration student checkpoint."""

    students = sorted((output_root / "students").glob("iter_*/student.pt"))
    entries = {}
    for checkpoint in students:
        iteration = int(checkpoint.parent.name.split("_")[1])
        report_path = checkpoint.parent / "offline.json"
        report = evaluate(
            dataset,
            checkpoint,
            report_path,
            device,
            batch_size=64,
            probe_ridge=0.001,
        )
        held_out = report["held_out_imitation"]
        entries[iteration] = {
            "checkpoint": str(checkpoint),
            "command_nrmse_by_teacher_std": held_out["command_nrmse_by_teacher_std"],
            "per_sequence_rmse": held_out["per_sequence_rmse"],
            "time_conditioned_mae": held_out["time_conditioned_mae"],
            "memory_ablation_rmse_ratio_reset_over_recurrent": report[
                "memory_ablation_rmse_ratio_reset_over_recurrent"
            ],
            "hidden_parameter_probe_median_r2": report["hidden_parameter_probe"][
                "median_r2"
            ],
        }
    return entries


def build_summary_table(output_root: Path) -> Mapping[str, Any]:
    gates = sorted((output_root / "gates").glob("gate_iter_*.json"))
    rows = []
    for gate_path in gates:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        iteration = int(gate_path.stem.split("_")[2])
        rows.append(
            {
                "iteration": iteration,
                "gru_safe_fraction": gate["variants"]["gru"]["safe_fraction"],
                "gru_reset_safe_fraction": gate["variants"]["gru_reset"]["safe_fraction"],
                "oracle_safe_fraction": gate["variants"]["oracle_lqi"]["safe_fraction"],
                "gru_rate_rms_rad_s": gate["variants"]["gru"]["rate_tracking_rms_rad_s"][
                    "mean"
                ],
                "gru_attitude_rms_rad": gate["variants"]["gru"][
                    "attitude_tracking_rms_rad"
                ]["mean"],
                "gru_survival_mean": gate["variants"]["gru"]["survival_fraction"]["mean"],
                "gru_command_movement": gate["variants"]["gru"]["command_movement_mean"][
                    "mean"
                ],
            }
        )
    return {"gate_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate DAgger iteration metrics for the report"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    if args.offline:
        result = collect_offline_metrics(
            Path(args.dataset).resolve(), root, torch.device(args.device)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        (root / "offline_metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    else:
        result = build_summary_table(root)
        print(json.dumps(result, indent=2, sort_keys=True))
        (root / "gate_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
