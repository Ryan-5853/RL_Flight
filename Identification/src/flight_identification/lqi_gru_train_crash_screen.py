from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_experiment_config
from .lqi_gru_closed_loop import _simulate
from .lqi_gru_experiment import _raw_config


@torch.no_grad()
def screen_train_crashes(
    dataset: Path,
    config_path: Path,
    checkpoint_path: Path,
    output: Path,
    device: torch.device,
    trials: int,
    duration_s: float,
    seed: int,
) -> dict:
    """Find train-split groups where nominal LQI crashes (safety failure)."""

    config = load_experiment_config(config_path)
    raw = _raw_config(config_path)
    payload = torch.load(
        dataset / "parameter_groups_audit_only.pt",
        map_location="cpu",
        weights_only=False,
    )
    labels_all = payload["labels_audit_only"]
    split = payload["split_assignment"]
    train_ids = torch.nonzero(split == 0).flatten()
    labels = labels_all[train_ids]
    ids = train_ids

    nominal = _simulate(
        "nominal_lqi", config, raw, labels, ids, trials, duration_s, seed,
        checkpoint_path, device, "fixed_target",
    )
    nominal_safe = nominal["safe"]
    nominal_groups = nominal["group_id"]
    per_group = {}
    for index, group in enumerate(ids):
        mask = nominal_groups == group
        safe_fraction = float(nominal_safe[mask].to(torch.float32).mean())
        per_group[int(group)] = {
            "nominal_safe_fraction": round(safe_fraction, 4),
            "nominal_rate_rms": round(
                float(nominal["rate_tracking_rms_rad_s"][mask].mean()), 4
            ),
        }
    crash_ids = [
        int(group)
        for group, values in per_group.items()
        if values["nominal_safe_fraction"] < 1.0
    ]
    detail = {}
    if crash_ids:
        crash_tensor = torch.as_tensor(crash_ids, dtype=torch.int64)
        crash_labels = labels_all[crash_tensor]
        for mode in ("nominal_lqi", "oracle_lqi", "gru"):
            result = _simulate(
                mode, config, raw, crash_labels, crash_tensor, trials, duration_s,
                seed + 1, checkpoint_path, device, "fixed_target",
            )
            detail[mode] = {
                "safe_fraction": float(
                    result["safe"].to(torch.float32).mean()
                ),
                "rate_rms_mean": float(result["rate_tracking_rms_rad_s"].mean()),
                "per_group_safe": {
                    int(group): round(
                        float(
                            result["safe"][result["group_id"] == group]
                            .to(torch.float32)
                            .mean()
                        ),
                        4,
                    )
                    for group in crash_ids
                },
            }
    report = {
        "schema_version": 1,
        "scope": "train-split nominal-LQI safety screen",
        "groups_screened": len(ids),
        "trials": trials,
        "duration_s": duration_s,
        "nominal_crash_groups": crash_ids,
        "per_group": per_group,
        "detail_on_crashes": detail,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen train-split groups for nominal-LQI safety crashes"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    report = screen_train_crashes(
        Path(args.dataset).resolve(),
        Path(args.config).resolve(),
        Path(args.checkpoint).resolve(),
        Path(args.output).resolve(),
        torch.device(args.device),
        args.trials,
        args.duration_s,
        args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
