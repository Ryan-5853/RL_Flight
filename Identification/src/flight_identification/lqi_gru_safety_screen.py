from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .config import load_experiment_config
from .lqi_gru_closed_loop import _simulate
from .lqi_gru_experiment import _raw_config


@torch.no_grad()
def screen_safety_crashes(
    parameter_groups: Path,
    config_path: Path,
    output: Path,
    device: torch.device,
    maximum_groups: int,
    trials: int,
    duration_s: float,
    seed: int,
    split: str = "train",
    motor_tau_lower_min: float | None = None,
    motor_tau_upper_max: float | None = None,
    motor_tau_upper_min: float | None = None,
    group_ids: Sequence[int] | None = None,
) -> dict:
    """6-second safety screen: nominal crashes, then oracle rescue confirmation."""

    config = load_experiment_config(config_path)
    raw = _raw_config(config_path)
    payload = torch.load(parameter_groups, map_location="cpu", weights_only=False)
    labels_all = payload["labels"]
    split_index = {"train": 0, "validation": 1, "test": 2}[split]
    mask = payload["split_assignment"] == split_index
    names = list(payload["label_names"])
    if motor_tau_lower_min is not None:
        mask &= labels_all[:, names.index("motor_tau_lower_s")] >= motor_tau_lower_min
    if motor_tau_upper_max is not None:
        mask &= labels_all[:, names.index("motor_tau_upper_s")] <= motor_tau_upper_max
    if motor_tau_upper_min is not None:
        mask &= labels_all[:, names.index("motor_tau_upper_s")] >= motor_tau_upper_min
    if group_ids is not None:
        requested = torch.as_tensor(list(group_ids), dtype=payload["group_id"].dtype)
        ids = payload["group_id"][mask]
        selected = torch.isin(ids, requested)
        if int(selected.sum()) != len(requested):
            missing = set(int(g) for g in requested) - set(int(g) for g in ids[selected])
            raise ValueError(f"requested group ids not in pool: {sorted(missing)}")
        ids = ids[selected]
        labels = labels_all[mask][selected]
    else:
        ids = payload["group_id"][mask]
        labels = labels_all[mask]
    if len(ids) == 0:
        raise ValueError("no groups match the requested label filters")
    rng = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(ids), generator=rng)[:maximum_groups]
    ids = ids[permutation]
    labels = labels[permutation]

    nominal = _simulate(
        "nominal_lqi", config, raw, labels, ids, trials, duration_s, seed,
        Path("/nonexistent.pt"), device, "fixed_target",
    )
    nominal_safe = nominal["safe"]
    nominal_groups = nominal["group_id"]
    crash_rows = []
    for group in ids:
        selected = nominal_groups == group
        safe_fraction = float(nominal_safe[selected].to(torch.float32).mean())
        if safe_fraction < 1.0:
            crash_rows.append(
                {
                    "group_id": int(group),
                    "nominal_safe_fraction": round(safe_fraction, 4),
                    "nominal_rate_rms": round(
                        float(nominal["rate_tracking_rms_rad_s"][selected].mean()), 4
                    ),
                }
            )
    crash_rows.sort(key=lambda row: row["nominal_safe_fraction"])
    crash_ids = torch.as_tensor(
        [row["group_id"] for row in crash_rows], dtype=torch.int64
    )
    rescuable = []
    if len(crash_ids):
        crash_labels = labels_all[crash_ids]
        oracle = _simulate(
            "oracle_lqi", config, raw, crash_labels, crash_ids, trials, duration_s,
            seed + 1, Path("/nonexistent.pt"), device, "fixed_target",
        )
        oracle_safe = oracle["safe"]
        oracle_groups = oracle["group_id"]
        lookup = {row["group_id"]: row for row in crash_rows}
        for group in crash_ids:
            selected = oracle_groups == group
            oracle_safe_fraction = float(
                oracle_safe[selected].to(torch.float32).mean()
            )
            nominal_fraction = lookup[int(group)]["nominal_safe_fraction"]
            rescuable.append(
                {
                    "group_id": int(group),
                    "nominal_safe_fraction": nominal_fraction,
                    "oracle_safe_fraction": round(oracle_safe_fraction, 4),
                    "rescue_gap": round(oracle_safe_fraction - nominal_fraction, 4),
                }
            )
        rescuable.sort(key=lambda row: -row["rescue_gap"])

    report = {
        "schema_version": 1,
        "scope": f"{split}-split 6-second safety crash screen",
        "parameter_groups": str(parameter_groups),
        "groups_screened": len(ids),
        "label_filters": {
            "motor_tau_lower_min": motor_tau_lower_min,
            "motor_tau_upper_max": motor_tau_upper_max,
            "motor_tau_upper_min": motor_tau_upper_min,
        },
        "trials": trials,
        "duration_s": duration_s,
        "nominal_crash_groups": crash_rows,
        "oracle_rescue_analysis": rescuable,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen for 6-second nominal-LQI crashes and oracle rescue"
    )
    parser.add_argument("--parameter-groups", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-groups", type=int, default=600)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--motor-tau-lower-min", type=float)
    parser.add_argument("--motor-tau-upper-max", type=float)
    parser.add_argument("--motor-tau-upper-min", type=float)
    parser.add_argument("--group-ids", type=int, nargs="+")
    args = parser.parse_args()
    report = screen_safety_crashes(
        Path(args.parameter_groups).resolve(),
        Path(args.config).resolve(),
        Path(args.output).resolve(),
        torch.device(args.device),
        args.maximum_groups,
        args.trials,
        args.duration_s,
        args.seed,
        args.split,
        args.motor_tau_lower_min,
        args.motor_tau_upper_max,
        args.motor_tau_upper_min,
        args.group_ids,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
