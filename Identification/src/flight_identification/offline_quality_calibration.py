from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .offline_log_inference import (
    _flight_information_score,
    _log_quality_metrics,
    checkpoint_sha256,
)
from .repeated_trial_training import _retain_converged_trials, load_repeated_split


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


@torch.no_grad()
def calibrate(args: argparse.Namespace) -> Mapping[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_type") != "sim2real_composite_servo_identifier":
        raise ValueError("expected a composite servo identifier checkpoint")
    if args.minimum_selected_flights < 1:
        raise ValueError("minimum-selected-flights must be positive")
    if not 0.0 <= args.lower_quantile < args.upper_quantile <= 1.0:
        raise ValueError("quality quantiles must satisfy 0 <= lower < upper <= 1")

    dataset = Path(args.dataset).expanduser().resolve()
    split = load_repeated_split(dataset, "validation", int(checkpoint["downsample"]))
    _retain_converged_trials(split)
    feature_names = tuple(split["feature_names"])
    if feature_names != tuple(checkpoint["feature_names"]):
        raise ValueError("validation feature schema differs from checkpoint")
    normalization = checkpoint["normalization"]
    records: list[dict[str, Any]] = []
    maximum_trials = int(checkpoint["trials_per_group"])
    for group_index in range(len(split["features"])):
        features = split["features"][group_index]
        valid = split["valid_mask"][group_index]
        eligible = split["trial_mask"][group_index]
        if int(eligible.sum()) < args.minimum_selected_flights:
            continue
        scores = _flight_information_score(features, valid, feature_names)
        scores = scores.masked_fill(~eligible, -torch.inf)
        selected = torch.argsort(scores, descending=True)[:maximum_trials]
        selected = selected[torch.isfinite(scores[selected])]
        selected_features = features[selected]
        selected_valid = valid[selected]
        normalized = (
            selected_features - normalization["feature_mean"]
        ) / normalization["feature_std"]
        normalized = torch.where(
            selected_valid[..., None], normalized, torch.zeros_like(normalized)
        )
        records.append(
            _log_quality_metrics(
                selected_features, selected_valid, feature_names, normalized
            )
        )
    if not records:
        raise ValueError("validation split has no group with enough successful flights")

    lower = args.lower_quantile
    upper = args.upper_quantile
    thresholds = {
        "minimum_axis_rate_rms_rad_s": {
            axis: _quantile(
                [record["axis_rate_rms_rad_s"][axis] for record in records], lower
            )
            for axis in "xyz"
        },
        "minimum_command_movement_rms_per_sample": _quantile(
            [record["command_movement_rms_per_sample"] for record in records], lower
        ),
        "maximum_feature_z_score_abs_p95": _quantile(
            [record["feature_z_score_abs_p95"] for record in records], upper
        ),
        "maximum_feature_z_score_abs_max": _quantile(
            [record["feature_z_score_abs_max"] for record in records], upper
        ),
    }
    report = {
        "schema_version": 1,
        "artifact_type": "offline_log_quality_calibration",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "dataset": str(dataset),
        "split": "validation",
        "feature_names": feature_names,
        "effective_sample_hz": 500.0 / int(checkpoint["downsample"]),
        "history_steps": int(checkpoint["history_steps"]),
        "calibration_parameter_groups": len(records),
        "minimum_selected_flights": int(args.minimum_selected_flights),
        "lower_quantile": lower,
        "upper_quantile": upper,
        "thresholds": thresholds,
        "deployment_mode": "offline_analysis_only",
        "meaning": (
            "Thresholds use validation closed-loop flights only. Passing permits a "
            "candidate to enter independent simulation/HIL validation, not flight."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate offline closed-loop log quality limits on validation data"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-selected-flights", type=int, default=4)
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    return parser


def main() -> None:
    calibrate(build_parser().parse_args())


if __name__ == "__main__":
    main()
