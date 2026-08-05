from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .offline_log_inference import _flight_information_score, checkpoint_sha256
from .repeated_trial_training import _retain_converged_trials, load_repeated_split


def _slice_split(
    split: Mapping[str, Any], count: int
) -> dict[str, Any]:
    output = dict(split)
    for name, value in tuple(output.items()):
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == len(
            split["features"]
        ):
            output[name] = value[:count]
    return output


@torch.no_grad()
def _group_quality_statistics(
    raw_features: torch.Tensor,
    raw_valid: torch.Tensor,
    raw_eligible: torch.Tensor,
    feature_names: Sequence[str],
    trial_count: int,
    downsampled_features: torch.Tensor,
    downsampled_valid: torch.Tensor,
    normalization: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Per-group log-information and input z-score statistics.

    Flights are selected exactly like the offline tool: top ``trial_count``
    eligible flights by the raw 500 Hz information score. The input z-score
    is computed on the primary checkpoint's downsampled history using its
    training-split normalization, mirroring what the identifier actually sees.
    """
    if raw_features.ndim != 4:
        raise ValueError("raw features must be [group, trial, time, feature]")
    if downsampled_features.shape[:2] != raw_features.shape[:2]:
        raise ValueError("raw and downsampled splits must cover the same groups/trials")
    scores = torch.stack(
        [
            _flight_information_score(
                raw_features[group], raw_valid[group], feature_names
            )
            for group in range(len(raw_features))
        ]
    )
    scores = scores.masked_fill(~raw_eligible, -torch.inf)
    selected = torch.argsort(scores, descending=True)[:, :trial_count]
    selected_scores = scores.gather(1, selected)
    finite = torch.isfinite(selected_scores)
    information = selected_scores.masked_fill(~finite, 0.0).clamp_min(0.0)
    information_mean = information.sum(dim=1) / finite.sum(dim=1).clamp_min(1)
    selected_count = finite.sum(dim=1)

    feature_mean = normalization["feature_mean"].to(downsampled_features.dtype)
    feature_std = normalization["feature_std"].to(downsampled_features.dtype)
    z_p95 = []
    for group in range(len(raw_features)):
        picks = selected[group][finite[group]]
        if not picks.numel():
            z_p95.append(torch.zeros((), dtype=downsampled_features.dtype))
            continue
        features = downsampled_features[group, picks]
        valid = downsampled_valid[group, picks]
        normalized = (features - feature_mean) / feature_std
        normalized = torch.where(
            valid[..., None], normalized, torch.zeros_like(normalized)
        )
        z_p95.append(torch.quantile(normalized[valid].abs(), 0.95))
    return {
        "log_information_score_mean": information_mean,
        "selected_flight_count": selected_count,
        "input_z_score_abs_p95": torch.stack(z_p95),
    }


def _quality_stratified_summary(
    group_deltas: Sequence[float],
    statistics: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
    delta = torch.as_tensor(group_deltas, dtype=torch.float32)
    if delta.ndim != 1:
        raise ValueError("group deltas must be one value per parameter group")
    centered_delta = delta - delta.mean()
    entries = []
    for name in ("log_information_score_mean", "input_z_score_abs_p95"):
        value = statistics[name]
        if value.ndim != 1 or len(value) != len(delta):
            raise ValueError(f"{name} must be one value per parameter group")
        centered = value - value.mean()
        denominator = torch.sqrt(
            centered.square().sum() * centered_delta.square().sum()
        ).clamp_min(1e-12)
        correlation = float((centered * centered_delta).sum() / denominator)
        lower = value <= torch.quantile(value, 0.25)
        upper = value >= torch.quantile(value, 0.75)
        entries.append(
            {
                "name": name,
                "pearson_correlation_with_group_convergence_delta": correlation,
                "lowest_quartile_mean_delta": float(delta[lower].mean()),
                "highest_quartile_mean_delta": float(delta[upper].mean()),
            }
        )
    return {
        "improved_group_fraction": float((delta > 0).to(torch.float32).mean()),
        "degraded_group_fraction": float((delta < 0).to(torch.float32).mean()),
        "unchanged_group_fraction": float((delta == 0).to(torch.float32).mean()),
        "quality_associations": entries,
    }


@torch.no_grad()
def analyze(args: argparse.Namespace) -> Mapping[str, Any]:
    audit_path = Path(args.audit).expanduser().resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    deltas = audit.get("per_group_convergence_deltas_vs_composite_nominal")
    if not isinstance(deltas, dict) or not deltas:
        raise ValueError(
            "audit lacks per_group_convergence_deltas_vs_composite_nominal; "
            "re-run sim2real_composite_evaluation with the current code"
        )
    group_count = int(audit["test_parameter_groups"])
    trial_count = int(audit["trial_count"])
    dataset = Path(args.dataset).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_type") != "sim2real_composite_servo_identifier":
        raise ValueError("expected a composite servo identifier checkpoint")
    if not 1 <= trial_count <= int(checkpoint["trials_per_group"]):
        raise ValueError("audit trial count is outside the checkpoint contract")

    raw_split = load_repeated_split(dataset, "test", 1)
    _retain_converged_trials(raw_split)
    raw_split = _slice_split(raw_split, group_count)
    downsampled_split = load_repeated_split(
        dataset, "test", int(checkpoint["downsample"])
    )
    _retain_converged_trials(downsampled_split)
    downsampled_split = _slice_split(downsampled_split, group_count)
    if not torch.equal(raw_split["group_id"], downsampled_split["group_id"]):
        raise ValueError("raw and downsampled test group ordering differs")
    if len(raw_split["features"]) != group_count:
        raise ValueError("audit group count does not match the dataset slice")
    statistics = _group_quality_statistics(
        raw_split["features"],
        raw_split["valid_mask"],
        raw_split["trial_mask"],
        tuple(raw_split["feature_names"]),
        trial_count,
        downsampled_split["features"],
        downsampled_split["valid_mask"],
        checkpoint["normalization"],
    )
    stratified = {
        name: _quality_stratified_summary(values, statistics)
        for name, values in deltas.items()
    }
    report = {
        "schema_version": 1,
        "artifact_type": "offline_quality_stratification",
        "audit": str(audit_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "dataset": str(dataset),
        "test_parameter_groups": group_count,
        "trial_count": trial_count,
        "statistics": {
            "log_information_score_mean": statistics[
                "log_information_score_mean"
            ].tolist(),
            "input_z_score_abs_p95": statistics["input_z_score_abs_p95"].tolist(),
            "selected_flight_count": statistics["selected_flight_count"].tolist(),
        },
        "stratified_vs_composite_nominal": stratified,
        "deployment_mode": "offline_analysis_only",
        "meaning": (
            "Correlations and quartile deltas describe where the candidate gain "
            "wins or loses relative to the fixed composite nominal LQI. They are "
            "analysis evidence, not a flight-acceptance gate."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stratify composite candidate win/loss by raw log information and "
            "primary input OOD z-score using an audit JSON with per-group deltas"
        )
    )
    parser.add_argument("--audit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
