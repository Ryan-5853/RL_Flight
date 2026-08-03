from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .training import (
    _aggregate_by_group,
    _predict,
    apply_target_mode,
    create_identifier_model,
    load_split,
    regression_metrics,
)


def _load_member(path: Path, device: torch.device) -> tuple[Mapping[str, Any], torch.nn.Module]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = create_identifier_model(
        str(checkpoint.get("architecture", "mlp")),
        int(checkpoint["history_steps"]),
        int(checkpoint["feature_count"]),
        len(checkpoint["label_names"]),
        tuple(checkpoint["hidden_sizes"]),
        float(checkpoint.get("dropout", 0.0)),
        int(checkpoint.get("tcn_channels", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model


def _member_predictions(
    member_paths: Sequence[Path],
    split: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, list[Mapping[str, Any]]]:
    predictions = []
    checkpoints = []
    for path in member_paths:
        checkpoint, model = _load_member(path, device)
        if tuple(checkpoint["label_names"]) != tuple(split["label_names"]):
            raise ValueError(f"label schema mismatch in ensemble member: {path}")
        normalization = checkpoint["normalization"]
        features = (split["features"] - normalization["feature_mean"]) / normalization[
            "feature_std"
        ]
        normalized = _predict(model, features, device, batch_size)
        predictions.append(
            normalized * normalization["label_std"] + normalization["label_mean"]
        )
        checkpoints.append(checkpoint)
        del model
    return torch.stack(predictions, dim=0), checkpoints


def predict_checkpoint_members(
    checkpoints: Sequence[Mapping[str, Any]],
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Run embedded ensemble checkpoints on raw, windowed feature histories."""
    predictions = []
    for checkpoint in checkpoints:
        model = create_identifier_model(
            str(checkpoint.get("architecture", "mlp")),
            int(checkpoint["history_steps"]),
            int(checkpoint["feature_count"]),
            len(checkpoint["label_names"]),
            tuple(checkpoint["hidden_sizes"]),
            float(checkpoint.get("dropout", 0.0)),
            int(checkpoint.get("tcn_channels", 128)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        normalization = checkpoint["normalization"]
        normalized_features = (
            features - normalization["feature_mean"]
        ) / normalization["feature_std"]
        normalized_prediction = _predict(
            model, normalized_features, device, batch_size
        )
        predictions.append(
            normalized_prediction * normalization["label_std"]
            + normalization["label_mean"]
        )
        del model
    if not predictions:
        raise ValueError("ensemble contains no members")
    return torch.stack(predictions, dim=0)


def _aggregate_member_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    members = []
    group_labels = None
    for prediction in predictions:
        member, current_labels = _aggregate_by_group(prediction, labels, group_ids)
        members.append(member)
        if group_labels is None:
            group_labels = current_labels
    if group_labels is None:
        raise ValueError("ensemble has no members")
    return torch.stack(members, dim=0), group_labels


def fit_uncertainty_calibration(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    coverage: float = 0.90,
) -> dict[str, torch.Tensor | float]:
    mean = predictions.mean(dim=0)
    epistemic_variance = predictions.var(dim=0, unbiased=False)
    residual_square = (labels - mean).square()
    noise_floor = (residual_square.mean(dim=0) - epistemic_variance.mean(dim=0)).clamp_min(
        1e-8
    ).sqrt()
    base_std = (epistemic_variance + noise_floor.square()).sqrt().clamp_min(1e-6)
    normalized_error = (labels - mean).abs() / base_std
    gaussian_quantile = 1.6448536269514722
    scale = (
        torch.quantile(normalized_error, coverage, dim=0) / gaussian_quantile
    ).clamp_min(1e-3)
    return {
        "coverage": coverage,
        "noise_floor": noise_floor,
        "scale": scale,
    }


def calibrated_std(
    predictions: torch.Tensor,
    calibration: Mapping[str, Any],
) -> torch.Tensor:
    variance = predictions.var(dim=0, unbiased=False)
    noise_floor = calibration["noise_floor"]
    scale = calibration["scale"]
    return (variance + noise_floor.square()).sqrt() * scale


def _coverage_report(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    mean = predictions.mean(dim=0)
    std = calibrated_std(predictions, calibration).clamp_min(1e-6)
    normalized = (labels - mean).abs() / std
    result: dict[str, Any] = {}
    for name, multiplier in (
        ("one_sigma", 1.0),
        ("two_sigma", 2.0),
        ("calibration_90", 1.6448536269514722),
    ):
        inside = normalized <= multiplier
        result[name] = {
            "joint": float(inside.all(dim=1).to(torch.float32).mean()),
            "mean_marginal": float(inside.to(torch.float32).mean()),
            "per_parameter": inside.to(torch.float32).mean(dim=0).tolist(),
        }
    result["normalized_absolute_error_p90"] = torch.quantile(
        normalized, 0.9, dim=0
    ).tolist()
    return result


def build_ensemble(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset = Path(args.dataset).expanduser().resolve()
    member_paths = [Path(value).expanduser().resolve() for value in args.checkpoints]
    output = Path(args.output).expanduser().resolve()
    device = torch.device(args.device)
    first = torch.load(member_paths[0], map_location="cpu", weights_only=False)
    target_mode = str(first["target_mode"])
    load_arguments = {
        "downsample": int(first["downsample"]),
        "minimum_start_step": round(args.minimum_start_s * 500),
        "maximum_start_step": round(args.maximum_start_s * 500),
        "minimum_information": None,
    }
    splits = {
        name: load_split(dataset, name, **load_arguments)
        for name in ("validation", "test")
    }
    for split in splits.values():
        apply_target_mode(split, target_mode)
    member_predictions = {}
    checkpoints = None
    for name, split in splits.items():
        prediction, current_checkpoints = _member_predictions(
            member_paths, split, device, args.batch_size
        )
        member_predictions[name] = prediction
        checkpoints = current_checkpoints
    if checkpoints is None:
        raise RuntimeError("no ensemble checkpoints were loaded")
    compatibility_fields = (
        "target_mode",
        "label_names",
        "feature_names",
        "history_steps",
        "feature_count",
        "downsample",
    )
    for path, checkpoint in zip(member_paths, checkpoints, strict=True):
        mismatches = [
            field
            for field in compatibility_fields
            if checkpoint.get(field) != first.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"incompatible ensemble member {path}: {', '.join(mismatches)}"
            )

    window_calibration = fit_uncertainty_calibration(
        member_predictions["validation"], splits["validation"]["labels"]
    )
    validation_group_prediction, validation_group_labels = _aggregate_member_predictions(
        member_predictions["validation"],
        splits["validation"]["labels"],
        splits["validation"]["group_id"],
    )
    group_calibration = fit_uncertainty_calibration(
        validation_group_prediction, validation_group_labels
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(dataset),
        "target_mode": target_mode,
        "members": [str(path) for path in member_paths],
        "member_count": len(member_paths),
        "calibration_scope": "validation one-second windows; parameter-group split is disjoint",
        "metrics": {},
    }
    for name, split in splits.items():
        prediction = member_predictions[name]
        mean = prediction.mean(dim=0)
        window_metrics = regression_metrics(mean, split["labels"], split["label_names"])
        group_prediction, group_labels = _aggregate_member_predictions(
            prediction, split["labels"], split["group_id"]
        )
        group_mean = group_prediction.mean(dim=0)
        report["metrics"][name] = {
            "window": window_metrics,
            "group": regression_metrics(group_mean, group_labels, split["label_names"]),
            "window_uncertainty_coverage": _coverage_report(
                prediction, split["labels"], window_calibration
            ),
            "group_uncertainty_coverage": _coverage_report(
                group_prediction, group_labels, group_calibration
            ),
        }

    artifact = {
        "schema_version": 1,
        "artifact_type": "flight_identification_ensemble",
        "target_mode": target_mode,
        "label_names": tuple(first["label_names"]),
        "feature_names": tuple(first["feature_names"]),
        "history_steps": int(first["history_steps"]),
        "feature_count": int(first["feature_count"]),
        "downsample": int(first["downsample"]),
        "calibration": window_calibration,
        "group_calibration": group_calibration,
        "members": checkpoints,
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output / "ensemble.pt")
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bundle and calibrate an identifier deep ensemble"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--minimum-start-s", type=float, default=0.0)
    parser.add_argument("--maximum-start-s", type=float, default=0.0)
    return parser


def main() -> None:
    build_ensemble(build_parser().parse_args())


if __name__ == "__main__":
    main()
