from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .control_evaluation import (
    _closed_loop_result,
    _discrete_model,
    _lqr_gain,
    _nominal_actuator_model,
    _scaled_actuator_model,
    _summary,
)
from .gain_training import _lqr_weights
from .training import (
    _aggregate_by_group,
    _predict,
    create_identifier_model,
    effective_lqr_labels,
    load_split,
)


def evaluate_gain_identifier(args: argparse.Namespace) -> Mapping[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_type") != "lqr_gain_pca_identifier":
        raise ValueError("expected an LQR gain PCA identifier")
    split = load_split(
        Path(args.dataset).expanduser().resolve(),
        args.split,
        int(checkpoint["downsample"]),
        0,
        0,
        None,
    )
    normalization = checkpoint["normalization"]
    features = (
        split["features"] - normalization["feature_mean"]
    ) / normalization["feature_std"]
    device = torch.device(args.device)
    model = create_identifier_model(
        "mlp",
        int(checkpoint["history_steps"]),
        int(checkpoint["feature_count"]),
        int(checkpoint["pca_components"].shape[0]),
        tuple(checkpoint["hidden_sizes"]),
        0.0,
        128,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    normalized_prediction = _predict(
        model, features, device, args.batch_size
    )
    latent = (
        normalized_prediction * normalization["label_std"]
        + normalization["label_mean"]
    )
    gains = checkpoint["pca_gain_mean"] + latent @ checkpoint["pca_components"]
    effective_labels = effective_lqr_labels(split["labels"])
    if args.aggregation == "group":
        gains, effective_labels = _aggregate_by_group(
            gains, effective_labels, split["group_id"]
        )
    nominal_gain = checkpoint["nominal_gain"].numpy()
    nominal_effectiveness, command_slopes, nominal_tau = _nominal_actuator_model(
        Path(args.simulator_config).expanduser().resolve()
    )
    q, r = _lqr_weights(Path(args.controller_config).expanduser().resolve())
    state_scales = 1.0 / np.sqrt(np.diag(q))
    initial_covariance = np.diag(state_scales**2)
    variants = ("nominal", "blend_25", "blend_50", "blend_75", "predicted", "oracle")
    radii = {name: [] for name in variants}
    costs = {name: [] for name in variants}
    gain_errors = []
    for predicted_flat, label in zip(gains.numpy(), effective_labels.numpy()):
        effectiveness, tau = _scaled_actuator_model(
            label, nominal_effectiveness, nominal_tau
        )
        true_a, true_b = _discrete_model(
            effectiveness, command_slopes, tau, 1.0 / 500.0
        )
        oracle = _lqr_gain(true_a, true_b, q, r)
        predicted = predicted_flat.reshape(5, 10)
        current = {
            "nominal": nominal_gain,
            "blend_25": nominal_gain + 0.25 * (predicted - nominal_gain),
            "blend_50": nominal_gain + 0.50 * (predicted - nominal_gain),
            "blend_75": nominal_gain + 0.75 * (predicted - nominal_gain),
            "predicted": predicted,
            "oracle": oracle,
        }
        for name, gain in current.items():
            radius, cost = _closed_loop_result(
                true_a, true_b, gain, q, r, initial_covariance
            )
            radii[name].append(radius)
            costs[name].append(cost)
        gain_errors.append(
            np.linalg.norm(predicted - oracle) / np.linalg.norm(oracle)
        )
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "aggregation": args.aggregation,
        "evaluation_count": len(gains),
        "unique_group_count": int(split["group_id"].unique().numel()),
        "stable_fraction": {
            name: float(np.mean(np.asarray(value) < 1.0))
            for name, value in radii.items()
        },
        "pole_radius": {name: _summary(value) for name, value in radii.items()},
        "absolute_cost": {name: _summary(value) for name, value in costs.items()},
        "predicted_gain_relative_error": _summary(gain_errors),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a PCA gain identifier")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--aggregation", default="window", choices=("window", "group"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--simulator-config", default="SimEnv/configs/example.yaml")
    parser.add_argument(
        "--controller-config",
        default="Controller/configs/lqr_identification_nominal.yaml",
    )
    return parser


def main() -> None:
    evaluate_gain_identifier(build_parser().parse_args())


if __name__ == "__main__":
    main()
