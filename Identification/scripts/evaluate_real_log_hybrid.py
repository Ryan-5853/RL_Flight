#!/usr/bin/env python3
"""Per-window v7 hybrid identification on a cleaned real-log canonical bundle.

The v7 offline hybrid (Identification/runs/repro_20260806) uses:
    roll  -> TCN v7 (500 Hz)
    pitch -> BiGRU v7b (500 Hz)
    yaw   -> MLP v7 (100 Hz view of the same 500 Hz bundle)

This script runs every window through all three encoders, merges per-axis
response snapshots, fits the composite servo coefficients, and synthesizes the
4-output LQI gain for each window and for the official top-8 set aggregate.
Output is analysis-only metadata; no gain is accepted for flight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from flight_identification.control_evaluation import _lqr_gain, _nominal_actuator_model
from flight_identification.offline_log_inference import (
    _flight_information_score,
    _log_quality_metrics,
    _predict_logs,
    load_canonical_log_bundle,
)
from flight_identification.sim2real_composite import (
    composite_discrete_model,
    composite_lqr_weights,
    fit_coefficients_from_step_response,
    merge_adaptive_composite_coefficients,
)
from flight_identification.config import load_experiment_config


def _load_predictions(
    checkpoint_path: Path,
    bundle: Path,
    feature_names: tuple[str, ...],
    device: torch.device,
) -> tuple[Mapping[str, Any], torch.Tensor]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    features, valid, sample_hz = load_canonical_log_bundle(
        bundle,
        feature_names,
        500.0 / int(checkpoint["downsample"]),
        int(checkpoint["history_steps"]),
    )
    with torch.no_grad():
        predictions = torch.stack(
            [
                _predict_logs(
                    checkpoint,
                    features,
                    valid,
                    torch.tensor([index], dtype=torch.int64),
                    device,
                )[0]
                for index in range(len(features))
            ]
        )
    return checkpoint, predictions


def _hybrid_prediction(
    predictions: Mapping[str, torch.Tensor],
    label_names: tuple[str, ...],
    axes: Mapping[str, str],
) -> torch.Tensor:
    first = predictions[axes["roll"]]
    output = first[None].clone() if first.ndim == 1 else first.clone()
    for name, source in axes.items():
        columns = [
            index
            for index, label in enumerate(label_names)
            if f".{name}." in label
        ]
        value = predictions[source]
        value = value[None] if value.ndim == 1 else value
        output[:, columns] = value[:, columns]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--tcn-checkpoint", required=True, type=Path)
    parser.add_argument("--bigru-checkpoint", required=True, type=Path)
    parser.add_argument("--mlp-checkpoint", required=True, type=Path)
    parser.add_argument("--analysis-gain-blend", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tcn = torch.load(args.tcn_checkpoint, map_location="cpu", weights_only=False)
    feature_names = tuple(tcn["feature_names"])
    label_names = tuple(tcn["label_names"])
    bundle_features, bundle_valid, bundle_hz = load_canonical_log_bundle(
        args.bundle, feature_names, 500.0, 1000
    )
    tcn_ckpt, tcn_pred = _load_predictions(
        args.tcn_checkpoint, args.bundle, feature_names, device
    )
    bigru_ckpt, bigru_pred = _load_predictions(
        args.bigru_checkpoint, args.bundle, feature_names, device
    )
    mlp_ckpt, mlp_pred = _load_predictions(
        args.mlp_checkpoint, args.bundle, feature_names, device
    )
    hybrid = _hybrid_prediction(
        {"roll": tcn_pred, "pitch": bigru_pred, "yaw": mlp_pred},
        label_names,
        {"roll": "roll", "pitch": "pitch", "yaw": "yaw"},
    )

    scores = _flight_information_score(bundle_features, bundle_valid, feature_names)
    top_count = min(8, len(bundle_features))
    top_selected = torch.argsort(scores, descending=True)[:top_count]
    # Quality metrics follow the MLP checkpoint's 100 Hz view, matching the
    # official validation calibration artifact.
    mlp_features, mlp_valid, _ = load_canonical_log_bundle(
        args.bundle,
        feature_names,
        100.0,
        int(mlp_ckpt["history_steps"]),
    )
    top_mlp = mlp_features[top_selected]
    top_mlp_valid = mlp_valid[top_selected]
    top_normalized = (
        top_mlp - mlp_ckpt["normalization"]["feature_mean"]
    ) / mlp_ckpt["normalization"]["feature_std"]
    top_normalized = torch.where(
        top_mlp_valid[..., None], top_normalized, torch.zeros_like(top_normalized)
    )
    quality_metrics = _log_quality_metrics(
        top_mlp, top_mlp_valid, feature_names, top_normalized
    )

    basis_tau = tuple(float(value) for value in tcn_ckpt["basis_time_constants_s"])
    mode_indices = tuple(int(value) for value in tcn_ckpt["adaptive_mode_indices"])
    servo_slopes = tcn_ckpt["servo_command_slopes"].numpy()
    nominal_coefficients = tcn_ckpt["nominal_composite_coefficients"].numpy()
    snapshot_times = tuple(
        float(value) for value in tcn_ckpt["response_snapshot_times_s"]
    )

    def coefficients_from_response(response: torch.Tensor) -> np.ndarray:
        response = response.reshape(len(snapshot_times), 3, len(mode_indices))
        adaptive = fit_coefficients_from_step_response(
            response[None],
            snapshot_times,
            float(np.mean(servo_slopes)),
            basis_tau,
        ).numpy().reshape(1, -1)
        return merge_adaptive_composite_coefficients(
            adaptive, nominal_coefficients, mode_indices
        )[0]

    experiment = load_experiment_config(args.experiment_config)
    nominal_effectiveness, command_slopes, nominal_tau = _nominal_actuator_model(
        experiment.simulator_config
    )
    from flight_controller import load_controller_config

    lqr = load_controller_config(experiment.controller_config)["params"]["lqr"]
    q, r = composite_lqr_weights(
        np.asarray(lqr["state_scales"], dtype=np.float64),
        np.asarray(lqr["integral_state_scales"], dtype=np.float64),
        np.asarray(lqr["input_scales"], dtype=np.float64),
        float(lqr["input_weight_scale"]),
        len(basis_tau),
    )

    def gain_from_coefficients(coefficients: np.ndarray) -> tuple[np.ndarray, float]:
        a, b = composite_discrete_model(
            nominal_effectiveness[:, :2],
            command_slopes[:2],
            nominal_tau[:2],
            coefficients,
            tcn_ckpt["mode_transform"].numpy(),
            servo_slopes,
            basis_tau,
            upper_external=True,
        )
        value = _lqr_gain(a, b, q, r)
        radius = float(np.max(np.abs(np.linalg.eigvals(a - b @ value))))
        return value, radius

    nominal_gain, nominal_radius = gain_from_coefficients(nominal_coefficients)
    window_records = []
    for index in range(len(bundle_features)):
        coefficients = coefficients_from_response(hybrid[index])
        predicted_gain, predicted_radius = gain_from_coefficients(coefficients)
        blend = float(args.analysis_gain_blend)
        analysis_gain = nominal_gain + blend * (predicted_gain - nominal_gain)
        window_records.append(
            {
                "window_index": index,
                "information_score": float(scores[index]),
                "predicted_composite_coefficients": coefficients.tolist(),
                "predicted_gain": predicted_gain.tolist(),
                "analysis_gain": analysis_gain.tolist(),
                "identified_model_pole_radius": predicted_radius,
            }
        )

    with torch.no_grad():
        aggregate_prediction = _hybrid_prediction(
            {
                "roll": _predict_logs(
                    tcn_ckpt, bundle_features, bundle_valid, top_selected, device
                )[0],
                "pitch": _predict_logs(
                    bigru_ckpt, bundle_features, bundle_valid, top_selected, device
                )[0],
                "yaw": _predict_logs(
                    mlp_ckpt, mlp_features, mlp_valid, top_selected, device
                )[0],
            },
            label_names,
            {"roll": "roll", "pitch": "pitch", "yaw": "yaw"},
        )
    aggregate_coefficients = coefficients_from_response(aggregate_prediction)
    aggregate_gain, aggregate_radius = gain_from_coefficients(aggregate_coefficients)
    aggregate_blend = float(args.analysis_gain_blend)
    aggregate_analysis_gain = nominal_gain + aggregate_blend * (
        aggregate_gain - nominal_gain
    )

    report = {
        "schema_version": 1,
        "artifact_type": "real_log_v7_hybrid_offline_identification",
        "bundle": str(args.bundle),
        "axis_checkpoints": {
            "roll": str(args.tcn_checkpoint),
            "pitch": str(args.bigru_checkpoint),
            "yaw": str(args.mlp_checkpoint),
        },
        "window_count": len(bundle_features),
        "selected_window_count": int(top_count),
        "selected_window_indices": top_selected.tolist(),
        "selected_information_scores": scores[top_selected].tolist(),
        **quality_metrics,
        "quality_calibration": (
            "runs/repro_20260806/sim2real_offline_hybrid_v7/"
            "offline_log_quality_calibration_v1.json"
        ),
        "windows": window_records,
        "aggregate": {
            "predicted_composite_coefficients": aggregate_coefficients.tolist(),
            "predicted_gain": aggregate_gain.tolist(),
            "analysis_gain": aggregate_analysis_gain.tolist(),
            "identified_model_pole_radius": aggregate_radius,
            "nominal_gain_on_nominal_model": nominal_radius,
        },
        "deployment_mode": "offline_analysis_only",
        "gain_accepted_for_flight": False,
        "note": (
            "Per-window and top-8 aggregate identification on cleaned real logs. "
            "Analysis-only; no gain is accepted for flight."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=lambda value: value.item()),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "windows": len(bundle_features),
                "selected": int(top_count),
                "aggregate_pole_radius": aggregate_radius,
                "nominal_pole_radius": nominal_radius,
                "quality_feature_z_p95": quality_metrics[
                    "feature_z_score_abs_p95"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
