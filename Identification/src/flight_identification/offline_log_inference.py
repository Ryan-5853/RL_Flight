from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import load_experiment_config
from .control_evaluation import _lqr_gain, _nominal_actuator_model
from .offline_models import build_offline_identifier
from .sim2real_composite import (
    DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
    composite_discrete_model,
    composite_lqr_weights,
    fit_coefficients_from_step_response,
    merge_adaptive_composite_coefficients,
)


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical_log_bundle(
    path: str | Path,
    expected_feature_names: Sequence[str],
    expected_sample_hz: float,
    expected_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        if "features" not in payload or "feature_names" not in payload:
            raise ValueError("log bundle requires features and feature_names arrays")
        features = np.asarray(payload["features"], dtype=np.float32)
        if features.ndim == 2:
            features = features[None]
        if features.ndim != 3:
            raise ValueError("features must have shape [flights,time,features]")
        names = tuple(str(value) for value in payload["feature_names"].tolist())
        if len(names) != features.shape[2] or len(set(names)) != len(names):
            raise ValueError("feature_names must uniquely describe the feature axis")
        missing = set(expected_feature_names) - set(names)
        if missing:
            raise ValueError(f"log bundle is missing features: {sorted(missing)}")
        order = [names.index(name) for name in expected_feature_names]
        features = features[:, :, order]
        sample_hz = float(np.asarray(payload["sample_hz"]).reshape(()))
        valid = (
            np.asarray(payload["valid_mask"], dtype=np.bool_)
            if "valid_mask" in payload
            else np.ones(features.shape[:2], dtype=np.bool_)
        )
    if valid.shape != features.shape[:2]:
        raise ValueError("valid_mask must match the flight and time axes")
    sample_ratio = sample_hz / expected_sample_hz
    integer_ratio = round(sample_ratio)
    if (
        not np.isfinite(sample_hz)
        or integer_ratio < 1
        or not np.isclose(sample_ratio, integer_ratio)
    ):
        raise ValueError(
            f"sample_hz must equal or be an integer multiple of the checkpoint "
            f"rate {expected_sample_hz:g} Hz"
        )
    if features.shape[1] != expected_steps * integer_ratio:
        raise ValueError(
            f"each flight must contain {expected_steps * integer_ratio} source "
            "samples for this checkpoint and sample rate"
        )
    features = features[:, ::integer_ratio]
    valid = valid[:, ::integer_ratio]
    if not valid.any(axis=1).all():
        raise ValueError("every flight must contain at least one valid sample")
    if not np.isfinite(features[valid]).all():
        raise ValueError("valid log samples must be finite")
    features[~valid] = 0.0
    return torch.from_numpy(features), torch.from_numpy(valid), expected_sample_hz


def _flight_information_score(
    features: torch.Tensor,
    valid: torch.Tensor,
    names: Sequence[str],
) -> torch.Tensor:
    rate_indices = [names.index(f"angular_velocity_b.{axis}") for axis in "xyz"]
    command_indices = [
        names.index(name)
        for name in (
            "command.motor_upper",
            "command.motor_lower",
            "command.servo_1",
            "command.servo_2",
            "command.servo_3",
        )
    ]
    rate = features[:, :, rate_indices]
    command = features[:, :, command_indices]
    weights = valid.to(features.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    rate_rms = torch.sqrt(
        (rate.square().sum(dim=2) * weights).sum(dim=1) / count
    )
    movement_valid = valid[:, 1:] & valid[:, :-1]
    movement_count = movement_valid.sum(dim=1).clamp_min(1).to(features.dtype)
    movement = (command[:, 1:] - command[:, :-1]).square().sum(dim=2)
    command_movement_rms = torch.sqrt(
        (movement * movement_valid).sum(dim=1) / movement_count
    )
    return rate_rms + 2.0 * command_movement_rms


def _log_quality_metrics(
    features: torch.Tensor,
    valid: torch.Tensor,
    names: Sequence[str],
    normalized: torch.Tensor,
) -> dict[str, Any]:
    if features.ndim != 3 or valid.shape != features.shape[:2]:
        raise ValueError("quality metrics require [flight,time,feature] logs")
    if normalized.shape != features.shape:
        raise ValueError("normalized log shape must match physical features")
    weights = valid.to(features.dtype)
    count = weights.sum().clamp_min(1.0)
    rate_rms = {}
    for axis in "xyz":
        index = names.index(f"angular_velocity_b.{axis}")
        rate_rms[axis] = float(
            torch.sqrt((features[..., index].square() * weights).sum() / count)
        )
    command_indices = [
        names.index(name)
        for name in (
            "command.motor_upper",
            "command.motor_lower",
            "command.servo_1",
            "command.servo_2",
            "command.servo_3",
        )
    ]
    movement_valid = valid[:, 1:] & valid[:, :-1]
    movement_count = movement_valid.sum().clamp_min(1).to(features.dtype)
    movement = (
        features[:, 1:, command_indices] - features[:, :-1, command_indices]
    ).square().sum(dim=2)
    command_movement_rms = float(
        torch.sqrt((movement * movement_valid).sum() / movement_count)
    )
    valid_normalized = normalized[valid]
    return {
        "axis_rate_rms_rad_s": rate_rms,
        "command_movement_rms_per_sample": command_movement_rms,
        "feature_z_score_abs_p95": float(
            torch.quantile(valid_normalized.abs(), 0.95)
        ),
        "feature_z_score_abs_max": float(valid_normalized.abs().max()),
    }


def _quality_gate(
    metrics: Mapping[str, Any],
    selected_flight_count: int,
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if calibration is None:
        return {
            "available": False,
            "passed_for_independent_validation": False,
            "reasons": ["quality_calibration_not_provided"],
        }
    if calibration.get("artifact_type") != "offline_log_quality_calibration":
        raise ValueError("quality calibration has an unexpected artifact type")
    thresholds = calibration["thresholds"]
    reasons: list[str] = []
    minimum_flights = int(calibration["minimum_selected_flights"])
    if selected_flight_count < minimum_flights:
        reasons.append(
            f"selected_flight_count_below_{minimum_flights}"
        )
    maximum_p95 = float(thresholds["maximum_feature_z_score_abs_p95"])
    if float(metrics["feature_z_score_abs_p95"]) > maximum_p95:
        reasons.append("feature_z_score_abs_p95_above_validation_limit")
    maximum_absolute = float(thresholds["maximum_feature_z_score_abs_max"])
    if float(metrics["feature_z_score_abs_max"]) > maximum_absolute:
        reasons.append("feature_z_score_abs_max_above_validation_limit")
    minimum_movement = float(thresholds["minimum_command_movement_rms_per_sample"])
    if float(metrics["command_movement_rms_per_sample"]) < minimum_movement:
        reasons.append("command_movement_below_validation_limit")
    for axis in "xyz":
        if float(metrics["axis_rate_rms_rad_s"][axis]) < float(
            thresholds["minimum_axis_rate_rms_rad_s"][axis]
        ):
            reasons.append(f"angular_rate_{axis}_coverage_below_validation_limit")
    return {
        "available": True,
        "passed_for_independent_validation": not reasons,
        "reasons": reasons,
        "minimum_selected_flights": minimum_flights,
        "thresholds": thresholds,
        "meaning": (
            "Passing only permits independent simulation and HIL validation. "
            "It never accepts a gain for flight."
        ),
    }


def _build_model(checkpoint: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_offline_identifier(
        str(checkpoint.get("architecture", "mlp")),
        int(checkpoint["history_steps"]),
        int(checkpoint["feature_count"]),
        len(checkpoint["label_names"]),
        tuple(checkpoint["trial_hidden_sizes"]),
        tuple(checkpoint["head_hidden_sizes"]),
        tuple(checkpoint.get("temporal_channels", (64, 96, 128))),
        int(checkpoint.get("recurrent_hidden_size", 96)),
        int(checkpoint.get("recurrent_layers", 2)),
        int(checkpoint.get("trial_embedding_size", 128)),
        float(checkpoint.get("temporal_dropout", 0.05)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _predict_logs(
    checkpoint: Mapping[str, Any],
    features: torch.Tensor,
    valid: torch.Tensor,
    selected: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_features = features[selected]
    selected_valid = valid[selected]
    normalization = checkpoint["normalization"]
    normalized = (
        selected_features - normalization["feature_mean"]
    ) / normalization["feature_std"]
    normalized = torch.where(
        selected_valid[..., None], normalized, torch.zeros_like(normalized)
    )
    histories = torch.cat(
        (normalized, selected_valid[..., None].to(normalized.dtype)), dim=2
    )[None]
    trial_mask = torch.ones(1, len(selected), dtype=torch.bool, device=device)
    model = _build_model(checkpoint, device)
    normalized_prediction = model(histories.to(device), trial_mask).cpu()[0]
    prediction = (
        normalized_prediction * normalization["label_std"]
        + normalization["label_mean"]
    )
    prediction = torch.maximum(
        torch.minimum(prediction, checkpoint["label_max"]), checkpoint["label_min"]
    )
    return prediction, normalized, selected_valid


@torch.no_grad()
def infer(args: argparse.Namespace) -> Mapping[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not 0.0 < args.minimum_valid_fraction <= 1.0:
        raise ValueError("minimum-valid-fraction must be in (0, 1]")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_type") != "sim2real_composite_servo_identifier":
        raise ValueError("expected a composite servo identifier checkpoint")
    feature_names = tuple(checkpoint["feature_names"])
    expected_sample_hz = 500.0 / int(checkpoint["downsample"])
    features, valid, sample_hz = load_canonical_log_bundle(
        args.logs,
        feature_names,
        expected_sample_hz,
        int(checkpoint["history_steps"]),
    )
    score = _flight_information_score(features, valid, feature_names)
    valid_fraction = valid.to(torch.float32).mean(dim=1)
    eligible = valid_fraction >= args.minimum_valid_fraction
    if not bool(eligible.any()):
        raise ValueError(
            "no flight satisfies the minimum valid-sample fraction"
        )
    score = score.masked_fill(~eligible, -torch.inf)
    maximum_trials = int(checkpoint["trials_per_group"])
    selected = torch.argsort(score, descending=True)[: min(maximum_trials, int(eligible.sum()))]
    device = torch.device(args.device)
    prediction, normalized, selected_valid = _predict_logs(
        checkpoint, features, valid, selected, device
    )
    selected_features = features[selected]
    quality_metrics = _log_quality_metrics(
        selected_features, selected_valid, feature_names, normalized
    )
    quality_calibration = None
    quality_calibration_path = None
    if args.quality_calibration:
        quality_calibration_path = Path(args.quality_calibration).expanduser().resolve()
        quality_calibration = json.loads(
            quality_calibration_path.read_text(encoding="utf-8")
        )
        expected_digest = str(quality_calibration.get("checkpoint_sha256", ""))
        actual_digest = checkpoint_sha256(checkpoint_path)
        if expected_digest != actual_digest:
            raise ValueError(
                "quality calibration checkpoint digest does not match the primary checkpoint"
            )
        if tuple(quality_calibration["feature_names"]) != feature_names:
            raise ValueError("quality calibration feature schema differs from checkpoint")
    quality_gate = _quality_gate(
        quality_metrics, len(selected), quality_calibration
    )
    secondary_checkpoint_path = None
    secondary_output_axes: tuple[str, ...] = ()
    secondary_sample_hz = None
    if args.secondary_checkpoint:
        secondary_checkpoint_path = Path(args.secondary_checkpoint).expanduser().resolve()
        secondary = torch.load(
            secondary_checkpoint_path, map_location="cpu", weights_only=False
        )
        if secondary.get("artifact_type") != "sim2real_composite_servo_identifier":
            raise ValueError("secondary checkpoint must be a composite identifier")
        for name in (
            "feature_names",
            "label_names",
            "adaptive_mode_indices",
            "basis_time_constants_s",
            "response_snapshot_times_s",
        ):
            if tuple(secondary[name]) != tuple(checkpoint[name]):
                raise ValueError(f"secondary checkpoint differs in {name}")
        secondary_features, secondary_valid, secondary_sample_hz = load_canonical_log_bundle(
            args.logs,
            feature_names,
            500.0 / int(secondary["downsample"]),
            int(secondary["history_steps"]),
        )
        if len(secondary_features) != len(features):
            raise ValueError("primary and secondary log flight counts differ")
        secondary_prediction, _, _ = _predict_logs(
            secondary, secondary_features, secondary_valid, selected, device
        )
        secondary_output_axes = tuple(
            value.strip()
            for value in args.secondary_output_axes.split(",")
            if value.strip()
        )
        if not secondary_output_axes or any(
            axis not in {"roll", "pitch", "yaw"}
            for axis in secondary_output_axes
        ):
            raise ValueError("secondary-output-axes must select roll, pitch, or yaw")
        selected_columns = [
            index
            for index, name in enumerate(checkpoint["label_names"])
            if any(f".{axis}." in name for axis in secondary_output_axes)
        ]
        prediction[selected_columns] = secondary_prediction[selected_columns]
    basis_tau = tuple(float(value) for value in checkpoint["basis_time_constants_s"])
    mode_indices = tuple(int(value) for value in checkpoint["adaptive_mode_indices"])
    servo_slopes = checkpoint["servo_command_slopes"].numpy()
    nominal_coefficients = checkpoint["nominal_composite_coefficients"].numpy()
    target_representation = str(checkpoint.get("target_representation", "coefficients"))
    if target_representation == "step_response":
        snapshot_times = tuple(
            float(value)
            for value in checkpoint.get(
                "response_snapshot_times_s", DEFAULT_RESPONSE_SNAPSHOT_TIMES_S
            )
        )
        response = prediction.reshape(
            len(snapshot_times), 3, len(mode_indices)
        )
        adaptive_coefficients = fit_coefficients_from_step_response(
            response[None],
            snapshot_times,
            float(np.mean(servo_slopes)),
            basis_tau,
        ).numpy().reshape(1, -1)
    else:
        snapshot_times = ()
        response = None
        adaptive_coefficients = prediction.numpy().reshape(1, -1)
    predicted_coefficients = merge_adaptive_composite_coefficients(
        adaptive_coefficients, nominal_coefficients, mode_indices
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

    def gain(coefficients: np.ndarray) -> tuple[np.ndarray, float]:
        a, b = composite_discrete_model(
            nominal_effectiveness[:, :2],
            command_slopes[:2],
            nominal_tau[:2],
            coefficients,
            checkpoint["mode_transform"].numpy(),
            servo_slopes,
            basis_tau,
        )
        value = _lqr_gain(a, b, q, r)
        radius = float(np.max(np.abs(np.linalg.eigvals(a - b @ value))))
        return value, radius

    nominal_gain, nominal_radius = gain(nominal_coefficients)
    predicted_gain, predicted_radius = gain(predicted_coefficients)
    blend_fraction = float(args.analysis_gain_blend)
    if not 0.0 <= blend_fraction <= 1.0:
        raise ValueError("analysis-gain-blend must be between zero and one")
    analysis_gain = nominal_gain + blend_fraction * (predicted_gain - nominal_gain)
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "secondary_checkpoint": (
            None if secondary_checkpoint_path is None else str(secondary_checkpoint_path)
        ),
        "secondary_output_axes": secondary_output_axes,
        "log_bundle": str(Path(args.logs).expanduser().resolve()),
        "architecture": str(checkpoint.get("architecture", "mlp")),
        "sample_hz": sample_hz,
        "effective_sample_hz_by_checkpoint": {
            "primary": sample_hz,
            "secondary": secondary_sample_hz,
        },
        "selected_flight_indices": selected.tolist(),
        "selected_information_scores": score[selected].tolist(),
        "selected_valid_fractions": valid_fraction[selected].tolist(),
        **quality_metrics,
        "quality_calibration": (
            None if quality_calibration_path is None else str(quality_calibration_path)
        ),
        "identification_quality_gate": quality_gate,
        "response_snapshot_times_s": snapshot_times,
        "predicted_response": None if response is None else response.tolist(),
        "predicted_composite_coefficients": predicted_coefficients.tolist(),
        "nominal_gain": nominal_gain.tolist(),
        "predicted_gain": predicted_gain.tolist(),
        "analysis_gain_blend_fraction": blend_fraction,
        "analysis_gain": analysis_gain.tolist(),
        "identified_model_pole_radius": {
            "nominal_gain_on_nominal_model": nominal_radius,
            "predicted_gain_on_identified_model": predicted_radius,
        },
        "deployment_mode": "offline_analysis_only",
        "gain_accepted_for_flight": False,
        "note": (
            "This tool consumes closed-loop flight logs after landing. The analysis "
            "gain is not a flight-release artifact and has not been validated on the "
            "unknown true plant."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_flights": len(selected),
                "feature_z_score_abs_p95": report["feature_z_score_abs_p95"],
                "quality_gate_passed_for_independent_validation": quality_gate[
                    "passed_for_independent_validation"
                ],
                "quality_gate_reasons": quality_gate["reasons"],
                "analysis_gain_blend_fraction": blend_fraction,
                "identified_model_pole_radius": report[
                    "identified_model_pole_radius"
                ],
                "deployment_mode": report["deployment_mode"],
                "gain_accepted_for_flight": False,
            },
            indent=2,
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer a composite LQI model from post-flight closed-loop logs"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--secondary-checkpoint")
    parser.add_argument("--secondary-output-axes", default="roll,pitch")
    parser.add_argument("--logs", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--analysis-gain-blend", type=float, default=0.9)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.99)
    parser.add_argument("--quality-calibration")
    return parser


def main() -> None:
    infer(build_parser().parse_args())


if __name__ == "__main__":
    main()
