#!/usr/bin/env python3
"""Clean real PX4 ULogs into the canonical 500 Hz offline-identification bundle.

Usage (from the repository root):

    PYTHONNOUSERSITE=1 PYTHONPATH=Identification/src \
      .venv/bin/python Identification/scripts/build_real_log_dataset.py \
      --logs Identification/real_log/26.8.12 \
      --output Identification/datasets/real_logs_26_8_12_v1 \
      --mlp-checkpoint Identification/runs/repro_20260806/sim2real_offline_logs_step_response_mlp_v7/identifier.pt

The output directory contains:
    bundle.npz          canonical [flights, 1000, 14] 500 Hz NPZ + valid mask + metadata
    windows.json        per-window selection report
    windows_summary.png one panel per accepted window (attitude/rate/command)
    build_report.json   per-log parse and selection summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from flight_identification.real_log_adapter import (
    FEATURE_NAMES,
    ParsedFlight,
    parse_ulog,
    yaw_free_tilt_quaternion,
)


WINDOW_STEPS = 1000
SAMPLE_HZ = 500
PRE_RELEASE_SAMPLES = 75  # 0.15 s
ACTIVE_THRESHOLD = 0.05
ACTIVE_MIN_S = 0.4
ACTIVE_FILL_S = 0.3
RELEASE_RATE_RAD_S = 0.5
RELEASE_TILT_CHANGE_DEG = 0.7
RELEASE_SUSTAIN_SAMPLES = 25
DISTURBANCE_LEVEL_DEG = 4.0
DISTURBANCE_ONSET_DEG = 8.0
DISTURBANCE_PEAK_DEG = 12.0

TIER_A = "A"
TIER_B = "B"
TIER_C = "C"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tilt_angle_deg(attitude_q_tilt: np.ndarray) -> np.ndarray:
    return np.degrees(
        2.0 * np.arccos(np.clip(np.abs(attitude_q_tilt[..., 0]), 0.0, 1.0))
    )


def roll_pitch_deg(attitude_q_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = np.moveaxis(np.asarray(attitude_q_wb, dtype=np.float64), -1, 0)
    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.degrees(roll), np.degrees(pitch)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _active_epochs(active: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) runs, removing bursts and filling short gaps."""
    min_run = max(1, round(ACTIVE_MIN_S * SAMPLE_HZ))
    fill = round(ACTIVE_FILL_S * SAMPLE_HZ)
    mask = active.copy()
    # Remove short bursts.
    for start, stop in _runs(mask):
        if stop - start < min_run:
            mask[start:stop] = False
    # Fill short gaps between runs.
    runs = list(_runs(mask))
    merged: list[tuple[int, int]] = []
    for start, stop in runs:
        if merged and start - merged[-1][1] <= fill:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    starts = [indices[0]]
    stops: list[int] = []
    previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            stops.append(previous + 1)
            starts.append(index)
        previous = index
    stops.append(previous + 1)
    return list(zip(starts, stops))


def _release_index(
    rate_norm: np.ndarray,
    tilt: np.ndarray,
    epoch_start: int,
    epoch_stop: int,
) -> int:
    """First sustained motion inside an armed epoch (hand release)."""
    smooth = _smooth(np.nan_to_num(tilt, nan=0.0), 25)
    baseline = smooth[epoch_start]
    for index in range(epoch_start, epoch_stop - RELEASE_SUSTAIN_SAMPLES):
        moving = (
            rate_norm[index : index + RELEASE_SUSTAIN_SAMPLES].mean() > RELEASE_RATE_RAD_S
            or abs(smooth[index + RELEASE_SUSTAIN_SAMPLES - 1] - baseline)
            > RELEASE_TILT_CHANGE_DEG
        )
        if moving:
            return index
    return epoch_start + min(round(0.3 * SAMPLE_HZ), max(0, epoch_stop - epoch_start - 1))


def _disturbance_events(
    tilt: np.ndarray,
    active: np.ndarray,
    epoch_start: int,
    epoch_stop: int,
) -> list[int]:
    """Onset indices of closed-loop tilt excursions inside an armed epoch."""
    # Invalid samples are already excluded by ``active``; fill them before the
    # moving average so NaN cannot stall the event scan.
    smooth = _smooth(np.nan_to_num(tilt, nan=0.0), 25)
    events: list[int] = []
    index = epoch_start
    while index < epoch_stop:
        if (
            not active[index]
            or not np.isfinite(smooth[index])
            or smooth[index] >= DISTURBANCE_ONSET_DEG
        ):
            index += 1
            continue
        onset = index
        while index < epoch_stop and active[index] and smooth[index] < DISTURBANCE_ONSET_DEG:
            index += 1
        peak_index = index
        peak = smooth[index] if index < epoch_stop else 0.0
        while index < epoch_stop and active[index] and smooth[index] >= DISTURBANCE_LEVEL_DEG:
            if smooth[index] > peak:
                peak = smooth[index]
                peak_index = index
            index += 1
        if peak >= DISTURBANCE_PEAK_DEG and peak_index - onset >= round(0.2 * SAMPLE_HZ):
            events.append(onset)
    return events


def _saturation_fraction(command: np.ndarray) -> float:
    lower_sat = (command[:, 0] <= 1e-6) | (command[:, 1] >= 1.0 - 1e-6)
    servo_sat = np.abs(command[:, 2:]) >= 1.0 - 1e-6
    return float(((lower_sat[:, None] | servo_sat).any(axis=1)).mean())


def _information_score(features: np.ndarray) -> float:
    rate = features[:, 4:7]
    command = features[:, 9:14]
    rate_rms = float(np.sqrt(np.square(rate).sum(axis=1).mean()))
    movement = np.sqrt(
        np.square(command[1:] - command[:-1]).sum(axis=1).mean()
    )
    return rate_rms + 2.0 * movement


def _window_metrics(
    features: np.ndarray,
    roll: np.ndarray,
    pitch: np.ndarray,
    tilt: np.ndarray,
    rate_norm: np.ndarray,
    command: np.ndarray,
) -> dict[str, float]:
    return {
        "initial_tilt_deg": float(tilt[:50].mean()),
        "initial_roll_deg": float(roll[:50].mean()),
        "initial_pitch_deg": float(pitch[:50].mean()),
        "initial_rate_norm_rad_s": float(rate_norm[0]),
        "peak_tilt_deg": float(tilt.max()),
        "peak_rate_rad_s": float(rate_norm.max()),
        "axis_rate_rms_rad_s": {
            axis: float(np.sqrt(np.square(features[:, 4 + index]).mean()))
            for index, axis in enumerate("xyz")
        },
        "command_movement_rms_per_sample": float(
            np.sqrt(np.square(command[1:] - command[:-1]).sum(axis=1).mean())
        ),
        "saturation_fraction": _saturation_fraction(command),
        "information_score": _information_score(features),
    }


def _tier(metrics: Mapping[str, float]) -> str:
    if (
        5.0 <= metrics["initial_tilt_deg"] <= 25.0
        and metrics["initial_rate_norm_rad_s"] <= 2.5
        and metrics["peak_tilt_deg"] <= 45.0
        and metrics["peak_rate_rad_s"] <= 8.0
    ):
        return TIER_A
    if (
        5.0 <= metrics["initial_tilt_deg"] <= 40.0
        and metrics["initial_rate_norm_rad_s"] <= 5.0
        and metrics["peak_tilt_deg"] <= 55.0
        and metrics["peak_rate_rad_s"] <= 12.0
    ):
        return TIER_B
    return TIER_C


def _hard_accept(
    features: np.ndarray,
    valid: np.ndarray,
    active: np.ndarray,
    metrics: Mapping[str, float],
) -> str | None:
    if not bool(valid.all()):
        return "invalid_samples"
    if not bool(active.all()):
        return "not_armed"
    if not 5.0 <= metrics["initial_tilt_deg"] <= 60.0:
        return "initial_tilt_out_of_range"
    if metrics["peak_tilt_deg"] > 60.0:
        return "peak_tilt_out_of_range"
    if metrics["peak_rate_rad_s"] > 20.0:
        return "peak_rate_out_of_range"
    return None


def _candidate_windows(
    flight: ParsedFlight,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    q_tilt = flight.attitude_q_tilt
    tilt = tilt_angle_deg(q_tilt)
    roll, pitch = roll_pitch_deg(flight.attitude_q_wb)
    rate_norm = np.linalg.norm(flight.angular_velocity_b, axis=1)
    command = flight.command
    features = np.concatenate(
        (q_tilt, flight.angular_velocity_b, flight.motor_speed, command), axis=1
    )
    active_raw = (
        (command[:, 0] > ACTIVE_THRESHOLD)
        | (command[:, 1] > ACTIVE_THRESHOLD)
    ) & flight.valid
    epochs = _active_epochs(active_raw)

    candidates: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for epoch_index, (epoch_start, epoch_stop) in enumerate(epochs):
        if epoch_stop - epoch_start < WINDOW_STEPS:
            rejects.append(
                {
                    "epoch": epoch_index,
                    "epoch_start_us": int(flight.timestamp_us[epoch_start]),
                    "epoch_stop_us": int(flight.timestamp_us[epoch_stop - 1]),
                    "duration_s": (epoch_stop - epoch_start) / SAMPLE_HZ,
                    "reason": "epoch_shorter_than_window",
                }
            )
            continue
        release = _release_index(rate_norm, tilt, epoch_start, epoch_stop)
        starts: list[tuple[int, str]] = [
            (max(epoch_start, release + offset), f"takeoff_offset_{offset}")
            for offset in (-PRE_RELEASE_SAMPLES, 0, 50, 100, 150, 200)
        ]
        starts.extend(
            (max(epoch_start, onset - PRE_RELEASE_SAMPLES), "disturbance")
            for onset in _disturbance_events(tilt, active_raw, epoch_start, epoch_stop)
        )
        takeoff_remaining = True
        for start, kind in sorted(starts):
            if kind.startswith("takeoff_offset") and not takeoff_remaining:
                continue
            stop = start + WINDOW_STEPS
            if stop > len(features):
                continue
            window_features = features[start:stop]
            window_valid = flight.valid[start:stop]
            window_active = active_raw[start:stop]
            metrics = _window_metrics(
                window_features,
                roll[start:stop],
                pitch[start:stop],
                tilt[start:stop],
                rate_norm[start:stop],
                command[start:stop],
            )
            reject_reason = _hard_accept(
                window_features, window_valid, window_active, metrics
            )
            entry = {
                "start_index": start,
                "stop_index": stop,
                "start_us": int(flight.timestamp_us[start]),
                "stop_us": int(flight.timestamp_us[stop - 1]),
                "epoch": epoch_index,
                "release_index": release,
                "window_kind": kind,
                "tier": _tier(metrics),
                **metrics,
            }
            if reject_reason is not None:
                entry["reason"] = reject_reason
                rejects.append(entry)
            else:
                candidates.append(entry)
                if kind.startswith("takeoff_offset"):
                    takeoff_remaining = False
    return candidates, rejects


def _select_windows(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    tier_rank = {TIER_A: 0, TIER_B: 1, TIER_C: 2}
    ordered = sorted(
        candidates,
        key=lambda entry: (
            tier_rank[entry["tier"]],
            0 if str(entry["window_kind"]).startswith("takeoff") else 1,
            -entry["information_score"],
        ),
    )
    selected: list[dict[str, Any]] = []
    for entry in ordered:
        start = int(entry["start_index"])
        source = entry["source"]
        if any(
            other["source"] == source
            and abs(start - int(other["start_index"])) < WINDOW_STEPS
            for other in selected
        ):
            continue
        selected.append(entry)
    selected.sort(key=lambda entry: entry["start_us"])
    return selected


def _mlp_quality_metrics(
    features: np.ndarray,
    checkpoint: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Quality metrics at the MLP's 100 Hz view, using its normalization."""
    downsampled = features[::5]
    mean = checkpoint["normalization"]["feature_mean"].numpy()
    std = checkpoint["normalization"]["feature_std"].numpy()
    normalized = (downsampled - mean) / std
    command = downsampled[:, 9:14]
    return {
        "axis_rate_rms_rad_s": {
            axis: float(np.sqrt(np.square(downsampled[:, 4 + index]).mean()))
            for index, axis in enumerate("xyz")
        },
        "command_movement_rms_per_sample": float(
            np.sqrt(np.square(command[1:] - command[:-1]).sum(axis=1).mean())
        ),
        "feature_z_score_abs_p95": float(np.quantile(np.abs(normalized), 0.95)),
        "feature_z_score_abs_max": float(np.abs(normalized).max()),
        "thresholds": calibration["thresholds"],
    }


def _quality_gate_pass(metrics: Mapping[str, Any]) -> tuple[bool, list[str]]:
    thresholds = metrics["thresholds"]
    reasons: list[str] = []
    for axis in "xyz":
        if metrics["axis_rate_rms_rad_s"][axis] < thresholds[
            "minimum_axis_rate_rms_rad_s"
        ][axis]:
            reasons.append(f"angular_rate_{axis}_coverage_below_validation_limit")
    if (
        metrics["command_movement_rms_per_sample"]
        < thresholds["minimum_command_movement_rms_per_sample"]
    ):
        reasons.append("command_movement_below_validation_limit")
    if metrics["feature_z_score_abs_p95"] > thresholds[
        "maximum_feature_z_score_abs_p95"
    ]:
        reasons.append("feature_z_score_abs_p95_above_validation_limit")
    if metrics["feature_z_score_abs_max"] > thresholds[
        "maximum_feature_z_score_abs_max"
    ]:
        reasons.append("feature_z_score_abs_max_above_validation_limit")
    return (not reasons), reasons


def _summary_plot(
    output: Path,
    windows: Sequence[dict[str, Any]],
    flights: Mapping[str, ParsedFlight],
    selected_entries: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(selected_entries)
    if count == 0:
        return
    columns = 2
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.2 * rows), squeeze=False)
    for index, entry in enumerate(selected_entries):
        axis = axes[index // columns][index % columns]
        flight = flights[entry["source"]]
        start = int(entry["start_index"])
        stop = int(entry["stop_index"])
        time_s = (flight.timestamp_us[start:stop] - flight.timestamp_us[start]) / 1e6
        roll, pitch = roll_pitch_deg(flight.attitude_q_wb[start:stop])
        tilt = tilt_angle_deg(flight.attitude_q_tilt[start:stop])
        rate = flight.angular_velocity_b[start:stop]
        command = flight.command[start:stop]
        axis.plot(time_s, roll, label="roll")
        axis.plot(time_s, pitch, label="pitch")
        axis.plot(time_s, tilt, label="tilt", linestyle="--")
        axis.plot(time_s, np.linalg.norm(rate, axis=1), label="|rate|")
        axis.plot(time_s, command[:, 0], label="cmd upper")
        axis.plot(time_s, command[:, 1], label="cmd lower")
        axis.plot(time_s, command[:, 2], label="cmd s1")
        axis.plot(time_s, command[:, 3], label="cmd s2")
        axis.plot(time_s, command[:, 4], label="cmd s3")
        axis.set_title(
            f"{entry['source']}  tier {entry['tier']}  "
            f"tilt {entry['initial_tilt_deg']:.1f}->{entry['peak_tilt_deg']:.1f} deg"
        )
        axis.set_xlabel("t [s]")
        axis.grid(True, alpha=0.3)
    for index in range(count, rows * columns):
        axes[index // columns][index % columns].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=6, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output / "windows_summary.png", dpi=110)
    plt.close(figure)


def build(
    logs_directory: Path,
    output_directory: Path,
    mlp_checkpoint: Path | None,
    quality_calibration: Path | None,
) -> Mapping[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    files = sorted(logs_directory.glob("*.ulg"))
    seen: set[str] = set()
    sources: list[Path] = []
    duplicates: list[tuple[str, str]] = []
    for path in files:
        digest = md5(path)
        if digest in seen:
            duplicates.append((path.name, digest))
            continue
        seen.add(digest)
        sources.append(path)

    flights: dict[str, ParsedFlight] = {}
    parse_errors: dict[str, str] = {}
    all_candidates: list[dict[str, Any]] = []
    all_rejects: list[dict[str, Any]] = []
    per_log: list[dict[str, Any]] = []
    for path in sources:
        try:
            flight = parse_ulog(path)
        except Exception as error:  # pragma: no cover - defensive
            parse_errors[path.name] = str(error)
            continue
        flights[path.name] = flight
        print(f"parsed {path.name}: {len(flight.timestamp_us)} samples", flush=True)
        candidates, rejects = _candidate_windows(flight)
        for entry in candidates:
            entry["source"] = path.name
        for entry in rejects:
            entry["source"] = path.name
        all_candidates.extend(candidates)
        all_rejects.extend(rejects)
        per_log.append(
            {
                "source": path.name,
                "md5": md5(path),
                "duration_s": round((flight.timestamp_us[-1] - flight.timestamp_us[0]) / 1e6, 3),
        "valid_fraction": float(flight.valid.mean()),
        "max_contiguous_valid_run_s": _max_contiguous_valid_run_s(flight.valid),
                "candidate_windows": len(candidates),
                "rejected_windows": len(rejects),
                "active_fraction": float(
                    (((flight.command[:, 0] > ACTIVE_THRESHOLD) | (flight.command[:, 1] > ACTIVE_THRESHOLD)) & flight.valid).mean()
                ),
            }
        )
        print(
            f"  {path.name}: {len(candidates)} candidates, {len(rejects)} rejects",
            flush=True,
        )

    selected = _select_windows(all_candidates)
    print(f"selected {len(selected)} windows", flush=True)
    selected.sort(key=lambda entry: entry["start_us"])
    for index, entry in enumerate(selected):
        entry["window_index"] = index
    # Overlap between windows from the same log is already prevented; also
    # require each selected window to keep the canonical bundle sorted.
    selected = sorted(selected, key=lambda entry: entry["start_us"])

    feature_rows = []
    valid_rows = []
    meta: dict[str, list[Any]] = {
        "source_log": [],
        "window_start_us": [],
        "window_stop_us": [],
        "tier": [],
        "information_score": [],
        "initial_tilt_deg": [],
        "peak_tilt_deg": [],
        "peak_rate_rad_s": [],
        "saturation_fraction": [],
    }
    for entry in selected:
        flight = flights[entry["source"]]
        start = int(entry["start_index"])
        stop = int(entry["stop_index"])
        features = np.concatenate(
            (
                flight.attitude_q_tilt[start:stop],
                flight.angular_velocity_b[start:stop],
                flight.motor_speed[start:stop],
                flight.command[start:stop],
            ),
            axis=1,
        ).astype(np.float32)
        feature_rows.append(features)
        valid_rows.append(flight.valid[start:stop])
        meta["source_log"].append(entry["source"])
        meta["window_start_us"].append(entry["start_us"])
        meta["window_stop_us"].append(entry["stop_us"])
        meta["tier"].append(entry["tier"])
        meta["information_score"].append(entry["information_score"])
        meta["initial_tilt_deg"].append(entry["initial_tilt_deg"])
        meta["peak_tilt_deg"].append(entry["peak_tilt_deg"])
        meta["peak_rate_rad_s"].append(entry["peak_rate_rad_s"])
        meta["saturation_fraction"].append(entry["saturation_fraction"])

    if feature_rows:
        bundle_features = np.stack(feature_rows)
        bundle_valid = np.stack(valid_rows)
        np.savez_compressed(
            output_directory / "bundle.npz",
            features=bundle_features,
            feature_names=np.asarray(FEATURE_NAMES, dtype="<U64"),
            sample_hz=np.float64(SAMPLE_HZ),
            valid_mask=bundle_valid,
            source_log=np.asarray(meta["source_log"], dtype="<U64"),
            window_start_us=np.asarray(meta["window_start_us"], dtype=np.int64),
            window_stop_us=np.asarray(meta["window_stop_us"], dtype=np.int64),
            tier=np.asarray(meta["tier"], dtype="<U4"),
            information_score=np.asarray(meta["information_score"], dtype=np.float32),
            initial_tilt_deg=np.asarray(meta["initial_tilt_deg"], dtype=np.float32),
            peak_tilt_deg=np.asarray(meta["peak_tilt_deg"], dtype=np.float32),
            peak_rate_rad_s=np.asarray(meta["peak_rate_rad_s"], dtype=np.float32),
            saturation_fraction=np.asarray(meta["saturation_fraction"], dtype=np.float32),
        )

    quality: list[dict[str, Any]] = []
    if mlp_checkpoint is not None and quality_calibration is not None and feature_rows:
        checkpoint = torch.load(mlp_checkpoint, map_location="cpu", weights_only=False)
        calibration = json.loads(quality_calibration.read_text(encoding="utf-8"))
        for entry, features in zip(selected, feature_rows):
            metrics = _mlp_quality_metrics(features, checkpoint, calibration)
            passed, reasons = _quality_gate_pass(metrics)
            quality.append(
                {
                    "source": entry["source"],
                    "window_index": entry["window_index"],
                    "passed_validation_calibration": passed,
                    "reasons": reasons,
                    **{key: value for key, value in metrics.items() if key != "thresholds"},
                }
            )

    report = {
        "schema_version": 1,
        "input_directory": str(logs_directory),
        "total_ulg_files": len(files),
        "unique_ulg_files": len(sources),
        "duplicates": duplicates,
        "parse_errors": parse_errors,
        "sample_hz": SAMPLE_HZ,
        "window_steps": WINDOW_STEPS,
        "selected_windows": len(selected),
        "tier_counts": {
            tier: sum(1 for entry in selected if entry["tier"] == tier)
            for tier in (TIER_A, TIER_B, TIER_C)
        },
        "per_log": per_log,
        "candidates": all_candidates,
        "windows": selected,
        "rejected_windows": all_rejects,
        "mlp_quality_calibration": quality,
        "bundle": str(output_directory / "bundle.npz"),
    }
    (output_directory / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (output_directory / "windows.json").write_text(
        json.dumps(
            {"schema_version": 1, "windows": selected},
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    _summary_plot(output_directory, [], flights, selected)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    return report


def _max_contiguous_valid_run_s(valid: np.ndarray) -> float:
    count = 0
    maximum = 0
    for value in valid:
        count = count + 1 if value else 0
        maximum = max(maximum, count)
    return maximum / SAMPLE_HZ


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mlp-checkpoint", type=Path)
    parser.add_argument("--quality-calibration", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build(
        args.logs,
        args.output,
        args.mlp_checkpoint,
        args.quality_calibration,
    )


if __name__ == "__main__":
    main()
