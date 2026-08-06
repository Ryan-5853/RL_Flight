from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .lqi_gru_closed_loop import evaluate_closed_loop
from .lqi_gru_inference_benchmark import benchmark


def run_estimator_robustness(
    dataset: Path,
    config_path: Path,
    checkpoint: Path,
    output: Path,
    device: torch.device,
    maximum_groups: int,
    trials: int,
    duration_s: float,
    seed: int,
    variants: Sequence[str],
    baseline: bool = True,
) -> list[Mapping[str, Any]]:
    """Single-factor estimator perturbation sweeps (cheap, interpretable)."""

    rows: list[Mapping[str, Any]] = []
    configurations: list[tuple[str, float, float, int]] = []
    if baseline:
        configurations.append(("baseline", 0.0, 0.0, 0))
    for noise in (2.0, 5.0, 10.0):
        configurations.append((f"noise{noise:g}deg", noise, 0.0, 0))
    for bias in (1.0, 3.0):
        configurations.append((f"noise5deg_bias{bias:g}deg", 5.0, bias, 0))
    for latency in (2, 5):
        configurations.append((f"noise5deg_lat{latency}steps", 5.0, 0.0, latency))
    for label, noise, bias, latency in configurations:
        report_path = output / f"{label}.json"
        report = evaluate_closed_loop(
            dataset, config_path, checkpoint, report_path, device,
            maximum_groups=maximum_groups, trials=trials,
            duration_s=duration_s, seed=seed,
            pilot_height_mode="fixed_target", split="test",
            variants=variants,
            attitude_noise_deg=noise,
            attitude_bias_deg=bias,
            attitude_latency_steps=latency,
        )
        rows.append(
            {
                "label": label,
                "noise_deg": noise,
                "bias_deg": bias,
                "latency_steps": latency,
                "variants": {
                    mode: {
                        "safe_fraction": report["variants"][mode]["safe_fraction"],
                        "rate_rms": report["variants"][mode][
                            "rate_tracking_rms_rad_s"
                        ]["mean"],
                        "attitude_rms": report["variants"][mode][
                            "attitude_tracking_rms_rad"
                        ]["mean"],
                    }
                    for mode in report["variants"]
                },
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    (output / "estimator_robustness_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    return rows


def run_long_horizon_audit(
    dataset: Path,
    config_path: Path,
    checkpoint: Path,
    output: Path,
    device: torch.device,
    maximum_groups: int,
    trials: int,
    duration_s: float,
    seed: int,
    variants: Sequence[str],
    segments_s: Sequence[float],
) -> Mapping[str, Any]:
    report = evaluate_closed_loop(
        dataset, config_path, checkpoint, output, device,
        maximum_groups=maximum_groups, trials=trials, duration_s=duration_s,
        seed=seed, pilot_height_mode="fixed_target", split="test",
        variants=variants, report_segments_s=segments_s,
    )
    return report


def run_verification(
    dataset: Path,
    config_path: Path,
    checkpoint: Path,
    output_root: Path,
    device: torch.device,
    robustness_groups: int,
    robustness_trials: int,
    robustness_duration_s: float,
    audit_groups: int,
    audit_trials: int,
    audit_duration_s: float,
    seed: int,
    cpu_benchmark_steps: int,
) -> Mapping[str, Any]:
    variants = ("oracle_lqi", "nominal_lqi", "gru", "gru_reset")
    robustness = run_estimator_robustness(
        dataset, config_path, checkpoint, output_root / "estimator_robustness",
        device, robustness_groups, robustness_trials, robustness_duration_s,
        seed, variants,
    )
    long_horizon = run_long_horizon_audit(
        dataset, config_path, checkpoint, output_root / "long_horizon_30s.json",
        device, audit_groups, audit_trials, audit_duration_s, seed + 1,
        variants, (6.0, 15.0, 30.0),
    )
    gpu_bench = benchmark(checkpoint, output_root / "inference_benchmark_gpu.json", device, cpu_benchmark_steps)
    cpu_bench = benchmark(
        checkpoint, output_root / "inference_benchmark_cpu.json",
        torch.device("cpu"), cpu_benchmark_steps,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "estimator_robustness": robustness,
        "long_horizon_30s": {
            "variants": {
                mode: {
                    "safe_fraction": long_horizon["variants"][mode]["safe_fraction"],
                    "rate_rms": long_horizon["variants"][mode][
                        "rate_tracking_rms_rad_s"
                    ]["mean"],
                    "survival_mean": long_horizon["variants"][mode][
                        "survival_fraction"
                    ]["mean"],
                }
                for mode in long_horizon["variants"]
            },
            "segments": long_horizon["segments"],
            "hidden_norms": long_horizon["hidden_norms"],
        },
        "inference_gpu_ms": gpu_bench["per_step_ms"],
        "inference_cpu_ms": cpu_bench["per_step_ms"],
    }
    (output_root / "verification_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5.3 verification: estimator robustness, long-horizon drift, latency"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robustness-groups", type=int, default=12)
    parser.add_argument("--robustness-trials", type=int, default=2)
    parser.add_argument("--robustness-duration-s", type=float, default=6.0)
    parser.add_argument("--audit-groups", type=int, default=8)
    parser.add_argument("--audit-trials", type=int, default=2)
    parser.add_argument("--audit-duration-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--cpu-benchmark-steps", type=int, default=1000)
    args = parser.parse_args()
    summary = run_verification(
        Path(args.dataset).resolve(),
        Path(args.config).resolve(),
        Path(args.checkpoint).resolve(),
        Path(args.output_root).resolve(),
        torch.device(args.device),
        args.robustness_groups,
        args.robustness_trials,
        args.robustness_duration_s,
        args.audit_groups,
        args.audit_trials,
        args.audit_duration_s,
        args.seed,
        args.cpu_benchmark_steps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
