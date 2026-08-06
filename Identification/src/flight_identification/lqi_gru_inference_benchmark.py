from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .lqi_gru_distillation import StudentStepState, load_student, step_student


def benchmark(
    checkpoint: Path,
    output: Path,
    device: torch.device,
    steps: int = 2000,
) -> dict[str, float]:
    """Steady-state single-instance (batch 1) inference latency at 500 Hz."""

    model, payload = load_student(checkpoint, device)
    model.eval()
    mean = payload["normalization"]["mean"].to(device)
    std = payload["normalization"]["std"].to(device)
    observation_size = len(payload["observation_names"])
    state = StudentStepState()
    observation = torch.randn(1, observation_size, device=device)
    normalized = (observation - mean) / std
    # warm up
    for _ in range(50):
        _, state = step_student(model, payload, normalized, state, full_context=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        _, state = step_student(model, payload, normalized, state, full_context=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    per_step_ms = 1000.0 * elapsed / steps
    report = {
        "device": str(device),
        "arch": str(payload.get("arch", "gru")),
        "steps": steps,
        "per_step_ms": per_step_ms,
        "max_500hz_budget_ms": 2.0,
        "fits_500hz": per_step_ms < 2.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark single-instance student inference at 500 Hz"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    report = benchmark(
        Path(args.checkpoint).resolve(),
        Path(args.output).resolve(),
        torch.device(args.device),
        args.steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
