from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .lqi_gru_closed_loop import evaluate_closed_loop
from .lqi_gru_distillation import evaluate, train


ARCH_DEFAULTS: Mapping[str, Mapping[str, int | None]] = {
    "gru": {"context_steps": None},
    "lstm": {"context_steps": None},
    "tcn": {"context_steps": 128},
    "transformer": {"context_steps": 128},
}

RECURRENT_ARCHS = {"gru", "lstm"}


def compare_architectures(
    dataset: Path,
    config_path: Path,
    output_root: Path,
    device: torch.device,
    epochs: int,
    architectures: Sequence[str],
    exclude_previous_command: bool,
    previous_command_noise_std: float,
    previous_command_reset_prob: float,
    seed: int,
    maximum_test_groups: int,
    trials: int,
    duration_s: float,
) -> Mapping[str, Any]:
    reports: dict[str, Mapping[str, Any]] = {}
    for arch in architectures:
        if arch not in ARCH_DEFAULTS:
            raise ValueError(f"unknown architecture {arch!r}")
        run_dir = output_root / arch
        summary_path = run_dir / "arch_summary.json"
        if summary_path.exists():
            reports[arch] = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"{arch} already complete, reusing {summary_path}")
            continue
        student_dir = run_dir / "student"
        result = train(
            dataset,
            config_path,
            student_dir,
            device,
            epochs_override=epochs,
            exclude_previous_command=exclude_previous_command,
            arch=arch,
            context_steps=ARCH_DEFAULTS[arch]["context_steps"],
            previous_command_noise_std=previous_command_noise_std,
            previous_command_reset_prob=previous_command_reset_prob,
        )
        checkpoint = Path(result["checkpoint"])
        offline_path = run_dir / "offline.json"
        offline = evaluate(
            dataset, checkpoint, offline_path, device, batch_size=64, probe_ridge=0.001
        )
        closed_loop_path = run_dir / "closed_loop_test.json"
        closed_loop = evaluate_closed_loop(
            dataset,
            config_path,
            checkpoint,
            closed_loop_path,
            device,
            maximum_groups=maximum_test_groups,
            trials=trials,
            duration_s=duration_s,
            seed=seed,
            pilot_height_mode="fixed_target",
            split="test",
            variants=(
                ("oracle_lqi", "nominal_lqi", "gru", "gru_reset")
                if arch in RECURRENT_ARCHS
                else ("oracle_lqi", "nominal_lqi", "gru")
            ),
        )
        reports[arch] = {
            "training": result,
            "offline": offline,
            "closed_loop": closed_loop,
        }
        summary_path.write_text(
            json.dumps(reports[arch], indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"{arch} done: {json.dumps(result, sort_keys=True)}")
    comparison_path = output_root / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "config": str(config_path),
                "architectures": architectures,
                "epochs": epochs,
                "exclude_previous_command": exclude_previous_command,
                "previous_command_noise_std": previous_command_noise_std,
                "previous_command_reset_prob": previous_command_reset_prob,
                "reports": reports,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare memory-capable student architectures on the DAgger dataset"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--architectures",
        nargs="*",
        default=("gru", "lstm", "tcn", "transformer"),
    )
    parser.add_argument("--exclude-previous-command", action="store_true")
    parser.add_argument("--previous-command-noise-std", type=float, default=0.05)
    parser.add_argument("--previous-command-reset-prob", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--maximum-test-groups", type=int, default=37)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--duration-s", type=float, default=6.0)
    args = parser.parse_args()
    reports = compare_architectures(
        Path(args.dataset).resolve(),
        Path(args.config).resolve(),
        Path(args.output_root).resolve(),
        torch.device(args.device),
        args.epochs,
        tuple(args.architectures),
        args.exclude_previous_command,
        args.previous_command_noise_std,
        args.previous_command_reset_prob,
        args.seed,
        args.maximum_test_groups,
        args.trials,
        args.duration_s,
    )
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
