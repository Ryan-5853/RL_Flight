from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import load_experiment_config
from .lqi_gru_closed_loop import _simulate, evaluate_closed_loop
from .lqi_gru_dagger import (
    _base_manifest,
    extend_parameter_payload,
    generate_dagger_shards,
)
from .lqi_gru_distillation import train
from .lqi_gru_experiment import _raw_config


def _copy_dataset_tree(source: Path, output: Path) -> None:
    """Copy a dataset directory.

    Shards are hardlinked (never modified in place), while payload and manifest
    are real copies because torch.save truncates files in place and would
    otherwise corrupt the source dataset through the shared inode.
    """

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output dataset is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = output / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.name in {"parameter_groups_audit_only.pt", "manifest.json"}:
                import shutil

                shutil.copy2(path, target)
            else:
                try:
                    os.link(path, target)
                except OSError:
                    import shutil

                    shutil.copy2(path, target)


def _labels_from_candidates(
    points: Sequence[Mapping[str, Any]], label_names: Sequence[str]
) -> torch.Tensor:
    rows = []
    for point in points:
        rows.append([float(point["parameters"][name]) for name in label_names])
    return torch.as_tensor(rows, dtype=torch.float32)


@torch.no_grad()
def evaluate_groups(
    config_path: Path,
    checkpoint_path: Path,
    labels: torch.Tensor,
    group_ids: Sequence[int],
    trials: int,
    duration_s: float,
    seed: int,
    device: torch.device,
    pilot_height_mode: str = "fixed_target",
    variants: Sequence[str] = ("oracle_lqi", "nominal_lqi", "gru", "gru_reset"),
) -> Mapping[str, Any]:
    """Paired closed-loop evaluation on arbitrary parameter groups."""

    config = load_experiment_config(config_path)
    raw = _raw_config(config_path)
    ids = torch.as_tensor(list(group_ids), dtype=torch.int64)
    if labels.shape[0] != len(ids):
        raise ValueError("labels and group_ids must have the same row count")
    results: dict[str, Mapping[str, torch.Tensor]] = {}
    for mode in variants:
        results[mode] = _simulate(
            mode, config, raw, labels, ids, trials, duration_s, seed,
            checkpoint_path, device, pilot_height_mode,
        )
    summaries = {}
    for mode, values in results.items():
        summaries[mode] = {
            "safe_fraction": float(values["safe"].to(torch.float32).mean()),
            "rate_tracking_rms_rad_s": float(
                values["rate_tracking_rms_rad_s"].mean()
            ),
            "attitude_tracking_rms_rad": float(
                values["attitude_tracking_rms_rad"].mean()
            ),
            "survival_fraction_mean": float(values["survival_fraction"].mean()),
        }
    return {
        "parameter_groups": len(ids),
        "trials": trials,
        "duration_s": duration_s,
        "variants": summaries,
        "per_episode": {
            mode: {
                "group_id": values["group_id"].tolist(),
                "safe": values["safe"].tolist(),
                "rate_rms": values["rate_tracking_rms_rad_s"].tolist(),
                "attitude_rms": values["attitude_tracking_rms_rad"].tolist(),
            }
            for mode, values in results.items()
        },
    }


def run_hard_plant_dagger(
    candidates_json: Path,
    base_dataset: Path,
    config_path: Path,
    start_checkpoint: Path,
    output_root: Path,
    device: torch.device,
    train_candidates: int,
    eval_candidates: int,
    rollout_trials: int,
    epochs: int,
    seed: int,
    previous_command_noise_std: float,
    previous_command_reset_prob: float,
    eval_trials: int,
    duration_s: float,
) -> Mapping[str, Any]:
    candidates = json.loads(candidates_json.read_text(encoding="utf-8"))[
        "selected_points"
    ]
    candidates = sorted(candidates, key=lambda point: -float(point["selection_score"]))
    train_points = candidates[:train_candidates]
    eval_points = candidates[train_candidates : train_candidates + eval_candidates]
    if len(train_points) < 1 or len(eval_points) < 1:
        raise ValueError("need at least one training and one evaluation candidate")

    payload = torch.load(
        base_dataset / "parameter_groups_audit_only.pt",
        map_location="cpu",
        weights_only=False,
    )
    label_names = tuple(payload["label_names"])
    train_labels = _labels_from_candidates(train_points, label_names)
    eval_labels = _labels_from_candidates(eval_points, label_names)
    eval_group_ids = [2000 + index for index in range(len(eval_points))]
    hard_group_ids = list(
        range(payload["labels_audit_only"].shape[0], payload["labels_audit_only"].shape[0] + len(train_points))
    )

    merged = output_root / "datasets" / "merged"
    if not (merged.exists() and any(merged.iterdir())):
        _copy_dataset_tree(base_dataset, merged)
        extend_parameter_payload(merged, train_labels, extra_split=0)
    manifest = _base_manifest(merged)
    extended = manifest.get("extended_parameter_groups", {})
    if extended.get("new_group_ids") != hard_group_ids:
        raise ValueError("hard-plant dataset payload does not match the requested groups")

    reports: list[Mapping[str, Any]] = []

    def evaluate_all(checkpoint: Path, label: str) -> Mapping[str, Any]:
        test_path = output_root / "evals" / f"test_{label}.json"
        test = evaluate_closed_loop(
            merged,
            config_path,
            checkpoint,
            test_path,
            device,
            maximum_groups=37,
            trials=eval_trials,
            duration_s=duration_s,
            seed=seed + 5,
            pilot_height_mode="fixed_target",
            split="test",
            variants=("oracle_lqi", "nominal_lqi", "gru", "gru_reset"),
        )
        hard_path = output_root / "evals" / f"hard_holdout_{label}.json"
        hard = evaluate_groups(
            config_path,
            checkpoint,
            eval_labels,
            eval_group_ids,
            eval_trials,
            duration_s,
            seed + 9,
            device,
            variants=("oracle_lqi", "nominal_lqi", "gru"),
        )
        hard_path.write_text(
            json.dumps(hard, indent=2, sort_keys=True), encoding="utf-8"
        )
        return {"test": test, "hard_holdout": hard}

    baseline = evaluate_all(start_checkpoint, "before")
    reports.append({"stage": "baseline", "checkpoint": str(start_checkpoint), **baseline})

    checkpoint = start_checkpoint
    for round_index in range(2):
        student_dir = output_root / "students" / f"hard_r{round_index + 1}"
        if student_dir.exists() and any(student_dir.iterdir()):
            checkpoint = student_dir / "student.pt"
            train_result = {"checkpoint": str(checkpoint), "resumed_existing": True}
            rollout = {"skipped_existing": True, "iteration": 100 + round_index}
        else:
            rollout = generate_dagger_shards(
                config_path,
                merged,
                checkpoint,
                merged,
                train_groups=0,
                validation_groups=0,
                trials=rollout_trials,
                device=device,
                seed=seed + 100 * (round_index + 1),
                selected_group_ids=hard_group_ids,
                dagger_iteration=100 + round_index,
            )
            train_result = train(
                merged,
                config_path,
                student_dir,
                device,
                epochs_override=epochs,
                arch="gru",
                context_steps=None,
                previous_command_noise_std=previous_command_noise_std,
                previous_command_reset_prob=previous_command_reset_prob,
            )
            checkpoint = Path(train_result["checkpoint"])
        evaluation = evaluate_all(checkpoint, f"r{round_index + 1}")
        entry = {
            "stage": f"round_{round_index + 1}",
            "rollout": rollout,
            "train": train_result,
            **evaluation,
        }
        reports.append(entry)
        print(json.dumps(entry, indent=2, sort_keys=True))
        summary_path = output_root / "hard_dagger_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "candidates": str(candidates_json),
                    "base_dataset": str(base_dataset),
                    "train_candidates": train_candidates,
                    "eval_candidates": eval_candidates,
                    "hard_group_ids": hard_group_ids,
                    "eval_group_ids": eval_group_ids,
                    "stages": reports,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return {"hard_group_ids": hard_group_ids, "stages": reports}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAgger rounds targeting screened hard plants where nominal LQI fails"
    )
    parser.add_argument("--candidates-json", required=True)
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-candidates", type=int, default=24)
    parser.add_argument("--eval-candidates", type=int, default=10)
    parser.add_argument("--rollout-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--previous-command-noise-std", type=float, default=0.05)
    parser.add_argument("--previous-command-reset-prob", type=float, default=0.05)
    parser.add_argument("--eval-trials", type=int, default=4)
    parser.add_argument("--duration-s", type=float, default=6.0)
    args = parser.parse_args()
    summary = run_hard_plant_dagger(
        Path(args.candidates_json).resolve(),
        Path(args.base_dataset).resolve(),
        Path(args.config).resolve(),
        Path(args.start_checkpoint).resolve(),
        Path(args.output_root).resolve(),
        torch.device(args.device),
        args.train_candidates,
        args.eval_candidates,
        args.rollout_trials,
        args.epochs,
        args.seed,
        args.previous_command_noise_std,
        args.previous_command_reset_prob,
        args.eval_trials,
        args.duration_s,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
