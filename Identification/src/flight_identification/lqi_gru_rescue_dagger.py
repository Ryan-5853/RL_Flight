from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .lqi_gru_dagger import extend_parameter_payload, generate_dagger_shards
from .lqi_gru_distillation import train
from .lqi_gru_hard_dagger import evaluate_groups


def _label_rows_from_groups(
    parameter_groups: Path, group_ids: Sequence[int]
) -> torch.Tensor:
    payload = torch.load(parameter_groups, map_location="cpu", weights_only=False)
    ids = payload["group_id"]
    lookup = {int(group): index for index, group in enumerate(ids)}
    rows = torch.stack([payload["labels"][lookup[group]] for group in group_ids])
    return rows


def run_rescue_dagger(
    screen_json: Path,
    parameter_groups: Path,
    dataset: Path,
    config_path: Path,
    start_checkpoint: Path,
    output_root: Path,
    device: torch.device,
    maximum_groups: int,
    minimum_rescue_gap: float,
    rollout_trials: int,
    epochs: int,
    seed: int,
    previous_command_noise_std: float,
    previous_command_reset_prob: float,
    eval_trials: int,
    duration_s: float,
) -> Mapping[str, Any]:
    screen = json.loads(screen_json.read_text(encoding="utf-8"))
    rescuable = [
        row
        for row in screen["oracle_rescue_analysis"]
        if row["rescue_gap"] >= minimum_rescue_gap
    ]
    rescuable = rescuable[:maximum_groups]
    if not rescuable:
        raise ValueError("no rescuable groups above the gap threshold")
    group_ids = [row["group_id"] for row in rescuable]
    labels = _label_rows_from_groups(parameter_groups, group_ids)

    payload = torch.load(
        dataset / "parameter_groups_audit_only.pt",
        map_location="cpu",
        weights_only=False,
    )
    base_count = payload["labels_audit_only"].shape[0]
    extended = extend_parameter_payload(dataset, labels, extra_split=0)
    new_group_ids = list(range(base_count, base_count + len(group_ids)))
    if extended["new_group_ids"] != new_group_ids:
        raise RuntimeError("payload extension mismatch")

    rollout = generate_dagger_shards(
        config_path,
        dataset,
        start_checkpoint,
        dataset,
        train_groups=0,
        validation_groups=0,
        trials=rollout_trials,
        device=device,
        seed=seed + 7,
        selected_group_ids=new_group_ids,
        dagger_iteration=200,
    )
    student_dir = output_root / "students" / "rescue_r1"
    train_result = train(
        dataset,
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

    evals: dict[str, Mapping[str, Any]] = {}
    payload_current = torch.load(
        dataset / "parameter_groups_audit_only.pt",
        map_location="cpu",
        weights_only=False,
    )
    labels_all = payload_current["labels_audit_only"]
    if 249 < labels_all.shape[0]:
        evals["group_249"] = evaluate_groups(
            config_path, checkpoint, labels_all[249:250], [249],
            eval_trials, duration_s, seed + 11, device,
            variants=("nominal_lqi", "oracle_lqi", "gru"),
        )
    rescue_groups = [int(g) for g in group_ids]
    evals["new_rescue_groups"] = evaluate_groups(
        config_path, checkpoint, labels, rescue_groups,
        eval_trials, duration_s, seed + 13, device,
        variants=("nominal_lqi", "oracle_lqi", "gru"),
    )
    summary = {
        "screen": str(screen_json),
        "selected_group_ids": group_ids,
        "new_dataset_group_ids": new_group_ids,
        "rollout": rollout,
        "train": train_result,
        "checkpoint": str(checkpoint),
        "evals": evals,
    }
    summary_path = output_root / "rescue_dagger_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAgger round on oracle-rescuable safety-crash plants"
    )
    parser.add_argument("--screen-json", required=True)
    parser.add_argument("--parameter-groups", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-groups", type=int, default=12)
    parser.add_argument("--minimum-rescue-gap", type=float, default=0.5)
    parser.add_argument("--rollout-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--previous-command-noise-std", type=float, default=0.05)
    parser.add_argument("--previous-command-reset-prob", type=float, default=0.05)
    parser.add_argument("--eval-trials", type=int, default=4)
    parser.add_argument("--duration-s", type=float, default=6.0)
    args = parser.parse_args()
    summary = run_rescue_dagger(
        Path(args.screen_json).resolve(),
        Path(args.parameter_groups).resolve(),
        Path(args.dataset).resolve(),
        Path(args.config).resolve(),
        Path(args.start_checkpoint).resolve(),
        Path(args.output_root).resolve(),
        torch.device(args.device),
        args.maximum_groups,
        args.minimum_rescue_gap,
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
