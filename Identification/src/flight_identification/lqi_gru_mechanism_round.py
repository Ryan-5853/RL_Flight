from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .lqi_gru_closed_loop import evaluate_closed_loop
from .lqi_gru_dagger import extend_parameter_payload, generate_dagger_shards
from .lqi_gru_distillation import train
from .lqi_gru_hard_dagger import evaluate_groups
from .lqi_gru_rescue_dagger import _label_rows_from_groups


def run_mechanism_round(
    screen_json: Path,
    parameter_groups: Path,
    dataset: Path,
    config_path: Path,
    start_checkpoint: Path,
    output_root: Path,
    device: torch.device,
    train_count: int,
    holdout_count: int,
    minimum_rescue_gap: float,
    rollout_trials: int,
    epochs: int,
    seed: int,
    previous_command_noise_std: float,
    previous_command_reset_prob: float,
    eval_trials: int,
    duration_s: float,
    exclude_group_ids: Sequence[int] = (),
) -> Mapping[str, Any]:
    screen = json.loads(screen_json.read_text(encoding="utf-8"))
    excluded = set(exclude_group_ids)
    rescuable = [
        row
        for row in screen["oracle_rescue_analysis"]
        if row["rescue_gap"] >= minimum_rescue_gap
        and row["group_id"] not in excluded
    ]
    rescuable.sort(key=lambda row: -row["rescue_gap"])
    train_rows = rescuable[:train_count]
    holdout_rows = rescuable[train_count : train_count + holdout_count]
    if not train_rows:
        raise ValueError("no training groups above the rescue gap threshold")
    train_ids = [row["group_id"] for row in train_rows]
    holdout_ids = [row["group_id"] for row in holdout_rows]
    train_labels = _label_rows_from_groups(parameter_groups, train_ids)

    payload = torch.load(
        dataset / "parameter_groups_audit_only.pt",
        map_location="cpu",
        weights_only=False,
    )
    base_count = payload["labels_audit_only"].shape[0]
    extended = extend_parameter_payload(dataset, train_labels, extra_split=0)
    new_group_ids = list(range(base_count, base_count + len(train_ids)))
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
        dagger_iteration=300,
    )
    student_dir = output_root / "students" / "mechanism_r1"
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

    payload_current = torch.load(
        dataset / "parameter_groups_audit_only.pt",
        map_location="cpu",
        weights_only=False,
    )
    labels_all = payload_current["labels_audit_only"]
    evals: dict[str, Mapping[str, Any]] = {}
    if 249 < labels_all.shape[0]:
        evals["group_249"] = evaluate_groups(
            config_path, checkpoint, labels_all[249:250], [249],
            eval_trials, duration_s, seed + 11, device,
            variants=("nominal_lqi", "oracle_lqi", "gru"),
        )
    if holdout_ids:
        holdout_labels = _label_rows_from_groups(parameter_groups, holdout_ids)
        evals["holdout_mechanism_groups"] = evaluate_groups(
            config_path, checkpoint, holdout_labels, holdout_ids,
            eval_trials, duration_s, seed + 13, device,
            variants=("nominal_lqi", "oracle_lqi", "gru"),
        )
    evals["trained_mechanism_groups"] = evaluate_groups(
        config_path, checkpoint, train_labels, new_group_ids,
        eval_trials, duration_s, seed + 17, device,
        variants=("nominal_lqi", "oracle_lqi", "gru"),
    )
    rescue_labels = _label_rows_from_groups(
        parameter_groups,
        [1673, 1692, 2571, 3628, 3865, 1462, 1434, 573, 1820, 251],
    )
    evals["previous_rescue_groups"] = evaluate_groups(
        config_path, checkpoint, rescue_labels,
        [1673, 1692, 2571, 3628, 3865, 1462, 1434, 573, 1820, 251],
        eval_trials, duration_s, seed + 19, device,
        variants=("nominal_lqi", "oracle_lqi", "gru"),
    )
    test_path = output_root / "test_mechanism_r1.json"
    evals["full_test"] = evaluate_closed_loop(
        dataset, config_path, checkpoint, test_path, device,
        maximum_groups=37, trials=eval_trials, duration_s=duration_s,
        seed=seed + 5, pilot_height_mode="fixed_target", split="test",
        variants=("oracle_lqi", "nominal_lqi", "gru", "gru_reset"),
    )
    summary = {
        "screen": str(screen_json),
        "mechanism_filter": screen.get("label_filters"),
        "train_group_ids": train_ids,
        "holdout_group_ids": holdout_ids,
        "new_dataset_group_ids": new_group_ids,
        "rollout": rollout,
        "train": train_result,
        "checkpoint": str(checkpoint),
        "evals": evals,
    }
    summary_path = output_root / "mechanism_round_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P0b: failure-mode stratified DAgger round (slow lower motor)"
    )
    parser.add_argument("--screen-json", required=True)
    parser.add_argument("--parameter-groups", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--holdout-count", type=int, default=6)
    parser.add_argument("--minimum-rescue-gap", type=float, default=0.5)
    parser.add_argument("--rollout-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--previous-command-noise-std", type=float, default=0.05)
    parser.add_argument("--previous-command-reset-prob", type=float, default=0.05)
    parser.add_argument("--eval-trials", type=int, default=4)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--exclude-group-ids", type=int, nargs="+", default=())
    args = parser.parse_args()
    summary = run_mechanism_round(
        Path(args.screen_json).resolve(),
        Path(args.parameter_groups).resolve(),
        Path(args.dataset).resolve(),
        Path(args.config).resolve(),
        Path(args.start_checkpoint).resolve(),
        Path(args.output_root).resolve(),
        torch.device(args.device),
        args.train_count,
        args.holdout_count,
        args.minimum_rescue_gap,
        args.rollout_trials,
        args.epochs,
        args.seed,
        args.previous_command_noise_std,
        args.previous_command_reset_prob,
        args.eval_trials,
        args.duration_s,
        tuple(args.exclude_group_ids),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
