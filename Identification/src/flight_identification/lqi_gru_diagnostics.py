from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .lqi_gru_distillation import load_student


@torch.no_grad()
def fixed_state_action_feedback_diagnostic(
    dataset: Path,
    checkpoint: Path,
    output: Path,
    device: torch.device,
    maximum_episodes: int,
) -> dict[str, Any]:
    shard_path = sorted((dataset / "test").glob("shard_*.pt"))[0]
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    observations = shard["observations"][:maximum_episodes].to(device)
    targets = shard["teacher_actions"][:maximum_episodes].to(device)
    valid_mask = shard["valid_mask"][:maximum_episodes].to(device)
    model, checkpoint_payload = load_student(checkpoint, device)
    model.eval()
    mean = checkpoint_payload["normalization"]["mean"].to(device)
    std = checkpoint_payload["normalization"]["std"].to(device)

    full_prediction, _, _ = model((observations - mean) / std)
    hidden = None
    stepwise_parts = []
    for step in range(observations.shape[1]):
        prediction, hidden, _ = model(
            (observations[:, step : step + 1] - mean) / std, hidden
        )
        stepwise_parts.append(prediction)
    stepwise_prediction = torch.cat(stepwise_parts, dim=1)

    def run(autoregressive_action: bool, reset_hidden: bool) -> dict[str, Any]:
        hidden_state = None
        previous_command = observations[:, 0, 20:25].clone()
        squared_error = torch.zeros(5, device=device)
        absolute_error = torch.zeros(5, device=device)
        sample_count = 0
        hidden_norm: dict[str, float] = {}
        instant_rmse: dict[str, float] = {}
        for step in range(observations.shape[1]):
            current = observations[:, step].clone()
            if autoregressive_action:
                current[:, 20:25] = previous_command
            prediction_sequence, next_hidden, _ = model(
                ((current - mean) / std).unsqueeze(1),
                None if reset_hidden else hidden_state,
            )
            prediction = prediction_sequence[:, 0]
            selected = valid_mask[:, step]
            error = prediction[selected] - targets[:, step][selected]
            squared_error += error.square().sum(dim=0)
            absolute_error += error.abs().sum(dim=0)
            sample_count += int(selected.sum())
            previous_command = prediction
            if not reset_hidden:
                hidden_state = next_hidden
            if step in {249, 999, observations.shape[1] - 1}:
                key = f"{(step + 1) / 500.0:g}s"
                hidden_norm[key] = float(next_hidden.norm(dim=2).mean())
                instant_rmse[key] = float(error.square().mean().sqrt())
        return {
            "rmse": (squared_error / sample_count).sqrt().cpu().tolist(),
            "mae": (absolute_error / sample_count).cpu().tolist(),
            "hidden_norm": hidden_norm,
            "instant_rmse": instant_rmse,
        }

    report = {
        "schema_version": 1,
        "episodes": len(observations),
        "full_sequence_vs_stepwise_hidden_max_abs": float(
            (full_prediction - stepwise_prediction).abs().max()
        ),
        "teacher_action_recurrent": run(False, False),
        "teacher_action_reset": run(False, True),
        "autoregressive_action_recurrent": run(True, False),
        "autoregressive_action_reset": run(True, True),
        "interpretation": (
            "Physical observations remain on the recorded oracle trajectory. Only "
            "the previous-command fields are optionally replaced by student outputs."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose recurrent stepping and previous-action feedback"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-episodes", type=int, default=32)
    args = parser.parse_args()
    report = fixed_state_action_feedback_diagnostic(
        Path(args.dataset).resolve(),
        Path(args.checkpoint).resolve(),
        Path(args.output).resolve(),
        torch.device(args.device),
        args.maximum_episodes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
