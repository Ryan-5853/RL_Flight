from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import load_experiment_config
from .control_evaluation import _nominal_actuator_model
from .experiment import _sim2real_lqr_targets
from .offline_models import build_offline_identifier
from .repeated_trial_training import (
    _fixed_trial_mask,
    _random_trial_mask,
    _retain_converged_trials,
    fit_repeated_normalization,
    load_repeated_split,
    prepare_histories,
)
from .sim2real_composite import (
    DEFAULT_BASIS_TIME_CONSTANTS_S,
    DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
    composite_response_target_names,
    composite_target_names,
    fit_coefficients_from_step_response,
    fit_composite_coefficients,
    reconstruct_composite_step_response,
    servo_mode_transform,
    true_servo_mode_step_response,
)
from simenv.config import load_and_materialize


@torch.no_grad()
def _predict(
    model: nn.Module,
    histories: torch.Tensor,
    available: torch.Tensor,
    device: torch.device,
    batch_size: int,
    trial_count: int,
) -> torch.Tensor:
    model.eval()
    output = []
    for start in range(0, len(histories), batch_size):
        value = histories[start : start + batch_size].to(device)
        mask = _fixed_trial_mask(
            available[start : start + batch_size].to(device), trial_count
        )
        output.append(model(value, mask).cpu())
    return torch.cat(output)


def _coefficient_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    names: tuple[str, ...],
) -> dict[str, Any]:
    error = prediction - target
    total = (target - target.mean(dim=0)).square().sum(dim=0)
    r2 = 1.0 - error.square().sum(dim=0) / total.clamp_min(1e-12)
    normalized_rmse = error.square().mean(dim=0).sqrt() / target.std(
        dim=0
    ).clamp_min(1e-12)
    return {
        "mean_r2": float(r2.mean()),
        "median_r2": float(r2.median()),
        "mean_normalized_rmse": float(normalized_rmse.mean()),
        "parameters": [
            {
                "name": name,
                "r2": float(r2[index]),
                "normalized_rmse": float(normalized_rmse[index]),
            }
            for index, name in enumerate(names)
        ],
    }


def _response_metrics(
    prediction: torch.Tensor,
    target_coefficients: torch.Tensor,
    true_targets: torch.Tensor,
    nominal_coefficients: torch.Tensor,
    mode_transform: np.ndarray,
    servo_slopes: np.ndarray,
    adaptive_mode_indices: tuple[int, ...],
    target_representation: str,
) -> dict[str, float]:
    if target_representation == "coefficients":
        shape = (
            len(prediction),
            len(DEFAULT_BASIS_TIME_CONSTANTS_S),
            3,
            len(adaptive_mode_indices),
        )
        predicted_coefficients = prediction.reshape(shape)
        fitted_coefficients = target_coefficients.reshape(shape)
    else:
        response_shape = (
            len(prediction),
            len(DEFAULT_RESPONSE_SNAPSHOT_TIMES_S),
            3,
            len(adaptive_mode_indices),
        )
        predicted_coefficients = fit_coefficients_from_step_response(
            prediction.reshape(response_shape),
            DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
            float(np.mean(servo_slopes)),
        )
        fitted_coefficients = fit_coefficients_from_step_response(
            target_coefficients.reshape(response_shape),
            DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
            float(np.mean(servo_slopes)),
        )
    predicted_response = reconstruct_composite_step_response(
        predicted_coefficients
    )
    fitted_response = reconstruct_composite_step_response(fitted_coefficients)
    true_response = true_servo_mode_step_response(
        true_targets, mode_transform, servo_slopes
    )[..., list(adaptive_mode_indices)]
    nominal_response = reconstruct_composite_step_response(
        nominal_coefficients[None]
    )[..., list(adaptive_mode_indices)].expand_as(true_response)
    scale = true_response.square().mean().sqrt().clamp_min(1e-12)
    return {
        "predicted_response_normalized_rmse": float(
            (predicted_response - true_response).square().mean().sqrt() / scale
        ),
        "nominal_response_normalized_rmse": float(
            (nominal_response - true_response).square().mean().sqrt() / scale
        ),
        "fixed_basis_oracle_normalized_rmse": float(
            (fitted_response - true_response).square().mean().sqrt() / scale
        ),
    }


def _supervised_loss(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    target_representation: str,
    adaptive_mode_count: int,
    servo_command_slope: float,
    direct_weight: float,
    projected_weight: float,
    loss_function: nn.Module,
) -> torch.Tensor:
    loss = normalized_prediction.new_zeros(())
    if direct_weight:
        loss = loss + direct_weight * loss_function(
            normalized_prediction, normalized_target
        )
    if projected_weight:
        if target_representation != "step_response":
            raise ValueError(
                "projected response loss requires step_response targets"
            )
        shape = (
            len(normalized_prediction),
            len(DEFAULT_RESPONSE_SNAPSHOT_TIMES_S),
            3,
            adaptive_mode_count,
        )
        physical_prediction = (
            normalized_prediction * label_std + label_mean
        ).reshape(shape)
        coefficients = fit_coefficients_from_step_response(
            physical_prediction,
            DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
            servo_command_slope,
        )
        reconstructed = reconstruct_composite_step_response(
            coefficients,
            response_times_s=DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
            servo_command_slope=servo_command_slope,
        ).reshape_as(normalized_prediction).to(normalized_prediction.dtype)
        normalized_reconstructed = (reconstructed - label_mean) / label_std
        loss = loss + projected_weight * loss_function(
            normalized_reconstructed, normalized_target
        )
    return loss


def train(args: argparse.Namespace) -> Mapping[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment_config(args.experiment_config)
    nominal_effectiveness, command_slopes, _ = _nominal_actuator_model(
        experiment.simulator_config
    )
    mode_transform = servo_mode_transform(
        nominal_effectiveness, command_slopes[2:]
    )
    adaptive_mode_indices = tuple(
        int(value) for value in args.adaptive_mode_indices.split(",")
    )
    if (
        not adaptive_mode_indices
        or len(set(adaptive_mode_indices)) != len(adaptive_mode_indices)
        or min(adaptive_mode_indices) < 0
        or max(adaptive_mode_indices) > 2
    ):
        raise ValueError("adaptive-mode-indices must be unique values from 0,1,2")
    target_names = (
        composite_target_names(DEFAULT_BASIS_TIME_CONSTANTS_S, adaptive_mode_indices)
        if args.target_representation == "coefficients"
        else composite_response_target_names(
            DEFAULT_RESPONSE_SNAPSHOT_TIMES_S, adaptive_mode_indices
        )
    )
    splits = {
        name: load_repeated_split(dataset, name, args.downsample)
        for name in ("train", "validation", "test")
    }
    for name, split in splits.items():
        _retain_converged_trials(split)
        if args.target_representation == "coefficients":
            target_value = fit_composite_coefficients(
                split["targets"], mode_transform, command_slopes[2:]
            )[..., list(adaptive_mode_indices)]
        else:
            target_value = true_servo_mode_step_response(
                split["targets"],
                mode_transform,
                command_slopes[2:],
                DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
            )[..., list(adaptive_mode_indices)]
        split["labels"] = target_value.reshape(len(target_value), -1).to(torch.float32)
        split["label_names"] = target_names

    materialized = load_and_materialize(
        experiment.simulator_config, 1, torch.device("cpu"), torch.float64
    )
    nominal_target = _sim2real_lqr_targets(materialized.parameters)
    nominal_coefficients = fit_composite_coefficients(
        nominal_target, mode_transform, command_slopes[2:]
    )[0]
    normalization = fit_repeated_normalization(splits["train"])
    histories = {
        name: prepare_histories(split, normalization)
        for name, split in splits.items()
    }
    normalized_labels = {
        name: (split["labels"] - normalization["label_mean"])
        / normalization["label_std"]
        for name, split in splits.items()
    }
    device = torch.device(args.device)
    trial_hidden = tuple(int(value) for value in args.trial_hidden_sizes.split(","))
    head_hidden = tuple(int(value) for value in args.head_hidden_sizes.split(","))
    temporal_channels = tuple(
        int(value) for value in args.temporal_channels.split(",")
    )
    model = build_offline_identifier(
        args.architecture,
        histories["train"].shape[2],
        histories["train"].shape[3],
        len(target_names),
        trial_hidden,
        head_hidden,
        temporal_channels,
        args.recurrent_hidden_size,
        args.recurrent_layers,
        args.trial_embedding_size,
        args.temporal_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(1, args.patience // 3)
    )
    train_loader = DataLoader(
        TensorDataset(
            histories["train"],
            normalized_labels["train"],
            splits["train"]["trial_mask"],
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    validation_loader = DataLoader(
        TensorDataset(
            histories["validation"],
            normalized_labels["validation"],
            splits["validation"]["trial_mask"],
        ),
        batch_size=args.batch_size,
    )
    loss_function = nn.HuberLoss(delta=1.0)
    if (
        args.direct_response_loss_weight < 0.0
        or args.projected_response_loss_weight < 0.0
        or args.direct_response_loss_weight + args.projected_response_loss_weight
        <= 0.0
    ):
        raise ValueError("response loss weights must be nonnegative with a positive sum")
    label_mean_device = normalization["label_mean"].to(device)
    label_std_device = normalization["label_std"].to(device)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        for batch_histories, batch_labels, available in train_loader:
            batch_histories = batch_histories.to(device)
            batch_labels = batch_labels.to(device)
            available = available.to(device)
            trial_mask = _random_trial_mask(available)
            optimizer.zero_grad(set_to_none=True)
            loss = _supervised_loss(
                model(batch_histories, trial_mask),
                batch_labels,
                label_mean_device,
                label_std_device,
                args.target_representation,
                len(adaptive_mode_indices),
                float(np.mean(command_slopes[2:])),
                args.direct_response_loss_weight,
                args.projected_response_loss_weight,
                loss_function,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_sum += float(loss.detach()) * len(batch_histories)
        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for batch_histories, batch_labels, available in validation_loader:
                batch_histories = batch_histories.to(device)
                batch_labels = batch_labels.to(device)
                available = available.to(device)
                validation_sum += float(
                    _supervised_loss(
                        model(batch_histories, available),
                        batch_labels,
                        label_mean_device,
                        label_std_device,
                        args.target_representation,
                        len(adaptive_mode_indices),
                        float(np.mean(command_slopes[2:])),
                        args.direct_response_loss_weight,
                        args.projected_response_loss_weight,
                        loss_function,
                    )
                ) * len(batch_histories)
        train_loss = train_sum / len(histories["train"])
        validation_loss = validation_sum / len(histories["validation"])
        scheduler.step(validation_loss)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"epoch={epoch} train={train_loss:.6f} "
                f"validation={validation_loss:.6f}"
            )
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("composite training produced no checkpoint")
    model.load_state_dict(best_state)
    trial_counts = (1, 2, 4, 8)
    metrics = {}
    for split_name, split in splits.items():
        metrics[split_name] = {}
        for trial_count in trial_counts:
            normalized_prediction = _predict(
                model,
                histories[split_name],
                split["trial_mask"],
                device,
                args.batch_size,
                trial_count,
            )
            prediction = (
                normalized_prediction * normalization["label_std"]
                + normalization["label_mean"]
            )
            metrics[split_name][str(trial_count)] = {
                **_coefficient_metrics(prediction, split["labels"], target_names),
                **_response_metrics(
                    prediction,
                    split["labels"],
                    split["targets"],
                    nominal_coefficients,
                    mode_transform,
                    command_slopes[2:],
                    adaptive_mode_indices,
                    args.target_representation,
                ),
            }
    checkpoint = {
        "schema_version": 1,
        "artifact_type": "sim2real_composite_servo_identifier",
        "model_state": best_state,
        "normalization": normalization,
        "history_steps": histories["train"].shape[2],
        "feature_count": histories["train"].shape[3],
        "raw_feature_count": splits["train"]["features"].shape[3],
        "trials_per_group": histories["train"].shape[1],
        "downsample": args.downsample,
        "trial_hidden_sizes": trial_hidden,
        "head_hidden_sizes": head_hidden,
        "architecture": args.architecture,
        "temporal_channels": temporal_channels,
        "recurrent_hidden_size": args.recurrent_hidden_size,
        "recurrent_layers": args.recurrent_layers,
        "trial_embedding_size": args.trial_embedding_size,
        "temporal_dropout": args.temporal_dropout,
        "direct_response_loss_weight": args.direct_response_loss_weight,
        "projected_response_loss_weight": args.projected_response_loss_weight,
        "label_names": target_names,
        "label_min": splits["train"]["labels"].amin(dim=0),
        "label_max": splits["train"]["labels"].amax(dim=0),
        "feature_names": splits["train"]["feature_names"],
        "basis_time_constants_s": DEFAULT_BASIS_TIME_CONSTANTS_S,
        "adaptive_mode_indices": adaptive_mode_indices,
        "target_representation": args.target_representation,
        "response_snapshot_times_s": DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
        "mode_transform": torch.from_numpy(mode_transform),
        "nominal_composite_coefficients": nominal_coefficients,
        "servo_command_slopes": torch.from_numpy(command_slopes[2:].copy()),
        "simulator_config": str(experiment.simulator_config),
        "controller_config": str(experiment.controller_config),
    }
    torch.save(checkpoint, output / "identifier.pt")
    report = {
        "schema_version": 1,
        "dataset": str(dataset),
        "device": str(device),
        "seed": args.seed,
        "target_mode": "fixed_filter_servo_composite",
        "architecture": args.architecture,
        "direct_response_loss_weight": args.direct_response_loss_weight,
        "projected_response_loss_weight": args.projected_response_loss_weight,
        "target_representation": args.target_representation,
        "response_snapshot_times_s": (
            DEFAULT_RESPONSE_SNAPSHOT_TIMES_S
            if args.target_representation == "step_response"
            else None
        ),
        "basis_time_constants_s": DEFAULT_BASIS_TIME_CONSTANTS_S,
        "adaptive_mode_indices": adaptive_mode_indices,
        "mode_transform": mode_transform.tolist(),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "group_counts": {name: len(value["labels"]) for name, value in splits.items()},
        "metrics": metrics,
        "history": history,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "parameter_count": report["parameter_count"],
                "test_by_trial_count": {
                    count: {
                        "mean_r2": value["mean_r2"],
                        "response_nrmse": value[
                            "predicted_response_normalized_rmse"
                        ],
                        "nominal_response_nrmse": value[
                            "nominal_response_normalized_rmse"
                        ],
                    }
                    for count, value in metrics["test"].items()
                },
            },
            indent=2,
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fixed-filter servo composite identification"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument(
        "--architecture", choices=("mlp", "tcn", "bigru"), default="mlp"
    )
    parser.add_argument(
        "--adaptive-mode-indices",
        default="1,2",
        help="zero-based servo SVD modes to identify; mode 0 is nominal yaw/common",
    )
    parser.add_argument(
        "--target-representation",
        choices=("step_response", "coefficients"),
        default="step_response",
    )
    parser.add_argument("--trial-hidden-sizes", default="512,256,128")
    parser.add_argument("--head-hidden-sizes", default="256,128")
    parser.add_argument("--temporal-channels", default="64,96,128")
    parser.add_argument("--recurrent-hidden-size", type=int, default=96)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument("--trial-embedding-size", type=int, default=128)
    parser.add_argument("--temporal-dropout", type=float, default=0.05)
    parser.add_argument("--direct-response-loss-weight", type=float, default=1.0)
    parser.add_argument("--projected-response-loss-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=20)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
