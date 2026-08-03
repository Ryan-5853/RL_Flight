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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .control_evaluation import (
    _discrete_model,
    _lqr_gain,
    _nominal_actuator_model,
    _scaled_actuator_model,
)
from .training import (
    _group_balanced_weights,
    _predict,
    create_identifier_model,
    effective_lqr_labels,
    fit_normalization,
    load_split,
)


def _lqr_weights(controller_config: Path) -> tuple[np.ndarray, np.ndarray]:
    from flight_controller import load_controller_config

    config = load_controller_config(controller_config)["params"]["lqr"]
    state_scales = np.asarray(config["state_scales"], dtype=np.float64)
    input_scales = np.asarray(config["input_scales"], dtype=np.float64)
    return (
        np.diag(1.0 / state_scales**2),
        float(config["input_weight_scale"]) * np.diag(1.0 / input_scales**2),
    )


def oracle_gains(
    split: Mapping[str, Any],
    nominal_effectiveness: np.ndarray,
    command_slopes: np.ndarray,
    nominal_tau: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> torch.Tensor:
    group_ids = split["group_id"]
    unique_groups, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    first_indices = torch.empty(len(unique_groups), dtype=torch.int64)
    for index, group_id in enumerate(unique_groups):
        first_indices[index] = torch.nonzero(group_ids == group_id)[0, 0]
    effective = effective_lqr_labels(split["labels"][first_indices]).numpy()
    group_gains = []
    for label in effective:
        effectiveness, tau = _scaled_actuator_model(
            label, nominal_effectiveness, nominal_tau
        )
        a, b = _discrete_model(
            effectiveness, command_slopes, tau, 1.0 / 500.0
        )
        group_gains.append(_lqr_gain(a, b, q, r).reshape(-1))
    return torch.from_numpy(np.stack(group_gains)).to(torch.float32)[inverse]


def _group_first(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    unique_groups = torch.unique(group_ids, sorted=True)
    return torch.stack(
        [values[torch.nonzero(group_ids == group_id)[0, 0]] for group_id in unique_groups]
    )


def _gain_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    relative = torch.linalg.vector_norm(predicted - target, dim=1) / torch.linalg.vector_norm(
        target, dim=1
    ).clamp_min(1e-8)
    return {
        "mean_relative_error": float(relative.mean()),
        "median_relative_error": float(relative.median()),
        "p90_relative_error": float(torch.quantile(relative, 0.9)),
        "rmse": float(torch.sqrt((predicted - target).square().mean())),
    }


def train_gain_identifier(args: argparse.Namespace) -> Mapping[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = {
        name: load_split(dataset, name, args.downsample, 0, 0, None)
        for name in ("train", "validation", "test")
    }
    nominal_effectiveness, command_slopes, nominal_tau = _nominal_actuator_model(
        Path(args.simulator_config).expanduser().resolve()
    )
    q, r = _lqr_weights(Path(args.controller_config).expanduser().resolve())
    gains = {
        name: oracle_gains(
            split,
            nominal_effectiveness,
            command_slopes,
            nominal_tau,
            q,
            r,
        )
        for name, split in splits.items()
    }
    train_group_gains = _group_first(gains["train"], splits["train"]["group_id"])
    gain_mean = train_group_gains.mean(dim=0)
    _, singular_values, right_vectors = torch.linalg.svd(
        train_group_gains - gain_mean, full_matrices=False
    )
    components = right_vectors[: args.pca_components]
    explained = torch.cumsum(singular_values.square(), dim=0) / singular_values.square().sum()
    for name, split in splits.items():
        split["labels"] = (gains[name] - gain_mean) @ components.T
        split["label_names"] = tuple(
            f"lqr_gain_pca_{index}" for index in range(args.pca_components)
        )
    normalization = fit_normalization(splits["train"])
    train_features = (
        splits["train"]["features"] - normalization["feature_mean"]
    ) / normalization["feature_std"]
    train_labels = (
        splits["train"]["labels"] - normalization["label_mean"]
    ) / normalization["label_std"]
    validation_features = (
        splits["validation"]["features"] - normalization["feature_mean"]
    ) / normalization["feature_std"]
    validation_labels = (
        splits["validation"]["labels"] - normalization["label_mean"]
    ) / normalization["label_std"]
    hidden_sizes = tuple(int(value) for value in args.hidden_sizes.split(","))
    device = torch.device(args.device)
    model = create_identifier_model(
        "mlp",
        train_features.shape[1],
        train_features.shape[2],
        args.pca_components,
        hidden_sizes,
        0.0,
        128,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(1, args.patience // 3)
    )
    sampler = WeightedRandomSampler(
        _group_balanced_weights(splits["train"]["group_id"]),
        len(train_features),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        sampler=sampler,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_labels),
        batch_size=args.batch_size,
    )
    loss_function = nn.HuberLoss(delta=1.0)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_sum += float(loss.detach()) * len(features)
        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)
                validation_sum += float(loss_function(model(features), labels)) * len(features)
        train_loss = train_sum / len(train_features)
        validation_loss = validation_sum / len(validation_features)
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
            print(f"epoch={epoch} train={train_loss:.6f} validation={validation_loss:.6f}")
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("gain training did not produce a checkpoint")
    model.load_state_dict(best_state)
    metrics = {}
    for name, split in splits.items():
        normalized_prediction = _predict(
            model,
            (split["features"] - normalization["feature_mean"])
            / normalization["feature_std"],
            device,
            args.batch_size,
        )
        latent = (
            normalized_prediction * normalization["label_std"]
            + normalization["label_mean"]
        )
        reconstructed = gain_mean + latent @ components
        group_reconstructed = _group_first(reconstructed, split["group_id"])
        group_target = _group_first(gains[name], split["group_id"])
        metrics[name] = {
            "window": _gain_metrics(reconstructed, gains[name]),
            "group_first_window": _gain_metrics(group_reconstructed, group_target),
        }
    pca_ceiling = gain_mean + (train_group_gains - gain_mean) @ components.T @ components
    report = {
        "schema_version": 1,
        "target_mode": "lqr_gain_pca",
        "dataset": str(dataset),
        "device": str(device),
        "seed": args.seed,
        "pca_components": args.pca_components,
        "pca_explained_variance": float(explained[args.pca_components - 1]),
        "pca_train_reconstruction": _gain_metrics(pca_ceiling, train_group_gains),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "sample_counts": {name: len(split["features"]) for name, split in splits.items()},
        "group_counts": {
            name: int(split["group_id"].unique().numel()) for name, split in splits.items()
        },
        "metrics": metrics,
        "history": history,
    }
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "lqr_gain_pca_identifier",
            "target_mode": "lqr_gain_pca",
            "model_state": best_state,
            "normalization": normalization,
            "history_steps": train_features.shape[1],
            "feature_count": train_features.shape[2],
            "hidden_sizes": hidden_sizes,
            "architecture": "mlp",
            "downsample": args.downsample,
            "pca_components": components,
            "pca_gain_mean": gain_mean,
            "nominal_gain": torch.from_numpy(
                _lqr_gain(
                    *_discrete_model(
                        nominal_effectiveness, command_slopes, nominal_tau, 1.0 / 500.0
                    ),
                    q,
                    r,
                )
            ).to(torch.float32),
        },
        output / "identifier.pt",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("best_epoch", "parameter_count", "pca_explained_variance", "metrics")}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PCA LQR-gain identifier")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--hidden-sizes", default="512,256,128")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--simulator-config", default="SimEnv/configs/example.yaml")
    parser.add_argument(
        "--controller-config",
        default="Controller/configs/lqr_identification_nominal.yaml",
    )
    return parser


def main() -> None:
    train_gain_identifier(build_parser().parse_args())


if __name__ == "__main__":
    main()
