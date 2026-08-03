from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


EFFECTIVE_LABEL_NAMES = (
    "log10.effectiveness.roll_servo_1_scale",
    "log10.effectiveness.roll_servo_2_scale",
    "log10.effectiveness.roll_servo_3_scale",
    "log10.effectiveness.pitch_servo_2_scale",
    "log10.effectiveness.pitch_servo_3_scale",
    "log10.effectiveness.yaw_motor_scale",
    "log10.effectiveness.yaw_servo_1_scale",
    "log10.effectiveness.yaw_servo_2_scale",
    "log10.effectiveness.yaw_servo_3_scale",
    "log10.motor_tau_upper_scale",
    "log10.motor_tau_lower_scale",
    "log10.servo_1_tau_scale",
    "log10.servo_2_tau_scale",
    "log10.servo_3_tau_scale",
)

LQR_LATENT_LABEL_NAMES = (
    "log10.effectiveness.roll_servo_1_scale",
    "log10.effectiveness.roll_servo_2_scale",
    "log10.effectiveness.roll_servo_3_scale",
    "log10.inertia_x_over_y_scale",
    "log10.inertia_x_over_z_scale",
    "log10.effectiveness.yaw_motor_scale",
    "log10.motor_tau_upper_scale",
    "log10.motor_tau_lower_scale",
    "log10.servo_1_tau_scale",
    "log10.servo_2_tau_scale",
    "log10.servo_3_tau_scale",
)


class HistoryMLP(nn.Module):
    def __init__(
        self,
        history_steps: int,
        feature_count: int,
        output_count: int,
        hidden_sizes: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        sizes = [history_steps * feature_count, *hidden_sizes, output_count]
        layers: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(sizes) - 2:
                layers.append(nn.SiLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.network(history.flatten(start_dim=1))


class _TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        groups = max(1, min(8, channels // 8))
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.network(value))


class TemporalConvIdentifier(nn.Module):
    def __init__(
        self,
        feature_count: int,
        output_count: int,
        channels: int,
        hidden_sizes: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(feature_count, channels, kernel_size=7, padding=3),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *(_TemporalResidualBlock(channels, dilation) for dilation in (1, 2, 4, 8))
        )
        sizes = [channels * 4, *hidden_sizes, output_count]
        head: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
            head.append(nn.Linear(input_size, output_size))
            if index < len(sizes) - 2:
                head.append(nn.SiLU())
                if dropout > 0.0:
                    head.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*head)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(self.stem(history.transpose(1, 2)))
        summary = torch.cat(
            (
                encoded.mean(dim=2),
                encoded.amax(dim=2),
                encoded[:, :, 0],
                encoded[:, :, -1],
            ),
            dim=1,
        )
        return self.head(summary)


def create_identifier_model(
    architecture: str,
    history_steps: int,
    feature_count: int,
    output_count: int,
    hidden_sizes: Sequence[int],
    dropout: float,
    tcn_channels: int,
) -> nn.Module:
    if architecture == "mlp":
        return HistoryMLP(
            history_steps,
            feature_count,
            output_count,
            hidden_sizes,
            dropout,
        )
    if architecture == "tcn":
        return TemporalConvIdentifier(
            feature_count,
            output_count,
            tcn_channels,
            hidden_sizes,
            dropout,
        )
    raise ValueError(f"unsupported identifier architecture: {architecture}")


def _iter_shards(dataset_directory: Path, split: str) -> Iterable[Mapping[str, Any]]:
    paths = sorted((dataset_directory / split).glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no dataset shards found for split {split!r}")
    for path in paths:
        yield torch.load(path, map_location="cpu", weights_only=False)


def load_split(
    dataset_directory: Path,
    split: str,
    downsample: int,
    minimum_start_step: int | None,
    maximum_start_step: int | None,
    minimum_information: float | None,
) -> dict[str, Any]:
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    accumulated: dict[str, list[torch.Tensor]] = defaultdict(list)
    feature_names: tuple[str, ...] | None = None
    label_names: tuple[str, ...] | None = None
    for shard in _iter_shards(dataset_directory, split):
        mask = torch.ones(len(shard["labels"]), dtype=torch.bool)
        if minimum_start_step is not None:
            mask &= shard["window_start_step"] >= minimum_start_step
        if maximum_start_step is not None:
            mask &= shard["window_start_step"] <= maximum_start_step
        if minimum_information is not None:
            mask &= shard["information_score"] >= minimum_information
        if not bool(mask.any()):
            continue
        accumulated["features"].append(
            shard["features"][mask, ::downsample].to(torch.float32)
        )
        for name in (
            "labels",
            "group_id",
            "episode_id",
            "window_start_step",
            "information_score",
        ):
            accumulated[name].append(shard[name][mask])
        current_features = tuple(shard["feature_names"])
        current_labels = tuple(shard["label_names"])
        if feature_names is not None and current_features != feature_names:
            raise ValueError("feature schema differs between shards")
        if label_names is not None and current_labels != label_names:
            raise ValueError("label schema differs between shards")
        feature_names, label_names = current_features, current_labels
    if not accumulated:
        raise ValueError(f"all samples were filtered from split {split!r}")
    return {
        name: torch.cat(values, dim=0) for name, values in accumulated.items()
    } | {
        "feature_names": feature_names,
        "label_names": label_names,
    }


def fit_normalization(train: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    features = train["features"]
    labels = train["labels"]
    feature_mean = features.mean(dim=(0, 1))
    feature_std = features.std(dim=(0, 1)).clamp_min(1e-6)
    label_mean = labels.mean(dim=0)
    label_std = labels.std(dim=0).clamp_min(1e-6)
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "label_mean": label_mean,
        "label_std": label_std,
    }


def effective_lqr_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map physical scales to the local angular-acceleration model LQR uses.

    The transform is exact for the multiplicative randomization around hover:
    servo authority scales as thrust * grid gain / axis inertia, while motor
    yaw authority scales as reaction torque / yaw inertia.
    """

    inertia_x, inertia_y, inertia_z = labels[:, 0], labels[:, 1], labels[:, 2]
    collective, reaction = labels[:, 3], labels[:, 4]
    grid_1, grid_2, grid_3 = labels[:, 7], labels[:, 8], labels[:, 9]
    return torch.stack(
        (
            collective + grid_1 - inertia_x,
            collective + grid_2 - inertia_x,
            collective + grid_3 - inertia_x,
            collective + grid_2 - inertia_y,
            collective + grid_3 - inertia_y,
            reaction - inertia_z,
            collective + grid_1 - inertia_z,
            collective + grid_2 - inertia_z,
            collective + grid_3 - inertia_z,
            labels[:, 5],
            labels[:, 6],
            labels[:, 10],
            labels[:, 11],
            labels[:, 12],
        ),
        dim=1,
    )


def lqr_latent_labels(labels: torch.Tensor) -> torch.Tensor:
    inertia_x, inertia_y, inertia_z = labels[:, 0], labels[:, 1], labels[:, 2]
    collective, reaction = labels[:, 3], labels[:, 4]
    return torch.stack(
        (
            collective + labels[:, 7] - inertia_x,
            collective + labels[:, 8] - inertia_x,
            collective + labels[:, 9] - inertia_x,
            inertia_x - inertia_y,
            inertia_x - inertia_z,
            reaction - inertia_z,
            labels[:, 5],
            labels[:, 6],
            labels[:, 10],
            labels[:, 11],
            labels[:, 12],
        ),
        dim=1,
    )


def lqr_latent_to_effective(labels: torch.Tensor) -> torch.Tensor:
    roll_1, roll_2, roll_3 = labels[:, 0], labels[:, 1], labels[:, 2]
    x_over_y, x_over_z, yaw_motor = labels[:, 3], labels[:, 4], labels[:, 5]
    return torch.stack(
        (
            roll_1,
            roll_2,
            roll_3,
            roll_2 + x_over_y,
            roll_3 + x_over_y,
            yaw_motor,
            roll_1 + x_over_z,
            roll_2 + x_over_z,
            roll_3 + x_over_z,
            labels[:, 6],
            labels[:, 7],
            labels[:, 8],
            labels[:, 9],
            labels[:, 10],
        ),
        dim=1,
    )


def apply_target_mode(split: dict[str, Any], target_mode: str) -> None:
    if target_mode == "physical":
        return
    if target_mode == "lqr_effective":
        split["labels"] = effective_lqr_labels(split["labels"])
        split["label_names"] = EFFECTIVE_LABEL_NAMES
        return
    if target_mode == "lqr_latent":
        split["labels"] = lqr_latent_labels(split["labels"])
        split["label_names"] = LQR_LATENT_LABEL_NAMES
        return
    raise ValueError(f"unsupported target mode: {target_mode}")


def _normalized_tensors(
    split: Mapping[str, torch.Tensor], normalization: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    features = (split["features"] - normalization["feature_mean"]) / normalization[
        "feature_std"
    ]
    labels = (split["labels"] - normalization["label_mean"]) / normalization[
        "label_std"
    ]
    return features, labels


def _group_balanced_weights(group_ids: torch.Tensor) -> torch.Tensor:
    _, inverse, counts = torch.unique(
        group_ids, sorted=True, return_inverse=True, return_counts=True
    )
    return counts[inverse].to(torch.float64).reciprocal()


def _predict(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size):
            outputs.append(model(batch.to(device)).cpu())
    return torch.cat(outputs, dim=0)


def _aggregate_by_group(
    predictions: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    unique, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    prediction_sum = torch.zeros(len(unique), predictions.shape[1])
    label_sum = torch.zeros_like(prediction_sum)
    counts = torch.zeros(len(unique), 1)
    prediction_sum.index_add_(0, inverse, predictions)
    label_sum.index_add_(0, inverse, labels)
    counts.index_add_(0, inverse, torch.ones(len(inverse), 1))
    return prediction_sum / counts, label_sum / counts


def regression_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    label_names: Sequence[str],
) -> dict[str, Any]:
    error = predictions - labels
    squared = error.square()
    denominator = ((labels - labels.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1.0 - squared.sum(dim=0) / denominator.clamp_min(1e-12)
    absolute = error.abs()
    metrics = []
    for index, name in enumerate(label_names):
        metrics.append(
            {
                "name": name,
                "rmse_log10": float(squared[:, index].mean().sqrt()),
                "mae_log10": float(absolute[:, index].mean()),
                "r2": float(r2[index]),
                "median_factor_error": float(
                    torch.pow(10.0, absolute[:, index].median())
                ),
                "p90_factor_error": float(
                    torch.pow(10.0, torch.quantile(absolute[:, index], 0.9))
                ),
            }
        )
    return {
        "mean_rmse_log10": float(squared.mean(dim=0).sqrt().mean()),
        "mean_r2": float(r2.mean()),
        "median_r2": float(r2.median()),
        "parameters": metrics,
    }


def _evaluate_split(
    model: nn.Module,
    split: Mapping[str, Any],
    normalization: Mapping[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    normalized_features, _ = _normalized_tensors(split, normalization)
    normalized_prediction = _predict(model, normalized_features, device, batch_size)
    prediction = (
        normalized_prediction * normalization["label_std"]
        + normalization["label_mean"]
    )
    window_metrics = regression_metrics(
        prediction, split["labels"], split["label_names"]
    )
    group_prediction, group_labels = _aggregate_by_group(
        prediction, split["labels"], split["group_id"]
    )
    group_metrics = regression_metrics(
        group_prediction, group_labels, split["label_names"]
    )
    by_start_step = {}
    for start_step in torch.unique(split["window_start_step"], sorted=True):
        mask = split["window_start_step"] == start_step
        start_prediction, start_labels = _aggregate_by_group(
            prediction[mask], split["labels"][mask], split["group_id"][mask]
        )
        by_start_step[str(int(start_step))] = regression_metrics(
            start_prediction, start_labels, split["label_names"]
        )
    return {
        "window": window_metrics,
        "group": group_metrics,
        "group_by_window_start_step": by_start_step,
    }, prediction


def train_identifier(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_directory = Path(args.dataset).expanduser().resolve()
    output_directory = Path(args.output).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_start_step = (
        None if args.maximum_start_s is None else round(args.maximum_start_s * 500)
    )
    minimum_start_step = (
        None if args.minimum_start_s is None else round(args.minimum_start_s * 500)
    )
    load_arguments = {
        "downsample": args.downsample,
        "minimum_start_step": minimum_start_step,
        "maximum_start_step": maximum_start_step,
        "minimum_information": args.minimum_information,
    }
    splits = {
        name: load_split(dataset_directory, name, **load_arguments)
        for name in ("train", "validation", "test")
    }
    if args.maximum_train_groups is not None:
        train_groups = torch.unique(splits["train"]["group_id"], sorted=True)
        selected = train_groups[: args.maximum_train_groups]
        mask = torch.isin(splits["train"]["group_id"], selected)
        for name, value in tuple(splits["train"].items()):
            if isinstance(value, torch.Tensor) and value.shape[:1] == mask.shape:
                splits["train"][name] = value[mask]
    for split in splits.values():
        apply_target_mode(split, args.target_mode)
    normalization = fit_normalization(splits["train"])
    train_features, train_labels = _normalized_tensors(
        splits["train"], normalization
    )
    validation_features, validation_labels = _normalized_tensors(
        splits["validation"], normalization
    )
    device = torch.device(args.device)
    hidden_sizes = tuple(int(value) for value in args.hidden_sizes.split(",") if value)
    model = create_identifier_model(
        args.architecture,
        train_features.shape[1],
        train_features.shape[2],
        train_labels.shape[1],
        hidden_sizes,
        args.dropout,
        args.tcn_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(1, args.patience // 3)
    )
    loss_function = nn.HuberLoss(delta=args.huber_delta)
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
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), labels)
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_sum += loss.detach().item() * len(features)
            train_count += len(features)
        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)
                loss = loss_function(model(features), labels)
                validation_sum += loss.item() * len(features)
                validation_count += len(features)
        train_loss = train_sum / train_count
        validation_loss = validation_sum / validation_count
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss - args.minimum_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"epoch={epoch} train={train_loss:.6f} "
                f"validation={validation_loss:.6f}"
            )
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics = {}
    for name, split in splits.items():
        metrics[name], _ = _evaluate_split(
            model, split, normalization, device, args.batch_size
        )
    baseline = normalization["label_mean"].expand_as(splits["test"]["labels"])
    baseline_group, baseline_labels = _aggregate_by_group(
        baseline, splits["test"]["labels"], splits["test"]["group_id"]
    )
    metrics["test_train_mean_baseline"] = {
        "window": regression_metrics(
            baseline, splits["test"]["labels"], splits["test"]["label_names"]
        ),
        "group": regression_metrics(
            baseline_group, baseline_labels, splits["test"]["label_names"]
        ),
    }
    report = {
        "schema_version": 1,
        "dataset": str(dataset_directory),
        "device": str(device),
        "seed": args.seed,
        "downsample": args.downsample,
        "effective_sample_hz": 500 / args.downsample,
        "maximum_start_s": args.maximum_start_s,
        "minimum_start_s": args.minimum_start_s,
        "minimum_information": args.minimum_information,
        "maximum_train_groups": args.maximum_train_groups,
        "target_mode": args.target_mode,
        "hidden_sizes": list(hidden_sizes),
        "architecture": args.architecture,
        "tcn_channels": args.tcn_channels,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "sample_counts": {name: len(split["labels"]) for name, split in splits.items()},
        "group_counts": {
            name: int(split["group_id"].unique().numel())
            for name, split in splits.items()
        },
        "history": history,
        "metrics": metrics,
    }
    torch.save(
        {
            "schema_version": 1,
            "model_state": best_state,
            "normalization": normalization,
            "history_steps": train_features.shape[1],
            "feature_count": train_features.shape[2],
            "hidden_sizes": hidden_sizes,
            "architecture": args.architecture,
            "tcn_channels": args.tcn_channels,
            "dropout": args.dropout,
            "feature_names": splits["train"]["feature_names"],
            "label_names": splits["train"]["label_names"],
            "downsample": args.downsample,
            "target_mode": args.target_mode,
        },
        output_directory / "identifier.pt",
    )
    (output_directory / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("best_epoch", "parameter_count", "sample_counts", "group_counts")}, indent=2))
    print(json.dumps(report["metrics"]["test"]["group"], indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a flattened-history MLP identifier")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--minimum-start-s", type=float, default=0.0)
    parser.add_argument("--maximum-start-s", type=float, default=1.0)
    parser.add_argument("--minimum-information", type=float)
    parser.add_argument("--maximum-train-groups", type=int)
    parser.add_argument(
        "--target-mode",
        choices=("physical", "lqr_effective", "lqr_latent"),
        default="physical",
    )
    parser.add_argument("--hidden-sizes", default="512,256,128")
    parser.add_argument("--architecture", choices=("mlp", "tcn"), default="mlp")
    parser.add_argument("--tcn-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--minimum-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    train_identifier(build_parser().parse_args())


if __name__ == "__main__":
    main()
