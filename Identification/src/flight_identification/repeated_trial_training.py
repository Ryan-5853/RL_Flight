from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .training import (
    effective_lqr_label_names,
    effective_lqr_labels,
    regression_metrics,
)
from .experiment import SIM2REAL_IDENTIFICATION_TARGET_NAMES


class TrialSetIdentifier(nn.Module):
    def __init__(
        self,
        history_steps: int,
        feature_count: int,
        output_count: int,
        trial_hidden_sizes: Sequence[int] = (512, 256, 128),
        head_hidden_sizes: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        trial_sizes = [history_steps * feature_count, *trial_hidden_sizes]
        trial_layers: list[nn.Module] = []
        for input_size, output_size in zip(trial_sizes, trial_sizes[1:]):
            trial_layers.extend((nn.Linear(input_size, output_size), nn.SiLU()))
        self.trial_encoder = nn.Sequential(*trial_layers)
        embedding_size = trial_hidden_sizes[-1]
        head_sizes = [3 * embedding_size, *head_hidden_sizes, output_count]
        head_layers: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(
            zip(head_sizes, head_sizes[1:])
        ):
            head_layers.append(nn.Linear(input_size, output_size))
            if index < len(head_sizes) - 2:
                head_layers.append(nn.SiLU())
        self.head = nn.Sequential(*head_layers)

    def forward(
        self, histories: torch.Tensor, trial_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if histories.ndim != 4:
            raise ValueError("histories must have shape [batch, trials, time, features]")
        batch_size, trial_count = histories.shape[:2]
        encoded = self.trial_encoder(histories.flatten(start_dim=2))
        if trial_mask is None:
            trial_mask = torch.ones(
                batch_size, trial_count, device=histories.device, dtype=torch.bool
            )
        if trial_mask.shape != (batch_size, trial_count):
            raise ValueError("trial_mask shape must match batch and trial dimensions")
        if not bool(trial_mask.any(dim=1).all().item()):
            raise ValueError("every sample must include at least one trial")
        weights = trial_mask.to(encoded.dtype)[..., None]
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (encoded * weights).sum(dim=1) / count
        centered = (encoded - mean[:, None]) * weights
        standard_deviation = torch.sqrt(
            centered.square().sum(dim=1) / count + 1e-8
        )
        maximum = encoded.masked_fill(~trial_mask[..., None], -torch.inf).amax(dim=1)
        return self.head(torch.cat((mean, maximum, standard_deviation), dim=1))


def load_repeated_split(
    dataset: Path, split: str, downsample: int
) -> dict[str, Any]:
    shards = sorted((dataset / split).glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no repeated-trial shards for split {split!r}")
    accumulated: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "features",
            "valid_mask",
            "labels",
            "group_id",
            "failure_step",
            "failure_code",
            "saturation_fraction",
        )
    }
    optional_targets: list[torch.Tensor] = []
    feature_names = None
    label_names = None
    parameterization = None
    for path in shards:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        accumulated["features"].append(
            shard["features"][:, :, ::downsample].to(torch.float32)
        )
        accumulated["valid_mask"].append(shard["valid_mask"][:, :, ::downsample])
        for name in accumulated.keys() - {"features", "valid_mask"}:
            accumulated[name].append(shard[name])
        if "targets" in shard:
            optional_targets.append(shard["targets"])
        current_features = tuple(shard["feature_names"])
        current_labels = tuple(shard["label_names"])
        current_parameterization = str(
            shard.get("parameterization", "legacy_collective")
        )
        if feature_names is not None and feature_names != current_features:
            raise ValueError("feature schema differs between repeated-trial shards")
        if label_names is not None and label_names != current_labels:
            raise ValueError("label schema differs between repeated-trial shards")
        if (
            parameterization is not None
            and parameterization != current_parameterization
        ):
            raise ValueError("parameterization differs between repeated-trial shards")
        feature_names, label_names = current_features, current_labels
        parameterization = current_parameterization
    result = {name: torch.cat(values) for name, values in accumulated.items()}
    result["feature_names"] = feature_names
    result["label_names"] = label_names
    result["parameterization"] = parameterization
    if optional_targets:
        if len(optional_targets) != len(shards):
            raise ValueError("targets must be present in every repeated-trial shard")
        result["targets"] = torch.cat(optional_targets)
        result["target_names"] = tuple(shard["target_names"])
    return result


def fit_repeated_normalization(split: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    features = split["features"]
    valid = split["valid_mask"][..., None]
    count = valid.sum(dim=(0, 1, 2)).clamp_min(1).to(features.dtype)
    feature_mean = (features * valid).sum(dim=(0, 1, 2)) / count
    variance = ((features - feature_mean).square() * valid).sum(
        dim=(0, 1, 2)
    ) / count
    labels = split["labels"]
    return {
        "feature_mean": feature_mean,
        "feature_std": torch.sqrt(variance).clamp_min(1e-6),
        "label_mean": labels.mean(dim=0),
        "label_std": labels.std(dim=0).clamp_min(1e-6),
    }


def prepare_histories(
    split: Mapping[str, Any], normalization: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    valid = split["valid_mask"]
    normalized = (
        split["features"] - normalization["feature_mean"]
    ) / normalization["feature_std"]
    normalized = torch.where(valid[..., None], normalized, torch.zeros_like(normalized))
    return torch.cat((normalized, valid[..., None].to(normalized.dtype)), dim=3)


def _random_trial_mask(available: torch.Tensor) -> torch.Tensor:
    if available.ndim != 2 or not bool(available.any(dim=1).all().item()):
        raise ValueError("every sample must have at least one available trial")
    available_count = available.sum(dim=1)
    counts = (
        torch.rand(len(available), device=available.device) * available_count
    ).floor().to(torch.int64) + 1
    scores = torch.rand(available.shape, device=available.device).masked_fill(
        ~available, torch.inf
    )
    order = scores.argsort(dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(
        1,
        order,
        torch.arange(available.shape[1], device=available.device)[None].expand_as(order),
    )
    return available & (rank < counts[:, None])


def _fixed_trial_mask(available: torch.Tensor, trial_count: int) -> torch.Tensor:
    return available & (available.to(torch.int64).cumsum(dim=1) <= trial_count)


@torch.no_grad()
def _predict(
    model: nn.Module,
    histories: torch.Tensor,
    available: torch.Tensor,
    device: torch.device,
    batch_size: int,
    trial_count: int,
) -> torch.Tensor:
    predictions = []
    model.eval()
    for start in range(0, len(histories), batch_size):
        value = histories[start : start + batch_size].to(device)
        mask = _fixed_trial_mask(
            available[start : start + batch_size].to(device), trial_count
        )
        predictions.append(model(value, mask).cpu())
    return torch.cat(predictions)


def _retain_converged_trials(split: dict[str, Any]) -> None:
    trial_mask = split["failure_code"] == 0
    keep = trial_mask.any(dim=1)
    if not bool(keep.any().item()):
        raise ValueError("split has no parameter group with a converged trial")
    group_count = len(keep)
    for name, value in tuple(split.items()):
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == group_count:
            split[name] = value[keep]
    split["trial_mask"] = trial_mask[keep]
    split["valid_mask"] &= split["trial_mask"][..., None]


def _sim2real_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    names: Sequence[str],
) -> dict[str, Any]:
    error = prediction - target
    squared = error.square()
    absolute = error.abs()
    target_mean = target.mean(dim=0)
    total = (target - target_mean).square().sum(dim=0)
    r2 = 1.0 - squared.sum(dim=0) / total.clamp_min(1e-12)
    target_std = target.std(dim=0).clamp_min(1e-12)
    per_target = []
    for index, name in enumerate(names):
        item = {
            "name": name,
            "rmse": float(squared[:, index].mean().sqrt()),
            "mae": float(absolute[:, index].mean()),
            "normalized_rmse": float(
                squared[:, index].mean().sqrt() / target_std[index]
            ),
            "r2": float(r2[index]),
        }
        if name.startswith("log10."):
            item["median_factor_error"] = float(
                torch.pow(10.0, absolute[:, index].median())
            )
            item["p90_factor_error"] = float(
                torch.pow(10.0, torch.quantile(absolute[:, index], 0.9))
            )
        per_target.append(item)
    return {
        "mean_normalized_rmse": float(
            (squared.mean(dim=0).sqrt() / target_std).mean()
        ),
        "mean_r2": float(r2.mean()),
        "median_r2": float(r2.median()),
        "parameters": per_target,
    }


def train_repeated_identifier(args: argparse.Namespace) -> Mapping[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = {
        name: load_repeated_split(dataset, name, args.downsample)
        for name in ("train", "validation", "test")
    }
    parameterization = str(splits["train"]["parameterization"])
    if any(split["parameterization"] != parameterization for split in splits.values()):
        raise ValueError("all repeated-trial splits must use one parameterization")
    if parameterization == "sim2real_micro":
        full_names = tuple(splits["train"]["target_names"])
        effective_names = SIM2REAL_IDENTIFICATION_TARGET_NAMES
        target_indices = [full_names.index(name) for name in effective_names]
        for split in splits.values():
            if tuple(split["target_names"]) != full_names:
                raise ValueError("sim2real target schema differs between splits")
            split["labels"] = split["targets"][:, target_indices]
            split["label_names"] = effective_names
            _retain_converged_trials(split)
    else:
        effective_names = effective_lqr_label_names(parameterization)
        for split in splits.values():
            split["labels"] = effective_lqr_labels(
                split["labels"], parameterization
            )
            split["label_names"] = effective_names
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
    model = TrialSetIdentifier(
        histories["train"].shape[2],
        histories["train"].shape[3],
        normalized_labels["train"].shape[1],
        trial_hidden,
        head_hidden,
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
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        for batch_histories, batch_labels, batch_available in train_loader:
            batch_histories = batch_histories.to(device)
            batch_labels = batch_labels.to(device)
            batch_available = batch_available.to(device)
            trial_mask = (
                _random_trial_mask(batch_available)
                if args.trial_sampling == "random_count"
                else batch_available
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_histories, trial_mask), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_sum += float(loss.detach()) * len(batch_histories)
        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for batch_histories, batch_labels, batch_available in validation_loader:
                batch_histories = batch_histories.to(device)
                batch_labels = batch_labels.to(device)
                batch_available = batch_available.to(device)
                validation_sum += float(
                    loss_function(
                        model(batch_histories, batch_available), batch_labels
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
        raise RuntimeError("repeated-trial training produced no checkpoint")
    model.load_state_dict(best_state)
    available_trials = histories["train"].shape[1]
    trial_counts = sorted(
        set(min(available_trials, value) for value in (1, 2, 4, 8, available_trials))
    )
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
            metric_function = (
                _sim2real_metrics
                if parameterization == "sim2real_micro"
                else regression_metrics
            )
            metrics[split_name][str(trial_count)] = metric_function(
                prediction, split["labels"], split["label_names"]
            )
    report = {
        "schema_version": 1,
        "dataset": str(dataset),
        "device": str(device),
        "seed": args.seed,
        "target_mode": "lqr_effective",
        "parameterization": parameterization,
        "trial_sampling": args.trial_sampling,
        "downsample": args.downsample,
        "effective_sample_hz": 500 / args.downsample,
        "trials_per_group": available_trials,
        "trial_hidden_sizes": trial_hidden,
        "head_hidden_sizes": head_hidden,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "group_counts": {name: len(value["labels"]) for name, value in splits.items()},
        "metrics": metrics,
        "history": history,
    }
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "repeated_trial_lqr_effective_identifier",
            "model_state": best_state,
            "normalization": normalization,
            "history_steps": histories["train"].shape[2],
            "feature_count": histories["train"].shape[3],
            "raw_feature_count": splits["train"]["features"].shape[3],
            "trials_per_group": available_trials,
            "trial_sampling": args.trial_sampling,
            "downsample": args.downsample,
            "trial_hidden_sizes": trial_hidden,
            "head_hidden_sizes": head_hidden,
            "label_names": effective_names,
            "label_min": splits["train"]["labels"].amin(dim=0),
            "label_max": splits["train"]["labels"].amax(dim=0),
            "parameterization": parameterization,
            "feature_names": splits["train"]["feature_names"],
        },
        output / "identifier.pt",
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "parameter_count": report["parameter_count"],
                "test_by_trial_count": {
                    count: value["mean_r2"]
                    for count, value in metrics["test"].items()
                },
            },
            indent=2,
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a permutation-invariant repeated-trial identifier"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument(
        "--trial-sampling",
        choices=("random_count", "all"),
        default="random_count",
        help="random_count supports variable trial counts; all specializes for the complete set",
    )
    parser.add_argument("--trial-hidden-sizes", default="512,256,128")
    parser.add_argument("--head-hidden-sizes", default="256,128")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=20)
    return parser


def main() -> None:
    train_repeated_identifier(build_parser().parse_args())


if __name__ == "__main__":
    main()
