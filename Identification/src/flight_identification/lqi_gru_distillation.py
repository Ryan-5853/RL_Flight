from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml


@dataclass(frozen=True)
class SequenceSet:
    observations: torch.Tensor
    actions: torch.Tensor
    valid_mask: torch.Tensor
    group_id: torch.Tensor
    labels_audit_only: torch.Tensor
    observation_names: tuple[str, ...]
    action_names: tuple[str, ...]
    label_names: tuple[str, ...]


class _SequenceDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, values: SequenceSet) -> None:
        self.values = values

    def __len__(self) -> int:
        return self.values.observations.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.values.observations[index],
            self.values.actions[index],
            self.values.valid_mask[index],
        )


class _WindowedSequenceDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Random causal window crops of full sequences for windowed architectures."""

    def __init__(
        self,
        values: SequenceSet,
        window_steps: int,
        generator: torch.Generator,
    ) -> None:
        self.values = values
        self.window_steps = window_steps
        self.length = values.observations.shape[1]
        self.count = values.observations.shape[0]
        if window_steps >= self.length:
            raise ValueError("window must be shorter than the sequence")
        self.generator = generator
        self.starts = torch.randint(
            0, self.length - window_steps + 1, (self.count,), generator=generator
        )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        start = int(self.starts[index])
        stop = start + self.window_steps
        return (
            self.values.observations[index, start:stop],
            self.values.actions[index, start:stop],
            self.values.valid_mask[index, start:stop],
        )


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        padded = F.pad(value, (self.padding, 0))
        out = self.dropout(self.activation(self.conv1(padded)))
        out = self.dropout(self.activation(self.conv2(F.pad(out, (self.padding, 0)))))
        return self.activation(out + residual)


class TCNEncoder(nn.Module):
    """Causal dilated temporal convolutional encoder with fixed receptive field."""

    def __init__(
        self,
        observation_size: int,
        channels: int,
        context_steps: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_steps = context_steps
        self.receptive_field = 1
        layers: list[nn.Module] = []
        dilation = 1
        while self.receptive_field < context_steps:
            layers.append(_TCNBlock(channels, kernel_size, dilation, dropout))
            self.receptive_field += (kernel_size - 1) * dilation
            dilation *= 2
        self.project = nn.Conv1d(observation_size, channels, 1)
        self.blocks = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        value = self.project(observations.transpose(1, 2))
        return self.blocks(value).transpose(1, 2)


class CausalTransformerEncoder(nn.Module):
    """Causal self-attention encoder over a fixed-length sliding window."""

    def __init__(
        self,
        observation_size: int,
        d_model: int,
        context_steps: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_steps = context_steps
        self.d_model = d_model
        self.embed = nn.Linear(observation_size, d_model)
        position = torch.zeros(context_steps, d_model)
        for step in range(context_steps):
            for index in range(0, d_model, 2):
                value = step / math.pow(10000.0, index / d_model)
                position[step, index] = math.sin(value)
                if index + 1 < d_model:
                    position[step, index + 1] = math.cos(value)
        self.register_buffer("position", position.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        causal = torch.full(
            (context_steps, context_steps), float("-inf"), dtype=torch.float32
        )
        lower = torch.tril(torch.ones(context_steps, context_steps, dtype=torch.bool))
        causal[lower] = 0.0
        self.register_buffer("causal_mask", causal)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        length = observations.shape[1]
        if length > self.context_steps:
            raise ValueError(
                f"transformer input length {length} exceeds context window "
                f"{self.context_steps}; slice the causal window before calling"
            )
        value = self.embed(observations) * math.sqrt(self.d_model)
        value = value + self.position[:, :length]
        return self.encoder(value, mask=self.causal_mask[:length, :length])


def select_observations(
    values: SequenceSet, observation_names: Sequence[str]
) -> SequenceSet:
    requested = tuple(observation_names)
    name_to_index = {
        name: index for index, name in enumerate(values.observation_names)
    }
    missing = [name for name in requested if name not in name_to_index]
    if missing:
        raise ValueError(f"dataset is missing requested observations: {missing}")
    indices = torch.tensor(
        [name_to_index[name] for name in requested], dtype=torch.int64
    )
    return SequenceSet(
        observations=values.observations.index_select(2, indices),
        actions=values.actions,
        valid_mask=values.valid_mask,
        group_id=values.group_id,
        labels_audit_only=values.labels_audit_only,
        observation_names=requested,
        action_names=values.action_names,
        label_names=values.label_names,
    )


class CausalLQIStudent(nn.Module):
    """Causal recurrent policy from deployable observations to five commands."""

    def __init__(
        self,
        observation_size: int,
        hidden_size: int,
        recurrent_layers: int,
        head_sizes: Sequence[int],
        dropout: float,
        arch: str = "gru",
        context_steps: int | None = None,
        kernel_size: int = 3,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.arch = arch
        if arch in {"tcn", "transformer"} and context_steps is None:
            raise ValueError(f"{arch} requires a context_steps window")
        self.context_steps = context_steps
        if arch == "gru":
            self.encoder = nn.GRU(
                observation_size,
                hidden_size,
                num_layers=recurrent_layers,
                batch_first=True,
                dropout=dropout if recurrent_layers > 1 else 0.0,
            )
        elif arch == "lstm":
            self.encoder = nn.LSTM(
                observation_size,
                hidden_size,
                num_layers=recurrent_layers,
                batch_first=True,
                dropout=dropout if recurrent_layers > 1 else 0.0,
            )
        elif arch == "tcn":
            self.encoder = TCNEncoder(
                observation_size, hidden_size, int(context_steps), kernel_size, dropout
            )
        elif arch == "transformer":
            self.encoder = CausalTransformerEncoder(
                observation_size,
                hidden_size,
                int(context_steps),
                recurrent_layers,
                num_heads,
                dropout,
            )
        else:
            raise ValueError(f"unknown student architecture {arch!r}")
        sizes = [hidden_size, *head_sizes, 5]
        layers: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(sizes) - 2:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*layers)

    def forward(
        self, observations: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.arch in {"gru", "lstm"}:
            encoded, next_hidden = self.encoder(observations, hidden)
        else:
            encoded = self.encoder(observations)
            next_hidden = None
        raw = self.head(encoded)
        commands = torch.cat((torch.sigmoid(raw[..., :2]), torch.tanh(raw[..., 2:])), dim=-1)
        return commands, next_hidden, encoded


@dataclass
class StudentStepState:
    """Opaque stepping state for closed-loop student inference."""

    hidden: Any = None
    window: torch.Tensor | None = None


@torch.no_grad()
def step_student(
    model: CausalLQIStudent,
    checkpoint: Mapping[str, Any],
    normalized: torch.Tensor,
    state: StudentStepState,
    full_context: bool = True,
) -> tuple[torch.Tensor, StudentStepState]:
    """Advance the student one control step with its own action feedback."""

    arch = str(checkpoint.get("arch", "gru"))
    model.eval()
    if arch in {"gru", "lstm"}:
        command, next_hidden, _ = model(
            normalized.unsqueeze(1), state.hidden if full_context else None
        )
        return command[:, 0], StudentStepState(hidden=next_hidden)
    context_steps = int(checkpoint.get("context_steps", 1))
    if full_context:
        if state.window is None:
            window = normalized.unsqueeze(1).expand(-1, context_steps, -1).clone()
        else:
            window = torch.cat((state.window[:, 1:], normalized.unsqueeze(1)), dim=1)
        next_state = StudentStepState(window=window)
        command, _, _ = model(window)
        return command[:, -1], next_state
    command, _, _ = model(normalized.unsqueeze(1))
    return command[:, 0], StudentStepState()


def _shards(dataset: Path, split: str) -> Iterable[Mapping[str, Any]]:
    paths = sorted((dataset / split).glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no shards for split {split!r} in {dataset}")
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        forbidden = {"parameter_labels", "lqi_gain"} & set(shard)
        if forbidden:
            raise ValueError(f"ambiguous non-audit tensors violate leakage contract: {forbidden}")
        yield shard


def load_sequences(dataset: str | Path, split: str) -> SequenceSet:
    root = Path(dataset).expanduser().resolve()
    observations, actions, masks, groups, labels = [], [], [], [], []
    schemas: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None = None
    for shard in _shards(root, split):
        if shard.get("leakage_contract") != "audit tensors are never student inputs":
            raise ValueError("dataset does not declare the strict leakage contract")
        current = (
            tuple(shard["observation_names"]),
            tuple(shard["action_names"]),
            tuple(shard["label_names"]),
        )
        if schemas is not None and schemas != current:
            raise ValueError("shard schemas differ")
        schemas = current
        observations.append(shard["observations"].to(torch.float32))
        actions.append(shard["teacher_actions"].to(torch.float32))
        masks.append(shard["valid_mask"].to(torch.bool))
        groups.append(shard["group_id"].to(torch.int64))
        labels.append(shard["parameter_labels_audit_only"].to(torch.float32))
    if schemas is None:
        raise RuntimeError("unreachable empty dataset")
    return SequenceSet(
        observations=torch.cat(observations),
        actions=torch.cat(actions),
        valid_mask=torch.cat(masks),
        group_id=torch.cat(groups),
        labels_audit_only=torch.cat(labels),
        observation_names=schemas[0],
        action_names=schemas[1],
        label_names=schemas[2],
    )


def fit_observation_normalization(train: SequenceSet) -> tuple[torch.Tensor, torch.Tensor]:
    values = train.observations[train.valid_mask]
    return values.mean(dim=0), values.std(dim=0).clamp_min(1e-5)


def _normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean) / std


def _masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta_weight: float,
) -> torch.Tensor:
    weights = prediction.new_tensor([1.0, 1.0, 0.5, 0.5, 0.5])
    squared = (prediction - target).square() * weights
    point = squared.sum(dim=-1) / weights.sum()
    loss = point[mask].mean()
    if delta_weight > 0 and prediction.shape[1] > 1:
        pair_mask = mask[:, 1:] & mask[:, :-1]
        if bool(pair_mask.any()):
            predicted_delta = prediction[:, 1:] - prediction[:, :-1]
            target_delta = target[:, 1:] - target[:, :-1]
            delta = ((predicted_delta - target_delta).square() * weights).sum(dim=-1) / weights.sum()
            loss = loss + delta_weight * delta[pair_mask].mean()
    return loss


@torch.no_grad()
def _predict(
    model: CausalLQIStudent,
    values: SequenceSet,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    reset_each_step: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs, hidden_features = [], []
    loader = DataLoader(_SequenceDataset(values), batch_size=batch_size, shuffle=False)
    model.eval()
    if model.arch in {"tcn", "transformer"}:
        window_steps = int(model.context_steps)
        for observations, _, _ in loader:
            normalized = _normalize(
                observations.to(device), mean.to(device), std.to(device)
            )
            length = normalized.shape[1]
            if reset_each_step:
                predictions, encoded_parts = [], []
                for step in range(length):
                    command, _, encoded = model(normalized[:, step : step + 1])
                    predictions.append(command)
                    encoded_parts.append(encoded)
                prediction = torch.cat(predictions, dim=1)
                encoded = torch.cat(encoded_parts, dim=1)
            else:
                starts = list(range(0, length, window_steps))
                prediction_parts, encoded_parts = [], []
                for start in starts:
                    stop = min(start + window_steps, length)
                    window = normalized[:, start:stop]
                    command, _, encoded = model(window)
                    prediction_parts.append(command)
                    encoded_parts.append(encoded)
                prediction = torch.cat(prediction_parts, dim=1)
                encoded = torch.cat(encoded_parts, dim=1)
            outputs.append(prediction.cpu())
            hidden_features.append(encoded.cpu())
    else:
        for observations, _, _ in loader:
            normalized = _normalize(
                observations.to(device), mean.to(device), std.to(device)
            )
            if reset_each_step:
                commands, encoded_parts = [], []
                for step in range(normalized.shape[1]):
                    command, _, encoded = model(normalized[:, step : step + 1])
                    commands.append(command)
                    encoded_parts.append(encoded)
                prediction = torch.cat(commands, dim=1)
                encoded = torch.cat(encoded_parts, dim=1)
            else:
                prediction, _, encoded = model(normalized)
            outputs.append(prediction.cpu())
            hidden_features.append(encoded.cpu())
    return torch.cat(outputs), torch.cat(hidden_features)


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(value.mean()),
        "p50": float(torch.quantile(value, 0.5)),
        "p90": float(torch.quantile(value, 0.9)),
        "p99": float(torch.quantile(value, 0.99)),
    }


def imitation_metrics(
    prediction: torch.Tensor, values: SequenceSet, stored_hz: int
) -> dict[str, Any]:
    selected_error = prediction[values.valid_mask] - values.actions[values.valid_mask]
    per_axis_rmse = selected_error.square().mean(dim=0).sqrt()
    action_scale = values.actions[values.valid_mask].std(dim=0).clamp_min(1e-5)
    sequence_error = (prediction - values.actions).square().mean(dim=2).sqrt()
    sequence_mae = (prediction - values.actions).abs().mean(dim=2)
    bins = {}
    for name, start_s, end_s in (
        ("early_0_0.5s", 0.0, 0.5),
        ("identification_0.5_2s", 0.5, 2.0),
        ("late_after_2s", 2.0, 1e9),
    ):
        start = min(round(start_s * stored_hz), prediction.shape[1])
        stop = min(round(end_s * stored_hz), prediction.shape[1])
        current_mask = values.valid_mask[:, start:stop]
        bins[name] = (
            float(sequence_mae[:, start:stop][current_mask].mean())
            if stop > start and bool(current_mask.any())
            else None
        )
    return {
        "command_rmse": dict(zip(values.action_names, per_axis_rmse.tolist())),
        "command_nrmse_by_teacher_std": dict(zip(values.action_names, (per_axis_rmse / action_scale).tolist())),
        "per_sequence_rmse": _quantiles(
            (sequence_error * values.valid_mask).sum(dim=1)
            / values.valid_mask.sum(dim=1).clamp_min(1)
        ),
        "time_conditioned_mae": bins,
    }


def _group_last_hidden(encoded: torch.Tensor, values: SequenceSet) -> tuple[torch.Tensor, torch.Tensor]:
    last_index = values.valid_mask.sum(dim=1).clamp_min(1) - 1
    final = encoded[torch.arange(len(encoded)), last_index]
    unique, inverse = torch.unique(values.group_id, sorted=True, return_inverse=True)
    hidden = torch.zeros(len(unique), final.shape[1])
    labels = torch.zeros(len(unique), values.labels_audit_only.shape[1])
    counts = torch.zeros(len(unique), 1)
    hidden.index_add_(0, inverse, final)
    labels.index_add_(0, inverse, values.labels_audit_only)
    counts.index_add_(0, inverse, torch.ones(len(inverse), 1))
    return hidden / counts, labels / counts


def hidden_parameter_probe(
    train_hidden: torch.Tensor,
    train_values: SequenceSet,
    test_hidden: torch.Tensor,
    test_values: SequenceSet,
    ridge: float,
) -> dict[str, Any]:
    x_train, y_train = _group_last_hidden(train_hidden, train_values)
    x_test, y_test = _group_last_hidden(test_hidden, test_values)
    x_mean, x_std = x_train.mean(0), x_train.std(0).clamp_min(1e-5)
    y_mean, y_std = y_train.mean(0), y_train.std(0).clamp_min(1e-5)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    augmented = torch.cat((x_train, torch.ones(len(x_train), 1)), dim=1)
    identity = torch.eye(augmented.shape[1])
    identity[-1, -1] = 0.0
    weight = torch.linalg.solve(
        augmented.T @ augmented + ridge * identity, augmented.T @ y_train_n
    )
    prediction = torch.cat((x_test, torch.ones(len(x_test), 1)), dim=1) @ weight
    prediction = prediction * y_std + y_mean
    squared = (prediction - y_test).square()
    denominator = (y_test - y_test.mean(0)).square().sum(0).clamp_min(1e-12)
    r2 = 1.0 - squared.sum(0) / denominator
    return {
        "purpose": "diagnostic only; labels were not controller inputs",
        "mean_r2": float(r2.mean()),
        "median_r2": float(r2.median()),
        "per_parameter_r2": dict(zip(test_values.label_names, r2.tolist())),
    }


def _training_config(config_path: Path) -> Mapping[str, Any]:
    root = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return root["distillation"]["training"]


def train(
    dataset_directory: Path,
    config_path: Path,
    output_directory: Path,
    device: torch.device,
    epochs_override: int | None = None,
    resume_checkpoint: Path | None = None,
    exclude_previous_command: bool = False,
    arch: str | None = None,
    context_steps: int | None = None,
    previous_command_noise_std: float = 0.0,
    previous_command_reset_prob: float = 0.0,
    kernel_size: int = 3,
    num_heads: int = 4,
) -> Mapping[str, Any]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    training = _training_config(config_path)
    arch = arch or str(training.get("arch", "gru"))
    if context_steps is None:
        context_steps = training.get("context_steps")
    if context_steps is not None:
        context_steps = int(context_steps)
    if previous_command_noise_std == 0.0:
        previous_command_noise_std = float(
            training.get("previous_command_noise_std", 0.0)
        )
    if previous_command_reset_prob == 0.0:
        previous_command_reset_prob = float(
            training.get("previous_command_reset_prob", 0.0)
        )
    seed = int(yaml.safe_load(config_path.read_text(encoding="utf-8"))["run"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    train_set = load_sequences(dataset_directory, "train")
    validation_set = load_sequences(dataset_directory, "validation")
    if exclude_previous_command:
        selected_names = tuple(
            name
            for name in train_set.observation_names
            if not name.startswith("previous_command.")
        )
        train_set = select_observations(train_set, selected_names)
        validation_set = select_observations(validation_set, selected_names)
    mean, std = fit_observation_normalization(train_set)
    model = CausalLQIStudent(
        len(train_set.observation_names),
        int(training["hidden_size"]),
        int(training["recurrent_layers"]),
        tuple(int(value) for value in training["head_sizes"]),
        float(training["dropout"]),
        arch=arch,
        context_steps=context_steps,
        kernel_size=kernel_size,
        num_heads=num_heads,
    ).to(device)
    if resume_checkpoint is not None:
        parent = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        if parent.get("artifact_type") not in {
            "causal_oracle_lqi_gru_student_v1",
            "causal_oracle_lqi_student_v2",
        }:
            raise ValueError("resume checkpoint is not a LQI student")
        if tuple(parent["observation_names"]) != train_set.observation_names:
            raise ValueError("resume checkpoint observation schema differs")
        model.load_state_dict(parent["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=1e-5)
    batch_size = int(training["batch_size"])
    epochs = epochs_override or int(training["epochs"])
    delta_weight = float(training["command_delta_loss_weight"])
    loader_generator = torch.Generator().manual_seed(seed + 2)
    previous_indices = [
        index
        for index, name in enumerate(train_set.observation_names)
        if name.startswith("previous_command.")
    ]
    initial_previous_normalized = None
    if previous_indices and previous_command_reset_prob > 0.0:
        manifest = json.loads((dataset_directory / "manifest.json").read_text(encoding="utf-8"))
        initial_previous = torch.as_tensor(
            manifest["initial_previous_command"], dtype=torch.float32
        )
        initial_previous_normalized = _normalize(
            initial_previous, mean[previous_indices], std[previous_indices]
        )
    windowed = model.arch in {"tcn", "transformer"}
    sequence_length = train_set.observations.shape[1]
    best_state = None
    best_validation = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        loader = DataLoader(
            (
                _WindowedSequenceDataset(
                    train_set, int(context_steps), loader_generator
                )
                if windowed
                else _SequenceDataset(train_set)
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        for observations, actions, mask in loader:
            normalized = _normalize(
                observations.to(device), mean.to(device), std.to(device)
            )
            if previous_indices:
                if previous_command_noise_std > 0.0:
                    noise = torch.randn(
                        *normalized[..., previous_indices].shape,
                        device=normalized.device,
                    ) * previous_command_noise_std
                    normalized[..., previous_indices] = normalized[..., previous_indices] + noise
                if previous_command_reset_prob > 0.0 and initial_previous_normalized is not None:
                    reset = (
                        torch.rand(
                            normalized.shape[0],
                            normalized.shape[1],
                            1,
                            device=normalized.device,
                        )
                        < previous_command_reset_prob
                    )
                    normalized[..., previous_indices] = torch.where(
                        reset,
                        initial_previous_normalized.to(normalized.device),
                        normalized[..., previous_indices],
                    )
            prediction, _, _ = model(normalized)
            loss = _masked_loss(prediction, actions.to(device), mask.to(device), delta_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach())
            batches += 1
        validation_prediction, _ = _predict(model, validation_set, mean, std, device, batch_size)
        validation_loss = float(
            _masked_loss(validation_prediction, validation_set.actions, validation_set.valid_mask, delta_weight)
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss / max(batches, 1), "validation_loss": validation_loss})
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    checkpoint = {
        "artifact_type": "causal_oracle_lqi_student_v2",
        "model_state": best_state,
        "model": {
            "observation_size": len(train_set.observation_names),
            "hidden_size": int(training["hidden_size"]),
            "recurrent_layers": int(training["recurrent_layers"]),
            "head_sizes": tuple(int(value) for value in training["head_sizes"]),
            "dropout": float(training["dropout"]),
            "arch": arch,
            "context_steps": context_steps,
            "kernel_size": kernel_size,
            "num_heads": num_heads,
        },
        "normalization": {"mean": mean, "std": std},
        "observation_names": train_set.observation_names,
        "action_names": train_set.action_names,
        "leakage_contract": "no parameters, gains, or truth-only actuator state in model input",
        "previous_command_input": not exclude_previous_command,
        "previous_command_noise_std": previous_command_noise_std,
        "previous_command_reset_prob": previous_command_reset_prob,
        "source_dataset": str(dataset_directory),
        "parent_checkpoint": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
        "sequence_length": sequence_length,
    }
    torch.save(checkpoint, output_directory / "student.pt")
    (output_directory / "training.json").write_text(
        json.dumps({"best_validation_loss": best_validation, "history": history}, indent=2), encoding="utf-8"
    )
    return {"best_validation_loss": best_validation, "epochs": epochs, "checkpoint": str(output_directory / "student.pt")}


def load_student(checkpoint_path: Path, device: torch.device) -> tuple[CausalLQIStudent, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    artifact_type = checkpoint.get("artifact_type")
    if artifact_type not in {
        "causal_oracle_lqi_gru_student_v1",
        "causal_oracle_lqi_student_v2",
    }:
        raise ValueError("not a LQI student checkpoint")
    cfg = checkpoint["model"]
    arch = str(cfg.get("arch", "gru"))
    model = CausalLQIStudent(
        int(cfg["observation_size"]),
        int(cfg["hidden_size"]),
        int(cfg["recurrent_layers"]),
        tuple(cfg["head_sizes"]),
        float(cfg["dropout"]),
        arch=arch,
        context_steps=cfg.get("context_steps"),
        kernel_size=int(cfg.get("kernel_size", 3)),
        num_heads=int(cfg.get("num_heads", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def evaluate(
    dataset_directory: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    probe_ridge: float,
) -> Mapping[str, Any]:
    manifest = json.loads((dataset_directory / "manifest.json").read_text(encoding="utf-8"))
    stored_hz = int(manifest["stored_hz"])
    train_set = load_sequences(dataset_directory, "train")
    test_set = load_sequences(dataset_directory, "test")
    model, checkpoint = load_student(checkpoint_path, device)
    train_set = select_observations(train_set, checkpoint["observation_names"])
    test_set = select_observations(test_set, checkpoint["observation_names"])
    mean = checkpoint["normalization"]["mean"]
    std = checkpoint["normalization"]["std"]
    prediction, test_hidden = _predict(model, test_set, mean, std, device, batch_size)
    reset_prediction, _ = _predict(model, test_set, mean, std, device, batch_size, reset_each_step=True)
    _, train_hidden = _predict(model, train_set, mean, std, device, batch_size)
    normal = imitation_metrics(prediction, test_set, stored_hz)
    reset = imitation_metrics(reset_prediction, test_set, stored_hz)
    normal_rmse = normal["per_sequence_rmse"]["mean"]
    reset_rmse = reset["per_sequence_rmse"]["mean"]
    report = {
        "schema_version": 1,
        "verdict_scope": "offline held-out-airframe capacity test; not deployment validation",
        "held_out_imitation": normal,
        "memory_ablation_reset_hidden_each_step": reset,
        "memory_ablation_rmse_ratio_reset_over_recurrent": reset_rmse / max(normal_rmse, 1e-12),
        "hidden_parameter_probe": hidden_parameter_probe(
            train_hidden, train_set, test_hidden, test_set, probe_ridge
        ),
        "required_next_gate": "paired nonlinear closed-loop student vs oracle and memoryless baseline",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train a causal GRU on oracle LQI commands")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--exclude-previous-command", action="store_true")
    parser.add_argument(
        "--arch",
        choices=("gru", "lstm", "tcn", "transformer"),
        default=None,
    )
    parser.add_argument("--context-steps", type=int)
    parser.add_argument("--previous-command-noise-std", type=float, default=0.0)
    parser.add_argument("--previous-command-reset-prob", type=float, default=0.0)
    args = parser.parse_args()
    result = train(
        Path(args.dataset).resolve(),
        Path(args.config).resolve(),
        Path(args.output_directory).resolve(),
        torch.device(args.device),
        args.epochs,
        None if args.resume_checkpoint is None else Path(args.resume_checkpoint).resolve(),
        args.exclude_previous_command,
        args.arch,
        args.context_steps,
        args.previous_command_noise_std,
        args.previous_command_reset_prob,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def eval_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hidden identification and imitation")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-ridge", type=float, default=0.001)
    args = parser.parse_args()
    report = evaluate(Path(args.dataset).resolve(), Path(args.checkpoint).resolve(), Path(args.output).resolve(), torch.device(args.device), args.batch_size, args.probe_ridge)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    train_main()
