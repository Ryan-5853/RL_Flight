from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class _TemporalPooling(nn.Module):
    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.to(value.dtype)[:, None, :]
        count = weights.sum(dim=2).clamp_min(1.0)
        mean = (value * weights).sum(dim=2) / count
        centered = (value - mean[:, :, None]) * weights
        standard_deviation = torch.sqrt(
            centered.square().sum(dim=2) / count + 1e-8
        )
        maximum = value.masked_fill(~valid[:, None, :], -torch.inf).amax(dim=2)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum, standard_deviation), dim=1)


class _ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = 2 * dilation
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(1, output_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
            nn.GroupNorm(1, output_channels),
        )
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(value) + self.residual(value))


class TemporalConvTrialEncoder(nn.Module):
    """Non-causal full-log encoder; inference is intentionally offline."""

    def __init__(
        self,
        feature_count: int,
        channels: Sequence[int],
        embedding_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("temporal convolution requires at least one channel")
        blocks: list[nn.Module] = []
        current = feature_count
        for index, output in enumerate(channels):
            blocks.append(
                _ResidualTemporalBlock(
                    current, int(output), 2**index, dropout
                )
            )
            current = int(output)
        self.temporal = nn.Sequential(*blocks)
        self.pool = _TemporalPooling()
        self.project = nn.Sequential(
            nn.Linear(3 * current, embedding_size),
            nn.SiLU(),
        )

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        value = self.temporal(history.transpose(1, 2))
        return self.project(self.pool(value, valid))


class BidirectionalGRUTrialEncoder(nn.Module):
    """Bidirectional recurrent encoder for complete post-flight logs."""

    def __init__(
        self,
        feature_count: int,
        hidden_size: int,
        layer_count: int,
        embedding_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.recurrent = nn.GRU(
            feature_count,
            hidden_size,
            num_layers=layer_count,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layer_count > 1 else 0.0,
        )
        self.pool = _TemporalPooling()
        self.project = nn.Sequential(
            nn.Linear(6 * hidden_size, embedding_size),
            nn.SiLU(),
        )

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        value, _ = self.recurrent(history)
        return self.project(self.pool(value.transpose(1, 2), valid))


class OfflineTemporalSetIdentifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_size: int,
        output_count: int,
        head_hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        sizes = [3 * embedding_size, *head_hidden_sizes, output_count]
        layers: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(sizes) - 2:
                layers.append(nn.SiLU())
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        histories: torch.Tensor,
        trial_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if histories.ndim != 4:
            raise ValueError("histories must have shape [batch, trials, time, features]")
        batch_size, trial_count, time_count, feature_count = histories.shape
        if trial_mask is None:
            trial_mask = torch.ones(
                batch_size, trial_count, device=histories.device, dtype=torch.bool
            )
        if trial_mask.shape != (batch_size, trial_count):
            raise ValueError("trial_mask shape must match batch and trial dimensions")
        if not bool(trial_mask.any(dim=1).all().item()):
            raise ValueError("every sample must include at least one trial")
        flattened = histories.reshape(
            batch_size * trial_count, time_count, feature_count
        )
        valid_time = flattened[:, :, -1] > 0.5
        encoded = self.encoder(flattened, valid_time).reshape(
            batch_size, trial_count, -1
        )
        weights = trial_mask.to(encoded.dtype)[..., None]
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (encoded * weights).sum(dim=1) / count
        centered = (encoded - mean[:, None]) * weights
        standard_deviation = torch.sqrt(
            centered.square().sum(dim=1) / count + 1e-8
        )
        maximum = encoded.masked_fill(~trial_mask[..., None], -torch.inf).amax(dim=1)
        return self.head(torch.cat((mean, maximum, standard_deviation), dim=1))


def build_offline_identifier(
    architecture: str,
    history_steps: int,
    feature_count: int,
    output_count: int,
    trial_hidden_sizes: Sequence[int],
    head_hidden_sizes: Sequence[int],
    temporal_channels: Sequence[int] = (64, 96, 128),
    recurrent_hidden_size: int = 96,
    recurrent_layers: int = 2,
    trial_embedding_size: int = 128,
    temporal_dropout: float = 0.05,
) -> nn.Module:
    if architecture == "mlp":
        from .repeated_trial_training import TrialSetIdentifier

        return TrialSetIdentifier(
            history_steps,
            feature_count,
            output_count,
            trial_hidden_sizes,
            head_hidden_sizes,
        )
    if architecture == "tcn":
        encoder: nn.Module = TemporalConvTrialEncoder(
            feature_count,
            temporal_channels,
            trial_embedding_size,
            temporal_dropout,
        )
    elif architecture == "bigru":
        encoder = BidirectionalGRUTrialEncoder(
            feature_count,
            recurrent_hidden_size,
            recurrent_layers,
            trial_embedding_size,
            temporal_dropout,
        )
    else:
        raise ValueError(f"unsupported offline architecture: {architecture!r}")
    return OfflineTemporalSetIdentifier(
        encoder,
        trial_embedding_size,
        output_count,
        head_hidden_sizes,
    )
