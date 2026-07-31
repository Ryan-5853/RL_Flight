from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn


_ACTIVATIONS: Mapping[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class FeedForwardPolicy(nn.Module):
    """Portable deterministic MLP used by the first built-in adapter.

    The class deliberately stores a plain sequence of standard PyTorch
    operators. It has no TorchRL or TensorDict dependency and can be scripted,
    exported, or translated by a target-specific backend.
    """

    def __init__(
        self,
        dimensions: tuple[int, ...],
        *,
        activation: str = "silu",
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if len(dimensions) < 2 or any(value <= 0 for value in dimensions):
            raise ValueError("dimensions must contain at least two positive values")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unsupported activation: {activation}")
        if output_activation not in _ACTIVATIONS:
            raise ValueError(f"unsupported output activation: {output_activation}")
        layers: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(dimensions) - 2:
                layers.append(_ACTIVATIONS[activation]())
        layers.append(_ACTIVATIONS[output_activation]())
        self.network = nn.Sequential(*layers)
        self.input_dim = dimensions[0]
        self.output_dim = dimensions[-1]

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


@dataclass
class ConvertedPolicy:
    """Adapter-neutral result consumed by bundle backends."""

    module: nn.Module
    input_dim: int
    output_dim: int
    architecture: Mapping[str, Any]
    contract: Mapping[str, Any] = field(default_factory=dict)
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    state_inputs: tuple[Mapping[str, Any], ...] = ()
    state_outputs: tuple[Mapping[str, Any], ...] = ()

    @property
    def stateful(self) -> bool:
        return bool(self.state_inputs or self.state_outputs)
