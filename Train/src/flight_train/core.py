from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from tensordict import TensorDictBase


@dataclass(frozen=True)
class EnvSpec:
    parallel_count: int
    observation_dim: int
    action_dim: int
    device: torch.device
    dtype: torch.dtype
    physics_hz: int
    control_hz: int
    observation_fields: tuple[str, ...]

class BatchedControlEnv(Protocol):
    @property
    def spec(self) -> EnvSpec: ...

    def reset(self, mask: torch.Tensor | None = None) -> TensorDictBase: ...

    def step(self, standard_action: torch.Tensor) -> TensorDictBase: ...

    def close(self) -> None: ...


def assert_tensor_on(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    shape: tuple[int, ...] | None = None,
    name: str,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
