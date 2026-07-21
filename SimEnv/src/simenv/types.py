from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, Mapping

import torch


class ErrorCode(IntEnum):
    OK = 0
    INVALID_CONTROL = 1
    NONFINITE_STATE = 2
    NUMERICAL_FAILURE = 3


@dataclass(frozen=True)
class Observation:
    batch_id: str
    instance_ids: tuple[str, ...]
    physics_step: torch.Tensor
    control_step: torch.Tensor
    sim_time_s: torch.Tensor
    source: Literal["truth", "sensor"]
    values: Mapping[str, torch.Tensor]
    valid: torch.Tensor


@dataclass(frozen=True)
class AdvanceResult:
    batch_id: str
    instance_ids: tuple[str, ...]
    physics_step: torch.Tensor
    control_step: torch.Tensor
    sim_time_s: torch.Tensor
    physics_steps_advanced: torch.Tensor
    valid: torch.Tensor
    error_code: torch.Tensor


@dataclass(frozen=True)
class ResetResult:
    batch_id: str
    reset_mask: torch.Tensor
    previous_instance_ids: tuple[str, ...]
    instance_ids: tuple[str, ...]
    generation: torch.Tensor
