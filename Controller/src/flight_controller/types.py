from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class ControllerContext:
    """Static information shared by every controller implementation."""

    batch_size: int
    device: torch.device
    dtype: torch.dtype
    control_dt: float
    parameters: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ControllerState:
    """One batched SimEnv observation at a controller boundary."""

    position_n: torch.Tensor
    velocity_n: torch.Tensor
    attitude_q_wb: torch.Tensor
    angular_velocity_b: torch.Tensor
    linear_acceleration_n: torch.Tensor
    motor_speed: torch.Tensor
    servo_angle: torch.Tensor

    @classmethod
    def from_truth(cls, values: Mapping[str, torch.Tensor]) -> "ControllerState":
        return cls(**{name: values[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ControllerReference:
    """Controller-independent command after input filtering and mode mapping."""

    target_position_n: torch.Tensor
    target_velocity_n: torch.Tensor
    target_attitude_q_wb: torch.Tensor
    target_angular_velocity_b: torch.Tensor
    collective_command: torch.Tensor


@dataclass(frozen=True)
class ControllerOutput:
    """Physical SimEnv command plus controller-specific diagnostic tensors."""

    command: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor]

    @classmethod
    def create(
        cls,
        command: torch.Tensor,
        diagnostics: Mapping[str, torch.Tensor] | None = None,
    ) -> "ControllerOutput":
        return cls(command, MappingProxyType(dict(diagnostics or {})))


def validate_context(context: ControllerContext) -> None:
    if context.batch_size <= 0:
        raise ValueError("controller batch_size must be positive")
    if context.control_dt <= 0:
        raise ValueError("controller control_dt must be positive")
    if not context.dtype.is_floating_point:
        raise ValueError("controller dtype must be floating point")


def tensor_diagnostics_to_python(
    diagnostics: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        name: value.detach().cpu().tolist()
        for name, value in diagnostics.items()
    }
