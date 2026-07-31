from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch

from .types import (
    ControllerContext,
    ControllerOutput,
    ControllerReference,
    ControllerState,
    validate_context,
)


class FlightController(ABC):
    """Common lifecycle used by PID, LQR, hybrid and learned policies."""

    controller_type = "abstract"

    def __init__(
        self,
        context: ControllerContext,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        validate_context(context)
        self.context = context
        self.config = dict(config or {})

    @abstractmethod
    def reset(self, reset_mask: torch.Tensor) -> None:
        """Reset only the selected batch slots."""

    @abstractmethod
    def step(
        self,
        state: ControllerState,
        reference: ControllerReference,
        active_mask: torch.Tensor | None = None,
    ) -> ControllerOutput:
        """Return a physical ``[B,5]`` SimEnv command."""

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.controller_type,
            "batch_size": self.context.batch_size,
            "control_dt": self.context.control_dt,
        }

    def _active_mask(self, active_mask: torch.Tensor | None) -> torch.Tensor:
        if active_mask is None:
            return torch.ones(
                self.context.batch_size,
                device=self.context.device,
                dtype=torch.bool,
            )
        if (
            active_mask.shape != (self.context.batch_size,)
            or active_mask.device != self.context.device
            or active_mask.dtype != torch.bool
        ):
            raise ValueError("active_mask must be a bool [B] tensor on the controller device")
        return active_mask
