from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import FlightController
from .types import ControllerContext, ControllerOutput, ControllerReference, ControllerState


class NeuralNetworkController(FlightController):
    """Adapter for current MLP/GRU policies with the same physical command API."""

    controller_type = "neural"

    def __init__(
        self,
        context: ControllerContext,
        config: Mapping[str, Any] | None,
        *,
        model: Any,
    ) -> None:
        super().__init__(context, config)
        self.model = model
        self.maximum_angular_rate = float(self.config.get("maximum_angular_rate_rad_s", 20.0))
        self.output_mode = str(self.config.get("output_mode", "residual_4"))
        if self.output_mode not in {"residual_4", "physical_5"}:
            raise ValueError("neural output_mode must be residual_4 or physical_5")
        self.previous_action = torch.zeros(
            context.batch_size, 4, device=context.device, dtype=context.dtype
        )
        self.is_init = torch.ones(
            context.batch_size, 1, device=context.device, dtype=torch.bool
        )
        self.recurrent_state = None

    def reset(self, reset_mask: torch.Tensor) -> None:
        mask = self._active_mask(reset_mask)
        self.previous_action.masked_fill_(mask[:, None], 0.0)
        self.is_init.masked_fill_(mask[:, None], True)
        # The current Train wrapper does not expose a generic hidden-state mask
        # contract. Interactive runtime uses B=1, so clearing is exact there.
        if bool(mask.any().item()):
            self.recurrent_state = None

    def step(
        self,
        state: ControllerState,
        reference: ControllerReference,
        active_mask: torch.Tensor | None = None,
    ) -> ControllerOutput:
        active = self._active_mask(active_mask)
        observation = torch.cat(
            (
                state.attitude_q_wb,
                state.angular_velocity_b / self.maximum_angular_rate,
                state.linear_acceleration_n / 9.80665,
                state.motor_speed / 1800.0,
                reference.target_attitude_q_wb,
                reference.collective_command * 2.0 - 1.0,
                self.previous_action,
            ),
            dim=1,
        )
        if hasattr(self.model, "forward_step"):
            action, recurrent = self.model.forward_step(
                observation, self.recurrent_state, self.is_init
            )
            self.recurrent_state = recurrent
        else:
            result = self.model(observation)
            if isinstance(result, tuple):
                action, self.recurrent_state = result
            else:
                action = result
        self.is_init.masked_fill_(active[:, None], False)
        action = action.clamp(-1.0, 1.0)
        if self.output_mode == "physical_5":
            if action.shape != (self.context.batch_size, 5):
                raise ValueError("physical_5 neural controller must output [B,5]")
            command = torch.cat(
                (action[:, :2].clamp(0.0, 1.0), action[:, 2:].clamp(-1.0, 1.0)),
                dim=1,
            )
        else:
            if action.shape != (self.context.batch_size, 4):
                raise ValueError("residual_4 neural controller must output [B,4]")
            command = torch.cat(
                (
                    reference.collective_command.clamp(0.0, 1.0),
                    (action[:, :1] + 1.0) * 0.5,
                    action[:, 1:],
                ),
                dim=1,
            )
        command = torch.where(
            active[:, None],
            command,
            command,
        )
        if self.output_mode == "residual_4":
            self.previous_action.copy_(
                torch.where(active[:, None], action, self.previous_action)
            )
        return ControllerOutput.create(
            command,
            {
                "controller.mode_code": torch.full(
                    (self.context.batch_size,),
                    4.0,
                    device=self.context.device,
                    dtype=self.context.dtype,
                ),
                "controller.policy_action": action,
                "controller.command": command,
            },
        )
