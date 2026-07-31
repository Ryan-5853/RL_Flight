from __future__ import annotations

import torch


class WeightedControlAllocator:
    def __init__(
        self,
        effectiveness: torch.Tensor,
        input_weights: torch.Tensor,
        damping: float = 1e-5,
    ) -> None:
        if effectiveness.ndim != 3 or effectiveness.shape[1:] != (4, 5):
            raise ValueError("effectiveness must have shape [B,4,5]")
        if input_weights.shape != (effectiveness.shape[0], 5):
            raise ValueError("input_weights must have shape [B,5]")
        if not bool((input_weights > 0).all().item()):
            raise ValueError("input_weights must be positive")
        self.effectiveness = effectiveness
        self.input_weights = input_weights
        inverse_weights = torch.diag_embed(input_weights.reciprocal())
        middle = (
            effectiveness @ inverse_weights @ effectiveness.transpose(1, 2)
            + damping
            * torch.eye(
                4,
                device=effectiveness.device,
                dtype=effectiveness.dtype,
            )[None]
        )
        self._mixer = (
            inverse_weights
            @ effectiveness.transpose(1, 2)
            @ torch.linalg.inv(middle)
        )

    def allocate(
        self,
        base_command: torch.Tensor,
        wrench_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unconstrained = base_command + torch.bmm(
            self._mixer, wrench_delta.unsqueeze(-1)
        ).squeeze(-1)
        lower = unconstrained.new_tensor([0.0, 0.0, -1.0, -1.0, -1.0])
        upper = unconstrained.new_tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        command = unconstrained.clamp(lower, upper)
        achieved = torch.bmm(
            self.effectiveness, (command - base_command).unsqueeze(-1)
        ).squeeze(-1)
        return command, achieved
