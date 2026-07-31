from __future__ import annotations

import torch


class ResidualActionTransform:
    def __init__(
        self,
        trim: tuple[float, ...],
        scale: tuple[float, ...],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not trim or len(trim) != len(scale):
            raise ValueError("trim and scale must have equal non-zero length")
        if any(value <= 0 for value in scale):
            raise ValueError("residual scales must be positive")
        self.trim = torch.tensor(trim, device=device, dtype=dtype)
        self.scale = torch.tensor(scale, device=device, dtype=dtype)

    def __call__(self, policy_action: torch.Tensor) -> torch.Tensor:
        if policy_action.shape[-1] != self.trim.numel():
            raise ValueError("policy action width does not match transform")
        return self.trim + self.scale * policy_action.clamp(-1.0, 1.0)
