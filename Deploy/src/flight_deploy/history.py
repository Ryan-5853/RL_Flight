from __future__ import annotations

import torch


class UniformHistoryBuffer:
    """Batch-first history with Train-compatible oldest-to-newest flattening."""

    def __init__(
        self,
        batch_size: int,
        frame_dim: int,
        frames: int,
        *,
        stride_steps: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if min(batch_size, frame_dim, frames, stride_steps) <= 0:
            raise ValueError("history dimensions and stride must be positive")
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.frames = frames
        self.stride_steps = stride_steps
        self.capacity = (frames - 1) * stride_steps + 1
        self.values = torch.zeros(
            batch_size,
            self.capacity,
            frame_dim,
            device=device,
            dtype=dtype,
        )
        self.index = self.capacity - 1
        self._offsets = (
            torch.arange(frames - 1, -1, -1, device=device) * stride_steps
        )
        self.initialized = torch.zeros(batch_size, device=device, dtype=torch.bool)

    def reset(self, frame: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        self._check_frame(frame)
        if mask is None:
            mask = torch.ones(
                self.batch_size, device=frame.device, dtype=torch.bool
            )
        if mask.shape != (self.batch_size,) or mask.dtype != torch.bool:
            raise ValueError("reset mask must have shape [B] and bool dtype")
        replacement = frame[:, None, :].expand(-1, self.capacity, -1)
        self.values.copy_(
            torch.where(mask[:, None, None], replacement, self.values)
        )
        self.initialized |= mask

    def append(
        self,
        frame: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_frame(frame)
        if not bool(self.initialized.all()):
            raise RuntimeError("history must be reset before the first append")
        self.index = (self.index + 1) % self.capacity
        self.values[:, self.index].copy_(frame)
        if reset_mask is not None:
            self.reset(frame, reset_mask)
        return self.observation()

    def observation(self) -> torch.Tensor:
        indices = (self.index - self._offsets) % self.capacity
        return self.values.index_select(1, indices).reshape(
            self.batch_size, self.frames * self.frame_dim
        )

    def _check_frame(self, frame: torch.Tensor) -> None:
        if frame.shape != (self.batch_size, self.frame_dim):
            raise ValueError(
                f"frame must have shape {(self.batch_size, self.frame_dim)}"
            )
        if frame.device != self.values.device or frame.dtype != self.values.dtype:
            raise ValueError("frame device and dtype must match history buffer")
