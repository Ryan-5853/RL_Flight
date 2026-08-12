from __future__ import annotations

import torch


def compose_external_upper_command(
    upper_throttle: torch.Tensor,
    controlled_command: torch.Tensor,
) -> torch.Tensor:
    """Compose the physical five-channel command without ceding upper ownership.

    The attitude controller supplies exactly ``lower + three servos``.  The
    external pilot value is copied into channel zero and cannot be overwritten
    by the four-output policy.
    """

    if upper_throttle.ndim != 2 or upper_throttle.shape[1] != 1:
        raise ValueError("upper_throttle must have shape [batch, 1]")
    if controlled_command.ndim != 2 or controlled_command.shape[1] != 4:
        raise ValueError("controlled_command must have shape [batch, 4]")
    if upper_throttle.shape[0] != controlled_command.shape[0]:
        raise ValueError("external and controlled command batches must match")
    if (
        upper_throttle.device != controlled_command.device
        or upper_throttle.dtype != controlled_command.dtype
    ):
        raise ValueError("external and controlled commands must share device/dtype")
    return torch.cat((upper_throttle.clamp(0.0, 1.0), controlled_command), dim=1)
