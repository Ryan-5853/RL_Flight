from __future__ import annotations

import torch


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat(
        (
            aw * bw - (av * bv).sum(dim=-1, keepdim=True),
            aw * bv + bw * av + torch.linalg.cross(av, bv),
        ),
        dim=-1,
    )


def quaternion_rotation_error(
    current_q_wb: torch.Tensor,
    target_q_wb: torch.Tensor,
) -> torch.Tensor:
    """Return the target-to-current local rotation vector.

    The sign is canonicalized so ``q`` and ``-q`` produce the same error.
    """

    error = quaternion_multiply(
        quaternion_conjugate(target_q_wb), current_q_wb
    )
    error = torch.where(error[..., :1] < 0, -error, error)
    vector_norm = torch.linalg.vector_norm(error[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, error[..., :1].clamp_min(1e-12))
    return error[..., 1:] * (angle / vector_norm.clamp_min(1e-12))


def tilt_cosine(q_wb: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q_wb.unbind(dim=-1)
    return (1.0 - 2.0 * (x.square() + y.square())).clamp(-1.0, 1.0)


def lookup(value: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear lookup with a table shaped ``[..., N, 2]``."""

    x_axis = table[..., 0]
    y_axis = table[..., 1]
    clamped = torch.maximum(
        torch.minimum(value, x_axis[..., -1]), x_axis[..., 0]
    )
    upper = torch.searchsorted(x_axis.contiguous(), clamped.unsqueeze(-1)).squeeze(-1)
    upper = upper.clamp(1, x_axis.shape[-1] - 1)
    lower = upper - 1
    x0 = torch.gather(x_axis, -1, lower.unsqueeze(-1)).squeeze(-1)
    x1 = torch.gather(x_axis, -1, upper.unsqueeze(-1)).squeeze(-1)
    y0 = torch.gather(y_axis, -1, lower.unsqueeze(-1)).squeeze(-1)
    y1 = torch.gather(y_axis, -1, upper.unsqueeze(-1)).squeeze(-1)
    return y0 + (clamped - x0) * (y1 - y0) / (x1 - x0)


def inverse_lookup(value: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    swapped = torch.stack((table[..., 1], table[..., 0]), dim=-1)
    return lookup(value, swapped)
