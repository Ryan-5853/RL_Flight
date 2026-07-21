from __future__ import annotations

import torch


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    normalized = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    sign = torch.where(normalized[..., :1] < 0, -1.0, 1.0)
    return normalized * sign


def quaternion_geodesic_angle(q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    qn = normalize_quaternion(q)
    tn = normalize_quaternion(target)
    dot = (qn * tn).sum(dim=-1, keepdim=True).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def euler_to_quaternion(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return normalize_quaternion(torch.stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ), dim=-1))


def tilt_angle(q_wb: torch.Tensor) -> torch.Tensor:
    """Angle between body z and world z; Hamilton q rotates body to world."""
    q = normalize_quaternion(q_wb)
    w, x, y, _z = q.unbind(dim=-1)
    body_z_dot_world_z = (1.0 - 2.0 * (x.square() + y.square())).clamp(-1.0, 1.0)
    return torch.acos(body_z_dot_world_z).unsqueeze(-1)

