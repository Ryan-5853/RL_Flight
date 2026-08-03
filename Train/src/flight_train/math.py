from __future__ import annotations

import torch


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """批量归一化 Hamilton 四元数，并统一到实部非负的等价表示。"""

    normalized = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    sign = torch.where(normalized[..., :1] < 0, -1.0, 1.0)
    return normalized * sign


def quaternion_geodesic_angle(q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """返回姿态四元数之间的最短测地角，输出形状为 ``[..., 1]``。

    点积取绝对值使 ``q`` 与 ``-q`` 表示同一姿态，避免奖励在符号翻转处跳变。
    """

    qn = normalize_quaternion(q)
    tn = normalize_quaternion(target)
    dot = (qn * tn).sum(dim=-1, keepdim=True).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """批量 Hamilton 四元数乘法，输入尾维均为 4。"""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def relative_quaternion(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """返回从当前姿态旋转到目标姿态的规范化相对四元数。"""

    current = normalize_quaternion(current)
    target = normalize_quaternion(target)
    conjugate = current.clone()
    conjugate[..., 1:] = -conjugate[..., 1:]
    return normalize_quaternion(quaternion_multiply(target, conjugate))


def attitude_error_rotation_vector(
    current: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Return the shortest current-to-target rotation vector in body coordinates."""

    current = normalize_quaternion(current)
    target = normalize_quaternion(target)
    conjugate = current.clone()
    conjugate[..., 1:] = -conjugate[..., 1:]
    error = normalize_quaternion(quaternion_multiply(conjugate, target))
    vector = error[..., 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, error[..., :1].clamp_min(0.0))
    scale = torch.where(
        vector_norm > eps,
        angle / vector_norm.clamp_min(eps),
        torch.full_like(vector_norm, 2.0),
    )
    return vector * scale


def euler_to_quaternion(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """将同批次的滚转、俯仰、偏航角转换为 Hamilton 四元数 ``[w,x,y,z]``。"""

    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return normalize_quaternion(torch.stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ), dim=-1))


def quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
    """Hamilton ``q_wb`` 转 roll/pitch/yaw，输出尾维为 3。"""

    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x.square() + y.square()),
    )
    pitch = torch.asin(
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    )
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )
    return torch.stack((roll, pitch, yaw), dim=-1)


def tilt_angle(q_wb: torch.Tensor) -> torch.Tensor:
    """计算机体坐标系 z 轴与世界坐标系 z 轴夹角，输出形状为 ``[..., 1]``。

    ``q_wb`` 采用 Hamilton 约定，表示从机体系旋转到世界系。
    """
    q = normalize_quaternion(q_wb)
    w, x, y, _z = q.unbind(dim=-1)
    body_z_dot_world_z = (1.0 - 2.0 * (x.square() + y.square())).clamp(-1.0, 1.0)
    return torch.acos(body_z_dot_world_z).unsqueeze(-1)
