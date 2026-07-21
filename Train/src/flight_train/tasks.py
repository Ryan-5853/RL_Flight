from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .config import TaskConfig
from .math import euler_to_quaternion, quaternion_geodesic_angle, tilt_angle


@dataclass(frozen=True)
class TaskTransition:
    reward: torch.Tensor
    terminated: torch.Tensor
    info: Mapping[str, torch.Tensor]


class AttitudeTrackingTask:
    """Fully batched reward/termination kernel; no tensor leaves its device."""

    def __init__(self, config: TaskConfig, batch_size: int, device: torch.device, dtype: torch.dtype, seed: int) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.target_attitude = torch.zeros((batch_size, 4), device=device, dtype=dtype)
        self.target_attitude[:, 0] = 1.0

    def reset(self, mask: torch.Tensor) -> None:
        shape = (self.batch_size,)
        roll = (torch.rand(shape, device=self.device, dtype=self.dtype, generator=self.generator) * 2 - 1) * self.config.max_target_tilt_rad
        pitch = (torch.rand(shape, device=self.device, dtype=self.dtype, generator=self.generator) * 2 - 1) * self.config.max_target_tilt_rad
        yaw = (torch.rand(shape, device=self.device, dtype=self.dtype, generator=self.generator) * 2 - 1) * self.config.max_target_yaw_rad
        sampled = euler_to_quaternion(roll, pitch, yaw)
        self.target_attitude = torch.where(mask[:, None], sampled, self.target_attitude)

    def transition(
        self,
        attitude: torch.Tensor,
        angular_velocity: torch.Tensor,
        standard_action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> TaskTransition:
        attitude_error = quaternion_geodesic_angle(attitude, self.target_attitude)
        rate_cost = angular_velocity.square().sum(dim=-1, keepdim=True)
        action_rate_cost = (standard_action - previous_action).square().sum(dim=-1, keepdim=True)
        saturation_cost = torch.relu(standard_action.abs() - 0.95).square().sum(dim=-1, keepdim=True)
        reward_config = self.config.reward
        reward = (
            reward_config.alive_bonus
            - reward_config.attitude_weight * attitude_error.square()
            - reward_config.angular_rate_weight * rate_cost
            - reward_config.action_rate_weight * action_rate_cost
            - reward_config.saturation_weight * saturation_cost
        )
        tilt = tilt_angle(attitude)
        rate_norm = angular_velocity.norm(dim=-1, keepdim=True)
        terminated = (tilt > self.config.terminate_tilt_rad) | (rate_norm > self.config.terminate_angular_rate_rad_s)
        return TaskTransition(
            reward=reward,
            terminated=terminated,
            info={
                "reward.attitude": -reward_config.attitude_weight * attitude_error.square(),
                "reward.angular_rate": -reward_config.angular_rate_weight * rate_cost,
                "reward.action_rate": -reward_config.action_rate_weight * action_rate_cost,
                "reward.saturation": -reward_config.saturation_weight * saturation_cost,
                "attitude_error_rad": attitude_error,
                "tilt_rad": tilt,
                "angular_rate_norm": rate_norm,
            },
        )

