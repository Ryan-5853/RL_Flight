from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from tensordict import TensorDict, TensorDictBase

from .config import TaskConfig
from .math import quaternion_geodesic_angle, quaternion_to_euler, tilt_angle
from .rewards import AttitudeRewardCalculator, RewardCalculator


@dataclass(frozen=True)
class TaskTransition:
    """任务层对一次状态转移的批量评价结果。"""

    reward: torch.Tensor
    terminated: torch.Tensor
    valid: torch.Tensor
    info: Mapping[str, torch.Tensor]


class AttitudeTrackingTask:
    """姿态跟踪任务的批量奖励和终止条件内核。

    目标姿态由独立 VirtualPilot command source 提供；Task 不再拥有命令 RNG，
    避免奖励任务和飞手逻辑同时修改目标。
    """

    def __init__(
        self,
        config: TaskConfig,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        reward_calculator: RewardCalculator | None = None,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.reward_calculator = reward_calculator or AttitudeRewardCalculator()

    def transition(
        self,
        attitude: torch.Tensor,
        angular_velocity: torch.Tensor,
        target_attitude: torch.Tensor,
        standard_action: torch.Tensor,
        previous_action: torch.Tensor,
        episode_step: torch.Tensor | None = None,
        max_episode_steps: int | None = None,
        env_context: TensorDictBase | None = None,
        desired_yaw_rate: torch.Tensor | None = None,
    ) -> TaskTransition:
        """批量计算奖励分量和安全终止条件。

        输入的首维均为并行环境维 ``B``；奖励与终止标志保持 ``[B,1]``，
        便于直接写入 TorchRL 的 ``("next", ...)`` 转移字段。
        """

        # 兼容 reward 单元测试/第三方调用的旧第六参数 env_context。
        if isinstance(episode_step, TensorDictBase):
            env_context = episode_step
            episode_step = None
        if episode_step is None:
            episode_step = torch.zeros(
                self.batch_size, device=self.device, dtype=torch.int64
            )
        if max_episode_steps is None:
            max_episode_steps = max(1, int(self.config.episode_duration_s * 500.0))
        attitude_error = quaternion_geodesic_angle(attitude, target_attitude)
        current_euler = quaternion_to_euler(attitude)
        target_euler = quaternion_to_euler(target_attitude)
        roll_pitch_error = torch.atan2(
            torch.sin(current_euler[:, :2] - target_euler[:, :2]),
            torch.cos(current_euler[:, :2] - target_euler[:, :2]),
        )
        if desired_yaw_rate is None:
            desired_yaw_rate = torch.zeros(
                self.batch_size, 1, device=self.device, dtype=self.dtype
            )
        yaw_rate_error = angular_velocity[:, 2:3] - desired_yaw_rate
        tilt = tilt_angle(attitude)
        termination_rate = (
            angular_velocity[:, :2]
            if self.config.terminate_angular_rate_axes == "roll_pitch"
            else angular_velocity
        )
        rate_norm = termination_rate.norm(dim=-1, keepdim=True)
        terminated = (tilt > self.config.terminate_tilt_rad) | (rate_norm > self.config.terminate_angular_rate_rad_s)
        episode_age_fraction = (
            (episode_step[:, None].to(self.dtype) + 1.0) / float(max_episode_steps)
        ).clamp(0.0, 1.0)
        context = TensorDict(
            {
                "attitude_geodesic_rad": attitude_error,
                "roll_pitch_error_rad": roll_pitch_error,
                "yaw_rate_error_rad_s": yaw_rate_error,
                "tilt_rad": tilt,
                "attitude_q_wb": attitude,
                "target_attitude_q_wb": target_attitude,
                "angular_velocity_b": angular_velocity,
                "action": standard_action,
                "previous_action": previous_action,
                "terminated": terminated,
                "tilt_ratio": tilt / self.config.terminate_tilt_rad,
                "rate_ratio": rate_norm / self.config.terminate_angular_rate_rad_s,
                "episode_age_fraction": episode_age_fraction,
                "episode_remaining_fraction": 1.0 - episode_age_fraction,
            },
            batch_size=[self.batch_size],
            device=self.device,
        )
        if env_context is not None:
            context.update(env_context)
        reward_output = self.reward_calculator(context)
        if (
            reward_output.reward.shape != (self.batch_size, 1)
            or reward_output.reward.device != self.device
            or reward_output.reward.dtype != self.dtype
        ):
            raise ValueError(
                "RewardCalculator.reward must be a [B, 1] tensor on the task device/dtype"
            )
        reward_valid = reward_output.valid
        if reward_valid is None:
            reward_valid = torch.ones_like(terminated)
        if reward_valid.shape != (self.batch_size, 1) or reward_valid.dtype != torch.bool:
            raise ValueError("RewardCalculator.valid must be a bool tensor with shape [B, 1]")
        reward_valid = reward_valid & torch.isfinite(reward_output.reward)
        safe_reward = torch.where(
            reward_valid, reward_output.reward, torch.zeros_like(reward_output.reward)
        )
        info = dict(reward_output.terms.items())
        if reward_output.diagnostics is not None:
            diagnostics = reward_output.diagnostics
            if (
                diagnostics.batch_size != torch.Size([self.batch_size])
                or diagnostics.device != self.device
            ):
                raise ValueError(
                    "RewardCalculator.diagnostics must use the task batch/device"
                )
            info.update(dict(diagnostics.items()))
        info.update(
            {
                "attitude_error_rad": attitude_error,
                "roll_pitch_error_rad": roll_pitch_error,
                "yaw_rate_error_rad_s": yaw_rate_error,
                "tilt_rad": tilt,
                "angular_rate_norm": rate_norm,
            }
        )
        return TaskTransition(
            reward=safe_reward,
            terminated=terminated,
            valid=reward_valid,
            info=info,
        )
