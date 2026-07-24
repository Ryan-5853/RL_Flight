from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import torch
from tensordict import TensorDict, TensorDictBase


@dataclass(frozen=True)
class RewardOutput:
    """奖励计算器的批量输出，所有张量必须与输入 context 同设备。"""

    reward: torch.Tensor
    terms: TensorDictBase
    valid: torch.Tensor | None = None


class RewardCalculator(Protocol):
    """可注入的 GPU 张量奖励接口。"""

    def __call__(self, context: TensorDictBase) -> RewardOutput: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class AttitudeRewardCalculator:
    """姿态自稳奖励 v3：生存优先、分轴跟踪与提前安全 barrier。"""

    version = 3

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        params = params or {}
        # 这些参数只是默认实现的构造参数，不再是 Task 的固定接口。
        self.roll_pitch_weight = float(
            params.get("roll_pitch_weight", params.get("attitude_weight", 2.0))
        )
        self.tilt_weight = float(params.get("tilt_weight", 1.0))
        self.yaw_rate_weight = float(params.get("yaw_rate_weight", 0.05))
        self.angular_rate_weight = float(params.get("angular_rate_weight", 0.05))
        self.action_rate_weight = float(params.get("action_rate_weight", 0.01))
        self.saturation_weight = float(params.get("saturation_weight", 0.02))
        self.alive_bonus = float(params.get("alive_bonus", 0.0))
        self.survival_progress_weight = float(
            params.get("survival_progress_weight", 0.0)
        )
        self.tilt_barrier_weight = float(params.get("tilt_barrier_weight", 0.0))
        self.rate_barrier_weight = float(params.get("rate_barrier_weight", 0.0))
        self.barrier_start_fraction = float(
            params.get("barrier_start_fraction", 0.6)
        )
        self.barrier_exponent = float(params.get("barrier_exponent", 4.0))
        self.termination_penalty = float(params.get("termination_penalty", 0.0))
        self.early_termination_penalty = float(
            params.get("early_termination_penalty", 0.0)
        )

    def __call__(self, context: TensorDictBase) -> RewardOutput:
        legacy_attitude_error = context.get(
            "attitude_geodesic_rad",
            torch.zeros(
                (*context.batch_size, 1),
                device=context.device,
                dtype=context["action"].dtype,
            ),
        )
        roll_pitch_error = context.get(
            "roll_pitch_error_rad",
            torch.cat(
                (legacy_attitude_error, torch.zeros_like(legacy_attitude_error)),
                dim=-1,
            ),
        )
        yaw_rate_error = context.get(
            "yaw_rate_error_rad_s",
            torch.zeros_like(legacy_attitude_error),
        )
        tilt = context.get("tilt_rad", legacy_attitude_error)
        angular_velocity = context["angular_velocity_b"]
        action = context["action"]
        previous_action = context.get("previous_action", torch.zeros_like(action))
        terminated = context.get(
            "terminated", torch.zeros_like(tilt, dtype=torch.bool)
        )
        tilt_ratio = context.get("tilt_ratio", torch.zeros_like(tilt))
        rate_ratio = context.get("rate_ratio", torch.zeros_like(tilt))
        episode_age_fraction = context.get(
            "episode_age_fraction", torch.zeros_like(tilt)
        ).clamp(0.0, 1.0)
        remaining_fraction = context.get(
            "episode_remaining_fraction", 1.0 - episode_age_fraction
        ).clamp(0.0, 1.0)
        rate_cost = angular_velocity[:, :2].square().sum(dim=-1, keepdim=True)
        action_rate_cost = (action - previous_action).square().sum(dim=-1, keepdim=True)
        saturation_cost = torch.relu(action.abs() - 0.95).square().sum(dim=-1, keepdim=True)
        reward_attitude = -self.roll_pitch_weight * roll_pitch_error.square().sum(
            dim=-1, keepdim=True
        )
        reward_tilt = -self.tilt_weight * tilt.square()
        reward_yaw_rate = -self.yaw_rate_weight * yaw_rate_error.square()
        reward_rate = -self.angular_rate_weight * rate_cost
        reward_action_rate = -self.action_rate_weight * action_rate_cost
        reward_saturation = -self.saturation_weight * saturation_cost
        barrier_scale = max(1.0 - self.barrier_start_fraction, 1e-6)
        tilt_risk = (
            torch.relu(tilt_ratio - self.barrier_start_fraction) / barrier_scale
        ).pow(self.barrier_exponent)
        rate_risk = (
            torch.relu(rate_ratio - self.barrier_start_fraction) / barrier_scale
        ).pow(self.barrier_exponent)
        reward_risk = (
            -self.tilt_barrier_weight * tilt_risk
            - self.rate_barrier_weight * rate_risk
        )
        reward_survival = self.survival_progress_weight * episode_age_fraction
        termination_cost = terminated.to(action.dtype) * (
            self.termination_penalty
            + self.early_termination_penalty * remaining_fraction
        )
        reward = (
            self.alive_bonus + reward_attitude + reward_tilt + reward_yaw_rate
            + reward_rate + reward_action_rate
            + reward_saturation + reward_risk + reward_survival - termination_cost
        )
        terms = TensorDict(
            {
                "reward.attitude": reward_attitude,
                "reward.tilt": reward_tilt,
                "reward.yaw_rate": reward_yaw_rate,
                "reward.angular_rate": reward_rate,
                "reward.action_rate": reward_action_rate,
                "reward.saturation": reward_saturation,
                "reward.risk": reward_risk,
                "reward.survival": reward_survival,
                "reward.termination": -termination_cost,
            },
            batch_size=context.batch_size,
            device=context.device,
        )
        if reward.shape != (context.batch_size[0], 1):
            raise ValueError("RewardCalculator must return reward with shape [B, 1]")
        return RewardOutput(reward=reward, terms=terms, valid=torch.isfinite(reward))

    def state_dict(self) -> Mapping[str, Any]:
        return {"version": self.version}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", self.version)) != self.version:
            raise ValueError("incompatible reward calculator state")
