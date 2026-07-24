from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from tensordict import TensorDict

from .config import VirtualPilotConfig
from .core import assert_tensor_on
from .math import euler_to_quaternion


@dataclass(frozen=True)
class VirtualPilotSnapshot:
    """当前控制周期使用的批量虚拟飞手命令。"""

    upper_throttle: torch.Tensor
    target_attitude_q_wb: torch.Tensor
    stick_target: torch.Tensor
    filtered_stick: torch.Tensor
    throttle_target: torch.Tensor
    height_target: torch.Tensor
    height_error: torch.Tensor


class VirtualPilotCommandSource:
    """GPU 批量虚拟飞手：开环上桨油门与自稳模式三轴摇杆命令。

    roll/pitch 摇杆表示目标角，yaw 摇杆表示目标角速度。每个并行实例独立
    保持随机目标一段时间，滤波后的摇杆按一阶惯性接近目标。高度真值只供
    外部飞手的增量式 PI 更新油门，不进入策略观测或姿态奖励。
    """

    version = 2

    def __init__(
        self,
        config: VirtualPilotConfig,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        control_hz: int,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.dt = 1.0 / float(control_hz)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(config.seed)

        self.upper_throttle = torch.full(
            (batch_size, 1), config.throttle_minimum, device=device, dtype=dtype
        )
        self.throttle_target = self.upper_throttle.clone()
        self.spool_remaining = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        self.spool_throttle = self.upper_throttle.clone()
        self.height_controller_output = self.upper_throttle.clone()
        self.height_previous_error = torch.zeros_like(self.upper_throttle)
        self.height_error = torch.zeros_like(self.upper_throttle)
        self.height_target = torch.full_like(
            self.upper_throttle, config.height_target_m
        )

        self.stick_target = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        self.filtered_stick = torch.zeros_like(self.stick_target)
        self.hold_remaining = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        self.target_yaw = torch.zeros(batch_size, device=device, dtype=dtype)
        self.target_attitude = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        self.target_attitude[:, 0] = 1.0
        self.curriculum_scale = config.initial_target_scale

        tau = torch.tensor(
            [config.roll_time_constant_s, config.pitch_time_constant_s, config.yaw_time_constant_s],
            device=device,
            dtype=dtype,
        )
        self._filter_alpha = 1.0 - torch.exp(-torch.tensor(self.dt, device=device, dtype=dtype) / tau)

    def reset(self, mask: torch.Tensor) -> None:
        """仅重置 mask 指定的飞手状态，并为其采样新的独立命令段。"""

        assert_tensor_on(
            mask,
            device=self.device,
            dtype=torch.bool,
            shape=(self.batch_size,),
            name="virtual pilot reset mask",
        )
        expanded = mask[:, None]
        initial = torch.full_like(self.upper_throttle, self.config.throttle_minimum)
        self.upper_throttle = torch.where(expanded, initial, self.upper_throttle)
        self.spool_remaining = torch.where(
            expanded,
            torch.full_like(self.spool_remaining, self.config.spool_duration_s),
            self.spool_remaining,
        )

        self.spool_throttle = self._replace_sampled_range(
            self.spool_throttle, mask, self.config.spool_target_range
        )
        self.height_controller_output = self._replace_sampled_range(
            self.height_controller_output,
            mask,
            self.config.height_initial_throttle_range,
        )
        self.height_previous_error = torch.where(
            expanded,
            torch.zeros_like(self.height_previous_error),
            self.height_previous_error,
        )
        self.height_error = torch.where(
            expanded, torch.zeros_like(self.height_error), self.height_error
        )
        self.height_target = torch.where(
            expanded,
            torch.full_like(self.height_target, self.config.height_target_m),
            self.height_target,
        )
        self.throttle_target = torch.where(expanded, self.spool_throttle, self.throttle_target)

        sampled_stick = self._sample_stick_target(self.curriculum_scale)
        self.stick_target = torch.where(expanded, sampled_stick, self.stick_target)
        self.filtered_stick = torch.where(
            expanded, torch.zeros_like(self.filtered_stick), self.filtered_stick
        )
        sampled_hold = self._sample_uniform(*self.config.hold_duration_range_s)
        self.hold_remaining = torch.where(expanded, sampled_hold, self.hold_remaining)
        self.target_yaw = torch.where(mask, torch.zeros_like(self.target_yaw), self.target_yaw)
        self._refresh_target_attitude()

    @torch.no_grad()
    def step(self, height_m: torch.Tensor, active_mask: torch.Tensor | None = None) -> None:
        """根据当前高度推进一个 500 Hz 飞手控制周期。"""

        assert_tensor_on(
            height_m,
            device=self.device,
            dtype=self.dtype,
            shape=(self.batch_size, 1),
            name="height_m",
        )
        if active_mask is None:
            active_mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        assert_tensor_on(
            active_mask,
            device=self.device,
            dtype=torch.bool,
            shape=(self.batch_size,),
            name="virtual pilot active mask",
        )
        active = active_mask[:, None]

        next_hold = self.hold_remaining - self.dt
        change = active_mask & (next_hold[:, 0] <= 0.0)
        sampled_target = self._sample_stick_target(self.curriculum_scale)
        self.stick_target = torch.where(change[:, None], sampled_target, self.stick_target)
        sampled_hold = self._sample_uniform(*self.config.hold_duration_range_s)
        next_hold = torch.where(change[:, None], sampled_hold, next_hold)
        self.hold_remaining = torch.where(active, next_hold, self.hold_remaining)

        filtered = self.filtered_stick + self._filter_alpha * (
            self.stick_target - self.filtered_stick
        )
        self.filtered_stick = torch.where(active, filtered, self.filtered_stick)
        yaw_next = self.target_yaw + (
            self.filtered_stick[:, 2] * self.config.max_yaw_rate_rad_s * self.dt
        )
        yaw_next = torch.remainder(yaw_next + torch.pi, 2.0 * torch.pi) - torch.pi
        self.target_yaw = torch.where(active_mask, yaw_next, self.target_yaw)

        # 增量式 PI：u[k] = u[k-1] + Kp(e[k]-e[k-1]) + Ki*e[k]*dt。
        # 误差和输出均限幅，且没有独立积分累加器，输出饱和时不会继续 windup。
        height_error = (self.height_target - height_m).clamp(
            -self.config.height_error_limit_m,
            self.config.height_error_limit_m,
        )
        in_spool = self.spool_remaining > 0.0
        controller_active = active & ~in_spool
        increment = (
            self.config.height_proportional_gain
            * (height_error - self.height_previous_error)
            + self.config.height_integral_gain * height_error * self.dt
        )
        controller_output = (self.height_controller_output + increment).clamp(
            self.config.throttle_minimum, self.config.throttle_maximum
        )
        self.height_controller_output = torch.where(
            controller_active, controller_output, self.height_controller_output
        )
        self.height_previous_error = torch.where(
            controller_active, height_error, self.height_previous_error
        )
        self.height_error = torch.where(active, height_error, self.height_error)
        target = torch.where(
            in_spool, self.spool_throttle, self.height_controller_output
        )
        self.throttle_target = torch.where(active, target, self.throttle_target)
        self.spool_remaining = torch.where(
            active,
            torch.clamp(self.spool_remaining - self.dt, min=0.0),
            self.spool_remaining,
        )

        lower = self.upper_throttle - self.config.throttle_fall_rate_per_s * self.dt
        upper = self.upper_throttle + self.config.throttle_rise_rate_per_s * self.dt
        throttle = torch.maximum(torch.minimum(self.throttle_target, upper), lower)
        throttle = throttle.clamp(self.config.throttle_minimum, self.config.throttle_maximum)
        self.upper_throttle = torch.where(active, throttle, self.upper_throttle)
        self._refresh_target_attitude()

    def snapshot(self) -> VirtualPilotSnapshot:
        return VirtualPilotSnapshot(
            upper_throttle=self.upper_throttle,
            target_attitude_q_wb=self.target_attitude,
            stick_target=self.stick_target,
            filtered_stick=self.filtered_stick,
            throttle_target=self.throttle_target,
            height_target=self.height_target,
            height_error=self.height_error,
        )

    @property
    def desired_yaw_rate(self) -> torch.Tensor:
        """飞手滤波后的偏航角速度指令，形状为 ``[B,1]``。"""

        return self.filtered_stick[:, 2:3] * self.config.max_yaw_rate_rad_s

    def set_curriculum_scale(self, scale: float) -> None:
        """由 episode 课程直接设置命令幅度比例。"""

        self.curriculum_scale = min(max(float(scale), 0.0), 1.0)

    def info(self) -> TensorDict:
        return TensorDict(
            {
                "pilot.upper_throttle": self.upper_throttle,
                "pilot.throttle_target": self.throttle_target,
                "pilot.height_target_m": self.height_target,
                "pilot.height_error_m": self.height_error,
                "pilot.height_controller_output": self.height_controller_output,
                "pilot.stick_target": self.stick_target,
                "pilot.stick_filtered": self.filtered_stick,
                "pilot.target_attitude_q_wb": self.target_attitude,
            },
            batch_size=[self.batch_size],
            device=self.device,
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "version": self.version,
            "generator_state": self.generator.get_state(),
            "upper_throttle": self.upper_throttle,
            "throttle_target": self.throttle_target,
            "spool_remaining": self.spool_remaining,
            "spool_throttle": self.spool_throttle,
            "height_controller_output": self.height_controller_output,
            "height_previous_error": self.height_previous_error,
            "height_error": self.height_error,
            "height_target": self.height_target,
            "stick_target": self.stick_target,
            "filtered_stick": self.filtered_stick,
            "hold_remaining": self.hold_remaining,
            "target_yaw": self.target_yaw,
            "target_attitude": self.target_attitude,
            "curriculum_scale": self.curriculum_scale,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", -1)) != self.version:
            raise ValueError("incompatible virtual pilot state")
        self.generator.set_state(state["generator_state"])
        self.curriculum_scale = float(
            state.get("curriculum_scale", self.config.initial_target_scale)
        )
        for name in (
            "upper_throttle", "throttle_target", "spool_remaining", "spool_throttle",
            "height_controller_output", "height_previous_error", "height_error",
            "height_target",
            "stick_target", "filtered_stick", "hold_remaining", "target_yaw",
            "target_attitude",
        ):
            setattr(self, name, state[name].to(self.device))

    def _refresh_target_attitude(self) -> None:
        self.target_attitude = euler_to_quaternion(
            self.filtered_stick[:, 0], self.filtered_stick[:, 1], self.target_yaw
        )

    def _sample_stick_target(self, scale: float) -> torch.Tensor:
        radial = torch.rand(
            self.batch_size, device=self.device, dtype=self.dtype, generator=self.generator
        ).pow(self.config.center_exponent)
        azimuth = torch.rand(
            self.batch_size, device=self.device, dtype=self.dtype, generator=self.generator
        ) * (2.0 * torch.pi)
        roll = radial * torch.cos(azimuth) * self.config.max_roll_rad * scale
        pitch = radial * torch.sin(azimuth) * self.config.max_pitch_rad * scale
        raw_yaw = torch.rand(
            self.batch_size, device=self.device, dtype=self.dtype, generator=self.generator
        ) * 2.0 - 1.0
        yaw = raw_yaw.sign() * raw_yaw.abs().pow(self.config.center_exponent) * scale
        return torch.stack((roll, pitch, yaw), dim=-1)

    def _sample_uniform(self, low: float, high: float) -> torch.Tensor:
        random = torch.rand(
            self.batch_size, 1, device=self.device, dtype=self.dtype, generator=self.generator
        )
        return low + (high - low) * random

    def _replace_sampled_range(
        self, current: torch.Tensor, mask: torch.Tensor, bounds: tuple[float, float]
    ) -> torch.Tensor:
        sampled = self._sample_uniform(*bounds)
        return torch.where(mask[:, None], sampled, current)
