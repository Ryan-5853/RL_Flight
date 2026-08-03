from __future__ import annotations

import math
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
    diagnostics: TensorDictBase | None = None


class RewardCalculator(Protocol):
    """可注入的 GPU 张量奖励接口。"""

    def __call__(self, context: TensorDictBase) -> RewardOutput: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class AttitudeRewardCalculator:
    """姿态自稳奖励 v3：生存优先、可选联合跟踪与安全 barrier。"""

    version = 3

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        params = params or {}
        # 这些参数只是默认实现的构造参数，不再是 Task 的固定接口。
        self.roll_pitch_weight = float(
            params.get("roll_pitch_weight", params.get("attitude_weight", 2.0))
        )
        roll_pitch_cost_cap_rad = params.get("roll_pitch_cost_cap_rad")
        self.roll_pitch_cost_cap_rad = (
            None if roll_pitch_cost_cap_rad is None else float(roll_pitch_cost_cap_rad)
        )
        if self.roll_pitch_cost_cap_rad is not None and (
            not math.isfinite(self.roll_pitch_cost_cap_rad)
            or self.roll_pitch_cost_cap_rad <= 0
        ):
            raise ValueError("roll_pitch_cost_cap_rad must be positive when configured")
        roll_pitch_huber_delta_rad = params.get(
            "roll_pitch_huber_delta_rad"
        )
        self.roll_pitch_huber_delta_rad = (
            None
            if roll_pitch_huber_delta_rad is None
            else float(roll_pitch_huber_delta_rad)
        )
        if self.roll_pitch_huber_delta_rad is not None and (
            not math.isfinite(self.roll_pitch_huber_delta_rad)
            or self.roll_pitch_huber_delta_rad <= 0
        ):
            raise ValueError(
                "roll_pitch_huber_delta_rad must be positive when configured"
            )
        if (
            self.roll_pitch_cost_cap_rad is not None
            and self.roll_pitch_huber_delta_rad is not None
        ):
            raise ValueError(
                "roll_pitch_cost_cap_rad and roll_pitch_huber_delta_rad "
                "are mutually exclusive"
            )
        self.tilt_weight = float(params.get("tilt_weight", 1.0))
        self.yaw_rate_weight = float(params.get("yaw_rate_weight", 0.05))
        yaw_rate_huber_delta_rad_s = params.get("yaw_rate_huber_delta_rad_s")
        self.yaw_rate_huber_delta_rad_s = (
            None
            if yaw_rate_huber_delta_rad_s is None
            else float(yaw_rate_huber_delta_rad_s)
        )
        if self.yaw_rate_huber_delta_rad_s is not None and (
            not math.isfinite(self.yaw_rate_huber_delta_rad_s)
            or self.yaw_rate_huber_delta_rad_s <= 0
        ):
            raise ValueError(
                "yaw_rate_huber_delta_rad_s must be positive when configured"
            )
        yaw_rate_cost_cap = params.get("yaw_rate_cost_cap")
        self.yaw_rate_cost_cap = (
            None if yaw_rate_cost_cap is None else float(yaw_rate_cost_cap)
        )
        if self.yaw_rate_cost_cap is not None and (
            not math.isfinite(self.yaw_rate_cost_cap)
            or self.yaw_rate_cost_cap <= 0
        ):
            raise ValueError("yaw_rate_cost_cap must be positive when configured")
        self.angular_rate_weight = float(params.get("angular_rate_weight", 0.05))
        self.joint_tracking_weight = float(
            params.get("joint_tracking_weight", 0.0)
        )
        self.joint_tracking_mean_weight = float(
            params.get("joint_tracking_mean_weight", 0.0)
        )
        self.joint_roll_pitch_scale_rad = float(
            params.get(
                "joint_roll_pitch_scale_rad",
                math.radians(10.0),
            )
        )
        self.joint_yaw_rate_scale_rad_s = float(
            params.get("joint_yaw_rate_scale_rad_s", 1.0)
        )
        self.joint_tracking_huber_delta = float(
            params.get("joint_tracking_huber_delta", 1.0)
        )
        if (
            not math.isfinite(self.joint_tracking_weight)
            or self.joint_tracking_weight < 0
        ):
            raise ValueError("joint_tracking_weight must be finite and nonnegative")
        if (
            not math.isfinite(self.joint_tracking_mean_weight)
            or self.joint_tracking_mean_weight < 0
        ):
            raise ValueError(
                "joint_tracking_mean_weight must be finite and nonnegative"
            )
        for name, value in (
            (
                "joint_roll_pitch_scale_rad",
                self.joint_roll_pitch_scale_rad,
            ),
            (
                "joint_yaw_rate_scale_rad_s",
                self.joint_yaw_rate_scale_rad_s,
            ),
            (
                "joint_tracking_huber_delta",
                self.joint_tracking_huber_delta,
            ),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.joint_tracking_weight > 0
            or self.joint_tracking_mean_weight > 0
        ) and (
            self.roll_pitch_weight != 0
            or self.tilt_weight != 0
            or self.yaw_rate_weight != 0
        ):
            raise ValueError(
                "joint tracking is mutually exclusive with roll_pitch_weight, "
                "tilt_weight, and yaw_rate_weight"
            )
        self.action_rate_weight = float(params.get("action_rate_weight", 0.01))
        self.servo_cyclic_weight = float(
            params.get("servo_cyclic_weight", 0.0)
        )
        self.motor_effort_weight = float(
            params.get("motor_effort_weight", 0.0)
        )
        self.servo_common_effort_weight = float(
            params.get("servo_common_effort_weight", 0.0)
        )
        self.servo_cyclic_effort_weight = float(
            params.get("servo_cyclic_effort_weight", 0.0)
        )
        self.physical_servo_common_effort_weight = float(
            params.get("physical_servo_common_effort_weight", 0.0)
        )
        self.motor_movement_weight = float(
            params.get("motor_movement_weight", 0.0)
        )
        self.servo_common_movement_weight = float(
            params.get("servo_common_movement_weight", 0.0)
        )
        self.servo_cyclic_movement_weight = float(
            params.get("servo_cyclic_movement_weight", 0.0)
        )
        movement_gate_roll_pitch_scale = params.get(
            "servo_cyclic_movement_gate_roll_pitch_scale_rad"
        )
        movement_gate_yaw_rate_scale = params.get(
            "servo_cyclic_movement_gate_yaw_rate_scale_rad_s"
        )
        if (movement_gate_roll_pitch_scale is None) != (
            movement_gate_yaw_rate_scale is None
        ):
            raise ValueError(
                "servo cyclic movement gate roll/pitch and yaw scales must be "
                "configured together"
            )
        self.servo_cyclic_movement_gate_roll_pitch_scale_rad = (
            None
            if movement_gate_roll_pitch_scale is None
            else float(movement_gate_roll_pitch_scale)
        )
        self.servo_cyclic_movement_gate_yaw_rate_scale_rad_s = (
            None
            if movement_gate_yaw_rate_scale is None
            else float(movement_gate_yaw_rate_scale)
        )
        self.servo_cyclic_movement_gate_minimum = float(
            params.get("servo_cyclic_movement_gate_minimum", 0.0)
        )
        for name, value in (
            (
                "servo_cyclic_movement_gate_roll_pitch_scale_rad",
                self.servo_cyclic_movement_gate_roll_pitch_scale_rad,
            ),
            (
                "servo_cyclic_movement_gate_yaw_rate_scale_rad_s",
                self.servo_cyclic_movement_gate_yaw_rate_scale_rad_s,
            ),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.servo_cyclic_movement_gate_minimum)
            or not 0.0 <= self.servo_cyclic_movement_gate_minimum <= 1.0
        ):
            raise ValueError(
                "servo_cyclic_movement_gate_minimum must be between 0 and 1"
            )
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
        for name, value in (
            ("action_rate_weight", self.action_rate_weight),
            ("servo_cyclic_weight", self.servo_cyclic_weight),
            ("motor_effort_weight", self.motor_effort_weight),
            (
                "servo_common_effort_weight",
                self.servo_common_effort_weight,
            ),
            (
                "servo_cyclic_effort_weight",
                self.servo_cyclic_effort_weight,
            ),
            (
                "physical_servo_common_effort_weight",
                self.physical_servo_common_effort_weight,
            ),
            ("motor_movement_weight", self.motor_movement_weight),
            (
                "servo_common_movement_weight",
                self.servo_common_movement_weight,
            ),
            (
                "servo_cyclic_movement_weight",
                self.servo_cyclic_movement_weight,
            ),
            ("saturation_weight", self.saturation_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

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
        servo_action = action[:, 1:4]
        servo_common = servo_action.mean(dim=-1, keepdim=True)
        # 三个舵面按 120 度布置：共同模态主要承担偏航/反扭矩，去均值后的
        # cyclic 模态主要承担 roll/pitch。只惩罚 cyclic 幅值，避免为了消除
        # 姿态极限环而破坏必要的稳态偏航配平。
        servo_cyclic_cost = (
            servo_action - servo_common
        ).square().sum(dim=-1, keepdim=True)
        motor_effort_cost = action[:, :1].square()
        servo_common_effort_cost = servo_common.square()
        servo_cyclic_effort_cost = servo_cyclic_cost
        actuator_effort_proxy = (
            motor_effort_cost
            + servo_common_effort_cost
            + servo_cyclic_effort_cost
        )
        simulator_command = context.get("simulator_command")
        if simulator_command is None:
            if self.physical_servo_common_effort_weight > 0:
                raise ValueError(
                    "physical servo common effort requires simulator_command"
                )
            physical_servo_common_effort_cost = torch.zeros_like(
                servo_common_effort_cost
            )
        else:
            if simulator_command.ndim != 2 or simulator_command.shape[1] != 5:
                raise ValueError("simulator_command must have shape [B, 5]")
            physical_servo_common_effort_cost = (
                simulator_command[:, 2:5].mean(dim=-1, keepdim=True).square()
            )
        action_delta = action - previous_action
        motor_movement_cost = action_delta[:, :1].abs()
        servo_delta = action_delta[:, 1:4]
        servo_common_delta = servo_delta.mean(dim=-1, keepdim=True)
        # L1 total variation 对低频、小步长的持续往返运动保持一阶敏感；
        # 与逐步平方差相比，不会因 500 Hz 控制周期而被 dt² 过度缩小。
        # 共同模态和 cyclic 模态分开计价，避免 roll/pitch 稳定性目标无意中
        # 抑制必要的偏航/反扭矩共同模态。
        servo_common_movement_cost = servo_common_delta.abs()
        servo_cyclic_movement_cost = (
            servo_delta - servo_common_delta
        ).abs().sum(dim=-1, keepdim=True)
        # Weight-independent normalized actuator travel. This is exposed as a
        # diagnostic so runs remain comparable when reward weights change.
        actuator_energy_proxy = (
            motor_movement_cost
            + servo_common_movement_cost
            + servo_cyclic_movement_cost
        )
        saturation_cost = torch.relu(action.abs() - 0.95).square().sum(dim=-1, keepdim=True)
        roll_pitch_cost = roll_pitch_error.square().sum(dim=-1, keepdim=True)
        if self.roll_pitch_cost_cap_rad is not None:
            roll_pitch_cost = roll_pitch_cost.clamp_max(
                self.roll_pitch_cost_cap_rad**2
            )
        elif self.roll_pitch_huber_delta_rad is not None:
            roll_pitch_norm = roll_pitch_cost.sqrt()
            delta = self.roll_pitch_huber_delta_rad
            roll_pitch_cost = torch.where(
                roll_pitch_norm <= delta,
                roll_pitch_cost,
                2.0 * delta * roll_pitch_norm - delta**2,
            )
        reward_attitude = -self.roll_pitch_weight * roll_pitch_cost
        reward_tilt = -self.tilt_weight * tilt.square()
        yaw_rate_cost = yaw_rate_error.square()
        if self.yaw_rate_huber_delta_rad_s is not None:
            yaw_rate_abs = yaw_rate_error.abs()
            delta = self.yaw_rate_huber_delta_rad_s
            yaw_rate_cost = torch.where(
                yaw_rate_abs <= delta,
                yaw_rate_cost,
                2.0 * delta * yaw_rate_abs - delta**2,
            )
        if self.yaw_rate_cost_cap is not None:
            # 有理饱和保持 cost < cap，同时在高 yaw 区间保留非零梯度。
            yaw_rate_cost = (
                self.yaw_rate_cost_cap
                * yaw_rate_cost
                / (self.yaw_rate_cost_cap + yaw_rate_cost)
            )
        reward_yaw_rate = -self.yaw_rate_weight * yaw_rate_cost
        normalized_roll_pitch = (
            torch.linalg.vector_norm(
                roll_pitch_error,
                dim=-1,
                keepdim=True,
            )
            / self.joint_roll_pitch_scale_rad
        )
        normalized_yaw_rate = (
            yaw_rate_error.abs() / self.joint_yaw_rate_scale_rad_s
        )
        delta = self.joint_tracking_huber_delta
        joint_roll_pitch_cost = torch.where(
            normalized_roll_pitch <= delta,
            normalized_roll_pitch.square(),
            2.0 * delta * normalized_roll_pitch - delta**2,
        )
        joint_yaw_rate_cost = torch.where(
            normalized_yaw_rate <= delta,
            normalized_yaw_rate.square(),
            2.0 * delta * normalized_yaw_rate - delta**2,
        )
        # 最差轴（Chebyshev）聚合是非补偿式标量化：已经较好的轴继续变好
        # 不会掩盖另一轴的坏结果，也不会给“牺牲好轴换取坏轴收益”提供奖励。
        # 在两项不相等时，主奖励变化完全由当前较差的一项决定。
        reward_joint_tracking_worst = (
            -self.joint_tracking_weight
            * torch.maximum(
                joint_roll_pitch_cost,
                joint_yaw_rate_cost,
            )
        )
        # 小权重均值辅助项让非主导轴也持续获得梯度，避免纯 worst-axis
        # 优化在 roll/pitch 与 yaw 之间反复交换能力。主项仍由最差轴决定，
        # 因而一个轴特别好时不能完全补偿另一个轴的坏结果。
        reward_joint_tracking_mean = (
            -self.joint_tracking_mean_weight
            * 0.5
            * (joint_roll_pitch_cost + joint_yaw_rate_cost)
        )
        reward_joint_tracking = (
            reward_joint_tracking_worst + reward_joint_tracking_mean
        )
        # 主导轴比例用于判断 worst-axis 奖励是否在两个目标之间正常切换。
        # 完全相等时 torch.maximum 会在两边分配次梯度，诊断也各记 0.5，
        # 从而保证两个 dominance 指标逐样本之和恒为 1。
        joint_cost_tie = joint_roll_pitch_cost == joint_yaw_rate_cost
        joint_roll_pitch_dominant = (
            (joint_roll_pitch_cost > joint_yaw_rate_cost).to(action.dtype)
            + 0.5 * joint_cost_tie.to(action.dtype)
        )
        joint_yaw_rate_dominant = 1.0 - joint_roll_pitch_dominant
        reward_rate = -self.angular_rate_weight * rate_cost
        reward_action_rate = -self.action_rate_weight * action_rate_cost
        reward_servo_cyclic = -self.servo_cyclic_weight * servo_cyclic_cost
        reward_motor_effort = -self.motor_effort_weight * motor_effort_cost
        reward_servo_common_effort = (
            -self.servo_common_effort_weight * servo_common_effort_cost
        )
        reward_servo_cyclic_effort = (
            -self.servo_cyclic_effort_weight * servo_cyclic_effort_cost
        )
        reward_physical_servo_common_effort = (
            -self.physical_servo_common_effort_weight
            * physical_servo_common_effort_cost
        )
        reward_control_effort = (
            reward_motor_effort
            + reward_servo_common_effort
            + reward_servo_cyclic_effort
            + reward_physical_servo_common_effort
        )
        reward_motor_movement = (
            -self.motor_movement_weight * motor_movement_cost
        )
        reward_servo_common_movement = (
            -self.servo_common_movement_weight
            * servo_common_movement_cost
        )
        if self.servo_cyclic_movement_gate_roll_pitch_scale_rad is None:
            servo_cyclic_movement_gate = torch.ones_like(
                servo_cyclic_movement_cost
            )
        else:
            roll_pitch_gate_ratio = (
                torch.linalg.vector_norm(
                    roll_pitch_error,
                    dim=-1,
                    keepdim=True,
                )
                / self.servo_cyclic_movement_gate_roll_pitch_scale_rad
            )
            yaw_rate_gate_ratio = (
                yaw_rate_error.abs()
                / self.servo_cyclic_movement_gate_yaw_rate_scale_rad_s
            )
            full_gate = torch.exp(
                -0.5
                * (
                    roll_pitch_gate_ratio.square()
                    + yaw_rate_gate_ratio.square()
                )
            )
            minimum_gate = self.servo_cyclic_movement_gate_minimum
            servo_cyclic_movement_gate = (
                minimum_gate + (1.0 - minimum_gate) * full_gate
            )
        reward_servo_cyclic_movement = (
            -self.servo_cyclic_movement_weight
            * servo_cyclic_movement_cost
            * servo_cyclic_movement_gate
        )
        reward_control_movement = (
            reward_motor_movement
            + reward_servo_common_movement
            + reward_servo_cyclic_movement
        )
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
        reward_alive = torch.full_like(legacy_attitude_error, self.alive_bonus)
        reward_survival = self.survival_progress_weight * episode_age_fraction
        termination_cost = terminated.to(action.dtype) * (
            self.termination_penalty
            + self.early_termination_penalty * remaining_fraction
        )
        reward = (
            reward_alive + reward_attitude + reward_tilt + reward_yaw_rate
            + reward_joint_tracking
            + reward_rate + reward_action_rate + reward_servo_cyclic
            + reward_control_effort + reward_control_movement
            + reward_saturation + reward_risk + reward_survival - termination_cost
        )
        terms = TensorDict(
            {
                "reward.alive": reward_alive,
                "reward.attitude": reward_attitude,
                "reward.tilt": reward_tilt,
                "reward.yaw_rate": reward_yaw_rate,
                "reward.joint_tracking": reward_joint_tracking,
                "reward.joint_tracking_worst": reward_joint_tracking_worst,
                "reward.joint_tracking_mean": reward_joint_tracking_mean,
                "reward.angular_rate": reward_rate,
                "reward.action_rate": reward_action_rate,
                "reward.servo_cyclic": reward_servo_cyclic,
                "reward.control_effort": reward_control_effort,
                "reward.motor_effort": reward_motor_effort,
                "reward.servo_common_effort": reward_servo_common_effort,
                "reward.servo_cyclic_effort": reward_servo_cyclic_effort,
                "reward.physical_servo_common_effort": (
                    reward_physical_servo_common_effort
                ),
                "reward.control_movement": reward_control_movement,
                "reward.motor_movement": reward_motor_movement,
                "reward.servo_common_movement": reward_servo_common_movement,
                "reward.servo_cyclic_movement": reward_servo_cyclic_movement,
                "reward.saturation": reward_saturation,
                "reward.risk": reward_risk,
                "reward.survival": reward_survival,
                "reward.termination": -termination_cost,
            },
            batch_size=context.batch_size,
            device=context.device,
        )
        diagnostics = TensorDict(
            {
                "joint_roll_pitch_cost": joint_roll_pitch_cost,
                "joint_yaw_rate_cost": joint_yaw_rate_cost,
                "joint_roll_pitch_dominant": joint_roll_pitch_dominant,
                "joint_yaw_rate_dominant": joint_yaw_rate_dominant,
                "servo_cyclic_movement_gate": servo_cyclic_movement_gate,
                "actuator_energy_proxy": actuator_energy_proxy,
                "actuator_effort_proxy": actuator_effort_proxy,
                "physical_servo_common_effort_proxy": (
                    physical_servo_common_effort_cost
                ),
            },
            batch_size=context.batch_size,
            device=context.device,
        )
        if reward.shape != (context.batch_size[0], 1):
            raise ValueError("RewardCalculator must return reward with shape [B, 1]")
        return RewardOutput(
            reward=reward,
            terms=terms,
            valid=torch.isfinite(reward),
            diagnostics=diagnostics,
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {"version": self.version}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", self.version)) != self.version:
            raise ValueError("incompatible reward calculator state")


class AngularAccelerationTrackingRewardCalculator:
    """Reward only body angular-acceleration command tracking error."""

    version = 1

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        params = dict(params or {})
        allowed = {
            "weight",
            "error_scale_rad_s2",
            "huber_delta",
            "reward_form",
            "aggregation",
        }
        unexpected = set(params) - allowed
        if unexpected:
            raise ValueError(
                "unsupported angular-acceleration reward parameters: "
                f"{sorted(unexpected)}"
            )
        self.weight = float(params.get("weight", 1.0))
        self.error_scale = tuple(
            float(value)
            for value in params.get("error_scale_rad_s2", (5.0, 5.0, 2.5))
        )
        self.huber_delta = float(params.get("huber_delta", 1.0))
        self.reward_form = str(params.get("reward_form", "negative_huber"))
        self.aggregation = str(params.get("aggregation", "mean"))
        if (
            not math.isfinite(self.weight)
            or not math.isfinite(self.huber_delta)
            or self.weight <= 0.0
            or self.huber_delta <= 0.0
        ):
            raise ValueError("reward weight and huber_delta must be positive")
        if len(self.error_scale) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.error_scale
        ):
            raise ValueError("error_scale_rad_s2 must contain 3 positive values")
        if self.reward_form not in {"negative_huber", "positive_exponential"}:
            raise ValueError(
                "reward_form must be 'negative_huber' or 'positive_exponential'"
            )
        if self.aggregation not in {"mean", "worst_axis"}:
            raise ValueError("aggregation must be 'mean' or 'worst_axis'")

    def __call__(self, context: TensorDictBase) -> RewardOutput:
        desired = context["desired_angular_acceleration_b"]
        actual = context["actual_angular_acceleration_b"]
        scale = desired.new_tensor(self.error_scale)
        normalized_error = (actual - desired) / scale
        absolute_error = normalized_error.abs()
        delta = self.huber_delta
        axis_cost = torch.where(
            absolute_error <= delta,
            0.5 * normalized_error.square(),
            delta * (absolute_error - 0.5 * delta),
        )
        if self.aggregation == "worst_axis":
            tracking_cost = axis_cost.amax(dim=-1, keepdim=True)
        else:
            tracking_cost = axis_cost.mean(dim=-1, keepdim=True)
        if self.reward_form == "positive_exponential":
            reward = self.weight * torch.exp(-tracking_cost)
        else:
            reward = -self.weight * tracking_cost
        terms = TensorDict(
            {"reward.angular_acceleration_tracking": reward},
            batch_size=context.batch_size,
            device=context.device,
        )
        diagnostics = TensorDict(
            {
                "angular_acceleration_error_b": actual - desired,
                "angular_acceleration_tracking_cost": tracking_cost,
            },
            batch_size=context.batch_size,
            device=context.device,
        )
        return RewardOutput(
            reward=reward,
            terms=terms,
            valid=torch.isfinite(reward),
            diagnostics=diagnostics,
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {"version": self.version}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", self.version)) != self.version:
            raise ValueError("incompatible reward calculator state")
