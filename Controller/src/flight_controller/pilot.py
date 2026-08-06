from __future__ import annotations

import math
from typing import Any, Mapping

import torch


def _node(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"pilot.{name} must be a mapping")
    return value


def _number(config: Mapping[str, Any], name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"pilot parameter {name} must be finite")
    return value


class VirtualPilotHeightController:
    """Throttle-domain altitude-hold virtual pilot that owns the upper rotor.

    Mirrors the deployment contract: the pilot sets the coaxial collective
    (upper throttle, with the lower rotor base following it), so every scheme
    shares the same external altitude behavior. Supports a damped PID mode
    (used by the identification line) and an incremental PI mode.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        control_dt: float,
    ) -> None:
        throttle = _node(config, "throttle")
        height = _node(config, "height_controller")
        self.throttle_minimum = _number(throttle, "minimum", 0.20)
        self.throttle_maximum = _number(throttle, "maximum", 0.96)
        self.throttle_initial = _number(throttle, "initial", 0.30)
        self.spool_duration_s = _number(throttle, "spool_duration_s", 0.10)
        self.target_height_m = _number(height, "target_m", 0.0)
        self.height_kp = _number(height, "proportional_gain", 0.10)
        self.height_ki = _number(height, "integral_gain", 0.20)
        self.height_kd = _number(height, "derivative_gain", 0.0)
        self.height_integral_limit = _number(height, "integral_limit", 1.0)
        self.height_error_limit_m = _number(height, "error_limit_m", 5.0)
        self.mode = str(height.get("mode", "incremental"))
        if self.mode not in {"incremental", "pid"}:
            raise ValueError("height_controller mode must be incremental or pid")
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.control_dt = control_dt
        self.upper_throttle = torch.full(
            (batch_size, 1),
            self.throttle_minimum,
            device=device,
            dtype=dtype,
        )
        self.spool_remaining = torch.full(
            (batch_size,),
            self.spool_duration_s,
            device=device,
            dtype=dtype,
        )
        self.previous_height_error = torch.zeros(
            batch_size, device=device, dtype=dtype
        )
        self.height_integral = torch.zeros(
            batch_size, device=device, dtype=dtype
        )

    def reset(self, mask: torch.Tensor) -> None:
        expanded = mask[:, None]
        self.upper_throttle.masked_fill_(expanded, self.throttle_minimum)
        self.spool_remaining.masked_fill_(mask, self.spool_duration_s)
        self.previous_height_error.masked_fill_(mask, 0.0)
        self.height_integral.masked_fill_(mask, 0.0)

    def step(
        self,
        height_m: torch.Tensor,
        vertical_speed_m_s: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        error = (self.target_height_m - height_m.squeeze(-1)).clamp(
            -self.height_error_limit_m,
            self.height_error_limit_m,
        )
        vertical_speed = vertical_speed_m_s.squeeze(-1)
        dt = self.control_dt
        in_spool = self.spool_remaining > 0.0
        controller_active = active & ~in_spool
        if self.mode == "pid":
            integral = (self.height_integral + error * dt).clamp(
                -self.height_integral_limit,
                self.height_integral_limit,
            )
            self.height_integral = torch.where(
                active, integral, self.height_integral
            )
            target = (
                self.throttle_initial
                + self.height_kp * error
                + self.height_ki * integral
                - self.height_kd * vertical_speed
            )
            candidate = target.clamp(
                self.throttle_minimum,
                self.throttle_maximum,
            )
            self.upper_throttle = torch.where(
                controller_active[:, None],
                candidate[:, None],
                self.upper_throttle,
            )
        else:
            increment = (
                self.height_kp * (error - self.previous_height_error)
                + self.height_ki * error * dt
            )
            candidate = (self.upper_throttle[:, 0] + increment).clamp(
                self.throttle_minimum,
                self.throttle_maximum,
            )
            self.upper_throttle = torch.where(
                controller_active[:, None],
                candidate[:, None],
                self.upper_throttle,
            )
            self.previous_height_error = torch.where(
                controller_active, error, self.previous_height_error
            )
        if self.spool_duration_s > 0.0:
            fraction = (
                1.0
                - self.spool_remaining / self.spool_duration_s
            ).clamp(0.0, 1.0)
            spool_value = (
                self.throttle_minimum
                + fraction
                * (self.throttle_initial - self.throttle_minimum)
            )
            self.upper_throttle = torch.where(
                in_spool[:, None],
                spool_value[:, None],
                self.upper_throttle,
            )
        self.spool_remaining = torch.where(
            active,
            (self.spool_remaining - dt).clamp_min(0.0),
            self.spool_remaining,
        )
        return self.upper_throttle.clamp(
            self.throttle_minimum,
            self.throttle_maximum,
        )
