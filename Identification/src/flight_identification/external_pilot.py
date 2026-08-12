from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _pair(
    node: Mapping[str, Any],
    name: str,
    default: tuple[float, float],
    *,
    minimum: float | None = None,
) -> tuple[float, float]:
    raw = node.get(name, default)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ValueError(f"external_pilot profile {name} must contain two numbers")
    low, high = float(raw[0]), float(raw[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"invalid external_pilot profile range {name}: {raw}")
    if minimum is not None and low < minimum:
        raise ValueError(f"external_pilot profile {name} must be >= {minimum}")
    return low, high


@dataclass(frozen=True)
class ExternalPilotProfile:
    name: str
    weight: float
    mode: str
    feedback_gain_scale: tuple[float, float]
    reaction_delay_s: tuple[float, float]
    update_period_s: tuple[float, float]
    height_bias_m: tuple[float, float]
    height_noise_std_m: tuple[float, float]
    lapse_probability: tuple[float, float]
    manual_throttle_range: tuple[float, float]
    manual_hold_s: tuple[float, float]


def load_external_pilot_profiles(
    raw_experiment: Mapping[str, Any],
) -> tuple[ExternalPilotProfile, ...]:
    """Parse optional deployment-pilot capability profiles.

    An absent block preserves the historical ideal incremental-PI pilot.  The
    profile model deliberately lives outside the attitude controller: it owns
    upper throttle and may be slower, biased, intermittent, or fully manual.
    """

    block = raw_experiment.get("external_pilot")
    if block is None:
        return (
            ExternalPilotProfile(
                name="legacy_ideal",
                weight=1.0,
                mode="feedback",
                feedback_gain_scale=(1.0, 1.0),
                reaction_delay_s=(0.0, 0.0),
                update_period_s=(0.0, 0.0),
                height_bias_m=(0.0, 0.0),
                height_noise_std_m=(0.0, 0.0),
                lapse_probability=(0.0, 0.0),
                manual_throttle_range=(0.0, 1.0),
                manual_hold_s=(1.0, 1.0),
            ),
        )
    profiles_raw = _mapping(block, "external_pilot").get("profiles")
    if not isinstance(profiles_raw, Sequence) or isinstance(profiles_raw, (str, bytes)):
        raise ValueError("external_pilot.profiles must be a non-empty sequence")
    profiles: list[ExternalPilotProfile] = []
    for index, value in enumerate(profiles_raw):
        node = _mapping(value, f"external_pilot.profiles[{index}]")
        name = str(node.get("name", f"profile_{index}"))
        weight = float(node.get("weight", 1.0))
        mode = str(node.get("mode", "feedback"))
        if not name or not math.isfinite(weight) or weight < 0.0:
            raise ValueError("external_pilot profile name/weight is invalid")
        if mode not in {"feedback", "manual_sample_hold"}:
            raise ValueError(
                "external_pilot profile mode must be feedback or manual_sample_hold"
            )
        profiles.append(
            ExternalPilotProfile(
                name=name,
                weight=weight,
                mode=mode,
                feedback_gain_scale=_pair(
                    node, "feedback_gain_scale", (1.0, 1.0), minimum=0.0
                ),
                reaction_delay_s=_pair(
                    node, "reaction_delay_s", (0.0, 0.0), minimum=0.0
                ),
                update_period_s=_pair(
                    node, "update_period_s", (0.0, 0.0), minimum=0.0
                ),
                height_bias_m=_pair(node, "height_bias_m", (0.0, 0.0)),
                height_noise_std_m=_pair(
                    node, "height_noise_std_m", (0.0, 0.0), minimum=0.0
                ),
                lapse_probability=_pair(
                    node, "lapse_probability", (0.0, 0.0), minimum=0.0
                ),
                manual_throttle_range=_pair(
                    node, "manual_throttle_range", (0.3, 0.8), minimum=0.0
                ),
                manual_hold_s=_pair(
                    node, "manual_hold_s", (0.5, 2.0), minimum=1e-6
                ),
            )
        )
        if profiles[-1].lapse_probability[1] > 1.0:
            raise ValueError("external_pilot lapse_probability must be <= 1")
        if profiles[-1].manual_throttle_range[1] > 1.0:
            raise ValueError("external_pilot manual_throttle_range must be <= 1")
    if not profiles or sum(profile.weight for profile in profiles) <= 0.0:
        raise ValueError("external_pilot profiles must have positive total weight")
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)):
        raise ValueError("external_pilot profile names must be unique")
    return tuple(profiles)


class ConfigurableExternalPilot:
    """Capability-randomized upper-rotor owner plus the existing stick pilot.

    ``VirtualPilotCommandSource`` remains responsible for roll/pitch/yaw target
    generation.  This adapter perturbs or replaces only its upper-throttle
    loop, keeping the ownership boundary explicit and auditable.
    """

    def __init__(
        self,
        base: Any,
        raw_experiment: Mapping[str, Any],
        control_hz: int,
        profile_names: Sequence[str] | None = None,
    ) -> None:
        self.base = base
        profiles = load_external_pilot_profiles(raw_experiment)
        if profile_names:
            selected = set(profile_names)
            profiles = tuple(profile for profile in profiles if profile.name in selected)
            missing = selected - {profile.name for profile in profiles}
            if missing:
                raise ValueError(f"unknown external pilot profiles: {sorted(missing)}")
        if not profiles or sum(profile.weight for profile in profiles) <= 0.0:
            raise ValueError("selected external pilot profiles have no positive weight")
        self.profiles = profiles
        self.dt = 1.0 / float(control_hz)
        self.batch_size = base.batch_size
        self.device = base.device
        self.dtype = base.dtype
        self.generator = base.generator
        self.profile_index = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.int64
        )
        self.feedback_gain_scale = torch.ones(
            self.batch_size, 1, device=self.device, dtype=self.dtype
        )
        self.reaction_delay_steps = torch.zeros_like(self.profile_index)
        self.update_period_steps = torch.ones_like(self.profile_index)
        self.update_remaining_steps = torch.zeros_like(self.profile_index)
        self.height_bias = torch.zeros_like(self.feedback_gain_scale)
        self.height_noise_std = torch.zeros_like(self.feedback_gain_scale)
        self.lapse_probability = torch.zeros_like(self.feedback_gain_scale)
        self.manual_mode = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.bool
        )
        self.manual_target = torch.zeros_like(self.feedback_gain_scale)
        self.manual_hold_remaining = torch.zeros_like(self.feedback_gain_scale)
        max_delay = max(profile.reaction_delay_s[1] for profile in profiles)
        self.max_delay_steps = int(math.ceil(max_delay * control_hz))
        self.height_history = torch.zeros(
            self.batch_size,
            self.max_delay_steps + 1,
            device=self.device,
            dtype=self.dtype,
        )
        self.height_history_initialized = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.bool
        )

    def _uniform(self, low: float, high: float, shape: tuple[int, ...]) -> torch.Tensor:
        random = torch.rand(
            shape,
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )
        return low + (high - low) * random

    def _sample_profile_values(self, mask: torch.Tensor) -> None:
        count = int(mask.sum())
        if count == 0:
            return
        weights = torch.tensor(
            [profile.weight for profile in self.profiles],
            device=self.device,
            dtype=self.dtype,
        )
        sampled = torch.multinomial(
            weights, count, replacement=True, generator=self.generator
        )
        self.profile_index[mask] = sampled
        selected_rows = torch.nonzero(mask).flatten()
        for index, profile in enumerate(self.profiles):
            rows = selected_rows[sampled == index]
            if len(rows) == 0:
                continue
            n = len(rows)
            self.feedback_gain_scale[rows] = self._uniform(
                *profile.feedback_gain_scale, (n, 1)
            )
            delay = self._uniform(*profile.reaction_delay_s, (n,)) / self.dt
            self.reaction_delay_steps[rows] = delay.round().to(torch.int64)
            period = self._uniform(*profile.update_period_s, (n,)) / self.dt
            self.update_period_steps[rows] = period.round().to(torch.int64).clamp_min(1)
            self.height_bias[rows] = self._uniform(*profile.height_bias_m, (n, 1))
            self.height_noise_std[rows] = self._uniform(
                *profile.height_noise_std_m, (n, 1)
            )
            self.lapse_probability[rows] = self._uniform(
                *profile.lapse_probability, (n, 1)
            )
            self.manual_mode[rows] = profile.mode == "manual_sample_hold"
            self.manual_target[rows] = self._uniform(
                *profile.manual_throttle_range, (n, 1)
            )
            self.manual_hold_remaining[rows] = self._uniform(
                *profile.manual_hold_s, (n, 1)
            )

    def reset(self, mask: torch.Tensor) -> None:
        self.base.reset(mask)
        self._sample_profile_values(mask)
        self.update_remaining_steps.masked_fill_(mask, 0)
        self.height_history_initialized.masked_fill_(mask, False)
        self.height_history[mask] = 0.0

    @torch.no_grad()
    def step(self, height_m: torch.Tensor, active_mask: torch.Tensor) -> None:
        first = active_mask & ~self.height_history_initialized
        self.height_history[first] = height_m[first].expand(-1, self.max_delay_steps + 1)
        self.height_history_initialized |= first
        if self.max_delay_steps:
            self.height_history[:, 1:] = self.height_history[:, :-1].clone()
        self.height_history[:, 0] = height_m[:, 0]
        delayed = self.height_history.gather(
            1, self.reaction_delay_steps[:, None].clamp_max(self.max_delay_steps)
        )
        noise = torch.randn(
            delayed.shape,
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        ) * self.height_noise_std
        observed = delayed + self.height_bias + noise
        target = self.base.height_target
        scaled_observed = target - self.feedback_gain_scale * (target - observed)

        update_due = active_mask & (self.update_remaining_steps <= 0)
        lapse = torch.rand(
            self.batch_size,
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        ) < self.lapse_probability[:, 0]
        feedback_update = update_due & ~lapse & ~self.manual_mode
        previous = {
            name: getattr(self.base, name).clone()
            for name in (
                "upper_throttle",
                "throttle_target",
                "height_controller_output",
                "height_previous_error",
            )
        }
        self.base.step(scaled_observed, active_mask)
        hold_feedback = active_mask & ~feedback_update & ~self.manual_mode
        for name, value in previous.items():
            current = getattr(self.base, name)
            setattr(self.base, name, torch.where(hold_feedback[:, None], value, current))

        manual_active = active_mask & self.manual_mode
        next_hold = self.manual_hold_remaining - self.dt
        change = manual_active & (next_hold[:, 0] <= 0.0)
        for index, profile in enumerate(self.profiles):
            rows = change & (self.profile_index == index)
            n = int(rows.sum())
            if n:
                self.manual_target[rows] = self._uniform(
                    *profile.manual_throttle_range, (n, 1)
                )
                next_hold[rows] = self._uniform(*profile.manual_hold_s, (n, 1))
        lower = previous["upper_throttle"] - self.base.config.throttle_fall_rate_per_s * self.dt
        upper = previous["upper_throttle"] + self.base.config.throttle_rise_rate_per_s * self.dt
        manual_throttle = torch.maximum(torch.minimum(self.manual_target, upper), lower)
        manual_throttle = manual_throttle.clamp(
            self.base.config.throttle_minimum, self.base.config.throttle_maximum
        )
        self.base.upper_throttle = torch.where(
            manual_active[:, None], manual_throttle, self.base.upper_throttle
        )
        self.base.throttle_target = torch.where(
            manual_active[:, None], self.manual_target, self.base.throttle_target
        )
        self.manual_hold_remaining = torch.where(
            manual_active[:, None], next_hold, self.manual_hold_remaining
        )
        self.update_remaining_steps = torch.where(
            update_due,
            self.update_period_steps - 1,
            (self.update_remaining_steps - 1).clamp_min(0),
        )

    def snapshot(self) -> Any:
        return self.base.snapshot()

    @property
    def desired_yaw_rate(self) -> torch.Tensor:
        return self.base.desired_yaw_rate

    @property
    def profile_names(self) -> tuple[str, ...]:
        return tuple(profile.name for profile in self.profiles)

    def episode_profile_names(self) -> list[str]:
        names = self.profile_names
        return [names[index] for index in self.profile_index.detach().cpu().tolist()]


def make_external_pilot(
    pilot_config: Any,
    raw_experiment: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    control_hz: int,
    profile_names: Sequence[str] | None = None,
) -> ConfigurableExternalPilot:
    from flight_train.commands import VirtualPilotCommandSource

    base = VirtualPilotCommandSource(
        pilot_config, batch_size, device, dtype, control_hz
    )
    return ConfigurableExternalPilot(
        base, raw_experiment, control_hz, profile_names=profile_names
    )
