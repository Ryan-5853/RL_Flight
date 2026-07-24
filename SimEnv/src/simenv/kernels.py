from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Protocol

import torch


class DynamicsKernel(Protocol):
    implemented: bool

    def step(
        self,
        state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        control: torch.Tensor,
        active_mask: torch.Tensor,
        physics_dt: float,
        instance_seeds: torch.Tensor,
        motor_noise_counters: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]: ...


class SensorKernel(Protocol):
    implemented: bool

    def step(
        self,
        sensor_state: Mapping[str, torch.Tensor],
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        active_mask: torch.Tensor,
        physics_step: torch.Tensor,
        instance_seeds: torch.Tensor,
        sample_counters: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]: ...

    def reset(
        self,
        sensor_state: Mapping[str, torch.Tensor],
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        reset_mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]: ...


class NullSensorKernel:
    """Framework placeholder: preserves the sensor cache without resampling."""

    implemented = False

    def step(
        self,
        sensor_state: Mapping[str, torch.Tensor],
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        active_mask: torch.Tensor,
        physics_step: torch.Tensor,
        instance_seeds: torch.Tensor,
        sample_counters: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        del truth_state, parameters, active_mask, physics_step, instance_seeds, sample_counters
        return sensor_state

    def reset(
        self,
        sensor_state: Mapping[str, torch.Tensor],
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        reset_mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        del truth_state, parameters, reset_mask
        return sensor_state


class TensorSensorKernel:
    """Batched sampled sensors with bias, noise, and physical-time delay."""

    implemented = True

    def __init__(
        self,
        sensor_names: tuple[str, ...],
        physics_hz: int,
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
    ) -> None:
        self._sensor_names = sensor_names
        self._physics_hz = physics_hz
        first = next(iter(truth_state.values()))
        self._parallel_count = first.shape[0]
        self._device = first.device
        self._dtype = first.dtype
        self._batch_index = torch.arange(
            self._parallel_count, dtype=torch.int64, device=self._device
        )
        self._history: dict[str, torch.Tensor] = {}
        self._capacity: dict[str, int] = {}
        self._subsystem_ids: dict[str, int] = {}
        self._component_index: dict[str, torch.Tensor] = {}
        self._sample_period: dict[str, torch.Tensor] = {}
        self._delay_lower: dict[str, torch.Tensor] = {}
        self._delay_upper: dict[str, torch.Tensor] = {}
        self._delay_fraction: dict[str, torch.Tensor] = {}

        for name in sensor_names:
            source = self._source(name, truth_state, parameters)
            capacity = self._required_capacity(name, parameters)
            self._capacity[name] = capacity
            self._history[name] = source[:, None, ...].expand(
                self._parallel_count, capacity, *source.shape[1:]
            ).clone()
            digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
            self._subsystem_ids[name] = int.from_bytes(digest, "little") & (
                (1 << 62) - 1
            )
            width = math.prod(source.shape[1:])
            self._component_index[name] = torch.arange(
                width, dtype=torch.int64, device=self._device
            )[None, :]
        self.refresh_parameters(parameters)

    def refresh_parameters(
        self,
        parameters: Mapping[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> None:
        """缓存传感器采样周期和延迟分解，并支持 masked reset 局部刷新。"""

        for name in self._sensor_names:
            sample_period = torch.round(
                self._physics_hz / parameters[f"sensors.{name}.sample_hz"]
            ).to(torch.int64)
            delay_steps = parameters[f"sensors.{name}.delay"] * self._physics_hz
            lower = torch.floor(delay_steps).to(torch.int64)
            upper = torch.ceil(delay_steps).to(torch.int64)
            fraction = delay_steps - lower.to(self._dtype)
            candidates = (
                ("sample_period", self._sample_period, sample_period),
                ("delay_lower", self._delay_lower, lower),
                ("delay_upper", self._delay_upper, upper),
                ("delay_fraction", self._delay_fraction, fraction),
            )
            for _label, cache, candidate in candidates:
                if mask is None or name not in cache:
                    cache[name] = candidate
                else:
                    cache[name].copy_(torch.where(mask, candidate, cache[name]))

    def initial_state(
        self,
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            name: self._source(name, truth_state, parameters)
            + parameters[f"sensors.{name}.bias"]
            for name in self._sensor_names
        }

    def validate_parameters(self, parameters: Mapping[str, torch.Tensor]) -> None:
        for name in self._sensor_names:
            required = self._required_capacity(name, parameters)
            if required > self._capacity[name]:
                raise ValueError(
                    f"sensor {name!r} delay requires history length {required}, "
                    f"but the batch capacity is {self._capacity[name]}"
                )

    def state_dict(self) -> Mapping[str, Any]:
        """导出影响未来传感器输出的全部延迟历史状态。"""

        return {
            "schema_version": 1,
            "sensor_names": self._sensor_names,
            "physics_hz": self._physics_hz,
            "capacity": dict(self._capacity),
            "history": {name: value.clone() for name, value in self._history.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """严格校验并恢复传感器延迟环形缓冲。"""

        if not isinstance(state, Mapping):
            raise TypeError("sensor-kernel state must be a mapping")
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("unsupported sensor-kernel state schema")
        if tuple(state.get("sensor_names", ())) != self._sensor_names:
            raise ValueError("sensor-kernel state changes sensor names or order")
        if int(state.get("physics_hz", -1)) != self._physics_hz:
            raise ValueError("sensor-kernel state changes physics_hz")
        if dict(state.get("capacity", {})) != self._capacity:
            raise ValueError("sensor-kernel state changes delay-buffer capacity")
        history = state.get("history")
        if not isinstance(history, Mapping) or set(history) != set(self._history):
            raise ValueError("sensor-kernel history fields are incompatible")
        checked: dict[str, torch.Tensor] = {}
        for name, current in self._history.items():
            value = history[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"sensor-kernel history {name} must be a Tensor")
            if value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError(f"sensor-kernel history {name} has incompatible shape/dtype")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"sensor-kernel history {name} contains non-finite values")
            checked[name] = value.to(self._device)
        for name, value in checked.items():
            self._history[name].copy_(value)

    def step(
        self,
        sensor_state: Mapping[str, torch.Tensor],
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        active_mask: torch.Tensor,
        physics_step: torch.Tensor,
        instance_seeds: torch.Tensor,
        sample_counters: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        next_state: dict[str, torch.Tensor] = {}
        for name in self._sensor_names:
            source = self._source(name, truth_state, parameters)
            history = self._history[name]
            capacity = self._capacity[name]
            write_index = torch.remainder(physics_step, capacity)
            previous_slot = history[self._batch_index, write_index]
            history[self._batch_index, write_index] = torch.where(
                self._expand(active_mask, source), source, previous_slot
            )

            sample_period = self._sample_period[name]
            sample_due = active_mask & (torch.remainder(physics_step, sample_period) == 0)
            delayed = self._delayed_value(name, physics_step, parameters)
            noise = self._normal_noise(
                instance_seeds,
                sample_counters[name],
                self._subsystem_ids[name],
                self._component_index[name],
                sensor_state[name].shape[1:],
            )
            measured = (
                delayed
                + parameters[f"sensors.{name}.bias"]
                + noise * parameters[f"sensors.{name}.noise.stddev"]
            )
            next_state[name] = torch.where(
                self._expand(sample_due, measured), measured, sensor_state[name]
            )
            sample_counters[name].add_(sample_due.to(torch.int64))
        return next_state

    def reset(
        self,
        sensor_state: Mapping[str, torch.Tensor],
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        reset_mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        self.validate_parameters(parameters)
        self.refresh_parameters(parameters, reset_mask)
        next_state: dict[str, torch.Tensor] = {}
        for name in self._sensor_names:
            source = self._source(name, truth_state, parameters)
            history = self._history[name]
            initial_history = source[:, None, ...].expand_as(history)
            history.copy_(
                torch.where(
                    self._expand(reset_mask, history), initial_history, history
                )
            )
            initial_output = source + parameters[f"sensors.{name}.bias"]
            next_state[name] = torch.where(
                self._expand(reset_mask, initial_output),
                initial_output,
                sensor_state[name],
            )
        return next_state

    def _delayed_value(
        self,
        name: str,
        physics_step: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        history = self._history[name]
        capacity = self._capacity[name]
        lower_delay = self._delay_lower[name]
        upper_delay = self._delay_upper[name]
        newer_index = torch.remainder(physics_step - lower_delay, capacity)
        older_index = torch.remainder(physics_step - upper_delay, capacity)
        newer = history[self._batch_index, newer_index]
        older = history[self._batch_index, older_index]
        fraction = self._delay_fraction[name]
        return newer + self._expand(fraction, newer) * (older - newer)

    def _normal_noise(
        self,
        seeds: torch.Tensor,
        counters: torch.Tensor,
        subsystem_id: int,
        component: torch.Tensor,
        tail_shape: torch.Size,
    ) -> torch.Tensor:
        base = (
            seeds[:, None]
            ^ ((counters[:, None] + 1) * 6364136223846793005)
            ^ ((self._batch_index[:, None] + 1) * 1442695040888963407)
            ^ subsystem_id
            ^ ((component + 1) * 3202034522624059733)
        )
        u1 = self._uniform_from_key(base ^ 2862933555777941757)
        u2 = self._uniform_from_key(base ^ 3037000493)
        normal = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
        return normal.reshape(self._parallel_count, *tail_shape)

    def _uniform_from_key(self, key: torch.Tensor) -> torch.Tensor:
        key = key - 7046029254386353131
        key = (key ^ self._logical_right_shift(key, 30)) * -4658895280553007687
        key = (key ^ self._logical_right_shift(key, 27)) * -7723592293110705685
        key = key ^ self._logical_right_shift(key, 31)
        mantissa = torch.bitwise_and(key, (1 << 53) - 1).to(self._dtype)
        return (mantissa + 1.0) / float((1 << 53) + 1)

    @staticmethod
    def _logical_right_shift(value: torch.Tensor, bits: int) -> torch.Tensor:
        mask = (1 << (64 - bits)) - 1
        return torch.bitwise_and(torch.bitwise_right_shift(value, bits), mask)

    def _required_capacity(
        self, name: str, parameters: Mapping[str, torch.Tensor]
    ) -> int:
        delay_steps = parameters[f"sensors.{name}.delay"] * self._physics_hz
        return int(torch.ceil(delay_steps.max()).item()) + 1

    def _source(
        self,
        name: str,
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if name == "gyro":
            return truth_state["angular_velocity_b"]
        if name == "motor_speed":
            return truth_state["motor_speed"]
        if name == "accelerometer":
            # 质心处加速度计测量机体系非重力比力。直接使用同一物理步的
            # force_b/m，避免把步初姿态计算的世界系加速度再用步末姿态旋回，
            # 从而引入 O(|omega|*dt) 的伪横向分量。
            return (
                truth_state["force_b"]
                / parameters["body.mass"][:, None]
            )
        raise RuntimeError(f"unsupported sensor {name!r}")

    @staticmethod
    def _expand(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mask.reshape(mask.shape[0], *([1] * (target.ndim - 1)))
