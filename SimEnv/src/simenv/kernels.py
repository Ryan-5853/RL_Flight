from __future__ import annotations

import hashlib
import math
from typing import Mapping, Protocol

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
    _GRAVITY_N = (0.0, 0.0, 9.80665)

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

        for name in sensor_names:
            source = self._source(name, truth_state)
            capacity = self._required_capacity(name, parameters)
            self._capacity[name] = capacity
            self._history[name] = source[:, None, ...].expand(
                self._parallel_count, capacity, *source.shape[1:]
            ).clone()
            digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
            self._subsystem_ids[name] = int.from_bytes(digest, "little") & (
                (1 << 62) - 1
            )

    def initial_state(
        self,
        truth_state: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            name: self._source(name, truth_state)
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
            source = self._source(name, truth_state)
            history = self._history[name]
            capacity = self._capacity[name]
            write_index = torch.remainder(physics_step, capacity)
            previous_slot = history[self._batch_index, write_index]
            history[self._batch_index, write_index] = torch.where(
                self._expand(active_mask, source), source, previous_slot
            )

            sample_period = torch.round(
                self._physics_hz / parameters[f"sensors.{name}.sample_hz"]
            ).to(torch.int64)
            sample_due = active_mask & (torch.remainder(physics_step, sample_period) == 0)
            delayed = self._delayed_value(name, physics_step, parameters)
            noise = self._normal_noise(
                instance_seeds,
                sample_counters[name],
                self._subsystem_ids[name],
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
        next_state: dict[str, torch.Tensor] = {}
        for name in self._sensor_names:
            source = self._source(name, truth_state)
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
        delay_steps = parameters[f"sensors.{name}.delay"] * self._physics_hz
        lower_delay = torch.floor(delay_steps).to(torch.int64)
        upper_delay = torch.ceil(delay_steps).to(torch.int64)
        newer_index = torch.remainder(physics_step - lower_delay, capacity)
        older_index = torch.remainder(physics_step - upper_delay, capacity)
        newer = history[self._batch_index, newer_index]
        older = history[self._batch_index, older_index]
        fraction = delay_steps - lower_delay.to(self._dtype)
        return newer + self._expand(fraction, newer) * (older - newer)

    def _normal_noise(
        self,
        seeds: torch.Tensor,
        counters: torch.Tensor,
        subsystem_id: int,
        tail_shape: torch.Size,
    ) -> torch.Tensor:
        width = math.prod(tail_shape)
        component = torch.arange(width, dtype=torch.int64, device=self._device)[None, :]
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
        self, name: str, truth_state: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if name == "gyro":
            return truth_state["angular_velocity_b"]
        if name == "motor_speed":
            return truth_state["motor_speed"]
        if name == "accelerometer":
            gravity_n = torch.as_tensor(
                self._GRAVITY_N, dtype=self._dtype, device=self._device
            )
            specific_force_n = truth_state["linear_acceleration_n"] - gravity_n
            return self._rotate_world_to_body(
                truth_state["attitude_q_wb"], specific_force_n
            )
        raise RuntimeError(f"unsupported sensor {name!r}")

    @staticmethod
    def _rotate_world_to_body(q_wb: torch.Tensor, vector_n: torch.Tensor) -> torch.Tensor:
        q_vector = q_wb[:, 1:]
        cross = torch.linalg.cross(q_vector, vector_n)
        return vector_n - 2 * q_wb[:, :1] * cross + 2 * torch.linalg.cross(
            q_vector, cross
        )

    @staticmethod
    def _expand(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mask.reshape(mask.shape[0], *([1] * (target.ndim - 1)))
