from __future__ import annotations

import uuid
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

import torch

from .config import MaterializedConfig, load_and_materialize
from .dynamics import TensorDynamicsKernel
from .errors import ConfigurationError, EnvironmentClosedError
from .kernels import TensorSensorKernel
from .logging import TensorChunkLogger
from .types import AdvanceResult, ErrorCode, Observation, ResetResult


class SimulationEnvironment:
    """Batched simulation interface with isolated parameters and state."""

    def __init__(
        self,
        materialized: MaterializedConfig,
        parallel_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.parallel_count = parallel_count
        self.batch_shape = torch.Size([parallel_count])
        self.device = device
        self.dtype = dtype
        self.batch_id = str(uuid.uuid4())
        self.instance_ids = tuple(str(uuid.uuid4()) for _ in range(parallel_count))
        self._config = materialized
        self._parameters = dict(materialized.parameters)
        self._truth = self._initialize_truth(materialized.initial_state)
        self._sensor_kernel = TensorSensorKernel(
            tuple(materialized.sensor_state),
            materialized.timing.physics_hz,
            self._truth,
            self._parameters,
        )
        self._sensors = self._sensor_kernel.initial_state(
            self._truth, self._parameters
        )
        self._physics_step = torch.zeros(parallel_count, dtype=torch.int64, device=device)
        self._control_step = torch.zeros(parallel_count, dtype=torch.int64, device=device)
        self._valid = torch.ones(parallel_count, dtype=torch.bool, device=device)
        self._error_code = torch.zeros(parallel_count, dtype=torch.int32, device=device)
        self._generation = torch.zeros(parallel_count, dtype=torch.int64, device=device)
        self._instance_seeds = torch.full(
            (parallel_count,), materialized.seed, dtype=torch.int64, device=device
        )
        self._random_counters = {
            "motors": torch.zeros(
                (parallel_count, 2), dtype=torch.int64, device=device
            ),
            **{
                f"sensor.{name}": torch.zeros(
                    parallel_count, dtype=torch.int64, device=device
                )
                for name in materialized.sensor_state
            },
        }
        self._control = torch.zeros((parallel_count, 5), dtype=dtype, device=device)
        self._dynamics = TensorDynamicsKernel(parallel_count, device, dtype)
        self._closed = False
        self._logger = TensorChunkLogger(
            materialized.logging,
            self.batch_id,
            self.instance_ids,
            materialized.source_path,
            materialized.raw,
            self._parameters,
        )
        self._append_log(
            active_mask=torch.zeros_like(self._valid),
            event_code=torch.ones(parallel_count, dtype=torch.int32, device=device),
        )

    @classmethod
    def create(
        cls,
        config_path: str | Path,
        parallel_count: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "SimulationEnvironment":
        if isinstance(parallel_count, bool) or not isinstance(parallel_count, int) or parallel_count <= 0:
            raise ConfigurationError("parallel_count must be a positive integer")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError(f"CUDA device requested but unavailable: {resolved_device}")
        if not dtype.is_floating_point:
            raise ConfigurationError("dtype must be a floating-point torch dtype")
        materialized = load_and_materialize(
            config_path, parallel_count, resolved_device, dtype
        )
        return cls(materialized, parallel_count, resolved_device, dtype)

    @property
    def parameters(self) -> Mapping[str, torch.Tensor]:
        self._ensure_open()
        return MappingProxyType({name: value.clone() for name, value in self._parameters.items()})

    @property
    def dynamics_implemented(self) -> bool:
        return self._dynamics.implemented

    @property
    def sensors_implemented(self) -> bool:
        return self._sensor_kernel.implemented

    def observe(
        self,
        source: Literal["truth", "sensor"],
        fields: tuple[str, ...] | None = None,
    ) -> Observation:
        self._ensure_open()
        if source == "truth":
            available = self._truth
        elif source == "sensor":
            available = self._sensors
        else:
            raise ValueError("source must be 'truth' or 'sensor'")

        selected = tuple(available) if fields is None else fields
        unknown = [name for name in selected if name not in available]
        if unknown:
            raise KeyError(f"unknown {source} observation fields: {unknown}")
        values = MappingProxyType({name: available[name].clone() for name in selected})
        return Observation(
            batch_id=self.batch_id,
            instance_ids=self.instance_ids,
            physics_step=self._physics_step.clone(),
            control_step=self._control_step.clone(),
            sim_time_s=self._simulation_time(),
            source=source,
            values=values,
            valid=self._valid.clone(),
        )

    def reset(
        self, reset_mask: torch.Tensor, config_path: str | Path
    ) -> ResetResult:
        """Replace selected batch slots using a fully batched candidate config."""
        self._ensure_open()
        mask = self._validate_reset_mask(reset_mask)
        previous_instance_ids = self.instance_ids
        if not bool(mask.any().item()):
            return ResetResult(
                batch_id=self.batch_id,
                reset_mask=mask.clone(),
                previous_instance_ids=previous_instance_ids,
                instance_ids=self.instance_ids,
                generation=self._generation.clone(),
            )

        replacement = load_and_materialize(
            config_path, self.parallel_count, self.device, self.dtype
        )
        replacement_truth = self._initialize_truth(replacement.initial_state)
        self._validate_reset_compatibility(replacement, replacement_truth)

        for name, value in self._parameters.items():
            self._masked_copy(value, replacement.parameters[name], mask)
        for name, value in self._truth.items():
            self._masked_copy(value, replacement_truth[name], mask)
        self._sensors = dict(
            self._sensor_kernel.reset(
                self._sensors, self._truth, self._parameters, mask
            )
        )

        self._physics_step.masked_fill_(mask, 0)
        self._control_step.masked_fill_(mask, 0)
        self._control.masked_fill_(mask[:, None], 0)
        self._valid.masked_fill_(mask, True)
        self._error_code.masked_fill_(mask, int(ErrorCode.OK))
        self._generation.add_(mask.to(torch.int64))
        self._instance_seeds.masked_fill_(mask, replacement.seed)
        for counter in self._random_counters.values():
            expanded = mask.reshape(
                self.parallel_count, *([1] * (counter.ndim - 1))
            )
            counter.masked_fill_(expanded, 0)

        selected = mask.detach().cpu().tolist()
        self.instance_ids = tuple(
            str(uuid.uuid4()) if should_reset else current
            for current, should_reset in zip(self.instance_ids, selected)
        )

        self._logger.record_reset(
            reset_mask=mask,
            generation=self._generation,
            previous_instance_ids=previous_instance_ids,
            instance_ids=self.instance_ids,
            config_path=replacement.source_path,
            parameters=replacement.parameters,
        )
        event_code = torch.where(
            mask,
            torch.full_like(self._error_code, 2),
            torch.full_like(self._error_code, -1),
        )
        self._append_log(torch.zeros_like(self._valid), event_code)
        return ResetResult(
            batch_id=self.batch_id,
            reset_mask=mask.clone(),
            previous_instance_ids=previous_instance_ids,
            instance_ids=self.instance_ids,
            generation=self._generation.clone(),
        )

    def advance(
        self,
        control: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> AdvanceResult:
        self._ensure_open()
        self._validate_control_tensor(control)
        active = self._validate_active_mask(active_mask)

        finite = torch.isfinite(control).all(dim=1)
        motor_in_range = ((control[:, :2] >= 0) & (control[:, :2] <= 1)).all(dim=1)
        servo_in_range = ((control[:, 2:] >= -1) & (control[:, 2:] <= 1)).all(dim=1)
        legal = finite & motor_in_range & servo_in_range
        invalid = active & self._valid & ~legal
        self._valid = self._valid & ~invalid
        self._error_code = torch.where(
            invalid,
            torch.full_like(self._error_code, int(ErrorCode.INVALID_CONTROL)),
            self._error_code,
        )
        advance_mask = active & self._valid & legal
        self._control = torch.where(advance_mask[:, None], control, self._control)

        steps_advanced = torch.zeros_like(self._physics_step)
        for _ in range(self._config.timing.substeps):
            step_mask = advance_mask & self._valid
            candidate = self._dynamics.step(
                self._truth,
                self._parameters,
                self._control,
                step_mask,
                self._config.timing.physics_dt,
                self._instance_seeds,
                self._random_counters["motors"],
            )
            candidate_finite = self._state_is_finite(candidate)
            failed = step_mask & ~candidate_finite
            self._valid = self._valid & ~failed
            self._error_code = torch.where(
                failed,
                torch.full_like(self._error_code, int(ErrorCode.NONFINITE_STATE)),
                self._error_code,
            )
            commit = step_mask & candidate_finite
            self._commit_state(candidate, commit)
            self._random_counters["motors"].add_(
                commit[:, None].to(torch.int64)
            )
            self._physics_step = self._physics_step + commit.to(torch.int64)
            steps_advanced = steps_advanced + commit.to(torch.int64)
            self._sensors = dict(
                self._sensor_kernel.step(
                    self._sensors,
                    self._truth,
                    self._parameters,
                    commit,
                    self._physics_step,
                    self._instance_seeds,
                    {
                        name: self._random_counters[f"sensor.{name}"]
                        for name in self._sensors
                    },
                )
            )
            self._append_log(
                active_mask=commit,
                event_code=torch.zeros(
                    self.parallel_count, dtype=torch.int32, device=self.device
                ),
            )

        completed = steps_advanced == self._config.timing.substeps
        self._control_step = self._control_step + completed.to(torch.int64)
        return AdvanceResult(
            batch_id=self.batch_id,
            instance_ids=self.instance_ids,
            physics_step=self._physics_step.clone(),
            control_step=self._control_step.clone(),
            sim_time_s=self._simulation_time(),
            physics_steps_advanced=steps_advanced,
            valid=self._valid.clone(),
            error_code=self._error_code.clone(),
        )

    def flush_logs(self) -> None:
        self._ensure_open()
        self._logger.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._logger.close()
        self._closed = True

    def __enter__(self) -> "SimulationEnvironment":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _initialize_truth(
        self, initial_state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        count = initial_state["position_n"].shape[0]
        zeros_3 = torch.zeros((count, 3), dtype=self.dtype, device=self.device)
        zeros_2 = torch.zeros((count, 2), dtype=self.dtype, device=self.device)
        zeros_grid = torch.zeros(
            (count, 3, 3), dtype=self.dtype, device=self.device
        )
        return {
            **{name: value.clone() for name, value in initial_state.items()},
            "linear_acceleration_n": zeros_3.clone(),
            "angular_acceleration_b": zeros_3.clone(),
            "motor_speed": zeros_2.clone(),
            "motor_thrust": zeros_2.clone(),
            "motor_torque": zeros_2.clone(),
            "servo_angle": zeros_3.clone(),
            "servo_effective_pwm": zeros_3.clone(),
            "servo_command_angle": zeros_3.clone(),
            "servo_target_angle": zeros_3.clone(),
            "servo_motion_direction": zeros_3.clone(),
            "servo_backlash_remaining": zeros_3.clone(),
            "direct_force_b": zeros_3.clone(),
            "grid_force_b": zeros_grid.clone(),
            "direct_moment_b": zeros_3.clone(),
            "grid_moment_b": zeros_grid.clone(),
            "motor_reaction_moment_b": zeros_3.clone(),
            "grid_effective_attenuation": torch.ones_like(zeros_3),
            "force_b": zeros_3.clone(),
            "moment_b": zeros_3.clone(),
        }

    def _validate_control_tensor(self, control: torch.Tensor) -> None:
        if not isinstance(control, torch.Tensor):
            raise TypeError("control must be a torch.Tensor")
        if control.shape != (self.parallel_count, 5):
            raise ValueError(f"control must have shape ({self.parallel_count}, 5)")
        if control.device != self.device:
            raise ValueError(f"control must be on {self.device}, got {control.device}")
        if control.dtype != self.dtype:
            raise ValueError(f"control must have dtype {self.dtype}, got {control.dtype}")

    def _validate_active_mask(self, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return torch.ones(self.parallel_count, dtype=torch.bool, device=self.device)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("active_mask must be a torch.Tensor")
        if mask.shape != (self.parallel_count,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError(
                f"active_mask must have shape ({self.parallel_count},), dtype bool, and device {self.device}"
            )
        return mask

    def _validate_reset_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("reset_mask must be a torch.Tensor")
        if (
            mask.shape != (self.parallel_count,)
            or mask.dtype != torch.bool
            or mask.device != self.device
        ):
            raise ValueError(
                f"reset_mask must have shape ({self.parallel_count},), "
                f"dtype bool, and device {self.device}"
            )
        return mask

    def _state_is_finite(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        finite = torch.ones(self.parallel_count, dtype=torch.bool, device=self.device)
        if set(state) != set(self._truth):
            raise RuntimeError("dynamics kernel changed the truth-state schema")
        for name, value in state.items():
            expected = self._truth[name]
            if value.shape != expected.shape or value.dtype != self.dtype or value.device != self.device:
                raise RuntimeError(f"dynamics kernel returned an invalid tensor for {name}")
            finite = finite & torch.isfinite(value).reshape(self.parallel_count, -1).all(dim=1)
        return finite

    def _commit_state(self, candidate: Mapping[str, torch.Tensor], mask: torch.Tensor) -> None:
        for name, current in self._truth.items():
            expanded = mask.reshape(self.parallel_count, *([1] * (current.ndim - 1)))
            self._truth[name] = torch.where(expanded, candidate[name], current)

    def _masked_copy(
        self, current: torch.Tensor, replacement: torch.Tensor, mask: torch.Tensor
    ) -> None:
        expanded = mask.reshape(
            self.parallel_count, *([1] * (current.ndim - 1))
        )
        current.copy_(torch.where(expanded, replacement, current))

    def _simulation_time(self) -> torch.Tensor:
        return self._physics_step.to(self.dtype) * self._config.timing.physics_dt

    def _append_log(
        self, active_mask: torch.Tensor, event_code: torch.Tensor
    ) -> None:
        record = {
            "physics_step": self._physics_step,
            "control_step": self._control_step,
            "sim_time_s": self._simulation_time(),
            "control": self._control,
            "active_mask": active_mask,
            "valid": self._valid,
            "error_code": self._error_code,
            "generation": self._generation,
            "event_code": event_code,
        }
        record.update({f"truth.{name}": value for name, value in self._truth.items()})
        record.update({f"sensor.{name}": value for name, value in self._sensors.items()})
        self._logger.append(record)

    def _validate_reset_compatibility(
        self,
        replacement: MaterializedConfig,
        replacement_truth: Mapping[str, torch.Tensor],
    ) -> None:
        current_timing = self._config.timing
        if (
            replacement.timing.physics_hz != current_timing.physics_hz
            or replacement.timing.control_hz != current_timing.control_hz
        ):
            raise ConfigurationError(
                "reset config must use the batch physics_hz and control_hz"
            )
        self._require_same_schema(
            self._parameters, replacement.parameters, "parameter"
        )
        self._require_same_schema(self._truth, replacement_truth, "truth-state")
        self._require_same_schema(
            self._sensors, replacement.sensor_state, "sensor-state"
        )
        if replacement.sensor_interpolation != self._config.sensor_interpolation:
            raise ConfigurationError(
                "reset config changes sensor interpolation modes"
            )
        try:
            self._sensor_kernel.validate_parameters(replacement.parameters)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc

    @staticmethod
    def _require_same_schema(
        current: Mapping[str, torch.Tensor],
        replacement: Mapping[str, torch.Tensor],
        label: str,
    ) -> None:
        if set(current) != set(replacement):
            raise ConfigurationError(f"reset config changes the {label} fields")
        for name, value in current.items():
            other = replacement[name]
            if value.shape[1:] != other.shape[1:]:
                raise ConfigurationError(
                    f"reset config changes {label} shape for {name}: "
                    f"{tuple(value.shape[1:])} != {tuple(other.shape[1:])}"
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise EnvironmentClosedError("simulation environment is closed")
