from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import torch

from .config import MaterializedConfig, load_and_materialize
from .dynamics import TensorDynamicsKernel
from .errors import ConfigurationError, EnvironmentClosedError
from .kernels import TensorSensorKernel
from .logging import NullTensorLogger, TensorChunkLogger
from .types import AdvanceResult, ErrorCode, Observation, ResetResult


class SimulationEnvironment:
    """Batched simulation interface with isolated parameters and state."""

    def __init__(
        self,
        materialized: MaterializedConfig,
        parallel_count: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        dynamic_parameter_names: tuple[str, ...] = (),
        dynamic_seed: int | None = None,
        logging_enabled: bool = True,
    ) -> None:
        self.parallel_count = parallel_count
        self.batch_shape = torch.Size([parallel_count])
        self.device = device
        self.dtype = dtype
        self.batch_id = str(uuid.uuid4())
        self.instance_ids = tuple(str(uuid.uuid4()) for _ in range(parallel_count))
        self._config = materialized
        self._parameters = dict(materialized.parameters)
        self._dynamic_parameter_names = dynamic_parameter_names
        self._dynamic_seed = dynamic_seed
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
        self._dynamics.refresh_parameters(
            self._parameters, materialized.timing.physics_dt
        )
        self._all_active = torch.ones(
            parallel_count, dtype=torch.bool, device=device
        )
        self._zero_event_code = torch.zeros(
            parallel_count, dtype=torch.int32, device=device
        )
        self._closed = False
        if logging_enabled:
            self._logger = TensorChunkLogger(
                materialized.logging,
                self.batch_id,
                self.instance_ids,
                materialized.source_path,
                materialized.raw,
                self._parameters,
                simulation_hz=materialized.timing.physics_hz,
            )
        else:
            self._logger = NullTensorLogger(
                materialized.logging.directory / self.batch_id
            )
        self._append_log(
            active_mask=torch.zeros_like(self._valid),
            event_code=torch.ones(parallel_count, dtype=torch.int32, device=device),
            force=True,
        )

    @classmethod
    def create(
        cls,
        config_path: str | Path,
        parallel_count: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        *,
        dynamic_randomization: Mapping[str, Mapping[str, Any]] | None = None,
        dynamic_seed: int | None = None,
        logging_enabled: bool = True,
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
        if dynamic_randomization:
            parameters = _apply_dynamic_randomization(
                materialized.parameters,
                dynamic_randomization,
                materialized.seed if dynamic_seed is None else dynamic_seed,
                parallel_count,
                resolved_device,
                dtype,
            )
            materialized = replace(
                materialized,
                seed=materialized.seed if dynamic_seed is None else dynamic_seed,
                parameters=parameters,
            )
        return cls(
            materialized,
            parallel_count,
            resolved_device,
            dtype,
            dynamic_parameter_names=tuple((dynamic_randomization or {}).keys()),
            dynamic_seed=dynamic_seed,
            logging_enabled=logging_enabled,
        )

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

    @property
    def log_directory(self) -> Path:
        """本批次完整物理时间线和 reset 快照所在目录。"""

        self._ensure_open()
        return self._logger.directory

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
        self,
        reset_mask: torch.Tensor,
        config_path: str | Path,
        *,
        static_parameters: Mapping[str, torch.Tensor] | None = None,
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
        if self._dynamic_parameter_names:
            parameters = dict(replacement.parameters)
            for name in self._dynamic_parameter_names:
                parameters[name] = self._parameters[name]
            replacement = replace(
                replacement,
                seed=(replacement.seed if self._dynamic_seed is None else self._dynamic_seed),
                parameters=parameters,
            )
        if static_parameters is not None and len(static_parameters) > 0:
            parameters = dict(replacement.parameters)
            for name, candidate in static_parameters.items():
                if name not in parameters:
                    raise ConfigurationError(f"unknown static parameter: {name}")
                expected = parameters[name]
                if not isinstance(candidate, torch.Tensor):
                    raise ConfigurationError(f"static parameter {name} must be a Tensor")
                if candidate.device != self.device or candidate.dtype != self.dtype:
                    raise ConfigurationError(
                        f"static parameter {name} must use {self.device}/{self.dtype}"
                    )
                if candidate.shape != expected.shape:
                    raise ConfigurationError(
                        f"static parameter {name} shape {tuple(candidate.shape)} "
                        f"does not match {tuple(expected.shape)}"
                    )
                parameters[name] = candidate
            replacement = replace(replacement, parameters=parameters)
        replacement_truth = self._initialize_truth(replacement.initial_state)
        self._validate_reset_compatibility(replacement, replacement_truth)

        for name, value in self._parameters.items():
            self._masked_copy(value, replacement.parameters[name], mask)
        for name, value in self._truth.items():
            self._masked_copy(value, replacement_truth[name], mask)
        self._dynamics.refresh_parameters(
            self._parameters, self._config.timing.physics_dt, mask
        )
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

        event_code = torch.where(
            mask,
            torch.full_like(self._error_code, 2),
            torch.full_like(self._error_code, -1),
        )
        reset_record = self._log_record(torch.zeros_like(self._valid), event_code)
        self._logger.record_reset(
            reset_mask=mask,
            generation=self._generation,
            previous_instance_ids=previous_instance_ids,
            instance_ids=self.instance_ids,
            config_path=replacement.source_path,
            parameters=replacement.parameters,
            timeline_record=(
                reset_record if self._logger.uses_sparse_reset_events else None
            ),
        )
        if not self._logger.uses_sparse_reset_events:
            self._logger.append(reset_record, force=True)
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

        # 物理与控制统一为 500 Hz：每次调用只执行一个 2 ms 仿真步。
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
        self._random_counters["motors"].add_(commit[:, None].to(torch.int64))
        steps_advanced = commit.to(torch.int64)
        self._physics_step = self._physics_step + steps_advanced
        self._control_step = self._control_step + steps_advanced
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
            event_code=self._zero_event_code,
        )

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

    def state_dict(self) -> Mapping[str, Any]:
        """导出决定未来数值轨迹的完整、版本化仿真状态。

        日志线程、batch UUID 和 instance UUID 只影响产物身份，不参与数值演化，
        因此不作为可恢复状态；源身份保留为审计元数据。恢复后的新环境继续写入
        自己的新日志目录，同时物理、执行器、传感器和随机流逐张量连续。
        """

        self._ensure_open()
        return {
            "schema_version": 1,
            "compatibility": {
                "parallel_count": self.parallel_count,
                "dtype": str(self.dtype),
                "physics_hz": self._config.timing.physics_hz,
                "control_hz": self._config.timing.control_hz,
                "config_sha256": self._state_config_sha256(),
                "dynamic_parameter_names": self._dynamic_parameter_names,
                "dynamic_seed": self._dynamic_seed,
            },
            "source_identity": {
                "batch_id": self.batch_id,
                "instance_ids": self.instance_ids,
            },
            "parameters": {name: value.clone() for name, value in self._parameters.items()},
            "truth": {name: value.clone() for name, value in self._truth.items()},
            "sensors": {name: value.clone() for name, value in self._sensors.items()},
            "sensor_kernel": self._sensor_kernel.state_dict(),
            "physics_step": self._physics_step.clone(),
            "control_step": self._control_step.clone(),
            "valid": self._valid.clone(),
            "error_code": self._error_code.clone(),
            "generation": self._generation.clone(),
            "instance_seeds": self._instance_seeds.clone(),
            "random_counters": {
                name: value.clone() for name, value in self._random_counters.items()
            },
            "control": self._control.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """严格校验并恢复完整仿真状态，允许 checkpoint 张量来自 CPU。"""

        self._ensure_open()
        if not isinstance(state, Mapping):
            raise TypeError("simulation state must be a mapping")
        if int(state.get("schema_version", -1)) != 1:
            raise ConfigurationError("unsupported simulation state schema")
        compatibility = state.get("compatibility")
        if not isinstance(compatibility, Mapping):
            raise ConfigurationError("simulation state compatibility metadata is missing")
        expected = {
            "parallel_count": self.parallel_count,
            "dtype": str(self.dtype),
            "physics_hz": self._config.timing.physics_hz,
            "control_hz": self._config.timing.control_hz,
            "config_sha256": self._state_config_sha256(),
            "dynamic_parameter_names": self._dynamic_parameter_names,
            "dynamic_seed": self._dynamic_seed,
        }
        actual = dict(compatibility)
        actual["dynamic_parameter_names"] = tuple(
            actual.get("dynamic_parameter_names", ())
        )
        actual_config_sha256 = actual.pop("config_sha256", None)
        expected_without_config = dict(expected)
        expected_without_config.pop("config_sha256")
        if (
            actual != expected_without_config
            or actual_config_sha256 not in self._compatible_state_config_sha256s()
        ):
            raise ConfigurationError(
                "simulation state is incompatible: "
                f"expected {expected}, got "
                f"{dict(actual, config_sha256=actual_config_sha256)}"
            )

        parameters = self._validated_tensor_mapping(
            state.get("parameters"), self._parameters, "parameter", require_finite=True
        )
        truth = self._validated_tensor_mapping(
            state.get("truth"), self._truth, "truth", require_finite=True
        )
        sensors = self._validated_tensor_mapping(
            state.get("sensors"), self._sensors, "sensor", require_finite=True
        )
        random_counters = self._validated_tensor_mapping(
            state.get("random_counters"), self._random_counters, "random counter"
        )
        scalar_tensors = {
            "physics_step": self._validated_tensor(state, "physics_step", self._physics_step),
            "control_step": self._validated_tensor(state, "control_step", self._control_step),
            "valid": self._validated_tensor(state, "valid", self._valid),
            "error_code": self._validated_tensor(state, "error_code", self._error_code),
            "generation": self._validated_tensor(state, "generation", self._generation),
            "instance_seeds": self._validated_tensor(state, "instance_seeds", self._instance_seeds),
            "control": self._validated_tensor(state, "control", self._control, require_finite=True),
        }
        try:
            self._sensor_kernel.validate_parameters(parameters)
            self._sensor_kernel.load_state_dict(state.get("sensor_kernel", {}))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(str(exc)) from exc

        for name, value in parameters.items():
            self._parameters[name].copy_(value)
        self._dynamics.refresh_parameters(
            self._parameters, self._config.timing.physics_dt
        )
        self._sensor_kernel.refresh_parameters(self._parameters)
        self._truth = {name: value.clone() for name, value in truth.items()}
        self._sensors = {name: value.clone() for name, value in sensors.items()}
        for name, value in random_counters.items():
            self._random_counters[name].copy_(value)
        self._physics_step.copy_(scalar_tensors["physics_step"])
        self._control_step.copy_(scalar_tensors["control_step"])
        self._valid.copy_(scalar_tensors["valid"])
        self._error_code.copy_(scalar_tensors["error_code"])
        self._generation.copy_(scalar_tensors["generation"])
        self._instance_seeds.copy_(scalar_tensors["instance_seeds"])
        self._control.copy_(scalar_tensors["control"])

        # 在新日志中留下明确的 resume 边界；不修改任何数值状态。
        self._append_log(
            active_mask=torch.zeros_like(self._valid),
            event_code=torch.full_like(self._error_code, 3),
            force=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._logger.close()
        finally:
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
        gravity_n = zeros_3.clone()
        gravity_n[:, 2] = 9.80665
        return {
            **{name: value.clone() for name, value in initial_state.items()},
            # 初始执行器速度和气动力均为零，因此初态质心立即处于自由落体；
            # 真值加速度从 NED 重力开始，而不是在第一个物理步前暂时为零。
            "linear_acceleration_n": gravity_n,
            "angular_acceleration_b": zeros_3.clone(),
            "motor_speed": zeros_2.clone(),
            "effective_motor_speed": zeros_2.clone(),
            "total_thrust": torch.zeros(count, dtype=self.dtype, device=self.device),
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
            return self._all_active
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

    def _state_config_sha256(self) -> str:
        raw = dict(self._config.raw)
        # 日志只决定产物位置和写盘策略，不改变仿真数值轨迹。
        raw.pop("logging", None)
        return self._config_sha256(raw)

    def _compatible_state_config_sha256s(self) -> set[str]:
        """返回当前数值配置及已发布状态格式可接受的配置摘要。"""

        raw = dict(self._config.raw)
        compatible = {
            self._state_config_sha256(),
            # schema v1 最初对完整配置取摘要，包括与状态无关的日志配置。
            self._config_sha256(raw),
        }
        logging = raw.get("logging")
        if isinstance(logging, Mapping) and "minimum_free_space_bytes" in logging:
            legacy_raw = dict(raw)
            legacy_logging = dict(logging)
            legacy_logging.pop("minimum_free_space_bytes")
            legacy_raw["logging"] = legacy_logging
            compatible.add(self._config_sha256(legacy_raw))
        return compatible

    @staticmethod
    def _config_sha256(raw: Mapping[str, Any]) -> str:
        resolved = json.dumps(
            raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(resolved.encode("utf-8")).hexdigest()

    def _validated_tensor_mapping(
        self,
        value: Any,
        reference: Mapping[str, torch.Tensor],
        label: str,
        *,
        require_finite: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(value, Mapping) or set(value) != set(reference):
            raise ConfigurationError(f"simulation state {label} fields are incompatible")
        return {
            name: self._validated_tensor(
                value, name, expected, require_finite=require_finite, label=label
            )
            for name, expected in reference.items()
        }

    def _validated_tensor(
        self,
        state: Mapping[str, Any],
        name: str,
        expected: torch.Tensor,
        *,
        require_finite: bool = False,
        label: str = "field",
    ) -> torch.Tensor:
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            raise ConfigurationError(f"simulation state {label} {name} must be a Tensor")
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ConfigurationError(
                f"simulation state {label} {name} has incompatible shape/dtype"
            )
        if require_finite and not bool(torch.isfinite(value).all().item()):
            raise ConfigurationError(f"simulation state {label} {name} is non-finite")
        return value.to(self.device)

    def _append_log(
        self,
        active_mask: torch.Tensor,
        event_code: torch.Tensor,
        *,
        force: bool = False,
    ) -> None:
        self._logger.append_lazy(
            lambda: self._log_record(active_mask, event_code),
            force=force,
        )

    def _log_record(
        self,
        active_mask: torch.Tensor,
        event_code: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        return record

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


def _apply_dynamic_randomization(
    baseline_parameters: Mapping[str, torch.Tensor],
    specs: Mapping[str, Mapping[str, Any]],
    seed: int,
    parallel_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Mapping[str, torch.Tensor]:
    """由 SimEnv 消费 v2 动态随机化规格并生成设备端参数。"""

    parameters = dict(baseline_parameters)
    for name, spec in specs.items():
        if name not in parameters:
            raise ConfigurationError(f"unknown dynamic parameter: {name}")
        distribution = spec.get("distribution")
        allowed = {
            "distribution", "seed_stream", "valid_range", "on_out_of_range", "unit"
        }
        if distribution in {"normal", "truncated_normal"}:
            allowed.add("stddev")
        elif distribution == "uniform":
            allowed.add("range")
        unknown = set(spec) - allowed
        if unknown:
            raise ConfigurationError(
                f"unknown dynamic randomization fields for {name}: {sorted(unknown)}"
            )
        if "seed_stream" not in spec:
            raise ConfigurationError(
                f"dynamic parameter {name} requires seed_stream"
            )
        # 标称值只来自环境配置；动态规格只能描述围绕它的分布。
        baseline = parameters[name]
        if baseline.shape[0] != parallel_count:
            raise ConfigurationError(f"dynamic parameter batch mismatch: {name}")
        generator = torch.Generator(device=device)
        digest = hashlib.blake2b(
            f"{seed}:{spec['seed_stream']}".encode(), digest_size=8
        ).digest()
        generator.manual_seed(int.from_bytes(digest, "little") & ((1 << 63) - 1))
        if distribution in {"normal", "truncated_normal"}:
            stddev = torch.as_tensor(
                spec["stddev"], device=device, dtype=dtype
            ).expand_as(baseline)
            sampled = baseline + torch.randn(
                baseline.shape, device=device, dtype=dtype, generator=generator
            ) * stddev
            if distribution == "truncated_normal":
                if "valid_range" not in spec:
                    raise ConfigurationError(
                        f"truncated_normal parameter {name} requires valid_range"
                    )
                low = torch.as_tensor(
                    spec["valid_range"][0], device=device, dtype=dtype
                ).expand_as(baseline)
                high = torch.as_tensor(
                    spec["valid_range"][1], device=device, dtype=dtype
                ).expand_as(baseline)
                invalid = (sampled < low) | (sampled > high)
                for _ in range(64):
                    candidate = baseline + torch.randn(
                        baseline.shape,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    ) * stddev
                    sampled = torch.where(invalid, candidate, sampled)
                    invalid = (sampled < low) | (sampled > high)
                if bool(invalid.any().item()):
                    raise ConfigurationError(
                        f"truncated_normal parameter {name} failed after 64 attempts"
                    )
        elif distribution == "uniform":
            low, high = spec["range"]
            low_tensor = torch.as_tensor(low, device=device, dtype=dtype).expand_as(
                baseline
            )
            high_tensor = torch.as_tensor(high, device=device, dtype=dtype).expand_as(
                baseline
            )
            sampled = low_tensor + torch.rand(
                baseline.shape, device=device, dtype=dtype, generator=generator
            ) * (high_tensor - low_tensor)
        else:
            raise ConfigurationError(
                f"unsupported dynamic distribution for {name}: {distribution}"
            )
        if "valid_range" in spec:
            low = torch.as_tensor(
                spec["valid_range"][0], device=device, dtype=dtype
            ).expand_as(sampled)
            high = torch.as_tensor(
                spec["valid_range"][1], device=device, dtype=dtype
            ).expand_as(sampled)
            if distribution != "truncated_normal" and spec.get("on_out_of_range", "fail") == "clamp":
                sampled = torch.maximum(torch.minimum(sampled, high), low)
            elif bool(torch.any((sampled < low) | (sampled > high)).item()):
                raise ConfigurationError(
                    f"dynamic parameter {name} exceeded valid_range"
                )
        if not bool(torch.isfinite(sampled).all().item()):
            raise ConfigurationError(
                f"dynamic parameter {name} produced non-finite values"
            )
        parameters[name] = sampled
    return parameters
