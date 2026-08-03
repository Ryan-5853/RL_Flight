"""Deployment-only neural inference package contract for the realtime WebUI.

This module intentionally has no dependency on the training framework.  A
deployment layer may register a loader which returns an object implementing the
protocol below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class InferencePackageMetadata:
    format_version: int
    package_id: str
    observation_dim: int
    action_dim: int
    output_mode: str
    recurrent: bool = False

    def validate(self) -> None:
        if self.format_version != 1:
            raise ValueError("unsupported inference package format_version")
        if not self.package_id:
            raise ValueError("inference package_id must not be empty")
        if self.observation_dim != 21:
            raise ValueError("realtime controller requires observation_dim=21")
        if self.output_mode == "residual_4":
            expected_actions = 4
        elif self.output_mode == "physical_5":
            expected_actions = 5
        elif self.output_mode == "coaxial_differential_cyclic_3":
            expected_actions = 3
        else:
            raise ValueError(
                "inference output_mode must be residual_4, physical_5, or "
                "coaxial_differential_cyclic_3"
            )
        if self.action_dim != expected_actions:
            raise ValueError(
                f"{self.output_mode} requires action_dim={expected_actions}"
            )


@runtime_checkable
class RealtimeInferencePackage(Protocol):
    """Minimal stateful inference component consumed by the realtime runtime."""

    metadata: InferencePackageMetadata

    def infer(
        self,
        observation: torch.Tensor,
        recurrent_state: Any | None,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        """Return one deterministic action and the next recurrent state."""

    def reset(self) -> None:
        """Clear package-owned recurrent or temporal state."""

    def warmup(self, observation: torch.Tensor) -> None:
        """Compile and allocate package runtime resources before the loop."""

    def close(self) -> None:
        """Release package-owned resources."""

    def describe(self) -> Mapping[str, Any]:
        """Return static deployment metadata safe to expose to the WebUI."""


InferencePackageLoader = Callable[
    [Path, torch.device, torch.dtype],
    RealtimeInferencePackage,
]


class InferenceModelAdapter:
    """Adapter from a deployment package to Controller's neural model surface."""

    def __init__(self, package: RealtimeInferencePackage) -> None:
        package.metadata.validate()
        self.package = package

    def forward_step(
        self,
        observation: torch.Tensor,
        recurrent_state: Any | None = None,
        is_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any | None]:
        if observation.shape != (1, self.package.metadata.observation_dim):
            raise ValueError(
                "inference package received an incompatible observation shape"
            )
        if is_init is None:
            is_init = torch.zeros(
                (1, 1), device=observation.device, dtype=torch.bool
            )
        action, next_state = self.package.infer(
            observation, recurrent_state, is_init
        )
        expected = (1, self.package.metadata.action_dim)
        if action.shape != expected:
            raise ValueError(
                f"inference package must return action shape {expected}"
            )
        if action.device != observation.device or action.dtype != observation.dtype:
            raise ValueError(
                "inference package action must match observation device/dtype"
            )
        return action, next_state

    def forward_control(
        self,
        state: Any,
        reference: Any,
        previous_action: torch.Tensor,
        recurrent_state: Any | None,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        method = getattr(self.package, "infer_control", None)
        if method is None:
            observation = torch.cat(
                (
                    state.attitude_q_wb,
                    state.angular_velocity_b / 20.0,
                    state.linear_acceleration_n / 9.80665,
                    state.motor_speed / 1800.0,
                    reference.target_attitude_q_wb,
                    reference.collective_command * 2.0 - 1.0,
                    previous_action,
                ),
                dim=1,
            )
            action, next_state = self.forward_step(
                observation,
                recurrent_state,
                is_init,
            )
        else:
            action, next_state = method(
                state,
                reference,
                previous_action,
                recurrent_state,
                is_init,
            )
        expected = (1, self.package.metadata.action_dim)
        if action.shape != expected:
            raise ValueError(
                f"inference package must return action shape {expected}"
            )
        return action, next_state

    def action_to_command(
        self,
        policy_action: torch.Tensor,
        external_action: torch.Tensor,
    ) -> torch.Tensor:
        method = getattr(self.package, "action_to_command", None)
        if method is None:
            if self.package.metadata.output_mode == "physical_5":
                return torch.cat(
                    (
                        policy_action[:, :2].clamp(0.0, 1.0),
                        policy_action[:, 2:].clamp(-1.0, 1.0),
                    ),
                    dim=1,
                )
            return torch.cat(
                (
                    external_action.clamp(0.0, 1.0),
                    (policy_action[:, :1] + 1.0) * 0.5,
                    policy_action[:, 1:],
                ),
                dim=1,
            )
        return method(policy_action, external_action)

    def control_diagnostics(self) -> Mapping[str, torch.Tensor]:
        method = getattr(self.package, "control_diagnostics", None)
        return {} if method is None else method()


class FlightDeployInferencePackage:
    """Adapter for the repository's training-independent Deploy bundle."""

    def __init__(
        self,
        path: Path,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if dtype != torch.float32:
            raise ValueError(
                "flight_deploy bundles currently require float32"
            )
        from flight_deploy import PolicyRuntime
        from flight_deploy.history import UniformHistoryBuffer

        self.path = path
        self.runtime = PolicyRuntime.load(path, device=device)
        if self.runtime.stateful:
            raise ValueError(
                "the current WebUI adapter requires a stateless deployment bundle"
            )
        manifest = self.runtime.bundle.manifest
        contract = manifest.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError("flight_deploy bundle has no control contract")
        if (
            contract.get("version") != "self_stabilize_v1"
            or contract.get("observation_profile")
            != "attitude_self_stabilize_21d_v3"
        ):
            raise ValueError(
                "WebUI requires the self_stabilize_v1 / "
                "attitude_self_stabilize_21d_v3 contract"
            )
        history = contract.get("observation_history")
        if not isinstance(history, Mapping) or history.get("mode") != "uniform":
            raise ValueError(
                "the current WebUI adapter requires uniform observation history"
            )
        frames = int(history.get("frames", 0))
        stride_steps = int(history.get("stride_steps", 0))
        if frames <= 0 or stride_steps <= 0:
            raise ValueError("invalid uniform observation history dimensions")
        if self.runtime.input_dim != 21 * frames:
            raise ValueError(
                "bundle runtime input width does not match 21-D history contract"
            )
        if self.runtime.output_dim != 4:
            raise ValueError(
                "self_stabilize_v1 requires four policy action outputs"
            )
        normalization = contract.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ValueError("flight_deploy bundle has no normalization contract")
        self.angular_velocity_scale = float(
            normalization["angular_velocity_rad_s"]
        )
        self.acceleration_scale = float(
            normalization["acceleration_m_s2"]
        )
        self.motor_speed_scale = float(
            normalization["motor_speed_rad_s"]
        )
        self.servo_angle_scale = float(
            normalization["servo_angle_rad"]
        )
        self.desired_yaw_rate_scale = float(
            normalization["desired_yaw_rate_rad_s"]
        )
        if min(
            self.angular_velocity_scale,
            self.acceleration_scale,
            self.motor_speed_scale,
            self.servo_angle_scale,
            self.desired_yaw_rate_scale,
        ) <= 0:
            raise ValueError("bundle normalization scales must be positive")
        transform = contract.get("action_transform")
        if (
            not isinstance(transform, Mapping)
            or transform.get("type") != "residual_around_trim"
        ):
            raise ValueError(
                "the current WebUI adapter requires residual_around_trim"
            )
        trim = tuple(float(value) for value in transform.get("trim_command", ()))
        scale = tuple(
            float(value) for value in transform.get("residual_scale", ())
        )
        if len(trim) != 4 or len(scale) != 4 or min(scale) <= 0:
            raise ValueError("bundle action trim/scale must contain four values")
        self.action_trim = torch.tensor(trim, device=device, dtype=dtype)
        self.action_scale = torch.tensor(scale, device=device, dtype=dtype)
        self.history = UniformHistoryBuffer(
            1,
            21,
            frames,
            stride_steps=stride_steps,
            device=device,
            dtype=dtype,
        )
        self.history_frames = frames
        self.history_stride_steps = stride_steps
        self.control_features_compiled = False
        self._control_features = self._build_control_features
        if device.type == "cpu":
            self._control_features = torch.compile(
                self._build_control_features,
                fullgraph=True,
                mode="reduce-overhead",
            )
            self.control_features_compiled = True
        self.metadata = InferencePackageMetadata(
            format_version=1,
            package_id=path.name,
            observation_dim=21,
            action_dim=4,
            output_mode="residual_4",
            recurrent=False,
        )
        self.metadata.validate()

    def infer(
        self,
        observation: torch.Tensor,
        recurrent_state: Any | None,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        del recurrent_state
        if observation.shape != (1, 21):
            raise ValueError("flight control base observation must have shape [1,21]")
        reset = bool(is_init.reshape(-1)[0].item())
        if reset or not bool(self.history.initialized.all()):
            self.history.reset(observation)
            runtime_observation = self.history.observation()
        else:
            runtime_observation = self.history.append(observation)
        return self.runtime.infer(runtime_observation), None

    def infer_control(
        self,
        state: Any,
        reference: Any,
        previous_action: torch.Tensor,
        recurrent_state: Any | None,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        frame = self._control_features(
            state.attitude_q_wb,
            state.angular_velocity_b,
            state.linear_acceleration_n,
            state.motor_speed,
            state.servo_angle,
            reference.target_attitude_q_wb,
            reference.target_angular_velocity_b,
            reference.collective_command,
            previous_action,
        )
        return self.infer(frame, recurrent_state, is_init)

    def _build_control_features(
        self,
        attitude_q_wb: torch.Tensor,
        angular_velocity_b: torch.Tensor,
        linear_acceleration_n: torch.Tensor,
        motor_speed: torch.Tensor,
        servo_angle: torch.Tensor,
        target_attitude_q_wb: torch.Tensor,
        target_angular_velocity_b: torch.Tensor,
        collective_command: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        current_euler = self._quaternion_to_euler(attitude_q_wb)
        target_euler = self._quaternion_to_euler(target_attitude_q_wb)
        roll_pitch_error = torch.atan2(
            torch.sin(target_euler[:, :2] - current_euler[:, :2]),
            torch.cos(target_euler[:, :2] - current_euler[:, :2]),
        )
        relative_attitude = self._euler_to_quaternion(
            roll_pitch_error[:, 0],
            roll_pitch_error[:, 1],
            torch.zeros_like(roll_pitch_error[:, 0]),
        )
        frame = torch.cat(
            (
                relative_attitude,
                angular_velocity_b / self.angular_velocity_scale,
                linear_acceleration_n / self.acceleration_scale,
                motor_speed / self.motor_speed_scale,
                servo_angle / self.servo_angle_scale,
                target_angular_velocity_b[:, 2:3]
                / self.desired_yaw_rate_scale,
                collective_command * 2.0 - 1.0,
                previous_action,
            ),
            dim=1,
        )
        return frame

    def action_to_command(
        self,
        policy_action: torch.Tensor,
        external_action: torch.Tensor,
    ) -> torch.Tensor:
        physical = (
            self.action_trim
            + self.action_scale * policy_action.clamp(-1.0, 1.0)
        )
        return torch.cat(
            (
                external_action.clamp(0.0, 1.0),
                physical[:, :1].clamp(0.0, 1.0),
                physical[:, 1:].clamp(-1.0, 1.0),
            ),
            dim=1,
        )

    def reset(self) -> None:
        self.history.values.zero_()
        self.history.index = self.history.capacity - 1
        self.history.initialized.zero_()

    def warmup(self, observation: torch.Tensor) -> None:
        is_init = torch.ones((1, 1), device=observation.device, dtype=torch.bool)
        for index in range(10):
            self.infer(
                observation,
                None,
                is_init if index == 0 else torch.zeros_like(is_init),
            )
        if observation.device.type == "cuda":
            torch.cuda.synchronize(observation.device)
        self.reset()

    def close(self) -> None:
        return None

    def describe(self) -> Mapping[str, Any]:
        return {
            "package_id": self.metadata.package_id,
            "backend": self.runtime.backend,
            "input_dim": self.runtime.input_dim,
            "output_dim": self.runtime.output_dim,
            "base_observation_dim": self.metadata.observation_dim,
            "history_mode": "uniform",
            "history_frames": self.history_frames,
            "history_stride_steps": self.history_stride_steps,
            "control_features_compiled": self.control_features_compiled,
            "bundle_path": str(self.path),
        }

    @staticmethod
    def _quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = q.unbind(dim=-1)
        return torch.stack(
            (
                torch.atan2(
                    2.0 * (w * x + y * z),
                    1.0 - 2.0 * (x.square() + y.square()),
                ),
                torch.asin(
                    (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
                ),
                torch.atan2(
                    2.0 * (w * z + x * y),
                    1.0 - 2.0 * (y.square() + z.square()),
                ),
            ),
            dim=-1,
        )

    @staticmethod
    def _euler_to_quaternion(
        roll: torch.Tensor,
        pitch: torch.Tensor,
        yaw: torch.Tensor,
    ) -> torch.Tensor:
        cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
        cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        result = torch.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dim=-1,
        )
        return result / result.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class AngularAccelerationCascadePackage:
    """Stateful attitude-PID wrapper around the exported 3-D inner policy."""

    def __init__(
        self,
        path: Path,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if dtype != torch.float32:
            raise ValueError("flight_deploy bundles currently require float32")
        from flight_deploy import PolicyRuntime
        from flight_deploy.history import UniformHistoryBuffer

        self.path = path
        self.runtime = PolicyRuntime.load(path, device=device)
        if self.runtime.stateful:
            raise ValueError("angular-acceleration cascade requires a stateless MLP")
        contract = self.runtime.bundle.manifest.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError("flight_deploy bundle has no control contract")
        if (
            contract.get("version") != "angular_acceleration_cascade_v1"
            or contract.get("observation_profile")
            != "angular_acceleration_allocated_inner_loop_21d_v2"
        ):
            raise ValueError("bundle is not an angular-acceleration cascade")
        history = self._mapping(contract, "observation_history")
        if history.get("mode") != "uniform":
            raise ValueError("angular-acceleration cascade requires uniform history")
        frames = int(history.get("frames", 0))
        stride_steps = int(history.get("stride_steps", 0))
        if frames <= 0 or stride_steps <= 0:
            raise ValueError("invalid uniform observation history dimensions")
        if self.runtime.input_dim != 21 * frames or self.runtime.output_dim != 3:
            raise ValueError("bundle network dimensions do not match cascade ABI")

        normalization = self._mapping(contract, "normalization")
        self.command_scale = self._tensor3(
            normalization,
            "desired_angular_acceleration_rad_s2",
            device,
            dtype,
        )
        actual_scale = self._tensor3(
            normalization,
            "actual_angular_acceleration_rad_s2",
            device,
            dtype,
        )
        if not torch.equal(self.command_scale, actual_scale):
            raise ValueError("desired and actual angular-acceleration scales differ")
        self.angular_velocity_scale = self._positive(
            normalization, "angular_velocity_rad_s"
        )
        self.acceleration_scale = self._positive(
            normalization, "acceleration_m_s2"
        )
        self.motor_speed_scale = self._positive(
            normalization, "motor_speed_rad_s"
        )
        self.servo_angle_scale = self._positive(
            normalization, "servo_angle_rad"
        )

        controller = self._mapping(contract, "controller")
        if controller.get("type") != "attitude_pid_angular_acceleration_cascade":
            raise ValueError("unsupported cascade controller type")
        self.required_control_hz = int(controller.get("control_hz", 0))
        if self.required_control_hz <= 0:
            raise ValueError("cascade control_hz must be positive")
        self.control_dt = 1.0 / float(self.required_control_hz)
        compatibility = contract.get("simulator_compatibility")
        self.required_simulator_fingerprint = None
        if compatibility is not None:
            if not isinstance(compatibility, Mapping):
                raise ValueError("simulator compatibility contract must be a mapping")
            if int(compatibility.get("fingerprint_version", 0)) != 1:
                raise ValueError("unsupported simulator compatibility fingerprint")
            fingerprint = str(compatibility.get("sha256", ""))
            if len(fingerprint) != 64:
                raise ValueError("invalid simulator compatibility fingerprint")
            self.required_simulator_fingerprint = fingerprint
        outer = self._mapping(controller, "outer_loop")
        self.proportional_gain = self._tensor3(
            outer, "proportional_gain", device, dtype, nonnegative=True
        )
        self.integral_gain = self._tensor3(
            outer, "integral_gain", device, dtype, nonnegative=True
        )
        self.derivative_gain = self._tensor3(
            outer, "derivative_gain", device, dtype, nonnegative=True
        )
        self.integral_limit = self._tensor3(
            outer, "integral_limit_rad_s", device, dtype
        )
        command_limit = self._tensor3(
            outer, "max_angular_acceleration_rad_s2", device, dtype
        )
        if not torch.equal(command_limit, self.command_scale):
            raise ValueError("outer-loop limit does not match observation scale")

        transform = self._mapping(contract, "action_transform")
        if transform.get("type") != "coaxial_differential_cyclic":
            raise ValueError("cascade requires coaxial differential/cyclic allocation")
        self.lower_motor_upper_ratio = self._positive(
            transform, "lower_motor_upper_ratio"
        )
        if self.lower_motor_upper_ratio > 1.0:
            raise ValueError("lower_motor_upper_ratio must not exceed one")
        trim = tuple(float(value) for value in transform.get("trim_command", ()))
        scale = tuple(float(value) for value in transform.get("residual_scale", ()))
        if len(trim) != 4 or len(scale) != 3 or min(scale) <= 0:
            raise ValueError("cascade action trim/scale dimensions are invalid")
        self.action_trim = torch.tensor(trim, device=device, dtype=dtype)
        self.action_scale = torch.tensor(scale, device=device, dtype=dtype)

        self.history = UniformHistoryBuffer(
            1,
            21,
            frames,
            stride_steps=stride_steps,
            device=device,
            dtype=dtype,
        )
        self.history_frames = frames
        self.history_stride_steps = stride_steps
        self.integral_error = torch.zeros((1, 3), device=device, dtype=dtype)
        self.previous_angular_velocity = torch.zeros_like(self.integral_error)
        self.estimator_initialized = torch.zeros(1, device=device, dtype=torch.bool)
        self.last_desired_angular_acceleration = torch.zeros_like(
            self.integral_error
        )
        self.last_actual_angular_acceleration = torch.zeros_like(
            self.integral_error
        )
        self.metadata = InferencePackageMetadata(
            format_version=1,
            package_id=path.name,
            observation_dim=21,
            action_dim=3,
            output_mode="coaxial_differential_cyclic_3",
            recurrent=False,
        )
        self.metadata.validate()

    def infer(
        self,
        observation: torch.Tensor,
        recurrent_state: Any | None,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        del recurrent_state
        if observation.shape != (1, 21):
            raise ValueError("cascade base observation must have shape [1,21]")
        reset = bool(is_init.reshape(-1)[0].item())
        if reset or not bool(self.history.initialized.all()):
            self.history.reset(observation)
            runtime_observation = self.history.observation()
        else:
            runtime_observation = self.history.append(observation)
        return self.runtime.infer(runtime_observation), None

    def infer_control(
        self,
        state: Any,
        reference: Any,
        previous_action: torch.Tensor,
        recurrent_state: Any | None,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        if previous_action.shape != (1, 3):
            raise ValueError("cascade previous action must have shape [1,3]")
        reset = bool(is_init.reshape(-1)[0].item())
        angular_velocity = state.angular_velocity_b
        if reset or not bool(self.estimator_initialized[0].item()):
            self.integral_error.zero_()
            actual_angular_acceleration = torch.zeros_like(angular_velocity)
            self.estimator_initialized.fill_(True)
        else:
            actual_angular_acceleration = (
                angular_velocity - self.previous_angular_velocity
            ) / self.control_dt
        self.previous_angular_velocity.copy_(angular_velocity)

        attitude_error = self._attitude_error_rotation_vector(
            state.attitude_q_wb,
            reference.target_attitude_q_wb,
        )
        candidate_integral = torch.clamp(
            self.integral_error + attitude_error * self.control_dt,
            min=-self.integral_limit,
            max=self.integral_limit,
        )
        self.integral_error.copy_(candidate_integral)
        rate_error = reference.target_angular_velocity_b - angular_velocity
        desired_angular_acceleration = torch.clamp(
            self.proportional_gain * attitude_error
            + self.integral_gain * self.integral_error
            + self.derivative_gain * rate_error,
            min=-self.command_scale,
            max=self.command_scale,
        )
        self.last_desired_angular_acceleration.copy_(
            desired_angular_acceleration
        )
        self.last_actual_angular_acceleration.copy_(actual_angular_acceleration)
        frame = torch.cat(
            (
                desired_angular_acceleration / self.command_scale,
                actual_angular_acceleration / self.command_scale,
                angular_velocity / self.angular_velocity_scale,
                state.linear_acceleration_n / self.acceleration_scale,
                state.motor_speed / self.motor_speed_scale,
                state.servo_angle / self.servo_angle_scale,
                reference.collective_command * 2.0 - 1.0,
                previous_action,
            ),
            dim=1,
        )
        return self.infer(frame, recurrent_state, is_init)

    def action_to_command(
        self,
        policy_action: torch.Tensor,
        external_action: torch.Tensor,
    ) -> torch.Tensor:
        if policy_action.shape != (1, 3):
            raise ValueError("cascade policy action must have shape [1,3]")
        bounded = policy_action.clamp(-1.0, 1.0)
        lower = torch.clamp(
            self.lower_motor_upper_ratio * external_action
            + self.action_scale[:1] * bounded[:, :1],
            0.0,
            1.0,
        )
        cyclic_a = self.action_scale[1] * bounded[:, 1:2]
        cyclic_b = self.action_scale[2] * bounded[:, 2:3]
        root_three = 3.0**0.5
        servos = torch.cat(
            (
                cyclic_a,
                -0.5 * cyclic_a + 0.5 * root_three * cyclic_b,
                -0.5 * cyclic_a - 0.5 * root_three * cyclic_b,
            ),
            dim=1,
        )
        servos = torch.clamp(servos + self.action_trim[1:], -1.0, 1.0)
        return torch.cat(
            (external_action.clamp(0.0, 1.0), lower, servos), dim=1
        )

    def control_diagnostics(self) -> Mapping[str, torch.Tensor]:
        return {
            "controller.desired_angular_acceleration_b": (
                self.last_desired_angular_acceleration
            ),
            "controller.actual_angular_acceleration_b": (
                self.last_actual_angular_acceleration
            ),
            "controller.angular_acceleration_error_b": (
                self.last_actual_angular_acceleration
                - self.last_desired_angular_acceleration
            ),
            "controller.attitude_pid_integral_error": self.integral_error,
        }

    def reset(self) -> None:
        self.history.values.zero_()
        self.history.index = self.history.capacity - 1
        self.history.initialized.zero_()
        self.integral_error.zero_()
        self.previous_angular_velocity.zero_()
        self.estimator_initialized.zero_()
        self.last_desired_angular_acceleration.zero_()
        self.last_actual_angular_acceleration.zero_()

    def warmup(self, observation: torch.Tensor) -> None:
        is_init = torch.ones((1, 1), device=observation.device, dtype=torch.bool)
        for index in range(10):
            self.infer(
                observation,
                None,
                is_init if index == 0 else torch.zeros_like(is_init),
            )
        if observation.device.type == "cuda":
            torch.cuda.synchronize(observation.device)
        self.reset()

    def close(self) -> None:
        return None

    def describe(self) -> Mapping[str, Any]:
        return {
            "package_id": self.metadata.package_id,
            "backend": self.runtime.backend,
            "controller": "attitude_pid_angular_acceleration_cascade",
            "input_dim": self.runtime.input_dim,
            "output_dim": self.runtime.output_dim,
            "base_observation_dim": 21,
            "history_mode": "uniform",
            "history_frames": self.history_frames,
            "history_stride_steps": self.history_stride_steps,
            "required_control_hz": self.required_control_hz,
            "required_simulator_fingerprint": (
                self.required_simulator_fingerprint
            ),
            "outer_loop": {
                "proportional_gain": self.proportional_gain.tolist(),
                "integral_gain": self.integral_gain.tolist(),
                "derivative_gain": self.derivative_gain.tolist(),
                "integral_limit_rad_s": self.integral_limit.tolist(),
                "max_angular_acceleration_rad_s2": self.command_scale.tolist(),
            },
            "bundle_path": str(self.path),
        }

    @staticmethod
    def _mapping(parent: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        value = parent.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"cascade contract {name} must be a mapping")
        return value

    @staticmethod
    def _positive(parent: Mapping[str, Any], name: str) -> float:
        value = float(parent[name])
        if value <= 0:
            raise ValueError(f"cascade contract {name} must be positive")
        return value

    @staticmethod
    def _tensor3(
        parent: Mapping[str, Any],
        name: str,
        device: torch.device,
        dtype: torch.dtype,
        *,
        nonnegative: bool = False,
    ) -> torch.Tensor:
        raw = parent.get(name)
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError(f"cascade contract {name} must have length three")
        value = torch.tensor(raw, device=device, dtype=dtype)
        valid = value >= 0 if nonnegative else value > 0
        if not bool(valid.all()):
            qualifier = "nonnegative" if nonnegative else "positive"
            raise ValueError(f"cascade contract {name} must be {qualifier}")
        return value

    @staticmethod
    def _quaternion_multiply(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        lw, lx, ly, lz = left.unbind(dim=-1)
        rw, rx, ry, rz = right.unbind(dim=-1)
        return torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )

    @classmethod
    def _attitude_error_rotation_vector(
        cls, current: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        current = current / current.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        conjugate = current.clone()
        conjugate[..., 1:] = -conjugate[..., 1:]
        error = cls._quaternion_multiply(conjugate, target)
        error = error / error.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        error = torch.where(error[..., :1] < 0, -error, error)
        vector = error[..., 1:]
        vector_norm = vector.norm(dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(
            vector_norm, error[..., :1].clamp_min(0.0)
        )
        scale = torch.where(
            vector_norm > 1e-8,
            angle / vector_norm.clamp_min(1e-8),
            torch.full_like(vector_norm, 2.0),
        )
        return vector * scale


def load_flight_deploy_package(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> RealtimeInferencePackage:
    """Default WebUI loader for a verified ``flight_deploy`` bundle."""

    from flight_deploy import DeploymentBundle

    bundle = DeploymentBundle.load(path)
    contract = bundle.manifest.get("contract")
    if (
        isinstance(contract, Mapping)
        and contract.get("version") == "angular_acceleration_cascade_v1"
    ):
        return AngularAccelerationCascadePackage(path, device, dtype)
    return FlightDeployInferencePackage(path, device, dtype)
