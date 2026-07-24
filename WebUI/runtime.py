"""Single-environment CPU runtime linking controller input, inference, and SimEnv.

The HTTP server is deliberately kept outside this module.  This module owns one
long-lived worker thread per session; simulation, observation construction, model
inference, recurrent state, and telemetry all remain on CPU.
"""

from __future__ import annotations

import copy
import math
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for source_root in (PROJECT_ROOT / "SimEnv" / "src", PROJECT_ROOT / "Train" / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


class RuntimeConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeOptions:
    device: str
    dtype: str
    observation_source: str
    cpu_threads: int
    telemetry_hz: float
    command_timeout_s: float
    episode_duration_s: float
    max_tilt_rad: float
    max_angular_rate_rad_s: float
    throttle_minimum: float
    throttle_maximum: float
    throttle_rise_per_s: float
    throttle_fall_per_s: float
    max_roll_rad: float
    max_pitch_rad: float
    max_yaw_rate_rad_s: float
    stick_time_constants: tuple[float, float, float]


def _mapping(node: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise RuntimeConfigurationError(f"{path} must be a mapping")
    return node


def _number(node: Mapping[str, Any], key: str, default: float) -> float:
    value = float(node.get(key, default))
    if not math.isfinite(value):
        raise RuntimeConfigurationError(f"{key} must be finite")
    return value


def parse_runtime_options(config: Mapping[str, Any]) -> RuntimeOptions:
    run = _mapping(config.get("run", {}), "run")
    environment = _mapping(config.get("environment", {}), "environment")
    task = _mapping(config.get("task", {}), "task")
    termination = _mapping(task.get("termination", {}), "task.termination")
    command = _mapping(config.get("command_source", {}), "command_source")
    params = _mapping(command.get("params", {}), "command_source.params")
    throttle = _mapping(params.get("throttle", {}), "command_source.params.throttle")
    slew = _mapping(throttle.get("slew_rate", {}), "command_source.params.throttle.slew_rate")
    sticks = _mapping(params.get("sticks", {}), "command_source.params.sticks")
    roll = _mapping(sticks.get("roll", {}), "command_source.params.sticks.roll")
    pitch = _mapping(sticks.get("pitch", {}), "command_source.params.sticks.pitch")
    yaw = _mapping(sticks.get("yaw", {}), "command_source.params.sticks.yaw")
    runtime = _mapping(config.get("runtime", {}), "runtime")
    options = RuntimeOptions(
        device=str(run.get("device", "cpu")),
        dtype=str(run.get("dtype", "float32")),
        observation_source=str(environment.get("observation_source", "truth")),
        cpu_threads=int(runtime.get("cpu_threads", 1)),
        telemetry_hz=_number(runtime, "telemetry_hz", 30.0),
        command_timeout_s=_number(runtime, "command_timeout_ms", 250.0) / 1000.0,
        episode_duration_s=_number(task, "episode_duration_s", 30.0),
        max_tilt_rad=_number(termination, "max_tilt_rad", 1.3),
        max_angular_rate_rad_s=_number(termination, "max_angular_rate_rad_s", 20.0),
        throttle_minimum=_number(throttle, "minimum", 0.20),
        throttle_maximum=_number(throttle, "maximum", 0.85),
        throttle_rise_per_s=_number(slew, "rise_per_s", 0.50),
        throttle_fall_per_s=_number(slew, "fall_per_s", 0.35),
        max_roll_rad=_number(roll, "limit_rad", 0.35),
        max_pitch_rad=_number(pitch, "limit_rad", 0.35),
        max_yaw_rate_rad_s=_number(yaw, "limit_rad_s", 1.50),
        stick_time_constants=(
            _number(roll, "time_constant_s", 0.20),
            _number(pitch, "time_constant_s", 0.20),
            _number(yaw, "time_constant_s", 0.30),
        ),
    )
    if options.device != "cpu":
        raise RuntimeConfigurationError("interactive WebUI runtime requires run.device=cpu")
    if options.dtype not in {"float32", "float64"}:
        raise RuntimeConfigurationError("runtime dtype must be float32 or float64")
    if options.observation_source not in {"truth", "sensor"}:
        raise RuntimeConfigurationError("observation_source must be truth or sensor")
    if options.telemetry_hz <= 0 or options.command_timeout_s <= 0:
        raise RuntimeConfigurationError("telemetry rate and command timeout must be positive")
    if options.cpu_threads <= 0 or options.cpu_threads > 64:
        raise RuntimeConfigurationError("runtime.cpu_threads must be between 1 and 64")
    if options.episode_duration_s <= 0:
        raise RuntimeConfigurationError("episode duration must be positive")
    if options.max_tilt_rad <= 0 or options.max_angular_rate_rad_s <= 0:
        raise RuntimeConfigurationError("safety limits must be positive")
    if any(value <= 0 for value in options.stick_time_constants):
        raise RuntimeConfigurationError("stick time constants must be positive")
    if options.throttle_rise_per_s < 0 or options.throttle_fall_per_s < 0:
        raise RuntimeConfigurationError("throttle slew rates must be non-negative")
    if not 0 <= options.throttle_minimum < options.throttle_maximum <= 1:
        raise RuntimeConfigurationError("throttle limits must satisfy 0 <= minimum < maximum <= 1")
    return options


def _model_config(raw: Mapping[str, Any]):
    from flight_train.config import ModelConfig

    node = _mapping(raw.get("model", {}), "model")
    model_type = str(node.get("type", "gru_actor_critic"))
    if model_type == "gru_actor_critic":
        encoder = tuple(int(value) for value in _mapping(node.get("encoder", {}), "model.encoder").get("hidden_sizes", [128, 128]))
        hidden = int(_mapping(node.get("recurrent", {}), "model.recurrent").get("hidden_size", 128))
        head_values = _mapping(node.get("actor_head", {}), "model.actor_head").get("hidden_sizes", [128])
        head = int(head_values[0])
        if not encoder or min((*encoder, hidden, head)) <= 0:
            raise RuntimeConfigurationError("GRU model dimensions must be positive")
        return ModelConfig(encoder, hidden, head, "gru")
    if model_type == "mlp_actor_critic":
        sizes = tuple(int(value) for value in node.get("hidden_sizes", [256, 256, 128]))
        if not sizes or min(sizes) <= 0:
            raise RuntimeConfigurationError("MLP hidden sizes must be positive")
        return ModelConfig(sizes, sizes[-1], sizes[-1], "mlp")
    raise RuntimeConfigurationError("unsupported runtime model type")


def _dynamic_randomization(raw: Mapping[str, Any]) -> tuple[Mapping[str, Mapping[str, Any]], int | None]:
    randomization = _mapping(raw.get("randomization", {}), "randomization")
    dynamic = _mapping(randomization.get("dynamic", {}), "randomization.dynamic")
    parameters = _mapping(dynamic.get("parameters", {}), "randomization.dynamic.parameters")
    normalized: dict[str, Mapping[str, Any]] = {}
    for name, spec in parameters.items():
        item = dict(_mapping(spec, f"randomization.dynamic.parameters.{name}"))
        # Train templates retain baseline for experiment documentation, while
        # SimEnv intentionally owns the nominal value in its physical config.
        item.pop("baseline", None)
        normalized[str(name)] = item
    seed = dynamic.get("seed")
    return normalized, None if seed is None else int(seed)


def _sim_timing(raw: Mapping[str, Any]) -> tuple[int, int]:
    timing = _mapping(raw.get("timing", {}), "timing")

    def frequency(name: str) -> int:
        node = _mapping(timing.get(name, {}), f"timing.{name}")
        value = node.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeConfigurationError(f"timing.{name}.value must be a positive integer")
        return value

    physics_hz, control_hz = frequency("physics_hz"), frequency("control_hz")
    if physics_hz % control_hz:
        raise RuntimeConfigurationError("control_hz must divide physics_hz")
    return physics_hz, control_hz


class CpuRuntimeSession:
    """Own one CPU simulation/controller loop and its lifecycle."""

    def __init__(
        self,
        simenv_yaml: str,
        test_yaml: str,
        checkpoint_path: str | Path,
        log_root: str | Path,
    ) -> None:
        import torch
        from flight_train.models import build_actor_critic
        from flight_train.recording import load_checkpoint
        from simenv import SimulationEnvironment

        self.id = str(uuid.uuid4())
        self._torch = torch
        self.raw_simenv = yaml.safe_load(simenv_yaml)
        self.raw_test = yaml.safe_load(test_yaml)
        if not isinstance(self.raw_simenv, Mapping):
            raise RuntimeConfigurationError("SimEnv YAML must contain a mapping")
        if not isinstance(self.raw_test, Mapping):
            raise RuntimeConfigurationError("controller test YAML must contain a mapping")
        self.options = parse_runtime_options(self.raw_test)
        self.device = torch.device(self.options.device)
        torch.set_num_threads(self.options.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only allows changing inter-op threads before the first
            # parallel operation. Intra-op threads above remain effective.
            pass
        self.dtype = torch.float32 if self.options.dtype == "float32" else torch.float64
        self._tempdir = tempfile.TemporaryDirectory(prefix=f"rl-flight-{self.id[:8]}-")
        temp = Path(self._tempdir.name)
        self.simenv_path = temp / "simenv.yaml"
        self.test_path = temp / "controller_test.yaml"
        runtime_log_root = Path(log_root).expanduser().resolve()
        runtime_log_root.mkdir(parents=True, exist_ok=True)
        normalized_simenv = copy.deepcopy(dict(self.raw_simenv))
        logging = normalized_simenv.setdefault("logging", {})
        if not isinstance(logging, dict):
            self._tempdir.cleanup()
            raise RuntimeConfigurationError("SimEnv logging must be a mapping")
        # Uploaded YAML no longer has a trustworthy source directory for
        # relative paths. Keep all interactive logs inside the server-owned
        # runtime log root instead of honoring a client-provided write target.
        logging["directory"] = str(runtime_log_root)
        self.simenv_path.write_text(yaml.safe_dump(normalized_simenv, sort_keys=False), encoding="utf-8")
        self.test_path.write_text(test_yaml, encoding="utf-8")
        self.environment = None
        try:
            dynamic_parameters, dynamic_seed = _dynamic_randomization(self.raw_test)
            self.environment = SimulationEnvironment.create(
                self.simenv_path,
                1,
                self.device,
                self.dtype,
                dynamic_randomization=dynamic_parameters,
                dynamic_seed=dynamic_seed,
            )
            if not self.environment.dynamics_implemented or not self.environment.sensors_implemented:
                raise RuntimeConfigurationError("SimEnv dynamics and sensors must both be implemented")
            self.log_directory = str(self.environment.log_directory)
            self.physics_hz, self.control_hz = _sim_timing(self.raw_simenv)
            self.control_period = 1.0 / self.control_hz
            self.telemetry_period = 1.0 / self.options.telemetry_hz
            self.max_episode_steps = max(1, round(self.options.episode_duration_s * self.control_hz))

            self.model = build_actor_critic(21, 4, _model_config(self.raw_test), self.device, self.dtype)
            checkpoint = load_checkpoint(Path(checkpoint_path).expanduser().resolve())
            actor_state = checkpoint.get("actor")
            if not isinstance(actor_state, Mapping):
                raise RuntimeConfigurationError("checkpoint does not contain an actor state")
            self.model.actor.load_state_dict(actor_state)
            self.model.actor.eval()
        except Exception:
            if self.environment is not None:
                self.environment.close()
            self._tempdir.cleanup()
            raise

        try:
            self.input_tensor = torch.zeros((1, 4), device=self.device, dtype=self.dtype)
            self.filtered_stick = torch.zeros((1, 3), device=self.device, dtype=self.dtype)
            self.target_yaw = torch.zeros(1, device=self.device, dtype=self.dtype)
            self.upper_throttle = torch.full((1, 1), self.options.throttle_minimum, device=self.device, dtype=self.dtype)
            self.previous_action = torch.zeros((1, 4), device=self.device, dtype=self.dtype)
            self.recurrent_state = None
            self.is_init = torch.ones((1, 1), device=self.device, dtype=torch.bool)
            self.episode_step = 0
            self.episode_id = 0
            self._filter_alpha = 1.0 - torch.exp(-torch.tensor(
                self.control_period, device=self.device, dtype=self.dtype
            ) / torch.tensor(self.options.stick_time_constants, device=self.device, dtype=self.dtype))
        except Exception:
            self.environment.close()
            self._tempdir.cleanup()
            raise

        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._state = "ready"
        self._fault: str | None = None
        self._command = (0.0, 0.0, 0.0, -1.0)
        self._command_sequence = -1
        self._command_received = 0.0
        self._telemetry_sequence = 0
        self._telemetry: dict[str, Any] | None = None
        self._control_steps = 0
        self._overruns = 0
        self._measured_control_hz = 0.0
        self._rate_sample_steps = 0
        self._rate_sample_at = time.monotonic()
        self._started_at: float | None = None
        self._last_telemetry_at = time.monotonic()
        self._reset_requested = False
        self._step_requests = 0
        self._thread = threading.Thread(target=self._run, name=f"cpu-runtime-{self.id[:8]}", daemon=True)
        try:
            self._thread.start()
        except Exception:
            self.environment.close()
            self._tempdir.cleanup()
            raise

    def update_command(self, sequence: int, channels: Mapping[str, Any]) -> bool:
        values = tuple(float(channels[name]) for name in ("roll", "pitch", "yaw", "throttle"))
        if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values):
            raise ValueError("controller channels must be finite values inside [-1,1]")
        with self._lock:
            if sequence <= self._command_sequence:
                return False
            self._command = values
            self._command_sequence = sequence
            self._command_received = time.monotonic()
        return True

    def start(self) -> None:
        with self._lock:
            if self._state == "closed":
                raise RuntimeError("runtime session is closed")
            if time.monotonic() - self._command_received > self.options.command_timeout_s:
                raise RuntimeError("a fresh controller frame is required before start")
            self._fault = None
            self._state = "running"
            self._started_at = self._started_at or time.monotonic()
        self._wake.set()

    def pause(self) -> None:
        with self._lock:
            if self._state != "closed":
                self._state = "paused"

    def reset(self) -> None:
        with self._lock:
            self._reset_requested = True
        self._wake.set()

    def step_once(self) -> None:
        with self._lock:
            if self._state in {"running", "closed", "faulted"}:
                raise RuntimeError("single-step requires a ready or paused runtime session")
            if time.monotonic() - self._command_received > self.options.command_timeout_s:
                raise RuntimeError("a fresh controller frame is required before single-step")
            self._state = "paused"
            self._step_requests += 1
        self._wake.set()

    def close(self) -> None:
        with self._lock:
            if self._state == "closed":
                return
            self._state = "closed"
        self._closed.set()
        self._wake.set()
        self._thread.join(timeout=5)
        self.environment.close()
        self._tempdir.cleanup()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.id,
                "state": self._state,
                "fault": self._fault,
                "device": str(self.device),
                "cpu_threads": self.options.cpu_threads,
                "dtype": str(self.dtype),
                "control_hz": self.control_hz,
                "physics_hz": self.physics_hz,
                "control_steps": self._control_steps,
                "episode_id": self.episode_id,
                "episode_step": self.episode_step,
                "command_sequence": self._command_sequence,
                "telemetry_sequence": self._telemetry_sequence,
                "loop_overruns": self._overruns,
                "measured_control_hz": self._measured_control_hz,
                "log_directory": self.log_directory,
            }

    def telemetry(self, after: int = -1) -> dict[str, Any] | None:
        with self._lock:
            if self._telemetry is None or self._telemetry_sequence <= after:
                return None
            return copy.deepcopy(self._telemetry)

    def _run(self) -> None:
        torch = self._torch
        deadline = time.perf_counter()
        try:
            while not self._closed.is_set():
                with self._lock:
                    state = self._state
                    reset_requested = self._reset_requested
                    self._reset_requested = False
                    single_step = state != "running" and self._step_requests > 0
                    if single_step:
                        self._step_requests -= 1
                    command = self._command
                    command_age = time.monotonic() - self._command_received
                if reset_requested:
                    self._reset_cpu()
                if state != "running" and not single_step:
                    self._wake.wait(.1)
                    self._wake.clear()
                    deadline = time.perf_counter()
                    continue
                if command_age > self.options.command_timeout_s:
                    with self._lock:
                        self._state = "paused"
                        self._fault = "controller_input_timeout"
                    continue
                deadline += self.control_period
                for index, value in enumerate(command):
                    self.input_tensor[0, index] = value
                publish_due = single_step or time.monotonic() - self._last_telemetry_at >= self.telemetry_period
                with torch.no_grad():
                    self._step_cpu(inspect_safety=publish_due)
                self._control_steps += 1
                if publish_due:
                    self._publish_telemetry()
                    self._last_telemetry_at = time.monotonic()
                if single_step:
                    deadline = time.perf_counter()
                    continue
                delay = deadline - time.perf_counter()
                if delay > 0:
                    self._closed.wait(delay)
                else:
                    self._overruns += 1
                    deadline = time.perf_counter()
        except Exception as error:
            with self._lock:
                self._fault = f"{type(error).__name__}: {error}"
                self._state = "faulted"

    def _step_cpu(self, *, inspect_safety: bool) -> None:
        torch = self._torch
        roll, pitch, yaw, throttle = self.input_tensor.unbind(dim=1)
        raw_stick = torch.stack((roll, pitch, yaw), dim=1)
        self.filtered_stick.add_(self._filter_alpha * (raw_stick - self.filtered_stick))
        self.target_yaw.add_(self.filtered_stick[:, 2] * self.options.max_yaw_rate_rad_s * self.control_period)
        self.target_yaw.copy_(torch.remainder(self.target_yaw + torch.pi, 2 * torch.pi) - torch.pi)
        target_attitude = self._euler_to_quaternion(
            self.filtered_stick[:, 0] * self.options.max_roll_rad,
            self.filtered_stick[:, 1] * self.options.max_pitch_rad,
            self.target_yaw,
        )
        desired_throttle = self.options.throttle_minimum + (throttle[:, None] + 1) * .5 * (
            self.options.throttle_maximum - self.options.throttle_minimum
        )
        lower = self.upper_throttle - self.options.throttle_fall_per_s * self.control_period
        upper = self.upper_throttle + self.options.throttle_rise_per_s * self.control_period
        self.upper_throttle.copy_(torch.maximum(torch.minimum(desired_throttle, upper), lower))

        truth = self.environment.observe("truth", (
            "position_n", "attitude_q_wb", "angular_velocity_b", "linear_acceleration_n", "motor_speed",
        )).values
        sensor = self.environment.observe("sensor").values
        attitude = truth["attitude_q_wb"]
        if self.options.observation_source == "sensor":
            if "attitude_q_wb" not in sensor:
                raise RuntimeError("sensor observation requested but SimEnv has no attitude_q_wb sensor")
            attitude = sensor["attitude_q_wb"]
        angular_velocity = sensor.get("gyro", truth["angular_velocity_b"])
        acceleration = sensor.get("accelerometer", truth["linear_acceleration_n"])
        motor_speed = sensor.get("motor_speed", truth["motor_speed"])
        observation = torch.cat((
            attitude,
            angular_velocity / self.options.max_angular_rate_rad_s,
            acceleration / 9.80665,
            motor_speed / 1800.0,
            target_attitude,
            self.upper_throttle * 2 - 1,
            self.previous_action,
        ), dim=1)
        action, self.recurrent_state = self.model.forward_step(observation, self.recurrent_state, self.is_init)
        self.is_init.zero_()
        action = action.clamp(-1, 1)
        command = torch.cat((self.upper_throttle, (action[:, :1] + 1) * .5, action[:, 1:]), dim=1)
        result = self.environment.advance(command)
        self.previous_action.copy_(action)
        self.episode_step += 1

        tilt = 2 * torch.acos(torch.sqrt((attitude[:, 0].square() + attitude[:, 3].square()).clamp(0, 1)))
        # Safety inspection is limited to the telemetry boundary (or the
        # deterministic episode timeout) to keep the hot path compact.
        inspect_safety = self.episode_step >= self.max_episode_steps or inspect_safety
        reset_mask = (~result.valid) | (tilt > self.options.max_tilt_rad) | (
            torch.linalg.vector_norm(angular_velocity, dim=1) > self.options.max_angular_rate_rad_s
        )
        should_reset = self.episode_step >= self.max_episode_steps
        if inspect_safety and not should_reset:
            should_reset = bool(reset_mask.any().item())
        if should_reset:
            if self.episode_step >= self.max_episode_steps:
                reset_mask = torch.ones_like(result.valid)
            self.environment.reset(reset_mask, self.simenv_path)
            self.recurrent_state = None
            self.previous_action.zero_()
            self.filtered_stick.zero_()
            self.target_yaw.zero_()
            self.is_init.fill_(True)
            self.episode_step = 0
            self.episode_id += 1

    def _reset_cpu(self) -> None:
        torch = self._torch
        with torch.no_grad():
            mask = torch.ones(1, device=self.device, dtype=torch.bool)
            self.environment.reset(mask, self.simenv_path)
            self.previous_action.zero_()
            self.filtered_stick.zero_()
            self.target_yaw.zero_()
            self.upper_throttle.fill_(self.options.throttle_minimum)
            self.recurrent_state = None
            self.is_init.fill_(True)
        self.episode_step = 0
        self.episode_id += 1

    def _publish_telemetry(self) -> None:
        now = time.monotonic()
        elapsed = now - self._rate_sample_at
        if elapsed > 0:
            self._measured_control_hz = (self._control_steps - self._rate_sample_steps) / elapsed
        self._rate_sample_steps = self._control_steps
        self._rate_sample_at = now
        fields = (
            "position_n", "velocity_n", "attitude_q_wb", "angular_velocity_b", "linear_acceleration_n",
            "motor_speed", "servo_angle", "grid_moment_b", "moment_b", "grid_force_b", "force_b",
        )
        truth = self.environment.observe("truth", fields).values
        cpu = {name: value.detach().tolist() for name, value in truth.items()}
        self._telemetry_sequence += 1
        packet = {
            "type": "telemetry",
            "session_id": self.id,
            "sequence": self._telemetry_sequence,
            "server_time_ns": time.time_ns(),
            "truth": cpu,
            "runtime": self.status(),
        }
        with self._lock:
            self._telemetry = packet

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        torch = __import__("torch")
        cr, sr = torch.cos(roll * .5), torch.sin(roll * .5)
        cp, sp = torch.cos(pitch * .5), torch.sin(pitch * .5)
        cy, sy = torch.cos(yaw * .5), torch.sin(yaw * .5)
        return torch.stack((
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ), dim=1)


class RuntimeRegistry:
    def __init__(self, log_root: str | Path) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, CpuRuntimeSession] = {}
        self._log_root = Path(log_root).expanduser().resolve()

    @staticmethod
    def capabilities() -> dict[str, Any]:
        try:
            import os
            import platform
            import torch
            return {
                "cpu_available": True,
                "processor": platform.processor() or platform.machine(),
                "logical_cpu_count": os.cpu_count(),
                "torch": torch.__version__,
                "devices": [{"type": "cpu", "name": platform.processor() or platform.machine()}],
                "runtime": "cpu-single-environment-v1",
            }
        except Exception as error:
            return {"cpu_available": False, "error": str(error), "devices": [], "runtime": "cpu-single-environment-v1"}

    def create(self, simenv_yaml: str, test_yaml: str, checkpoint_path: str) -> CpuRuntimeSession:
        with self._lock:
            if self._sessions:
                raise RuntimeError("only one interactive simulation session may exist at a time")
            session = CpuRuntimeSession(simenv_yaml, test_yaml, checkpoint_path, self._log_root)
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> CpuRuntimeSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as error:
                raise KeyError("runtime session not found") from error

    def close(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise KeyError("runtime session not found")
        session.close()

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()
