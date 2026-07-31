"""Single-environment CPU runtime linking controller input, inference, and SimEnv.

The HTTP server is deliberately kept outside this module.  This module owns one
long-lived worker thread per session; simulation, observation construction, model
inference, recurrent state, and telemetry all remain on CPU.
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for source_root in (
    PROJECT_ROOT / "Controller" / "src",
    PROJECT_ROOT / "SimEnv" / "src",
    PROJECT_ROOT / "Train" / "src",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


class RuntimeConfigurationError(ValueError):
    pass


class RolloutCancelled(RuntimeError):
    """Raised internally when a queued/offline rollout is cancelled."""


def _finite_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class RuntimeOptions:
    device: str
    dtype: str
    observation_source: str
    cpu_threads: int
    execution_hz: float
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
        execution_hz=_number(runtime, "execution_hz", 80.0),
        telemetry_hz=_number(runtime, "telemetry_hz", 30.0),
        command_timeout_s=_number(runtime, "command_timeout_ms", 5000.0) / 1000.0,
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
    if (
        options.execution_hz <= 0
        or options.telemetry_hz <= 0
        or options.command_timeout_s <= 0
    ):
        raise RuntimeConfigurationError(
            "execution rate, telemetry rate and command timeout must be positive"
        )
    if options.telemetry_hz > options.execution_hz:
        raise RuntimeConfigurationError(
            "runtime.telemetry_hz must not exceed runtime.execution_hz"
        )
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
    if physics_hz != 500 or control_hz != 500:
        raise RuntimeConfigurationError(
            "the current single-step SimEnv requires physics_hz=500 "
            "and control_hz=500"
        )
    return physics_hz, control_hz


class CpuRuntimeSession:
    """Own one CPU simulation/controller loop and its lifecycle."""

    def __init__(
        self,
        simenv_yaml: str,
        test_yaml: str,
        checkpoint_path: str | Path | None,
        log_root: str | Path,
    ) -> None:
        import torch
        from flight_controller import ControllerContext, create_controller
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
        sensors = normalized_simenv.get("sensors", {})
        if not isinstance(sensors, dict):
            self._tempdir.cleanup()
            raise RuntimeConfigurationError("SimEnv sensors must be a mapping")
        # Current SimEnv accepts only explicit linear interpolation for a
        # fractional physics-step delay. Supplying it for every WebUI sensor is
        # also valid for integer/zero delays and keeps imported legacy configs
        # compatible with the fixed 500 Hz single-step interface.
        for name, sensor in sensors.items():
            if not isinstance(sensor, dict):
                self._tempdir.cleanup()
                raise RuntimeConfigurationError(
                    f"SimEnv sensor {name} must be a mapping"
                )
            sensor.setdefault("interpolation", "linear")
        normalized_simenv_yaml = yaml.safe_dump(
            normalized_simenv,
            sort_keys=False,
        )
        self.configuration_id = hashlib.sha256(
            normalized_simenv_yaml.encode("utf-8")
        ).hexdigest()[:12]
        self.simenv_path.write_text(
            normalized_simenv_yaml,
            encoding="utf-8",
        )
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
            self.execution_hz = min(
                self.options.execution_hz,
                float(self.control_hz),
            )
            self.execution_period = 1.0 / self.execution_hz
            self.telemetry_period = 1.0 / self.options.telemetry_hz
            self.max_episode_steps = max(1, round(self.options.episode_duration_s * self.control_hz))

            controller_node = self.raw_test.get("controller", {"type": "neural"})
            controller_config = dict(_mapping(controller_node, "controller"))
            controller_type = str(controller_config.get("type", "neural"))
            neural_model = None
            if controller_type == "neural":
                if checkpoint_path is None:
                    raise RuntimeConfigurationError(
                        "neural controller requires runtime.checkpoint_path"
                    )
                from flight_train.models import build_actor_critic
                from flight_train.recording import load_checkpoint

                neural_model = build_actor_critic(
                    21,
                    4,
                    _model_config(self.raw_test),
                    self.device,
                    self.dtype,
                )
                checkpoint = load_checkpoint(
                    Path(checkpoint_path).expanduser().resolve()
                )
                actor_state = checkpoint.get("actor")
                if not isinstance(actor_state, Mapping):
                    raise RuntimeConfigurationError(
                        "checkpoint does not contain an actor state"
                    )
                neural_model.actor.load_state_dict(actor_state)
                neural_model.actor.eval()
                params = controller_config.setdefault("params", {})
                if not isinstance(params, dict):
                    raise RuntimeConfigurationError(
                        "controller.params must be a mapping"
                    )
                params.setdefault(
                    "maximum_angular_rate_rad_s",
                    self.options.max_angular_rate_rad_s,
                )
            self.controller = create_controller(
                controller_config,
                ControllerContext(
                    batch_size=1,
                    device=self.device,
                    dtype=self.dtype,
                    control_dt=self.control_period,
                    parameters=self.environment.parameters,
                ),
                neural_model=neural_model,
            )
            parameters = self.environment.parameters
            self.effective_configuration = {
                "id": self.configuration_id,
                "body": {
                    "mass": float(
                        parameters["body.mass"][0].detach().cpu().item()
                    ),
                    "center_of_mass_b": parameters[
                        "body.center_of_mass_b"
                    ][0].detach().cpu().tolist(),
                    "inertia_diagonal_b": parameters[
                        "body.inertia_diagonal_b"
                    ][0].detach().cpu().tolist(),
                },
                "controller_parameter_source": (
                    "checkpoint"
                    if controller_type == "neural"
                    else "environment"
                ),
            }
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
            initial_truth = self.environment.observe(
                "truth", ("position_n",)
            ).values
            self.target_position_n = initial_truth["position_n"].clone()
            self.target_velocity_n = torch.zeros(
                (1, 3), device=self.device, dtype=self.dtype
            )
            self.last_controller_output = None
            self.last_reference = None
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
        self._failsafe_command = (0.0, 0.0, 0.0, -1.0)
        self._command_sequence = -1
        self._command_trace: dict[str, Any] | None = None
        self._pending_latency_trace: dict[str, Any] | None = None
        # A newly created session owns a neutral virtual controller frame.
        # This lets PID/LQR demonstrations start without a physical gamepad,
        # while the normal timeout still protects an already running session.
        self._command_received = time.monotonic()
        self._telemetry_sequence = 0
        self._telemetry: dict[str, Any] | None = None
        self._control_steps = 0
        self._overruns = 0
        self._measured_control_hz = 0.0
        self._effective_execution_hz = self.execution_hz
        self._step_compute_ema_s = 0.0
        self._input_stale = False
        self._input_timeout_events = 0
        self._rate_sample_steps = 0
        self._rate_sample_at = time.monotonic()
        self._started_at: float | None = None
        self._last_telemetry_at = time.monotonic()
        self._reset_requested = False
        self._step_requests = 0
        self._last_reset: dict[str, Any] | None = None
        self._reset_counts: dict[str, int] = {}
        self._thread = threading.Thread(target=self._run, name=f"cpu-runtime-{self.id[:8]}", daemon=True)
        try:
            self._thread.start()
        except Exception:
            self.environment.close()
            self._tempdir.cleanup()
            raise

    def _apply_command_locked(
        self,
        sequence: int,
        channels: Mapping[str, Any],
        *,
        trace: Mapping[str, Any] | None = None,
        server_received_ns: int | None = None,
    ) -> bool:
        values = tuple(
            float(channels[name])
            for name in ("roll", "pitch", "yaw", "throttle")
        )
        if not all(
            math.isfinite(value) and -1.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError(
                "controller channels must be finite values inside [-1,1]"
            )
        if sequence <= self._command_sequence:
            return False
        self._command = values
        self._command_sequence = sequence
        self._command_received = time.monotonic()
        self._input_stale = False
        trace = trace or {}
        self._command_trace = {
            "transport_sequence": sequence,
            "source_sequence": int(trace.get("source_sequence", -1)),
            "input_source": str(trace.get("input_source", "unknown"))[:32],
            "input_captured_epoch_ms": _finite_float_or_none(
                trace.get("input_captured_epoch_ms")
            ),
            "hardware_updated_epoch_ms": _finite_float_or_none(
                trace.get("hardware_updated_epoch_ms")
            ),
            "gamepad_poll_interval_ms": _finite_float_or_none(
                trace.get("gamepad_poll_interval_ms")
            ),
            "client_send_epoch_ms": _finite_float_or_none(
                trace.get("client_send_epoch_ms")
            ),
            "server_control_received_ns": int(
                server_received_ns or time.time_ns()
            ),
            "command_applied_ns": time.time_ns(),
        }
        return True

    def update_command(
        self,
        sequence: int,
        channels: Mapping[str, Any],
        *,
        trace: Mapping[str, Any] | None = None,
        server_received_ns: int | None = None,
    ) -> bool:
        with self._lock:
            return self._apply_command_locked(
                sequence,
                channels,
                trace=trace,
                server_received_ns=server_received_ns,
            )

    def start(
        self,
        sequence: int | None = None,
        channels: Mapping[str, Any] | None = None,
        *,
        trace: Mapping[str, Any] | None = None,
        server_received_ns: int | None = None,
    ) -> bool:
        with self._lock:
            if self._state == "closed":
                raise RuntimeError("runtime session is closed")
            accepted = True
            if sequence is not None or channels is not None:
                if sequence is None or channels is None:
                    raise ValueError(
                        "start requires both controller sequence and channels"
                    )
                accepted = self._apply_command_locked(
                    sequence,
                    channels,
                    trace=trace,
                    server_received_ns=server_received_ns,
                )
                if not accepted:
                    raise RuntimeError(
                        "the controller frame supplied to start is stale"
                    )
            if time.monotonic() - self._command_received > self.options.command_timeout_s:
                raise RuntimeError("a fresh controller frame is required before start")
            self._fault = None
            self._state = "running"
            self._started_at = self._started_at or time.monotonic()
        self._wake.set()
        return accepted

    def pause(self) -> None:
        with self._lock:
            if self._state != "closed":
                self._state = "paused"

    def reset(self) -> None:
        with self._lock:
            self._reset_requested = True
        self._wake.set()

    def step_once(
        self,
        sequence: int | None = None,
        channels: Mapping[str, Any] | None = None,
        *,
        trace: Mapping[str, Any] | None = None,
        server_received_ns: int | None = None,
    ) -> bool:
        with self._lock:
            if self._state in {"running", "closed", "faulted"}:
                raise RuntimeError("single-step requires a ready or paused runtime session")
            accepted = True
            if sequence is not None or channels is not None:
                if sequence is None or channels is None:
                    raise ValueError(
                        "single-step requires both controller sequence and channels"
                    )
                accepted = self._apply_command_locked(
                    sequence,
                    channels,
                    trace=trace,
                    server_received_ns=server_received_ns,
                )
                if not accepted:
                    raise RuntimeError(
                        "the controller frame supplied to single-step is stale"
                    )
            if time.monotonic() - self._command_received > self.options.command_timeout_s:
                raise RuntimeError("a fresh controller frame is required before single-step")
            self._state = "paused"
            self._step_requests += 1
        self._wake.set()
        return accepted

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
                "target_execution_hz": self.execution_hz,
                "effective_execution_hz": self._effective_execution_hz,
                "control_steps": self._control_steps,
                "episode_id": self.episode_id,
                "episode_step": self.episode_step,
                "last_reset": copy.deepcopy(self._last_reset),
                "reset_counts": dict(self._reset_counts),
                "controller_input": {
                    "sequence": self._command_sequence,
                    "channels": {
                        name: value
                        for name, value in zip(
                            ("roll", "pitch", "yaw", "throttle"),
                            self._command,
                        )
                    },
                    "age_ms": max(
                        0.0,
                        (time.monotonic() - self._command_received) * 1000.0,
                    ),
                    "stale": self._input_stale,
                    "timeout_events": self._input_timeout_events,
                },
                "telemetry_sequence": self._telemetry_sequence,
                "loop_overruns": self._overruns,
                "measured_control_hz": self._measured_control_hz,
                "real_time_factor": (
                    self._measured_control_hz / self.control_hz
                ),
                "configuration": copy.deepcopy(
                    self.effective_configuration
                ),
                "log_directory": self.log_directory,
                "controller": self.controller.describe(),
            }

    def telemetry(self, after: int = -1) -> dict[str, Any] | None:
        with self._lock:
            if self._telemetry is None or self._telemetry_sequence <= after:
                return None
            return copy.deepcopy(self._telemetry)

    def _run(self) -> None:
        torch = self._torch
        try:
            while not self._closed.is_set():
                with self._lock:
                    state = self._state
                    reset_requested = self._reset_requested
                    self._reset_requested = False
                    single_step = state != "running" and self._step_requests > 0
                    if single_step:
                        self._step_requests -= 1
                    command_age = time.monotonic() - self._command_received
                    input_stale = (
                        command_age > self.options.command_timeout_s
                    )
                    if input_stale and not self._input_stale:
                        self._input_timeout_events += 1
                    self._input_stale = input_stale
                    command = (
                        self._failsafe_command
                        if input_stale
                        else self._command
                    )
                    command_trace = copy.deepcopy(self._command_trace)
                    # A command trace describes the first simulation step that
                    # consumes that command. Reusing it on subsequent steps
                    # reports command age as scheduler delay on high-RTT links.
                    self._command_trace = None
                if reset_requested:
                    self._reset_cpu()
                if state != "running" and not single_step:
                    self._wake.wait(.1)
                    self._wake.clear()
                    continue
                step_started = time.perf_counter()
                step_started_ns = time.time_ns()
                for index, value in enumerate(command):
                    self.input_tensor[0, index] = value
                publish_due = single_step or time.monotonic() - self._last_telemetry_at >= self.telemetry_period
                with torch.no_grad():
                    step_timing = self._step_cpu(inspect_safety=publish_due)
                step_finished_ns = time.time_ns()
                step_trace = command_trace or {
                    "transport_sequence": self._command_sequence,
                    "input_source": "failsafe" if input_stale else "unknown",
                }
                step_trace.update(step_timing)
                step_trace.update({
                    "backend_step_started_ns": step_started_ns,
                    "backend_step_finished_ns": step_finished_ns,
                    "input_stale": input_stale,
                })
                if command_trace is not None:
                    # Keep the newest first-consumption sample until the next
                    # configured telemetry publication. Intermediate commands
                    # may be superseded, but no stale timestamp is reused.
                    self._pending_latency_trace = step_trace
                self._control_steps += 1
                if publish_due:
                    latency_trace = self._pending_latency_trace or step_trace
                    self._pending_latency_trace = None
                    self._publish_telemetry(latency_trace)
                    self._last_telemetry_at = time.monotonic()
                if single_step:
                    continue
                compute_s = time.perf_counter() - step_started
                if self._step_compute_ema_s <= 0:
                    self._step_compute_ema_s = compute_s
                else:
                    self._step_compute_ema_s = (
                        .9 * self._step_compute_ema_s + .1 * compute_s
                    )
                # Reserve roughly 20% wall-clock headroom for HTTP control and
                # telemetry threads. If inference/physics is slower than the
                # configured execution rate, reduce wall execution rate rather
                # than spinning and starving communication.
                adaptive_period = max(
                    self.execution_period,
                    self._step_compute_ema_s / .8,
                )
                self._effective_execution_hz = 1.0 / adaptive_period
                delay = adaptive_period - compute_s
                if delay > 0:
                    self._closed.wait(delay)
                else:
                    self._overruns += 1
        except Exception as error:
            with self._lock:
                self._fault = f"{type(error).__name__}: {error}"
                self._state = "faulted"

    def _step_cpu(
        self,
        *,
        inspect_safety: bool,
        reset_on_termination: bool = True,
    ) -> dict[str, Any]:
        torch = self._torch
        from flight_controller import ControllerReference, ControllerState

        preparation_started_ns = time.time_ns()
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
            "position_n", "velocity_n", "attitude_q_wb", "angular_velocity_b",
            "linear_acceleration_n", "motor_speed", "servo_angle",
        )).values
        sensor = self.environment.observe("sensor").values
        controller_values = dict(truth)
        if self.controller.controller_type == "neural":
            # Preserve the deployed 21-D policy contract: gyro,
            # accelerometer and motor-speed channels use configured sensors
            # even when attitude itself is selected from truth.
            controller_values["angular_velocity_b"] = sensor.get(
                "gyro", truth["angular_velocity_b"]
            )
            controller_values["linear_acceleration_n"] = sensor.get(
                "accelerometer", truth["linear_acceleration_n"]
            )
            controller_values["motor_speed"] = sensor.get(
                "motor_speed", truth["motor_speed"]
            )
        if self.options.observation_source == "sensor":
            if "attitude_q_wb" not in sensor:
                raise RuntimeError("sensor observation requested but SimEnv has no attitude_q_wb sensor")
            controller_values["attitude_q_wb"] = sensor["attitude_q_wb"]
            controller_values["angular_velocity_b"] = sensor.get(
                "gyro", truth["angular_velocity_b"]
            )
            controller_values["linear_acceleration_n"] = sensor.get(
                "accelerometer", truth["linear_acceleration_n"]
            )
            controller_values["motor_speed"] = sensor.get(
                "motor_speed", truth["motor_speed"]
            )
        state = ControllerState.from_truth(controller_values)
        target_rate = torch.zeros((1, 3), device=self.device, dtype=self.dtype)
        target_rate[:, 2] = (
            self.filtered_stick[:, 2] * self.options.max_yaw_rate_rad_s
        )
        reference = ControllerReference(
            target_position_n=self.target_position_n,
            target_velocity_n=self.target_velocity_n,
            target_attitude_q_wb=target_attitude,
            target_angular_velocity_b=target_rate,
            collective_command=self.upper_throttle,
        )
        controller_started_perf_ns = time.perf_counter_ns()
        controller_started_ns = time.time_ns()
        controller_output = self.controller.step(state, reference)
        controller_finished_ns = time.time_ns()
        controller_elapsed_ns = (
            time.perf_counter_ns() - controller_started_perf_ns
        )
        environment_started_perf_ns = time.perf_counter_ns()
        environment_started_ns = time.time_ns()
        result = self.environment.advance(controller_output.command)
        environment_finished_ns = time.time_ns()
        environment_elapsed_ns = (
            time.perf_counter_ns() - environment_started_perf_ns
        )
        self.last_controller_output = controller_output
        self.last_reference = reference
        self.episode_step += 1

        attitude = state.attitude_q_wb
        angular_velocity = state.angular_velocity_b
        tilt = 2 * torch.acos(torch.sqrt((attitude[:, 0].square() + attitude[:, 3].square()).clamp(0, 1)))
        # Safety inspection is limited to the telemetry boundary (or the
        # deterministic episode timeout) to keep the hot path compact.
        inspect_safety = self.episode_step >= self.max_episode_steps or inspect_safety
        angular_rate_norm = torch.linalg.vector_norm(
            angular_velocity, dim=1
        )
        invalid_mask = ~result.valid
        tilt_mask = tilt > self.options.max_tilt_rad
        angular_rate_mask = (
            angular_rate_norm > self.options.max_angular_rate_rad_s
        )
        reset_mask = invalid_mask | tilt_mask | angular_rate_mask
        episode_timeout = self.episode_step >= self.max_episode_steps
        should_reset = episode_timeout
        if inspect_safety and not should_reset:
            should_reset = bool(reset_mask.any().item())
        termination: dict[str, Any] | None = None
        if should_reset:
            if bool(invalid_mask.any().item()):
                reason = "simenv_invalid"
            elif bool(tilt_mask.any().item()):
                reason = "tilt_limit"
            elif bool(angular_rate_mask.any().item()):
                reason = "angular_rate_limit"
            else:
                reason = "episode_timeout"
            termination = {
                "reason": reason,
                "episode_id": self.episode_id,
                "episode_step": self.episode_step,
                "tilt_rad": float(tilt[0].item()),
                "angular_rate_rad_s": float(
                    angular_rate_norm[0].item()
                ),
                "simenv_valid": bool(result.valid[0].item()),
                "simenv_error_code": int(
                    result.error_code[0].item()
                ),
            }
            if reset_on_termination:
                self._record_reset(reason, termination)
                if episode_timeout:
                    reset_mask = torch.ones_like(result.valid)
                self.environment.reset(reset_mask, self.simenv_path)
                self.controller.reset(reset_mask)
                self.filtered_stick.zero_()
                self.target_yaw.zero_()
                self.target_position_n.copy_(
                    self.environment.observe(
                        "truth", ("position_n",)
                    ).values["position_n"]
                )
                self.target_velocity_n.zero_()
                self.last_controller_output = None
                self.last_reference = None
                self.episode_step = 0
                self.episode_id += 1
        return {
            "preparation_started_ns": preparation_started_ns,
            "controller_started_ns": controller_started_ns,
            "controller_finished_ns": controller_finished_ns,
            "controller_elapsed_ns": controller_elapsed_ns,
            "environment_started_ns": environment_started_ns,
            "environment_finished_ns": environment_finished_ns,
            "environment_elapsed_ns": environment_elapsed_ns,
            "post_step_finished_ns": time.time_ns(),
            "termination": termination,
        }

    def sample_virtual_pilot_rollout(
        self,
        *,
        fps: float,
        progress: Callable[[int, int], None] | None = None,
        cancelled: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Run one deterministic episode as fast as CPU allows.

        Every 500 Hz control step is evaluated locally.  Only video-rate
        snapshots are retained, keeping the result compact enough to transfer
        once after the rollout instead of using the interactive telemetry
        stream as part of the closed loop.
        """

        import torch
        from flight_train.commands import VirtualPilotCommandSource
        from flight_train.config import _virtual_pilot

        if not math.isfinite(fps) or fps < 1 or fps > 60:
            raise RuntimeConfigurationError(
                "rollout fps must be between 1 and 60"
            )
        command_config = _virtual_pilot(
            _mapping(
                self.raw_test.get("command_source", {}),
                "command_source",
            )
        )
        pilot = VirtualPilotCommandSource(
            command_config,
            1,
            self.device,
            self.dtype,
            self.control_hz,
        )
        pilot.reset(
            torch.ones(1, device=self.device, dtype=torch.bool)
        )
        with self._lock:
            if self._state not in {"ready", "paused"}:
                raise RuntimeError(
                    "offline rollout requires a ready or paused session"
                )
            self._state = "sampling"
        total_steps = self.max_episode_steps
        requested_duration_s = self.options.episode_duration_s
        frame_targets = {
            min(total_steps, round(index * self.control_hz / fps))
            for index in range(
                math.ceil(requested_duration_s * fps) + 1
            )
        }
        frame_targets.add(0)
        frame_targets.add(total_steps)
        frames: list[dict[str, Any]] = []
        termination: dict[str, Any] | None = None

        try:
            with torch.no_grad():
                frames.append(
                    self._rollout_snapshot(0.0, pilot, None)
                )
                if progress is not None:
                    progress(0, total_steps)
                for step_index in range(1, total_steps + 1):
                    if cancelled is not None and cancelled.is_set():
                        raise RolloutCancelled("rollout cancelled")
                    position = self.environment.observe(
                        "truth", ("position_n",)
                    ).values["position_n"]
                    pilot.step(-position[:, 2:3])
                    pilot_snapshot = pilot.snapshot()
                    channels_tensor = torch.zeros(
                        (1, 4),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    if command_config.max_roll_rad > 0:
                        channels_tensor[:, 0] = (
                            pilot_snapshot.stick_target[:, 0]
                            / command_config.max_roll_rad
                        )
                    if command_config.max_pitch_rad > 0:
                        channels_tensor[:, 1] = (
                            pilot_snapshot.stick_target[:, 1]
                            / command_config.max_pitch_rad
                        )
                    channels_tensor[:, 2] = (
                        pilot_snapshot.stick_target[:, 2]
                    )
                    throttle_span = (
                        command_config.throttle_maximum
                        - command_config.throttle_minimum
                    )
                    channels_tensor[:, 3] = (
                        2.0
                        * (
                            pilot_snapshot.throttle_target[:, 0]
                            - command_config.throttle_minimum
                        )
                        / throttle_span
                        - 1.0
                    )
                    channels_tensor.clamp_(-1.0, 1.0)
                    self.input_tensor.copy_(channels_tensor)
                    step_result = self._step_cpu(
                        inspect_safety=True,
                        reset_on_termination=False,
                    )
                    self._control_steps += 1
                    termination = step_result["termination"]
                    if (
                        step_index in frame_targets
                        or termination is not None
                    ):
                        frames.append(
                            self._rollout_snapshot(
                                step_index / self.control_hz,
                                pilot,
                                channels_tensor,
                            )
                        )
                    if progress is not None and (
                        step_index == total_steps
                        or step_index % max(1, self.control_hz // 5)
                        == 0
                    ):
                        progress(step_index, total_steps)
                    if (
                        termination is not None
                        and termination["reason"] != "episode_timeout"
                    ):
                        break
        finally:
            with self._lock:
                if self._state == "sampling":
                    self._state = "paused"

        actual_duration_s = (
            float(frames[-1]["time_s"]) if frames else 0.0
        )
        return {
            "schema_version": 1,
            "rollout_id": self.id,
            "created_at_epoch_ms": time.time() * 1000.0,
            "fps": fps,
            "control_hz": self.control_hz,
            "requested_duration_s": requested_duration_s,
            "duration_s": actual_duration_s,
            "termination": termination,
            "controller": self.controller.describe(),
            "configuration": copy.deepcopy(
                self.effective_configuration
            ),
            "frames": frames,
        }

    def _rollout_snapshot(
        self,
        time_s: float,
        pilot: Any,
        channels: Any | None,
    ) -> dict[str, Any]:
        from flight_controller import tensor_diagnostics_to_python

        fields = (
            "position_n",
            "velocity_n",
            "attitude_q_wb",
            "angular_velocity_b",
            "motor_speed",
            "servo_angle",
            "grid_moment_b",
            "moment_b",
            "grid_force_b",
            "force_b",
        )
        truth = self.environment.observe("truth", fields).values
        pilot_snapshot = pilot.snapshot()
        if channels is None:
            channel_values = [0.0, 0.0, 0.0, -1.0]
        else:
            channel_values = channels[0].detach().cpu().tolist()
        frame: dict[str, Any] = {
            "time_s": time_s,
            "truth": {
                name: value.detach().cpu().tolist()
                for name, value in truth.items()
            },
            "pilot": {
                "channels": {
                    name: float(value)
                    for name, value in zip(
                        ("roll", "pitch", "yaw", "throttle"),
                        channel_values,
                    )
                },
                "stick_target": pilot_snapshot.stick_target[
                    0
                ].detach().cpu().tolist(),
                "stick_filtered": pilot_snapshot.filtered_stick[
                    0
                ].detach().cpu().tolist(),
                "target_attitude_q_wb": pilot_snapshot.target_attitude_q_wb[
                    0
                ].detach().cpu().tolist(),
                "upper_throttle": float(
                    pilot_snapshot.upper_throttle[0, 0]
                    .detach()
                    .cpu()
                    .item()
                ),
                "throttle_target": float(
                    pilot_snapshot.throttle_target[0, 0]
                    .detach()
                    .cpu()
                    .item()
                ),
                "height_target_m": float(
                    pilot_snapshot.height_target[0, 0]
                    .detach()
                    .cpu()
                    .item()
                ),
                "height_error_m": float(
                    pilot_snapshot.height_error[0, 0]
                    .detach()
                    .cpu()
                    .item()
                ),
            },
        }
        if self.last_controller_output is not None:
            frame["controller"] = {
                "type": self.controller.controller_type,
                "command": self.last_controller_output.command.detach()
                .cpu()
                .tolist(),
                "diagnostics": tensor_diagnostics_to_python(
                    self.last_controller_output.diagnostics
                ),
            }
        if self.last_reference is not None:
            frame["reference"] = {
                "target_position_n": self.last_reference.target_position_n.detach()
                .cpu()
                .tolist(),
                "target_velocity_n": self.last_reference.target_velocity_n.detach()
                .cpu()
                .tolist(),
                "target_attitude_q_wb": self.last_reference.target_attitude_q_wb.detach()
                .cpu()
                .tolist(),
                "target_angular_velocity_b": self.last_reference.target_angular_velocity_b.detach()
                .cpu()
                .tolist(),
                "collective_command": self.last_reference.collective_command.detach()
                .cpu()
                .tolist(),
            }
        return frame

    def _reset_cpu(self) -> None:
        torch = self._torch
        self._record_reset(
            "manual",
            {
                "episode_id": self.episode_id,
                "episode_step": self.episode_step,
            },
        )
        with torch.no_grad():
            mask = torch.ones(1, device=self.device, dtype=torch.bool)
            self.environment.reset(mask, self.simenv_path)
            self.controller.reset(mask)
            self.filtered_stick.zero_()
            self.target_yaw.zero_()
            self.upper_throttle.fill_(self.options.throttle_minimum)
            self.target_position_n.copy_(
                self.environment.observe(
                    "truth", ("position_n",)
                ).values["position_n"]
            )
            self.target_velocity_n.zero_()
            self.last_controller_output = None
            self.last_reference = None
        self.episode_step = 0
        self.episode_id += 1

    def _record_reset(
        self,
        reason: str,
        details: Mapping[str, Any],
    ) -> None:
        record = {
            "reason": reason,
            **dict(details),
            "control_steps": self._control_steps,
            "server_time_ns": time.time_ns(),
        }
        with self._lock:
            self._last_reset = record
            self._reset_counts[reason] = (
                self._reset_counts.get(reason, 0) + 1
            )

    def _publish_telemetry(
        self,
        latency_trace: Mapping[str, Any] | None = None,
    ) -> None:
        from flight_controller import tensor_diagnostics_to_python

        telemetry_pack_started_ns = time.time_ns()
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
            "latency_trace": dict(latency_trace or {}),
        }
        if self.last_controller_output is not None:
            packet["controller"] = {
                "type": self.controller.controller_type,
                "command": self.last_controller_output.command.detach().cpu().tolist(),
                "diagnostics": tensor_diagnostics_to_python(
                    self.last_controller_output.diagnostics
                ),
            }
        if self.last_reference is not None:
            packet["reference"] = {
                "target_position_n": self.last_reference.target_position_n.detach().cpu().tolist(),
                "target_velocity_n": self.last_reference.target_velocity_n.detach().cpu().tolist(),
                "target_attitude_q_wb": self.last_reference.target_attitude_q_wb.detach().cpu().tolist(),
                "target_angular_velocity_b": self.last_reference.target_angular_velocity_b.detach().cpu().tolist(),
                "collective_command": self.last_reference.collective_command.detach().cpu().tolist(),
            }
        packet["latency_trace"].update({
            "telemetry_pack_started_ns": telemetry_pack_started_ns,
            "telemetry_pack_finished_ns": time.time_ns(),
        })
        packet["latency_trace"]["telemetry_published_ns"] = time.time_ns()
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

    def create(
        self,
        simenv_yaml: str,
        test_yaml: str,
        checkpoint_path: str | None,
        *,
        replace_existing: bool = False,
    ) -> CpuRuntimeSession:
        with self._lock:
            if self._sessions:
                if not replace_existing:
                    raise RuntimeError(
                        "only one interactive simulation session may exist at a time"
                    )
                # The WebUI intentionally exposes a single interactive slot.
                # A browser refresh can lose its unload DELETE, so a new WebUI
                # owner must be able to reclaim that slot atomically instead
                # of leaving an orphaned session that requires a server restart.
                stale_sessions = list(self._sessions.values())
                self._sessions.clear()
                for stale_session in stale_sessions:
                    stale_session.close()
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


class OfflineRolloutJob:
    """Background owner for one complete virtual-pilot rollout."""

    def __init__(
        self,
        simenv_yaml: str,
        test_yaml: str,
        checkpoint_path: str | Path | None,
        log_root: str | Path,
        *,
        fps: float,
    ) -> None:
        self.id = str(uuid.uuid4())
        self._simenv_yaml = simenv_yaml
        self._test_yaml = test_yaml
        self._checkpoint_path = checkpoint_path
        self._log_root = log_root
        self._fps = fps
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._state = "queued"
        self._error: str | None = None
        self._completed_steps = 0
        self._total_steps = 0
        self._result: dict[str, Any] | None = None
        self._created_at = time.time()
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"offline-rollout-{self.id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _set_progress(self, completed: int, total: int) -> None:
        with self._lock:
            self._completed_steps = completed
            self._total_steps = total

    def _run(self) -> None:
        session: CpuRuntimeSession | None = None
        with self._lock:
            self._state = "sampling"
            self._started_at = time.time()
        try:
            session = CpuRuntimeSession(
                self._simenv_yaml,
                self._test_yaml,
                self._checkpoint_path,
                self._log_root,
            )
            result = session.sample_virtual_pilot_rollout(
                fps=self._fps,
                progress=self._set_progress,
                cancelled=self._cancelled,
            )
            with self._lock:
                self._result = result
                self._state = "completed"
        except RolloutCancelled:
            with self._lock:
                self._state = "cancelled"
        except Exception as error:
            with self._lock:
                self._error = f"{type(error).__name__}: {error}"
                self._state = "failed"
        finally:
            if session is not None:
                session.close()
            with self._lock:
                self._finished_at = time.time()

    def cancel(self) -> None:
        self._cancelled.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            total = self._total_steps
            completed = self._completed_steps
            return {
                "job_id": self.id,
                "state": self._state,
                "error": self._error,
                "completed_steps": completed,
                "total_steps": total,
                "progress": completed / total if total else 0.0,
                "fps": self._fps,
                "created_at_epoch_ms": self._created_at * 1000.0,
                "started_at_epoch_ms": (
                    None
                    if self._started_at is None
                    else self._started_at * 1000.0
                ),
                "finished_at_epoch_ms": (
                    None
                    if self._finished_at is None
                    else self._finished_at * 1000.0
                ),
                "result_ready": self._result is not None,
            }

    def result(self) -> dict[str, Any]:
        with self._lock:
            if self._state != "completed" or self._result is None:
                raise RuntimeError("rollout result is not ready")
            return copy.deepcopy(self._result)


class OfflineRolloutRegistry:
    """Keeps one active rollout and a small set of completed results."""

    def __init__(self, log_root: str | Path, retain: int = 3) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, OfflineRolloutJob] = {}
        self._log_root = Path(log_root).expanduser().resolve()
        self._retain = max(1, retain)

    def create(
        self,
        simenv_yaml: str,
        test_yaml: str,
        checkpoint_path: str | Path | None,
        *,
        fps: float,
    ) -> OfflineRolloutJob:
        with self._lock:
            active = [
                job
                for job in self._jobs.values()
                if job.status()["state"] in {"queued", "sampling"}
            ]
            if active:
                raise RuntimeError(
                    "only one offline rollout can run at a time"
                )
            finished = sorted(
                (
                    job
                    for job in self._jobs.values()
                    if job.status()["state"]
                    in {"completed", "failed", "cancelled"}
                ),
                key=lambda job: job.status()["created_at_epoch_ms"],
            )
            while len(finished) >= self._retain:
                expired = finished.pop(0)
                self._jobs.pop(expired.id, None)
            job = OfflineRolloutJob(
                simenv_yaml,
                test_yaml,
                checkpoint_path,
                self._log_root,
                fps=fps,
            )
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> OfflineRolloutJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as error:
                raise KeyError(f"unknown rollout job {job_id}") from error

    def cancel(self, job_id: str) -> OfflineRolloutJob:
        job = self.get(job_id)
        job.cancel()
        return job

    def close_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.cancel()
