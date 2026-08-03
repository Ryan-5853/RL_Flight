"""Single-environment CPU runtime linking controller input, inference, and SimEnv.

The HTTP server is deliberately kept outside this module.  This module owns one
long-lived worker thread per session; simulation, observation construction, model
inference, recurrent state, and telemetry all remain on CPU.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
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

from inference_package import (
    InferenceModelAdapter,
    InferencePackageLoader,
    RealtimeInferencePackage,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for source_root in (
    PROJECT_ROOT / "Controller" / "src",
    PROJECT_ROOT / "Deploy" / "src",
    PROJECT_ROOT / "SimEnv" / "src",
    PROJECT_ROOT / "Train" / "src",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


class RuntimeConfigurationError(ValueError):
    pass


class RolloutCancelled(RuntimeError):
    """Raised internally when a queued/offline rollout is cancelled."""


def _simulator_compatibility_fingerprint(config: Mapping[str, Any]) -> str:
    def canonical(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        raise RuntimeConfigurationError(
            f"unsupported simulator configuration value {type(value).__name__}"
        )

    relevant = {
        key: value
        for key, value in config.items()
        if key not in {"seed", "initial_state", "logging"}
    }
    encoded = json.dumps(
        canonical(relevant),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


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
    compile_kernels: bool
    warmup_steps: int
    spin_us: float
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
    compile_kernels = runtime.get("compile_kernels", True)
    if not isinstance(compile_kernels, bool):
        raise RuntimeConfigurationError("runtime.compile_kernels must be boolean")
    options = RuntimeOptions(
        device=str(run.get("device", "cpu")),
        dtype=str(run.get("dtype", "float32")),
        observation_source=str(environment.get("observation_source", "truth")),
        cpu_threads=int(runtime.get("cpu_threads", 1)),
        compile_kernels=compile_kernels,
        warmup_steps=int(runtime.get("warmup_steps", 3)),
        spin_us=_number(runtime, "spin_us", 200.0),
        execution_hz=_number(runtime, "execution_hz", 500.0),
        telemetry_hz=_number(runtime, "telemetry_hz", 60.0),
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
    if options.warmup_steps <= 0 or options.warmup_steps > 100:
        raise RuntimeConfigurationError(
            "runtime.warmup_steps must be between 1 and 100"
        )
    if options.spin_us < 0 or options.spin_us > 1000:
        raise RuntimeConfigurationError(
            "runtime.spin_us must be between 0 and 1000"
        )
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
        *,
        inference_package_loader: InferencePackageLoader | None = None,
    ) -> None:
        import torch
        from flight_controller import ControllerContext, create_controller
        from simenv import RealtimeSimulationEnvironment

        self.id = str(uuid.uuid4())
        self._torch = torch
        self.raw_simenv = yaml.safe_load(simenv_yaml)
        self.raw_test = yaml.safe_load(test_yaml)
        self.inference_package: RealtimeInferencePackage | None = None
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
        # relative paths. RealtimeSimulationEnvironment disables persistence,
        # but reset still reparses this normalized config; retain a safe
        # server-owned directory instead of a client-controlled write target.
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
        self.simulation = None
        self.environment = None
        try:
            dynamic_parameters, dynamic_seed = _dynamic_randomization(self.raw_test)
            self.simulation = RealtimeSimulationEnvironment.create(
                self.simenv_path,
                device=self.device,
                dtype=self.dtype,
                compile_kernels=self.options.compile_kernels,
                dynamic_randomization=dynamic_parameters,
                dynamic_seed=dynamic_seed,
            )
            self.environment = self.simulation.environment
            if not self.environment.dynamics_implemented or not self.environment.sensors_implemented:
                raise RuntimeConfigurationError("SimEnv dynamics and sensors must both be implemented")
            self.log_directory = None
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
                        "neural controller requires a deployment package path"
                    )
                if inference_package_loader is None:
                    raise RuntimeConfigurationError(
                        "neural controller requires a deployment inference "
                        "package loader; the realtime path does not construct "
                        "models through Train"
                    )
                self.inference_package = inference_package_loader(
                    Path(checkpoint_path).expanduser().resolve(),
                    self.device,
                    self.dtype,
                )
                self.inference_package.metadata.validate()
                required_control_hz = getattr(
                    self.inference_package, "required_control_hz", None
                )
                if (
                    required_control_hz is not None
                    and int(required_control_hz) != self.control_hz
                ):
                    raise RuntimeConfigurationError(
                        "deployment package requires control_hz="
                        f"{required_control_hz}, simulator provides {self.control_hz}"
                    )
                required_simulator_fingerprint = getattr(
                    self.inference_package,
                    "required_simulator_fingerprint",
                    None,
                )
                if (
                    required_simulator_fingerprint is not None
                    and _simulator_compatibility_fingerprint(self.raw_simenv)
                    != required_simulator_fingerprint
                ):
                    raise RuntimeConfigurationError(
                        "deployment package is incompatible with the selected "
                        "SimEnv dynamics; import the training environment config"
                    )
                neural_model = InferenceModelAdapter(self.inference_package)
                params = controller_config.setdefault("params", {})
                if not isinstance(params, dict):
                    raise RuntimeConfigurationError(
                        "controller.params must be a mapping"
                    )
                params.setdefault(
                    "maximum_angular_rate_rad_s",
                    self.options.max_angular_rate_rad_s,
                )
                configured_output = str(
                    params.get(
                        "output_mode",
                        self.inference_package.metadata.output_mode,
                    )
                )
                if configured_output != self.inference_package.metadata.output_mode:
                    raise RuntimeConfigurationError(
                        "controller output_mode does not match inference package"
                    )
                params["output_mode"] = configured_output
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
                "simulation_backend": "simenv-realtime-single-v1",
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
                    "deployment_package"
                    if controller_type == "neural"
                    else "environment"
                ),
            }
            self.warmup_seconds = self.simulation.warmup(
                steps=self.options.warmup_steps
            )
            if self.inference_package is not None:
                self.inference_package.warmup(
                    torch.zeros(
                        (1, self.inference_package.metadata.observation_dim),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
        except Exception:
            if self.inference_package is not None:
                self.inference_package.close()
            if self.simulation is not None:
                self.simulation.close()
            elif self.environment is not None:
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
            self.target_rate = torch.zeros(
                (1, 3), device=self.device, dtype=self.dtype
            )
            self.last_controller_output = None
            self.last_reference = None
            self.episode_step = 0
            self.episode_id = 0
            self.controller_compiled = False
            self._filter_alpha = 1.0 - torch.exp(-torch.tensor(
                self.control_period, device=self.device, dtype=self.dtype
            ) / torch.tensor(self.options.stick_time_constants, device=self.device, dtype=self.dtype))
            if self.options.compile_kernels:
                if self.controller.controller_type != "neural":
                    self.controller.step = torch.compile(
                        self.controller.step,
                        fullgraph=True,
                        mode="reduce-overhead",
                    )
                checkpoint = self.environment.state_dict()
                controller_warmup_started = time.perf_counter()
                try:
                    with torch.no_grad():
                        for _ in range(self.options.warmup_steps):
                            self._step_cpu(
                                inspect_safety=False,
                                capture_timing=False,
                            )
                finally:
                    self.environment.load_state_dict(checkpoint)
                    reset_mask = torch.ones(
                        1, device=self.device, dtype=torch.bool
                    )
                    self.controller.reset(reset_mask)
                    if self.inference_package is not None:
                        self.inference_package.reset()
                    self.filtered_stick.zero_()
                    self.target_yaw.zero_()
                    self.upper_throttle.fill_(
                        self.options.throttle_minimum
                    )
                    self.target_rate.zero_()
                    self.last_controller_output = None
                    self.last_reference = None
                    self.episode_step = 0
                    self.episode_id = 0
                self.warmup_seconds += (
                    time.perf_counter() - controller_warmup_started
                )
                self.controller_compiled = True
            self.controller_description = self.controller.describe()
            if self.inference_package is not None:
                self.controller_description["inference_package"] = dict(
                    self.inference_package.describe()
                )
        except Exception:
            if self.inference_package is not None:
                self.inference_package.close()
            self.simulation.close()
            self._tempdir.cleanup()
            raise

        self._lock = threading.RLock()
        self._telemetry_ready = threading.Condition(self._lock)
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
            self.simulation.close()
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
        with self._telemetry_ready:
            self._telemetry_ready.notify_all()
        self._thread.join(timeout=5)
        try:
            if self.inference_package is not None:
                self.inference_package.close()
        finally:
            self.simulation.close()
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
                "simulation_backend": "simenv-realtime-single-v1",
                "simulation_compiled": self.simulation.compiled,
                "controller_compiled": self.controller_compiled,
                "simulation_warmup_s": self.warmup_seconds,
                "persistent_logging": False,
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
                "controller": copy.deepcopy(
                    self.controller_description
                ),
            }

    def telemetry(self, after: int = -1) -> dict[str, Any] | None:
        with self._lock:
            if self._telemetry is None or self._telemetry_sequence <= after:
                return None
            return copy.deepcopy(self._telemetry)

    def wait_telemetry(
        self,
        after: int = -1,
        timeout: float = 1.0,
    ) -> dict[str, Any] | None:
        """Block without polling until a newer latest-value packet exists."""

        deadline = time.monotonic() + max(0.0, timeout)
        with self._telemetry_ready:
            while (
                self._telemetry is None
                or self._telemetry_sequence <= after
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._state in {"closed", "faulted"}:
                    return None
                self._telemetry_ready.wait(remaining)
            return copy.deepcopy(self._telemetry)

    def _run(self) -> None:
        torch = self._torch
        next_deadline_ns = time.perf_counter_ns()
        period_ns = max(1, round(self.execution_period * 1e9))
        spin_ns = round(self.options.spin_us * 1000)
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
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
                    next_deadline_ns = time.perf_counter_ns()
                    continue
                step_started_ns_perf = time.perf_counter_ns()
                step_started_ns = time.time_ns()
                for index, value in enumerate(command):
                    self.input_tensor[0, index] = value
                publish_due = single_step or time.monotonic() - self._last_telemetry_at >= self.telemetry_period
                with torch.no_grad():
                    step_timing = self._step_cpu(
                        inspect_safety=publish_due,
                        capture_timing=publish_due or command_trace is not None,
                    )
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
                compute_s = (
                    time.perf_counter_ns() - step_started_ns_perf
                ) / 1e9
                if self._step_compute_ema_s <= 0:
                    self._step_compute_ema_s = compute_s
                else:
                    self._step_compute_ema_s = (
                        .9 * self._step_compute_ema_s + .1 * compute_s
                    )
                self._effective_execution_hz = self.execution_hz
                next_deadline_ns += period_ns
                remaining_ns = next_deadline_ns - time.perf_counter_ns()
                if remaining_ns > 0:
                    # Sleep most of the slack, then spin briefly to avoid the
                    # millisecond-scale overshoot of a pure Event.wait.
                    if remaining_ns > spin_ns:
                        self._closed.wait(
                            (remaining_ns - spin_ns) / 1e9
                        )
                    while (
                        not self._closed.is_set()
                        and time.perf_counter_ns() < next_deadline_ns
                    ):
                        pass
                else:
                    self._overruns += 1
                    # Do not execute an unbounded catch-up burst after an OS
                    # scheduling stall; resume from the current wall clock.
                    if remaining_ns < -period_ns:
                        next_deadline_ns = time.perf_counter_ns()
        except Exception as error:
            with self._lock:
                self._fault = f"{type(error).__name__}: {error}"
                self._state = "faulted"
                self._telemetry_ready.notify_all()
        finally:
            if gc_was_enabled:
                gc.enable()

    def _step_cpu(
        self,
        *,
        inspect_safety: bool,
        reset_on_termination: bool = True,
        capture_timing: bool = True,
    ) -> dict[str, Any]:
        torch = self._torch
        from flight_controller import ControllerReference, ControllerState

        preparation_started_ns = time.time_ns() if capture_timing else 0
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

        truth_fields = (
            "position_n", "velocity_n", "attitude_q_wb", "angular_velocity_b",
            "linear_acceleration_n", "motor_speed", "servo_angle",
        )
        truth = dict(zip(
            truth_fields,
            self.simulation.state_views("truth", truth_fields),
        ))
        sensor_fields = self.simulation.observation_layout["sensor"]
        sensor = dict(zip(
            sensor_fields,
            self.simulation.state_views("sensor", sensor_fields),
        ))
        controller_values = dict(truth)
        cascade_truth_contract = (
            self.inference_package is not None
            and self.inference_package.metadata.output_mode
            == "coaxial_differential_cyclic_3"
        )
        if self.controller.controller_type == "neural":
            # Train always uses configured accelerometer and motor-speed
            # channels. The cascade profile alone keeps truth angular velocity
            # because both its rate feature and backward-difference alpha did so.
            if not cascade_truth_contract:
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
        self.target_rate.zero_()
        self.target_rate[:, 2] = (
            self.filtered_stick[:, 2] * self.options.max_yaw_rate_rad_s
        )
        reference = ControllerReference(
            target_position_n=self.target_position_n,
            target_velocity_n=self.target_velocity_n,
            target_attitude_q_wb=target_attitude,
            target_angular_velocity_b=self.target_rate,
            collective_command=self.upper_throttle,
        )
        controller_started_perf_ns = (
            time.perf_counter_ns() if capture_timing else 0
        )
        controller_started_ns = time.time_ns() if capture_timing else 0
        controller_output = self.controller.step(state, reference)
        controller_finished_ns = time.time_ns() if capture_timing else 0
        controller_elapsed_ns = (
            time.perf_counter_ns() - controller_started_perf_ns
            if capture_timing
            else 0
        )
        environment_started_perf_ns = (
            time.perf_counter_ns() if capture_timing else 0
        )
        environment_started_ns = time.time_ns() if capture_timing else 0
        result = self.simulation.advance(controller_output.command)
        environment_finished_ns = time.time_ns() if capture_timing else 0
        environment_elapsed_ns = (
            time.perf_counter_ns() - environment_started_perf_ns
            if capture_timing
            else 0
        )
        self.last_controller_output = controller_output
        self.last_reference = reference
        self.episode_step += 1

        inspect_safety = self.episode_step >= self.max_episode_steps or inspect_safety
        episode_timeout = self.episode_step >= self.max_episode_steps
        termination: dict[str, Any] | None = None
        if inspect_safety:
            attitude = state.attitude_q_wb
            angular_velocity = state.angular_velocity_b
            tilt = 2 * torch.acos(torch.sqrt((
                attitude[:, 0].square() + attitude[:, 3].square()
            ).clamp(0, 1)))
            angular_rate_norm = torch.linalg.vector_norm(
                angular_velocity, dim=1
            )
            invalid_mask = ~result.valid
            tilt_mask = tilt > self.options.max_tilt_rad
            angular_rate_mask = (
                angular_rate_norm > self.options.max_angular_rate_rad_s
            )
            reset_mask = invalid_mask | tilt_mask | angular_rate_mask
            should_reset = episode_timeout or bool(reset_mask.any().item())
        else:
            should_reset = False
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
                if self.inference_package is not None:
                    self.inference_package.reset()
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
            "post_step_finished_ns": (
                time.time_ns() if capture_timing else 0
            ),
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
            if self.inference_package is not None:
                self.inference_package.reset()
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
        truth = self.simulation.state_views("truth", fields)
        cpu = {
            name: value.detach().tolist()
            for name, value in zip(fields, truth)
        }
        next_telemetry_sequence = self._telemetry_sequence + 1
        packet = {
            "type": "telemetry",
            "session_id": self.id,
            "sequence": next_telemetry_sequence,
            "server_time_ns": time.time_ns(),
            "truth": cpu,
            "runtime": self.status(),
            "latency_trace": dict(latency_trace or {}),
        }
        packet["runtime"]["telemetry_sequence"] = next_telemetry_sequence
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
        with self._telemetry_ready:
            self._telemetry_sequence = next_telemetry_sequence
            self._telemetry = packet
            self._telemetry_ready.notify_all()

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
    def __init__(
        self,
        log_root: str | Path,
        *,
        inference_package_loader: InferencePackageLoader | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, CpuRuntimeSession] = {}
        self._log_root = Path(log_root).expanduser().resolve()
        self._inference_package_loader = inference_package_loader

    def capabilities(self) -> dict[str, Any]:
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
                "runtime": "cpu-realtime-single-environment-v2",
                "simulation_backend": "simenv-realtime-single-v1",
                "torch_compile_available": hasattr(torch, "compile"),
                "inference_package_loader": (
                    self._inference_package_loader is not None
                ),
            }
        except Exception as error:
            return {
                "cpu_available": False,
                "error": str(error),
                "devices": [],
                "runtime": "cpu-realtime-single-environment-v2",
                "simulation_backend": "simenv-realtime-single-v1",
                "inference_package_loader": (
                    self._inference_package_loader is not None
                ),
            }

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
            session = CpuRuntimeSession(
                simenv_yaml,
                test_yaml,
                checkpoint_path,
                self._log_root,
                inference_package_loader=self._inference_package_loader,
            )
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
        inference_package_loader: InferencePackageLoader | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self._simenv_yaml = simenv_yaml
        self._test_yaml = test_yaml
        self._checkpoint_path = checkpoint_path
        self._log_root = log_root
        self._fps = fps
        self._inference_package_loader = inference_package_loader
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
                inference_package_loader=self._inference_package_loader,
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

    def __init__(
        self,
        log_root: str | Path,
        retain: int = 3,
        *,
        inference_package_loader: InferencePackageLoader | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, OfflineRolloutJob] = {}
        self._log_root = Path(log_root).expanduser().resolve()
        self._retain = max(1, retain)
        self._inference_package_loader = inference_package_loader

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
                inference_package_loader=self._inference_package_loader,
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
