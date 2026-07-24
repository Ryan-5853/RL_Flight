from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .errors import ConfigurationError
from .randomization import ParameterRandomizer


@dataclass(frozen=True)
class TimingConfig:
    physics_hz: int
    control_hz: int
    physics_dt: float
    substeps: int


@dataclass(frozen=True)
class LoggingConfig:
    directory: Path
    chunk_steps: int
    queue_chunks: int
    overflow: str
    mode: str
    physics_step_stride: int
    fields: tuple[str, ...] | None


@dataclass(frozen=True)
class MaterializedConfig:
    raw: Mapping[str, Any]
    source_path: Path
    seed: int
    timing: TimingConfig
    logging: LoggingConfig
    parameters: Mapping[str, torch.Tensor]
    initial_state: Mapping[str, torch.Tensor]
    sensor_state: Mapping[str, torch.Tensor]
    sensor_interpolation: Mapping[str, str | None]


def load_and_materialize(
    config_path: str | Path,
    parallel_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> MaterializedConfig:
    path = Path(config_path).expanduser().resolve()
    raw = _load_mapping(path)
    _require(raw.get("schema_version") == 1, "schema_version must equal 1")
    seed = _require_int(raw.get("seed"), "seed")
    _require(0 <= seed < (1 << 63), "seed must be within [0, 2^63)")
    parallel = _mapping(raw, "parallel")
    _require(parallel.get("independent_rng") is True, "parallel.independent_rng must be true")

    timing_node = _mapping(raw, "timing")
    physics_hz = _structural_int(timing_node.get("physics_hz"), "timing.physics_hz")
    control_hz = _structural_int(timing_node.get("control_hz"), "timing.control_hz")
    _require(physics_hz > 0 and control_hz > 0, "timing frequencies must be positive")
    _require(
        physics_hz % control_hz == 0,
        "timing.physics_hz / timing.control_hz must be a positive integer",
    )
    timing = TimingConfig(physics_hz, control_hz, 1.0 / physics_hz, physics_hz // control_hz)
    logging = _logging_config(raw.get("logging", {}), path.parent)
    randomizer = ParameterRandomizer(seed, parallel_count, device, dtype)

    initial = _initial_state(_mapping(raw, "initial_state"), randomizer)
    parameters: dict[str, torch.Tensor] = {}
    _body(_mapping(raw, "body"), randomizer, parameters)
    _motors(_sequence(raw, "motors"), randomizer, parameters)
    _servos(_sequence(raw, "servos"), randomizer, parameters)
    _aerodynamics(_mapping(raw, "aerodynamics"), randomizer, parameters)
    sensor_state, sensor_interpolation = _sensors(
        _mapping(raw, "sensors"), randomizer, parameters, physics_hz
    )
    _validate_materialized(parameters, initial, physics_hz)

    return MaterializedConfig(
        raw=raw,
        source_path=path,
        seed=seed,
        timing=timing,
        logging=logging,
        parameters=parameters,
        initial_state=initial,
        sensor_state=sensor_state,
        sensor_interpolation=sensor_interpolation,
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigurationError(
                "non-JSON YAML configuration requires PyYAML; install the project dependencies"
            ) from exc
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"invalid YAML configuration: {path}") from exc
    if not isinstance(loaded, Mapping):
        raise ConfigurationError("configuration root must be a mapping")
    return loaded


def _initial_state(node: Mapping[str, Any], r: ParameterRandomizer) -> dict[str, torch.Tensor]:
    state = {
        "position_n": r.sample(node.get("position_n"), "initial_state.position_n"),
        "velocity_n": r.sample(node.get("velocity_n"), "initial_state.velocity_n"),
        "attitude_q_wb": r.sample(node.get("attitude_q_wb"), "initial_state.attitude_q_wb"),
        "angular_velocity_b": r.sample(
            node.get("angular_velocity_b"), "initial_state.angular_velocity_b"
        ),
    }
    _shape(state["position_n"], (3,), "initial_state.position_n")
    _shape(state["velocity_n"], (3,), "initial_state.velocity_n")
    _shape(state["attitude_q_wb"], (4,), "initial_state.attitude_q_wb")
    _shape(state["angular_velocity_b"], (3,), "initial_state.angular_velocity_b")
    return state


def _body(node: Mapping[str, Any], r: ParameterRandomizer, out: dict[str, torch.Tensor]) -> None:
    out["body.mass"] = r.sample(node.get("mass"), "body.mass")
    out["body.center_of_mass_b"] = r.sample(
        node.get("center_of_mass_b"), "body.center_of_mass_b"
    )
    out["body.inertia_diagonal_b"] = r.sample(
        node.get("inertia_diagonal_b"), "body.inertia_diagonal_b"
    )
    _shape(out["body.mass"], (), "body.mass")
    _shape(out["body.center_of_mass_b"], (3,), "body.center_of_mass_b")
    _shape(out["body.inertia_diagonal_b"], (3,), "body.inertia_diagonal_b")


def _motors(nodes: Sequence[Any], r: ParameterRandomizer, out: dict[str, torch.Tensor]) -> None:
    _require(len(nodes) == 2, "motors must contain exactly upper and lower")
    _require([_mapping_value(n, "name", f"motors[{i}]") for i, n in enumerate(nodes)] == ["upper", "lower"],
             "motor order must be [upper, lower]")
    fields = ("pwm_deadzone", "pwm_to_rpm_table", "time_constant", "torque_coefficient")
    for field in fields:
        tensors = [r.sample(_mapping_node(n, f"motors[{i}]").get(field), f"motors[{i}].{field}") for i, n in enumerate(nodes)]
        _same_tail_shape(tensors, f"motors.{field}")
        out[f"motors.{field}"] = torch.stack(tensors, dim=1)
    _table(out["motors.pwm_to_rpm_table"], "motors.pwm_to_rpm_table")
    noise_stddev = []
    for i, item in enumerate(nodes):
        noise = _mapping_node(
            _mapping_node(item, f"motors[{i}]").get("noise"), f"motors[{i}].noise"
        )
        _require(noise.get("distribution") == "normal", f"motors[{i}].noise.distribution must be normal")
        noise_stddev.append(r.sample(noise.get("stddev"), f"motors[{i}].noise.stddev"))
    out["motors.noise.stddev"] = torch.stack(noise_stddev, dim=1)


def _servos(nodes: Sequence[Any], r: ParameterRandomizer, out: dict[str, torch.Tensor]) -> None:
    _require(len(nodes) == 3, "servos must contain exactly servo_1, servo_2, servo_3")
    expected = ["servo_1", "servo_2", "servo_3"]
    _require([_mapping_value(n, "name", f"servos[{i}]") for i, n in enumerate(nodes)] == expected,
             "servo order must be [servo_1, servo_2, servo_3]")
    for field in ("pwm_angle_table", "tau", "max_speed", "backlash", "deadzone"):
        tensors = [r.sample(_mapping_node(n, f"servos[{i}]").get(field), f"servos[{i}].{field}") for i, n in enumerate(nodes)]
        _same_tail_shape(tensors, f"servos.{field}")
        out[f"servos.{field}"] = torch.stack(tensors, dim=1)
    _table(out["servos.pwm_angle_table"], "servos.pwm_angle_table")


def _aerodynamics(node: Mapping[str, Any], r: ParameterRandomizer, out: dict[str, torch.Tensor]) -> None:
    out["aerodynamics.thrust_coefficients"] = r.sample(
        node.get("thrust_coefficients"), "aerodynamics.thrust_coefficients"
    )
    out["aerodynamics.neutral_thrust_direction_b"] = r.sample(
        node.get("neutral_thrust_direction_b"), "aerodynamics.neutral_thrust_direction_b"
    )
    out["aerodynamics.direct_thrust_center_b"] = r.sample(
        node.get("direct_thrust_center_b"), "aerodynamics.direct_thrust_center_b"
    )
    partition = _mapping(node, "thrust_partition")
    out["aerodynamics.thrust_partition"] = torch.stack(
        [r.sample(partition.get(name), f"aerodynamics.thrust_partition.{name}") for name in ("direct", "grid_1", "grid_2", "grid_3")],
        dim=1,
    )
    grids = _sequence(node, "grids")
    _require(len(grids) == 3, "aerodynamics.grids must contain exactly three grids")
    expected = ["grid_1", "grid_2", "grid_3"]
    _require([_mapping_value(n, "name", f"aerodynamics.grids[{i}]") for i, n in enumerate(grids)] == expected,
             "grid order must be [grid_1, grid_2, grid_3]")
    for field in ("aerodynamic_center_b", "deflection_axis_b", "self_attenuation_curve"):
        tensors = [r.sample(_mapping_node(n, f"aerodynamics.grids[{i}]").get(field), f"aerodynamics.grids[{i}].{field}") for i, n in enumerate(grids)]
        _same_tail_shape(tensors, f"aerodynamics.grids.{field}")
        out[f"aerodynamics.grids.{field}"] = torch.stack(tensors, dim=1)
    _table(out["aerodynamics.grids.self_attenuation_curve"], "aerodynamics.grids.self_attenuation_curve")
    for field in ("gain", "offset"):
        tensors = []
        for i, grid in enumerate(grids):
            path = f"aerodynamics.grids[{i}].vector_deflection"
            vector_deflection = _mapping_node(
                _mapping_node(grid, f"aerodynamics.grids[{i}]").get("vector_deflection"), path
            )
            tensors.append(r.sample(vector_deflection.get(field), f"{path}.{field}"))
        out[f"aerodynamics.grids.vector_deflection.{field}"] = torch.stack(tensors, dim=1)
    out["aerodynamics.coupling_attenuation"] = r.sample(
        node.get("coupling_attenuation"), "aerodynamics.coupling_attenuation"
    )


def _sensors(
    node: Mapping[str, Any],
    r: ParameterRandomizer,
    out: dict[str, torch.Tensor],
    physics_hz: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str | None]]:
    sensor_state: dict[str, torch.Tensor] = {}
    interpolation_modes: dict[str, str | None] = {}
    supported_shapes = {"gyro": (3,), "accelerometer": (3,), "motor_speed": (2,)}
    for name, raw_sensor in node.items():
        _require(
            name in supported_shapes,
            f"unsupported sensor {name!r}; supported sensors are {tuple(supported_shapes)}",
        )
        sensor = _mapping_node(raw_sensor, f"sensors.{name}")
        sample_node = sensor.get("sample_hz", {"value": physics_hz})
        out[f"sensors.{name}.sample_hz"] = r.sample(sample_node, f"sensors.{name}.sample_hz")
        noise = _mapping(sensor, "noise")
        _require(noise.get("distribution") == "normal", f"sensors.{name}.noise.distribution must be normal")
        out[f"sensors.{name}.noise.stddev"] = r.sample(
            noise.get("stddev"), f"sensors.{name}.noise.stddev"
        )
        out[f"sensors.{name}.bias"] = r.sample(sensor.get("bias"), f"sensors.{name}.bias")
        out[f"sensors.{name}.delay"] = r.sample(sensor.get("delay"), f"sensors.{name}.delay")
        _shape(out[f"sensors.{name}.sample_hz"], (), f"sensors.{name}.sample_hz")
        _shape(out[f"sensors.{name}.delay"], (), f"sensors.{name}.delay")
        _shape(out[f"sensors.{name}.noise.stddev"], supported_shapes[name], f"sensors.{name}.noise.stddev")
        _shape(out[f"sensors.{name}.bias"], supported_shapes[name], f"sensors.{name}.bias")
        _same_tail_shape(
            [out[f"sensors.{name}.noise.stddev"], out[f"sensors.{name}.bias"]],
            f"sensors.{name} noise/bias",
        )
        interpolation = sensor.get("interpolation")
        _require(
            interpolation in {None, "linear"},
            f"sensors.{name}.interpolation must be linear when specified",
        )
        delay_steps = out[f"sensors.{name}.delay"] * physics_hz
        fractional = ~torch.isclose(
            delay_steps, delay_steps.round(), atol=1e-5, rtol=0
        )
        _require(
            not bool(torch.any(fractional).item()) or interpolation == "linear",
            f"sensors.{name}.interpolation must be linear for non-integer physics-step delay",
        )
        interpolation_modes[name] = interpolation
        sensor_state[name] = out[f"sensors.{name}.bias"].clone()
    _require(bool(sensor_state), "sensors must contain at least one sensor")
    return sensor_state, interpolation_modes


def _validate_materialized(
    p: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor], physics_hz: int
) -> None:
    _shape(p["motors.pwm_deadzone"], (2,), "motors.pwm_deadzone")
    _shape(p["motors.time_constant"], (2,), "motors.time_constant")
    _shape(p["motors.torque_coefficient"], (2,), "motors.torque_coefficient")
    _shape(p["motors.noise.stddev"], (2,), "motors.noise.stddev")
    _shape(p["servos.tau"], (3,), "servos.tau")
    _shape(p["servos.max_speed"], (3,), "servos.max_speed")
    _shape(p["servos.backlash"], (3,), "servos.backlash")
    _shape(p["servos.deadzone"], (3,), "servos.deadzone")
    _shape(p["aerodynamics.neutral_thrust_direction_b"], (3,), "aerodynamics.neutral_thrust_direction_b")
    _shape(p["aerodynamics.thrust_coefficients"], (3,), "aerodynamics.thrust_coefficients")
    _shape(p["aerodynamics.direct_thrust_center_b"], (3,), "aerodynamics.direct_thrust_center_b")
    _shape(p["aerodynamics.thrust_partition"], (4,), "aerodynamics.thrust_partition")
    _shape(p["aerodynamics.grids.aerodynamic_center_b"], (3, 3), "aerodynamics.grids.aerodynamic_center_b")
    _shape(p["aerodynamics.grids.deflection_axis_b"], (3, 3), "aerodynamics.grids.deflection_axis_b")
    _shape(p["aerodynamics.grids.vector_deflection.gain"], (3,), "aerodynamics.grids.vector_deflection.gain")
    _shape(p["aerodynamics.grids.vector_deflection.offset"], (3,), "aerodynamics.grids.vector_deflection.offset")
    _all(p["body.mass"] > 0, "body.mass must be positive after randomization")
    _all(p["body.inertia_diagonal_b"] > 0, "body inertia must be positive after randomization")
    q_norm = torch.linalg.vector_norm(state["attitude_q_wb"], dim=-1)
    _all(torch.isclose(q_norm, torch.ones_like(q_norm), atol=1e-5, rtol=1e-5), "initial quaternion must be normalized")
    _all(
        (p["motors.pwm_deadzone"] >= 0) & (p["motors.pwm_deadzone"] <= 1),
        "motor pwm_deadzone must be within [0,1]",
    )
    _all(p["motors.time_constant"] > 0, "motor time_constant must be positive")
    _all(p["motors.torque_coefficient"] >= 0, "motor torque_coefficient must be non-negative")
    _all(p["motors.noise.stddev"] >= 0, "motor noise stddev must be non-negative")
    _all(
        p["motors.pwm_to_rpm_table"][..., 1] >= 0,
        "motor target speed table must be non-negative",
    )
    _all(
        (p["motors.pwm_to_rpm_table"][..., 0] >= 0)
        & (p["motors.pwm_to_rpm_table"][..., 0] <= 1),
        "motor PWM table axis must be within [0,1]",
    )
    thrust_coefficients = p["aerodynamics.thrust_coefficients"]
    k1, k2, k3 = thrust_coefficients.unbind(dim=1)
    _all((k1 >= 0) & (k2 >= 0), "thrust coefficients k1 and k2 must be non-negative")
    _all(
        k3 + 2.0 * torch.sqrt(k1 * k2) >= 0,
        "thrust coefficients must produce non-negative thrust for non-negative rotor speeds",
    )
    _all(p["servos.tau"] > 0, "servo tau must be positive")
    _all(p["servos.max_speed"] > 0, "servo max_speed must be positive")
    _all(p["servos.backlash"] >= 0, "servo backlash must be non-negative")
    _all(p["servos.deadzone"] >= 0, "servo deadzone must be non-negative")
    _all(p["servos.deadzone"] <= 2, "servo deadzone must not exceed the PWM span")
    _all(
        (p["servos.pwm_angle_table"][..., 0] >= -1)
        & (p["servos.pwm_angle_table"][..., 0] <= 1),
        "servo PWM table axis must be within [-1,1]",
    )
    partition = p["aerodynamics.thrust_partition"]
    _all(partition >= 0, "thrust partition must be non-negative")
    _all(torch.isclose(partition.sum(dim=1), torch.ones_like(partition[:, 0]), atol=1e-5), "thrust partition must sum to 1")
    direction = p["aerodynamics.neutral_thrust_direction_b"]
    axes = p["aerodynamics.grids.deflection_axis_b"]
    _unit_vectors(direction, "neutral thrust direction")
    _unit_vectors(axes, "grid deflection axes")
    pairwise_axis_dot = torch.stack(
        (
            (axes[:, 0] * axes[:, 1]).sum(dim=1),
            (axes[:, 1] * axes[:, 2]).sum(dim=1),
            (axes[:, 2] * axes[:, 0]).sum(dim=1),
        ),
        dim=1,
    )
    _all(
        torch.isclose(
            pairwise_axis_dot,
            torch.full_like(pairwise_axis_dot, -0.5),
            atol=1e-4,
            rtol=1e-4,
        ),
        "grid deflection axes must be pairwise 120 degrees apart",
    )
    _all(
        torch.abs((axes * direction[:, None, :]).sum(dim=2)) <= 1e-4,
        "grid deflection axes must be perpendicular to neutral thrust",
    )
    coupling = p["aerodynamics.coupling_attenuation"]
    _shape(coupling, (3, 3), "aerodynamics.coupling_attenuation")
    _all((coupling >= 0) & (coupling <= 1), "coupling attenuation must be within [0,1]")
    diagonal = torch.diagonal(coupling, dim1=-2, dim2=-1)
    _all(diagonal == 0, "coupling attenuation diagonal must be zero")
    attenuation = p["aerodynamics.grids.self_attenuation_curve"]
    _all(
        (attenuation[..., 1] >= 0) & (attenuation[..., 1] <= 1),
        "grid thrust-retention values must be within [0,1]",
    )
    for name, value in p.items():
        if name.endswith(".sample_hz"):
            _all(value > 0, f"{name} must be positive")
            ratio = torch.as_tensor(physics_hz, device=value.device, dtype=value.dtype) / value
            _all(torch.isclose(ratio, ratio.round(), atol=1e-5), f"{name} must divide physics_hz")
        if name.endswith(".delay"):
            _all(value >= 0, f"{name} must be non-negative")
        if name.endswith(".noise.stddev"):
            _all(value >= 0, f"{name} must be non-negative")


def _logging_config(node: Any, base_dir: Path) -> LoggingConfig:
    if not isinstance(node, Mapping):
        raise ConfigurationError("logging must be a mapping")
    allowed = {
        "directory",
        "chunk_steps",
        "flush_interval_steps",
        "queue_chunks",
        "overflow",
        "mode",
        "physics_step_stride",
        "fields",
    }
    unknown = sorted(set(node) - allowed)
    _require(not unknown, f"unknown logging fields: {unknown}")
    directory = Path(node.get("directory", "logs"))
    if not directory.is_absolute():
        directory = (base_dir / directory).resolve()
    chunk_steps = _require_int(node.get("chunk_steps", node.get("flush_interval_steps", 1024)), "logging.chunk_steps")
    queue_chunks = _require_int(node.get("queue_chunks", 2), "logging.queue_chunks")
    overflow = node.get("overflow", "block")
    mode = node.get("mode", "full")
    _require(mode in {"full", "compact"}, "logging.mode must be full or compact")
    stride_default = 1 if mode == "full" else 10
    physics_step_stride = _require_int(
        node.get("physics_step_stride", stride_default),
        "logging.physics_step_stride",
    )
    fields_node = node.get("fields")
    fields: tuple[str, ...] | None
    if fields_node is None:
        fields = None
    else:
        _require(
            isinstance(fields_node, list)
            and bool(fields_node)
            and all(isinstance(field, str) and field for field in fields_node),
            "logging.fields must be a non-empty list of field names",
        )
        fields = tuple(fields_node)
        _require(len(set(fields)) == len(fields), "logging.fields must not contain duplicates")
    _require(chunk_steps > 0 and queue_chunks > 0, "logging chunk sizes must be positive")
    _require(physics_step_stride > 0, "logging.physics_step_stride must be positive")
    _require(overflow in {"block", "error"}, "logging.overflow must be block or error")
    return LoggingConfig(
        directory,
        chunk_steps,
        queue_chunks,
        overflow,
        mode,
        physics_step_stride,
        fields,
    )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _mapping_node(parent.get(key), key)


def _mapping_node(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    return value


def _sequence(parent: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ConfigurationError(f"{key} must be a list")
    return value


def _mapping_value(node: Any, key: str, path: str) -> Any:
    return _mapping_node(node, path).get(key)


def _structural_int(node: Any, path: str) -> int:
    if not isinstance(node, Mapping) or set(node) != {"value"}:
        raise ConfigurationError(f"{path} must contain only a non-randomized value")
    return _require_int(node["value"], f"{path}.value")


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer")
    return value


def _shape(tensor: torch.Tensor, tail: tuple[int, ...], path: str) -> None:
    if tensor.shape[1:] != tail:
        raise ConfigurationError(f"{path} must have per-instance shape {tail}, got {tuple(tensor.shape[1:])}")


def _same_tail_shape(tensors: Sequence[torch.Tensor], path: str) -> None:
    tails = {tensor.shape[1:] for tensor in tensors}
    if len(tails) != 1:
        raise ConfigurationError(f"{path} entries must have identical shapes")


def _table(tensor: torch.Tensor, path: str) -> None:
    if tensor.ndim < 4 or tensor.shape[-1] != 2 or tensor.shape[-2] < 2:
        raise ConfigurationError(f"{path} must contain at least two [x,y] points")
    _all(torch.diff(tensor[..., 0], dim=-1) > 0, f"{path} x axis must be strictly increasing")


def _unit_vectors(tensor: torch.Tensor, path: str) -> None:
    norms = torch.linalg.vector_norm(tensor, dim=-1)
    _all(torch.isclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5), f"{path} must be unit vectors")


def _all(condition: torch.Tensor, message: str) -> None:
    if not bool(torch.all(condition).item()):
        raise ConfigurationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)
