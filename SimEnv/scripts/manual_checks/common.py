"""Shared helpers for deterministic, inspectable SimEnv diagnostics."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def value(node: dict[str, Any], new_value: Any) -> None:
    """Replace a configuration value while retaining its schema wrapper."""

    node.clear()
    node["value"] = new_value


def _remove_randomization(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("randomization", None)
        for child in node.values():
            _remove_randomization(child)
    elif isinstance(node, list):
        for child in node:
            _remove_randomization(child)


def deterministic_config(
    log_directory: Path,
    *,
    physics_hz: int = 500,
    control_hz: int | None = None,
) -> dict[str, Any]:
    """Load the example topology and remove every source of randomness."""

    if physics_hz != 500 or (control_hz is not None and control_hz != 500):
        raise ValueError("SimEnv manual checks require the fixed 500 Hz timebase")

    with (REPO_ROOT / "configs" / "example.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    _remove_randomization(config)
    config["seed"] = 1
    value(config["timing"]["physics_hz"], physics_hz)
    value(config["timing"]["control_hz"], 500)
    config["logging"]["directory"] = str(log_directory)
    config["logging"]["mode"] = "compact"
    config["logging"]["physics_step_stride"] = 1000
    config["logging"]["chunk_steps"] = 64
    config["logging"]["queue_chunks"] = 1

    for motor in config["motors"]:
        value(motor["noise"]["stddev"], 0.0)
    for sensor in config["sensors"].values():
        value(sensor["sample_hz"], physics_hz)
        value(sensor["noise"]["stddev"], [0.0] * len(sensor["bias"]["value"]))
        value(sensor["bias"], [0.0] * len(sensor["bias"]["value"]))
        value(sensor["delay"], 0.0)
        sensor.pop("interpolation", None)
    return config


def configure_clean_actuators(config: dict[str, Any]) -> None:
    """Use simple deterministic actuator maps suitable for hand calculation."""

    for motor in config["motors"]:
        value(motor["pwm_deadzone"], 0.0)
        value(motor["pwm_to_rpm_table"], [[0.0, 0.0], [1.0, 100.0]])
        value(motor["time_constant"], 0.01)
        value(motor["torque_coefficient"], 0.0)
        value(motor["noise"]["stddev"], 0.0)
    value(config["aerodynamics"]["thrust_coefficients"], [1.0e-3, 1.0e-3, 0.0])

    for servo in config["servos"]:
        value(servo["pwm_angle_table"], [[-1.0, -0.2], [0.0, 0.0], [1.0, 0.2]])
        value(servo["tau"], 0.01)
        value(servo["max_speed"], 100.0)
        value(servo["backlash"], 0.0)
        value(servo["deadzone"], 0.0)


def configure_symmetric_grids(config: dict[str, Any]) -> None:
    """Install exact 120-degree radial grid geometry with no attenuation."""

    value(config["body"]["center_of_mass_b"], [0.0, 0.0, 0.0])
    value(config["body"]["inertia_diagonal_b"], [0.03, 0.03, 0.012])
    value(config["aerodynamics"]["neutral_thrust_direction_b"], [0.0, 0.0, -1.0])
    value(config["aerodynamics"]["direct_thrust_center_b"], [0.0, 0.0, 0.0])
    for name, fraction in (
        ("direct", 0.0),
        ("grid_1", 1.0 / 3.0),
        ("grid_2", 1.0 / 3.0),
        ("grid_3", 1.0 / 3.0),
    ):
        value(config["aerodynamics"]["thrust_partition"][name], fraction)

    for index, grid in enumerate(config["aerodynamics"]["grids"]):
        theta = index * 2.0 * math.pi / 3.0
        radial = [math.cos(theta), math.sin(theta), 0.0]
        center = [0.1 * radial[0], 0.1 * radial[1], 0.25]
        value(grid["aerodynamic_center_b"], center)
        value(grid["deflection_axis_b"], radial)
        value(grid["self_attenuation_curve"], [[0.0, 1.0], [0.2, 1.0]])
        value(grid["vector_deflection"]["gain"], 1.0)
        value(grid["vector_deflection"]["offset"], 0.0)
    value(
        config["aerodynamics"]["coupling_attenuation"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )


def write_config(config: dict[str, Any], path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def dot(a: list[float], b: list[float]) -> float:
    return sum(left * right for left, right in zip(a, b))


def rodrigues(
    vector: list[float], axis: list[float], angle: float
) -> list[float]:
    """Independent scalar implementation of right-hand axis rotation."""

    axis_cross_vector = cross(axis, vector)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    projection = dot(axis, vector)
    return [
        vector[i] * cosine
        + axis_cross_vector[i] * sine
        + axis[i] * projection * (1.0 - cosine)
        for i in range(3)
    ]


def rotate_body_to_world(q_wb: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
    q_vector = q_wb[1:]
    first_cross = torch.linalg.cross(q_vector, vector_b)
    return (
        vector_b
        + 2.0 * q_wb[0] * first_cross
        + 2.0 * torch.linalg.cross(q_vector, first_cross)
    )


def rotate_world_to_body(q_wb: torch.Tensor, vector_n: torch.Tensor) -> torch.Tensor:
    q_vector = q_wb[1:]
    first_cross = torch.linalg.cross(q_vector, vector_n)
    return (
        vector_n
        - 2.0 * q_wb[0] * first_cross
        + 2.0 * torch.linalg.cross(q_vector, first_cross)
    )


def integrate_quaternion(
    q_wb: torch.Tensor, angular_velocity_b: torch.Tensor, dt: float
) -> torch.Tensor:
    angular_speed = torch.linalg.vector_norm(angular_velocity_b)
    half_angle = 0.5 * dt * angular_speed
    if angular_speed.item() == 0.0:
        delta_vector = 0.5 * dt * angular_velocity_b
    else:
        delta_vector = (
            torch.sin(half_angle) / angular_speed * angular_velocity_b
        )
    delta_scalar = torch.cos(half_angle).reshape(1)
    scalar = q_wb[:1]
    vector = q_wb[1:]
    candidate = torch.cat(
        (
            scalar * delta_scalar - (vector * delta_vector).sum().reshape(1),
            scalar * delta_vector
            + delta_scalar * vector
            + torch.linalg.cross(vector, delta_vector),
        )
    )
    return candidate / torch.linalg.vector_norm(candidate)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2.0e-5,
    label: str,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=atol)
    except AssertionError as error:
        raise AssertionError(f"{label} failed:\n{error}") from error
