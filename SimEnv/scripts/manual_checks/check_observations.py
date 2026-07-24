#!/usr/bin/env python3
"""Check observation mapping, sampling/hold behavior, and sensor delay."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from common import (
    assert_close,
    configure_clean_actuators,
    deterministic_config,
    value,
    write_config,
)
from simenv import SimulationEnvironment


def gyro_sequence(
    root: Path,
    name: str,
    inputs: list[float],
    expected: list[float],
    *,
    sample_hz: int,
    delay: float,
    interpolation: str | None = None,
) -> None:
    file_stem = (
        name.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace(".", "-")
    )
    config = deterministic_config(root / f"{file_stem}-logs")
    configure_clean_actuators(config)
    value(config["aerodynamics"]["thrust_coefficients"], [0.0, 0.0, 0.0])
    gyro = config["sensors"]["gyro"]
    value(gyro["sample_hz"], sample_hz)
    value(gyro["delay"], delay)
    if interpolation is None:
        gyro.pop("interpolation", None)
    else:
        gyro["interpolation"] = interpolation
    path = write_config(config, root / f"{file_stem}.yaml")

    actual: list[float] = []
    with SimulationEnvironment.create(path, 1, "cpu") as env:
        control = torch.zeros((1, 5))
        for input_value in inputs:
            # Deliberate truth injection makes the sensor's source history inspectable.
            # Only p is nonzero, so torque-free rigid-body dynamics preserves it.
            env._truth["angular_velocity_b"][0] = torch.tensor(
                [input_value, 0.0, 0.0]
            )
            env.advance(control)
            actual.append(env.observe("sensor", ("gyro",)).values["gyro"][0, 0].item())
    assert_close(
        torch.tensor(actual),
        torch.tensor(expected),
        atol=1.0e-6,
        label=name,
    )
    print(
        f"{name:<18} input={inputs!s:<25} output="
        f"{[round(item, 4) for item in actual]}"
    )


def check_zero_delay_mapping(root: Path) -> None:
    config = deterministic_config(root / "mapping-logs")
    configure_clean_actuators(config)
    half_angle = math.pi / 8.0
    value(
        config["initial_state"]["attitude_q_wb"],
        [math.cos(half_angle), 0.0, math.sin(half_angle), 0.0],
    )
    gyro_bias = torch.tensor([0.1, -0.2, 0.3])
    accelerometer_bias = torch.tensor([0.4, -0.5, 0.6])
    motor_bias = torch.tensor([1.0, -2.0])
    value(config["sensors"]["gyro"]["bias"], gyro_bias.tolist())
    value(config["sensors"]["accelerometer"]["bias"], accelerometer_bias.tolist())
    value(config["sensors"]["motor_speed"]["bias"], motor_bias.tolist())
    path = write_config(config, root / "mapping.yaml")

    with SimulationEnvironment.create(path, 1, "cpu") as env:
        env._truth["angular_velocity_b"][0] = torch.tensor([0.2, -0.1, 0.3])
        env.advance(torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]))
        truth = env.observe("truth").values
        first = env.observe("sensor").values
        second = env.observe("sensor").values

        expected_accelerometer = (
            truth["force_b"][0] / env.parameters["body.mass"][0]
            + accelerometer_bias
        )
        assert_close(
            first["gyro"][0],
            truth["angular_velocity_b"][0] + gyro_bias,
            atol=2.0e-6,
            label="zero-delay gyro mapping",
        )
        assert_close(
            first["accelerometer"][0],
            expected_accelerometer,
            atol=2.0e-6,
            label="zero-delay accelerometer specific force/frame mapping",
        )
        assert_close(
            first["motor_speed"][0],
            truth["motor_speed"][0] + motor_bias,
            atol=2.0e-6,
            label="zero-delay motor-speed mapping",
        )
        for sensor_name in first:
            assert_close(
                first[sensor_name],
                second[sensor_name],
                atol=0.0,
                label=f"observe idempotence for {sensor_name}",
            )

        first["gyro"][0, 0] = 999.0
        if env.observe("sensor", ("gyro",)).values["gyro"][0, 0].item() == 999.0:
            raise AssertionError("observe() exposed mutable internal sensor state")

        print(
            "zero-delay mapping gyro="
            f"{second['gyro'][0].tolist()}, accel={second['accelerometer'][0].tolist()}, "
            f"motor={second['motor_speed'][0].tolist()}"
        )


def check_initial_accelerometer_semantics(root: Path) -> None:
    config = deterministic_config(root / "initial-accelerometer-logs")
    configure_clean_actuators(config)
    value(config["aerodynamics"]["thrust_coefficients"], [0.0, 0.0, 0.0])
    path = write_config(config, root / "initial-accelerometer.yaml")

    with SimulationEnvironment.create(path, 1, "cpu") as env:
        initial = env.observe("sensor", ("accelerometer",)).values[
            "accelerometer"
        ][0]
        expected_free_fall_specific_force = torch.zeros(3)
        assert_close(
            initial,
            expected_free_fall_specific_force,
            atol=1.0e-6,
            label="free-fall accelerometer at initialization",
        )
        assert_close(
            env.observe("truth", ("linear_acceleration_n",)).values[
                "linear_acceleration_n"
            ][0],
            torch.tensor([0.0, 0.0, 9.80665]),
            atol=1.0e-6,
            label="initial free-fall truth acceleration",
        )
        env.advance(torch.zeros((1, 5)))
        after_one_step = env.observe("sensor", ("accelerometer",)).values[
            "accelerometer"
        ][0]
        assert_close(
            after_one_step,
            expected_free_fall_specific_force,
            atol=1.0e-6,
            label="free-fall accelerometer after first dynamics step",
        )
    print(
        "initial free-fall accelerometer="
        f"{initial.tolist()}, after one step={after_one_step.tolist()}"
    )


def run() -> None:
    inputs = [10.0, 20.0, 30.0, 40.0]
    with tempfile.TemporaryDirectory(prefix="simenv-observation-check-") as directory:
        root = Path(directory)
        check_initial_accelerometer_semantics(root)
        check_zero_delay_mapping(root)
        print()
        gyro_sequence(
            root,
            "zero delay",
            inputs,
            inputs,
            sample_hz=100,
            delay=0.0,
        )
        gyro_sequence(
            root,
            "50 Hz sample/hold",
            inputs,
            [0.0, 20.0, 20.0, 40.0],
            sample_hz=50,
            delay=0.0,
        )
        gyro_sequence(
            root,
            "two-step delay",
            inputs,
            [0.0, 0.0, 10.0, 20.0],
            sample_hz=100,
            delay=0.02,
        )
        gyro_sequence(
            root,
            "1.5-step delay",
            inputs,
            [0.0, 5.0, 15.0, 25.0],
            sample_hz=100,
            delay=0.015,
            interpolation="linear",
        )
    print(
        "\nPASS: observation mapping, accelerometer time alignment, hold, delay, "
        "interpolation, and copying agree."
    )


if __name__ == "__main__":
    run()
