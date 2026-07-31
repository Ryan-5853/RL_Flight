#!/usr/bin/env python3
"""Recompute several rigid-body steps independently from the reported forces."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from common import (
    assert_close,
    configure_clean_actuators,
    configure_symmetric_grids,
    deterministic_config,
    integrate_quaternion,
    rotate_body_to_world,
    value,
    write_config,
)
from simenv import SimulationEnvironment


GRAVITY_N = torch.tensor([0.0, 0.0, 9.80665])


def check_free_fall(path: Path, dt: float, steps: int = 5) -> None:
    with SimulationEnvironment.create(path, 1, "cpu") as env:
        zero_control = torch.zeros((1, 5))
        for step in range(1, steps + 1):
            env.advance(zero_control)
            truth = env.observe("truth").values
            expected_velocity = GRAVITY_N * (step * dt)
            # 中点常加速度位置更新对自由落体是精确的。
            expected_position = 0.5 * GRAVITY_N * (step * dt) ** 2
            assert_close(
                truth["linear_acceleration_n"][0],
                GRAVITY_N,
                atol=1.0e-6,
                label=f"free-fall acceleration step {step}",
            )
            assert_close(
                truth["velocity_n"][0],
                expected_velocity,
                atol=1.0e-6,
                label=f"free-fall velocity step {step}",
            )
            assert_close(
                truth["position_n"][0],
                expected_position,
                atol=1.0e-6,
                label=f"free-fall position step {step}",
            )
        print(
            f"free fall ({steps} steps): vz={truth['velocity_n'][0, 2]:.7f} m/s, "
            f"z={truth['position_n'][0, 2]:.7f} m"
        )


def check_forced_steps(path: Path, dt: float, steps: int = 6) -> None:
    with SimulationEnvironment.create(path, 1, "cpu") as env:
        control = torch.tensor([[1.0, 1.0, 0.7, 0.7, 0.7]])
        print("\nstep  servo[rad]  thrust[N]    Mz[Nm]      r[rad/s]    qz         max residual")
        for step in range(1, steps + 1):
            previous = env.observe("truth").values
            env.advance(control)
            current = env.observe("truth").values

            mass = env.parameters["body.mass"][0]
            inertia = env.parameters["body.inertia_diagonal_b"][0]
            old_q = previous["attitude_q_wb"][0]
            old_velocity = previous["velocity_n"][0]
            old_position = previous["position_n"][0]
            old_omega = previous["angular_velocity_b"][0]

            angular_momentum = inertia * old_omega
            initial_angular_acceleration = (
                current["moment_b"][0]
                - torch.linalg.cross(old_omega, angular_momentum)
            ) / inertia
            midpoint_omega = old_omega + 0.5 * initial_angular_acceleration * dt
            midpoint_q = integrate_quaternion(old_q, midpoint_omega, 0.5 * dt)
            force_n = rotate_body_to_world(midpoint_q, current["force_b"][0])
            expected_linear_acceleration = force_n / mass + GRAVITY_N
            midpoint_angular_momentum = inertia * midpoint_omega
            midpoint_angular_acceleration = (
                current["moment_b"][0]
                - torch.linalg.cross(midpoint_omega, midpoint_angular_momentum)
            ) / inertia
            expected_velocity = old_velocity + expected_linear_acceleration * dt
            expected_position = (
                old_position
                + old_velocity * dt
                + 0.5 * expected_linear_acceleration * dt**2
            )
            expected_omega = old_omega + midpoint_angular_acceleration * dt
            expected_q = integrate_quaternion(old_q, midpoint_omega, dt)
            final_force_n = rotate_body_to_world(
                expected_q, current["force_b"][0]
            )
            expected_final_linear_acceleration = (
                final_force_n / mass + GRAVITY_N
            )
            expected_final_angular_acceleration = (
                current["moment_b"][0]
                - torch.linalg.cross(
                    expected_omega, inertia * expected_omega
                )
            ) / inertia

            checks = (
                (
                    current["linear_acceleration_n"][0],
                    expected_final_linear_acceleration,
                    "a",
                ),
                (
                    current["angular_acceleration_b"][0],
                    expected_final_angular_acceleration,
                    "alpha",
                ),
                (current["velocity_n"][0], expected_velocity, "velocity"),
                (current["position_n"][0], expected_position, "position"),
                (current["angular_velocity_b"][0], expected_omega, "omega"),
                (current["attitude_q_wb"][0], expected_q, "quaternion"),
            )
            residual = 0.0
            for actual, expected, label in checks:
                assert_close(
                    actual,
                    expected,
                    atol=3.0e-6,
                    label=f"forced step {step} {label}",
                )
                residual = max(residual, (actual - expected).abs().max().item())
            norm_error = abs(
                torch.linalg.vector_norm(current["attitude_q_wb"][0]).item() - 1.0
            )
            if norm_error > 2.0e-6:
                raise AssertionError(f"quaternion norm drift at step {step}: {norm_error}")

            print(
                f"{step:>4}  {current['servo_angle'][0, 0]:>10.6f}"
                f"  {current['total_thrust'][0]:>9.5f}"
                f"  {current['moment_b'][0, 2]:>+10.6f}"
                f"  {current['angular_velocity_b'][0, 2]:>+11.7f}"
                f"  {current['attitude_q_wb'][0, 3]:>+9.6f}"
                f"  {residual:.2e}"
            )


def run() -> None:
    physics_hz = 500
    dt = 1.0 / physics_hz
    with tempfile.TemporaryDirectory(prefix="simenv-step-check-") as directory:
        root = Path(directory)
        free_fall_config = deterministic_config(root / "free-fall-logs")
        configure_clean_actuators(free_fall_config)
        # Zero control then gives no aerodynamic force or moment.
        free_fall_path = write_config(free_fall_config, root / "free-fall.yaml")
        check_free_fall(free_fall_path, dt)

        forced_config = deterministic_config(root / "forced-logs")
        configure_clean_actuators(forced_config)
        # 令执行器在 2 ms 内达到指令，使该检查可独立核对恒定力/力矩的
        # 中点刚体积分，而不重复实现执行器内部的解析过渡过程。
        for motor in forced_config["motors"]:
            value(motor["time_constant"], 1.0e-6)
        for servo in forced_config["servos"]:
            value(servo["tau"], 1.0e-6)
            value(servo["max_speed"], 1.0e9)
        configure_symmetric_grids(forced_config)
        value(forced_config["body"]["mass"], 2.0)
        forced_path = write_config(forced_config, root / "forced.yaml")
        check_forced_steps(forced_path, dt)
    print(
        "\nPASS: 500 Hz midpoint rigid-body equations, exact translation, "
        "and quaternion exponential integration agree."
    )


if __name__ == "__main__":
    run()
