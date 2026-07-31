#!/usr/bin/env python3
"""Manually verify grid-force directions and r x F moments.

FRD convention reminder:
  x = forward, y = right, z = down.
Viewed from above, a positive right-hand moment about +z is clockwise yaw.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from common import (
    assert_close,
    configure_clean_actuators,
    configure_symmetric_grids,
    cross,
    deterministic_config,
    rodrigues,
    write_config,
)
from simenv import SimulationEnvironment


CASES = (
    ("neutral", (0.0, 0.0, 0.0)),
    ("servo_1 +", (1.0, 0.0, 0.0)),
    ("servo_2 +", (0.0, 1.0, 0.0)),
    ("servo_3 +", (0.0, 0.0, 1.0)),
    ("servo_1 -", (-1.0, 0.0, 0.0)),
    ("servo_2 -", (0.0, -1.0, 0.0)),
    ("servo_3 -", (0.0, 0.0, -1.0)),
    ("all +", (1.0, 1.0, 1.0)),
    ("all -", (-1.0, -1.0, -1.0)),
)


def expected_grid_values(
    thrust: float, servo_angles: tuple[float, float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    forces: list[list[float]] = []
    moments: list[list[float]] = []
    neutral = [0.0, 0.0, -1.0]
    for index, servo_angle in enumerate(servo_angles):
        theta = index * 2.0 * math.pi / 3.0
        axis = [math.cos(theta), math.sin(theta), 0.0]
        arm = [0.1 * axis[0], 0.1 * axis[1], 0.25]
        direction = rodrigues(neutral, axis, servo_angle)
        force = [thrust * component / 3.0 for component in direction]
        forces.append(force)
        moments.append(cross(arm, force))
    return torch.tensor(forces), torch.tensor(moments)


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="simenv-surface-check-") as directory:
        root = Path(directory)
        config = deterministic_config(root / "logs")
        configure_clean_actuators(config)
        configure_symmetric_grids(config)
        path = write_config(config, root / "surface-check.yaml")

        controls = torch.tensor(
            [[1.0, 1.0, *servo] for _, servo in CASES], dtype=torch.float32
        )
        with SimulationEnvironment.create(path, len(CASES), "cpu") as env:
            result = env.advance(controls)
            if not bool(result.valid.all()):
                raise AssertionError(f"invalid instances: {result.error_code.tolist()}")
            truth = env.observe("truth").values

        dt = 1.0 / 500.0
        expected_speed = 100.0 * (1.0 - math.exp(-dt / 0.01))
        expected_thrust = 2.0e-3 * expected_speed**2
        assert_close(
            truth["motor_speed"],
            torch.full((len(CASES), 2), expected_speed),
            label="first-order motor response",
        )
        assert_close(
            truth["total_thrust"],
            torch.full((len(CASES),), expected_thrust),
            label="coaxial thrust equation",
        )

        expected_total_moments = []
        for row, (_, servo) in enumerate(CASES):
            expected_servo_angles = tuple(
                0.2 * command * (1.0 - math.exp(-dt / 0.01))
                for command in servo
            )
            expected_force, expected_moment = expected_grid_values(
                expected_thrust, expected_servo_angles
            )
            assert_close(
                truth["servo_angle"][row],
                torch.tensor(expected_servo_angles),
                label=f"{CASES[row][0]} exact servo response",
            )
            assert_close(
                truth["grid_force_b"][row],
                expected_force,
                label=f"{CASES[row][0]} grid force",
            )
            assert_close(
                truth["grid_moment_b"][row],
                expected_moment,
                label=f"{CASES[row][0]} per-grid r x F",
            )
            expected_total_moments.append(expected_moment.sum(dim=0))
        assert_close(
            truth["moment_b"],
            torch.stack(expected_total_moments),
            label="total moment",
        )

        for row in (1, 2, 3, 7):
            if truth["moment_b"][row, 2].item() <= 0.0:
                raise AssertionError(f"{CASES[row][0]} should produce +Mz")
        for row in (4, 5, 6, 8):
            if truth["moment_b"][row, 2].item() >= 0.0:
                raise AssertionError(f"{CASES[row][0]} should produce -Mz")
        if truth["moment_b"][7, 2] <= truth["moment_b"][1:4, 2].max():
            raise AssertionError("three positive surfaces should reinforce one another")

        print("FRD: +Mz is clockwise when viewed from above (looking along +z).")
        print(
            f"Hand calculation: motor speed={expected_speed:.6f} rad/s, "
            f"total thrust={expected_thrust:.6f} N"
        )
        print("\ncase          servo angles [rad]          Mx [N m]    My [N m]    Mz [N m]")
        for row, (name, _) in enumerate(CASES):
            angles = ", ".join(f"{item:+.3f}" for item in truth["servo_angle"][row])
            moment = truth["moment_b"][row]
            print(
                f"{name:<13} [{angles}]"
                f"  {moment[0]:+10.6f}  {moment[1]:+10.6f}  {moment[2]:+10.6f}"
            )
        print("\nPASS: each grid force, each r x F moment, and all yaw signs agree.")


if __name__ == "__main__":
    run()
