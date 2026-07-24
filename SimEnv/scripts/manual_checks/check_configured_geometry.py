#!/usr/bin/env python3
"""Audit the declared grid geometry in any SimEnv YAML configuration.

This check does not substitute idealized grid locations. It reads the configured
centers, center of mass, deflection axes, gains, offsets, thrust partitions,
self-attenuation curves, and coupling matrix, then evaluates their scalar
Rodrigues/cross-product equations at a small positive and negative deflection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from common import REPO_ROOT, cross, rodrigues


def raw_value(node: dict[str, Any]) -> Any:
    return node["value"]


def lookup(value: float, table: list[list[float]]) -> float:
    value = max(table[0][0], min(table[-1][0], value))
    for lower, upper in zip(table, table[1:]):
        if value <= upper[0]:
            fraction = (value - lower[0]) / (upper[0] - lower[0])
            return lower[1] + fraction * (upper[1] - lower[1])
    return table[-1][1]


def vector_subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def vector_sum(vectors: list[list[float]]) -> list[float]:
    return [sum(vector[axis] for vector in vectors) for axis in range(3)]


def configured_grid_moments(
    config: dict[str, Any], servo_angles: list[float]
) -> list[list[float]]:
    aerodynamics = config["aerodynamics"]
    grids = aerodynamics["grids"]
    neutral = raw_value(aerodynamics["neutral_thrust_direction_b"])
    center_of_mass = raw_value(config["body"]["center_of_mass_b"])
    partitions = [
        raw_value(aerodynamics["thrust_partition"][f"grid_{index + 1}"])
        for index in range(3)
    ]
    attenuation = [
        lookup(
            abs(servo_angles[index]),
            raw_value(grids[index]["self_attenuation_curve"]),
        )
        for index in range(3)
    ]
    coupling = raw_value(aerodynamics["coupling_attenuation"])
    effective_attenuation = [
        max(
            0.0,
            min(
                1.0,
                attenuation[row]
                - sum(
                    coupling[row][column] * (1.0 - attenuation[column])
                    for column in range(3)
                ),
            ),
        )
        for row in range(3)
    ]

    moments = []
    for index, grid in enumerate(grids):
        gain = raw_value(grid["vector_deflection"]["gain"])
        offset = raw_value(grid["vector_deflection"]["offset"])
        vector_angle = gain * servo_angles[index] + offset
        direction = rodrigues(
            neutral,
            raw_value(grid["deflection_axis_b"]),
            vector_angle,
        )
        # Unit total thrust is sufficient for sign and geometry checks.
        force = [
            partitions[index] * effective_attenuation[index] * component
            for component in direction
        ]
        arm = vector_subtract(
            raw_value(grid["aerodynamic_center_b"]), center_of_mass
        )
        moments.append(cross(arm, force))
    return moments


def audit(path: Path, deflection_rad: float = 0.1) -> None:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    neutral = vector_sum(configured_grid_moments(config, [0.0, 0.0, 0.0]))
    positive_deltas: list[float] = []
    negative_deltas: list[float] = []
    for index in range(3):
        positive = [0.0, 0.0, 0.0]
        negative = [0.0, 0.0, 0.0]
        positive[index] = deflection_rad
        negative[index] = -deflection_rad
        positive_total = vector_sum(configured_grid_moments(config, positive))
        negative_total = vector_sum(configured_grid_moments(config, negative))
        positive_deltas.append(positive_total[2] - neutral[2])
        negative_deltas.append(negative_total[2] - neutral[2])

    all_positive = vector_sum(
        configured_grid_moments(config, [deflection_rad] * 3)
    )
    all_negative = vector_sum(
        configured_grid_moments(config, [-deflection_rad] * 3)
    )
    all_positive_delta = all_positive[2] - neutral[2]
    all_negative_delta = all_negative[2] - neutral[2]

    tolerance = 1.0e-10
    if not all(item > tolerance for item in positive_deltas):
        raise AssertionError(
            f"{path}: positive individual deflections do not all produce +delta Mz: "
            f"{positive_deltas}"
        )
    if not all(item < -tolerance for item in negative_deltas):
        raise AssertionError(
            f"{path}: negative individual deflections do not all produce -delta Mz: "
            f"{negative_deltas}"
        )
    if all_positive_delta <= tolerance or all_negative_delta >= -tolerance:
        raise AssertionError(
            f"{path}: same-direction grid deflections have wrong yaw signs: "
            f"{all_positive_delta=}, {all_negative_delta=}"
        )

    print(f"configuration: {path.resolve()}")
    print("  unit-thrust individual +delta Mz:", [f"{item:+.8f}" for item in positive_deltas])
    print("  unit-thrust individual -delta Mz:", [f"{item:+.8f}" for item in negative_deltas])
    print(
        "  unit-thrust all-surface delta Mz: "
        f"positive={all_positive_delta:+.8f}, negative={all_negative_delta:+.8f}"
    )
    print("  PASS: declared centers/axes/gains produce the expected yaw signs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        default=[REPO_ROOT / "configs" / "example.yaml"],
        help="SimEnv YAML files to audit",
    )
    parser.add_argument(
        "--deflection-rad",
        type=float,
        default=0.1,
        help="mechanical deflection used for the sign check (default: 0.1)",
    )
    args = parser.parse_args()
    for index, path in enumerate(args.configs):
        if index:
            print()
        audit(path, args.deflection_rad)


if __name__ == "__main__":
    main()

