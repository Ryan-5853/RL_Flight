#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _vector(values: list[float], name: str) -> list[float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values")
    return [float(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a cascade-evaluation config from an inner-loop config"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kp", nargs=3, type=float, required=True)
    parser.add_argument("--ki", nargs=3, type=float, required=True)
    parser.add_argument("--kd", nargs=3, type=float, required=True)
    parser.add_argument("--roll-pitch-limit-rad", type=float, default=0.20)
    parser.add_argument("--yaw-rate-limit-rad-s", type=float, default=0.35)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["command_source"]["type"] = (
        "flight_train.commands:VirtualPilotCommandSource"
    )
    config["command_source"]["version"] = "2"
    config["command_source"]["params"].pop(
        "direct_angular_acceleration", None
    )
    sticks = config["command_source"]["params"]["sticks"]
    sticks["roll"]["limit_rad"] = args.roll_pitch_limit_rad
    sticks["pitch"]["limit_rad"] = args.roll_pitch_limit_rad
    sticks["yaw"]["limit_rad_s"] = args.yaw_rate_limit_rad_s
    outer = config["task"]["outer_loop"]
    outer["proportional_gain"] = _vector(args.kp, "kp")
    outer["integral_gain"] = _vector(args.ki, "ki")
    outer["derivative_gain"] = _vector(args.kd, "kd")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
