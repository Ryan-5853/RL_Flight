from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import load_experiment_config
from .experiment import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate closed-loop LQR identification sequences"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    parser.add_argument("--device")
    parser.add_argument("--parameter-groups", type=int)
    parser.add_argument("--parallel-count", type=int)
    parser.add_argument("--initial-conditions-per-group", type=int)
    parser.add_argument("--episode-duration-s", type=float)
    parser.add_argument(
        "--log10-effectiveness-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.output_directory is not None:
        config = replace(
            config, output_directory=Path(args.output_directory).expanduser().resolve()
        )
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.parameter_groups is not None:
        if args.parameter_groups <= 0:
            parser.error("--parameter-groups must be positive")
        config = replace(config, parameter_groups=args.parameter_groups)
    if args.initial_conditions_per_group is not None:
        if args.initial_conditions_per_group <= 0:
            parser.error("--initial-conditions-per-group must be positive")
        config = replace(
            config,
            initial_conditions_per_group=args.initial_conditions_per_group,
        )
    if args.episode_duration_s is not None:
        if args.episode_duration_s <= 0.0:
            parser.error("--episode-duration-s must be positive")
        steps = args.episode_duration_s * config.control_hz
        if steps != round(steps):
            parser.error("--episode-duration-s must be an integer number of steps")
        if args.episode_duration_s < config.window.length_s:
            parser.error("--episode-duration-s cannot be shorter than the window")
        config = replace(config, episode_duration_s=args.episode_duration_s)
    if args.log10_effectiveness_range is not None:
        low, high = args.log10_effectiveness_range
        if low >= high:
            parser.error("--log10-effectiveness-range must be ordered")
        config = replace(config, log10_effectiveness_range=(low, high))
    if args.parallel_count is not None:
        if args.parallel_count < config.initial_conditions_per_group:
            parser.error("--parallel-count must fit one parameter group")
        parallel = args.parallel_count - (
            args.parallel_count % config.initial_conditions_per_group
        )
        config = replace(config, parallel_count=parallel)
    elif config.parallel_count < config.initial_conditions_per_group:
        config = replace(
            config, parallel_count=config.initial_conditions_per_group
        )
    else:
        config = replace(
            config,
            parallel_count=config.parallel_count
            - config.parallel_count % config.initial_conditions_per_group,
        )
    manifest = generate_dataset(config)
    print(config.output_directory)
    print(manifest["totals"])


if __name__ == "__main__":
    main()
