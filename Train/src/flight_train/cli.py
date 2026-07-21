from __future__ import annotations

import argparse

from .config import load_experiment_config
from .runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the recurrent flight controller")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_experiment(load_experiment_config(args.config))
        print(result["run_directory"])


if __name__ == "__main__":
    main()

