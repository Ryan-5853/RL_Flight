from __future__ import annotations

import argparse

from .config import load_experiment_config, override_run_config
from .evaluation import load_fixed_evaluation_suite, run_fixed_evaluation
from .registry import ComponentRegistry


def main() -> None:
    """解析命令行并把配置文件交给训练编排入口。"""

    parser = argparse.ArgumentParser(description="Train the recurrent flight controller")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--device")
    run_parser.add_argument("--output-root")
    run_parser.add_argument("--resume-from")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--suite", required=True)
    evaluate_parser.add_argument("--device")
    evaluate_parser.add_argument("--output-root")
    args = parser.parse_args()
    if args.command == "run":
        config = load_experiment_config(args.config)
        config = override_run_config(
            config,
            device=args.device,
            output_root=args.output_root,
            resume_from=args.resume_from,
        )
        entrypoint = ComponentRegistry().build_entrypoint(
            {"type": config.entrypoint.type, "version": config.entrypoint.version}
        )
        result = entrypoint(config)
        print(result["run_directory"])
    elif args.command == "evaluate":
        config = load_experiment_config(args.config)
        config = override_run_config(config, device=args.device)
        suite = load_fixed_evaluation_suite(args.suite)
        result = run_fixed_evaluation(
            config,
            args.checkpoint,
            suite,
            output_root=args.output_root,
        )
        print(result["evaluation_directory"])


if __name__ == "__main__":
    main()
