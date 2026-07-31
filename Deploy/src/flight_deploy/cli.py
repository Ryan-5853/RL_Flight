from __future__ import annotations

import argparse
import json
from typing import Sequence

from .adapters import default_registry
from .bundle import BundleBuilder, DeploymentBundle
from .errors import FlightDeployError
from .runtime import PolicyRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flight-deploy",
        description="Optimize checkpoints into portable low-latency policy bundles",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("checkpoint")
    inspect_parser.add_argument("--adapter")
    inspect_parser.add_argument("--skip-checksum", action="store_true")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("checkpoint")
    export_parser.add_argument("output")
    export_parser.add_argument("--adapter")
    export_parser.add_argument("--skip-checksum", action="store_true")
    export_parser.add_argument("--overwrite", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("bundle")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("bundle")
    benchmark_parser.add_argument("--device", default="cpu")
    benchmark_parser.add_argument(
        "--backend",
        choices=("torchscript", "torch_export"),
        default="torchscript",
    )
    benchmark_parser.add_argument("--steps", type=int, default=10_000)
    benchmark_parser.add_argument("--warmup-steps", type=int, default=100)
    benchmark_parser.add_argument("--batch-size", type=int, default=1)

    subparsers.add_parser("adapters")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "adapters":
            print(json.dumps({"adapters": default_registry().names()}, indent=2))
            return 0
        if args.command == "inspect":
            value = BundleBuilder().inspect(
                args.checkpoint,
                adapter_name=args.adapter,
                verify_checksum=not args.skip_checksum,
            )
        elif args.command == "export":
            bundle = BundleBuilder().build(
                args.checkpoint,
                args.output,
                adapter_name=args.adapter,
                verify_checksum=not args.skip_checksum,
                overwrite=args.overwrite,
            )
            value = {
                "bundle": str(bundle.directory),
                "manifest": bundle.manifest,
            }
        elif args.command == "verify":
            bundle = DeploymentBundle.load(args.bundle)
            value = {"bundle": str(bundle.directory), "valid": True}
        elif args.command == "benchmark":
            runtime = PolicyRuntime.load(
                args.bundle,
                device=args.device,
                backend=args.backend,
            )
            value = runtime.benchmark(
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                batch_size=args.batch_size,
            ).as_dict()
        else:
            raise AssertionError(args.command)
    except (FlightDeployError, FileExistsError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
