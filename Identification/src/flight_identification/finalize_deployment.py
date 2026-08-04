from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def finalize(args: argparse.Namespace) -> Mapping[str, Any]:
    source = Path(args.artifact).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    reports = [
        json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        for path in args.nonlinear_reports
    ]
    checks = []
    for report in reports:
        checks.append(
            {
                "report_duration_s": report["duration_s"],
                "prefix_reproducible": report["prefix_max_absolute_difference"] == 0.0,
                "overall_safety_non_degraded": (
                    report["adaptive"]["safe_fraction"]
                    >= report["nominal"]["safe_fraction"]
                ),
                "accepted_safety_non_degraded": (
                    report["accepted_subset"]["adaptive_safe_fraction"]
                    >= report["accepted_subset"]["nominal_safe_fraction"]
                ),
                "accepted_convergence_non_degraded": (
                    report["accepted_subset"]["adaptive_converged_fraction"]
                    >= report["accepted_subset"]["nominal_converged_fraction"]
                ),
                "accepted_count": report["accepted_count"],
            }
        )
    passed = all(
        check["prefix_reproducible"]
        and check["overall_safety_non_degraded"]
        and check["accepted_safety_non_degraded"]
        and check["accepted_convergence_non_degraded"]
        and check["accepted_count"] >= args.minimum_accepted
        for check in checks
    )
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    artifact["nonlinear_validation"] = {
        "passed": passed,
        "checks": checks,
        "report_paths": [str(Path(path).expanduser().resolve()) for path in args.nonlinear_reports],
    }
    artifact["deployment_mode"] = "adaptive" if passed else "shadow_only"
    artifact["gain_gate"] = dict(artifact["gain_gate"])
    artifact["gain_gate"]["enabled"] = bool(
        artifact["gain_gate"].get("enabled", False) and passed
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    report = {
        "schema_version": 1,
        "source_artifact": str(source),
        "output_artifact": str(output),
        "nonlinear_validation_passed": passed,
        "deployment_mode": artifact["deployment_mode"],
        "gain_updates_enabled": artifact["gain_gate"]["enabled"],
        "checks": checks,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze adaptive or shadow-only mode after nonlinear audits"
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--nonlinear-reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-accepted", type=int, default=100)
    return parser


def main() -> None:
    finalize(build_parser().parse_args())


if __name__ == "__main__":
    main()
