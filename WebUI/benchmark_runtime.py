#!/usr/bin/env python3
"""Benchmark the complete controller + single-environment realtime runtime."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from inference_package import load_flight_deploy_package
from runtime import CpuRuntimeSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("simenv", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument(
        "--inference-package",
        type=Path,
        default=None,
        help="required only for controller.type=neural",
    )
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")

    with tempfile.TemporaryDirectory(prefix="webui-benchmark-") as temporary:
        session = CpuRuntimeSession(
            args.simenv.read_text(encoding="utf-8"),
            args.test.read_text(encoding="utf-8"),
            args.inference_package,
            temporary,
            inference_package_loader=load_flight_deploy_package,
        )
        try:
            print(f"warmup_s: {session.warmup_seconds:.3f}")
            print(f"controller: {session.controller.controller_type}")
            print(f"controller_compiled: {session.controller_compiled}")
            started = time.monotonic()
            session.start()
            while time.monotonic() - started < args.seconds:
                time.sleep(min(1.0, args.seconds))
                status = session.status()
                if status["fault"]:
                    raise RuntimeError(status["fault"])
            session.pause()
            elapsed = time.monotonic() - started
            status = session.status()
            steps = status["control_steps"]
            overruns = status["loop_overruns"]
            print(f"elapsed_s: {elapsed:.3f}")
            print(f"steps: {steps}")
            print(f"achieved_hz: {steps / elapsed:.3f}")
            print(f"last_window_hz: {status['measured_control_hz']:.3f}")
            print(f"loop_overruns: {overruns}")
            print(f"overrun_ratio: {overruns / max(1, steps):.6f}")
        finally:
            session.close()


if __name__ == "__main__":
    main()
