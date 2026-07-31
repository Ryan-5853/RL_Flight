from __future__ import annotations

import argparse

import torch

from simenv import RealtimeSimulationEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm and benchmark the single-environment 500 Hz path."
    )
    parser.add_argument("config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--paced",
        action="store_true",
        help="pace at 500 Hz instead of measuring maximum throughput",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)

    with RealtimeSimulationEnvironment.create(
        args.config,
        device=args.device,
    ) as simulation:
        compile_seconds = simulation.warmup(steps=args.warmup_steps)
        control = torch.zeros((1, 5), device=simulation.device, dtype=simulation.dtype)

        def policy(_: torch.Tensor) -> torch.Tensor:
            return control

        stats = simulation.run_policy(
            policy,
            steps=args.steps,
            realtime=args.paced,
        )

    print(f"warmup:          {compile_seconds:.3f} s")
    print(f"achieved:        {stats.achieved_hz:.1f} Hz")
    print(f"mean compute:    {stats.mean_compute_us:.1f} us")
    print(f"p50 compute:     {stats.p50_compute_us:.1f} us")
    print(f"p99 compute:     {stats.p99_compute_us:.1f} us")
    print(f"max compute:     {stats.max_compute_us:.1f} us")
    print(f"deadline misses: {stats.deadline_misses}/{stats.steps}")
    print(f"max lateness:    {stats.max_lateness_us:.1f} us")


if __name__ == "__main__":
    main()
