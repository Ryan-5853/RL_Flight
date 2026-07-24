#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_train.diagnostics import (
    finite_horizon_action_sweep,
    load_checkpoint_experiment_config,
    load_sac_diagnostic_context,
    replay_action_gradient_report,
    transition_contract_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="比较 SAC 双 Q 梯度、replay 局部动力学和真实有限时域回报"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-count", type=int, default=100_000)
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--horizon-steps", type=int, default=250)
    parser.add_argument("--minimum-yaw-rate-rad-s", type=float, default=0.5)
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=(-0.10, -0.05, 0.0, 0.05, 0.10),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = load_checkpoint_experiment_config(
        args.config,
        checkpoint,
        device=args.device,
    )
    report = {
        "schema_version": 1,
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "replay": replay_action_gradient_report(
            config,
            checkpoint,
            sample_count=args.sample_count,
            minimum_yaw_rate_rad_s=args.minimum_yaw_rate_rad_s,
        ),
    }
    with load_sac_diagnostic_context(config, checkpoint) as context:
        report["transition_contract"] = transition_contract_report(context)
        report["finite_horizon_sweep"] = finite_horizon_action_sweep(
            context,
            action_axis=args.axis,
            deltas=tuple(args.deltas),
            horizon_steps=args.horizon_steps,
            minimum_yaw_rate_rad_s=args.minimum_yaw_rate_rad_s,
        )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(output)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
