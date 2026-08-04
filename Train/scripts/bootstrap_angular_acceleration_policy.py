#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from tensordict import TensorDict

from flight_train.angular_acceleration_teacher import (
    IncrementalAngularAccelerationTeacher,
)
from flight_train.config import load_experiment_config, override_run_config
from flight_train.envs import SimEnvAdapter
from flight_train.models import build_sac_actor_critic
from flight_train.randomization import StaticRandomizer
from flight_train.recording import load_checkpoint
from flight_train.registry import ComponentRegistry
from flight_train.runner import _restore_policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap a SAC actor by cloning a point-specific INDI teacher"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gain", type=float, default=0.08)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--retain-mean",
        action="store_true",
        help="Retain the source actor mean instead of resetting its output head",
    )
    parser.add_argument(
        "--execute-policy-fraction",
        type=float,
        default=0.0,
        help="Linearly ramp the executed teacher/policy blend to this fraction",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.log_interval <= 0:
        raise ValueError("iterations and log interval must be positive")
    if not 0.0 <= args.execute_policy_fraction <= 1.0:
        raise ValueError("execute policy fraction must be inside [0, 1]")

    config = override_run_config(
        load_experiment_config(args.config), device=args.device
    )
    if config.algorithm_name != "sac":
        raise ValueError("teacher bootstrap requires a SAC experiment")
    device = torch.device(args.device)
    torch.manual_seed(config.run.seed + 901)
    torch.cuda.manual_seed_all(config.run.seed + 901)

    reward_calculator = ComponentRegistry().build_reward(
        {
            "type": config.reward.calculator.type,
            "version": config.reward.calculator.version,
            "params": config.reward.calculator.params,
        }
    )
    randomizer = StaticRandomizer(
        config.static_parameter_specs(),
        config.static_randomization.seed,
        device,
        config.torch_dtype,
    )
    env = SimEnvAdapter.create(
        config.simulator_config,
        config.run.parallel_count,
        device,
        config.torch_dtype,
        config.task,
        reward_calculator=reward_calculator,
        static_randomizer=randomizer,
        dynamic_randomization=dict(config.dynamic_randomization.parameters),
        dynamic_seed=config.dynamic_randomization.seed,
        reward_context_fields=tuple(
            dict(field) for field in config.reward.context_fields
        ),
        command_source_config=config.command_source,
        control_contract_config=config.control_contract,
    )
    try:
        model = build_sac_actor_critic(
            env.spec.observation_dim,
            env.spec.action_dim,
            config.model,
            device,
            config.torch_dtype,
        )
        source_state = load_checkpoint(args.source_checkpoint)
        _restore_policy(
            source_state,
            model=model,
            reset_mean_output=not args.retain_mean,
        )
        del source_state
        teacher = IncrementalAngularAccelerationTeacher.create(
            env.simulator.parameters,
            config.control_contract,
            config.task,
            (args.gain, args.gain, args.gain),
        )
        optimizer = torch.optim.Adam(
            model.actor.parameters(), lr=args.learning_rate
        )
        observation_td = env.reset()
        observation = observation_td["observation"]
        is_init = observation_td["is_init"]
        env.command_source.set_curriculum_scale(1.0)
        interval_loss = 0.0
        interval_action_error = 0.0
        records: list[dict[str, float | int]] = []
        for iteration in range(1, args.iterations + 1):
            with torch.no_grad():
                teacher_action, _ = teacher.forward_step(
                    observation, None, is_init
                )
            policy_input = TensorDict(
                {"observation": observation},
                batch_size=[observation.shape[0]],
                device=device,
            )
            model.policy_module(policy_input)
            predicted_action = torch.tanh(policy_input["loc"])
            loss = torch.nn.functional.smooth_l1_loss(
                predicted_action, teacher_action
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                action_error = torch.sqrt(
                    torch.mean((predicted_action - teacher_action).square())
                )
                policy_fraction = (
                    args.execute_policy_fraction
                    * iteration
                    / args.iterations
                )
                executed_action = torch.lerp(
                    teacher_action,
                    predicted_action.detach(),
                    policy_fraction,
                )
                transition = env.step(executed_action)
                observation = transition["observation"]
                is_init = transition["is_init"]
            interval_loss += float(loss.detach())
            interval_action_error += float(action_error)
            if iteration % args.log_interval == 0:
                record = {
                    "iteration": iteration,
                    "examples": iteration * config.run.parallel_count,
                    "loss": interval_loss / args.log_interval,
                    "action_rmse": interval_action_error / args.log_interval,
                    "executed_policy_fraction": policy_fraction,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
                interval_loss = 0.0
                interval_action_error = 0.0

        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": 1,
            "kind": "angular_acceleration_teacher_bootstrap",
            "run_id": f"teacher_bootstrap_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "global_control_steps": 0,
            "actor": model.actor.state_dict(),
            "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
            "teacher_gain": [args.gain, args.gain, args.gain],
            "retained_source_mean": args.retain_mean,
            "execute_policy_fraction": args.execute_policy_fraction,
            "training_records": records,
        }
        torch.save(state, output)
        digest = _sha256(output)
        output.with_suffix(output.suffix + ".sha256").write_text(
            digest + "\n", encoding="ascii"
        )
        report = {
            "checkpoint": str(output),
            "sha256": digest,
            "iterations": args.iterations,
            "examples": args.iterations * config.run.parallel_count,
            "final": records[-1],
        }
        output.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
