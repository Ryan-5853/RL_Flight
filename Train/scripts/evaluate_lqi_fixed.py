#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch

from flight_train.config import load_experiment_config, override_run_config
from flight_train.evaluation import (
    _create_packed_evaluation_environment,
    _precomputed_scenario_trajectory,
    _quaternion_to_euler,
    _repeat_packed_scenario_initial_state,
    _score_trajectory,
    load_fixed_evaluation_suite,
)
from flight_train.math import tilt_angle


def _expand_first(
    values: Mapping[str, torch.Tensor], count: int
) -> dict[str, torch.Tensor]:
    return {
        name: value[:1].expand(count, *value.shape[1:]).clone()
        for name, value in values.items()
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate nominal LQI on the same packed fixed suite as Train"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--nominal-simulator", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from flight_controller import (
        ControllerContext,
        ControllerReference,
        ControllerState,
        create_controller,
        load_controller_config,
    )
    from simenv.config import load_and_materialize

    config = override_run_config(
        load_experiment_config(args.config), device=args.device
    )
    suite = load_fixed_evaluation_suite(args.suite)
    device = torch.device(args.device)
    env = _create_packed_evaluation_environment(config, suite, device)
    try:
        group_size = suite.parallel_count
        control_hz = env.spec.control_hz
        count = env.spec.parallel_count
        scenario_steps = tuple(
            round(item.duration_s * control_hz) for item in suite.scenarios
        )
        maximum_steps = max(scenario_steps)
        env.reset()
        _repeat_packed_scenario_initial_state(env, group_size)
        initial_position = env.simulator.observe(
            "truth", ("position_n",)
        ).values["position_n"]

        nominal = load_and_materialize(
            Path(args.nominal_simulator).expanduser().resolve(),
            1,
            device,
            config.torch_dtype,
        )
        nominal_parameters = _expand_first(nominal.parameters, count)
        controller = create_controller(
            load_controller_config(args.controller),
            ControllerContext(
                batch_size=count,
                device=device,
                dtype=config.torch_dtype,
                control_dt=1.0 / float(control_hz),
                parameters=nominal_parameters,
            ),
        )
        controller.reset(
            torch.ones(count, device=device, dtype=torch.bool)
        )

        trajectories = {}
        for index, (scenario, steps) in enumerate(
            zip(suite.scenarios, scenario_steps)
        ):
            group = slice(index * group_size, (index + 1) * group_size)
            trajectories[scenario.name] = _precomputed_scenario_trajectory(
                scenario,
                steps,
                group_size,
                control_hz,
                initial_position[group],
                device,
                config.torch_dtype,
            )

        trace_names = (
            "alive",
            "action",
            "position_n",
            "velocity_n",
            "attitude_q_wb",
            "yaw_rate_error_rad_s",
            "lower_motor_differential_pwm",
            "servo_common_command",
            "servo_cyclic_command_norm",
        )
        packed: dict[str, torch.Tensor] = {
            "alive": torch.empty(maximum_steps, count, device=device, dtype=torch.bool),
            "action": torch.empty(
                maximum_steps, count, 5, device=device, dtype=config.torch_dtype
            ),
        }
        for name in ("position_n", "velocity_n"):
            packed[name] = torch.empty(
                maximum_steps, count, 3, device=device, dtype=config.torch_dtype
            )
        packed["attitude_q_wb"] = torch.empty(
            maximum_steps, count, 4, device=device, dtype=config.torch_dtype
        )
        for name in trace_names[5:]:
            packed[name] = torch.empty(
                maximum_steps, count, device=device, dtype=config.torch_dtype
            )

        horizon_steps = torch.tensor(
            [steps for steps in scenario_steps for _ in range(group_size)],
            device=device,
            dtype=torch.int64,
        )
        alive = torch.ones(count, device=device, dtype=torch.bool)
        survival = torch.tensor(
            [item.duration_s for item in suite.scenarios for _ in range(group_size)],
            device=device,
            dtype=config.torch_dtype,
        )
        zeros = torch.zeros(count, 3, device=device, dtype=config.torch_dtype)
        collective = torch.zeros(count, 1, device=device, dtype=config.torch_dtype)

        for step in range(maximum_steps):
            active = alive & (step < horizon_steps)
            truth = env.simulator.observe("truth").values
            pilot = env.command_source.snapshot()
            desired_yaw_rate = env.command_source.desired_yaw_rate.clone()
            target_rate = zeros.clone()
            target_rate[:, 2:3] = desired_yaw_rate
            state = ControllerState.from_truth(truth)
            reference = ControllerReference(
                target_position_n=torch.stack(
                    [
                        trajectories[item.name]["target_position_n"][
                            min(step, scenario_steps[index] - 1)
                        ]
                        for index, item in enumerate(suite.scenarios)
                    ]
                ).reshape(count, 3),
                target_velocity_n=torch.stack(
                    [
                        trajectories[item.name]["target_velocity_n"][
                            min(step, scenario_steps[index] - 1)
                        ]
                        for index, item in enumerate(suite.scenarios)
                    ]
                ).reshape(count, 3),
                target_attitude_q_wb=pilot.target_attitude_q_wb,
                target_angular_velocity_b=target_rate,
                collective_command=collective,
            )
            command = controller.step(state, reference, active).command
            result = env.simulator.advance(command, active_mask=active)
            next_truth = env.simulator.observe("truth").values
            env.command_source.step(
                -next_truth["position_n"][:, 2:3], active_mask=active
            )
            rate = next_truth["angular_velocity_b"]
            tilt = tilt_angle(next_truth["attitude_q_wb"]).squeeze(-1)
            if config.task.terminate_angular_rate_axes == "yaw":
                rate_limit_value = rate[:, 2].abs()
            else:
                rate_limit_value = torch.linalg.vector_norm(rate, dim=-1)
            done = (
                ~result.valid
                | (tilt > config.task.terminate_tilt_rad)
                | (
                    rate_limit_value
                    > config.task.terminate_angular_rate_rad_s
                )
            )
            newly_done = active & done
            survival = torch.where(
                newly_done,
                torch.full_like(survival, (step + 1) / float(control_hz)),
                survival,
            )
            sample_alive = alive & ~done
            normalized_command = torch.cat(
                (2.0 * command[:, :2] - 1.0, command[:, 2:]), dim=-1
            )
            servo = command[:, 2:5]
            servo_common = servo.mean(dim=-1)
            packed["alive"][step].copy_(sample_alive)
            packed["action"][step].copy_(normalized_command)
            packed["position_n"][step].copy_(next_truth["position_n"])
            packed["velocity_n"][step].copy_(next_truth["velocity_n"])
            packed["attitude_q_wb"][step].copy_(next_truth["attitude_q_wb"])
            packed["yaw_rate_error_rad_s"][step].copy_(
                rate[:, 2] - desired_yaw_rate[:, 0]
            )
            packed["lower_motor_differential_pwm"][step].copy_(
                command[:, 1] - command[:, 0]
            )
            packed["servo_common_command"][step].copy_(servo_common)
            packed["servo_cyclic_command_norm"][step].copy_(
                torch.linalg.vector_norm(servo - servo_common[:, None], dim=-1)
            )
            alive = sample_alive

        report: dict[str, object] = {
            "schema_version": 1,
            "status": "completed",
            "controller": "nominal_lqi_5dof",
            "actual_simulator_config": str(config.simulator_config),
            "nominal_simulator_config": str(Path(args.nominal_simulator).resolve()),
            "controller_config": str(Path(args.controller).resolve()),
            "suite_config": str(Path(args.suite).resolve()),
            "device": str(device),
            "seed": suite.seed,
            "score_weights": dict(suite.score_weights),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "action_metric_note": (
                "LQI action is normalized full 5D physical command; neural reports "
                "3D residual action, so action scores are not directly homogeneous"
            ),
            "scenarios": {},
        }
        scores = []
        for index, (scenario, steps) in enumerate(
            zip(suite.scenarios, scenario_steps)
        ):
            group = slice(index * group_size, (index + 1) * group_size)
            trajectory = trajectories[scenario.name]
            trajectory.update(
                {name: value[:steps, group] for name, value in packed.items()}
            )
            trajectory["actual_euler_rad"] = _quaternion_to_euler(
                trajectory["attitude_q_wb"]
            )
            cpu = {name: value.cpu() for name, value in trajectory.items()}
            metrics = _score_trajectory(
                cpu,
                survival[group].cpu(),
                scenario,
                suite.limits,
                control_hz,
                suite.score_weights,
                self_stabilize_tracking=suite.self_stabilize_tracking,
                yaw_rate_tracking_weight=suite.yaw_rate_tracking_weight,
            )
            report["scenarios"][scenario.name] = metrics
            scores.append(float(metrics["total_score"]))
        report["total_score"] = sum(scores) / len(scores)
        common_weight = (
            suite.score_weights["survival"]
            + suite.score_weights["tracking"]
            + suite.score_weights["response"]
        )
        report["common_score_excluding_action"] = sum(
            (
                suite.score_weights["survival"] * value["scores"]["survival"]["mean"]
                + suite.score_weights["tracking"] * value["scores"]["tracking"]["mean"]
                + suite.score_weights["response"] * value["scores"]["response"]["mean"]
            )
            / common_weight
            for value in report["scenarios"].values()
        ) / len(suite.scenarios)

        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
