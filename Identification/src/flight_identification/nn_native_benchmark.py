from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .robustness_benchmark import (
    CROSS_CELLS,
    SINGLE_DIMS,
    SINGLE_STRENGTHS,
    apply_error,
    error_cells,
)


def _single_strong_cells() -> list[dict[str, Any]]:
    cells = [{"name": "baseline", "fields": (), "strengths": ((),)}]
    for dim in SINGLE_DIMS:
        strength = SINGLE_STRENGTHS[dim][-1]
        cells.append(
            {
                "name": f"{dim}_x{strength}",
                "fields": (dim,),
                "strengths": ((strength,),),
            }
        )
    for cross in CROSS_CELLS:
        strength = cross["strengths"][-1]
        cells.append(
            {
                "name": f"{cross['name']}_x{'_'.join(str(round(v, 3)) for v in strength)}",
                "fields": tuple(cross["fields"]),
                "strengths": (strength,),
            }
        )
    return cells


def run(args: argparse.Namespace) -> dict[str, Any]:
    from inference_package import (
        InferenceModelAdapter,
        load_flight_deploy_package,
    )
    from simenv import SimulationEnvironment
    from simenv.config import load_and_materialize

    from flight_controller import (
        ControllerContext,
        create_controller,
    )
    from flight_controller.pilot import VirtualPilotHeightController
    from flight_identification.robustness_benchmark import (
        _expand_mapping,
        _simulate_neural_episode,
    )
    from .experiment import FirstOrderServoObserver

    device = torch.device(args.device)
    m = load_and_materialize(args.simulator_config, 1, device, torch.float32)
    pilot_config = {
        "throttle": {
            "minimum": 0.20,
            "maximum": 0.85,
            "initial": 0.56,
            "spool_duration_s": 0.10,
        },
        "height_controller": {
            "mode": "incremental",
            "target_m": 0.0,
            "proportional_gain": 0.08,
            "integral_gain": 0.04,
            "error_limit_m": 5.0,
        },
    }
    experiment = argparse.Namespace(
        control_hz=500,
        convergence=argparse.Namespace(
            maximum_roll_pitch_error_rad=0.0349066,
            maximum_angular_rate_rad_s=0.10,
            hold_s=0.5,
            safety_tilt_rad=1.0471976,
            safety_angular_rate_rad_s=10.0,
        ),
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    repeats = args.n_episodes
    from flight_identification.repeated_trial_evaluation import (
        _sample_near_equilibrium,
    )

    attitude, angular_velocity = _sample_near_equilibrium(
        repeats,
        args.maximum_initial_tilt_rad,
        args.maximum_initial_rate_rad_s,
        generator,
        torch.float32,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = _single_strong_cells()
    bundles = {
        "e2e_nn": (args.e2e_bundle, "residual_4"),
        "aa_nn": (args.aa_bundle, "coaxial_differential_cyclic_3"),
    }
    for cell in cells:
        perturbed = {
            name: value.repeat(repeats, *([1] * (value.ndim - 1))).clone()
            for name, value in m.parameters.items()
        }
        if cell["fields"]:
            apply_error(perturbed, cell["fields"], cell["strengths"][0])
        for scheme, (bundle, output_mode) in bundles.items():
            target = output_dir / f"{cell['name']}__{scheme}.json"
            if target.exists() and not args.force:
                continue
            package = load_flight_deploy_package(
                Path(bundle), device, torch.float32
            )
            adapter = InferenceModelAdapter(package)
            collected: dict[str, list[Any]] = {key: [] for key in (
                "safe", "converged", "cum_roll_pitch_error_rad_s",
                "cum_rate_error_rad_s", "cum_yaw_rate_error_rad_s",
                "peak_tilt_rad", "peak_rate_rad_s",
                "final_attitude_error_rad", "final_rate_rad_s",
                "saturation_fraction", "height_min_m", "height_max_m",
            )}
            for index in range(repeats):
                package.reset()
                one_params = {
                    name: value[index : index + 1].clone()
                    for name, value in perturbed.items()
                }
                controller = create_controller(
                    {"type": "neural", "params": {"output_mode": output_mode}},
                    ControllerContext(
                        batch_size=1,
                        device=device,
                        dtype=torch.float32,
                        control_dt=1.0 / experiment.control_hz,
                        parameters={
                            name: value[index : index + 1].clone()
                            for name, value in m.parameters.items()
                        },
                    ),
                    neural_model=adapter,
                )
                controller.reset(
                    torch.ones(1, device=device, dtype=torch.bool)
                )
                episode_observer = FirstOrderServoObserver(
                    m.parameters["servos.pwm_angle_table"],
                    m.parameters["servos.tau"],
                    1.0 / experiment.control_hz,
                )
                episode_pilot = VirtualPilotHeightController(
                    pilot_config,
                    1,
                    device,
                    torch.float32,
                    1.0 / experiment.control_hz,
                )
                episode_pilot.reset(
                    torch.ones(1, device=device, dtype=torch.bool)
                )
                episode_initial = _expand_mapping(m.initial_state, 1)
                episode_initial["attitude_q_wb"].copy_(
                    attitude[index : index + 1].to(device)
                )
                episode_initial["angular_velocity_b"].copy_(
                    angular_velocity[index : index + 1].to(device)
                )
                env = SimulationEnvironment(
                    replace(
                        m,
                        parameters=one_params,
                        initial_state=episode_initial,
                    ),
                    1,
                    device,
                    torch.float32,
                    logging_enabled=False,
                )
                episode = _simulate_neural_episode(
                    argparse.Namespace(duration_s=args.duration_s),
                    experiment,
                    env,
                    controller,
                    episode_pilot,
                    episode_observer,
                    device,
                    torch.float32,
                )
                for key in collected:
                    collected[key].append(episode[key])
            metrics = {
                key: (
                    torch.stack(values).squeeze(-1).cpu().numpy().tolist()
                    if values
                    else []
                )
                for key, values in collected.items()
            }
            target.write_text(
                json.dumps(
                    {"cell": cell["name"], "scheme": scheme, "metrics": metrics},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(cell["name"], scheme, "done")
    return {"cells": [cell["name"] for cell in cells]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native-vehicle robustness benchmark for neural policies"
    )
    parser.add_argument("--simulator-config", required=True)
    parser.add_argument("--e2e-bundle", required=True)
    parser.add_argument("--aa-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-episodes", type=int, default=8)
    parser.add_argument("--maximum-initial-tilt-rad", type=float, default=0.2617994)
    parser.add_argument("--maximum-initial-rate-rad-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
