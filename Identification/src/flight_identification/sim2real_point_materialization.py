from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from scipy.optimize import least_squares

from .experiment import _apply_effectiveness_labels


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _python(value: torch.Tensor) -> Any:
    result = value.detach().cpu().to(torch.float64).tolist()
    return result


def _set_materialized_values(
    raw: dict[str, Any], parameters: Mapping[str, torch.Tensor], group_id: int
) -> None:
    raw["seed"] = 20260830 + int(group_id)
    raw["body"]["mass"]["value"] = float(parameters["body.mass"][0])
    raw["body"]["inertia_diagonal_b"]["value"] = _python(
        parameters["body.inertia_diagonal_b"][0]
    )
    for index, motor in enumerate(raw["motors"]):
        motor["time_constant"]["value"] = float(
            parameters["motors.time_constant"][0, index]
        )
        motor["torque_coefficient"]["value"] = float(
            parameters["motors.torque_coefficient"][0, index]
        )
    for index, servo in enumerate(raw["servos"]):
        servo["tau"]["value"] = float(parameters["servos.tau"][0, index])
        servo["max_speed"]["value"] = float(
            parameters["servos.max_speed"][0, index]
        )
    aero = raw["aerodynamics"]
    aero["thrust_coefficients"]["value"] = _python(
        parameters["aerodynamics.thrust_coefficients"][0]
    )
    aero["direct_thrust_center_b"]["value"] = _python(
        parameters["aerodynamics.direct_thrust_center_b"][0]
    )
    partition = parameters["aerodynamics.thrust_partition"][0]
    for index, name in enumerate(("direct", "grid_1", "grid_2", "grid_3")):
        aero["thrust_partition"][name]["value"] = float(partition[index])
    centers = parameters["aerodynamics.grids.aerodynamic_center_b"][0]
    gains = parameters["aerodynamics.grids.vector_deflection.gain"][0]
    for index, grid in enumerate(aero["grids"]):
        grid["aerodynamic_center_b"]["value"] = _python(centers[index])
        grid["vector_deflection"]["gain"]["value"] = float(gains[index])
    aero["coupling_attenuation"]["value"] = _python(
        parameters["aerodynamics.coupling_attenuation"][0]
    )
    raw["logging"]["directory"] = (
        f"../logs/sim2real_bad_points/group_{group_id}"
    )


def _allocated_hover(parameters: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    from flight_controller.math import lookup
    from flight_controller.plant import LocalPlantModel

    cpu_parameters = {
        name: value.detach().cpu().to(torch.float64) for name, value in parameters.items()
    }
    plant = LocalPlantModel(cpu_parameters)
    unconstrained = plant.hover_trim().command[0]
    mass = float(cpu_parameters["body.mass"][0])
    inertia = cpu_parameters["body.inertia_diagonal_b"][0].numpy()
    weight = mass * plant.gravity
    moment_scale = weight * 0.05

    def force_and_moment(command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = cpu_parameters
        motor_speed = lookup(command[:, :2], p["motors.pwm_to_rpm_table"])
        servo_angle = lookup(command[:, 2:], p["servos.pwm_angle_table"])
        coefficients = p["aerodynamics.thrust_coefficients"]
        total_thrust = (
            coefficients[:, 0] * motor_speed[:, 0].square()
            + coefficients[:, 1] * motor_speed[:, 1].square()
            + coefficients[:, 2] * motor_speed[:, 0] * motor_speed[:, 1]
        )
        neutral = p["aerodynamics.neutral_thrust_direction_b"]
        partition = p["aerodynamics.thrust_partition"]
        direct_force = total_thrust[:, None] * partition[:, :1] * neutral
        direct_arm = p["aerodynamics.direct_thrust_center_b"] - p["body.center_of_mass_b"]
        direct_moment = torch.linalg.cross(direct_arm, direct_force)
        attenuation = lookup(
            servo_angle.abs(), p["aerodynamics.grids.self_attenuation_curve"]
        )
        coupling_loss = torch.bmm(
            p["aerodynamics.coupling_attenuation"],
            (1.0 - attenuation).unsqueeze(-1),
        ).squeeze(-1)
        effective = (attenuation - coupling_loss).clamp(0.0, 1.0)
        vector_angle = (
            p["aerodynamics.grids.vector_deflection.gain"] * servo_angle
            + p["aerodynamics.grids.vector_deflection.offset"]
        )
        axes = p["aerodynamics.grids.deflection_axis_b"]
        base = neutral[:, None, :].expand(-1, 3, -1)
        cosine = torch.cos(vector_angle)[..., None]
        sine = torch.sin(vector_angle)[..., None]
        projection = (axes * base).sum(dim=-1, keepdim=True)
        direction = (
            base * cosine
            + torch.linalg.cross(axes, base) * sine
            + axes * projection * (1.0 - cosine)
        )
        grid_thrust = total_thrust[:, None] * partition[:, 1:] * effective
        grid_force = grid_thrust[..., None] * direction
        grid_arm = (
            p["aerodynamics.grids.aerodynamic_center_b"]
            - p["body.center_of_mass_b"][:, None, :]
        )
        grid_moment = torch.linalg.cross(grid_arm, grid_force).sum(dim=1)
        motor_torque = p["motors.torque_coefficient"] * motor_speed.square()
        reaction = (motor_torque[:, 0] - motor_torque[:, 1])[:, None] * (-neutral)
        return direct_force + grid_force.sum(dim=1), direct_moment + grid_moment + reaction

    def command_from_variables(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value)[None]

    regularization_center = unconstrained.numpy().copy()
    regularization_scale = np.array([0.20, 0.20, 0.35, 0.35, 0.35])

    def residual(value: np.ndarray) -> np.ndarray:
        command = command_from_variables(value)
        force, moment = force_and_moment(command)
        physical = np.concatenate(
            (((-float(force[0, 2]) - weight) / weight,), moment[0].numpy() / moment_scale)
        )
        regularization = 1e-5 * (
            (value - regularization_center) / regularization_scale
        )
        return np.concatenate((physical, regularization))

    zero_command = torch.cat((unconstrained[:2], torch.zeros(3, dtype=torch.float64)))
    bias_moment = plant.steady_wrench(zero_command[None])[0, 1:]
    servo_effectiveness = plant.control_effectiveness(zero_command[None])[0, 1:, 2:]
    servo_initial = torch.linalg.lstsq(
        servo_effectiveness, -bias_moment[:, None]
    ).solution[:, 0].numpy()
    initial = np.concatenate((unconstrained[:2].numpy(), servo_initial))
    solution = least_squares(
        residual,
        initial,
        bounds=(
            np.array([0.08, 0.08, -0.90, -0.90, -0.90]),
            np.array([1.00, 1.00, 0.90, 0.90, 0.90]),
        ),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=2000,
    )
    command = command_from_variables(solution.x)[0]
    if float(command[2:].abs().max()) >= 0.95:
        raise ValueError("allocated hover requires excessive servo command")
    force, moment = force_and_moment(command[None])
    acceleration_residual = torch.cat(
        (
            ((-force[0, 2] - weight) / mass)[None],
            moment[0] / torch.from_numpy(inertia),
        )
    )
    if float(acceleration_residual.abs().max()) > 1e-3:
        raise ValueError(
            "could not solve allocated hover; maximum acceleration residual is "
            f"{float(acceleration_residual.abs().max()):.6g}; "
            f"command={command.tolist()}; residual={acceleration_residual.tolist()}; "
            f"optimizer_status={solution.status}:{solution.message}"
        )
    upper = float(command[0])
    lower = float(command[1])
    return {
        "simulator_command": command.tolist(),
        "upper_motor_pwm": upper,
        "lower_motor_pwm": lower,
        "lower_motor_upper_ratio": lower / upper,
        "servo_trim_command": command[2:].tolist(),
        "acceleration_residual": acceleration_residual.tolist(),
        "lateral_acceleration_b_m_s2": (force[0, :2] / mass).tolist(),
        "optimizer_cost": float(solution.cost),
        "optimizer_optimality": float(solution.optimality),
    }


def materialize(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset = Path(args.dataset).expanduser().resolve()
    source_config = Path(args.source_config).expanduser().resolve()
    output_directory = Path(args.output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = torch.load(
        dataset / "parameter_groups.pt", map_location="cpu", weights_only=False
    )
    if payload.get("parameterization") != "sim2real_micro":
        raise ValueError("point materialization requires sim2real_micro labels")
    requested = tuple(dict.fromkeys(args.group_ids))
    group_ids = payload["group_id"]
    rows = []

    from simenv.config import load_and_materialize

    nominal = load_and_materialize(
        source_config, 1, torch.device("cpu"), torch.float64
    ).parameters
    for group_id in requested:
        matches = torch.nonzero(group_ids == group_id, as_tuple=False).flatten()
        if len(matches) != 1:
            raise ValueError(f"group ID {group_id} does not resolve uniquely")
        index = int(matches[0])
        labels = payload["labels"][index : index + 1].to(torch.float64)
        actual = _apply_effectiveness_labels(nominal, labels, "sim2real_micro")
        # YAML merge keys share nested mapping objects after safe_load. Break
        # those aliases before assigning per-actuator values.
        raw = json.loads(
            json.dumps(yaml.safe_load(source_config.read_text(encoding="utf-8")))
        )
        _set_materialized_values(raw, actual, group_id)
        output = output_directory / f"group_{group_id}.yaml"
        output.write_text(
            yaml.dump(
                raw,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
        round_trip = load_and_materialize(
            output, 1, torch.device("cpu"), torch.float64
        ).parameters
        mismatches = []
        for name, expected in actual.items():
            if not torch.allclose(round_trip[name], expected, rtol=1e-12, atol=1e-14):
                mismatches.append(name)
        if mismatches:
            raise RuntimeError(f"materialized config mismatch: {mismatches}")
        rows.append(
            {
                "group_id": group_id,
                "simulator_config": str(output),
                "label_names": list(payload["label_names"]),
                "labels": labels[0].tolist(),
                "allocated_hover": _allocated_hover(actual),
            }
        )
    report = {
        "schema_version": 1,
        "source_dataset": str(dataset),
        "source_simulator_config": str(source_config),
        "points": rows,
    }
    manifest = Path(args.manifest).expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize exact SimEnv configs for selected sim2real groups"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--group-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--manifest", required=True)
    return parser


def main() -> None:
    materialize(build_parser().parse_args())


if __name__ == "__main__":
    main()
