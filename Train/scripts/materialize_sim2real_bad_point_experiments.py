from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _cyclic_scale(trim: list[float], requested: float = 0.40) -> float:
    coefficients = (1.0, 0.5 + 0.5 * 3.0**0.5, 0.5 + 0.5 * 3.0**0.5)
    factor = min(
        (1.0 - abs(center)) / (requested * coefficient)
        for center, coefficient in zip(trim, coefficients, strict=True)
    )
    if factor <= 0.0:
        raise ValueError("servo trim has no cyclic control margin")
    return requested * min(1.0, 0.90 * factor)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve()
    point_manifest = Path(args.point_manifest).expanduser().resolve()
    output_directory = Path(args.output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(source.read_text(encoding="utf-8"))
    points = json.loads(point_manifest.read_text(encoding="utf-8"))["points"]
    rows = []
    acceleration_limits = [4.0, 4.0, 1.5]
    for point_index, point in enumerate(points):
        group_id = int(point["group_id"])
        hover = point["allocated_hover"]
        upper = float(hover["upper_motor_pwm"])
        lower = float(hover["lower_motor_pwm"])
        ratio = lower / upper
        servo_trim = [float(value) for value in hover["servo_trim_command"]]
        cyclic_scale = _cyclic_scale(servo_trim)
        motor_scale = min(0.12, 0.8 * lower, 0.8 * (1.0 - lower))
        throttle_maximum = min(0.85, 0.98 * (1.0 - motor_scale) / ratio)
        if throttle_maximum <= upper + 0.04:
            raise ValueError(f"group {group_id} has insufficient collective margin")
        throttle_minimum = max(0.10, upper - 0.15)
        target_half_width = min(0.015, upper - throttle_minimum, throttle_maximum - upper)
        target_range = [upper - target_half_width, upper + target_half_width]

        config = json.loads(json.dumps(base))
        config["experiment"]["name"] = (
            f"mlp_sac_angular_acceleration_bad_point_{group_id}_v1"
        )
        config["seed"]["base"] = 20260840 + point_index
        config["run"]["total_control_steps"] = 4_194_304
        config["run"]["output_root"] = "../../../runs"
        config["environment"]["config_path"] = (
            f"../../../../SimEnv/configs/sim2real_bad_points/group_{group_id}.yaml"
        )
        throttle = config["command_source"]["params"]["throttle"]
        throttle["minimum"] = throttle_minimum
        throttle["maximum"] = throttle_maximum
        throttle["spool"]["target_range"] = target_range
        throttle["height_controller"]["initial_throttle_range"] = target_range
        config["command_source"]["params"]["direct_angular_acceleration"][
            "limits_rad_s2"
        ] = acceleration_limits
        transform = config["control_contract"]["action_transform"]
        transform["lower_motor_upper_ratio"] = ratio
        transform["trim_command"] = [upper, *servo_trim]
        transform["residual_scale"] = [motor_scale, cyclic_scale, cyclic_scale]
        config["task"]["outer_loop"]["max_angular_acceleration_rad_s2"] = (
            acceleration_limits
        )
        distribution = config["model"]["policy_distribution"]
        distribution["initial_action_std"] = [0.025, 0.020, 0.020]
        distribution["minimum_action_std"] = [0.005, 0.005, 0.005]
        distribution["maximum_action_std"] = [0.050, 0.040, 0.040]
        config["algorithm"]["policy_anchor"] = {
            "weight": 5.0,
            "max_action_deviation": 0.05,
        }
        config["algorithm"]["critic_pretraining_updates"] = 1024
        config["algorithm"]["optimizer"]["actor_learning_rate"] = 0.000002
        config["checkpoint"]["resume"] = {
            "from": (
                "../../../runs/sim2real_bad_point_teacher_bootstrap/"
                f"group_{group_id}_v1.pt"
            ),
            "mode": "policy",
        }
        config["checkpoint"]["keep_last"] = 4
        output = output_directory / (
            f"mlp_sac_angular_acceleration_bad_point_{group_id}_v1.yaml"
        )
        output.write_text(
            yaml.dump(
                config,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "group_id": group_id,
                "experiment_config": str(output),
                "upper_hover_pwm": upper,
                "lower_hover_pwm": lower,
                "lower_motor_upper_ratio": ratio,
                "servo_trim_command": servo_trim,
                "motor_residual_scale": motor_scale,
                "cyclic_residual_scale": cyclic_scale,
                "throttle_range": [throttle_minimum, throttle_maximum],
                "acceleration_limits_rad_s2": acceleration_limits,
            }
        )
    report = {
        "schema_version": 1,
        "source_experiment": str(source),
        "source_point_manifest": str(point_manifest),
        "experiments": rows,
    }
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--point-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--output-manifest", required=True)
    materialize(parser.parse_args())


if __name__ == "__main__":
    main()
