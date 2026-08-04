from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


@torch.no_grad()
def validate(args: argparse.Namespace) -> dict[str, object]:
    from flight_controller.math import lookup, tilt_cosine
    from simenv import SimulationEnvironment

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    dtype = torch.float32
    point_reports = []
    for point in manifest["points"]:
        environment = SimulationEnvironment.create(
            point["simulator_config"], args.parallel_count, device, dtype
        )
        try:
            command = torch.tensor(
                point["allocated_hover"]["simulator_command"],
                device=device,
                dtype=dtype,
            )[None].expand(args.parallel_count, -1).clone()
            motor_speed = lookup(
                command[:, :2], environment.parameters["motors.pwm_to_rpm_table"]
            )
            servo_angle = lookup(
                command[:, 2:], environment.parameters["servos.pwm_angle_table"]
            )
            state = environment.state_dict()
            state["truth"]["motor_speed"].copy_(motor_speed)
            state["truth"]["effective_motor_speed"].copy_(motor_speed)
            state["truth"]["servo_angle"].copy_(servo_angle)
            state["truth"]["servo_effective_pwm"].copy_(command[:, 2:])
            state["truth"]["servo_command_angle"].copy_(servo_angle)
            state["truth"]["servo_target_angle"].copy_(servo_angle)
            state["truth"]["servo_motion_direction"].zero_()
            state["truth"]["servo_backlash_remaining"].zero_()
            state["control"].copy_(command)
            environment.load_state_dict(state)
            active = torch.ones(args.parallel_count, device=device, dtype=torch.bool)
            maximum_tilt = torch.zeros(args.parallel_count, device=device, dtype=dtype)
            maximum_rate = torch.zeros_like(maximum_tilt)
            for _ in range(round(args.duration_s * 500.0)):
                result = environment.advance(command, active)
                active &= result.valid
                truth = environment.observe("truth").values
                tilt = torch.acos(tilt_cosine(truth["attitude_q_wb"]).clamp(-1.0, 1.0))
                rate = truth["angular_velocity_b"].norm(dim=1)
                maximum_tilt = torch.maximum(maximum_tilt, tilt)
                maximum_rate = torch.maximum(maximum_rate, rate)
            truth = environment.observe("truth").values
            report = {
                "group_id": int(point["group_id"]),
                "valid_fraction": float(active.to(torch.float32).mean()),
                "maximum_tilt_rad_p95": float(torch.quantile(maximum_tilt, 0.95)),
                "maximum_rate_rad_s_p95": float(torch.quantile(maximum_rate, 0.95)),
                "final_vertical_velocity_m_s_p95_abs": float(
                    torch.quantile(truth["velocity_n"][:, 2].abs(), 0.95)
                ),
                "final_horizontal_velocity_m_s_p95": float(
                    torch.quantile(truth["velocity_n"][:, :2].norm(dim=1), 0.95)
                ),
                "final_horizontal_position_m_p95": float(
                    torch.quantile(truth["position_n"][:, :2].norm(dim=1), 0.95)
                ),
            }
            report["passed"] = bool(
                report["valid_fraction"] == 1.0
                and report["maximum_tilt_rad_p95"] <= 0.02
                and report["maximum_rate_rad_s_p95"] <= 0.05
                and report["final_vertical_velocity_m_s_p95_abs"] <= 0.10
            )
            point_reports.append(report)
        finally:
            environment.close()
    result = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "device": str(device),
        "parallel_count": args.parallel_count,
        "duration_s": args.duration_s,
        "points": point_reports,
        "all_passed": all(point["passed"] for point in point_reports),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parallel-count", type=int, default=64)
    parser.add_argument("--duration-s", type=float, default=2.0)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
