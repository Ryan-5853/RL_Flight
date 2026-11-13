#!/usr/bin/env python3
"""Generate the airframe-1 coaxial collective/servo LQI firmware payload."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_CONFIG = (
    ROOT
    / "px4_trans/configs/lqi_coaxial_collective_airframe1_manual_v3.yaml"
)
PAYLOAD_HEADER = (
    ROOT
    / "px4_trans/px4/src/modules/nn_control/LqiCoaxialCollectivePayload.hpp"
)
GOLDEN_HEADER = (
    ROOT / "px4_trans/tests/lqi_coaxial_collective_golden.hpp"
)
GOLDEN_JSON = (
    ROOT / "px4_trans/tests/lqi_coaxial_collective_golden.json"
)
DT_S = 0.002
RPM_TO_RAD_S = 2.0 * math.pi / 60.0


def f32(value: float | np.ndarray) -> np.float32 | np.ndarray:
    return np.asarray(value, dtype=np.float32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv1a32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def literal(value: float) -> str:
    rendered = f"{float(value):.9g}"
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "f"


def cpp_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def array_lines(name: str, values: np.ndarray) -> list[str]:
    result = [f"static constexpr float {name}[{values.size}] = {{"]
    result.extend(f"\t{literal(value)}," for value in values.ravel())
    result.append("};")
    return result


def load_calibration(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("upper-rotor calibration CSV is empty")
    parsed = []
    for row in rows:
        if row["quality"] != "正常" or row["mode"] != "HARDWARE":
            raise ValueError(f"rejected motor calibration row: {row}")
        if int(row["motor"]) != 1:
            raise ValueError("upper-rotor CSV must use motor=1")
        parsed.append(
            {
                "physical_command": float(row["command_normalized"]),
                "rpm": float(row["rpm_mean"]),
                "rad_s": float(row["rpm_mean"]) * RPM_TO_RAD_S,
                "samples": int(row["samples"]),
            }
        )
    return parsed


def interpolate_linear(
    point: float, points: np.ndarray, values: np.ndarray
) -> float:
    upper = np.searchsorted(points, point, side="right")
    upper = int(np.clip(upper, 1, len(points) - 1))
    lower = upper - 1
    fraction = (point - float(points[lower])) / (
        float(points[upper]) - float(points[lower])
    )
    return float(values[lower] + fraction * (values[upper] - values[lower]))


def load_model(
    deployment_config: Path,
) -> tuple[
    dict[str, np.ndarray | float | int], dict[str, Any], dict[str, Any]
]:
    import torch

    from flight_controller import (
        ControllerContext,
        create_controller,
        load_controller_config,
    )
    from simenv.config import load_and_materialize

    deployment = yaml.safe_load(
        deployment_config.read_text(encoding="utf-8")
    )
    if deployment.get("airframe_id") != "airframe-1":
        raise ValueError("coaxial collective firmware requires airframe-1")
    simulator_path = ROOT / deployment["simulator_config"]
    controller_path = ROOT / deployment["controller_config"]
    materialized = load_and_materialize(
        simulator_path, 1, torch.device("cpu"), torch.float64
    )
    controller_config = load_controller_config(controller_path)
    controller = create_controller(
        controller_config,
        ControllerContext(
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float64,
            control_dt=DT_S,
            parameters=materialized.parameters,
        ),
    )
    if controller.controller_type != "lqi_coaxial_collective_servo":
        raise ValueError("deployment controller type is not the new LQI")

    calibration_path = ROOT / deployment["motor_speed_calibration"]["path"]
    calibration = load_calibration(calibration_path)
    tables = materialized.parameters[
        "motors.pwm_to_rpm_table"
    ][0].detach().cpu().numpy().astype(np.float32)
    upper_table = tables[0]
    lower_table = tables[1]
    if not np.array_equal(upper_table, lower_table):
        raise ValueError(
            "airframe-1 policy requires both coaxial rotors to use the same "
            "measured upper-rotor in-air table"
        )
    for expected, table in zip(calibration, upper_table[2:7]):
        controller_command = (
            2.0 * expected["physical_command"]
        )
        if not math.isclose(
            float(table[0]), controller_command, abs_tol=1e-7
        ):
            raise ValueError(
                "upper table command points do not match CSV physical x2 mapping"
            )
        if not math.isclose(
            float(table[1]), expected["rad_s"], rel_tol=2e-6
        ):
            raise ValueError(
                "upper table speed points do not match CSV RPM conversion"
            )
    if not math.isclose(
        float(upper_table[-1, 1]), 1200.0, abs_tol=1e-6
    ):
        raise ValueError("upper extrapolation must terminate at prior 1200 rad/s")

    servo_jacobian = controller.hover_effectiveness[
        :, :, 2:
    ].detach().cpu().numpy()[0].astype(np.float64)
    normal = (
        servo_jacobian.T @ servo_jacobian
        + controller.servo_regularization * np.eye(3, dtype=np.float64)
    )
    desired = np.zeros((4, 2), dtype=np.float64)
    desired[1, 0] = 1.0
    desired[2, 1] = 1.0
    servo_direction = np.linalg.solve(
        normal, servo_jacobian.T @ desired
    ).astype(np.float32)

    manual = deployment["manual_reference"]
    command_tau = np.asarray(
        [
            manual["roll"]["time_constant_s"],
            manual["pitch"]["time_constant_s"],
            manual["yaw"]["time_constant_s"],
        ],
        dtype=np.float32,
    )
    if np.any(command_tau < 0.0):
        raise ValueError("manual reference time constants must be non-negative")
    positive_tau = command_tau > 0.0
    command_decay = np.zeros_like(command_tau)
    command_decay[positive_tau] = np.exp(
        -DT_S / command_tau[positive_tau]
    )
    model: dict[str, np.ndarray | float | int] = {
        "attitude_gain": controller._lqi_gain.detach().cpu().numpy()
        .astype(np.float32),
        "servo_direction": servo_direction,
        "maximum_moment": controller.maximum_moment.detach().cpu().numpy()
        .astype(np.float32),
        "motor_effectiveness_points": controller.motor_effectiveness_points
        .detach().cpu().numpy().astype(np.float32),
        "motor_effectiveness": controller.motor_effectiveness_grid.detach()
        .cpu().numpy()[0].reshape(-1, 4).astype(np.float32),
        "upper_table": tables[0],
        "lower_table": tables[1],
        "mass": float(controller.context.parameters["body.mass"][0]),
        "gravity": float(controller.plant.gravity),
        "thrust_coefficients": controller.context.parameters[
            "aerodynamics.thrust_coefficients"
        ][0].detach().cpu().numpy().astype(np.float32),
        "motor_torque_coefficients": controller.context.parameters[
            "motors.torque_coefficient"
        ][0].detach().cpu().numpy().astype(np.float32),
        "maximum_thrust": float(controller.maximum_thrust[0]),
        "minimum_thrust_fraction": controller.minimum_thrust_fraction,
        "maximum_thrust_fraction": controller.maximum_thrust_fraction,
        "tilt_cosine_minimum": 0.35,
        "pilot_minimum": controller.pilot_minimum,
        "pilot_maximum": controller.pilot_maximum,
        "maximum_vertical_acceleration": controller.maximum_acceleration,
        "acceleration_kp": controller.acceleration_kp,
        "acceleration_ki": controller.acceleration_ki,
        "acceleration_integral_limit": (
            controller.acceleration_integral_limit
        ),
        "acceleration_filter_time_constant": (
            controller.acceleration_filter_time_constant
        ),
        "servo_effectiveness_scale": controller.servo_effectiveness_scale,
        "servo_scale_minimum": 0.05,
        "servo_scale_maximum": 1.5,
        "motor_mix_regularization": controller.motor_regularization,
        "zero_throttle_threshold": controller.zero_throttle_threshold,
        "dt": DT_S,
        "max_roll_angle": float(manual["roll"]["limit_rad"]),
        "max_pitch_angle": float(manual["pitch"]["limit_rad"]),
        "max_yaw_rate": float(manual["yaw"]["limit_rad_s"]),
        "manual_command_decay": command_decay.astype(np.float32),
    }
    metadata = {
        "deployment_config": str(deployment_config.relative_to(ROOT)),
        "deployment_sha256": file_sha256(deployment_config),
        "simulator_config": str(simulator_path.relative_to(ROOT)),
        "simulator_sha256": file_sha256(simulator_path),
        "controller_config": str(controller_path.relative_to(ROOT)),
        "controller_sha256": file_sha256(controller_path),
        "calibration_path": str(calibration_path.relative_to(ROOT)),
        "calibration_sha256": file_sha256(calibration_path),
        "calibration_rows": calibration,
        "identification_report": deployment["identified_model"]["report"],
        "identification_report_sha256": file_sha256(
            ROOT / deployment["identified_model"]["report"]
        ),
        "identified_model_checksum": deployment["identified_model"][
            "expected_checksum"
        ],
        "trim_upper_command": float(controller.trim.command[0, 0]),
        "trim_lower_command": float(controller.trim.command[0, 1]),
        "trim_motor_speed_rad_s": controller.trim.motor_speed[0]
        .detach()
        .cpu()
        .tolist(),
        "lqi_pole_radius": float(
            np.max(np.abs(controller._lqi_poles))
        ),
    }
    return model, deployment, metadata


YAW_BALANCE_FIELDS = (
    "trim_speed_ratio", "maximum_speed_ratio", "motor_command_maximum",
    "grid_common_command", "grid_yaw_per_thrust_per_command",
    "yaw_integral_moment_limit_n_m", "lower_torque_scale",
)


def apply_yaw_balance_profile(model, deployment, metadata):
    profile = deployment.get("yaw_balance_profile")
    if profile is None:
        if deployment.get("allocation_safety") is not None:
            raise ValueError("allocation_safety requires yaw_balance_profile")
        return
    if set(profile) != set(YAW_BALANCE_FIELDS):
        raise ValueError("yaw_balance_profile must contain exactly the documented fields")
    policy = np.asarray([profile[key] for key in YAW_BALANCE_FIELDS], dtype=np.float32)
    ratio, maximum, cap, common, slope, limit, torque_scale = policy
    if (not np.all(np.isfinite(policy)) or not 1 <= ratio <= maximum <= 1.5
        or not 0 < cap <= 1 or abs(common) > .25 or slope <= 0
        or not 0 < limit <= .01 or not 0 < torque_scale <= 1):
        raise ValueError("invalid conservative yaw balance profile")
    model["yaw_balance"] = policy
    safety = deployment.get("allocation_safety")
    if safety is not None:
        expected = {"roll_pitch_integral_moment_fraction", "grid_common_preference_fraction"}
        if set(safety) != expected:
            raise ValueError("allocation_safety must contain the documented two fractions")
        rp = float(safety["roll_pitch_integral_moment_fraction"])
        grid = float(safety["grid_common_preference_fraction"])
        if not math.isfinite(rp) or not 0 < rp <= 1 or not math.isfinite(grid) or not 0 <= grid <= 1:
            raise ValueError("invalid allocation safety fractions")
        model.update(roll_pitch_integral_moment_fraction=rp, grid_common_preference_fraction=grid, allocation_revision=2)
        metadata["allocation_safety"] = dict(safety)
    elif model.get("allocation_revision") == 2:
        # Only explicit model-reuse mode permits this downgrade; reference-only
        # callers reject a changed allocation_safety before reaching here.
        for name in ("allocation_revision", "roll_pitch_integral_moment_fraction", "grid_common_preference_fraction"):
            model.pop(name, None)
        metadata.pop("allocation_safety", None)
    coefficients = model["thrust_coefficients"]
    denominator = coefficients[0] + coefficients[1] * ratio**2 + coefficients[2] * ratio
    upper_speed = math.sqrt(model["mass"] * model["gravity"] / denominator)
    speeds = [float(upper_speed), float(upper_speed * ratio)]
    commands = [float(np.interp(speed, model[key][:, 1], model[key][:, 0])) for speed, key in zip(speeds, ("upper_table", "lower_table"))]
    metadata["yaw_balance_profile"] = {key: float(value) for key, value in zip(YAW_BALANCE_FIELDS, policy)}
    metadata["policy_hover_trim_controller_commands"] = commands
    metadata["policy_hover_trim_physical_commands"] = [value * .5 for value in commands]
    metadata["policy_hover_trim_motor_speed_rad_s"] = speeds
    metadata["policy_scope"] = "Firmware deployment overlay; inherited plant/gains unchanged; lower torque scale is an experimental assumption. Ratios constrain predicted steady speed, not measured ESC speed."


def apply_attitude_response(model, deployment, metadata):
    """Optional moment-domain PID retune; no change to yaw or plant tables."""
    response = deployment.get("attitude_response")
    if response is None:
        if metadata.get("attitude_response") is not None:
            raise ValueError("removing a gain override requires reusing the original gain manifest")
        return
    names = ("roll_pitch_kp", "roll_pitch_kd", "roll_pitch_ki")
    if set(response) != set(names):
        raise ValueError("attitude_response requires exactly roll_pitch_kp/kd/ki")
    kp, kd, ki = (float(response[name]) for name in names)
    if not all(math.isfinite(value) and value > 0 for value in (kp, kd, ki)):
        raise ValueError("attitude response gains must be finite and positive")
    gain = np.asarray(model["attitude_gain"], dtype=np.float32).copy()
    for axis in range(2):
        gain[axis] = 0.0
        gain[axis, axis] = kp
        gain[axis, 2 + axis] = kd
        gain[axis, 5 + axis] = ki
    model["attitude_gain"] = gain
    metadata["attitude_response"] = dict(response)
    if "lqi_pole_radius" in metadata:
        metadata["inherited_lqi_pole_radius"] = metadata.pop("lqi_pole_radius")
    metadata["gain_design_scope"] = "Roll/pitch moment-domain PID retune with 50 Hz actuator ZOH/lag stability screen; original yaw row and plant retained. Not hardware validated."


def load_reference_update(deployment_config: Path, manifest_path: Path):
    """Preserve the verified plant/gains when only manual limits change.

    This path does not need SimEnv or Torch. Check the old model checksum and
    all plant source hashes before updating the reference and provenance.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = dict(manifest["source"])
    deployment = yaml.safe_load(deployment_config.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("simulator_config", "simulator_sha256"),
        ("controller_config", "controller_sha256"),
        ("calibration_path", "calibration_sha256"),
        ("identification_report", "identification_report_sha256"),
    ):
        if file_sha256(ROOT / metadata[path_key]) != metadata[hash_key]:
            raise ValueError(f"reference-only update cannot change {path_key}")
    if deployment["simulator_config"] != metadata["simulator_config"] or deployment["controller_config"] != metadata["controller_config"]:
        raise ValueError("reference-only update cannot change plant/controller paths")
    header = (ROOT / manifest["generated"]["candidate_payload"]).read_text()
    if file_sha256(ROOT / manifest["generated"]["candidate_payload"]) != manifest["generated"]["candidate_payload_sha256"]:
        raise ValueError("candidate payload hash mismatch")
    arrays = {
        name: np.array([float(v.strip().rstrip("f")) for v in values.split(",") if v.strip()], dtype=np.float32)
        for name, values in re.findall(r"static constexpr float (\w+)\[\d+\] = \{(.*?)\};", header, re.S)
    }
    fields = re.search(r"static constexpr LqiCoaxialCollectiveModel kModel \{(.*?)\n\};", header, re.S).group(1)
    tokens = [line.strip().rstrip(",") for line in fields.splitlines() if line.strip()][5:]
    names = (
        "attitude_gain", "servo_direction", "maximum_moment", "motor_effectiveness_points", "motor_effectiveness", "points_count",
        "upper_table", "lower_table", "upper_count", "lower_count", "mass", "gravity", "thrust_coefficients", "motor_torque_coefficients",
        "maximum_thrust", "minimum_thrust_fraction", "maximum_thrust_fraction", "tilt_cosine_minimum", "pilot_minimum", "pilot_maximum",
        "maximum_vertical_acceleration", "acceleration_kp", "acceleration_ki", "acceleration_integral_limit", "acceleration_filter_time_constant",
        "servo_effectiveness_scale", "servo_scale_minimum", "servo_scale_maximum", "motor_mix_regularization", "zero_throttle_threshold",
        "dt", "max_roll_angle", "max_pitch_angle", "max_yaw_rate", "manual_command_decay", "weight_bytes", "checksum",
    )
    if len(tokens) == len(names) + 3:
        names = (*names, "yaw_balance", "roll_pitch_integral_moment_fraction", "grid_common_preference_fraction")
    elif len(tokens) == len(names) + 1:
        names = (*names, "yaw_balance")
    if len(tokens) != len(names):
        raise ValueError("unsupported embedded model layout")
    model = {}
    for name, token in zip(names, tokens):
        if token in arrays:
            model[name] = arrays[token]
        elif token.startswith("{"):
            model[name] = np.array([float(v.strip().rstrip("f")) for v in token.strip("{}").split(",")], dtype=np.float32)
        elif token.startswith("0x"):
            model[name] = int(token.rstrip("u"), 16)
        else:
            model[name] = float(token.rstrip("fu"))
    if "PilotAccelerationSharedYawAntiWindup" in fields:
        model["allocation_revision"] = 2
    for name, width in (("attitude_gain", 8), ("servo_direction", 2), ("motor_effectiveness", 4), ("upper_table", 2), ("lower_table", 2)):
        model[name] = model[name].reshape(-1, width)
    if checksum(model) != int(manifest["model_checksum"], 16) or model["checksum"] != checksum(model):
        raise ValueError("embedded model checksum mismatch")
    manual = deployment["manual_reference"]
    model["max_roll_angle"] = float(manual["roll"]["limit_rad"])
    model["max_pitch_angle"] = float(manual["pitch"]["limit_rad"])
    model["max_yaw_rate"] = float(manual["yaw"]["limit_rad_s"])
    tau = np.asarray([manual[axis]["time_constant_s"] for axis in ("roll", "pitch", "yaw")], dtype=np.float32)
    if not np.all(np.isfinite(tau)) or np.any(tau < 0):
        raise ValueError("manual reference time constants must be finite and non-negative")
    decay = np.zeros_like(tau)
    decay[tau > 0] = np.exp(-model["dt"] / tau[tau > 0])
    model["manual_command_decay"] = decay
    metadata["deployment_config"] = str(deployment_config.relative_to(ROOT))
    metadata["deployment_sha256"] = file_sha256(deployment_config)
    return model, deployment, metadata


def checksum(model: Mapping[str, Any]) -> int:
    names = (
        "attitude_gain",
        "servo_direction",
        "maximum_moment",
        "motor_effectiveness_points",
        "motor_effectiveness",
        "upper_table",
        "lower_table",
        "mass",
        "gravity",
        "thrust_coefficients",
        "motor_torque_coefficients",
        "maximum_thrust",
        "minimum_thrust_fraction",
        "maximum_thrust_fraction",
        "tilt_cosine_minimum",
        "pilot_minimum",
        "pilot_maximum",
        "maximum_vertical_acceleration",
        "acceleration_kp",
        "acceleration_ki",
        "acceleration_integral_limit",
        "acceleration_filter_time_constant",
        "servo_effectiveness_scale",
        "servo_scale_minimum",
        "servo_scale_maximum",
        "motor_mix_regularization",
        "zero_throttle_threshold",
        "dt",
        "max_roll_angle",
        "max_pitch_angle",
        "max_yaw_rate",
        "manual_command_decay",
    )
    if "yaw_balance" in model:
        names = (*names, "yaw_balance")
    if model.get("allocation_revision") == 2:
        # Include the behavioral contract as well as its two new scalars.
        names = (*names, "roll_pitch_integral_moment_fraction", "grid_common_preference_fraction", "allocation_revision")
    return fnv1a32(
        b"".join(
            np.asarray(model[name], dtype=np.float32).tobytes()
            for name in names
        )
    )


def write_model_header(
    model: Mapping[str, Any],
    metadata: Mapping[str, Any],
    payload_path: Path,
) -> int:
    digest = checksum(model)
    weight_names = (
        "attitude_gain",
        "servo_direction",
        "maximum_moment",
        "motor_effectiveness_points",
        "motor_effectiveness",
        "upper_table",
        "lower_table",
        "manual_command_decay",
    )
    if "yaw_balance" in model:
        weight_names = (*weight_names, "yaw_balance")
    if model.get("allocation_revision") == 2:
        weight_names = (*weight_names, "roll_pitch_integral_moment_fraction", "grid_common_preference_fraction")
    weight_bytes = (
        sum(np.asarray(model[name]).size for name in weight_names) * 4
    )
    lines = [
        "/****************************************************************************",
        " * Generated by px4_trans/tools/generate_coaxial_collective_backend.py.",
        " * Do not edit.",
        f" * Source CSV: {metadata['calibration_path']}",
        f" * Identification: {metadata['identification_report']}",
        " ****************************************************************************/",
        "",
        "#pragma once",
        "",
        '#include "LqiCoaxialCollectiveModel.hpp"',
        "",
        "namespace lqi_coaxial_collective",
        "{",
        "",
        'static constexpr uint32_t kAirframeNumber{1u};',
        'static constexpr char kAirframeId[]{"airframe-1"};',
        'static constexpr char kPlatformFamily[]{"micoair_h743-v2_micro_coaxial_3servo"};',
        f"static constexpr char kCalibrationSha256[]{{{cpp_string(metadata['calibration_sha256'])}}};",
        f"static constexpr char kIdentificationArtifactSha256[]{{{cpp_string(metadata['identification_report_sha256'])}}};",
        "",
    ]
    for name, cpp_name in (
        ("attitude_gain", "kAttitudeGain"),
        ("servo_direction", "kServoDirection"),
        ("maximum_moment", "kMaximumMoment"),
        ("motor_effectiveness_points", "kMotorEffectivenessPoints"),
        ("motor_effectiveness", "kMotorEffectiveness"),
        ("upper_table", "kUpperTable"),
        ("lower_table", "kLowerTable"),
        ("manual_command_decay", "kManualCommandDecay"),
    ):
        lines += array_lines(
            cpp_name, np.asarray(model[name], dtype=np.float32)
        )
        lines.append("")
    if "yaw_balance" in model:
        lines += array_lines("kYawBalance", np.asarray(model["yaw_balance"], dtype=np.float32))
        lines.append("")
    thrust = np.asarray(model["thrust_coefficients"], dtype=np.float32)
    torque = np.asarray(
        model["motor_torque_coefficients"], dtype=np.float32
    )
    lines += [
        "static constexpr LqiCoaxialCollectiveModel kModel {",
        "\tLqiCoaxialCollectiveModel::kSchemaVersion,",
        "\tLqiCoaxialCollectiveModel::WorldFrame::NED,",
        "\tLqiCoaxialCollectiveModel::BodyFrame::FRD,",
        "\tLqiCoaxialCollectiveModel::QuaternionConvention::HamiltonWxyzBodyToWorld,",
        ("\tLqiCoaxialCollectiveModel::CollectiveContract::PilotAccelerationSharedYawAntiWindup,"
         if model.get("allocation_revision") == 2 else "\tLqiCoaxialCollectiveModel::CollectiveContract::PilotAccelerationBoundedRotorDifferentialGridYawTrim,"
         if "yaw_balance" in model else "\tLqiCoaxialCollectiveModel::CollectiveContract::PilotAccelerationDualMotorCommonDifferentialServoRollPitch,"),
        "\tkAttitudeGain,",
        "\tkServoDirection,",
        "\tkMaximumMoment,",
        "\tkMotorEffectivenessPoints,",
        "\tkMotorEffectiveness,",
        f"\t{np.asarray(model['motor_effectiveness_points']).size}u,",
        "\tkUpperTable,",
        "\tkLowerTable,",
        f"\t{np.asarray(model['upper_table']).shape[0]}u,",
        f"\t{np.asarray(model['lower_table']).shape[0]}u,",
        f"\t{literal(model['mass'])},",
        f"\t{literal(model['gravity'])},",
        "\t{" + ", ".join(literal(value) for value in thrust) + "},",
        "\t{" + ", ".join(literal(value) for value in torque) + "},",
        f"\t{literal(model['maximum_thrust'])},",
        f"\t{literal(model['minimum_thrust_fraction'])},",
        f"\t{literal(model['maximum_thrust_fraction'])},",
        f"\t{literal(model['tilt_cosine_minimum'])},",
        f"\t{literal(model['pilot_minimum'])},",
        f"\t{literal(model['pilot_maximum'])},",
        f"\t{literal(model['maximum_vertical_acceleration'])},",
        f"\t{literal(model['acceleration_kp'])},",
        f"\t{literal(model['acceleration_ki'])},",
        f"\t{literal(model['acceleration_integral_limit'])},",
        f"\t{literal(model['acceleration_filter_time_constant'])},",
        f"\t{literal(model['servo_effectiveness_scale'])},",
        f"\t{literal(model['servo_scale_minimum'])},",
        f"\t{literal(model['servo_scale_maximum'])},",
        f"\t{literal(model['motor_mix_regularization'])},",
        f"\t{literal(model['zero_throttle_threshold'])},",
        f"\t{literal(model['dt'])},",
        f"\t{literal(model['max_roll_angle'])},",
        f"\t{literal(model['max_pitch_angle'])},",
        f"\t{literal(model['max_yaw_rate'])},",
        "\tkManualCommandDecay,",
        f"\t{weight_bytes}u,",
        f"\t0x{digest:08x}u,",
        *(["\tkYawBalance,"] if "yaw_balance" in model else []),
        *([f"\t{literal(model['roll_pitch_integral_moment_fraction'])},", f"\t{literal(model['grid_common_preference_fraction'])},"]
          if model.get("allocation_revision") == 2 else []),
        "};",
        "",
        "} // namespace lqi_coaxial_collective",
        "",
    ]
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text("\n".join(lines), encoding="utf-8")
    return digest


def quaternion_error(
    current: np.ndarray, target: np.ndarray
) -> np.ndarray:
    tw, tx, ty, tz = [np.float32(value) for value in target]
    cw, cx, cy, cz = [np.float32(value) for value in current]
    error = np.asarray(
        [
            tw * cw + tx * cx + ty * cy + tz * cz,
            tw * cx - tx * cw - ty * cz + tz * cy,
            tw * cy + tx * cz - ty * cw - tz * cx,
            tw * cz - tx * cy + ty * cx - tz * cw,
        ],
        dtype=np.float32,
    )
    if error[0] < 0.0:
        error = -error
    norm = np.sqrt(
        np.sum(error[1:] * error[1:], dtype=np.float32),
        dtype=np.float32,
    )
    angle = np.float32(2.0) * np.arctan2(
        norm, max(error[0], np.float32(1e-12))
    )
    return (
        error[1:] * angle / max(norm, np.float32(1e-12))
    ).astype(np.float32)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _inverse_lookup(
    value: float, table: np.ndarray
) -> float:
    upper = 1
    while upper < len(table) - 1 and value > table[upper, 1]:
        upper += 1
    lower = upper - 1
    command_0, speed_0 = table[lower]
    command_1, speed_1 = table[upper]
    if speed_1 <= speed_0:
        return float(command_1)
    if value <= table[1, 1]:
        return float(table[1, 0])
    if value >= table[-1, 1]:
        return float(table[-1, 0])
    return float(
        command_0
        + (value - speed_0) * (command_1 - command_0)
        / (speed_1 - speed_0)
    )


def coaxial_step_float32(
    current_q: np.ndarray,
    target_q: np.ndarray,
    current_rate: np.ndarray,
    target_rate: np.ndarray,
    acceleration_b: np.ndarray,
    collective: float,
    persistent: np.ndarray,
    model: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    current_q = f32(current_q)
    target_q = f32(target_q)
    current_rate = f32(current_rate)
    target_rate = f32(target_rate)
    acceleration_b = f32(acceleration_b)
    persistent = f32(persistent)
    error = quaternion_error(current_q, target_q)
    rate_error = current_rate - target_rate
    collective_f = f32(_clamp(collective, 0.0, 1.0))
    if collective_f <= model["zero_throttle_threshold"]:
        next_state = persistent.copy()
        next_state[1:] = 0.0
        return np.zeros(5, dtype=np.float32), next_state

    qw, qx, qy, qz = [float(value) for value in current_q]
    rotation_31 = 2.0 * (qx * qz - qw * qy)
    rotation_32 = 2.0 * (qy * qz + qw * qx)
    rotation_33 = 1.0 - 2.0 * (qx * qx + qy * qy)
    specific_force_z = (
        rotation_31 * acceleration_b[0]
        + rotation_32 * acceleration_b[1]
        + rotation_33 * acceleration_b[2]
    )
    measured_up = -(specific_force_z + model["gravity"])
    tau = model["acceleration_filter_time_constant"]
    alpha = DT_S / (tau + DT_S) if tau > 0.0 else 1.0
    filtered_up = float(
        persistent[0] + alpha * (measured_up - persistent[0])
    )
    normalized_collective = (
        2.0
        * _clamp(
            (collective_f - model["pilot_minimum"])
            / (model["pilot_maximum"] - model["pilot_minimum"]),
            0.0,
            1.0,
        )
        - 1.0
    )
    target_up = (
        normalized_collective
        * model["maximum_vertical_acceleration"]
    )
    acceleration_error = target_up - filtered_up
    z_integral = _clamp(
        persistent[1] + acceleration_error * DT_S,
        -model["acceleration_integral_limit"],
        model["acceleration_integral_limit"],
    )
    corrected_up = (
        target_up
        + model["acceleration_kp"] * acceleration_error
        + model["acceleration_ki"] * z_integral
    )
    acceleration_limit = (
        1.5 * model["maximum_vertical_acceleration"]
        + model["acceleration_integral_limit"]
    )
    bounded_up = _clamp(
        corrected_up,
        -model["gravity"] - acceleration_limit,
        acceleration_limit,
    )
    tilt_cosine = max(
        1.0 - 2.0 * (qx * qx + qy * qy),
        model["tilt_cosine_minimum"],
    )
    desired_thrust = _clamp(
        model["mass"] * (model["gravity"] + bounded_up) / tilt_cosine,
        model["minimum_thrust_fraction"] * model["maximum_thrust"],
        model["maximum_thrust_fraction"] * model["maximum_thrust"],
    )
    policy = model.get("yaw_balance")
    if policy is not None:
        rate_error[2] = current_rate[2] - np.clip(target_rate[2], -model["max_yaw_rate"], model["max_yaw_rate"])
    speed_ratio = float(policy[0]) if policy is not None else math.sqrt(
        model["motor_torque_coefficients"][0]
        / max(model["motor_torque_coefficients"][1], 1.0e-16)
    )
    coefficients = model["thrust_coefficients"]
    denominator = (
        coefficients[0]
        + coefficients[1] * speed_ratio * speed_ratio
        + coefficients[2] * speed_ratio
    )
    upper_speed = math.sqrt(max(desired_thrust, 0.0) / max(denominator, 1e-16))
    base_saturated = False
    if policy is not None:
        speed_cap = min(float(np.interp(policy[2], model["upper_table"][:, 0], model["upper_table"][:, 1])),
                        float(np.interp(policy[2], model["lower_table"][:, 0], model["lower_table"][:, 1])) / speed_ratio)
        base_saturated = upper_speed > speed_cap
        upper_speed = min(upper_speed, speed_cap)
    lower_speed = speed_ratio * upper_speed
    upper_base = _inverse_lookup(upper_speed, model["upper_table"])
    lower_base = _inverse_lookup(lower_speed, model["lower_table"])

    attitude_integral = persistent[2:5] + np.asarray(
        [error[0], error[1], rate_error[2]], dtype=np.float32
    ) * DT_S
    yaw_ki = float(np.asarray(model["attitude_gain"])[2, 7])
    if policy is not None:
        integral_limit = policy[5] / max(abs(yaw_ki), 1e-9)
        attitude_integral[2] = np.clip(attitude_integral[2], -integral_limit, integral_limit)
    shared_yaw = model.get("allocation_revision") == 2
    base_thrust = denominator * upper_speed**2
    previous_rp = persistent[2:4].copy()
    if shared_yaw:
        for axis in range(2):
            gain = float(model["attitude_gain"][axis, 5 + axis])
            limit = model["roll_pitch_integral_moment_fraction"] * model["maximum_moment"][axis] / max(abs(gain), 1e-9)
            previous_rp[axis] = np.clip(previous_rp[axis], -limit, limit)
            attitude_integral[axis] = previous_rp[axis] if base_thrust < .5 * model["mass"] * model["gravity"] else np.clip(previous_rp[axis] + error[axis] * DT_S, -limit, limit)
    state = np.concatenate(
        (error[:2], rate_error, attitude_integral)
    ).astype(np.float32)
    desired_moment = (
        -np.asarray(model["attitude_gain"], dtype=np.float32) @ state
    )
    raw_moment = desired_moment.copy()
    raw_yaw_moment = float(desired_moment[2])
    desired_moment = np.clip(
        desired_moment,
        -np.asarray(model["maximum_moment"], dtype=np.float32),
        np.asarray(model["maximum_moment"], dtype=np.float32),
    ).astype(np.float32)
    servo_scale = _clamp(
        desired_thrust
        / (model["mass"] * model["gravity"])
        * model["servo_effectiveness_scale"],
        model["servo_scale_minimum"],
        model["servo_scale_maximum"],
    )
    servo_raw = (
        np.asarray(model["servo_direction"], dtype=np.float32)
        @ desired_moment[:2]
        / servo_scale
    )
    servo_limit = float(np.max(np.abs(servo_raw)))
    servo_saturated = servo_limit > 1.0
    servos = (
        servo_raw / max(servo_limit, 1.0) if servo_saturated else servo_raw
    )

    base_yaw = 0.0
    grid_yaw = 0.0
    if policy is not None:
        common_min = max(-1.0, float(np.max(-1.0 - servos)))
        common_max = min(1.0, float(np.min(1.0 - servos)))
        full_common = policy[3] * np.clip(desired_thrust / (model["mass"] * model["gravity"]), 0, 1)
        common_min = max(common_min, min(full_common, 0))
        common_max = min(common_max, max(full_common, 0))
        common = np.clip(full_common * (model["grid_common_preference_fraction"] if shared_yaw else 1), common_min, common_max)
        servos = np.clip(servos + common, -1, 1)
        base_yaw = model["motor_torque_coefficients"][0] * upper_speed**2 - model["motor_torque_coefficients"][1] * policy[6] * lower_speed**2
        grid_yaw = policy[4] * denominator * upper_speed**2 * common
    motor_yaw_request = float(desired_moment[2]) - base_yaw - grid_yaw
    effectiveness_points = np.asarray(
        model["motor_effectiveness_points"], dtype=np.float32
    )
    effectiveness_values = np.asarray(
        model["motor_effectiveness"], dtype=np.float32
    ).reshape(len(effectiveness_points), 4)
    average_command = _clamp(0.5 * (upper_base + lower_base), 0.0, 1.0)
    effectiveness = np.asarray(
        [
            interpolate_linear(
                average_command,
                effectiveness_points,
                effectiveness_values[:, column],
            )
            for column in range(4)
        ],
        dtype=np.float32,
    )
    if policy is not None:
        effectiveness[3] *= policy[6]
    common_regularization = 2.0 * model["motor_mix_regularization"]
    common_t = float(effectiveness[0] + effectiveness[1])
    differential_t = float(effectiveness[0] - effectiveness[1])
    common_yaw = float(effectiveness[2] + effectiveness[3])
    differential_yaw = float(effectiveness[2] - effectiveness[3])
    determinant = (
        (common_t + common_regularization)
        * (differential_yaw + common_regularization)
        - differential_t * common_yaw
    )
    if abs(determinant) < 1e-12:
        raise ValueError("singular motor mixer")
    common_output = -differential_t * motor_yaw_request / determinant
    differential_output = (
        (common_t + common_regularization)
        * motor_yaw_request
        / determinant
    )
    motor_raw = np.asarray(
        [
            upper_base + common_output + differential_output,
            lower_base + common_output - differential_output,
        ],
        dtype=np.float32,
    )
    motors = np.clip(motor_raw, 0.0, 1.0)
    motor_saturated = bool(
        np.any(motor_raw < 0.0) or np.any(motor_raw > 1.0)
    )
    yaw_limited = abs(raw_yaw_moment - desired_moment[2]) > 1e-7
    if policy is not None:
        def feasible(value):
            if np.any(value < 0) or np.any(value > policy[2]):
                return False
            speed = [float(np.interp(value[i], model[key][:, 0], model[key][:, 1])) for i, key in enumerate(("upper_table", "lower_table"))]
            return speed[0] <= policy[1] * speed[1] + 1e-3 and speed[1] <= policy[1] * speed[0] + 1e-3
        base = np.array([upper_base, lower_base], dtype=np.float32)
        def allocate(delta):
            fraction = 1.0
            if not feasible(base + delta):
                low, high = 0.0, 1.0
                for _ in range(16):
                    middle = .5 * (low + high)
                    if feasible(base + middle * delta):
                        low = middle
                    else:
                        high = middle
                fraction = low
            return base + fraction * delta, fraction
        motors, fraction = allocate(motor_raw - base)
        if shared_yaw and fraction < 1:
            achieved_motor = float(effectiveness[2:] @ (motors - base))
            grid_slope = policy[4] * base_thrust
            if abs(grid_slope) > 1e-9:
                selected = np.clip(common + (motor_yaw_request - achieved_motor) / grid_slope, common_min, common_max)
                servos = np.clip(servos + selected - common, -1, 1)
                common = selected
                grid_yaw = grid_slope * common
                motor_yaw_request = float(desired_moment[2]) - base_yaw - grid_yaw
                delta = np.asarray([(-differential_t + common_t + common_regularization) / determinant,
                                    (-differential_t - common_t - common_regularization) / determinant], dtype=np.float32) * motor_yaw_request
                motors, fraction = allocate(delta)
        motor_saturated = base_saturated or fraction < 1
        yaw_limited = yaw_limited or fraction < 1
    next_state = np.empty(5, dtype=np.float32)
    next_state[0] = filtered_up
    next_state[1] = persistent[1] if (base_saturated if policy is not None else motor_saturated) else z_integral
    next_state[2:5] = (
        persistent[2:5]
        if servo_saturated or motor_saturated
        else attitude_integral
    )
    if policy is not None:
        next_state[2:4] = persistent[2:4] if servo_saturated else attitude_integral[:2]
        if shared_yaw:
            for axis in range(2):
                gain = float(model["attitude_gain"][axis, 5 + axis])
                achieved = float(desired_moment[axis]) / max(servo_limit, 1.0)
                residual = float(raw_moment[axis]) - achieved
                change = -gain * (attitude_integral[axis] - previous_rp[axis])
                next_state[2 + axis] = previous_rp[axis] if abs(residual) > 1e-7 and change * residual > 0 else attitude_integral[axis]
        achieved_yaw = base_yaw + grid_yaw + float(effectiveness @ np.array([0, 0, motors[0] - upper_base, motors[1] - lower_base]))
        residual = raw_yaw_moment - achieved_yaw
        delta_moment = -yaw_ki * (attitude_integral[2] - persistent[4])
        next_state[4] = np.clip(persistent[4] if yaw_limited and delta_moment * residual > 0 else attitude_integral[2], -integral_limit, integral_limit)
    output = np.concatenate(
        (motors, np.clip(servos, -1.0, 1.0))
    ).astype(np.float32)
    return output, next_state


def golden_vectors(
    model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cases = (
        ([1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, -9.80665], 0.525, [0] * 5),
        ([0.9998, 0.01, 0, 0], [1, 0, 0, 0], [0.1, -0.2, 0.05], [0, 0, 0.3], [0, 0, -9.7], 0.70, [0] * 5),
        ([0.9, 0.2, -0.3, 0.1], [0.98, -0.1, 0.12, 0.02], [-1, 0.8, 0.6], [0, 0, -0.5], [0.2, -0.1, -9.5], 0.35, [0.2, -0.1, 0.05, -0.02, 0.03]),
        ([-0.9998, -0.01, 0, 0], [1, 0, 0, 0], [0.3, 0.4, 0.2], [0, 0, 1.2], [0, 0, -9.2], 0.85, [0.5, 0.3, 0.1, 0.05, 0.2]),
        ([0.8, 0.4, 0.4, 0.2], [0.9239, 0, 0, 0.3827], [3, -3, 2], [0, 0, 0], [0, 0, -8.0], 0.02, [-0.3, 0.2, -0.3, 0, 0.1]),
    )
    vectors = []
    for current_q, target_q, current_rate, target_rate, acceleration, collective, persistent in cases:
        arrays = [
            np.asarray(value, dtype=np.float32)
            for value in (
                current_q,
                target_q,
                current_rate,
                target_rate,
                acceleration,
                persistent,
            )
        ]
        output, next_state = coaxial_step_float32(
            arrays[0],
            arrays[1],
            arrays[2],
            arrays[3],
            arrays[4],
            collective,
            arrays[5],
            model,
        )
        vectors.append(
            {
                "attitude_q_wb": arrays[0].tolist(),
                "target_attitude_q_wb": arrays[1].tolist(),
                "angular_velocity_b": arrays[2].tolist(),
                "target_angular_velocity_b": arrays[3].tolist(),
                "acceleration_b": arrays[4].tolist(),
                "collective_base": collective,
                "persistent": arrays[5].tolist(),
                "expected_output": output.tolist(),
                "expected_next_persistent": next_state.tolist(),
            }
        )
    return vectors


def write_golden(
    vectors: list[dict[str, Any]],
    golden_header: Path,
    golden_json: Path,
) -> None:
    lines = [
        "#pragma once",
        "",
        "struct LqiCoaxialCollectiveGoldenVector {",
        "\tfloat attitude_q_wb[4];",
        "\tfloat target_attitude_q_wb[4];",
        "\tfloat angular_velocity_b[3];",
        "\tfloat target_angular_velocity_b[3];",
        "\tfloat acceleration_b[3];",
        "\tfloat collective_base;",
        "\tfloat persistent[5];",
        "\tfloat expected_output[5];",
        "\tfloat expected_next_persistent[5];",
        "};",
        "",
        "static constexpr LqiCoaxialCollectiveGoldenVector kLqiCoaxialCollectiveGoldenVectors[] = {",
    ]
    for vector in vectors:
        lines.append("\t{")
        for name in (
            "attitude_q_wb",
            "target_attitude_q_wb",
            "angular_velocity_b",
            "target_angular_velocity_b",
            "acceleration_b",
        ):
            lines.append(
                "\t\t{" + ", ".join(literal(v) for v in vector[name]) + "},"
            )
        lines.append(f"\t\t{literal(vector['collective_base'])},")
        for name in (
            "persistent",
            "expected_output",
            "expected_next_persistent",
        ):
            lines.append(
                "\t\t{" + ", ".join(literal(v) for v in vector[name]) + "},"
            )
        lines.append("\t},")
    lines += [
        "};",
        "",
        "static constexpr int kLqiCoaxialCollectiveGoldenVectorCount{"
        + str(len(vectors))
        + "};",
        "",
    ]
    golden_header.parent.mkdir(parents=True, exist_ok=True)
    golden_json.parent.mkdir(parents=True, exist_ok=True)
    golden_header.write_text("\n".join(lines), encoding="utf-8")
    golden_json.write_text(
        json.dumps({"schema_version": 1, "vectors": vectors}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment-config",
        type=Path,
        default=DEPLOYMENT_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "px4_trans/generated/airframe-1",
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="archive the candidate without replacing the PX4 build headers",
    )
    parser.add_argument("--reference-only-manifest", type=Path, help="Reuse a verified payload; update only manual reference constants")
    parser.add_argument("--reuse-model-manifest", type=Path, help="Reuse verified plant/gains and apply an explicit deployment yaw-balance profile")
    args = parser.parse_args()
    deployment_config = args.deployment_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    payload_path = output_dir / "LqiCoaxialCollectivePayload.hpp"
    golden_header = output_dir / "lqi_coaxial_collective_golden.hpp"
    golden_json = output_dir / "lqi_coaxial_collective_golden.json"
    manifest_path = output_dir / "lqi_coaxial_collective_manifest.json"

    if args.reference_only_manifest and args.reuse_model_manifest:
        raise ValueError("choose one verified-model reuse mode")
    if args.reference_only_manifest or args.reuse_model_manifest:
        manifest = args.reference_only_manifest or args.reuse_model_manifest
        model, deployment, metadata = load_reference_update(deployment_config, manifest.resolve())
        if args.reference_only_manifest and (
            (deployment.get("yaw_balance_profile") is None) != (metadata.get("yaw_balance_profile") is None)
            or (deployment.get("yaw_balance_profile") is not None and not np.array_equal(
                np.asarray([deployment["yaw_balance_profile"][key] for key in YAW_BALANCE_FIELDS], dtype=np.float32),
                model.get("yaw_balance")))
        ):
            raise ValueError("reference-only mode cannot change the yaw balance profile; use --reuse-model-manifest")
    else:
        model, deployment, metadata = load_model(deployment_config)
    if args.reference_only_manifest and deployment.get("allocation_safety") != metadata.get("allocation_safety"):
        raise ValueError("reference-only mode cannot change allocation safety; use --reuse-model-manifest")
    if args.reference_only_manifest and deployment.get("attitude_response") != metadata.get("attitude_response"):
        raise ValueError("reference-only mode cannot change attitude gains; use --reuse-model-manifest")
    apply_yaw_balance_profile(model, deployment, metadata)
    apply_attitude_response(model, deployment, metadata)
    digest = write_model_header(model, metadata, payload_path)
    vectors = golden_vectors(model)
    write_golden(vectors, golden_header, golden_json)

    active_payload = payload_path
    active_golden_header = golden_header
    active_golden_json = golden_json
    if not args.no_activate:
        if payload_path.resolve() != PAYLOAD_HEADER.resolve():
            shutil.copyfile(payload_path, PAYLOAD_HEADER)
        if golden_header.resolve() != GOLDEN_HEADER.resolve():
            shutil.copyfile(golden_header, GOLDEN_HEADER)
        if golden_json.resolve() != GOLDEN_JSON.resolve():
            shutil.copyfile(golden_json, GOLDEN_JSON)
        active_payload = PAYLOAD_HEADER
        active_golden_header = GOLDEN_HEADER
        active_golden_json = GOLDEN_JSON

    manifest = {
        "schema_version": 1,
        "artifact_type": "px4_coaxial_collective_lqi_model",
        "airframe_id": "airframe-1",
        "airframe_number": 1,
        "controller_contract": ("pilot_acceleration_shared_yaw_antiwindup"
                                if model.get("allocation_revision") == 2 else "pilot_acceleration_bounded_rotor_differential_grid_yaw_trim"
                                if "yaw_balance" in model else "pilot_acceleration_dual_motor_common_differential_servo_roll_pitch"),
        "model_checksum": f"0x{digest:08x}",
        "source": metadata,
        "generated": {
            "candidate_payload": str(payload_path.relative_to(ROOT)),
            "candidate_payload_sha256": file_sha256(payload_path),
            "active_header": str(active_payload.relative_to(ROOT)),
            "active_header_sha256": file_sha256(active_payload),
            "candidate_golden_header": str(golden_header.relative_to(ROOT)),
            "candidate_golden_header_sha256": file_sha256(golden_header),
            "candidate_golden_json": str(golden_json.relative_to(ROOT)),
            "candidate_golden_json_sha256": file_sha256(golden_json),
            "golden_header": str(active_golden_header.relative_to(ROOT)),
            "golden_header_sha256": file_sha256(active_golden_header),
            "golden_json": str(active_golden_json.relative_to(ROOT)),
            "golden_json_sha256": file_sha256(active_golden_json),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
