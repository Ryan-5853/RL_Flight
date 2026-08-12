#!/usr/bin/env python3
"""Generate the PX4 nominal-LQI model payload and host golden vectors.

The generated header contains data only.  The reusable C++ algorithm lives in
LqiControllerCore.hpp and consumes the stable descriptor in
LqiControllerModel.hpp.

Run from the repository root:
    PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
    python px4_trans/tools/generate_lqi_backend.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
MODEL_HEADER = ROOT / "px4_trans/px4/src/modules/nn_control/LqiNominalModel.hpp"
GOLDEN_HEADER = ROOT / "px4_trans/tests/lqi_golden.hpp"
GOLDEN_JSON = ROOT / "px4_trans/tests/lqi_golden.json"
COMPOSITE_MODEL_HEADER = ROOT / "px4_trans/px4/src/modules/nn_control/LqiIdentifiedModel.hpp"
COMPOSITE_GOLDEN_HEADER = ROOT / "px4_trans/tests/lqi_identified_golden.hpp"
COMPOSITE_GOLDEN_JSON = ROOT / "px4_trans/tests/lqi_identified_golden.json"
DEPLOYMENT_CONFIG = ROOT / "px4_trans/configs/lqi_nominal_deployment.yaml"
DT_S = 0.002
RPM_TO_RAD_S = 2.0 * math.pi / 60.0


def f32(value: float) -> np.float32:
    return np.float32(value)


def load_model() -> dict[str, np.ndarray | np.float32]:
    from flight_controller import ControllerContext, create_controller, load_controller_config
    from flight_identification.control_evaluation import _nominal_actuator_model
    from simenv.config import load_and_materialize

    deployment = yaml.safe_load(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))
    if deployment.get("schema_version") != 1:
        raise ValueError("unsupported LQI deployment schema")
    simulator_config = ROOT / deployment["simulator_config"]
    controller_config = ROOT / deployment["controller_config"]
    materialized = load_and_materialize(
        simulator_config, 1, torch.device("cpu"), torch.float64
    )
    config = load_controller_config(controller_config)
    controller = create_controller(
        config,
        ControllerContext(
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float64,
            control_dt=DT_S,
            parameters=materialized.parameters,
        ),
    )
    gain = controller._lqr_gain
    if gain is None or tuple(gain.shape) != (4, 13):
        raise RuntimeError(f"nominal controller must provide a 4x13 gain, got {None if gain is None else tuple(gain.shape)}")
    _, slopes, time_constants = _nominal_actuator_model(simulator_config)
    decay = np.exp(-DT_S / time_constants).astype(np.float32)
    sticks = deployment["manual_reference"]
    command_tau = np.asarray(
        [
            sticks["roll"]["time_constant_s"],
            sticks["pitch"]["time_constant_s"],
            sticks["yaw"]["time_constant_s"],
        ],
        dtype=np.float32,
    )
    attitude = config["params"]["pid"]["attitude"]
    trim = controller.trim
    return {
        "gain": gain.detach().cpu().numpy().astype(np.float32),
        "decay": decay,
        "slope_eff": (slopes.astype(np.float32) * (1.0 - decay)).astype(np.float32),
        "integral_limit": np.asarray(attitude["integral_limit"], dtype=np.float32),
        "motor_table": materialized.parameters["motors.pwm_to_rpm_table"][0, 0].detach().cpu().numpy().astype(np.float32),
        "dt": f32(DT_S),
        "upper_trim": f32(trim.command[0, 0].item()),
        "lower_trim": f32(trim.command[0, 1].item()),
        "upper_trim_rad_s": f32(trim.motor_speed[0, 0].item()),
        "lower_trim_rad_s": f32(trim.motor_speed[0, 1].item()),
        "rpm_to_rad_s": f32(RPM_TO_RAD_S),
        "max_roll_angle": f32(sticks["roll"]["limit_rad"]),
        "max_pitch_angle": f32(sticks["pitch"]["limit_rad"]),
        "max_yaw_rate": f32(sticks["yaw"]["limit_rad_s"]),
        "manual_command_decay": np.exp(-DT_S / command_tau).astype(np.float32),
    }


def quaternion_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    tw, tx, ty, tz = target
    cw, cx, cy, cz = current
    error = np.asarray(
        [
            tw * cw + tx * cx + ty * cy + tz * cz,
            tw * cx - tx * cw - ty * cz + tz * cy,
            tw * cy + tx * cz - ty * cw - tz * cx,
            tw * cz - tx * cy + ty * cx - tz * cw,
        ],
        dtype=np.float32,
    )
    if error[0] < 0:
        error = -error
    norm = np.sqrt(np.sum(error[1:] * error[1:], dtype=np.float32), dtype=np.float32)
    angle = f32(2.0) * np.arctan2(norm, np.maximum(error[0], f32(1e-12)))
    return (error[1:] * angle / np.maximum(norm, f32(1e-12))).astype(np.float32)


def lqi_step_float32(
    current_q: np.ndarray,
    target_q: np.ndarray,
    current_rate: np.ndarray,
    target_rate: np.ndarray,
    collective: float,
    motor_rpm: np.ndarray,
    motor_rpm_valid: bool,
    persistent: np.ndarray,
    model: dict[str, np.ndarray | np.float32],
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    error = quaternion_error(current_q, target_q)
    rate_error = (current_rate - target_rate).astype(np.float32)
    collective_f = np.clip(f32(collective), f32(0.0), f32(1.0)).astype(np.float32)
    if collective_f <= f32(0.0):
        # Idle throttle: no actuator output and persistent state stays frozen.
        # Must match LqiControllerCore::stepFromState.
        return (
            np.zeros(4, dtype=np.float32),
            persistent.copy(),
            collective_f,
        )
    target_speed = f32(np.interp(float(collective_f), model["motor_table"][:, 0], model["motor_table"][:, 1]))
    target_dev = target_speed - np.asarray(
        [model["upper_trim_rad_s"], model["lower_trim_rad_s"]], dtype=np.float32
    )
    if motor_rpm_valid:
        motor_error = np.abs(motor_rpm).astype(np.float32) * model["rpm_to_rad_s"] - np.asarray(
            [model["upper_trim_rad_s"], model["lower_trim_rad_s"]], dtype=np.float32
        ) - target_dev
    else:
        motor_error = persistent[:2] - target_dev
    state = np.concatenate(
        (error[:2], rate_error, motor_error, persistent[2:5], persistent[5:8])
    ).astype(np.float32)
    delta = np.empty(4, dtype=np.float32)
    for row in range(4):
        accumulator = f32(0.0)
        for col in range(13):
            accumulator = f32(accumulator + f32(model["gain"][row, col]) * state[col])
        delta[row] = -accumulator
    raw = np.concatenate((collective_f + delta[:1], delta[1:])).astype(np.float32)
    # Keep the lower-rotor opening within 20 percentage points of the pilot
    # upper-rotor command, additionally bounded to [0, 1]. Must match
    # LqiControllerCore::stepFromState.
    lower_min = np.maximum(collective_f - f32(0.2), f32(0.0)).astype(np.float32)
    lower_max = np.minimum(collective_f + f32(0.2), f32(1.0)).astype(np.float32)
    output = np.concatenate(
        (np.clip(raw[:1], lower_min, lower_max), np.clip(raw[1:], f32(-1.0), f32(1.0)))
    ).astype(np.float32)
    saturated = bool(
        np.any(raw[:1] < lower_min)
        or np.any(raw[:1] > lower_max)
        or np.any(raw[1:] < -1.0)
        or np.any(raw[1:] > 1.0)
    )
    integral_error = np.asarray([error[0], error[1], rate_error[2]], dtype=np.float32)
    candidate = np.clip(
        persistent[5:8] + integral_error * model["dt"],
        -model["integral_limit"],
        model["integral_limit"],
    ).astype(np.float32)
    command_dev = np.concatenate(([collective_f - model["upper_trim"]], output - np.asarray(
        [model["lower_trim"], 0.0, 0.0, 0.0], dtype=np.float32
    ))).astype(np.float32)
    actuator_next = (model["decay"] * persistent[:5] + model["slope_eff"] * command_dev).astype(np.float32)
    if motor_rpm_valid:
        actuator_next[:2] = np.abs(motor_rpm).astype(np.float32) * model["rpm_to_rad_s"] - np.asarray(
            [model["upper_trim_rad_s"], model["lower_trim_rad_s"]], dtype=np.float32
        )
    next_persistent = np.concatenate((actuator_next, persistent[5:8] if saturated else candidate)).astype(np.float32)
    return output, next_persistent, collective_f


def golden_vectors(model: dict[str, np.ndarray | np.float32]) -> list[dict[str, object]]:
    cases = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0], 0.5, [0, 0], False, [0] * 8),
        ([0.9998, 0.01, 0, 0], [1, 0, 0, 0], [0.1, -0.2, 0.05], [0, 0, 0.3], 0.75, [11000, -11500], True, [0] * 8),
        ([0.9, 0.2, -0.3, 0.1], [0.98, -0.1, 0.12, 0.02], [-1, 0.8, 0.6], [0, 0, -0.5], 0.2, [9800, 10200], True, [100, -50, 0.05, -0.02, 0.03, 0.1, -0.2, 0.3]),
        ([-0.9998, -0.01, 0, 0], [1, 0, 0, 0], [0.3, 0.4, 0.2], [0, 0, 1.2], 1.0, [0, 0], False, [500, 400, 0.3, 0.25, 0.2, 0.2, 0.2, 0.4]),
        ([0.8, 0.4, 0.4, 0.2], [0.9239, 0, 0, 0.3827], [3, -3, 2], [0, 0, 0], 0.0, [13500, 9000], True, [-300, 200, -0.3, 0, 0.1, -0.25, 0.15, -0.45]),
    ]
    vectors = []
    for current_q, target_q, current_rate, target_rate, collective, rpm, valid, persistent in cases:
        arrays = [np.asarray(value, dtype=np.float32) for value in (current_q, target_q, current_rate, target_rate, rpm, persistent)]
        output, next_state, upper = lqi_step_float32(*arrays[:4], collective, arrays[4], valid, arrays[5], model)
        vectors.append({
            "attitude_q_wb": arrays[0].tolist(), "target_attitude_q_wb": arrays[1].tolist(),
            "angular_velocity_b": arrays[2].tolist(), "target_angular_velocity_b": arrays[3].tolist(),
            "collective_base": float(collective), "motor_rpm": arrays[4].tolist(), "motor_rpm_valid": valid,
            "persistent": arrays[5].tolist(), "expected_output": output.tolist(),
            "expected_upper": float(upper),
            "expected_next_persistent": next_state.tolist(),
        })
    return vectors


def fnv1a32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def checksum(model: dict[str, np.ndarray | np.float32]) -> int:
    names = ("gain", "decay", "slope_eff", "integral_limit", "motor_table", "dt", "upper_trim", "lower_trim", "upper_trim_rad_s", "lower_trim_rad_s", "rpm_to_rad_s", "max_roll_angle", "max_pitch_angle", "max_yaw_rate", "manual_command_decay")
    return fnv1a32(b"".join(np.asarray(model[name], dtype=np.float32).tobytes() for name in names))


def literal(value: float) -> str:
    rendered = f"{float(value):.9g}"
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "f"


def array_lines(name: str, values: np.ndarray) -> list[str]:
    result = [f"static constexpr float {name}[{values.size}] = {{"]
    result.extend(f"\t{literal(value)}," for value in values.ravel())
    result.append("};")
    return result


def write_model_header(model: dict[str, np.ndarray | np.float32]) -> int:
    digest = checksum(model)
    lines = [
        "/****************************************************************************", " * Generated by px4_trans/tools/generate_lqi_backend.py. Do not edit.",
        " * Source: Controller/configs/lqi_sim2real_micro_coaxial_4out.yaml", " ****************************************************************************/", "", "#pragma once", "",
        '#include "LqiControllerModel.hpp"', "", "namespace lqi_nominal", "{", "",
    ]
    for name in ("gain", "decay", "slope_eff", "integral_limit", "motor_table", "manual_command_decay"):
        lines += array_lines("k" + "".join(part.title() for part in name.split("_")), np.asarray(model[name])) + [""]
    lines += [
        "static constexpr LqiControllerModel kModel {",
        "\tLqiControllerModel::kSchemaVersion,", "\tLqiControllerModel::WorldFrame::NED,", "\tLqiControllerModel::BodyFrame::FRD,",
        "\tLqiControllerModel::QuaternionConvention::HamiltonWxyzBodyToWorld,", "\tLqiControllerModel::CollectiveContract::ExternalUpperLqiLowerAndServos,",
        "\tLqiControllerModel::ActuatorObserver::MotorRpmOrCommandDriven,", "\tkGain,", "\tkDecay,", "\tkSlopeEff,", "\tkIntegralLimit,", "\tkMotorTable,", "\t7u,",
        f"\t{literal(model['dt'])},", f"\t{literal(model['upper_trim'])},", f"\t{literal(model['lower_trim'])},",
        f"\t{literal(model['upper_trim_rad_s'])},", f"\t{literal(model['lower_trim_rad_s'])},", f"\t{literal(model['rpm_to_rad_s'])},",
        f"\t{literal(model['max_roll_angle'])},", f"\t{literal(model['max_pitch_angle'])},", f"\t{literal(model['max_yaw_rate'])},", "\tkManualCommandDecay,",
        "\t364u,", f"\t0x{digest:08x}u,", "};", "", "} // namespace lqi_nominal", "",
    ]
    MODEL_HEADER.write_text("\n".join(lines), encoding="utf-8")
    return digest


def write_golden(vectors: list[dict[str, object]]) -> None:
    fields = [
        "\tfloat attitude_q_wb[4];", "\tfloat target_attitude_q_wb[4];", "\tfloat angular_velocity_b[3];",
        "\tfloat target_angular_velocity_b[3];", "\tfloat collective_base;", "\tfloat motor_rpm[2];", "\tbool motor_rpm_valid;",
        "\tfloat persistent[8];", "\tfloat expected_output[4];", "\tfloat expected_upper;", "\tfloat expected_next_persistent[8];",
    ]
    lines = ["#pragma once", "", "struct LqiGoldenVector {"] + fields + ["};", "", "static constexpr LqiGoldenVector kLqiGoldenVectors[] = {"]
    array_names = ("attitude_q_wb", "target_attitude_q_wb", "angular_velocity_b", "target_angular_velocity_b")
    for vector in vectors:
        lines.append("\t{")
        for name in array_names:
            lines.append("\t\t{" + ", ".join(literal(v) for v in vector[name]) + "},")
        lines.append(f"\t\t{literal(vector['collective_base'])},")
        lines.append("\t\t{" + ", ".join(literal(v) for v in vector["motor_rpm"]) + "},")
        lines.append("\t\t" + ("true," if vector["motor_rpm_valid"] else "false,"))
        for name in ("persistent", "expected_output"):
            lines.append("\t\t{" + ", ".join(literal(v) for v in vector[name]) + "},")
        lines.append(f"\t\t{literal(vector['expected_upper'])},")
        lines.append("\t\t{" + ", ".join(literal(v) for v in vector["expected_next_persistent"]) + "},")
        lines.append("\t},")
    lines += ["};", "", f"static constexpr int kLqiGoldenVectorCount{{{len(vectors)}}};", ""]
    GOLDEN_HEADER.write_text("\n".join(lines), encoding="utf-8")
    GOLDEN_JSON.write_text(json.dumps({"schema_version": 2, "vectors": vectors}, indent=2), encoding="utf-8")


def load_identified_model(
    identified_json: Path,
    nominal: dict[str, np.ndarray | np.float32],
    checkpoint_path: Path,
) -> dict[str, np.ndarray | np.float32]:
    """Build the 19-state composite payload from the offline identification."""

    report = json.loads(identified_json.read_text(encoding="utf-8"))
    aggregate = report["aggregate"]
    gain = np.asarray(aggregate["analysis_gain"], dtype=np.float32)
    coefficients = np.asarray(
        aggregate["predicted_composite_coefficients"], dtype=np.float32
    )
    if gain.shape != (4, 19):
        raise ValueError(f"identified gain must be 4x19, got {gain.shape}")
    if coefficients.shape != (3, 3, 3):
        raise ValueError(
            f"identified coefficients must be (3,3,3), got {coefficients.shape}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode_transform = np.asarray(checkpoint["mode_transform"], dtype=np.float32)
    servo_slopes = np.asarray(checkpoint["servo_command_slopes"], dtype=np.float32)
    basis_tau = np.asarray(
        tuple(float(value) for value in checkpoint["basis_time_constants_s"]),
        dtype=np.float32,
    )
    if mode_transform.shape != (3, 3) or servo_slopes.shape != (3,):
        raise ValueError("checkpoint mode_transform/slopes have unexpected shape")
    latent_decay = np.exp(-DT_S / basis_tau).astype(np.float32)
    servo_input = (mode_transform @ np.diag(servo_slopes)).astype(np.float32)
    latent_input = np.zeros((3, 3, 3), dtype=np.float32)
    for basis in range(3):
        latent_input[basis] = servo_input * (1.0 - latent_decay[basis])
    return {
        "gain": gain,
        "motor_decay": np.asarray(nominal["decay"])[:2].astype(np.float32),
        "motor_slope_eff": np.asarray(nominal["slope_eff"])[:2].astype(np.float32),
        "latent_decay": latent_decay,
        "latent_input": latent_input,
        "integral_limit": np.asarray(nominal["integral_limit"], dtype=np.float32),
        "motor_table": np.asarray(nominal["motor_table"], dtype=np.float32),
        "dt": f32(DT_S),
        "upper_trim": f32(float(nominal["upper_trim"])),
        "lower_trim": f32(float(nominal["lower_trim"])),
        "upper_trim_rad_s": f32(float(nominal["upper_trim_rad_s"])),
        "lower_trim_rad_s": f32(float(nominal["lower_trim_rad_s"])),
        "rpm_to_rad_s": f32(RPM_TO_RAD_S),
        "max_roll_angle": f32(float(nominal["max_roll_angle"])),
        "max_pitch_angle": f32(float(nominal["max_pitch_angle"])),
        "max_yaw_rate": f32(float(nominal["max_yaw_rate"])),
        "manual_command_decay": np.asarray(
            nominal["manual_command_decay"], dtype=np.float32
        ),
    }


def lqi_composite_step_float32(
    current_q: np.ndarray,
    target_q: np.ndarray,
    current_rate: np.ndarray,
    target_rate: np.ndarray,
    collective: float,
    motor_rpm: np.ndarray,
    motor_rpm_valid: bool,
    persistent: np.ndarray,
    model: dict[str, np.ndarray | np.float32],
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Python mirror of LqiCompositeCore::stepFromState (persistent[14])."""

    error = quaternion_error(current_q, target_q)
    rate_error = (current_rate - target_rate).astype(np.float32)
    collective_f = np.clip(f32(collective), f32(0.0), f32(1.0)).astype(np.float32)
    if collective_f <= f32(0.0):
        return (
            np.zeros(4, dtype=np.float32),
            persistent.copy(),
            collective_f,
        )
    target_speed = f32(
        np.interp(
            float(collective_f),
            model["motor_table"][:, 0],
            model["motor_table"][:, 1],
        )
    )
    target_dev = target_speed - np.asarray(
        [model["upper_trim_rad_s"], model["lower_trim_rad_s"]], dtype=np.float32
    )
    if motor_rpm_valid:
        motor_error = (
            np.abs(motor_rpm).astype(np.float32) * model["rpm_to_rad_s"]
            - np.asarray(
                [model["upper_trim_rad_s"], model["lower_trim_rad_s"]],
                dtype=np.float32,
            )
            - target_dev
        )
    else:
        motor_error = persistent[:2] - target_dev
    state = np.concatenate(
        (error[:2], rate_error, motor_error, persistent[2:11], persistent[11:14])
    ).astype(np.float32)
    delta = np.empty(4, dtype=np.float32)
    for row in range(4):
        accumulator = f32(0.0)
        for col in range(19):
            accumulator = f32(accumulator + f32(model["gain"][row, col]) * state[col])
        delta[row] = -accumulator
    raw = np.concatenate((collective_f + delta[:1], delta[1:])).astype(np.float32)
    lower_min = np.maximum(collective_f - f32(0.2), f32(0.0)).astype(np.float32)
    lower_max = np.minimum(collective_f + f32(0.2), f32(1.0)).astype(np.float32)
    output = np.concatenate(
        (np.clip(raw[:1], lower_min, lower_max), np.clip(raw[1:], f32(-1.0), f32(1.0)))
    ).astype(np.float32)
    saturated = bool(
        np.any(raw[:1] < lower_min)
        or np.any(raw[:1] > lower_max)
        or np.any(raw[1:] < -1.0)
        or np.any(raw[1:] > 1.0)
    )
    integral_error = np.asarray([error[0], error[1], rate_error[2]], dtype=np.float32)
    candidate = np.clip(
        persistent[11:14] + integral_error * model["dt"],
        -model["integral_limit"],
        model["integral_limit"],
    ).astype(np.float32)
    motor_command_dev = np.asarray(
        [
            collective_f - model["upper_trim"],
            output[0] - model["lower_trim"],
        ],
        dtype=np.float32,
    )
    motor_next = (
        model["motor_decay"] * persistent[:2]
        + model["motor_slope_eff"] * motor_command_dev
    ).astype(np.float32)
    if motor_rpm_valid:
        motor_next[:2] = (
            np.abs(motor_rpm).astype(np.float32) * model["rpm_to_rad_s"]
            - np.asarray(
                [model["upper_trim_rad_s"], model["lower_trim_rad_s"]],
                dtype=np.float32,
            )
        )
    latent_next = np.zeros(9, dtype=np.float32)
    for basis in range(3):
        for axis in range(3):
            value = model["latent_decay"][basis] * persistent[2 + 3 * basis + axis]
            for servo in range(3):
                value += (
                    model["latent_input"][basis, axis, servo] * output[1 + servo]
                )
            latent_next[3 * basis + axis] = f32(value)
    integral_next = persistent[11:14] if saturated else candidate
    next_persistent = np.concatenate(
        (motor_next, latent_next, integral_next)
    ).astype(np.float32)
    return output, next_persistent, collective_f


def composite_golden_vectors(
    model: dict[str, np.ndarray | np.float32],
) -> list[dict[str, object]]:
    cases = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0], 0.5, [0, 0], False, [0] * 14),
        ([0.9998, 0.01, 0, 0], [1, 0, 0, 0], [0.1, -0.2, 0.05], [0, 0, 0.3], 0.75,
         [11000, -11500], True, [0] * 14),
        ([0.9, 0.2, -0.3, 0.1], [0.98, -0.1, 0.12, 0.02], [-1, 0.8, 0.6], [0, 0, -0.5], 0.2,
         [9800, 10200], True, [100, -50, 0.05, -0.02, 0.03, 0.1, -0.2, 0.3, 0.01, -0.05, 0.07, 0.1, -0.2, 0.3]),
        ([-0.9998, -0.01, 0, 0], [1, 0, 0, 0], [0.3, 0.4, 0.2], [0, 0, 1.2], 1.0,
         [0, 0], False, [500, 400, 0.3, 0.25, 0.2, 0.2, 0.2, 0.4, -0.1, 0.15, -0.2, 0.2, 0.2, 0.4]),
        ([0.8, 0.4, 0.4, 0.2], [0.9239, 0, 0, 0.3827], [3, -3, 2], [0, 0, 0], 0.0,
         [13500, 9000], True, [-300, 200, -0.3, 0, 0.1, -0.25, 0.15, -0.45, 0.2, -0.1, 0.05, -0.25, 0.15, -0.45]),
    ]
    vectors = []
    for current_q, target_q, current_rate, target_rate, collective, rpm, valid, persistent in cases:
        arrays = [
            np.asarray(value, dtype=np.float32)
            for value in (current_q, target_q, current_rate, target_rate, rpm, persistent)
        ]
        output, next_state, upper = lqi_composite_step_float32(
            *arrays[:4], collective, arrays[4], valid, arrays[5], model
        )
        vectors.append(
            {
                "attitude_q_wb": arrays[0].tolist(),
                "target_attitude_q_wb": arrays[1].tolist(),
                "angular_velocity_b": arrays[2].tolist(),
                "target_angular_velocity_b": arrays[3].tolist(),
                "collective_base": float(collective),
                "motor_rpm": arrays[4].tolist(),
                "motor_rpm_valid": valid,
                "persistent": arrays[5].tolist(),
                "expected_output": output.tolist(),
                "expected_upper": float(upper),
                "expected_next_persistent": next_state.tolist(),
            }
        )
    return vectors


def composite_checksum(model: dict[str, np.ndarray | np.float32]) -> int:
    names = (
        "gain",
        "motor_decay",
        "motor_slope_eff",
        "latent_decay",
        "latent_input",
        "integral_limit",
        "motor_table",
        "dt",
        "upper_trim",
        "lower_trim",
        "upper_trim_rad_s",
        "lower_trim_rad_s",
        "rpm_to_rad_s",
        "max_roll_angle",
        "max_pitch_angle",
        "max_yaw_rate",
        "manual_command_decay",
    )
    return fnv1a32(
        b"".join(np.asarray(model[name], dtype=np.float32).tobytes() for name in names)
    )


def write_composite_model_header(
    model: dict[str, np.ndarray | np.float32],
) -> int:
    digest = composite_checksum(model)
    weight_bytes = int(
        sum(np.asarray(model[name]).size for name in (
            "gain",
            "motor_decay",
            "motor_slope_eff",
            "latent_decay",
            "latent_input",
            "integral_limit",
            "motor_table",
            "manual_command_decay",
        ))
        * 4
        + 9 * 4
    )
    lines = [
        "/****************************************************************************",
        " * Generated by px4_trans/tools/generate_lqi_backend.py. Do not edit.",
        " * Source: Identification/datasets/real_logs_26_8_12_v1/offline_identification_v7_hybrid.json",
        " ****************************************************************************/",
        "",
        "#pragma once",
        "",
        '#include "LqiCompositeControllerModel.hpp"',
        "",
        "namespace lqi_identified",
        "{",
        "",
    ]
    for name in (
        "gain",
        "motor_decay",
        "motor_slope_eff",
        "latent_decay",
        "latent_input",
        "integral_limit",
        "motor_table",
        "manual_command_decay",
    ):
        lines += array_lines(
            "k" + "".join(part.title() for part in name.split("_")),
            np.asarray(model[name]),
        )
        lines += [""]
    lines += [
        "static constexpr LqiCompositeControllerModel kModel {",
        "\tLqiCompositeControllerModel::kSchemaVersion,",
        "\tLqiCompositeControllerModel::WorldFrame::NED,",
        "\tLqiCompositeControllerModel::BodyFrame::FRD,",
        "\tLqiCompositeControllerModel::QuaternionConvention::HamiltonWxyzBodyToWorld,",
        "\tLqiCompositeControllerModel::CollectiveContract::ExternalUpperLqiLowerAndServos,",
        "\tLqiCompositeControllerModel::ActuatorObserver::MotorRpmOrCommandDriven,",
        "\tkGain,", "\tkMotorDecay,", "\tkMotorSlopeEff,", "\tkLatentDecay,", "\tkLatentInput,",
        "\tkIntegralLimit,", "\tkMotorTable,", "\t7u,",
        f"\t{literal(model['dt'])},", f"\t{literal(model['upper_trim'])},", f"\t{literal(model['lower_trim'])},",
        f"\t{literal(model['upper_trim_rad_s'])},", f"\t{literal(model['lower_trim_rad_s'])},", f"\t{literal(model['rpm_to_rad_s'])},",
        f"\t{literal(model['max_roll_angle'])},", f"\t{literal(model['max_pitch_angle'])},", f"\t{literal(model['max_yaw_rate'])},", "\tkManualCommandDecay,",
        f"\t{weight_bytes}u,", f"\t0x{digest:08x}u,", "};", "",
        "} // namespace lqi_identified", "",
    ]
    COMPOSITE_MODEL_HEADER.write_text("\n".join(lines), encoding="utf-8")
    return digest


def write_composite_golden(vectors: list[dict[str, object]]) -> None:
    fields = [
        "\tfloat attitude_q_wb[4];", "\tfloat target_attitude_q_wb[4];", "\tfloat angular_velocity_b[3];",
        "\tfloat target_angular_velocity_b[3];", "\tfloat collective_base;", "\tfloat motor_rpm[2];", "\tbool motor_rpm_valid;",
        "\tfloat persistent[14];", "\tfloat expected_output[4];", "\tfloat expected_upper;", "\tfloat expected_next_persistent[14];",
    ]
    lines = [
        "#pragma once", "",
        "struct LqiIdentifiedGoldenVector {",
    ] + fields + ["};", "", "static constexpr LqiIdentifiedGoldenVector kLqiIdentifiedGoldenVectors[] = {"]
    array_names = ("attitude_q_wb", "target_attitude_q_wb", "angular_velocity_b", "target_angular_velocity_b")
    for vector in vectors:
        lines.append("\t{")
        for name in array_names:
            lines.append("\t\t{" + ", ".join(literal(v) for v in vector[name]) + "},")
        lines.append(f"\t\t{literal(vector['collective_base'])},")
        lines.append("\t\t{" + ", ".join(literal(v) for v in vector["motor_rpm"]) + "},")
        lines.append("\t\t" + ("true," if vector["motor_rpm_valid"] else "false,"))
        for name in ("persistent", "expected_output"):
            lines.append("\t\t{" + ", ".join(literal(v) for v in vector[name]) + "},")
        lines.append(f"\t\t{literal(vector['expected_upper'])},")
        lines.append("\t\t{" + ", ".join(literal(v) for v in vector["expected_next_persistent"]) + "},")
        lines.append("\t},")
    lines += ["};", "", f"static constexpr int kLqiIdentifiedGoldenVectorCount{{{len(vectors)}}};", ""]
    COMPOSITE_GOLDEN_HEADER.write_text("\n".join(lines), encoding="utf-8")
    COMPOSITE_GOLDEN_JSON.write_text(
        json.dumps({"schema_version": 2, "vectors": vectors}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identified-json",
        type=Path,
        default=ROOT
        / "Identification/datasets/real_logs_26_8_12_v1/offline_identification_v7_hybrid.json",
        help="offline identification aggregate report (optional)",
    )
    parser.add_argument(
        "--identified-checkpoint",
        type=Path,
        default=ROOT
        / "Identification/runs/repro_20260806/sim2real_offline_logs_step_response_tcn_500hz_v7/identifier.pt",
        help="checkpoint carrying mode_transform/servo slopes/basis taus",
    )
    args = parser.parse_args()

    model = load_model()
    digest = write_model_header(model)
    vectors = golden_vectors(model)
    write_golden(vectors)
    print(f"wrote nominal 4x13 model (checksum 0x{digest:08x}) and {len(vectors)} golden vectors")

    identified_path = args.identified_json.expanduser().resolve()
    if identified_path.exists():
        identified = load_identified_model(
            identified_path, model, args.identified_checkpoint.expanduser().resolve()
        )
        composite_digest = write_composite_model_header(identified)
        composite_vectors = composite_golden_vectors(identified)
        write_composite_golden(composite_vectors)
        print(
            f"wrote identified 4x19 composite model (checksum 0x{composite_digest:08x}) "
            f"and {len(composite_vectors)} golden vectors"
        )


if __name__ == "__main__":
    main()
