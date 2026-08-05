#!/usr/bin/env python3
"""Generate the PX4 LQI backend weights header and golden test vectors.

The deployed controller is the Plan A external-collective 4-output LQI:
the upper rotor is owned by the pilot/RC throttle, and the fixed gain matrix
K4 (4x13) plus observer constants are embedded as float32 constexpr tables.

The Python reference in this module mirrors the C++ core operation order so
the golden vectors can be checked on the host with small float tolerances.

Run from the repository root:
    PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
    python px4_trans/tools/generate_lqi_backend.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_HEADER = (
    ROOT
    / "px4_trans/px4/src/modules/nn_control/LqiWeights.hpp"
)
GOLDEN_HEADER = ROOT / "px4_trans/tests/lqi_golden.hpp"
GOLDEN_JSON = ROOT / "px4_trans/tests/lqi_golden.json"

DT_S = 0.002
YAW_RATE_SCALE_RAD_S = 2.0


def _f32(value: float) -> np.float32:
    return np.float32(value)


def _load_controller() -> dict[str, np.ndarray]:
    from flight_identification.external_collective_lqi import (
        synthesize_external_gain,
    )

    sim = ROOT / "SimEnv/configs/sim2real_micro_coaxial.yaml"
    ctl = ROOT / "Controller/configs/lqi_sim2real_micro_coaxial.yaml"
    synth = synthesize_external_gain(sim, ctl)
    from flight_controller import load_controller_config

    cfg = load_controller_config(ctl)["params"]["pid"]["attitude"]
    return {
        "gain_4": synth["gain_4"].astype(np.float32),
        "upper_trim": _f32(synth["trim_upper_pwm"]),
        "lower_trim": _f32(synth["trim_lower_pwm"]),
        "slopes": synth["command_slopes"].astype(np.float32),
        "decay": np.exp(_f32(-DT_S) / synth["time_constants"].astype(np.float32)).astype(
            np.float32
        ),
        "slope_eff": (
            synth["command_slopes"].astype(np.float32)
            * (
                np.float32(1.0)
                - np.exp(
                    _f32(-DT_S)
                    / synth["time_constants"].astype(np.float32)
                ).astype(np.float32)
            )
        ).astype(np.float32),
        "integral_limit": np.asarray(
            cfg["integral_limit"], dtype=np.float32
        ),
        "dt": _f32(DT_S),
        "yaw_rate_scale": _f32(YAW_RATE_SCALE_RAD_S),
    }


def lqi_step_float32(
    attitude_q: np.ndarray,
    angular_velocity: np.ndarray,
    rc_throttle: float,
    rc_yaw: float,
    persistent: np.ndarray,
    weights: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Float32 mirror of LqiControllerCore::stepFromState.

    Returns (output4, next_persistent8, upper).
    """

    w = _f32(attitude_q[0])
    vx = _f32(attitude_q[1])
    vy = _f32(attitude_q[2])
    vz = _f32(attitude_q[3])
    if w < 0:
        w = -w
        vx = -vx
        vy = -vy
        vz = -vz
    vnorm = np.sqrt(_f32(vx * vx + vy * vy + vz * vz))
    vnorm_c = np.maximum(vnorm, _f32(1e-12))
    angle = _f32(2.0) * np.arctan2(vnorm, np.maximum(w, _f32(1e-12)))
    ratio = angle / vnorm_c
    e0 = vx * ratio
    e1 = vy * ratio

    yaw_target = _f32(rc_yaw) * weights["yaw_rate_scale"]
    r0 = _f32(angular_velocity[0])
    r1 = _f32(angular_velocity[1])
    r2 = _f32(angular_velocity[2]) - yaw_target

    x = np.concatenate(
        (
            np.asarray([e0, e1, r0, r1, r2], dtype=np.float32),
            persistent[:5],
            persistent[5:8],
        )
    ).astype(np.float32)
    delta = np.empty(4, dtype=np.float32)
    for row in range(4):
        acc = _f32(0.0)
        for col in range(13):
            acc = _f32(
                acc
                + _f32(weights["gain_4"][row, col]) * _f32(x[col])
            )
        delta[row] = -acc

    lower_raw = weights["lower_trim"] + delta[0]
    servos_raw = delta[1:4]
    lower = np.clip(lower_raw, _f32(0.0), _f32(1.0)).astype(np.float32)
    servos = np.clip(servos_raw, _f32(-1.0), _f32(1.0)).astype(np.float32)
    saturated = bool(
        lower_raw < 0.0
        or lower_raw > 1.0
        or float(servos_raw[0]) < -1.0
        or float(servos_raw[0]) > 1.0
        or float(servos_raw[1]) < -1.0
        or float(servos_raw[1]) > 1.0
        or float(servos_raw[2]) < -1.0
        or float(servos_raw[2]) > 1.0
    )

    errors = np.asarray([e0, e1, r2], dtype=np.float32)
    integral_next = np.empty(3, dtype=np.float32)
    for axis in range(3):
        candidate = np.clip(
            _f32(persistent[5 + axis]) + errors[axis] * weights["dt"],
            _f32(-weights["integral_limit"][axis]),
            _f32(weights["integral_limit"][axis]),
        ).astype(np.float32)
        integral_next[axis] = (
            _f32(persistent[5 + axis]) if saturated else candidate
        )

    upper = np.clip(
        _f32((_f32(rc_throttle) + _f32(1.0)) * _f32(0.5)),
        _f32(0.0),
        _f32(1.0),
    ).astype(np.float32)
    u_dev = np.asarray(
        [
            upper - weights["upper_trim"],
            lower - weights["lower_trim"],
            servos[0],
            servos[1],
            servos[2],
        ],
        dtype=np.float32,
    )
    actuator_next = (
        weights["decay"] * persistent[:5]
        + weights["slope_eff"] * u_dev
    ).astype(np.float32)
    next_persistent = np.concatenate((actuator_next, integral_next)).astype(
        np.float32
    )
    output = np.concatenate((lower[None], servos)).astype(np.float32)
    return output, next_persistent, upper


def _golden_vectors(weights: dict[str, np.ndarray]) -> list[dict[str, object]]:
    cases = [
        (
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            0.0,
            0.0,
            np.zeros(8, dtype=np.float32),
        ),
        (
            np.asarray([0.9998, 0.01, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.1, -0.2, 0.05], dtype=np.float32),
            0.5,
            0.25,
            np.zeros(8, dtype=np.float32),
        ),
        (
            np.asarray([0.9, 0.2, -0.3, 0.1], dtype=np.float32),
            np.asarray([-1.0, 0.8, 0.6], dtype=np.float32),
            -1.0,
            -0.5,
            np.asarray(
                [100.0, -50.0, 0.05, -0.02, 0.03, 0.1, -0.2, 0.3],
                dtype=np.float32,
            ),
        ),
        (
            np.asarray([-0.9998, -0.01, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.3, 0.4, 0.2], dtype=np.float32),
            1.0,
            1.0,
            np.asarray(
                [500.0, 400.0, 0.3, 0.25, 0.2, 0.2, 0.2, 0.4],
                dtype=np.float32,
            ),
        ),
        (
            np.asarray([0.8, 0.4, 0.4, 0.2], dtype=np.float32),
            np.asarray([3.0, -3.0, 2.0], dtype=np.float32),
            0.0,
            0.0,
            np.asarray(
                [-300.0, 200.0, -0.3, 0.0, 0.1, -0.25, 0.15, -0.45],
                dtype=np.float32,
            ),
        ),
        (
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            0.3,
            0.0,
            np.asarray(
                [700.0, 700.0, 0.34, 0.34, 0.34, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
    ]
    vectors = []
    for attitude_q, rate, throttle, yaw, persistent in cases:
        output, next_persistent, upper = lqi_step_float32(
            attitude_q, rate, throttle, yaw, persistent, weights
        )
        vectors.append(
            {
                "attitude_q": attitude_q.tolist(),
                "angular_velocity": rate.tolist(),
                "rc_throttle": float(throttle),
                "rc_yaw": float(yaw),
                "persistent": persistent.tolist(),
                "expected_output": output.tolist(),
                "expected_upper": float(upper),
                "expected_next_persistent": next_persistent.tolist(),
            }
        )
    return vectors


def _fnv1a32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _weights_checksum(weights: dict[str, np.ndarray]) -> int:
    payload = b"".join(
        weights[name].tobytes()
        for name in (
            "gain_4",
            "slopes",
            "decay",
            "slope_eff",
        )
    )
    payload += b"".join(
        np.asarray([weights[name]], dtype=np.float32).tobytes()
        for name in ("upper_trim", "lower_trim", "dt", "yaw_rate_scale")
    )
    payload += weights["integral_limit"].tobytes()
    return _fnv1a32(payload)


def _format_array(name: str, values: np.ndarray) -> list[str]:
    lines = [f"static constexpr float {name}[{values.size}] = {{"]
    for value in values.ravel():
        lines.append(f"\t{_float_literal(value)},")
    lines.append("};")
    return lines


def _float_literal(value: float) -> str:
    rendered = f"{float(value):.9g}"
    if "." not in rendered and "e" not in rendered and "E" not in rendered:
        rendered += ".0"
    return f"{rendered}f"


def write_weights_header(weights: dict[str, np.ndarray]) -> int:
    checksum = _weights_checksum(weights)
    lines = [
        "/****************************************************************************",
        " *",
        " *   Generated by px4_trans/tools/generate_lqi_backend.py",
        " *   Do not edit by hand; regenerate with the generator.",
        " *",
        " *   Plan A external-collective 4-output LQI weights.",
        " *   Upper rotor: external (RC throttle). Lower rotor + 3 servos: LQI.",
        " *   State: [roll_err, pitch_err, p, q, r, upper_f, lower_f, s1, s2, s3,",
        " *           i_roll, i_pitch, i_yaw_rate].",
        " *",
        " ****************************************************************************/",
        "",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        "namespace lqi_weights {",
        "",
        "static constexpr int kStateSize{13};",
        "static constexpr int kOutputSize{4};",
        "static constexpr int kPersistentSize{8};",
        "static constexpr float kDt{0.002f};",
        "static constexpr float kYawRateScaleRadS{2.0f};",
        "",
    ]
    lines += _format_array("kGain", weights["gain_4"].reshape(4, 13))
    lines.append("")
    lines += _format_array("kSlopes", weights["slopes"])
    lines.append("")
    lines += _format_array("kDecay", weights["decay"])
    lines.append("")
    lines += _format_array("kSlopeEff", weights["slope_eff"])
    lines.append("")
    lines += _format_array("kIntegralLimit", weights["integral_limit"])
    lines.append("")
    lines.append(
        "static constexpr float kUpperTrim{"
        + _float_literal(float(weights["upper_trim"]))
        + "};"
    )
    lines.append(
        "static constexpr float kLowerTrim{"
        + _float_literal(float(weights["lower_trim"]))
        + "};"
    )
    lines.append("")
    lines.append(f"static constexpr uint32_t kWeightBytes{{{296}}};")
    lines.append(f"static constexpr uint32_t kChecksum{{0x{checksum:08x}}};")
    lines.append("")
    lines.append("} // namespace lqi_weights")
    lines.append("")
    WEIGHTS_HEADER.write_text("\n".join(lines), encoding="utf-8")
    return checksum


def write_golden_files(vectors: list[dict[str, object]]) -> None:
    GOLDEN_HEADER.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "/****************************************************************************",
        " *",
        " *   Generated by px4_trans/tools/generate_lqi_backend.py",
        " *   Golden vectors for the host LQI core check (float tolerance).",
        " *",
        " ****************************************************************************/",
        "",
        "#pragma once",
        "",
        "struct LqiGoldenVector {",
        "\tfloat attitude_q[4];",
        "\tfloat angular_velocity[3];",
        "\tfloat rc_throttle;",
        "\tfloat rc_yaw;",
        "\tfloat persistent[8];",
        "\tfloat expected_output[4];",
        "\tfloat expected_upper;",
        "\tfloat expected_next_persistent[8];",
        "};",
        "",
        "static constexpr LqiGoldenVector kLqiGoldenVectors[] = {",
    ]
    for vector in vectors:
        att = ", ".join(_float_literal(value) for value in vector["attitude_q"])
        rate = ", ".join(
            _float_literal(value) for value in vector["angular_velocity"]
        )
        persistent = ", ".join(
            _float_literal(value) for value in vector["persistent"]
        )
        output = ", ".join(
            _float_literal(value) for value in vector["expected_output"]
        )
        next_persistent = ", ".join(
            _float_literal(value) for value in vector["expected_next_persistent"]
        )
        lines.append("\t{")
        lines.append(f"\t\t{{{att}}},")
        lines.append(f"\t\t{{{rate}}},")
        lines.append(f"\t\t{_float_literal(vector['rc_throttle'])}, ")
        lines.append(f"\t\t{_float_literal(vector['rc_yaw'])}, ")
        lines.append(f"\t\t{{{persistent}}},")
        lines.append(f"\t\t{{{output}}},")
        lines.append(f"\t\t{_float_literal(vector['expected_upper'])}, ")
        lines.append(f"\t\t{{{next_persistent}}},")
        lines.append("\t},")
    lines.append("};")
    lines.append("")
    lines.append(
        f"static constexpr int kLqiGoldenVectorCount{{{len(vectors)}}};"
    )
    lines.append("")
    GOLDEN_HEADER.write_text("\n".join(lines), encoding="utf-8")
    GOLDEN_JSON.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "px4_trans/tools/generate_lqi_backend.py",
                "vector_count": len(vectors),
                "vectors": vectors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate and verify the committed golden vectors match.",
    )
    args = parser.parse_args()
    weights = _load_controller()
    checksum = write_weights_header(weights)
    vectors = _golden_vectors(weights)
    write_golden_files(vectors)
    if args.check:
        previous = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
        if previous["vectors"] != vectors:
            raise SystemExit("golden vectors changed; inspect the diff")
        print(f"checksum 0x{checksum:08x}, {len(vectors)} golden vectors unchanged")
    else:
        print(f"wrote weights header (checksum 0x{checksum:08x}) and {len(vectors)} golden vectors")


if __name__ == "__main__":
    main()
