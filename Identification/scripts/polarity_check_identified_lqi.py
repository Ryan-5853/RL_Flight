#!/usr/bin/env python3
"""Closed-loop polarity check for the embedded identified 19-state LQI.

The firmware payload is generated from
``Identification/datasets/real_logs_26_8_12_v1/offline_identification_v7_hybrid.json``
by ``px4_trans/tools/generate_lqi_backend.py``.  This script rebuilds the
exact payload and runs the exact generator mirror of ``LqiCompositeCore``
against SimEnv:

* positive roll error must produce negative roll angular acceleration;
* positive pitch error must produce negative pitch angular acceleration;
* +/- roll runs must be antisymmetric in the first servo commands;
* the identified controller must agree in polarity with the known-good
  nominal 13-state LQI on the same initial condition;
* outputs must stay finite and inside actuator bounds.

Usage:
    PYTHONNOUSERSITE=1 PYTHONPATH=Identification/src:Controller/src:SimEnv/src \
      .venv/bin/python Identification/scripts/polarity_check_identified_lqi.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_generator_module() -> Any:
    path = ROOT / "px4_trans/tools/generate_lqi_backend.py"
    spec = importlib.util.spec_from_file_location("generate_lqi_backend", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["generate_lqi_backend"] = module
    spec.loader.exec_module(module)
    return module


def _quaternion(roll_rad: float, pitch_rad: float, yaw_rad: float = 0.0) -> np.ndarray:
    cr, sr = np.cos(0.5 * roll_rad), np.sin(0.5 * roll_rad)
    cp, sp = np.cos(0.5 * pitch_rad), np.sin(0.5 * pitch_rad)
    cy, sy = np.cos(0.5 * yaw_rad), np.sin(0.5 * yaw_rad)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


class CompositeLqiController:
    """Python mirror of the firmware LqiCompositeCore (persistent[14])."""

    def __init__(self, model: Mapping[str, Any]) -> None:
        self._model = model
        self._persistent = np.zeros(
            14, dtype=np.float32
        )

    def reset(self) -> None:
        self._persistent = np.zeros(14, dtype=np.float32)

    def step(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
        rate: np.ndarray,
        target_rate: np.ndarray,
        collective: float,
        motor_speed_rad_s: np.ndarray,
    ) -> np.ndarray:
        from generate_lqi_backend import lqi_composite_step_float32

        rpm = motor_speed_rad_s / float(self._model["rpm_to_rad_s"])
        output, next_persistent, upper = lqi_composite_step_float32(
            current_q,
            target_q,
            rate,
            target_rate,
            collective,
            rpm,
            True,
            self._persistent,
            self._model,
        )
        self._persistent = next_persistent
        return np.concatenate(([float(upper)], output)).astype(np.float32)


class NominalLqiController:
    """Python mirror of the firmware 13-state nominal LqiControllerCore."""

    def __init__(self, model: Mapping[str, Any]) -> None:
        self._model = model
        self._persistent = np.zeros(8, dtype=np.float32)

    def reset(self) -> None:
        self._persistent = np.zeros(8, dtype=np.float32)

    def step(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
        rate: np.ndarray,
        target_rate: np.ndarray,
        collective: float,
        motor_speed_rad_s: np.ndarray,
    ) -> np.ndarray:
        from generate_lqi_backend import lqi_step_float32

        rpm = motor_speed_rad_s / float(self._model["rpm_to_rad_s"])
        output, next_persistent, upper = lqi_step_float32(
            current_q,
            target_q,
            rate,
            target_rate,
            collective,
            rpm,
            True,
            self._persistent,
            self._model,
        )
        self._persistent = next_persistent
        return np.concatenate(([float(upper)], output)).astype(np.float32)


def _simulate(
    generator: Any,
    controller: Any,
    roll_deg: float,
    pitch_deg: float,
    collective: float,
    duration_s: float,
    seed: int = 20260812,
) -> dict[str, Any]:
    from simenv import SimulationEnvironment
    from simenv.config import load_and_materialize
    from flight_controller.plant import LocalPlantModel

    from flight_identification.experiment import _set_initial_observation_state

    device = torch.device("cpu")
    dtype = torch.float32
    materialized = load_and_materialize(
        ROOT / "SimEnv/configs/sim2real_micro_coaxial.yaml", 1, device, dtype
    )
    initial_state = dict(materialized.initial_state)
    initial_state["attitude_q_wb"] = torch.from_numpy(
        _quaternion(np.radians(roll_deg), np.radians(pitch_deg))
    )[None]
    initial_state["angular_velocity_b"] = torch.zeros(1, 3, dtype=dtype)
    environment = SimulationEnvironment(
        replace(
            materialized,
            initial_state=initial_state,
            sensor_state=dict(materialized.sensor_state),
        ),
        1,
        device,
        dtype,
        logging_enabled=False,
    )
    controller.reset()
    trim_speed = LocalPlantModel(materialized.parameters).hover_trim().motor_speed
    _set_initial_observation_state(
        environment,
        trim_speed,
        initial_state["angular_velocity_b"],
    )
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    target_rate = np.zeros(3, dtype=np.float32)
    active = torch.ones(1, dtype=torch.bool)
    records = {
        "roll_deg": [],
        "pitch_deg": [],
        "rate": [],
        "command": [],
        "time_s": [],
    }
    steps = round(duration_s * 500)
    try:
        for step in range(steps):
            truth = environment.observe("truth").values
            q = truth["attitude_q_wb"][0].numpy().astype(np.float32)
            rate = truth["angular_velocity_b"][0].numpy().astype(np.float32)
            motor_speed = truth["motor_speed"][0].numpy().astype(np.float32)
            command = controller.step(
                q, identity, rate, target_rate, collective, motor_speed
            )
            w, x, y, z = q
            roll = float(
                np.degrees(
                    np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                )
            )
            pitch = float(
                np.degrees(
                    np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
                )
            )
            records["roll_deg"].append(roll)
            records["pitch_deg"].append(pitch)
            records["rate"].append(rate.copy())
            records["command"].append(command.copy())
            records["time_s"].append(step / 500.0)
            result = environment.advance(
                torch.from_numpy(command)[None], active
            )
            active &= result.valid
    finally:
        environment.close()
    for key in records:
        records[key] = np.asarray(records[key])
    records["safe"] = bool(active.item())
    return records


def _initial_axis_response(
    records: Mapping[str, Any],
    axis: str,
) -> float:
    index = {"x": 0, "y": 1, "z": 2}[axis]
    rate = records["rate"]
    # Mean angular acceleration over the first 40 ms after commands engage.
    window = min(20, len(rate) - 1)
    return float((rate[window, index] - rate[0, index]) / records["time_s"][window])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--collective", type=float, default=0.45)
    parser.add_argument(
        "--identified-json",
        type=Path,
        default=ROOT
        / "Identification/datasets/real_logs_26_8_12_v1/offline_identification_v7_hybrid.json",
    )
    parser.add_argument(
        "--identified-checkpoint",
        type=Path,
        default=ROOT
        / "Identification/runs/repro_20260806/sim2real_offline_logs_step_response_tcn_500hz_v7/identifier.pt",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generator = _load_generator_module()
    nominal = generator.load_model()
    identified = generator.load_identified_model(
        args.identified_json, nominal, args.identified_checkpoint
    )
    identified_controller = CompositeLqiController(identified)
    nominal_controller = NominalLqiController(nominal)

    cases = [
        ("roll_plus", +12.0, 0.0),
        ("roll_minus", -12.0, 0.0),
        ("pitch_plus", 0.0, +10.0),
        ("pitch_minus", 0.0, -10.0),
    ]
    runs = {}
    for name, roll, pitch in cases:
        runs[name] = _simulate(
            generator, identified_controller, roll, pitch, args.collective, args.duration_s
        )
    runs["nominal_roll_plus"] = _simulate(
        generator, nominal_controller, +12.0, 0.0, args.collective, args.duration_s
    )

    results: dict[str, Any] = {}
    polarity_ok = True
    for name in ("roll_plus", "roll_minus", "pitch_plus", "pitch_minus"):
        records = runs[name]
        axis = "x" if name.startswith("roll") else "y"
        error_deg = {"roll_plus": 12.0, "roll_minus": -12.0, "pitch_plus": 10.0, "pitch_minus": -10.0}[name]
        accel = _initial_axis_response(records, axis)
        sign_ok = bool(np.sign(accel) == -np.sign(error_deg)) if accel != 0 else False
        polarity_ok &= sign_ok
        final_roll = float(records["roll_deg"][-1])
        final_pitch = float(records["pitch_deg"][-1])
        final_tilt = float(np.hypot(final_roll, final_pitch))
        peak_tilt = float(
            np.max(np.hypot(records["roll_deg"], records["pitch_deg"]))
        )
        finite = bool(
            np.isfinite(records["command"]).all()
            and (records["command"][:, :2] >= 0.0).all()
            and (records["command"][:, :2] <= 1.0).all()
            and (np.abs(records["command"][:, 2:]) <= 1.0).all()
        )
        polarity_ok &= finite
        results[name] = {
            "initial_error_deg": error_deg,
            "initial_axis_acceleration_rad_s2": accel,
            "polarity_sign_ok": sign_ok,
            "outputs_finite_and_bounded": finite,
            "peak_tilt_deg": peak_tilt,
            "final_tilt_deg": final_tilt,
            "safe": records["safe"],
        }

    roll_plus_cmd = runs["roll_plus"]["command"]
    roll_minus_cmd = runs["roll_minus"]["command"]
    # Servo commands must be antisymmetric for +/- roll.  The lower motor is
    # not an antisymmetric channel (it also balances yaw/collective coupling).
    antisymmetry = np.mean(
        np.abs((roll_plus_cmd[1:10, 2:] + roll_minus_cmd[1:10, 2:]))
    )
    nominal_accel = _initial_axis_response(runs["nominal_roll_plus"], "x")
    identified_accel = _initial_axis_response(runs["roll_plus"], "x")
    polarity_agreement = bool(
        np.sign(nominal_accel) == np.sign(identified_accel)
        and np.sign(identified_accel) == -1.0
    )
    polarity_ok &= polarity_agreement
    results["antisymmetry"] = {
        "mean_servo_cmd_sum_deg": float(antisymmetry),
        "ok": bool(antisymmetry < 0.02),
    }
    results["nominal_agreement"] = {
        "nominal_roll_accel_rad_s2": nominal_accel,
        "identified_roll_accel_rad_s2": identified_accel,
        "ok": polarity_agreement,
    }
    results["polarity_ok"] = polarity_ok

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0 if polarity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
