from __future__ import annotations

from pathlib import Path

import numpy as np

from flight_identification.real_log_adapter import (
    FEATURE_NAMES,
    MOTOR_RPM_TO_RAD_S,
    parse_ulog,
    yaw_free_tilt_quaternion,
)


LOG = Path(__file__).resolve().parents[1] / "real_log/26.8.12/log_14_UnknownDate.ulg"


def test_parse_ulog_schema() -> None:
    flight = parse_ulog(LOG)
    count = len(flight.timestamp_us)
    assert flight.attitude_q_wb.shape == (count, 4)
    assert flight.angular_velocity_b.shape == (count, 3)
    assert flight.motor_speed.shape == (count, 2)
    assert flight.command.shape == (count, 5)
    assert flight.valid.shape == (count,)
    assert flight.valid.mean() > 0.9
    assert flight.sample_hz == 500


def test_attitude_quaternion_unit_norm() -> None:
    flight = parse_ulog(LOG)
    norms = np.linalg.norm(flight.attitude_q_wb[flight.valid], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_motor_speed_esc_conversion() -> None:
    flight = parse_ulog(LOG)
    active = flight.command[:, 0] > 0.3
    # The adapter's rad/s values must be consistent with a fixed scale factor;
    # here we only verify they are finite and positive during active flight.
    assert bool(active.any())
    assert np.isfinite(flight.motor_speed[active]).all()
    assert (flight.motor_speed[active] > 0).all()
    assert MOTOR_RPM_TO_RAD_S == 2.0 * np.pi / 60.0


def test_yaw_free_tilt_matches_roll_pitch() -> None:
    flight = parse_ulog(LOG)
    tilt = yaw_free_tilt_quaternion(flight.attitude_q_wb[flight.valid])
    assert tilt.shape[1] == 4
    assert np.allclose(np.linalg.norm(tilt, axis=1), 1.0, atol=1e-6)
    w, x, y, z = tilt.T
    # w must equal cos(roll/2)*cos(pitch/2) for the de-yawed quaternion.
    q = flight.attitude_q_wb[flight.valid]
    roll = np.arctan2(
        2.0 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
        1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2),
    )
    pitch = np.arcsin(np.clip(2.0 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1, 1))
    expected_w = np.cos(0.5 * roll) * np.cos(0.5 * pitch)
    assert np.allclose(w, expected_w, atol=1e-5)


def test_feature_names_match_v7_schema() -> None:
    assert FEATURE_NAMES == (
        "attitude_q_tilt.w",
        "attitude_q_tilt.x",
        "attitude_q_tilt.y",
        "attitude_q_tilt.z",
        "angular_velocity_b.x",
        "angular_velocity_b.y",
        "angular_velocity_b.z",
        "motor_speed.upper",
        "motor_speed.lower",
        "command.motor_upper",
        "command.motor_lower",
        "command.servo_1",
        "command.servo_2",
        "command.servo_3",
    )
