"""Adapter from PX4 ULog real flights to the canonical offline log bundle.

The canonical 14-channel schema is defined by the v4 offline composite
identifier (see Identification/OFFLINE_LOG_IDENTIFICATION_zh.md):

    attitude_q_tilt.w/x/y/z
    angular_velocity_b.x/y/z
    motor_speed.upper/lower
    command.motor_upper/lower
    command.servo_1/2/3

Real PX4 ULogs carry the equivalent data on different topics and units:

    vehicle_attitude.q[0..3]            -> attitude (Hamilton wxyz, body FRD -> NED)
    vehicle_angular_velocity.xyz[0..2]  -> body angular velocity [rad/s]
    esc_status.esc[*].esc_rpm           -> motor speed [rpm] -> [rad/s] (2*pi/60)
    actuator_motors.control[0..1]       -> normalized upper/lower motor command [0,1]
    actuator_servos.control[0..2]       -> normalized servo command [-1,1]

The firmware contract (px4_trans/src/modules/nn_control) fixes motor channel 0
to the upper rotor and channel 1 to the lower rotor, and matches ESC telemetry
by actuator function (MOTOR1=101, MOTOR2=102).  This adapter is intentionally
independent of arming/segmentation: it only resamples physics to a fixed 500 Hz
grid and marks samples that lack logged sensor/actuator coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pyulog
except ImportError:  # pragma: no cover - only used by the real-log CLI
    pyulog = None


SAMPLE_HZ = 500
SAMPLE_PERIOD_US = int(round(1e6 / SAMPLE_HZ))
MOTOR_RPM_TO_RAD_S = 2.0 * math.pi / 60.0
ACTUATOR_FUNCTION_MOTOR1 = 101
ACTUATOR_FUNCTION_MOTOR2 = 102

FEATURE_NAMES = (
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


@dataclass(frozen=True)
class ParsedFlight:
    source: Path
    timestamp_us: np.ndarray
    attitude_q_wb: np.ndarray
    angular_velocity_b: np.ndarray
    motor_speed: np.ndarray
    command: np.ndarray
    valid: np.ndarray
    sample_hz: int = SAMPLE_HZ

    @property
    def attitude_q_tilt(self) -> np.ndarray:
        return yaw_free_tilt_quaternion(self.attitude_q_wb)


def _require_pyulog() -> None:
    if pyulog is None:
        raise RuntimeError("pyulog is required to parse ULog files")


def _topic_map(log: object) -> dict[str, object]:
    return {entry.name: entry for entry in log.data_list}


def _ordered(ts: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ts.ndim != 1 or values.shape[0] != ts.shape[0]:
        raise ValueError("timestamp/value arrays must share the leading axis")
    if not (np.diff(ts) >= 0).all():
        order = np.argsort(ts, kind="stable")
        return ts[order], values[order]
    return ts, values


def _field_arrays(
    topic: object,
    fields: Sequence[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    timestamp = np.asarray(topic.data["timestamp"], dtype=np.int64)
    arrays = []
    for field in fields:
        values = np.asarray(topic.data[field], dtype=np.float64)
        if values.ndim != 1 or len(values) != len(timestamp):
            raise ValueError(f"field {topic.name}.{field} has unexpected shape")
        arrays.append(values)
    return timestamp, arrays


def _resample_hold(
    ts: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    max_gap_us: int,
) -> np.ndarray:
    """Zero-order-hold resampling with explicit gap invalidation (NaN)."""
    out = np.full(grid.shape, np.nan, dtype=np.float64)
    if len(ts) == 0:
        return out
    indices = np.searchsorted(ts, grid, side="right") - 1
    inside = (grid >= ts[0]) & (grid <= ts[-1])
    valid_index = indices >= 0
    use = inside & valid_index
    out[use] = values[np.clip(indices[use], 0, len(ts) - 1)]
    # A held value is only trustworthy when a fresh sample exists nearby.
    nearest = np.empty(grid.shape, dtype=np.int64)
    left = np.searchsorted(ts, grid, side="right") - 1
    right = np.searchsorted(ts, grid, side="left")
    left = np.clip(left, 0, len(ts) - 1)
    right = np.clip(right, 0, len(ts) - 1)
    nearest = np.where(
        np.abs(grid - ts[left]) <= np.abs(ts[right] - grid), left, right
    )
    stale = np.abs(grid - ts[nearest]) > max_gap_us
    out[stale] = np.nan
    return out


def _resample_linear(
    ts: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    max_gap_us: int,
) -> np.ndarray:
    out = np.full(grid.shape, np.nan, dtype=np.float64)
    if len(ts) == 0:
        return out
    out[:] = np.interp(grid, ts, values, left=np.nan, right=np.nan)
    nearest = np.empty(grid.shape, dtype=np.int64)
    left = np.searchsorted(ts, grid, side="right") - 1
    right = np.searchsorted(ts, grid, side="left")
    left = np.clip(left, 0, len(ts) - 1)
    right = np.clip(right, 0, len(ts) - 1)
    nearest = np.where(
        np.abs(grid - ts[left]) <= np.abs(ts[right] - grid), left, right
    )
    stale = np.abs(grid - ts[nearest]) > max_gap_us
    out[stale] = np.nan
    return out


def _esc_motor_speed(
    topic: object,
    grid: np.ndarray,
    max_gap_us: int,
) -> np.ndarray:
    """Return [N,2] rad/s motor speeds ordered [upper, lower]."""
    ts = np.asarray(topic.data["timestamp"], dtype=np.int64)
    esc_count_values = np.asarray(topic.data["esc_count"], dtype=np.int64)
    esc_count = int(esc_count_values[0]) if len(esc_count_values) else 0
    rpm_by_function: dict[int, np.ndarray] = {}
    for index in range(min(esc_count, 8)):
        function = np.asarray(
            topic.data.get(f"esc[{index}].actuator_function", np.array([0])),
            dtype=np.int64,
        )
        rpm = np.asarray(topic.data.get(f"esc[{index}].esc_rpm", np.array([])), dtype=np.float64)
        if len(rpm) != len(ts):
            continue
        if len(function) == 1:
            function = np.full(len(ts), int(function[0]), dtype=np.int64)
        if len(function) != len(ts):
            continue
        for function_id in (ACTUATOR_FUNCTION_MOTOR1, ACTUATOR_FUNCTION_MOTOR2):
            mask = function == function_id
            if mask.any():
                rpm_by_function.setdefault(function_id, np.full(len(ts), np.nan))
                rpm_by_function[function_id][mask] = rpm[mask]
    if len(rpm_by_function) < 2:
        # Fallback used only by non-standard logs: assume esc[0]/esc[1] order.
        for index, function_id in enumerate(
            (ACTUATOR_FUNCTION_MOTOR1, ACTUATOR_FUNCTION_MOTOR2)
        ):
            rpm = np.asarray(
                topic.data.get(f"esc[{index}].esc_rpm", np.array([])),
                dtype=np.float64,
            )
            if len(rpm) == len(ts):
                rpm_by_function[function_id] = rpm
    output = np.stack(
        [
            _resample_hold(ts, rpm_by_function[function_id], grid, max_gap_us)
            for function_id in (ACTUATOR_FUNCTION_MOTOR1, ACTUATOR_FUNCTION_MOTOR2)
            if function_id in rpm_by_function
        ],
        axis=1,
    ) * MOTOR_RPM_TO_RAD_S
    if output.shape[1] != 2:
        missing = [f for f in (ACTUATOR_FUNCTION_MOTOR1, ACTUATOR_FUNCTION_MOTOR2) if f not in rpm_by_function]
        raise ValueError(f"ESC telemetry missing motor functions: {missing}")
    return output


def _grid(log: object, topics: Sequence[object]) -> np.ndarray:
    first = min(int(np.asarray(topic.data["timestamp"])[0]) for topic in topics)
    last = max(int(np.asarray(topic.data["timestamp"])[-1]) for topic in topics)
    count = int(math.ceil((last - first) / SAMPLE_PERIOD_US)) + 1
    return first + np.arange(count, dtype=np.int64) * SAMPLE_PERIOD_US


def yaw_free_tilt_quaternion(attitude_q_wb: np.ndarray) -> np.ndarray:
    """De-yawed tilt quaternion matching Identification.experiment._yaw_free_attitude."""
    w, x, y, z = np.moveaxis(np.asarray(attitude_q_wb, dtype=np.float64), -1, 0)
    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    cosine_roll = np.cos(half_roll)
    sine_roll = np.sin(half_roll)
    cosine_pitch = np.cos(half_pitch)
    sine_pitch = np.sin(half_pitch)
    return np.stack(
        (
            cosine_roll * cosine_pitch,
            sine_roll * cosine_pitch,
            cosine_roll * sine_pitch,
            -sine_roll * sine_pitch,
        ),
        axis=-1,
    )


def parse_ulog(path: str | Path, sample_hz: int = SAMPLE_HZ) -> ParsedFlight:
    """Parse one ULog and resample it onto a fixed ``sample_hz`` grid."""
    _require_pyulog()
    if sample_hz != SAMPLE_HZ:
        raise ValueError("the canonical offline bundle is fixed at 500 Hz")
    source = Path(path).expanduser().resolve()
    log = pyulog.ULog(str(source))
    topics = _topic_map(log)
    required = {
        "actuator_motors",
        "actuator_servos",
        "esc_status",
        "vehicle_angular_velocity",
        "vehicle_attitude",
    }
    missing = required - set(topics)
    if missing:
        raise ValueError(f"{source.name} missing required topics: {sorted(missing)}")

    grid = _grid(log, [topics[name] for name in required])
    max_gap_us = 5 * SAMPLE_PERIOD_US

    att_ts, att_values = _field_arrays(
        topics["vehicle_attitude"], [f"q[{index}]" for index in range(4)]
    )
    attitude = np.stack(
        [
            _resample_linear(att_ts, channel, grid, max_gap_us)
            for channel in att_values
        ],
        axis=1,
    )
    norm = np.linalg.norm(attitude, axis=1, keepdims=True)
    attitude = attitude / np.maximum(norm, 1e-12)

    gyro_ts, gyro_values = _field_arrays(
        topics["vehicle_angular_velocity"], [f"xyz[{index}]" for index in range(3)]
    )
    angular_velocity = np.stack(
        [
            _resample_linear(gyro_ts, channel, grid, max_gap_us)
            for channel in gyro_values
        ],
        axis=1,
    )

    motor_speed = _esc_motor_speed(topics["esc_status"], grid, max_gap_us)

    motor_ts, motor_values = _field_arrays(
        topics["actuator_motors"], [f"control[{index}]" for index in range(2)]
    )
    servo_ts, servo_values = _field_arrays(
        topics["actuator_servos"], [f"control[{index}]" for index in range(3)]
    )
    command = np.concatenate(
        (
            np.stack(
                [
                    _resample_hold(motor_ts, channel, grid, max_gap_us)
                    for channel in motor_values
                ],
                axis=1,
            ),
            np.stack(
                [
                    _resample_hold(servo_ts, channel, grid, max_gap_us)
                    for channel in servo_values
                ],
                axis=1,
            ),
        ),
        axis=1,
    )

    channels = np.concatenate(
        (attitude, angular_velocity, motor_speed, command), axis=1
    )
    valid = np.isfinite(channels).all(axis=1)
    if not bool(valid.any()):
        raise ValueError(f"{source.name} contains no valid resampled sample")
    return ParsedFlight(
        source=source,
        timestamp_us=grid,
        attitude_q_wb=attitude,
        angular_velocity_b=angular_velocity,
        motor_speed=motor_speed,
        command=command,
        valid=valid,
        sample_hz=SAMPLE_HZ,
    )
