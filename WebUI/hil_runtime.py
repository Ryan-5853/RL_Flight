"""PX4 hardware-in-the-loop controller backend for the existing WebUI.

The wire contract deliberately uses MAVLink common messages only.  SimEnv is
the plant, PX4 owns the controller, and the browser remains the pilot station.
All frames are NED/FRD and the attitude quaternion is Hamilton ``q_wb`` (body
FRD to world NED), which is the convention shared by SimEnv and PX4.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROTOCOL_VERSION = 6
EARTH_RADIUS_M = 6_378_137.0
HOME_LAT_DEG = 31.2304
HOME_LON_DEG = 121.4737
HOME_ALT_M = 20.0


class HilConfigurationError(ValueError):
    pass


class HilTransportError(RuntimeError):
    pass


def _finite_number(node: Mapping[str, Any], key: str, default: float) -> float:
    value = float(node.get(key, default))
    if not math.isfinite(value):
        raise HilConfigurationError(f"runtime.hil.{key} must be finite")
    return value


@dataclass(frozen=True)
class HilOptions:
    connection: str = "/dev/ttyACM0"
    baud: int = 2_000_000
    heartbeat_timeout_s: float = 5.0
    startup_timeout_s: float = 5.0
    actuator_timeout_s: float = 0.050
    pilot_timeout_s: float = 0.250
    auto_arm: bool = True
    auto_start_px4: bool = True
    pilot_source: str = "webui"
    px4_mavlink_device: str = "/dev/ttyACM0"
    home_lat_deg: float = HOME_LAT_DEG
    home_lon_deg: float = HOME_LON_DEG
    home_alt_m: float = HOME_ALT_M

    @classmethod
    def parse(cls, runtime: Mapping[str, Any]) -> "HilOptions":
        node = runtime.get("hil", {})
        if not isinstance(node, Mapping):
            raise HilConfigurationError("runtime.hil must be a mapping")
        auto_arm = node.get("auto_arm", True)
        if not isinstance(auto_arm, bool):
            raise HilConfigurationError("runtime.hil.auto_arm must be boolean")
        auto_start_px4 = node.get("auto_start_px4", True)
        if not isinstance(auto_start_px4, bool):
            raise HilConfigurationError(
                "runtime.hil.auto_start_px4 must be boolean"
            )
        options = cls(
            connection=str(node.get("connection", "/dev/ttyACM0")).strip(),
            baud=int(node.get("baud", 2_000_000)),
            heartbeat_timeout_s=_finite_number(
                node, "heartbeat_timeout_s", 5.0
            ),
            startup_timeout_s=_finite_number(
                node, "startup_timeout_s", 5.0
            ),
            actuator_timeout_s=(
                _finite_number(node, "actuator_timeout_ms", 50.0) / 1000.0
            ),
            pilot_timeout_s=(
                _finite_number(node, "pilot_timeout_ms", 250.0) / 1000.0
            ),
            auto_arm=auto_arm,
            auto_start_px4=auto_start_px4,
            pilot_source=str(node.get("pilot_source", "webui")).strip(),
            px4_mavlink_device=str(
                node.get("px4_mavlink_device", "/dev/ttyACM0")
            ).strip(),
            home_lat_deg=_finite_number(node, "home_lat_deg", HOME_LAT_DEG),
            home_lon_deg=_finite_number(node, "home_lon_deg", HOME_LON_DEG),
            home_alt_m=_finite_number(node, "home_alt_m", HOME_ALT_M),
        )
        if not options.connection:
            raise HilConfigurationError("runtime.hil.connection must not be empty")
        if options.pilot_source not in {"webui", "rc"}:
            raise HilConfigurationError(
                "runtime.hil.pilot_source must be webui or rc"
            )
        if options.auto_start_px4 and not options.px4_mavlink_device:
            raise HilConfigurationError(
                "runtime.hil.px4_mavlink_device must not be empty"
            )
        if options.auto_start_px4 and re.fullmatch(
            r"/dev/[A-Za-z0-9._-]+", options.px4_mavlink_device
        ) is None:
            raise HilConfigurationError(
                "runtime.hil.px4_mavlink_device must be a simple /dev path"
            )
        if not 9_600 <= options.baud <= 4_000_000:
            raise HilConfigurationError("runtime.hil.baud must be inside [9600,4000000]")
        if not 0.1 <= options.heartbeat_timeout_s <= 30.0:
            raise HilConfigurationError(
                "runtime.hil.heartbeat_timeout_s must be inside [0.1,30]"
            )
        if not 0.1 <= options.startup_timeout_s <= 10.0:
            raise HilConfigurationError(
                "runtime.hil.startup_timeout_s must be inside [0.1,10]"
            )
        if not 0.002 <= options.actuator_timeout_s <= 1.0:
            raise HilConfigurationError(
                "runtime.hil.actuator_timeout_ms must be inside [2,1000]"
            )
        if not 0.050 <= options.pilot_timeout_s <= 2.0:
            raise HilConfigurationError(
                "runtime.hil.pilot_timeout_ms must be inside [50,2000]"
            )
        if not -90.0 <= options.home_lat_deg <= 90.0:
            raise HilConfigurationError("runtime.hil.home_lat_deg is invalid")
        if not -180.0 <= options.home_lon_deg <= 180.0:
            raise HilConfigurationError("runtime.hil.home_lon_deg is invalid")
        return options


def ned_to_geodetic(
    position_n: Sequence[float],
    *,
    home_lat_deg: float,
    home_lon_deg: float,
    home_alt_m: float,
) -> tuple[int, int, int]:
    """Convert small local NED offsets to MAVLink integer global fields."""

    north, east, down = (float(value) for value in position_n)
    latitude = home_lat_deg + math.degrees(north / EARTH_RADIUS_M)
    longitude = home_lon_deg + math.degrees(
        east / (EARTH_RADIUS_M * max(math.cos(math.radians(home_lat_deg)), 1e-6))
    )
    altitude = home_alt_m - down
    return (
        round(latitude * 1e7),
        round(longitude * 1e7),
        round(altitude * 1000.0),
    )


def manual_control_fields(channels: Sequence[float]) -> tuple[int, int, int, int]:
    """Map WebUI ``roll,pitch,yaw,throttle`` [-1,1] to MANUAL_CONTROL."""

    roll, pitch, yaw, throttle = (
        min(max(float(value), -1.0), 1.0) for value in channels
    )
    # MAVLink x is stick-forward positive, while WebUI pitch is nose-up/stick-
    # back positive. Invert exactly once here; LqiManualReference then applies
    # PX4's stick-forward convention when it constructs the target attitude.
    # y=roll and r=yaw are right-positive in both contracts.
    return (
        round(-pitch * 1000.0),
        round(roll * 1000.0),
        round((throttle + 1.0) * 500.0),
        round(yaw * 1000.0),
    )


def actuator_command(message: Any) -> tuple[float, float, float, float, float]:
    values = tuple(float(value) for value in message.controls[:5])
    if len(values) != 5 or not all(math.isfinite(value) for value in values):
        raise HilTransportError("PX4 returned an invalid HIL actuator frame")
    if not all(0.0 <= value <= 1.0 for value in values[:2]):
        raise HilTransportError("PX4 motor outputs must be inside [0,1]")
    if not all(-1.0 <= value <= 1.0 for value in values[2:]):
        raise HilTransportError("PX4 servo outputs must be inside [-1,1]")
    return values


def rotate_world_to_body(
    q_wb: Sequence[float], vector_n: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate one NED vector into FRD using Hamilton q_wb."""

    w, x, y, z = (float(value) for value in q_wb)
    north, east, down = (float(value) for value in vector_n)
    # R_wb transpose, expanded to avoid allocations in the 50 Hz aux path.
    return (
        (1 - 2 * (y * y + z * z)) * north
        + 2 * (x * y + w * z) * east
        + 2 * (x * z - w * y) * down,
        2 * (x * y - w * z) * north
        + (1 - 2 * (x * x + z * z)) * east
        + 2 * (y * z + w * x) * down,
        2 * (x * z + w * y) * north
        + 2 * (y * z - w * x) * east
        + (1 - 2 * (x * x + y * y)) * down,
    )


class Px4HilController:
    """Controller-compatible MAVLink bridge used by ``CpuRuntimeSession``."""

    controller_type = "px4_hil"

    def __init__(
        self,
        options: HilOptions,
        *,
        torch_module: Any,
        device: Any,
        dtype: Any,
        control_dt: float,
        connection_factory: Callable[..., Any] | None = None,
        diagnostic_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.options = options
        self._torch = torch_module
        self._device = device
        self._dtype = dtype
        self._control_dt = float(control_dt)
        self._sample: tuple[Mapping[str, Any], Mapping[str, Any], Sequence[float]] | None = None
        self._origin_position_n: tuple[float, float, float] | None = None
        self._steps = 0
        self._timeouts = 0
        self._drained_frames = 0
        self._last_actuator_timestamp = -1
        self._last_rtt_ms = 0.0
        self._runtime_settle_until = 0.0
        self._last_mode = 0
        self._auto_arm_active = False
        self._last_arm_request = 0.0
        self._last_manual_mode_request = 0.0
        self._armed_by_us = False
        self._last_gcs_heartbeat = 0.0
        self._arming_feedback: list[str] = []
        self._hil_sensor_frames = 0
        self._hil_state_frames = 0
        self._hil_tx_sequence = 0
        self._bootstrap_commands: list[str] = []
        self._bootstrap_output = ""
        self._startup_trace: list[dict[str, Any]] = []
        self._startup_step = "initializing"
        self._startup_operation = "initialize PX4 HIL transport"
        self._last_completed_startup_step = "none"
        self._traced_live_operations: set[str] = set()
        self._diagnostic_path = (
            Path(diagnostic_path).expanduser().resolve()
            if diagnostic_path is not None
            else None
        )
        self._terminal_lock = threading.RLock()
        self._serial_io_lock = threading.RLock()

        if connection_factory is None:
            try:
                os.environ.setdefault("MAVLINK20", "1")
                from pymavlink import mavutil
                mavutil.set_dialect("common")
            except ImportError as error:
                raise HilConfigurationError(
                    "PX4 HIL requires pymavlink and pyserial"
                ) from error
            self._mavutil = mavutil
            connection_factory = mavutil.mavlink_connection
        else:
            # Tests may inject a complete MAVLink-compatible connection.
            try:
                from pymavlink import mavutil
            except ImportError:
                mavutil = None
            self._mavutil = mavutil

        self._trace_startup(
            "started",
            "connect",
            "open MAVLink transport and wait for PX4 heartbeat",
            connection=options.connection,
            baud=options.baud,
        )
        connect_started = time.perf_counter()
        try:
            self._connection = connection_factory(
                options.connection,
                baud=options.baud,
                source_system=255,
                source_component=190,
                autoreconnect=True,
                dialect="common",
            )
            heartbeat = self._connection.wait_heartbeat(
                timeout=options.heartbeat_timeout_s
            )
        except Exception as error:
            self._trace_startup(
                "failed",
                "connect",
                "open MAVLink transport and wait for PX4 heartbeat",
                elapsed_ms=(time.perf_counter() - connect_started) * 1000.0,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise HilTransportError(
                f"cannot connect to PX4 at {options.connection}: {error}; "
                f"diagnostic log: {self._diagnostic_path_text()}"
            ) from error
        if heartbeat is None:
            self._trace_startup(
                "failed",
                "connect",
                "open MAVLink transport and wait for PX4 heartbeat",
                elapsed_ms=(time.perf_counter() - connect_started) * 1000.0,
                error="heartbeat timeout",
                transport=self._transport_snapshot(),
            )
            self.close()
            raise HilTransportError(
                f"PX4 heartbeat timed out after {options.heartbeat_timeout_s:.1f}s; "
                f"diagnostic log: {self._diagnostic_path_text()}"
            )
        self._install_write_failure_detector()
        self._trace_startup(
            "completed",
            "connect",
            "open MAVLink transport and wait for PX4 heartbeat",
            elapsed_ms=(time.perf_counter() - connect_started) * 1000.0,
            transport=self._transport_snapshot(),
        )
        self._last_completed_startup_step = "connect"
        self.target_system = int(getattr(self._connection, "target_system", 1) or 1)
        self.target_component = int(
            getattr(self._connection, "target_component", 1) or 1
        )
        base_mode = int(getattr(heartbeat, "base_mode", 0))
        hil_flag = 32 if self._mavutil is None else int(
            self._mavutil.mavlink.MAV_MODE_FLAG_HIL_ENABLED
        )
        self._trace_startup(
            "started",
            "validate-hil-mode",
            "verify that the PX4 heartbeat has MAV_MODE_FLAG_HIL_ENABLED",
            base_mode=base_mode,
            hil_flag=hil_flag,
        )
        if not base_mode & hil_flag:
            self._trace_startup(
                "failed",
                "validate-hil-mode",
                "verify that the PX4 heartbeat has MAV_MODE_FLAG_HIL_ENABLED",
                error_type="HilTransportError",
                error=f"heartbeat base_mode={base_mode} does not contain HIL bit {hil_flag}",
                transport=self._transport_snapshot(),
            )
            self.close()
            raise HilTransportError(
                "PX4 is not in HIL mode; set SYS_HITL=1, save, and reboot; "
                f"diagnostic log: {self._diagnostic_path_text()}"
            )
        self._last_completed_startup_step = "validate-hil-mode"
        self._trace_startup(
            "completed",
            "validate-hil-mode",
            "verify that the PX4 heartbeat has MAV_MODE_FLAG_HIL_ENABLED",
            base_mode=base_mode,
            hil_flag=hil_flag,
        )
        try:
            if self.options.auto_start_px4:
                self._bootstrap_px4()
            self._run_startup_step(
                "message-rate",
                "request HIL_ACTUATOR_CONTROLS at the plant rate",
                self._request_actuator_rate,
                command="MAV_CMD_SET_MESSAGE_INTERVAL message_id=93 "
                f"interval_us={round(self._control_dt * 1e6)}",
            )
        except Exception:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
                self._connection = None
            raise

    def _diagnostic_path_text(self) -> str:
        return str(self._diagnostic_path) if self._diagnostic_path is not None else "console only"

    def _transport_snapshot(self) -> dict[str, Any]:
        connection = getattr(self, "_connection", None)
        port = getattr(connection, "port", None)
        snapshot: dict[str, Any] = {
            "connection": self.options.connection,
            "portdead": bool(getattr(connection, "portdead", False)),
        }
        if self.options.connection.startswith("/dev/"):
            snapshot["device_exists"] = os.path.exists(self.options.connection)
        if port is not None:
            is_open = getattr(port, "is_open", None)
            if is_open is not None:
                snapshot["port_is_open"] = bool(is_open)
        return snapshot

    def _trace_startup(
        self,
        status: str,
        step: str,
        operation: str,
        **details: Any,
    ) -> None:
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "monotonic_ns": time.monotonic_ns(),
            "status": status,
            "step": step,
            "operation": operation,
            **details,
        }
        self._startup_trace.append(event)
        self._startup_trace = self._startup_trace[-128:]
        detail_text = ""
        if status == "failed":
            detail_text = f": {details.get('error_type', 'error')}: {details.get('error', '')}"
        elif status == "progress":
            detail_text = f": {details.get('summary', 'in progress')}"
        elif details.get("received_message_type"):
            detail_text = (
                f": received {details['received_message_type']}"
                + (
                    f" mode={details['received_mode']}"
                    if "received_mode" in details
                    else ""
                )
            )
        print(
            f"[PX4 HIL startup] {status.upper()} {step}: {operation}{detail_text}",
            flush=True,
        )
        if self._diagnostic_path is None:
            return
        try:
            self._diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            with self._diagnostic_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
                stream.flush()
        except OSError as error:
            print(
                f"[PX4 HIL startup] WARNING cannot write diagnostic log "
                f"{self._diagnostic_path}: {error}",
                flush=True,
            )

    def _install_write_failure_detector(self) -> None:
        """Turn pymavlink's swallowed serial write failure into an exception.

        ``mavserial.write`` prints ``Device ... is dead`` and returns ``-1``;
        it does not raise. Wrapping the connection writer lets the active
        startup step record that failure instead of continuing with later
        commands and blaming the wrong operation.
        """

        connection = getattr(self, "_connection", None)
        writer = getattr(connection, "write", None)
        if not callable(writer):
            return

        def checked_write(payload: Any) -> Any:
            try:
                result = writer(payload)
            except Exception as error:
                raise HilTransportError(
                    f"MAVLink write raised {type(error).__name__}: {error}"
                ) from error
            if isinstance(result, int) and result < 0:
                raise HilTransportError(
                    "MAVLink serial write returned -1 after the device became unavailable"
                )
            return result

        connection.write = checked_write

    def _assert_transport_healthy(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is None:
            raise HilTransportError("MAVLink connection is closed")
        if bool(getattr(connection, "portdead", False)):
            raise HilTransportError(
                "pymavlink marked the serial device dead after a write failure"
            )
        port = getattr(connection, "port", None)
        if port is not None and getattr(port, "is_open", True) is False:
            raise HilTransportError("pyserial port is no longer open")
        if self.options.connection.startswith("/dev/") and not os.path.exists(
            self.options.connection
        ):
            raise HilTransportError(
                f"serial device {self.options.connection} disappeared"
            )

    def _startup_failure(
        self,
        step: str,
        operation: str,
        error: Exception,
        elapsed_ms: float,
        **details: Any,
    ) -> HilTransportError:
        snapshot = self._transport_snapshot()
        self._trace_startup(
            "failed",
            step,
            operation,
            elapsed_ms=elapsed_ms,
            error_type=type(error).__name__,
            error=str(error),
            last_completed_step=self._last_completed_startup_step,
            transport=snapshot,
            **details,
        )
        return HilTransportError(
            "PX4 HIL transport failure was first observed during "
            f"[{step}] {operation}; last completed step: "
            f"{self._last_completed_startup_step}; transport: {snapshot}; "
            f"underlying error: {type(error).__name__}: {error}; "
            f"diagnostic log: {self._diagnostic_path_text()}"
        )

    def _run_startup_step(
        self,
        step: str,
        operation: str,
        action: Callable[[], Any],
        *,
        fatal: bool = True,
        **details: Any,
    ) -> Any:
        self._startup_step = step
        self._startup_operation = operation
        self._trace_startup("started", step, operation, **details)
        started = time.perf_counter()
        try:
            result = action()
            self._assert_transport_healthy()
        except Exception as error:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            recoverable_capability_gap = (
                not fatal
                and isinstance(error, (AttributeError, NotImplementedError))
                and not bool(
                    getattr(getattr(self, "_connection", None), "portdead", False)
                )
            )
            if recoverable_capability_gap:
                self._trace_startup(
                    "warning",
                    step,
                    operation,
                    elapsed_ms=elapsed_ms,
                    error_type=type(error).__name__,
                    error=str(error),
                    **details,
                )
                return None
            raise self._startup_failure(
                step, operation, error, elapsed_ms, **details
            ) from error
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._last_completed_startup_step = step
        result_details = dict(result) if isinstance(result, Mapping) else {}
        if result is not None and callable(getattr(result, "get_type", None)):
            result_details["received_message_type"] = str(result.get_type())
            if hasattr(result, "mode"):
                result_details["received_mode"] = int(result.mode)
            if hasattr(result, "time_usec"):
                result_details["received_time_usec"] = int(result.time_usec)
        self._trace_startup(
            "completed",
            step,
            operation,
            elapsed_ms=elapsed_ms,
            transport=self._transport_snapshot(),
            **details,
            **result_details,
        )
        return result

    def _run_live_operation(
        self,
        step: str,
        operation: str,
        action: Callable[[], Any],
        **details: Any,
    ) -> Any:
        """Trace the first live invocation and every transport failure.

        Successful 500 Hz operations are logged once to avoid perturbing HIL
        timing. The current operation is still updated on every invocation, so
        a later USB loss is attributed to the exact message or arm request
        that first observed it.
        """

        if step not in self._traced_live_operations:
            self._traced_live_operations.add(step)
            return self._run_startup_step(step, operation, action, **details)

        if self._steps > 0:
            return action()

        self._startup_step = step
        self._startup_operation = operation
        started = time.perf_counter()
        try:
            return action()
        except Exception as error:
            raise self._startup_failure(
                step,
                operation,
                error,
                (time.perf_counter() - started) * 1000.0,
                **details,
            ) from error

    def _send_shell_text(self, value: str, *, respond: bool = True) -> None:
        try:
            encoded = value.encode("ascii")
        except UnicodeEncodeError as error:
            raise HilConfigurationError(
                "PX4 shell bootstrap commands must be ASCII"
            ) from error
        mavlink = self._mavutil.mavlink if self._mavutil is not None else None
        device = 10 if mavlink is None else int(mavlink.SERIAL_CONTROL_DEV_SHELL)
        respond_flag = 2 if mavlink is None else int(
            mavlink.SERIAL_CONTROL_FLAG_RESPOND
        )
        exclusive_flag = 4 if mavlink is None else int(
            mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE
        )
        flags = (respond_flag | exclusive_flag) if respond else 0
        if not encoded:
            self._connection.mav.serial_control_send(
                device, flags, 0, 0, 0, [0] * 70
            )
            self._assert_transport_healthy()
            return
        for offset in range(0, len(encoded), 70):
            chunk = encoded[offset : offset + 70]
            data = list(chunk) + [0] * (70 - len(chunk))
            self._connection.mav.serial_control_send(
                device, flags, 0, 0, len(chunk), data
            )
            self._assert_transport_healthy()

    def _drain_shell_output(self) -> None:
        fragments: list[str] = []
        while True:
            message = self._recv_match(
                type="SERIAL_CONTROL", blocking=False
            )
            if message is None:
                break
            count = int(getattr(message, "count", 0))
            data = bytes(getattr(message, "data", [])[:count])
            fragments.append(data.decode("utf-8", errors="replace"))
        if fragments:
            self._bootstrap_output = (
                self._bootstrap_output + "".join(fragments)
            )[-16384:]

    def _recv_match(self, *args: Any, **kwargs: Any) -> Any:
        """Serialize pyserial reads shared by the HIL loop and terminal."""
        with self._serial_io_lock:
            return self._connection.recv_match(*args, **kwargs)

    def terminal_command(self, command: str, *, timeout_s: float = 1.5) -> str:
        """Run one bounded PX4 NSH command through the existing MAVLink link."""
        command = command.strip()
        if not command:
            return ""
        if len(command) > 240 or any(ch in command for ch in ("\x00", "\r", "\n")):
            raise HilConfigurationError("terminal command must be one line of at most 240 characters")
        timeout_s = max(0.1, min(float(timeout_s), 5.0))
        with self._terminal_lock, self._serial_io_lock:
            self._send_shell_text("\n")
            self._send_shell_text(command + "\n")
            deadline = time.monotonic() + timeout_s
            output = ""
            while time.monotonic() < deadline:
                before = self._bootstrap_output
                self._drain_shell_output()
                output += self._bootstrap_output[len(before):]
                time.sleep(0.05)
            before = self._bootstrap_output
            self._drain_shell_output()
            output += self._bootstrap_output[len(before):]
            self._send_shell_text("", respond=False)
            return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", output[-16384:])

    def _bootstrap_px4(self) -> None:
        """Configure and start the PX4 HIL chain through MAVLink NSH."""

        rc_input_mode = 1 if self.options.pilot_source == "webui" else 0
        # Set actuator-function parameters through MAVLink as well as NSH.
        # SERIAL_CONTROL can be delayed while PX4 is restarting modules; a
        # direct PARAM_SET makes the mapping deterministic across restarts.
        parameters = {
            "NN_LQI_OUTPUT_EN": 1,
            "COM_RC_IN_MODE": rc_input_mode,
            # This HIL round injects truth attitude/state directly and
            # bypasses EKF2, so disable the estimator and allow arming
            # without GPS/home position. Restore EKF2_EN=1 before real
            # flight.
            "EKF2_EN": 0,
            "COM_ARM_WO_GPS": 1,
            # Force the simple single-IMU sensor path and lift the gyro rate
            # cap so the simulated 500 Hz IMU stream is not throttled to the
            # 400 Hz default (and worse after repeated sessions).
            "SENS_IMU_MODE": 1,
            "IMU_GYRO_RATEMAX": 1000,
            # HIL actuator direction must exactly match the SimEnv/controller
            # sign convention.  QGC actuator calibration can persist a
            # non-zero HIL_ACT_REV (especially after setting servo reverse for
            # the real airframe), which negates the HIL channel values in
            # MixingOutput and turns the closed loop into positive feedback.
            "HIL_ACT_REV": 0,
            "HIL_ACT_FUNC1": 101,
            "HIL_ACT_FUNC2": 102,
            "HIL_ACT_FUNC3": 201,
            "HIL_ACT_FUNC4": 202,
            "HIL_ACT_FUNC5": 203,
        }
        for index, (name, value) in enumerate(parameters.items(), start=1):
            def send_parameter(name: str = name, value: int = value) -> None:
                self._connection.mav.param_set_send(
                    self.target_system,
                    self.target_component,
                    name.encode("ascii"),
                    float(value),
                    9,  # MAV_PARAM_TYPE_REAL32
                )
            # Test shims and unusual MAVLink endpoints may not implement
            # PARAM_SET; the NSH commands below remain the fallback. A dead
            # serial transport is still fatal and is never downgraded.
            self._run_startup_step(
                f"param-set.{index:02d}",
                f"set PX4 parameter {name}={value} through MAVLink",
                send_parameter,
                fatal=False,
                parameter=name,
                value=value,
            )

        def force_disarm() -> None:
            # A previous WebUI session can leave PX4 armed (e.g. after an
            # abrupt server exit while the simulated vehicle was "not
            # landed").  pwm_out_sim must be (re)started while disarmed or
            # MixingOutput will not load the HIL_ACT_FUNC channel map.
            self._send_arm_command(False)
            time.sleep(0.20)

        self._run_startup_step(
            "disarm-force",
            "force-disarm PX4 before restarting the HIL module chain",
            force_disarm,
            command="MAV_CMD_COMPONENT_ARM_DISARM arm=0 force=21196",
        )

        commands = [
            "param show SYS_HITL",
            # A previous WebUI process can leave callback-driven modules alive
            # but dormant after HIL input disappears.  Recreate both ends of
            # the actuator uORB chain so a new host session never inherits an
            # old actuator_outputs_sim timestamp.
            "commander disarm",
            "nn_control stop",
            "pwm_out_sim stop",
            # Restart the sensor pipeline each session. Repeated HIL sessions
            # can leave VehicleIMU/VehicleAngularVelocity callbacks degraded
            # even though HIL_SENSOR still arrives at the MAVLink layer.
            "sensors stop",
            "param set SENS_IMU_MODE 1",
            "param set IMU_GYRO_RATEMAX 1000",
            "sensors start -h",
            "param set NN_LQI_OUTPUT_EN 1",
            f"param set COM_RC_IN_MODE {rc_input_mode}",
            "param set HIL_ACT_FUNC1 101",
            "param set HIL_ACT_FUNC2 102",
            "param set HIL_ACT_FUNC3 201",
            "param set HIL_ACT_FUNC4 202",
            "param set HIL_ACT_FUNC5 203",
            "param set HIL_ACT_REV 0",
            "param set EKF2_EN 0",
            "param set COM_ARM_WO_GPS 1",
            "param show HIL_ACT_FUNC1",
            "param show HIL_ACT_REV",
            "pwm_out_sim start -m hil",
            "nn_control start",
            "pwm_out_sim status",
            "nn_control status",
        ]
        operation_by_command = {
            "param show SYS_HITL": "read and verify the PX4 HIL-mode parameter",
            "commander disarm": "disarm PX4 before changing the HIL module chain",
            "nn_control stop": "stop the nn_control callback worker",
            "pwm_out_sim stop": "stop the simulated actuator-output driver",
            "sensors stop": "stop the IMU/sensor pipeline before a clean HIL restart",
            "sensors start -h": "start the HIL sensor pipeline with a fresh IMU voter",
            "param set NN_LQI_OUTPUT_EN 1": "enable LQI actuator output",
            f"param set COM_RC_IN_MODE {rc_input_mode}": (
                "select the WebUI pilot source"
                if rc_input_mode == 1
                else "select the physical RC pilot source"
            ),
            "param set HIL_ACT_FUNC1 101": "map HIL output 1 to motor 1",
            "param set HIL_ACT_FUNC2 102": "map HIL output 2 to motor 2",
            "param set HIL_ACT_FUNC3 201": "map HIL output 3 to servo 1",
            "param set HIL_ACT_FUNC4 202": "map HIL output 4 to servo 2",
            "param set HIL_ACT_FUNC5 203": "map HIL output 5 to servo 3",
            "param set HIL_ACT_REV 0": "force neutral HIL actuator direction (no reversed channels)",
            "param set EKF2_EN 0": "disable EKF2 for the direct-state HIL round",
            "param set COM_ARM_WO_GPS 1": "allow arming without GPS/home position in HIL",
            "param set SENS_IMU_MODE 1": "force the single-IMU sensor voter for HIL",
            "param set IMU_GYRO_RATEMAX 1000": "allow the simulated gyro stream up to 1000 Hz",
            "param show HIL_ACT_FUNC1": "read back the first HIL actuator mapping",
            "param show HIL_ACT_REV": "read back the HIL actuator reverse mask",
            "pwm_out_sim start -m hil": "start the simulated actuator-output driver in HIL mode",
            "nn_control start": "start the nn_control callback worker",
            "pwm_out_sim status": "read simulated actuator-output driver status",
            "nn_control status": "read nn_control status and timing counters",
        }
        self._bootstrap_commands = list(commands)

        def acquire_shell() -> None:
            self._send_shell_text("\n")
            time.sleep(0.10)
            self._drain_shell_output()

        self._run_startup_step(
            "shell-acquire",
            "acquire the PX4 NSH shell through exclusive MAVLink SERIAL_CONTROL",
            acquire_shell,
            command="SERIAL_CONTROL_DEV_SHELL flags=RESPOND|EXCLUSIVE",
        )

        for index, command in enumerate(commands, start=1):
            def run_shell_command(command: str = command) -> dict[str, Any]:
                output_before = self._bootstrap_output
                self._send_shell_text(command + "\n")
                # SERIAL_CONTROL is a reliable shell transport, but PX4's
                # NSH parser can still lag while modules are stopping.  A
                # short per-command settling interval prevents param-set and
                # module-start commands from being dropped on restart.
                if command.endswith(" stop"):
                    time.sleep(0.20)
                elif command.startswith("param set"):
                    time.sleep(0.16)
                else:
                    time.sleep(0.10)
                self._drain_shell_output()
                output_after = self._bootstrap_output
                output_delta = (
                    output_after[len(output_before):]
                    if output_after.startswith(output_before)
                    else output_after[-1000:]
                )
                return {"shell_output_tail": output_delta[-1000:]}

            self._run_startup_step(
                f"shell-command.{index:02d}",
                operation_by_command.get(command, f"execute PX4 NSH command {command}"),
                run_shell_command,
                command=command,
            )

        def release_shell() -> None:
            time.sleep(0.10)
            self._drain_shell_output()
            # A zero-length message without RESPOND releases the PX4 shell.
            self._send_shell_text("", respond=False)

        self._run_startup_step(
            "shell-release",
            "release the exclusive PX4 NSH SERIAL_CONTROL session",
            release_shell,
            command="SERIAL_CONTROL count=0 flags=0",
        )

    def _request_actuator_rate(self) -> None:
        """Ask PX4 for one actuator frame per 2 ms plant boundary."""

        command = 511 if self._mavutil is None else int(
            self._mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
        )
        message_id = 93  # MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS
        self._connection.mav.command_long_send(
            self.target_system,
            self.target_component,
            command,
            0,
            float(message_id),
            round(self._control_dt * 1e6),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def set_sample(
        self,
        truth: Mapping[str, Any],
        sensor: Mapping[str, Any],
        channels: Any,
    ) -> None:
        cpu_channels = channels[0].detach().cpu().tolist()
        self._sample = (truth, sensor, cpu_channels)

    @staticmethod
    def _row(values: Any) -> list[float]:
        return [float(value) for value in values[0].detach().cpu().tolist()]

    def _send_gcs_heartbeat(self, now: float) -> None:
        if now - self._last_gcs_heartbeat < 1.0:
            return
        mavlink = self._mavutil.mavlink if self._mavutil is not None else None
        self._connection.mav.heartbeat_send(
            6 if mavlink is None else mavlink.MAV_TYPE_GCS,
            8 if mavlink is None else mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            4 if mavlink is None else mavlink.MAV_STATE_ACTIVE,
        )
        self._last_gcs_heartbeat = now

    def _send_hil_sensor(
        self,
        timestamp_us: int,
        quaternion: Sequence[float],
        down_m: float,
        angular_velocity: Sequence[float],
        specific_force: Sequence[float],
        *,
        include_aux: bool,
    ) -> None:
        """Send the primary IMU and optionally low-rate auxiliary sensors.

        MAVLink HIL_SENSOR accelerometer fields are SI m/s^2, unlike the mG
        integer fields in HIL_STATE_QUATERNION.  Keeping this message at the
        500 Hz plant rate gives PX4's normal VehicleIMU/sensors pipeline the
        sample-rate information required to select the simulated gyro.
        """

        magnetic_b = (0.0, 0.0, 0.0)
        altitude_m = 0.0
        pressure_hpa = 0.0
        temperature_c = 0.0
        fields_updated = 0b111 | 0b111000
        if include_aux:
            # Representative Shanghai NED magnetic field in gauss. Rotate it
            # to FRD so the sample stays coherent with q_wb as the vehicle
            # moves. Barometric altitude is positive upward.
            magnetic_b = rotate_world_to_body(
                quaternion, (0.22, 0.01, 0.43)
            )
            altitude_m = self.options.home_alt_m - float(down_m)
            pressure_hpa = 1013.25 * max(
                1.0 - 2.25577e-5 * altitude_m, 0.01
            ) ** 5.25588
            temperature_c = 20.0
            fields_updated |= 0b111000000 | 0b1101000000000
        self._connection.mav.hil_sensor_send(
            timestamp_us,
            *specific_force,
            *angular_velocity,
            *magnetic_b,
            pressure_hpa,
            0.0,
            altitude_m,
            temperature_c,
            fields_updated,
        )
        self._hil_sensor_frames += 1

    def _collect_timeout_diagnostics(
        self, keepalive: Callable[[], Any] | None = None
    ) -> None:
        """Capture PX4 state while keeping the injected IMU stream live."""

        if not self.options.auto_start_px4:
            return
        commands = [
            # Include the transport counters in the same WebUI-owned shell
            # session; users do not need a second QGC/serial console while
            # HIL is running.
            "mavlink status",
            "sensors status",
            "commander check",
            "commander status",
            "nn_control status",
            "pwm_out_sim status",
            "listener actuator_outputs_sim -n 1",
            "listener manual_control_setpoint -n 1",
            "listener vehicle_status -n 1",
            "listener health_report -n 1",
            "listener failsafe_flags -n 1",
        ]

        def wait_for_shell(duration_s: float) -> None:
            deadline = time.perf_counter() + duration_s
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return
                if keepalive is not None:
                    keepalive()
                time.sleep(min(max(self._control_dt, 0.001), remaining))

        diagnostic_step = "timeout-diagnostics.shell-acquire"
        diagnostic_operation = "acquire PX4 NSH after actuator startup timeout"
        self._trace_startup(
            "started", diagnostic_step, diagnostic_operation,
            transport=self._transport_snapshot(),
        )
        try:
            self._bootstrap_output = (
                self._bootstrap_output + "\n--- live timeout diagnostics ---\n"
            )[-16384:]
            self._send_shell_text("\n")
            wait_for_shell(0.08)
            self._drain_shell_output()
            self._trace_startup(
                "completed", diagnostic_step, diagnostic_operation,
                transport=self._transport_snapshot(),
            )
            for index, command in enumerate(commands, 1):
                diagnostic_step = f"timeout-diagnostics.shell-command.{index:02d}"
                diagnostic_operation = (
                    f"run PX4 timeout diagnostic command: {command}"
                )
                before = self._bootstrap_output
                self._trace_startup(
                    "started", diagnostic_step, diagnostic_operation,
                    command=command,
                )
                self._send_shell_text(command + "\n")
                wait_for_shell(0.08)
                self._drain_shell_output()
                self._drain_startup_feedback()
                output = self._bootstrap_output[len(before):]
                self._trace_startup(
                    "completed", diagnostic_step, diagnostic_operation,
                    command=command,
                    shell_output_tail=output[-1000:],
                    transport=self._transport_snapshot(),
                )
            wait_for_shell(0.08)
            self._drain_shell_output()
        except Exception as error:
            self._bootstrap_output = (
                self._bootstrap_output
                + f"\npost-timeout diagnostics failed: {error}\n"
            )[-16384:]
            self._trace_startup(
                "failed", diagnostic_step, diagnostic_operation,
                error_type=type(error).__name__,
                error=str(error),
                transport=self._transport_snapshot(),
            )
        finally:
            release_step = "timeout-diagnostics.shell-release"
            release_operation = "release PX4 NSH after actuator timeout diagnostics"
            self._trace_startup(
                "started", release_step, release_operation,
                transport=self._transport_snapshot(),
            )
            try:
                self._send_shell_text("", respond=False)
                self._trace_startup(
                    "completed", release_step, release_operation,
                    transport=self._transport_snapshot(),
                )
            except Exception as error:
                self._trace_startup(
                    "failed", release_step, release_operation,
                    error_type=type(error).__name__,
                    error=str(error),
                    transport=self._transport_snapshot(),
                )

    def _send_arm_command(self, arm: bool) -> None:
        command = 400 if self._mavutil is None else int(
            self._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
        )
        # Force disarming (21196) so a HIL session can always recover from a
        # previous abrupt stop.  PX4 otherwise refuses to disarm while
        # "not landed", which leaves the next session armed; MixingOutput then
        # refuses to load HIL_ACT_FUNC* channel mappings while armed and the
        # whole actuator chain stays silent.
        force = 21196.0 if not arm else 0.0
        self._connection.mav.command_long_send(
            self.target_system,
            self.target_component,
            command,
            0,
            1.0 if arm else 0.0,
            force,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        if not arm:
            self._armed_by_us = False

    def _send_manual_mode_command(self) -> None:
        """Select PX4 Manual as the safety shell around direct LQI control.

        ``nn_control`` consumes manual input itself and does not register a PX4
        external mode.  A mode left at External 1 by QGC or a previous session
        is therefore unarmable by design.  Request Manual only after HIL has
        begun injecting the safe, fresh manual-control sample.
        """

        command = 176 if self._mavutil is None else int(
            self._mavutil.mavlink.MAV_CMD_DO_SET_MODE
        )
        if self._mavutil is None:
            # Keep the HIL bit set when using the lightweight test MAVLink
            # shim.  PX4 gates HIL_SENSOR reception on the MAVLink link's
            # HIL-enabled flag, so selecting Manual must not clear it.
            custom_mode_enabled = 1 | 32
        else:
            custom_mode_enabled = int(
                self._mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            ) | int(self._mavutil.mavlink.MAV_MODE_FLAG_HIL_ENABLED)
        px4_custom_main_mode_manual = 1
        self._connection.mav.command_long_send(
            self.target_system,
            self.target_component,
            command,
            0,
            float(custom_mode_enabled),
            float(px4_custom_main_mode_manual),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def _drain_startup_feedback(self) -> None:
        while True:
            message = self._recv_match(
                type=["COMMAND_ACK", "STATUSTEXT"], blocking=False
            )
            if message is None:
                break
            self._capture_arming_feedback(message)

    def _capture_arming_feedback(self, message: Any) -> None:
        """Retain bounded COMMAND_ACK/STATUSTEXT details during startup."""

        message_type = str(message.get_type())
        detail = ""
        command = int(getattr(message, "command", -1))
        if message_type == "COMMAND_ACK" and command in {176, 400}:
            result = int(getattr(message, "result", -1))
            result_names = {
                0: "accepted",
                1: "temporarily rejected",
                2: "denied",
                3: "unsupported",
                4: "failed",
                5: "in progress",
                6: "cancelled",
            }
            detail = (
                ("arm" if command == 400 else "manual-mode")
                + " COMMAND_ACK: "
                f"{result_names.get(result, 'unknown')} ({result})"
            )
            result_param2 = int(getattr(message, "result_param2", 0))
            if result_param2:
                detail += f", reason {result_param2}"
        elif message_type == "STATUSTEXT":
            text = str(getattr(message, "text", "")).strip("\x00 ")
            diagnostic_terms = (
                "arm",
                "throttle",
                "preflight",
                "health",
                "mode",
                "manual control",
                "sensor",
            )
            if text and any(term in text.lower() for term in diagnostic_terms):
                detail = f"PX4: {text}"
        if detail and (not self._arming_feedback or self._arming_feedback[-1] != detail):
            self._arming_feedback.append(detail)
            self._arming_feedback = self._arming_feedback[-8:]

    def start(self) -> None:
        # Without the bootstrap sequence the operator needs the terminal to
        # configure PX4 first.  Do not immediately fail that session by
        # attempting automatic mode switching/arming against an unconfigured
        # vehicle; arming can be issued manually from the terminal afterward.
        self._auto_arm_active = self.options.auto_arm and self.options.auto_start_px4
        self._last_arm_request = 0.0
        self._last_manual_mode_request = 0.0
        self._arming_feedback.clear()
        self._run_live_operation(
            "simulation-start",
            "accept WebUI start and enable the automatic HIL arming sequence",
            lambda: None,
            auto_arm_active=self._auto_arm_active,
        )

    def _request_manual_mode_then_arm(self, now: float, armed: bool) -> None:
        if not self._auto_arm_active or armed:
            return

        if (
            self._last_manual_mode_request == 0.0
            or now - self._last_manual_mode_request >= 0.5
        ):
            self._run_live_operation(
                "arming.manual-mode",
                "request PX4 Manual mode while preserving the HIL mode bit",
                self._send_manual_mode_command,
                command="MAV_CMD_DO_SET_MODE custom_main_mode=Manual",
            )
            self._last_manual_mode_request = now
            return

        # Give Commander one update cycle to apply Manual before arming.  This
        # also makes command ordering deterministic on slower USB links.
        if (
            now - self._last_manual_mode_request >= 0.1
            and now - self._last_arm_request >= 0.5
        ):
            self._run_live_operation(
                "arming.arm",
                "request PX4 arming after fresh safe HIL pilot input",
                lambda: self._send_arm_command(True),
                command="MAV_CMD_COMPONENT_ARM_DISARM arm=1",
            )
            self._last_arm_request = now

    def pause(self) -> None:
        self._auto_arm_active = False
        if getattr(self, "_connection", None) is not None:
            self._send_arm_command(False)

    def fault(self) -> None:
        """Best-effort HIL stop used for transport and plant safety faults."""

        self.pause()

    def reset(self, reset_mask: Any) -> None:
        if bool(reset_mask[0].item()):
            self._origin_position_n = None
            self._sample = None

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is None:
            return
        try:
            if hasattr(self, "target_system"):
                self._send_arm_command(False)
        finally:
            close = getattr(connection, "close", None)
            if close is not None:
                close()
            self._connection = None

    def step(self, state: Any, reference: Any, active_mask: Any = None) -> Any:
        del state, reference, active_mask
        if self._sample is None:
            raise HilTransportError("PX4 HIL step has no SimEnv sample")
        truth, sensor, channels = self._sample
        now = time.monotonic()
        self._run_live_operation(
            "first-cycle.gcs-heartbeat",
            "send the WebUI GCS heartbeat before the first HIL sample",
            lambda: self._send_gcs_heartbeat(now),
            message="HEARTBEAT",
        )

        # Remove asynchronous frames produced before this plant boundary. The
        # accepted response is therefore the first new PX4 output after the
        # current state/manual sample is transmitted.
        while True:
            stale = self._run_live_operation(
                "first-cycle.drain-actuator",
                "drain actuator frames that predate the first plant sample",
                lambda: self._recv_match(
                    type="HIL_ACTUATOR_CONTROLS", blocking=False
                ),
                message="HIL_ACTUATOR_CONTROLS",
            )
            if stale is None:
                break
            self._drained_frames += 1
            stale_timestamp = int(getattr(stale, "time_usec", -1))
            self._last_actuator_timestamp = max(
                self._last_actuator_timestamp, stale_timestamp
            )

        position = self._row(truth["position_n"])
        if self._origin_position_n is None:
            self._origin_position_n = tuple(position)
        relative_position = [
            value - origin
            for value, origin in zip(position, self._origin_position_n)
        ]
        latitude, longitude, altitude = ned_to_geodetic(
            relative_position,
            home_lat_deg=self.options.home_lat_deg,
            home_lon_deg=self.options.home_lon_deg,
            home_alt_m=self.options.home_alt_m,
        )
        quaternion = self._row(truth["attitude_q_wb"])
        angular_velocity = self._row(
            sensor.get("gyro", truth["angular_velocity_b"])
        )
        velocity = self._row(truth["velocity_n"])
        specific_force = self._row(
            sensor.get("accelerometer", truth["linear_acceleration_n"])
        )
        acceleration_mg = [
            round(min(max(value / 9.80665 * 1000.0, -32768), 32767))
            for value in specific_force
        ]
        velocity_cms = [
            round(min(max(value * 100.0, -32768), 32767))
            for value in velocity
        ]
        x, y, z, r = manual_control_fields(channels)
        armed_flag = 128 if self._mavutil is None else int(
            self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )

        def send_primary_sample() -> int:
            timestamp = time.monotonic_ns() // 1000
            include_state = self._hil_tx_sequence % 10 == 0
            if include_state:
                self._run_live_operation(
                    "first-cycle.hil-state",
                    "send HIL_STATE_QUATERNION truth state to PX4",
                    lambda: self._connection.mav.hil_state_quaternion_send(
                        timestamp,
                        quaternion,
                        *angular_velocity,
                        latitude,
                        longitude,
                        altitude,
                        *velocity_cms,
                        0,
                        0,
                        *acceleration_mg,
                    ),
                    message="HIL_STATE_QUATERNION",
                    sequence=self._hil_tx_sequence,
                )
                self._hil_state_frames += 1
            self._run_live_operation(
                "first-cycle.hil-sensor",
                "send the 500 Hz HIL_SENSOR IMU sample to PX4",
                lambda: self._send_hil_sensor(
                    timestamp,
                    quaternion,
                    relative_position[2],
                    angular_velocity,
                    specific_force,
                    include_aux=include_state,
                ),
                message="HIL_SENSOR",
                sequence=self._hil_tx_sequence,
            )
            if self.options.pilot_source == "webui":
                # A normal PX4 arm request requires throttle < -0.8. Browser
                # virtual input is centred at zero, so send an explicit safe
                # arming sample without modifying the pilot command stored by
                # WebUI or applied after SAFETY_ARMED is observed.
                manual_fields = (
                    (0, 0, 0, 0)
                    if self._auto_arm_active
                    and not self._last_mode & armed_flag
                    else (x, y, z, r)
                )
                self._run_live_operation(
                    "first-cycle.manual-control",
                    "send safe WebUI pilot input before PX4 arming",
                    lambda: self._connection.mav.manual_control_send(
                        self.target_system, *manual_fields, 0
                    ),
                    message="MANUAL_CONTROL",
                    sequence=self._hil_tx_sequence,
                    safe_arming_sample=bool(
                        self._auto_arm_active
                        and not self._last_mode & armed_flag
                    ),
                )
            self._hil_tx_sequence += 1
            return timestamp

        send_primary_sample()
        self._request_manual_mode_then_arm(
            now, bool(self._last_mode & armed_flag)
        )

        wait_started = time.perf_counter()
        starting = self._steps == 0
        if starting:
            response_timeout_s = self.options.startup_timeout_s
        elif time.perf_counter() < self._runtime_settle_until:
            # Right after the first armed boundary the HIL IMU may still be
            # discovered by the sensors module (it scans every 500 ms while
            # unarmed; with the firmware fix it also scans in HIL after
            # arming). Give the first runtime steps a bounded settling
            # window instead of failing on the strict 50 ms actuator timeout.
            response_timeout_s = max(
                self.options.actuator_timeout_s,
                min(self.options.startup_timeout_s, 2.0),
            )
        else:
            response_timeout_s = self.options.actuator_timeout_s
        deadline = wait_started + response_timeout_s
        retry_interval_s = max(self._control_dt, 0.001)
        next_retry = wait_started + retry_interval_s
        next_progress = wait_started + 1.0
        message = None
        received_unarmed = False
        receive_none_count = 0
        retry_count = 0
        stale_actuator_count = 0
        unarmed_actuator_count = 0
        message_counts: dict[str, int] = {}
        last_received_type = "none"
        last_received_mode: int | None = None
        last_received_timestamp: int | None = None
        boundary_operation = (
            "receive the first armed, fresh HIL_ACTUATOR_CONTROLS frame"
        )
        if starting:
            self._trace_startup(
                "started",
                "first-cycle.actuator-boundary",
                boundary_operation,
                timeout_ms=response_timeout_s * 1000.0,
                armed_flag=armed_flag,
                transport=self._transport_snapshot(),
            )
        while message is None:
            now_perf = time.perf_counter()
            if starting and now_perf >= next_progress:
                self._trace_startup(
                    "progress",
                    "first-cycle.actuator-boundary",
                    boundary_operation,
                    summary=(
                        f"still waiting; last={last_received_type}, "
                        f"unarmed={unarmed_actuator_count}, "
                        f"stale={stale_actuator_count}, retries={retry_count}"
                    ),
                    elapsed_ms=(now_perf - wait_started) * 1000.0,
                    received_message_counts=dict(message_counts),
                    receive_none_count=receive_none_count,
                    unarmed_actuator_count=unarmed_actuator_count,
                    stale_actuator_count=stale_actuator_count,
                    retry_count=retry_count,
                    last_received_type=last_received_type,
                    last_received_mode=last_received_mode,
                    last_received_timestamp=last_received_timestamp,
                    arming_feedback=list(self._arming_feedback),
                    transport=self._transport_snapshot(),
                )
                next_progress = now_perf + 1.0
            remaining = deadline - now_perf
            if remaining <= 0.0:
                # Manual bootstrap mode intentionally waits here until the
                # operator configures and arms PX4 from the WebUI terminal.
                # Do not fault the whole session merely because the first
                # actuator frame is unarmed.
                if starting and not self.options.auto_start_px4:
                    deadline = now_perf + response_timeout_s
                    continue
                break
            receive_timeout = remaining
            receive_timeout = min(
                receive_timeout, max(0.0, next_retry - now_perf)
            )
            candidate = self._run_live_operation(
                "first-cycle.receive",
                "read one MAVLink frame while searching for the first armed actuator response",
                lambda: self._recv_match(
                    blocking=True, timeout=receive_timeout
                ),
                expected_message="HIL_ACTUATOR_CONTROLS",
            )
            if candidate is None:
                receive_none_count += 1
                if time.perf_counter() < deadline:
                    send_primary_sample()
                    retry_count += 1
                    next_retry = time.perf_counter() + retry_interval_s
                    if starting:
                        retry_now = time.monotonic()
                        self._request_manual_mode_then_arm(
                            retry_now, bool(self._last_mode & armed_flag)
                        )
                    continue
                break
            candidate_type = str(candidate.get_type())
            last_received_type = candidate_type
            message_counts[candidate_type] = message_counts.get(candidate_type, 0) + 1
            if hasattr(candidate, "mode"):
                last_received_mode = int(candidate.mode)
            if hasattr(candidate, "time_usec"):
                last_received_timestamp = int(candidate.time_usec)
            if candidate_type != "HIL_ACTUATOR_CONTROLS":
                feedback_before = len(self._arming_feedback)
                self._capture_arming_feedback(candidate)
                if starting and len(self._arming_feedback) > feedback_before:
                    self._trace_startup(
                        "progress",
                        "arming.feedback",
                        "receive PX4 response while waiting for automatic arming",
                        summary=self._arming_feedback[-1],
                        received_message_type=candidate_type,
                        arming_feedback=list(self._arming_feedback),
                        transport=self._transport_snapshot(),
                    )
                # A busy MAVLink link can continuously deliver unrelated
                # messages (heartbeats, status text, etc.) without ever
                # returning None from recv_match.  Keep the HIL input alive
                # on that path as well; otherwise a stream of stale traffic
                # can starve the retry timer until the 50 ms deadline.
                now_retry = time.perf_counter()
                if now_retry >= next_retry and now_retry < deadline:
                    send_primary_sample()
                    retry_count += 1
                    next_retry = now_retry + retry_interval_s
                if starting and now_retry >= next_progress:
                    self._trace_startup(
                        "progress",
                        "first-cycle.actuator-boundary",
                        boundary_operation,
                        summary=(
                            f"still waiting; last={last_received_type}, "
                            f"unarmed={unarmed_actuator_count}, "
                            f"stale={stale_actuator_count}, retries={retry_count}"
                        ),
                        elapsed_ms=(now_retry - wait_started) * 1000.0,
                        received_message_counts=dict(message_counts),
                        receive_none_count=receive_none_count,
                        unarmed_actuator_count=unarmed_actuator_count,
                        stale_actuator_count=stale_actuator_count,
                        retry_count=retry_count,
                        last_received_type=last_received_type,
                        last_received_mode=last_received_mode,
                        last_received_timestamp=last_received_timestamp,
                        arming_feedback=list(self._arming_feedback),
                        transport=self._transport_snapshot(),
                    )
                    next_progress = now_retry + 1.0
                continue
            candidate_timestamp = int(getattr(candidate, "time_usec", -1))
            if (
                candidate_timestamp >= 0
                and candidate_timestamp <= self._last_actuator_timestamp
            ):
                self._drained_frames += 1
                stale_actuator_count += 1
                # PX4 may continue publishing the last actuator frame while
                # its sensor voter is recovering.  Treat that exactly like a
                # receive timeout and retransmit the current HIL sample so
                # the host-side wait cannot deadlock on stale output.
                now_retry = time.perf_counter()
                if now_retry >= next_retry and now_retry < deadline:
                    send_primary_sample()
                    retry_count += 1
                    next_retry = now_retry + retry_interval_s
                continue
            candidate_mode = int(getattr(candidate, "mode", 0))
            if not candidate_mode & armed_flag:
                received_unarmed = True
                unarmed_actuator_count += 1
                self._last_mode = candidate_mode
                if starting and unarmed_actuator_count == 1:
                    self._trace_startup(
                        "progress",
                        "first-cycle.actuator-boundary",
                        boundary_operation,
                        summary=(
                            "received HIL_ACTUATOR_CONTROLS, but PX4 mode "
                            f"{candidate_mode} does not contain armed flag {armed_flag}"
                        ),
                        received_message_type=candidate_type,
                        received_mode=candidate_mode,
                        received_time_usec=candidate_timestamp,
                        transport=self._transport_snapshot(),
                    )
                if starting:
                    now_retry = time.perf_counter()
                    if now_retry >= next_retry:
                        send_primary_sample()
                        retry_count += 1
                        next_retry = now_retry + retry_interval_s
                    retry_now = time.monotonic()
                    self._request_manual_mode_then_arm(
                        retry_now, bool(self._last_mode & armed_flag)
                    )
                    if now_retry >= next_progress:
                        self._trace_startup(
                            "progress",
                            "first-cycle.actuator-boundary",
                            boundary_operation,
                            summary=(
                                f"PX4 remains disarmed; mode={candidate_mode}, "
                                f"unarmed={unarmed_actuator_count}, "
                                f"retries={retry_count}"
                            ),
                            elapsed_ms=(now_retry - wait_started) * 1000.0,
                            received_message_counts=dict(message_counts),
                            receive_none_count=receive_none_count,
                            unarmed_actuator_count=unarmed_actuator_count,
                            stale_actuator_count=stale_actuator_count,
                            retry_count=retry_count,
                            last_received_type=last_received_type,
                            last_received_mode=last_received_mode,
                            last_received_timestamp=last_received_timestamp,
                            arming_feedback=list(self._arming_feedback),
                            transport=self._transport_snapshot(),
                        )
                        next_progress = now_retry + 1.0
                    continue
            message = candidate
            if candidate_timestamp >= 0:
                self._last_actuator_timestamp = candidate_timestamp
        self._last_rtt_ms = (time.perf_counter() - wait_started) * 1000.0
        if message is None:
            self._timeouts += 1
            phase = "startup" if starting else "runtime"
            if starting:
                self._trace_startup(
                    "failed",
                    "first-cycle.actuator-boundary",
                    "receive an armed, fresh HIL_ACTUATOR_CONTROLS frame",
                    elapsed_ms=self._last_rtt_ms,
                    error_type="TimeoutError",
                    error=(
                        f"no acceptable actuator frame within "
                        f"{response_timeout_s * 1000:.0f} ms"
                    ),
                    last_completed_step=self._last_completed_startup_step,
                    transport=self._transport_snapshot(),
                    received_unarmed=received_unarmed,
                    received_message_counts=dict(message_counts),
                    receive_none_count=receive_none_count,
                    unarmed_actuator_count=unarmed_actuator_count,
                    stale_actuator_count=stale_actuator_count,
                    retry_count=retry_count,
                    last_received_type=last_received_type,
                    last_received_mode=last_received_mode,
                    last_received_timestamp=last_received_timestamp,
                    arming_feedback=list(self._arming_feedback),
                )
            shell_detail = ""
            if self.options.auto_start_px4:
                self._collect_timeout_diagnostics(send_primary_sample)
                compact_output = " ".join(
                    self._bootstrap_output[-8000:].split()
                )
                shell_detail = (
                    f"; PX4 shell: {compact_output}"
                    if compact_output
                    else "; PX4 shell returned no bootstrap output"
                )
            if starting and received_unarmed:
                arm_hint = (
                    "PX4 rejected automatic arming; inspect the post-injection "
                    "arming check output"
                    if self.options.auto_arm
                    else "enable runtime.hil.auto_arm or arm PX4 before start"
                )
                feedback_detail = (
                    "; arming feedback: " + " | ".join(self._arming_feedback)
                    if self._arming_feedback
                    else ""
                )
                failure_message = (
                    f"PX4 HIL remained disarmed for "
                    f"{response_timeout_s * 1000:.0f} ms; {arm_hint}"
                    f"{feedback_detail}{shell_detail}"
                )
            else:
                failure_message = (
                    f"PX4 HIL actuator {phase} timeout after "
                    f"{response_timeout_s * 1000:.0f} ms; check pwm_out_sim, "
                    f"nn_control and HIL_ACT_FUNC1..5; "
                    f"WebUI sent {self._hil_sensor_frames} HIL_SENSOR frames"
                    f"{shell_detail}"
                )
            self._trace_startup(
                "failed",
                "first-cycle.startup-result",
                "abort the first HIL cycle after actuator/arming timeout diagnostics",
                error_type="HilTransportError",
                error=failure_message,
                last_completed_step=self._last_completed_startup_step,
                transport=self._transport_snapshot(),
                received_message_counts=dict(message_counts),
                receive_none_count=receive_none_count,
                unarmed_actuator_count=unarmed_actuator_count,
                stale_actuator_count=stale_actuator_count,
                retry_count=retry_count,
                arming_feedback=list(self._arming_feedback),
                shell_output_tail=self._bootstrap_output[-8000:],
            )
            raise HilTransportError(failure_message)
        if starting:
            self._last_completed_startup_step = "first-cycle.actuator-boundary"
            self._runtime_settle_until = (
                time.perf_counter() + min(self.options.startup_timeout_s, 2.0)
            )
            self._trace_startup(
                "completed",
                "first-cycle.actuator-boundary",
                boundary_operation,
                elapsed_rtt_ms=self._last_rtt_ms,
                actuator_timestamp=int(getattr(message, "time_usec", -1)),
                mavlink_mode=int(getattr(message, "mode", 0)),
                received_message_counts=dict(message_counts),
                receive_none_count=receive_none_count,
                unarmed_actuator_count=unarmed_actuator_count,
                stale_actuator_count=stale_actuator_count,
                retry_count=retry_count,
                transport=self._transport_snapshot(),
            )
        command = actuator_command(message)
        self._last_mode = int(getattr(message, "mode", 0))
        if not self._last_mode & armed_flag:
            raise HilTransportError(
                "PX4 disarmed during HIL; simulation was not advanced"
            )
        self._armed_by_us = bool(
            self._auto_arm_active and self._last_mode & armed_flag
        )
        self._steps += 1
        output = self._torch.tensor(
            [command], device=self._device, dtype=self._dtype
        )
        diagnostics = {
            "transport_rtt_ms": self._torch.tensor(
                [self._last_rtt_ms], device=self._device, dtype=self._dtype
            ),
            "mavlink_mode": self._torch.tensor(
                [self._last_mode], device=self._device, dtype=self._dtype
            ),
        }
        from flight_controller import ControllerOutput

        return ControllerOutput.create(output, diagnostics)

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.controller_type,
            "protocol": "mavlink2-px4-hil-v4",
            "protocol_version": PROTOCOL_VERSION,
            "connection": self.options.connection,
            "baud": self.options.baud,
            "auto_arm": self.options.auto_arm,
            "pilot_source": self.options.pilot_source,
            "auto_start_px4": self.options.auto_start_px4,
            "px4_mavlink_device": self.options.px4_mavlink_device,
            "startup_diagnostic_log": (
                str(self._diagnostic_path)
                if self._diagnostic_path is not None
                else None
            ),
            "startup_step": self._startup_step,
            "startup_operation": self._startup_operation,
            "last_completed_startup_step": self._last_completed_startup_step,
            "startup_trace": list(self._startup_trace),
            "bootstrap_commands": list(self._bootstrap_commands),
            "bootstrap_output": self._bootstrap_output,
            "frames": {"world": "NED", "body": "FRD", "quaternion": "q_wb"},
            "actuator_order": [
                "upper_motor",
                "lower_motor",
                "servo_1",
                "servo_2",
                "servo_3",
            ],
            "steps": self._steps,
            "timeouts": self._timeouts,
            "drained_frames": self._drained_frames,
            "hil_sensor_frames": self._hil_sensor_frames,
            "hil_state_frames": self._hil_state_frames,
            "last_rtt_ms": self._last_rtt_ms,
            "mavlink_mode": self._last_mode,
            "armed": bool(self._last_mode & 128),
            "arming_feedback": list(self._arming_feedback),
        }
