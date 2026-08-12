from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from hil_runtime import (
    HilConfigurationError,
    HilOptions,
    HilTransportError,
    Px4HilController,
    actuator_command,
    manual_control_fields,
    ned_to_geodetic,
    rotate_world_to_body,
)


class _FakeMav:
    def __init__(self, connection=None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.connection = connection

    def __getattr__(self, name):
        if not name.endswith("_send"):
            raise AttributeError(name)

        def send(*args):
            self.calls.append((name, args))
            if (
                name == "serial_control_send"
                and self.connection is not None
                and self.connection.fail_on_shell_command is not None
            ):
                payload = bytes(args[5][: args[4]])
                if self.connection.fail_on_shell_command in payload:
                    if self.connection.write_through:
                        return self.connection.write(payload)
                    self.connection.portdead = True

        return send


class _FakeConnection:
    target_system = 1
    target_component = 1

    def __init__(
        self,
        messages=None,
        fail_on_shell_command=None,
        *,
        write_through=False,
    ) -> None:
        self.fail_on_shell_command = fail_on_shell_command
        self.write_through = write_through
        self.portdead = False
        self.mav = _FakeMav(self)
        self.messages = list(messages or [])
        self.closed = False

    def wait_heartbeat(self, timeout):
        del timeout
        return SimpleNamespace(base_mode=32)

    def recv_match(self, **kwargs):
        if not kwargs.get("blocking", False):
            return None
        return self.messages.pop(0) if self.messages else None

    def close(self):
        self.closed = True

    def write(self, payload):
        self.portdead = True
        return -1


class HilProtocolTest(unittest.TestCase):
    def test_options_validate_hil_timeouts(self):
        options = HilOptions.parse(
            {"hil": {"connection": "udp:127.0.0.1:14560", "auto_arm": True}}
        )
        self.assertEqual(options.baud, 2_000_000)
        self.assertTrue(options.auto_arm)
        self.assertTrue(options.auto_start_px4)
        self.assertEqual(options.pilot_source, "webui")
        with self.assertRaisesRegex(HilConfigurationError, "actuator_timeout_ms"):
            HilOptions.parse({"hil": {"actuator_timeout_ms": 1}})
        with self.assertRaisesRegex(HilConfigurationError, "simple /dev path"):
            HilOptions.parse(
                {"hil": {"px4_mavlink_device": "/dev/ttyACM0; reboot"}}
            )
        with self.assertRaisesRegex(HilConfigurationError, "pilot_source"):
            HilOptions.parse({"hil": {"pilot_source": "both"}})

    def test_manual_control_axis_and_throttle_mapping(self):
        self.assertEqual(
            manual_control_fields((0.25, -0.5, 0.75, -1.0)),
            (500, 250, 0, 750),
        )
        self.assertEqual(manual_control_fields((0, 0, 0, 1)), (0, 0, 1000, 0))

    def test_ned_to_geodetic_preserves_down_sign(self):
        lat, lon, alt = ned_to_geodetic(
            (0, 0, 3),
            home_lat_deg=31.0,
            home_lon_deg=121.0,
            home_alt_m=20.0,
        )
        self.assertEqual((lat, lon), (310_000_000, 1_210_000_000))
        self.assertEqual(alt, 17_000)

    def test_q_wb_world_to_body_rotation(self):
        self.assertEqual(
            rotate_world_to_body((1, 0, 0, 0), (0.22, 0.01, 0.43)),
            (0.22, 0.01, 0.43),
        )

    def test_actuator_ranges_are_part_of_contract(self):
        message = SimpleNamespace(controls=[0.2, 0.3, -1, 0, 1] + [0] * 11)
        self.assertEqual(actuator_command(message), (0.2, 0.3, -1, 0, 1))
        with self.assertRaises(HilTransportError):
            actuator_command(SimpleNamespace(controls=[-0.1, 0, 0, 0, 0]))

    def test_step_sends_q_wb_and_returns_five_actuators(self):
        unarmed_response = SimpleNamespace(
            controls=[0.0] * 16,
            mode=32,
            get_type=lambda: "HIL_ACTUATOR_CONTROLS",
        )
        arm_denied = SimpleNamespace(
            command=400,
            result=2,
            result_param2=5,
            get_type=lambda: "COMMAND_ACK",
        )
        response = SimpleNamespace(
            controls=[0.21, 0.32, -0.4, 0.5, 0.6] + [0.0] * 11,
            mode=160,
            get_type=lambda: "HIL_ACTUATOR_CONTROLS",
        )
        connection = _FakeConnection(
            [unarmed_response, arm_denied] + [response] * 11
        )
        controller = Px4HilController(
            HilOptions(connection="fake", actuator_timeout_s=0.01),
            torch_module=torch,
            device=torch.device("cpu"),
            dtype=torch.float32,
            control_dt=0.002,
            connection_factory=lambda *args, **kwargs: connection,
        )
        truth = {
            "position_n": torch.tensor([[1.0, 2.0, 3.0]]),
            "velocity_n": torch.tensor([[4.0, 5.0, 6.0]]),
            "attitude_q_wb": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "angular_velocity_b": torch.tensor([[0.1, 0.2, 0.3]]),
            "linear_acceleration_n": torch.zeros((1, 3)),
        }
        sensor = {
            "gyro": torch.tensor([[0.11, 0.22, 0.33]]),
            "accelerometer": torch.tensor([[0.0, 0.0, -9.80665]]),
        }
        channels = torch.tensor([[0.2, -0.3, 0.4, -0.5]])
        controller.set_sample(truth, sensor, channels)
        controller.start()
        output = controller.step(None, None)
        self.assertTrue(
            torch.allclose(
                output.command,
                torch.tensor([[0.21, 0.32, -0.4, 0.5, 0.6]]),
            )
        )
        state_call = next(call for call in connection.mav.calls if call[0] == "hil_state_quaternion_send")
        self.assertEqual(list(state_call[1][1]), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(state_call[1][-3:], (0, 0, -1000))
        manual_call = next(call for call in connection.mav.calls if call[0] == "manual_control_send")
        self.assertEqual(manual_call[1][1:5], (0, 0, 0, 0))
        sensor_call = next(call for call in connection.mav.calls if call[0] == "hil_sensor_send")
        for actual, expected in zip(
            sensor_call[1][1:7],
            (0.0, 0.0, -9.80665, 0.11, 0.22, 0.33),
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(sensor_call[1][-1], 7167)
        shell_payload = b"".join(
            bytes(call[1][5][: call[1][4]])
            for call in connection.mav.calls
            if call[0] == "serial_control_send" and call[1][4] > 0
        )
        self.assertIn(b"pwm_out_sim start -m hil\n", shell_payload)
        self.assertIn(b"nn_control start\n", shell_payload)
        self.assertLess(
            shell_payload.index(b"nn_control stop\n"),
            shell_payload.index(b"nn_control start\n"),
        )
        self.assertLess(
            shell_payload.index(b"pwm_out_sim stop\n"),
            shell_payload.index(b"pwm_out_sim start -m hil\n"),
        )
        self.assertIn(b"param set COM_RC_IN_MODE 1\n", shell_payload)
        self.assertIn(b"param set EKF2_EN 0\n", shell_payload)
        self.assertIn(b"param set COM_ARM_WO_GPS 1\n", shell_payload)
        self.assertIn(b"param set SENS_IMU_MODE 1\n", shell_payload)
        self.assertIn(b"param set IMU_GYRO_RATEMAX 1000\n", shell_payload)
        self.assertIn(b"param set HIL_ACT_REV 0\n", shell_payload)
        sens_imu_param = next(
            call for call in connection.mav.calls
            if call[0] == "param_set_send" and call[1][2] == b"SENS_IMU_MODE"
        )
        self.assertEqual(sens_imu_param[1][3:5], (1.0, 9))
        gyro_rate_param = next(
            call for call in connection.mav.calls
            if call[0] == "param_set_send" and call[1][2] == b"IMU_GYRO_RATEMAX"
        )
        self.assertEqual(gyro_rate_param[1][3:5], (1000.0, 9))
        hil_rev_param = next(
            call for call in connection.mav.calls
            if call[0] == "param_set_send" and call[1][2] == b"HIL_ACT_REV"
        )
        self.assertEqual(hil_rev_param[1][3:5], (0.0, 9))
        ekf2_param = next(
            call for call in connection.mav.calls
            if call[0] == "param_set_send" and call[1][2] == b"EKF2_EN"
        )
        self.assertEqual(ekf2_param[1][3:5], (0.0, 9))
        wo_gps_param = next(
            call for call in connection.mav.calls
            if call[0] == "param_set_send" and call[1][2] == b"COM_ARM_WO_GPS"
        )
        self.assertEqual(wo_gps_param[1][3:5], (1.0, 9))
        message_interval = next(
            call for call in connection.mav.calls
            if call[0] == "command_long_send" and call[1][2] == 511
        )
        self.assertEqual(message_interval[1][4:7], (93.0, 2000.0, 0.0))
        self.assertNotIn(b"mavlink stream", shell_payload)
        self.assertTrue(math.isfinite(float(output.diagnostics["transport_rtt_ms"][0])))
        self.assertEqual(
            controller.describe()["arming_feedback"],
            ["arm COMMAND_ACK: denied (2), reason 5"],
        )
        for _ in range(10):
            controller.step(None, None)
        manual_calls = [
            call for call in connection.mav.calls
            if call[0] == "manual_control_send"
        ]
        self.assertEqual(manual_calls[-1][1][1:5], (300, 200, 250, 400))
        self.assertEqual(
            sum(
                call[0] == "hil_state_quaternion_send"
                for call in connection.mav.calls
            ),
            2,
        )
        self.assertEqual(
            sum(call[0] == "hil_sensor_send" for call in connection.mav.calls),
            11,
        )
        controller.close()
        self.assertTrue(connection.closed)
        disarm = next(
            call for call in connection.mav.calls
            if call[0] == "command_long_send"
            and call[1][2] == 400
            and call[1][4] == 0.0
        )
        self.assertEqual(disarm[1][5], 21196.0)

    def test_physical_rc_source_is_not_overwritten_by_webui(self):
        response = SimpleNamespace(
            controls=[0.0] * 16,
            mode=160,
            get_type=lambda: "HIL_ACTUATOR_CONTROLS",
        )
        connection = _FakeConnection([response])
        controller = Px4HilController(
            HilOptions(connection="fake", pilot_source="rc"),
            torch_module=torch,
            device=torch.device("cpu"),
            dtype=torch.float32,
            control_dt=0.002,
            connection_factory=lambda *args, **kwargs: connection,
        )
        controller.set_sample(
            {
                "position_n": torch.zeros((1, 3)),
                "velocity_n": torch.zeros((1, 3)),
                "attitude_q_wb": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                "angular_velocity_b": torch.zeros((1, 3)),
                "linear_acceleration_n": torch.zeros((1, 3)),
            },
            {
                "gyro": torch.zeros((1, 3)),
                "accelerometer": torch.tensor([[0.0, 0.0, -9.80665]]),
            },
            torch.zeros((1, 4)),
        )
        controller.step(None, None)
        self.assertFalse(
            any(call[0] == "manual_control_send" for call in connection.mav.calls)
        )
        shell_payload = b"".join(
            bytes(call[1][5][: call[1][4]])
            for call in connection.mav.calls
            if call[0] == "serial_control_send" and call[1][4] > 0
        )
        self.assertIn(b"param set COM_RC_IN_MODE 0\n", shell_payload)
        controller.close()

    def test_auto_arm_selects_manual_mode_before_arm_request(self):
        connection = _FakeConnection()
        controller = Px4HilController(
            HilOptions(connection="fake", auto_start_px4=False),
            torch_module=torch,
            device=torch.device("cpu"),
            dtype=torch.float32,
            control_dt=0.002,
            connection_factory=lambda *args, **kwargs: connection,
        )
        controller.start()
        # This unit test isolates mode/arm command ordering. Manual-bootstrap
        # sessions intentionally leave automatic arming disabled in start().
        controller._auto_arm_active = True
        controller._request_manual_mode_then_arm(10.0, False)
        controller._request_manual_mode_then_arm(10.05, False)
        controller._request_manual_mode_then_arm(10.11, False)

        startup_commands = [
            call[1]
            for call in connection.mav.calls
            if call[0] == "command_long_send" and call[1][2] in {176, 400}
        ]
        self.assertEqual([call[2] for call in startup_commands], [176, 400])
        # Keep MAV_MODE_FLAG_HIL_ENABLED together with custom-mode selection;
        # PX4 otherwise stops accepting HIL_SENSOR after switching to Manual.
        self.assertEqual(startup_commands[0][4:7], (33.0, 1.0, 0.0))
        controller.close()

    def test_startup_trace_identifies_the_command_that_loses_serial(self):
        connection = _FakeConnection(
            fail_on_shell_command=b"nn_control stop\n",
            write_through=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            diagnostic_path = Path(directory) / "hil-startup.jsonl"
            with self.assertRaises(HilTransportError) as raised:
                Px4HilController(
                    HilOptions(connection="fake"),
                    torch_module=torch,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    control_dt=0.002,
                    connection_factory=lambda *args, **kwargs: connection,
                    diagnostic_path=diagnostic_path,
                )

            message = str(raised.exception)
            self.assertIn("[shell-command.03]", message)
            self.assertIn("stop the nn_control callback worker", message)
            self.assertIn("last completed step: shell-command.02", message)
            self.assertIn(str(diagnostic_path), message)
            events = [
                json.loads(line)
                for line in diagnostic_path.read_text(encoding="utf-8").splitlines()
            ]
            failure = events[-1]
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["step"], "shell-command.03")
            self.assertEqual(failure["command"], "nn_control stop")
            self.assertEqual(
                failure["last_completed_step"], "shell-command.02"
            )
            self.assertTrue(failure["transport"]["portdead"])
            self.assertTrue(connection.closed)

    def test_startup_trace_distinguishes_raw_frames_from_armed_actuator_boundary(self):
        unarmed_response = SimpleNamespace(
            controls=[0.0] * 16,
            mode=32,
            time_usec=123,
            get_type=lambda: "HIL_ACTUATOR_CONTROLS",
        )
        arm_denied = SimpleNamespace(
            command=400,
            result=2,
            result_param2=5,
            get_type=lambda: "COMMAND_ACK",
        )
        connection = _FakeConnection([unarmed_response, arm_denied])
        with tempfile.TemporaryDirectory() as directory:
            diagnostic_path = Path(directory) / "hil-startup.jsonl"
            controller = Px4HilController(
                HilOptions(connection="fake", startup_timeout_s=0.1),
                torch_module=torch,
                device=torch.device("cpu"),
                dtype=torch.float32,
                control_dt=0.002,
                connection_factory=lambda *args, **kwargs: connection,
                diagnostic_path=diagnostic_path,
            )
            controller.set_sample(
                {
                    "position_n": torch.zeros((1, 3)),
                    "velocity_n": torch.zeros((1, 3)),
                    "attitude_q_wb": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    "angular_velocity_b": torch.zeros((1, 3)),
                    "linear_acceleration_n": torch.zeros((1, 3)),
                },
                {
                    "gyro": torch.zeros((1, 3)),
                    "accelerometer": torch.tensor([[0.0, 0.0, -9.80665]]),
                },
                torch.zeros((1, 4)),
            )
            # The timeout logger itself is tested through the boundary event;
            # avoid spending another second on fake NSH diagnostics here.
            controller._collect_timeout_diagnostics = lambda keepalive=None: None
            controller.start()
            with self.assertRaisesRegex(HilTransportError, "remained disarmed"):
                controller.step(None, None)

            events = [
                json.loads(line)
                for line in diagnostic_path.read_text(encoding="utf-8").splitlines()
            ]
            raw_receive = next(
                event for event in events
                if event["status"] == "completed"
                and event["step"] == "first-cycle.receive"
            )
            self.assertEqual(
                raw_receive["received_message_type"],
                "HIL_ACTUATOR_CONTROLS",
            )
            self.assertEqual(raw_receive["received_mode"], 32)
            failure = next(
                event for event in events
                if event["status"] == "failed"
                and event["step"] == "first-cycle.actuator-boundary"
            )
            self.assertEqual(failure["unarmed_actuator_count"], 1)
            self.assertEqual(
                failure["received_message_counts"],
                {"HIL_ACTUATOR_CONTROLS": 1, "COMMAND_ACK": 1},
            )
            self.assertEqual(
                failure["arming_feedback"],
                ["arm COMMAND_ACK: denied (2), reason 5"],
            )
            controller.close()


if __name__ == "__main__":
    unittest.main()
