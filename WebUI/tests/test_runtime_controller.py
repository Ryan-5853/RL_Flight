from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from runtime import (
    CpuRuntimeSession,
    RuntimeConfigurationError,
    RuntimeRegistry,
    _simulator_compatibility_fingerprint,
    parse_runtime_options,
)


ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG = ROOT / "SimEnv" / "configs" / "example.yaml"


class RuntimeOptionsTests(unittest.TestCase):
    def test_simulator_fingerprint_ignores_reset_and_logging_only(self) -> None:
        baseline = {
            "seed": 1,
            "initial_state": {"position_n": {"value": [0, 0, 0]}},
            "body": {"mass": {"value": 1}},
            "logging": {"directory": "first"},
        }
        equivalent = {
            **baseline,
            "seed": 2,
            "initial_state": {"position_n": {"value": [1, 2, 3]}},
            "logging": {"directory": "second"},
        }
        incompatible = {
            **equivalent,
            "body": {"mass": {"value": 2.0}},
        }
        self.assertEqual(
            _simulator_compatibility_fingerprint(baseline),
            _simulator_compatibility_fingerprint(equivalent),
        )
        self.assertNotEqual(
            _simulator_compatibility_fingerprint(baseline),
            _simulator_compatibility_fingerprint(incompatible),
        )

    def test_realtime_backend_defaults_to_compiled_500_hz_execution(self) -> None:
        options = parse_runtime_options({})

        self.assertTrue(options.compile_kernels)
        self.assertEqual(options.warmup_steps, 3)
        self.assertEqual(options.spin_us, 200.0)
        self.assertEqual(options.execution_hz, 500.0)

    def test_compile_kernels_requires_a_boolean(self) -> None:
        with self.assertRaisesRegex(
            RuntimeConfigurationError,
            "runtime.compile_kernels must be boolean",
        ):
            parse_runtime_options(
                {"runtime": {"compile_kernels": "false"}}
            )

    def test_spin_window_is_bounded_to_one_millisecond(self) -> None:
        with self.assertRaisesRegex(
            RuntimeConfigurationError,
            "runtime.spin_us must be between 0 and 1000",
        ):
            parse_runtime_options({"runtime": {"spin_us": 1001}})


class RuntimeControllerIntegrationTests(unittest.TestCase):
    def test_offline_rollout_uses_virtual_pilot_and_retains_video_frames(
        self,
    ) -> None:
        test_config = {
            "run": {"device": "cpu", "dtype": "float32"},
            "environment": {"observation_source": "truth"},
            "runtime": {
                "cpu_threads": 1,
                "compile_kernels": False,
                "execution_hz": 80,
                "telemetry_hz": 30,
                "command_timeout_ms": 1000,
            },
            "controller": {
                "type": "pid",
                "params": {"collective_mode": "hover"},
            },
            "command_source": {
                "type": (
                    "flight_train.commands:"
                    "VirtualPilotCommandSource"
                ),
                "version": "2",
                "seed": 51002,
                "params": {
                    "throttle": {
                        "minimum": 0.2,
                        "maximum": 0.85,
                        "spool": {
                            "duration_s": 0.01,
                            "target_range": [0.3, 0.4],
                        },
                        "height_controller": {
                            "observation_source": "truth",
                            "target_m": 0.0,
                            "initial_throttle_range": [0.48, 0.60],
                            "proportional_gain": 0.08,
                            "integral_gain": 0.04,
                            "error_limit_m": 5.0,
                        },
                        "slew_rate": {
                            "rise_per_s": 0.5,
                            "fall_per_s": 0.35,
                        },
                    },
                    "sticks": {
                        "roll": {
                            "mode": "angle",
                            "limit_rad": 0.35,
                            "time_constant_s": 0.2,
                        },
                        "pitch": {
                            "mode": "angle",
                            "limit_rad": 0.35,
                            "time_constant_s": 0.2,
                        },
                        "yaw": {
                            "mode": "rate",
                            "limit_rad_s": 1.5,
                            "time_constant_s": 0.3,
                        },
                        "target_sampling": {
                            "distribution": "centered",
                            "center_exponent": 2.0,
                            "hold_duration_s": {"range": [1.0, 4.0]},
                        },
                        "reset": {
                            "filtered_stick": "zero",
                            "initial_target_scale": 0.25,
                        },
                    },
                },
            },
            "task": {
                "episode_duration_s": 0.02,
                "termination": {
                    "max_tilt_rad": 1.3,
                    "max_angular_rate_rad_s": 20.0,
                },
            },
            "randomization": {"dynamic": {"parameters": {}}},
        }
        simenv_config = yaml.safe_load(
            SIM_CONFIG.read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as log_root:
            session = CpuRuntimeSession(
                yaml.safe_dump(simenv_config),
                yaml.safe_dump(test_config),
                None,
                log_root,
            )
            try:
                progress = []
                rollout = session.sample_virtual_pilot_rollout(
                    fps=10,
                    progress=lambda completed, total: progress.append(
                        (completed, total)
                    ),
                )
                self.assertEqual(rollout["schema_version"], 1)
                self.assertEqual(rollout["control_hz"], 500)
                self.assertEqual(rollout["requested_duration_s"], 0.02)
                self.assertEqual(
                    rollout["termination"]["reason"],
                    "episode_timeout",
                )
                self.assertEqual(progress[-1], (10, 10))
                self.assertGreaterEqual(len(rollout["frames"]), 2)
                first, last = rollout["frames"][0], rollout["frames"][-1]
                self.assertEqual(first["time_s"], 0.0)
                self.assertAlmostEqual(last["time_s"], 0.02)
                self.assertIn("target_attitude_q_wb", last["pilot"])
                self.assertEqual(
                    set(last["pilot"]["channels"]),
                    {"roll", "pitch", "yaw", "throttle"},
                )
                self.assertEqual(
                    len(last["controller"]["command"][0]),
                    5,
                )
                self.assertEqual(
                    len(last["reference"]["target_attitude_q_wb"][0]),
                    4,
                )
            finally:
                session.close()

    def test_classical_controller_needs_no_checkpoint_and_publishes_diagnostics(
        self,
    ) -> None:
        test_config = {
            "run": {"device": "cpu", "dtype": "float32"},
            "environment": {"observation_source": "truth"},
            "runtime": {
                "cpu_threads": 1,
                "compile_kernels": False,
                "telemetry_hz": 30,
                "command_timeout_ms": 1000,
            },
            "controller": {
                "type": "hybrid_pid_lqr",
                "params": {"collective_mode": "hover"},
            },
            "command_source": {
                "params": {
                    "throttle": {
                        "minimum": 0.2,
                        "maximum": 0.85,
                        "slew_rate": {"rise_per_s": 4.0, "fall_per_s": 0.35},
                    },
                    "sticks": {
                        "roll": {"limit_rad": 0.35, "time_constant_s": 0.2},
                        "pitch": {"limit_rad": 0.35, "time_constant_s": 0.2},
                        "yaw": {"limit_rad_s": 1.5, "time_constant_s": 0.3},
                    },
                }
            },
            "task": {
                "episode_duration_s": 30,
                "termination": {
                    "max_tilt_rad": 1.0,
                    "max_angular_rate_rad_s": 8.0,
                },
            },
            "randomization": {"dynamic": {"parameters": {}}},
        }
        simenv_config = yaml.safe_load(
            SIM_CONFIG.read_text(encoding="utf-8")
        )
        # WebUI accepts pre-single-step-interface configs and makes the only
        # supported fractional-delay interpolation policy explicit.
        simenv_config["sensors"]["gyro"].pop("interpolation", None)
        simenv_config["sensors"]["accelerometer"].pop(
            "interpolation", None
        )
        with tempfile.TemporaryDirectory() as log_root:
            session = CpuRuntimeSession(
                yaml.safe_dump(simenv_config),
                yaml.safe_dump(test_config),
                None,
                log_root,
            )
            try:
                initial_status = session.status()
                self.assertEqual(
                    initial_status["simulation_backend"],
                    "simenv-realtime-single-v1",
                )
                self.assertFalse(initial_status["simulation_compiled"])
                self.assertFalse(initial_status["persistent_logging"])
                self.assertIsNone(initial_status["log_directory"])
                self.assertEqual(
                    initial_status["controller"]["type"],
                    "hybrid_pid_lqr",
                )
                effective = initial_status["configuration"]
                self.assertRegex(effective["id"], r"^[0-9a-f]{12}$")
                self.assertEqual(
                    effective["controller_parameter_source"],
                    "environment",
                )
                for actual, expected in zip(
                    effective["body"]["center_of_mass_b"],
                    [0.0, 0.0, 0.08],
                ):
                    self.assertAlmostEqual(actual, expected, places=6)
                # Classical sessions own a neutral virtual input frame from
                # creation, so start cannot race physical-gamepad discovery.
                session.step_once()
                deadline = time.monotonic() + 3.0
                telemetry = None
                while telemetry is None and time.monotonic() < deadline:
                    telemetry = session.wait_telemetry(timeout=0.1)
                self.assertIsNotNone(telemetry)
                assert telemetry is not None
                self.assertEqual(
                    telemetry["controller"]["type"], "hybrid_pid_lqr"
                )
                self.assertEqual(
                    len(telemetry["controller"]["command"][0]), 5
                )
                self.assertIn(
                    "controller.lqr_blend",
                    telemetry["controller"]["diagnostics"],
                )
                previous_sequence = telemetry["sequence"]
                session._command_received = 0.0
                self.assertTrue(
                    session.step_once(
                        1,
                        {
                            "roll": 0.75,
                            "pitch": -0.50,
                            "yaw": 0.25,
                            "throttle": 0.0,
                        },
                        trace={
                            "source_sequence": 17,
                            "input_source": "gamepad",
                            "input_captured_epoch_ms": 1000.0,
                            "client_send_epoch_ms": 1001.0,
                        },
                        server_received_ns=time.time_ns(),
                    )
                )
                commanded = None
                deadline = time.monotonic() + 3.0
                while commanded is None and time.monotonic() < deadline:
                    candidate = session.telemetry(previous_sequence)
                    if candidate is not None:
                        commanded = candidate
                    time.sleep(0.01)
                self.assertIsNotNone(commanded)
                assert commanded is not None
                self.assertEqual(
                    commanded["runtime"]["controller_input"]["channels"],
                    {
                        "roll": 0.75,
                        "pitch": -0.5,
                        "yaw": 0.25,
                        "throttle": 0.0,
                    },
                )
                latency_trace = commanded["latency_trace"]
                self.assertEqual(
                    latency_trace["transport_sequence"], 1
                )
                self.assertEqual(
                    latency_trace["source_sequence"], 17
                )
                self.assertEqual(
                    latency_trace["input_source"], "gamepad"
                )
                for key in (
                    "server_control_received_ns",
                    "command_applied_ns",
                    "backend_step_started_ns",
                    "controller_elapsed_ns",
                    "environment_elapsed_ns",
                    "backend_step_finished_ns",
                    "telemetry_pack_started_ns",
                    "telemetry_pack_finished_ns",
                    "telemetry_published_ns",
                ):
                    self.assertIn(key, latency_trace)
                    self.assertGreaterEqual(latency_trace[key], 0)
                target_quaternion = commanded["reference"][
                    "target_attitude_q_wb"
                ][0]
                self.assertGreater(abs(target_quaternion[1]), 1e-4)
                self.assertGreater(abs(target_quaternion[2]), 1e-4)
                self.assertTrue(
                    session.start(
                        2,
                        {
                            "roll": 0.75,
                            "pitch": -0.5,
                            "yaw": 0.25,
                            "throttle": 0.0,
                        },
                    )
                )
                with session._lock:
                    session._command_received = 0.0
                starting_steps = session.status()["control_steps"]
                deadline = time.monotonic() + 3.0
                stale_status = None
                while time.monotonic() < deadline:
                    candidate = session.status()
                    if (
                        candidate["control_steps"] > starting_steps
                        and candidate["controller_input"]["stale"]
                    ):
                        stale_status = candidate
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(stale_status)
                assert stale_status is not None
                self.assertEqual(stale_status["state"], "running")
                self.assertIsNone(stale_status["fault"])
                self.assertGreaterEqual(
                    stale_status["controller_input"]["timeout_events"],
                    1,
                )
                self.assertEqual(
                    session.input_tensor[0].tolist(),
                    [0.0, 0.0, 0.0, -1.0],
                )
                session.pause()
            finally:
                session.close()


class RuntimeRegistryLifecycleTests(unittest.TestCase):
    def test_explicit_replacement_reclaims_orphaned_interactive_slot(
        self,
    ) -> None:
        first = MagicMock()
        first.id = "first-session"
        second = MagicMock()
        second.id = "second-session"
        with tempfile.TemporaryDirectory() as log_root:
            registry = RuntimeRegistry(log_root)
            with patch(
                "runtime.CpuRuntimeSession",
                side_effect=[first, second],
            ) as session_factory:
                self.assertIs(
                    registry.create("sim: first", "test: first", None),
                    first,
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "only one interactive simulation session",
                ):
                    registry.create("sim: blocked", "test: blocked", None)
                self.assertEqual(session_factory.call_count, 1)
                first.close.assert_not_called()

                self.assertIs(
                    registry.create(
                        "sim: replacement",
                        "test: replacement",
                        None,
                        replace_existing=True,
                    ),
                    second,
                )
                first.close.assert_called_once_with()
                with self.assertRaises(KeyError):
                    registry.get(first.id)
                self.assertIs(registry.get(second.id), second)

                registry.close_all()
                second.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
