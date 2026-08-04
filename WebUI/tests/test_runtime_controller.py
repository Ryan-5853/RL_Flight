from __future__ import annotations

import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
import torch

from inference_package import InferencePackageMetadata
from runtime import (
    CpuRuntimeSession,
    RuntimeConfigurationError,
    RuntimeRegistry,
    _simulator_compatibility_fingerprint,
    parse_runtime_options,
)


ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG = ROOT / "SimEnv" / "configs" / "example.yaml"


def _manual_controller_model(simenv: dict) -> dict:
    value = lambda node: node["value"]
    motors = simenv["motors"]
    servos = simenv["servos"]
    aerodynamics = simenv["aerodynamics"]
    grids = aerodynamics["grids"]
    partition = aerodynamics["thrust_partition"]
    return {
        "body": {
            "mass": value(simenv["body"]["mass"]),
            "center_of_mass_b": value(simenv["body"]["center_of_mass_b"]),
            "inertia_diagonal_b": value(
                simenv["body"]["inertia_diagonal_b"]
            ),
        },
        "motors": {
            "upper_pwm_to_rpm_table": value(motors[0]["pwm_to_rpm_table"]),
            "lower_pwm_to_rpm_table": value(motors[1]["pwm_to_rpm_table"]),
            "time_constant": [value(item["time_constant"]) for item in motors],
            "torque_coefficient": [
                value(item["torque_coefficient"]) for item in motors
            ],
        },
        "servos": {
            "tau": [value(item["tau"]) for item in servos],
            **{
                f"servo_{index + 1}_pwm_angle_table": value(
                    item["pwm_angle_table"]
                )
                for index, item in enumerate(servos)
            },
        },
        "aerodynamics": {
            "thrust_coefficients": value(
                aerodynamics["thrust_coefficients"]
            ),
            "neutral_thrust_direction_b": value(
                aerodynamics["neutral_thrust_direction_b"]
            ),
            "direct_thrust_center_b": value(
                aerodynamics["direct_thrust_center_b"]
            ),
            "thrust_partition": [
                value(partition[name])
                for name in ("direct", "grid_1", "grid_2", "grid_3")
            ],
            "coupling_attenuation": value(
                aerodynamics["coupling_attenuation"]
            ),
            "grids": {
                "aerodynamic_center_b": [
                    value(item["aerodynamic_center_b"]) for item in grids
                ],
                "deflection_axis_b": [
                    value(item["deflection_axis_b"]) for item in grids
                ],
                **{
                    f"grid_{index + 1}_self_attenuation_curve": value(
                        item["self_attenuation_curve"]
                    )
                    for index, item in enumerate(grids)
                },
                "vector_deflection_gain": [
                    value(item["vector_deflection"]["gain"])
                    for item in grids
                ],
                "vector_deflection_offset": [
                    value(item["vector_deflection"]["offset"])
                    for item in grids
                ],
            },
        },
    }


class _ZeroInferencePackage:
    metadata = InferencePackageMetadata(
        format_version=1,
        package_id="zero-policy",
        observation_dim=21,
        action_dim=4,
        output_mode="residual_4",
    )

    def infer(self, observation, recurrent_state, is_init):
        del recurrent_state, is_init
        return observation.new_zeros((1, 4)), None

    def reset(self):
        return None

    def warmup(self, observation):
        self.infer(
            observation,
            None,
            torch.ones((1, 1), device=observation.device, dtype=torch.bool),
        )

    def close(self):
        return None

    def describe(self):
        return {"package_id": self.metadata.package_id}


class _ZeroCascadeInferencePackage(_ZeroInferencePackage):
    metadata = InferencePackageMetadata(
        format_version=1,
        package_id="zero-cascade-policy",
        observation_dim=21,
        action_dim=3,
        output_mode="coaxial_differential_cyclic_3",
    )

    def infer(self, observation, recurrent_state, is_init):
        del recurrent_state, is_init
        return observation.new_zeros((1, 3)), None

    def infer_control(
        self,
        state,
        reference,
        previous_action,
        recurrent_state,
        is_init,
    ):
        del state, reference, recurrent_state, is_init
        return previous_action.new_zeros((1, 3)), None

    def action_to_command(self, policy_action, external_action):
        zeros = policy_action.new_zeros((1, 3))
        return torch.cat(
            (
                external_action.clamp(0.0, 1.0),
                external_action.clamp(0.0, 1.0) * 0.947558738884,
                zeros,
            ),
            dim=1,
        )


class RuntimeOptionsTests(unittest.TestCase):
    def test_simulator_fingerprint_ignores_nonphysical_top_level_fields(self) -> None:
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

    def test_simulator_fingerprint_normalizes_webui_round_trip(self) -> None:
        baseline = yaml.safe_load(
            (
                ROOT
                / "Train"
                / "configs"
                / "environment"
                / "gru_sac_upright_height_only_small_tip.yaml"
            ).read_text(encoding="utf-8")
        )
        equivalent = copy.deepcopy(baseline)
        equivalent["schema_version"] = 99
        equivalent["parallel"] = {"independent_rng": False}
        equivalent["motors"][0]["name"] = "display-only-name"
        equivalent["body"]["mass"]["randomization"] = {
            "distribution": "none",
            "mode": "relative",
            "stddev": 0.05,
        }
        equivalent["sensors"]["motor_speed"]["interpolation"] = "linear"

        self.assertEqual(
            _simulator_compatibility_fingerprint(baseline),
            _simulator_compatibility_fingerprint(equivalent),
        )
        equivalent["body"]["mass"]["value"] = 2.0
        self.assertNotEqual(
            _simulator_compatibility_fingerprint(baseline),
            _simulator_compatibility_fingerprint(equivalent),
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
    def test_cascade_neural_hover_keeps_collective_external(self) -> None:
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
                "type": "neural",
                "params": {
                    "collective_mode": "hover",
                    "flight_mode": "attitude",
                },
            },
            "command_source": {
                "params": {
                    "throttle": {
                        "minimum": 0.2,
                        "maximum": 0.85,
                        "slew_rate": {
                            "rise_per_s": 4.0,
                            "fall_per_s": 0.35,
                        },
                    }
                }
            },
            "task": {"episode_duration_s": 1.0},
            "randomization": {"dynamic": {"parameters": {}}},
        }
        simenv_config = yaml.safe_load(
            SIM_CONFIG.read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as log_root:
            session = CpuRuntimeSession(
                yaml.safe_dump(simenv_config),
                yaml.safe_dump(test_config),
                Path(log_root) / "zero-cascade-policy",
                log_root,
                inference_package_loader=(
                    lambda path, device, dtype: _ZeroCascadeInferencePackage()
                ),
            )
            try:
                self.assertEqual(
                    session.status()["controller"]["height_controller"],
                    "shared_altitude_pid",
                )
                self.assertEqual(
                    session.status()["controller"]["height_controller_backend"],
                    "scalar_b1",
                )
                self.assertTrue(session.status()["flush_denormal"])
                with torch.no_grad():
                    session._step_cpu(inspect_safety=False)
                command = session.last_controller_output.command
                collective = session.last_reference.collective_command
                torch.testing.assert_close(command[:, :1], collective)
                torch.testing.assert_close(
                    command[:, 1:2], collective * 0.947558738884
                )
                torch.testing.assert_close(
                    command[:, 2:], torch.zeros_like(command[:, 2:])
                )
            finally:
                session.close()

    def test_neural_hover_uses_shared_realtime_altitude_pid(
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
                "type": "neural",
                "params": {
                    "collective_mode": "hover",
                    "flight_mode": "position",
                    "position": {
                        "kp": 1.0,
                        "kd": 1.6,
                        "maximum_acceleration_m_s2": 3.0,
                        "maximum_tilt_rad": 0.35,
                    },
                    "pid": {
                        "altitude": {
                            "kp": 6.0,
                            "ki": 0.0,
                            "kd": 0.0,
                        }
                    },
                },
            },
            "command_source": {
                "type": "flight_train.commands:VirtualPilotCommandSource",
                "version": "2",
                "seed": 51002,
                "params": {
                    "throttle": {
                        "minimum": 0.2,
                        "maximum": 0.85,
                        "spool": {
                            "duration_s": 0.0,
                            "target_range": [0.4, 0.4],
                        },
                        "height_controller": {
                            "observation_source": "truth",
                            "target_m": 1.0,
                            "initial_throttle_range": [0.4, 0.4],
                            "proportional_gain": 0.1,
                            "integral_gain": 0.0,
                            "error_limit_m": 5.0,
                        },
                        "slew_rate": {
                            "rise_per_s": 100.0,
                            "fall_per_s": 100.0,
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
                "episode_duration_s": 30,
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
                Path(log_root) / "zero-policy",
                log_root,
                inference_package_loader=(
                    lambda path, device, dtype: _ZeroInferencePackage()
                ),
            )
            try:
                self.assertEqual(
                    session.status()["controller"]["height_controller"],
                    "shared_altitude_pid",
                )
                self.assertEqual(
                    session.realtime_height_controller.height_kp,
                    6.0,
                )
                # The throttle stick requests the minimum. A one-metre upward
                # target must instead make the shared PID command above trim.
                session.input_tensor[0, 3] = -1.0
                session.update_target_position([1.0, 0.0, -1.0])
                self.assertEqual(
                    session.status()["target_position_n"],
                    [1.0, 0.0, -1.0],
                )
                with self.assertRaisesRegex(ValueError, "three-element"):
                    session.update_target_position([1.0, 0.0])
                with torch.no_grad():
                    session._step_cpu(inspect_safety=False)
                self.assertGreater(
                    float(session.last_reference.collective_command[0, 0]),
                    0.5,
                )
                self.assertAlmostEqual(
                    float(session.last_height_controller_output["error_m"][0]),
                    1.0,
                    places=5,
                )
                expected_thrust = min(
                    float(session.controller_parameters["body.mass"][0])
                    * (9.80665 + 6.0),
                    session.realtime_height_controller.maximum_thrust,
                )
                self.assertAlmostEqual(
                    float(
                        session.last_height_controller_output[
                            "desired_thrust_n"
                        ][0]
                    ),
                    expected_thrust,
                    places=4,
                )
                target_attitude = session.last_reference.target_attitude_q_wb
                # Positive North acceleration requires negative pitch because
                # thrust acts along -Z_body in the NED/FRD convention.
                self.assertLess(float(target_attitude[0, 2]), -0.01)
                self.assertAlmostEqual(
                    float(
                        session.last_position_controller_output[
                            "position_error_n_m"
                        ][0, 0]
                    ),
                    1.0,
                    places=5,
                )
                session._publish_telemetry({})
                telemetry = session.telemetry()
                assert telemetry is not None
                self.assertEqual(
                    telemetry["height_controller"]["mode"],
                    "shared_altitude_pid",
                )
                self.assertAlmostEqual(
                    telemetry["height_controller"]["target_m"][0],
                    1.0,
                    places=5,
                )
                self.assertEqual(
                    telemetry["position_controller"]["mode"],
                    "position_outer_loop",
                )
            finally:
                session.close()

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
                "type": "hybrid_pid_lqr",
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
                "params": {
                    "collective_mode": "hover",
                    "flight_mode": "position",
                    "position": {
                        "kp": 1.0,
                        "kd": 1.6,
                        "maximum_acceleration_m_s2": 3.0,
                        "maximum_tilt_rad": 0.35,
                    },
                },
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
                    "synchronized",
                )
                self.assertEqual(
                    effective["controller_model_mismatch"][
                        "different_parameter_count"
                    ],
                    0,
                )
                self.assertNotEqual(
                    session.environment.parameters["body.mass"].data_ptr(),
                    session.controller.context.parameters["body.mass"].data_ptr(),
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
                session.update_target_position([1.0, -1.0, 0.0])
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
                # Target N>0/E<0 requires negative pitch and negative roll at
                # near-zero yaw under NED/FRD thrust polarity.
                self.assertLess(target_quaternion[1], -1e-4)
                self.assertLess(target_quaternion[2], -1e-4)
                initial_position = session.environment.observe(
                    "truth", ("position_n",)
                ).values["position_n"].clone()
                with torch.no_grad():
                    for _ in range(300):
                        session._step_cpu(inspect_safety=False)
                moved_position = session.environment.observe(
                    "truth", ("position_n",)
                ).values["position_n"]
                self.assertGreater(
                    float(moved_position[0, 0] - initial_position[0, 0]),
                    0.01,
                )
                self.assertLess(
                    float(moved_position[0, 1] - initial_position[0, 1]),
                    -0.01,
                )
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

    def test_manual_controller_model_is_independent_from_simulation_parameters(
        self,
    ) -> None:
        simenv_config = yaml.safe_load(
            SIM_CONFIG.read_text(encoding="utf-8")
        )
        manual = _manual_controller_model(simenv_config)
        manual["body"]["mass"] = 1.2
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
                "params": {
                    "collective_mode": "hover",
                    "model_parameters": {
                        "source": "manual",
                        "manual": manual,
                    },
                },
            },
            "command_source": {
                "params": {
                    "throttle": {
                        "minimum": 0.2,
                        "maximum": 0.85,
                        "slew_rate": {
                            "rise_per_s": 4.0,
                            "fall_per_s": 0.35,
                        },
                    }
                }
            },
            "task": {"episode_duration_s": 1},
            "randomization": {"dynamic": {"parameters": {}}},
        }
        with tempfile.TemporaryDirectory() as log_root:
            session = CpuRuntimeSession(
                yaml.safe_dump(simenv_config),
                yaml.safe_dump(test_config),
                None,
                log_root,
            )
            try:
                actual_mass = float(
                    session.environment.parameters["body.mass"][0]
                )
                controller_mass = float(
                    session.controller.context.parameters["body.mass"][0]
                )
                self.assertNotAlmostEqual(actual_mass, controller_mass)
                self.assertAlmostEqual(controller_mass, 1.2, places=6)
                self.assertNotEqual(
                    session.environment.parameters["body.mass"].data_ptr(),
                    session.controller.context.parameters["body.mass"].data_ptr(),
                )
                self.assertAlmostEqual(
                    float(session.controller.trim.thrust[0]),
                    controller_mass * session.controller.plant.gravity,
                    places=4,
                )
                self.assertIsNotNone(session.controller._lqr_gain)
                effective = session.status()["configuration"]
                self.assertEqual(
                    effective["controller_parameter_source"], "manual"
                )
                self.assertAlmostEqual(
                    effective["controller_model"]["body"]["mass"],
                    1.2,
                    places=6,
                )
                self.assertIn(
                    "body.mass",
                    effective["controller_model_mismatch"][
                        "different_parameters"
                    ],
                )
                session.step_once()
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
