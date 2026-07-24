from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from simenv import ConfigurationError, ErrorCode, SimulationEnvironment


def _randomizable(value, randomization=None):
    node = {"value": value}
    if randomization is not None:
        node["randomization"] = randomization
    return node


def _config(log_directory: str) -> dict:
    motor = {
        "name": "upper",
        "pwm_deadzone": _randomizable(0.08),
        "pwm_to_rpm_table": _randomizable([[0.0, 0.0], [0.08, 0.0], [1.0, 1800.0]]),
        "time_constant": _randomizable(0.03),
        "torque_coefficient": _randomizable(1.0e-7),
        "noise": {"distribution": "normal", "stddev": _randomizable(5.0)},
    }
    servo = {
        "name": "servo_1",
        "pwm_angle_table": _randomizable([[-1.0, -0.35], [0.0, 0.0], [1.0, 0.35]]),
        "tau": _randomizable(0.02),
        "max_speed": _randomizable(8.0),
        "backlash": _randomizable(0.01),
        "deadzone": _randomizable(0.015),
    }
    grid = {
        "name": "grid_1",
        "aerodynamic_center_b": _randomizable([0.1, 0.0, 0.25]),
        "deflection_axis_b": _randomizable([1.0, 0.0, 0.0]),
        "self_attenuation_curve": _randomizable([[0.0, 1.0], [0.35, 0.85]]),
        "vector_deflection": {"gain": _randomizable(1.0), "offset": _randomizable(0.0)},
    }
    lower_motor = deepcopy(motor)
    lower_motor["name"] = "lower"
    lower_motor["time_constant"] = _randomizable(0.05)
    lower_motor["torque_coefficient"] = _randomizable(1.2e-7)
    servos = []
    for index in range(3):
        item = deepcopy(servo)
        item["name"] = f"servo_{index + 1}"
        servos.append(item)
    grids = []
    grid_axes = (
        [1.0, 0.0, 0.0],
        [-0.5, 0.8660254037844386, 0.0],
        [-0.5, -0.8660254037844386, 0.0],
    )
    for index in range(3):
        item = deepcopy(grid)
        item["name"] = f"grid_{index + 1}"
        item["deflection_axis_b"] = _randomizable(grid_axes[index])
        grids.append(item)
    return {
        "schema_version": 1,
        "seed": 1234,
        "parallel": {"independent_rng": True},
        "timing": {"physics_hz": _randomizable(5000), "control_hz": _randomizable(500)},
        "initial_state": {
            "position_n": _randomizable([0.0, 0.0, 0.0]),
            "velocity_n": _randomizable([0.0, 0.0, 0.0]),
            "attitude_q_wb": _randomizable([1.0, 0.0, 0.0, 0.0]),
            "angular_velocity_b": _randomizable([0.0, 0.0, 0.0]),
        },
        "body": {
            "mass": _randomizable(
                2.4,
                {
                    "distribution": "normal",
                    "mode": "relative",
                    "mean": 0.0,
                    "stddev": 0.05,
                    "clip": [-0.2, 0.2],
                },
            ),
            "center_of_mass_b": _randomizable([0.0, 0.0, 0.08]),
            "inertia_diagonal_b": _randomizable([0.03, 0.028, 0.012]),
        },
        "motors": [motor, lower_motor],
        "servos": servos,
        "aerodynamics": {
            "thrust_coefficients": _randomizable([4.0e-6, 4.0e-6, 1.0e-6]),
            "neutral_thrust_direction_b": _randomizable([0.0, 0.0, -1.0]),
            "direct_thrust_center_b": _randomizable([0.0, 0.0, 0.2]),
            "thrust_partition": {
                "direct": _randomizable(0.4),
                "grid_1": _randomizable(0.2),
                "grid_2": _randomizable(0.2),
                "grid_3": _randomizable(0.2),
            },
            "grids": grids,
            "coupling_attenuation": _randomizable(
                [[0.0, 0.1, 0.1], [0.1, 0.0, 0.1], [0.1, 0.1, 0.0]]
            ),
        },
        "sensors": {
            "gyro": {
                "sample_hz": _randomizable(5000),
                "noise": {"distribution": "normal", "stddev": _randomizable([0.002] * 3)},
                "bias": _randomizable([0.0] * 3),
                "delay": _randomizable(0.001),
            }
        },
        "logging": {
            "directory": log_directory,
            "chunk_steps": 4,
            "queue_chunks": 2,
            "overflow": "block",
        },
    }


class SimulationEnvironmentTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        return self._write_named_config(root, "config.json", _config(str(root / "logs")))

    def _write_named_config(self, root: Path, name: str, config: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    @staticmethod
    def _disable_motor_noise(config: dict) -> None:
        for motor in config["motors"]:
            motor["noise"]["stddev"] = _randomizable(0.0)

    @staticmethod
    def _zero_sensor_delays(config: dict) -> None:
        physics_hz = config["timing"]["physics_hz"]["value"]
        for sensor in config["sensors"].values():
            sensor["sample_hz"] = _randomizable(physics_hz)
            sensor["delay"] = _randomizable(0.0)

    def test_create_materializes_all_parameters_with_batch_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SimulationEnvironment.create(self._write_config(root), 4, "cpu") as env:
                self.assertEqual(env.batch_shape, torch.Size([4]))
                self.assertTrue(env.dynamics_implemented)
                self.assertTrue(env.sensors_implemented)
                for tensor in env.parameters.values():
                    self.assertEqual(tensor.shape[0], 4)
                self.assertEqual(env.parameters["motors.pwm_to_rpm_table"].shape, (4, 2, 3, 2))
                self.assertEqual(env.parameters["servos.pwm_angle_table"].shape, (4, 3, 3, 2))
                self.assertEqual(env.parameters["aerodynamics.coupling_attenuation"].shape, (4, 3, 3))

    def test_dynamic_randomization_uses_environment_nominal_and_rejects_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_config(root)
            spec = {
                "sensors.gyro.noise.stddev": {
                    "distribution": "normal",
                    "stddev": [0.0, 0.0, 0.0],
                    "seed_stream": "dynamic.sensor.gyro.noise",
                }
            }
            with SimulationEnvironment.create(
                path, 3, "cpu", dynamic_randomization=spec, dynamic_seed=7
            ) as env:
                torch.testing.assert_close(
                    env.parameters["sensors.gyro.noise.stddev"],
                    torch.full((3, 3), 0.002),
                )

            duplicate = deepcopy(spec)
            duplicate["sensors.gyro.noise.stddev"]["baseline"] = [0.002] * 3
            with self.assertRaisesRegex(ConfigurationError, "unknown dynamic randomization fields"):
                SimulationEnvironment.create(
                    path, 3, "cpu", dynamic_randomization=duplicate, dynamic_seed=7
                )

    def test_observation_is_batched_and_does_not_expose_internal_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SimulationEnvironment.create(self._write_config(root), 1, "cpu") as env:
                first = env.observe("truth", ("position_n",))
                self.assertEqual(first.values["position_n"].shape, (1, 3))
                first.values["position_n"][0, 0] = 99
                second = env.observe("truth", ("position_n",))
                self.assertEqual(second.values["position_n"][0, 0].item(), 0.0)

    def test_complete_state_round_trip_replays_next_steps_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_config(root)
            batch = 3
            prefix = [
                torch.tensor([[0.52, 0.48, 0.1, -0.2, 0.3]]).expand(batch, -1),
                torch.tensor([[0.61, 0.43, -0.3, 0.2, -0.1]]).expand(batch, -1),
                torch.tensor([[0.57, 0.51, 0.2, 0.1, -0.2]]).expand(batch, -1),
            ]
            suffix = [
                torch.tensor([[0.63, 0.46, -0.1, 0.3, 0.2]]).expand(batch, -1),
                torch.tensor([[0.55, 0.54, 0.3, -0.1, -0.3]]).expand(batch, -1),
            ]
            env_a = SimulationEnvironment.create(path, batch, "cpu")
            env_b = SimulationEnvironment.create(path, batch, "cpu")
            try:
                for control in prefix:
                    env_a.advance(control)
                checkpoint = env_a.state_dict()

                expected = []
                for control in suffix:
                    result = env_a.advance(control)
                    expected.append(
                        (
                            result,
                            dict(env_a.observe("truth").values),
                            dict(env_a.observe("sensor").values),
                        )
                    )

                env_b.load_state_dict(checkpoint)
                for control, (expected_result, expected_truth, expected_sensor) in zip(
                    suffix, expected
                ):
                    actual_result = env_b.advance(control)
                    for name in (
                        "physics_step", "control_step", "sim_time_s",
                        "physics_steps_advanced", "valid", "error_code",
                    ):
                        torch.testing.assert_close(
                            getattr(actual_result, name), getattr(expected_result, name)
                        )
                    actual_truth = env_b.observe("truth").values
                    actual_sensor = env_b.observe("sensor").values
                    self.assertEqual(set(actual_truth), set(expected_truth))
                    self.assertEqual(set(actual_sensor), set(expected_sensor))
                    for name, value in expected_truth.items():
                        torch.testing.assert_close(actual_truth[name], value, rtol=0, atol=0)
                    for name, value in expected_sensor.items():
                        torch.testing.assert_close(actual_sensor[name], value, rtol=0, atol=0)

                final_a = env_a.state_dict()
                final_b = env_b.state_dict()
                for section in (
                    "parameters", "truth", "sensors", "random_counters"
                ):
                    for name, value in final_a[section].items():
                        torch.testing.assert_close(
                            final_b[section][name], value, rtol=0, atol=0
                        )
                for name in (
                    "physics_step", "control_step", "valid", "error_code",
                    "generation", "instance_seeds", "control",
                ):
                    torch.testing.assert_close(final_b[name], final_a[name], rtol=0, atol=0)
                for name, value in final_a["sensor_kernel"]["history"].items():
                    torch.testing.assert_close(
                        final_b["sensor_kernel"]["history"][name], value, rtol=0, atol=0
                    )
            finally:
                env_a.close()
                env_b.close()

    def test_complete_state_rejects_incompatible_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_config(root)
            with SimulationEnvironment.create(path, 2, "cpu") as source:
                state = source.state_dict()
            with SimulationEnvironment.create(path, 3, "cpu") as target:
                with self.assertRaises(ConfigurationError):
                    target.load_state_dict(state)

    def test_rigid_body_free_fall_uses_ned_gravity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["timing"] = {
                "physics_hz": _randomizable(100),
                "control_hz": _randomizable(100),
            }
            self._disable_motor_noise(config)
            self._zero_sensor_delays(config)
            path = self._write_named_config(root, "free-fall.json", config)
            with SimulationEnvironment.create(path, 2, "cpu") as env:
                env.advance(torch.zeros((2, 5)))
                truth = env.observe("truth").values
                expected_acceleration = torch.tensor([[0.0, 0.0, 9.80665]]).expand(2, 3)
                torch.testing.assert_close(
                    truth["linear_acceleration_n"], expected_acceleration
                )
                torch.testing.assert_close(
                    truth["velocity_n"], expected_acceleration * 0.01
                )
                torch.testing.assert_close(
                    truth["position_n"], expected_acceleration * 0.0001
                )
                torch.testing.assert_close(truth["force_b"], torch.zeros((2, 3)))
                torch.testing.assert_close(truth["moment_b"], torch.zeros((2, 3)))

    def test_coaxial_motor_thrust_and_reaction_torque(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["timing"] = {
                "physics_hz": _randomizable(100),
                "control_hz": _randomizable(100),
            }
            config["body"]["mass"] = _randomizable(1.0)
            config["body"]["center_of_mass_b"] = _randomizable([0.0, 0.0, 0.0])
            config["aerodynamics"]["direct_thrust_center_b"] = _randomizable([0.0, 0.0, 0.0])
            config["aerodynamics"]["thrust_partition"] = {
                "direct": _randomizable(1.0),
                "grid_1": _randomizable(0.0),
                "grid_2": _randomizable(0.0),
                "grid_3": _randomizable(0.0),
            }
            for motor in config["motors"]:
                motor["pwm_deadzone"] = _randomizable(0.0)
                motor["pwm_to_rpm_table"] = _randomizable([[0.0, 0.0], [1.0, 100.0]])
                motor["time_constant"] = _randomizable(0.01)
                motor["torque_coefficient"] = _randomizable(1.0e-4)
            config["aerodynamics"]["thrust_coefficients"] = _randomizable(
                [1.0e-3, 2.0e-3, 3.0e-4]
            )
            self._disable_motor_noise(config)
            self._zero_sensor_delays(config)
            path = self._write_named_config(root, "coaxial.json", config)

            with SimulationEnvironment.create(path, 2, "cpu") as env:
                control = torch.zeros((2, 5))
                control[0, :2] = 1.0
                control[1, 0] = 1.0
                env.advance(control)
                truth = env.observe("truth").values
                expected_speed = 100.0 * (1.0 - torch.exp(torch.tensor(-1.0)))
                torch.testing.assert_close(
                    truth["motor_speed"][0], expected_speed.expand(2)
                )
                self.assertAlmostEqual(truth["moment_b"][0, 2].item(), 0.0, places=6)
                self.assertGreater(truth["moment_b"][1, 2].item(), 0.0)
                speed = expected_speed.item()
                self.assertAlmostEqual(
                    truth["total_thrust"][0].item(),
                    (1.0e-3 + 2.0e-3 + 3.0e-4) * speed * speed,
                    places=5,
                )
                self.assertAlmostEqual(
                    truth["total_thrust"][1].item(), 1.0e-3 * speed * speed, places=5
                )
                self.assertLess(truth["force_b"][0, 2].item(), 0.0)
                torch.testing.assert_close(
                    truth["force_b"][0, :2], torch.zeros(2), atol=1e-6, rtol=0
                )

    def test_upper_and_lower_motor_parameters_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["timing"] = {
                "physics_hz": _randomizable(100),
                "control_hz": _randomizable(100),
            }
            config["aerodynamics"]["thrust_coefficients"] = _randomizable(
                [1.0e-3, 2.0e-3, 3.0e-4]
            )
            for motor in config["motors"]:
                motor["pwm_deadzone"] = _randomizable(0.0)
                motor["pwm_to_rpm_table"] = _randomizable(
                    [[0.0, 0.0], [1.0, 100.0]]
                )
            config["motors"][0]["time_constant"] = _randomizable(0.01)
            config["motors"][1]["time_constant"] = _randomizable(0.02)
            config["motors"][0]["torque_coefficient"] = _randomizable(1.0e-4)
            config["motors"][1]["torque_coefficient"] = _randomizable(2.0e-4)
            self._disable_motor_noise(config)
            self._zero_sensor_delays(config)
            path = self._write_named_config(root, "independent-motors.json", config)

            with SimulationEnvironment.create(path, 1, "cpu") as env:
                env.advance(torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]))
                truth = env.observe("truth").values
                upper_speed = 100.0 * (1.0 - torch.exp(torch.tensor(-1.0)))
                lower_speed = 100.0 * (1.0 - torch.exp(torch.tensor(-0.5)))
                expected_speeds = torch.stack((upper_speed, lower_speed))
                torch.testing.assert_close(truth["motor_speed"][0], expected_speeds)

                expected_torques = torch.tensor([1.0e-4, 2.0e-4]) * expected_speeds.square()
                torch.testing.assert_close(truth["motor_torque"][0], expected_torques)
                self.assertAlmostEqual(
                    truth["motor_reaction_moment_b"][0, 2].item(),
                    (expected_torques[0] - expected_torques[1]).item(),
                    places=5,
                )
                expected_thrust = (
                    1.0e-3 * upper_speed.square()
                    + 2.0e-3 * lower_speed.square()
                    + 3.0e-4 * upper_speed * lower_speed
                )
                torch.testing.assert_close(truth["total_thrust"][0], expected_thrust)

    def test_body_thrust_is_rotated_to_ned_by_q_wb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["timing"] = {
                "physics_hz": _randomizable(100),
                "control_hz": _randomizable(100),
            }
            config["initial_state"]["attitude_q_wb"] = _randomizable(
                [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]
            )
            config["body"]["mass"] = _randomizable(1.0)
            config["aerodynamics"]["thrust_partition"] = {
                "direct": _randomizable(1.0),
                "grid_1": _randomizable(0.0),
                "grid_2": _randomizable(0.0),
                "grid_3": _randomizable(0.0),
            }
            for motor in config["motors"]:
                motor["pwm_deadzone"] = _randomizable(0.0)
                motor["pwm_to_rpm_table"] = _randomizable([[0.0, 0.0], [1.0, 100.0]])
                motor["time_constant"] = _randomizable(0.01)
                motor["torque_coefficient"] = _randomizable(0.0)
            config["aerodynamics"]["thrust_coefficients"] = _randomizable(
                [1.0e-3, 1.0e-3, 0.0]
            )
            self._disable_motor_noise(config)
            self._zero_sensor_delays(config)
            path = self._write_named_config(root, "rotated-thrust.json", config)
            with SimulationEnvironment.create(path, 1, "cpu") as env:
                env.advance(torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]))
                truth = env.observe("truth").values
                total_thrust = truth["total_thrust"].squeeze(0)
                self.assertAlmostEqual(
                    truth["linear_acceleration_n"][0, 0].item(),
                    -total_thrust.item(),
                    places=5,
                )
                self.assertAlmostEqual(
                    truth["linear_acceleration_n"][0, 2].item(), 9.80665, places=5
                )

    def test_grid_vector_force_and_offset_moment_follow_right_hand_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["timing"] = {
                "physics_hz": _randomizable(100),
                "control_hz": _randomizable(100),
            }
            config["body"]["center_of_mass_b"] = _randomizable([0.0, 0.0, 0.0])
            config["aerodynamics"]["thrust_partition"] = {
                "direct": _randomizable(0.0),
                "grid_1": _randomizable(1.0),
                "grid_2": _randomizable(0.0),
                "grid_3": _randomizable(0.0),
            }
            config["aerodynamics"]["coupling_attenuation"] = _randomizable(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
            config["aerodynamics"]["grids"][0]["aerodynamic_center_b"] = _randomizable(
                [0.0, 0.0, 1.0]
            )
            config["aerodynamics"]["grids"][0]["deflection_axis_b"] = _randomizable(
                [1.0, 0.0, 0.0]
            )
            for grid in config["aerodynamics"]["grids"]:
                grid["self_attenuation_curve"] = _randomizable(
                    [[0.0, 1.0], [2.0, 1.0]]
                )
            for motor in config["motors"]:
                motor["pwm_deadzone"] = _randomizable(0.0)
                motor["pwm_to_rpm_table"] = _randomizable([[0.0, 0.0], [1.0, 100.0]])
                motor["time_constant"] = _randomizable(0.01)
                motor["torque_coefficient"] = _randomizable(0.0)
            config["aerodynamics"]["thrust_coefficients"] = _randomizable(
                [1.0e-3, 1.0e-3, 0.0]
            )
            for servo in config["servos"]:
                servo["pwm_angle_table"] = _randomizable(
                    [[-1.0, -1.5707963267948966], [0.0, 0.0], [1.0, 1.5707963267948966]]
                )
                servo["tau"] = _randomizable(0.01)
                servo["max_speed"] = _randomizable(1000.0)
                servo["backlash"] = _randomizable(0.0)
                servo["deadzone"] = _randomizable(0.0)
            self._disable_motor_noise(config)
            self._zero_sensor_delays(config)
            path = self._write_named_config(root, "grid-vector.json", config)

            with SimulationEnvironment.create(path, 1, "cpu") as env:
                control = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
                env.advance(control)
                truth = env.observe("truth").values
                thrust = truth["total_thrust"].item()
                self.assertAlmostEqual(truth["grid_force_b"][0, 0, 0].item(), 0.0, places=5)
                self.assertAlmostEqual(truth["grid_force_b"][0, 0, 1].item(), thrust, places=5)
                self.assertAlmostEqual(truth["grid_force_b"][0, 0, 2].item(), 0.0, places=5)
                self.assertAlmostEqual(truth["grid_moment_b"][0, 0, 0].item(), -thrust, places=5)
                self.assertAlmostEqual(truth["moment_b"][0, 0].item(), -thrust, places=5)

    def test_grid_coupling_attenuation_matches_documented_matrix_formula(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(root)
            with SimulationEnvironment.create(config, 2, "cpu") as env:
                components = env._dynamics._forces_and_moments(
                    env._parameters,
                    torch.ones(2),
                    torch.zeros((2, 2)),
                    torch.tensor([[0.35, 0.0, 0.0], [0.0, 0.35, 0.0]]),
                )
                torch.testing.assert_close(
                    components["grid_effective_attenuation"],
                    torch.tensor([[0.85, 0.985, 0.985], [0.985, 0.85, 0.985]]),
                    atol=1e-6,
                    rtol=0,
                )

    def test_servo_deadzone_and_reversal_backlash_are_stateful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["timing"] = {
                "physics_hz": _randomizable(100),
                "control_hz": _randomizable(100),
            }
            for servo in config["servos"]:
                servo["pwm_angle_table"] = _randomizable(
                    [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
                )
                servo["tau"] = _randomizable(1.0)
                servo["max_speed"] = _randomizable(100.0)
                servo["backlash"] = _randomizable(0.5)
                servo["deadzone"] = _randomizable(0.1)
            self._disable_motor_noise(config)
            self._zero_sensor_delays(config)
            path = self._write_named_config(root, "servo-state.json", config)
            with SimulationEnvironment.create(path, 1, "cpu") as env:
                env.advance(torch.tensor([[0.0, 0.0, 0.05, 0.0, 0.0]]))
                self.assertEqual(env._truth["servo_effective_pwm"][0, 0].item(), 0.0)
                env.advance(torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]))
                self.assertEqual(env._truth["servo_target_angle"][0, 0].item(), 1.0)
                env.advance(torch.tensor([[0.0, 0.0, -1.0, 0.0, 0.0]]))
                self.assertAlmostEqual(
                    env._truth["servo_target_angle"][0, 0].item(), -0.5, places=6
                )

    def test_sensor_sampling_and_integer_delay_use_physics_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["timing"] = {
                "physics_hz": _randomizable(100),
                "control_hz": _randomizable(50),
            }
            gyro = config["sensors"]["gyro"]
            for motor in config["motors"]:
                motor["noise"]["stddev"] = _randomizable(0.0)
            gyro["sample_hz"] = _randomizable(50)
            gyro["noise"]["stddev"] = _randomizable([0.0, 0.0, 0.0])
            gyro["delay"] = _randomizable(0.02)
            path = self._write_named_config(root, "sensor-delay.json", config)

            with SimulationEnvironment.create(path, 1, "cpu") as env:
                env._truth["angular_velocity_b"][0, 0] = 10.0
                control = torch.zeros((1, 5))
                env.advance(control)
                first = env.observe("sensor", ("gyro",))
                torch.testing.assert_close(first.values["gyro"], torch.zeros((1, 3)))
                env.advance(control)
                second = env.observe("sensor", ("gyro",))
                torch.testing.assert_close(
                    second.values["gyro"], torch.tensor([[10.0, 0.0, 0.0]])
                )

    def test_sensor_fractional_delay_uses_explicit_linear_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["timing"] = {
                "physics_hz": _randomizable(100),
                "control_hz": _randomizable(100),
            }
            gyro = config["sensors"]["gyro"]
            for motor in config["motors"]:
                motor["noise"]["stddev"] = _randomizable(0.0)
            gyro["sample_hz"] = _randomizable(100)
            gyro["noise"]["stddev"] = _randomizable([0.0, 0.0, 0.0])
            gyro["delay"] = _randomizable(0.015)
            gyro["interpolation"] = "linear"
            path = self._write_named_config(root, "sensor-linear.json", config)

            with SimulationEnvironment.create(path, 1, "cpu") as env:
                env._truth["angular_velocity_b"][0, 0] = 10.0
                control = torch.zeros((1, 5))
                env.advance(control)
                torch.testing.assert_close(
                    env.observe("sensor").values["gyro"], torch.zeros((1, 3))
                )
                env.advance(control)
                torch.testing.assert_close(
                    env.observe("sensor").values["gyro"],
                    torch.tensor([[5.0, 0.0, 0.0]]),
                )

    def test_sensor_bias_and_reference_frame_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["initial_state"]["attitude_q_wb"] = _randomizable(
                [0.0, 1.0, 0.0, 0.0]
            )
            config["sensors"] = {
                "gyro": {
                    "noise": {"distribution": "normal", "stddev": _randomizable([0.0] * 3)},
                    "bias": _randomizable([0.1, 0.2, 0.3]),
                    "delay": _randomizable(0.0),
                },
                "accelerometer": {
                    "noise": {"distribution": "normal", "stddev": _randomizable([0.0] * 3)},
                    "bias": _randomizable([0.0] * 3),
                    "delay": _randomizable(0.0),
                },
                "motor_speed": {
                    "noise": {"distribution": "normal", "stddev": _randomizable([0.0] * 2)},
                    "bias": _randomizable([1.0, -1.0]),
                    "delay": _randomizable(0.0),
                },
            }
            path = self._write_named_config(root, "sensor-mapping.json", config)
            with SimulationEnvironment.create(path, 2, "cpu") as env:
                values = env.observe("sensor").values
                torch.testing.assert_close(
                    values["gyro"], torch.tensor([[0.1, 0.2, 0.3]]).expand(2, 3)
                )
                torch.testing.assert_close(
                    values["accelerometer"],
                    torch.tensor([[0.0, 0.0, 9.80665]]).expand(2, 3),
                )
                torch.testing.assert_close(
                    values["motor_speed"], torch.tensor([[1.0, -1.0]]).expand(2, 2)
                )

    def test_sensor_noise_stream_is_isolated_by_instance_counter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["sensors"]["gyro"]["delay"] = _randomizable(0.0)
            path = self._write_named_config(root, "sensor-noise.json", config)
            control = torch.zeros((3, 5))
            with SimulationEnvironment.create(path, 3, "cpu") as all_active:
                all_active.advance(control)
                expected = all_active.observe("sensor").values["gyro"]
            with SimulationEnvironment.create(path, 3, "cpu") as partially_active:
                partially_active.advance(
                    control, torch.tensor([True, False, True])
                )
                partial = partially_active.observe("sensor").values["gyro"]
                torch.testing.assert_close(partial[[0, 2]], expected[[0, 2]], rtol=0, atol=0)
                partially_active.advance(
                    control, torch.tensor([False, True, False])
                )
                resumed = partially_active.observe("sensor").values["gyro"]
                torch.testing.assert_close(resumed[1], expected[1], rtol=0, atol=0)

    def test_motor_noise_stream_is_isolated_by_instance_counter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_named_config(
                root, "motor-noise.json", _config(str(root / "logs"))
            )
            control = torch.zeros((3, 5))
            control[:, :2] = 0.5
            with SimulationEnvironment.create(path, 3, "cpu") as all_active:
                all_active.advance(control)
                expected = all_active.observe("truth").values
            with SimulationEnvironment.create(path, 3, "cpu") as partially_active:
                partially_active.advance(
                    control, torch.tensor([True, False, True])
                )
                partial = partially_active.observe("truth").values
                for name in expected:
                    torch.testing.assert_close(
                        partial[name][[0, 2]], expected[name][[0, 2]], rtol=0, atol=0
                    )
                partially_active.advance(
                    control, torch.tensor([False, True, False])
                )
                resumed = partially_active.observe("truth").values
                for name in expected:
                    torch.testing.assert_close(
                        resumed[name][1], expected[name][1], rtol=0, atol=0
                    )

    def test_sensor_configuration_rejects_ambiguous_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fractional = _config(str(root / "fractional_logs"))
            fractional["sensors"]["gyro"]["delay"] = _randomizable(0.0001)
            fractional_path = self._write_named_config(
                root, "fractional.json", fractional
            )
            with self.assertRaises(ConfigurationError):
                SimulationEnvironment.create(fractional_path, 1, "cpu")

            unsupported = _config(str(root / "unsupported_logs"))
            unsupported["sensors"]["unknown"] = unsupported["sensors"].pop("gyro")
            unsupported_path = self._write_named_config(
                root, "unsupported.json", unsupported
            )
            with self.assertRaises(ConfigurationError):
                SimulationEnvironment.create(unsupported_path, 1, "cpu")

            negative_noise = _config(str(root / "negative_noise_logs"))
            negative_noise["sensors"]["gyro"]["noise"]["stddev"] = _randomizable(
                [-0.1, 0.0, 0.0]
            )
            negative_noise_path = self._write_named_config(
                root, "negative-noise.json", negative_noise
            )
            with self.assertRaises(ConfigurationError):
                SimulationEnvironment.create(negative_noise_path, 1, "cpu")

    def test_advance_isolates_inactive_and_invalid_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SimulationEnvironment.create(self._write_config(root), 4, "cpu") as env:
                control = torch.zeros((4, 5), dtype=torch.float32)
                control[2, 0] = -0.1
                active = torch.tensor([True, False, True, True])
                result = env.advance(control, active)
                self.assertEqual(result.physics_steps_advanced.tolist(), [10, 0, 0, 10])
                self.assertEqual(result.physics_step.tolist(), [10, 0, 0, 10])
                self.assertEqual(result.control_step.tolist(), [1, 0, 0, 1])
                self.assertEqual(result.valid.tolist(), [True, True, False, True])
                self.assertEqual(
                    result.error_code.tolist(),
                    [int(ErrorCode.OK), int(ErrorCode.OK), int(ErrorCode.INVALID_CONTROL), int(ErrorCode.OK)],
                )
                position = env.observe("truth", ("position_n",)).values["position_n"]
                torch.testing.assert_close(position[1], torch.zeros(3))
                torch.testing.assert_close(position[2], torch.zeros(3))
                self.assertGreater(position[0, 2].item(), 0.0)
                self.assertGreater(position[3, 2].item(), 0.0)

    def test_randomized_parameter_prefix_is_stable_when_batch_grows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(root)
            with SimulationEnvironment.create(config, 2, "cpu") as small:
                expected = small.parameters["body.mass"]
            with SimulationEnvironment.create(config, 4, "cpu") as large:
                actual = large.parameters["body.mass"][:2]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_timeline_contains_initial_and_every_physics_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SimulationEnvironment.create(self._write_config(root), 2, "cpu") as env:
                log_directory = env._logger.directory
                env.advance(torch.zeros((2, 5), dtype=torch.float32))
            chunks = sorted(log_directory.glob("timeline_*.pt"))
            records = [torch.load(path, weights_only=True) for path in chunks]
            for field in (
                "truth.direct_force_b",
                "truth.grid_force_b",
                "truth.direct_moment_b",
                "truth.grid_moment_b",
                "truth.motor_reaction_moment_b",
                "truth.force_b",
                "truth.moment_b",
            ):
                self.assertIn(field, records[0])
            self.assertEqual(sum(chunk["physics_step"].shape[0] for chunk in records), 11)
            timeline = torch.cat([chunk["physics_step"] for chunk in records], dim=0)
            self.assertEqual(timeline[0].tolist(), [0, 0])
            self.assertEqual(timeline[-1].tolist(), [10, 10])

    def test_compact_timeline_downsamples_fields_but_keeps_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["logging"].update(
                {
                    "mode": "compact",
                    "physics_step_stride": 10,
                    "fields": [
                        "physics_step",
                        "control_step",
                        "event_code",
                        "truth.position_n",
                    ],
                }
            )
            config_path = self._write_named_config(root, "compact.json", config)
            with SimulationEnvironment.create(config_path, 2, "cpu") as env:
                log_directory = env._logger.directory
                env.advance(torch.zeros((2, 5), dtype=torch.float32))
            records = [
                torch.load(path, weights_only=True)
                for path in sorted(log_directory.glob("timeline_*.pt"))
            ]
            self.assertEqual(set(records[0]), set(config["logging"]["fields"]))
            physics_steps = torch.cat(
                [chunk["physics_step"] for chunk in records], dim=0
            )
            event_codes = torch.cat(
                [chunk["event_code"] for chunk in records], dim=0
            )
            self.assertEqual(physics_steps[:, 0].tolist(), [0, 10])
            self.assertEqual(event_codes[0].tolist(), [1, 1])

    def test_compact_timeline_does_not_build_skipped_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["logging"].update(
                {"mode": "compact", "physics_step_stride": 100}
            )
            config_path = self._write_named_config(root, "lazy_log.json", config)
            with SimulationEnvironment.create(config_path, 1, "cpu") as env:
                calls = 0
                original = env._log_record

                def counted_record(active_mask, event_code):
                    nonlocal calls
                    calls += 1
                    return original(active_mask, event_code)

                env._log_record = counted_record
                env.advance(torch.zeros((1, 5), dtype=torch.float32))
                self.assertEqual(calls, 0)

    def test_compact_reset_saves_only_selected_post_reset_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(str(root / "logs"))
            config["logging"].update(
                {
                    "mode": "compact",
                    "physics_step_stride": 100,
                    "fields": [
                        "physics_step",
                        "control_step",
                        "event_code",
                        "generation",
                        "truth.position_n",
                    ],
                }
            )
            config_path = self._write_named_config(root, "compact_reset.json", config)
            with SimulationEnvironment.create(config_path, 4, "cpu") as env:
                log_directory = env._logger.directory
                reset_mask = torch.tensor([True, False, True, False])
                env.reset(reset_mask, config_path)

            chunks = [
                torch.load(path, weights_only=True)
                for path in sorted(log_directory.glob("timeline_*.pt"))
            ]
            timeline_events = torch.cat(
                [chunk["event_code"] for chunk in chunks], dim=0
            )
            self.assertEqual(timeline_events.shape, (1, 4))
            self.assertEqual(timeline_events[0].tolist(), [1, 1, 1, 1])

            snapshot = torch.load(
                log_directory / "resets" / "000000" / "parameters.pt",
                weights_only=True,
            )
            self.assertEqual(
                snapshot["parameter_instance_indices"].tolist(), [0, 2]
            )
            sparse = snapshot["post_reset_timeline"]
            self.assertEqual(set(sparse), set(config["logging"]["fields"]))
            self.assertEqual(sparse["physics_step"].shape, (2,))
            self.assertEqual(sparse["event_code"].tolist(), [2, 2])
            self.assertEqual(sparse["generation"].tolist(), [1, 1])
            self.assertEqual(sparse["truth.position_n"].shape, (2, 3))

    def test_mask_reset_replaces_selected_slots_and_matches_batched_create(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = self._write_config(root)
            replacement_config = _config(str(root / "replacement_logs"))
            replacement_config["seed"] = 9876
            replacement_config["initial_state"]["position_n"] = _randomizable([3.0, 2.0, 1.0])
            replacement_config["body"]["mass"] = _randomizable(3.5)
            replacement_path = self._write_named_config(
                root, "replacement.json", replacement_config
            )

            with SimulationEnvironment.create(base_path, 4, "cpu") as env:
                control = torch.zeros((4, 5), dtype=torch.float32)
                control[:, :2] = 0.2
                control[0, 0] = -1.0
                control[2, 0] = -1.0
                env.advance(control)
                before = env.parameters
                truth_before = env.observe("truth").values
                sensor_before = env.observe("sensor").values
                previous_ids = env.instance_ids
                reset_mask = torch.tensor([True, False, True, False])
                result = env.reset(reset_mask, replacement_path)

                self.assertEqual(result.previous_instance_ids, previous_ids)
                self.assertEqual(result.reset_mask.tolist(), reset_mask.tolist())
                self.assertEqual(result.generation.tolist(), [1, 0, 1, 0])
                for index in (0, 2):
                    self.assertNotEqual(result.instance_ids[index], previous_ids[index])
                for index in (1, 3):
                    self.assertEqual(result.instance_ids[index], previous_ids[index])

                observation = env.observe("truth", ("position_n",))
                self.assertEqual(observation.physics_step.tolist(), [0, 10, 0, 10])
                self.assertEqual(observation.control_step.tolist(), [0, 1, 0, 1])
                self.assertEqual(observation.valid.tolist(), [True, True, True, True])
                torch.testing.assert_close(
                    env._control,
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.2, 0.2, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.2, 0.2, 0.0, 0.0, 0.0],
                        ]
                    ),
                )
                torch.testing.assert_close(
                    observation.values["position_n"][[0, 2]],
                    torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]),
                )

                after = env.parameters
                truth_after = env.observe("truth").values
                sensor_after = env.observe("sensor").values
                for name in before:
                    torch.testing.assert_close(after[name][1], before[name][1])
                    torch.testing.assert_close(after[name][3], before[name][3])
                for name in sensor_before:
                    torch.testing.assert_close(sensor_after[name][1], sensor_before[name][1])
                    torch.testing.assert_close(sensor_after[name][3], sensor_before[name][3])
                for name in truth_before:
                    torch.testing.assert_close(truth_after[name][1], truth_before[name][1])
                    torch.testing.assert_close(truth_after[name][3], truth_before[name][3])

                log_directory = env._logger.directory

            with SimulationEnvironment.create(replacement_path, 4, "cpu") as standalone:
                expected = standalone.parameters
                expected_truth = standalone.observe("truth").values
                expected_sensors = standalone.observe("sensor").values
            for name, value in expected.items():
                torch.testing.assert_close(after[name][0], value[0], rtol=0, atol=0)
                torch.testing.assert_close(after[name][2], value[2], rtol=0, atol=0)
            for name, value in expected_sensors.items():
                torch.testing.assert_close(sensor_after[name][0], value[0], rtol=0, atol=0)
                torch.testing.assert_close(sensor_after[name][2], value[2], rtol=0, atol=0)
            for name, value in expected_truth.items():
                torch.testing.assert_close(truth_after[name][0], value[0], rtol=0, atol=0)
                torch.testing.assert_close(truth_after[name][2], value[2], rtol=0, atol=0)

            reset_events = (log_directory / "resets.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(reset_events), 1)
            event = json.loads(reset_events[0])
            self.assertEqual(event["instance_indices"], [0, 2])
            self.assertEqual([item["generation"] for item in event["instances"]], [1, 1])
            reset_snapshot = torch.load(
                log_directory / "resets" / "000000" / "parameters.pt",
                weights_only=True,
            )
            self.assertEqual(reset_snapshot["reset_mask"].tolist(), reset_mask.tolist())
            self.assertEqual(reset_snapshot["parameters"]["body.mass"].shape, (4,))
            chunks = [torch.load(path, weights_only=True) for path in sorted(log_directory.glob("timeline_*.pt"))]
            event_codes = torch.cat([chunk["event_code"] for chunk in chunks], dim=0)
            generations = torch.cat([chunk["generation"] for chunk in chunks], dim=0)
            self.assertEqual(event_codes[-1].tolist(), [2, -1, 2, -1])
            self.assertEqual(generations[-1].tolist(), [1, 0, 1, 0])

    def test_mask_reset_refreshes_cached_derived_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _config(str(root / "base_logs"))
            replacement = _config(str(root / "replacement_logs"))
            replacement["seed"] = 9876
            replacement["motors"][0]["time_constant"] = _randomizable(0.003)
            replacement["motors"][0]["pwm_to_rpm_table"] = _randomizable(
                [[0.0, 0.0], [0.08, 0.0], [1.0, 900.0]]
            )
            replacement["body"]["center_of_mass_b"] = _randomizable(
                [0.0, 0.0, 0.02]
            )
            replacement["sensors"]["gyro"]["sample_hz"] = _randomizable(2500)
            replacement["sensors"]["gyro"]["delay"] = _randomizable(0.0004)
            base_path = self._write_named_config(root, "base.json", base)
            replacement_path = self._write_named_config(
                root, "replacement.json", replacement
            )
            control = torch.tensor(
                [[0.8, 0.4, 0.2, -0.1, 0.3]] * 2, dtype=torch.float32
            )

            with SimulationEnvironment.create(base_path, 2, "cpu") as env:
                env.reset(torch.tensor([True, False]), replacement_path)
                env.advance(control)
                env.advance(control)
                actual_truth = env.observe("truth").values
                actual_sensors = env.observe("sensor").values

            with SimulationEnvironment.create(
                replacement_path, 2, "cpu"
            ) as standalone:
                standalone.advance(control)
                standalone.advance(control)
                expected_truth = standalone.observe("truth").values
                expected_sensors = standalone.observe("sensor").values

            for name, expected in expected_truth.items():
                torch.testing.assert_close(
                    actual_truth[name][0], expected[0], rtol=0, atol=0
                )
            for name, expected in expected_sensors.items():
                torch.testing.assert_close(
                    actual_sensors[name][0], expected[0], rtol=0, atol=0
                )

    def test_empty_reset_mask_is_noop_and_does_not_load_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SimulationEnvironment.create(self._write_config(root), 2, "cpu") as env:
                ids_before = env.instance_ids
                parameters_before = env.parameters
                result = env.reset(
                    torch.zeros(2, dtype=torch.bool), root / "does-not-exist.json"
                )
                self.assertEqual(result.instance_ids, ids_before)
                self.assertEqual(result.generation.tolist(), [0, 0])
                self.assertFalse((env._logger.directory / "resets.jsonl").exists())
                for name, value in parameters_before.items():
                    torch.testing.assert_close(env.parameters[name], value)

    def test_reset_rejects_invalid_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_config(root)
            with SimulationEnvironment.create(config_path, 2, "cpu") as env:
                with self.assertRaises(TypeError):
                    env.reset([True, False], config_path)
                with self.assertRaises(ValueError):
                    env.reset(torch.tensor([True]), config_path)
                with self.assertRaises(ValueError):
                    env.reset(torch.tensor([1, 0]), config_path)
                with self.assertRaises(ValueError):
                    env.reset(
                        torch.zeros(2, dtype=torch.bool, device="meta"), config_path
                    )

    def test_reset_rejects_structural_change_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = self._write_config(root)
            incompatible = _config(str(root / "other_logs"))
            incompatible["timing"]["control_hz"] = _randomizable(250)
            incompatible_path = self._write_named_config(
                root, "incompatible.json", incompatible
            )
            with SimulationEnvironment.create(base_path, 2, "cpu") as env:
                ids_before = env.instance_ids
                parameters_before = env.parameters
                with self.assertRaises(ConfigurationError):
                    env.reset(torch.tensor([True, False]), incompatible_path)
                self.assertEqual(env.instance_ids, ids_before)
                for name, value in parameters_before.items():
                    torch.testing.assert_close(env.parameters[name], value)


if __name__ == "__main__":
    unittest.main()
