from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict

from flight_train.config import AttitudePidConfig, ConfigError, load_experiment_config
from flight_train.envs import SimEnvAdapter
from flight_train.math import euler_to_quaternion
from flight_train.registry import ComponentRegistry
from flight_train.rewards import AngularAccelerationTrackingRewardCalculator
from flight_train.tasks import AttitudePidOuterLoop


ROOT = Path(__file__).parents[1]
CONFIG = (
    ROOT
    / "configs/experiments/mlp_sac_pid_angular_acceleration_tracking_v1.yaml"
)
CONFIG_V2 = (
    ROOT
    / "configs/experiments/mlp_sac_pid_angular_acceleration_tracking_v2.yaml"
)
CONFIG_V3 = (
    ROOT
    / "configs/experiments/mlp_sac_pid_angular_acceleration_tracking_v3.yaml"
)


class PidAngularAccelerationExperimentTests(unittest.TestCase):
    def test_experiment_contract_and_reward_are_acceleration_only(self):
        config = load_experiment_config(CONFIG)

        self.assertEqual(
            config.name, "mlp_sac_pid_angular_acceleration_tracking_v1"
        )
        self.assertEqual(
            config.control_contract.observation_profile,
            "angular_acceleration_inner_loop_22d_v1",
        )
        self.assertEqual(config.control_contract.base_observation_dim, 22)
        self.assertEqual(config.control_contract.observation_dim, 22 * 61)
        self.assertIsNotNone(config.task.outer_loop_pid)
        self.assertIsNone(config.checkpoint.resume_from)
        self.assertEqual(
            tuple(field["name"] for field in config.reward.context_fields),
            (
                "desired_angular_acceleration_b",
                "actual_angular_acceleration_b",
            ),
        )
        calculator = ComponentRegistry().build_reward(
            {
                "type": config.reward.calculator.type,
                "version": config.reward.calculator.version,
                "params": config.reward.calculator.params,
            }
        )
        self.assertIsInstance(
            calculator, AngularAccelerationTrackingRewardCalculator
        )

    def test_pid_uses_shortest_attitude_error_and_masked_integrator(self):
        pid = AttitudePidOuterLoop(
            AttitudePidConfig(
                proportional_gain=(10.0, 10.0, 10.0),
                integral_gain=(2.0, 2.0, 2.0),
                derivative_gain=(3.0, 3.0, 3.0),
                integral_limit_rad_s=(0.2, 0.2, 0.2),
                max_angular_acceleration_rad_s2=(5.0, 5.0, 5.0),
            ),
            2,
            torch.device("cpu"),
            torch.float32,
            500,
        )
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1)
        target = euler_to_quaternion(
            torch.tensor([0.1, 0.1]),
            torch.zeros(2),
            torch.zeros(2),
        )
        angular_velocity = torch.tensor(
            [[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]
        )
        target_rate = torch.zeros(2, 3)
        active = torch.tensor([True, False])

        command = pid.update(
            identity, target, angular_velocity, target_rate, active
        )

        expected = 10.0 * 0.1 + 2.0 * 0.1 / 500.0 - 3.0 * 0.02
        torch.testing.assert_close(command[0, 0], torch.tensor(expected))
        torch.testing.assert_close(command[1], torch.zeros(3))
        torch.testing.assert_close(pid.integral_error[0, 0], torch.tensor(0.0002))
        pid.reset(torch.tensor([True, False]))
        torch.testing.assert_close(pid.integral_error, torch.zeros(2, 3))

        sign_flipped = pid.update(
            identity, -target, torch.zeros_like(angular_velocity), target_rate,
            torch.ones(2, dtype=torch.bool),
        )
        self.assertGreater(float(sign_flipped[0, 0]), 0.0)

    def test_reward_has_exactly_one_tracking_term(self):
        calculator = AngularAccelerationTrackingRewardCalculator(
            {
                "weight": 1.0,
                "error_scale_rad_s2": [2.0, 2.0, 2.0],
                "huber_delta": 1.0,
            }
        )
        context = TensorDict(
            {
                "desired_angular_acceleration_b": torch.zeros(2, 3),
                "actual_angular_acceleration_b": torch.tensor(
                    [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
            },
            batch_size=[2],
        )

        output = calculator(context)

        self.assertEqual(
            tuple(output.terms.keys()),
            ("reward.angular_acceleration_tracking",),
        )
        torch.testing.assert_close(
            output.reward[:, 0], torch.tensor([-1.0 / 6.0, 0.0])
        )
        torch.testing.assert_close(
            output.reward,
            output.terms["reward.angular_acceleration_tracking"],
        )

    def test_positive_reward_uses_worst_axis_tracking_error_only(self):
        calculator = AngularAccelerationTrackingRewardCalculator(
            {
                "weight": 1.0,
                "error_scale_rad_s2": [2.0, 2.0, 2.0],
                "huber_delta": 1.0,
                "reward_form": "positive_exponential",
                "aggregation": "worst_axis",
            }
        )
        context = TensorDict(
            {
                "desired_angular_acceleration_b": torch.zeros(2, 3),
                "actual_angular_acceleration_b": torch.tensor(
                    [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
            },
            batch_size=[2],
        )

        output = calculator(context)

        self.assertEqual(
            tuple(output.terms.keys()),
            ("reward.angular_acceleration_tracking",),
        )
        torch.testing.assert_close(
            output.reward[:, 0], torch.tensor([torch.exp(torch.tensor(-0.5)), 1.0])
        )
        self.assertTrue(output.valid.all())

    def test_v2_config_addresses_v1_failure_modes(self):
        config = load_experiment_config(CONFIG_V2)

        self.assertEqual(
            config.name, "mlp_sac_pid_angular_acceleration_tracking_v2"
        )
        self.assertEqual(config.control_contract.base_observation_dim, 22)
        self.assertEqual(config.control_contract.observation_dim, 22 * 61)
        self.assertEqual(
            config.reward.calculator.params["reward_form"],
            "positive_exponential",
        )
        self.assertEqual(
            config.reward.calculator.params["aggregation"], "worst_axis"
        )
        self.assertEqual(config.task.terminate_angular_rate_axes, "all")
        self.assertEqual(
            config.task.outer_loop_pid.max_angular_acceleration_rad_s2,
            (6.0, 6.0, 3.0),
        )
        self.assertIsNotNone(config.sac)
        self.assertEqual(config.sac.gamma, 0.99)
        self.assertEqual(config.sac.n_step_return, 8)
        self.assertEqual(config.sac.critic_pretraining_updates, 2048)
        self.assertEqual(config.sac.actor_update_interval, 32)
        self.assertAlmostEqual(config.sac.actor_learning_rate, 0.000005)
        self.assertIsNone(config.checkpoint.resume_from)

    def test_v3_allocates_yaw_to_motor_difference_and_servos_to_cyclic(self):
        config = load_experiment_config(CONFIG_V3)
        contract = config.control_contract
        upper = torch.tensor([[0.565], [0.565]])
        action = torch.tensor(
            [
                [0.5, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        command = SimEnvAdapter.action_to_command(
            action,
            upper,
            contract.policy_action_trim,
            contract.policy_action_residual_scale,
            contract.action_transform_type,
            contract.lower_motor_upper_ratio,
        )

        balanced_lower = 0.565 * contract.lower_motor_upper_ratio
        self.assertAlmostEqual(command[0, 1].item(), balanced_lower + 0.06)
        self.assertAlmostEqual(command[1, 1].item(), balanced_lower)
        torch.testing.assert_close(
            command[0, 2:], torch.tensor([0.4, -0.2, -0.2])
        )
        torch.testing.assert_close(
            command[1, 2:],
            torch.tensor([0.0, 0.2 * 3.0**0.5, -0.2 * 3.0**0.5]),
        )
        torch.testing.assert_close(command[:, 2:].sum(dim=-1), torch.zeros(2))

    def test_v3_config_restores_learnable_actor_schedule(self):
        config = load_experiment_config(CONFIG_V3)

        self.assertEqual(
            config.name, "mlp_sac_pid_angular_acceleration_tracking_v3"
        )
        self.assertEqual(
            config.control_contract.action_transform_type,
            "coaxial_differential_cyclic",
        )
        self.assertEqual(config.control_contract.action_dim, 3)
        self.assertEqual(config.control_contract.base_observation_dim, 21)
        self.assertEqual(config.control_contract.observation_dim, 21 * 61)
        self.assertEqual(
            config.reward.calculator.params["error_scale_rad_s2"],
            [6.0, 6.0, 3.0],
        )
        self.assertEqual(config.sac.critic_pretraining_updates, 256)
        self.assertEqual(config.sac.actor_update_interval, 8)
        self.assertAlmostEqual(config.sac.actor_learning_rate, 0.000015)
        self.assertEqual(
            config.model.sac_initial_action_std,
            (0.15, 0.10, 0.10),
        )
        self.assertIsNone(config.checkpoint.resume_from)

    def test_acceleration_observation_contract_requires_pid_outer_loop(self):
        source = CONFIG.read_text(encoding="utf-8")
        start = source.index("  outer_loop:\n")
        end = source.index("  episode_curriculum:\n", start)
        without_outer_loop = source[:start] + source[end:]
        absolute_environment = str(
            (ROOT / "configs/environment/gru_sac_upright_height_only_small_tip.yaml")
            .resolve()
        )
        without_outer_loop = without_outer_loop.replace(
            "../environment/gru_sac_upright_height_only_small_tip.yaml",
            absolute_environment,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(without_outer_loop, encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigError,
                "angular_acceleration_inner_loop.*task.outer_loop",
            ):
                load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
