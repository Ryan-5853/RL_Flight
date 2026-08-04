from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict

from flight_train.config import (
    AttitudePidConfig,
    ConfigError,
    continuation_resume_config_sha256,
    load_experiment_config,
)
from flight_train.envs import SimEnvAdapter
from flight_train.math import euler_to_quaternion
from flight_train.commands import DirectAngularAccelerationCommandSource
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
CONFIG_V4 = (
    ROOT
    / "configs/experiments/mlp_sac_pid_angular_acceleration_tracking_v4.yaml"
)
CONFIG_V5 = (
    ROOT
    / "configs/experiments/mlp_sac_pid_angular_acceleration_tracking_v5.yaml"
)
CONFIG_V6 = (
    ROOT
    / "configs/experiments/mlp_sac_pid_angular_acceleration_tracking_v6.yaml"
)
CONFIG_V7 = (
    ROOT
    / "configs/experiments/mlp_sac_pid_angular_acceleration_tracking_v7.yaml"
)
CONFIG_V8 = (
    ROOT
    / "configs/experiments/mlp_sac_pid_angular_acceleration_tracking_v8.yaml"
)
BAD_POINT_CONFIGS = tuple(
    sorted(
        (
            ROOT / "configs/experiments/sim2real_bad_points"
        ).glob("mlp_sac_angular_acceleration_bad_point_*_v1.yaml")
    )
)


class PidAngularAccelerationExperimentTests(unittest.TestCase):
    def test_bad_point_configs_apply_exact_non_symmetric_hover_trim(self):
        self.assertEqual(len(BAD_POINT_CONFIGS), 3)
        saw_lower_ratio_above_one = False
        for path in BAD_POINT_CONFIGS:
            config = load_experiment_config(path)
            contract = config.control_contract
            zero_action = torch.zeros(1, 3)
            upper_hover = torch.tensor([[contract.policy_action_trim[0]]])
            command = SimEnvAdapter.action_to_command(
                zero_action,
                upper_hover,
                contract.policy_action_trim,
                contract.policy_action_residual_scale,
                contract.action_transform_type,
                contract.lower_motor_upper_ratio,
            )
            self.assertAlmostEqual(
                float(command[0, 1]),
                contract.policy_action_trim[0]
                * contract.lower_motor_upper_ratio,
                places=6,
            )
            torch.testing.assert_close(
                command[0, 2:],
                torch.tensor(contract.policy_action_trim[1:]),
            )
            self.assertEqual(
                config.task.outer_loop_pid.max_angular_acceleration_rad_s2,
                (4.0, 4.0, 1.5),
            )
            self.assertEqual(
                config.reward.calculator.params["aggregation"], "worst_axis"
            )
            saw_lower_ratio_above_one |= contract.lower_motor_upper_ratio > 1.0
        self.assertTrue(saw_lower_ratio_above_one)

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

    def test_v4_policy_restart_targets_the_full_command_envelope(self):
        config = load_experiment_config(CONFIG_V4)

        self.assertEqual(
            config.name, "mlp_sac_pid_angular_acceleration_tracking_v4"
        )
        self.assertEqual(config.checkpoint.resume_mode, "policy")
        self.assertEqual(config.checkpoint.resume_from.name, "step_15728640.pt")
        self.assertEqual(
            config.task.curriculum_target_scales,
            (0.25, 0.50, 0.75, 1.00),
        )
        self.assertEqual(
            config.task.curriculum_durations_s,
            (5.0, 10.0, 20.0, 30.0),
        )
        self.assertEqual(config.run.total_control_steps, 33_554_432)
        self.assertEqual(config.sac.critic_pretraining_updates, 2048)
        self.assertEqual(config.sac.policy_anchor_weight, 2.0)
        self.assertEqual(config.sac.policy_anchor_max_action_deviation, 0.20)
        self.assertEqual(
            config.reward.calculator.type,
            "flight_train.rewards.angular_acceleration:"
            "AngularAccelerationTrackingRewardCalculator",
        )
        self.assertEqual(
            tuple(field["name"] for field in config.reward.context_fields),
            (
                "desired_angular_acceleration_b",
                "actual_angular_acceleration_b",
            ),
        )

    def test_v5_uses_isolated_direct_command_training_contract(self):
        config = load_experiment_config(CONFIG_V5)

        self.assertEqual(
            config.name, "mlp_sac_pid_angular_acceleration_tracking_v5"
        )
        self.assertFalse(config.evaluation.enabled)
        self.assertEqual(config.checkpoint.resume_mode, "policy")
        self.assertEqual(config.run.total_control_steps, 8_388_608)
        self.assertEqual(
            config.command_source.type,
            "flight_train.commands:DirectAngularAccelerationCommandSource",
        )
        direct = config.command_source.direct_angular_acceleration
        self.assertIsNotNone(direct)
        assert direct is not None
        self.assertEqual(direct.limits_rad_s2, (6.0, 6.0, 3.0))
        self.assertAlmostEqual(direct.zero_probability, 0.35)
        self.assertAlmostEqual(direct.sine_probability, 0.45)
        self.assertAlmostEqual(direct.balanced_step_probability, 0.20)
        self.assertEqual(
            tuple(field["name"] for field in config.reward.context_fields),
            (
                "desired_angular_acceleration_b",
                "actual_angular_acceleration_b",
            ),
        )

    def test_direct_command_source_waveforms_and_state_round_trip(self):
        config = load_experiment_config(CONFIG_V5)
        source = DirectAngularAccelerationCommandSource(
            config.command_source,
            4,
            torch.device("cpu"),
            torch.float32,
            100,
        )
        source.curriculum_scale = 1.0
        source.direct_waveform.copy_(torch.tensor([0, 1, 2, 2]))
        source.direct_elapsed_s.copy_(
            torch.tensor([[0.5], [0.0], [0.25], [0.50]])
        )
        source.direct_duration_s.fill_(1.0)
        source.direct_frequency_hz.fill_(0.5)
        source.direct_amplitude.copy_(
            torch.tensor(
                [
                    [6.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 1.5],
                ]
            )
        )
        source._update_direct_command(torch.ones(4, dtype=torch.bool))
        torch.testing.assert_close(
            source.desired_angular_acceleration,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, -1.5],
                ]
            ),
        )

        restored = DirectAngularAccelerationCommandSource(
            config.command_source,
            4,
            torch.device("cpu"),
            torch.float32,
            100,
        )
        restored.load_state_dict(source.state_dict())
        for name in (
            "direct_waveform",
            "direct_elapsed_s",
            "direct_duration_s",
            "direct_frequency_hz",
            "direct_amplitude",
            "desired_angular_acceleration",
        ):
            left = (
                source.desired_angular_acceleration
                if name == "desired_angular_acceleration"
                else getattr(source, name)
            )
            right = (
                restored.desired_angular_acceleration
                if name == "desired_angular_acceleration"
                else getattr(restored, name)
            )
            torch.testing.assert_close(left, right)

    def test_v6_makes_measured_small_signal_bias_visible_to_reward(self):
        config = load_experiment_config(CONFIG_V6)

        self.assertEqual(
            config.name, "mlp_sac_pid_angular_acceleration_tracking_v6"
        )
        self.assertEqual(
            tuple(config.reward.calculator.params["error_scale_rad_s2"]),
            (0.75, 0.75, 0.375),
        )
        self.assertEqual(config.reward.calculator.params["aggregation"], "mean")
        self.assertEqual(
            config.reward.calculator.params["reward_form"],
            "positive_exponential",
        )
        calculator = ComponentRegistry().build_reward(
            {
                "type": config.reward.calculator.type,
                "version": config.reward.calculator.version,
                "params": config.reward.calculator.params,
            }
        )
        context = TensorDict(
            {
                "desired_angular_acceleration_b": torch.zeros(1, 3),
                "actual_angular_acceleration_b": torch.tensor(
                    [[-0.4, 0.1, 0.05]]
                ),
            },
            batch_size=[1],
        )
        reward = calculator(context).reward
        self.assertLess(float(reward.item()), 0.97)

    def test_v7_exactly_continues_v6_learning_state(self):
        v6 = load_experiment_config(CONFIG_V6)
        v7 = load_experiment_config(CONFIG_V7)

        self.assertEqual(
            v7.name, "mlp_sac_pid_angular_acceleration_tracking_v7"
        )
        self.assertEqual(v7.run.total_control_steps, 16_777_216)
        self.assertEqual(v7.checkpoint.resume_mode, "continuation")
        self.assertEqual(v7.checkpoint.resume_from.name, "step_8388608.pt")
        self.assertEqual(
            continuation_resume_config_sha256(v7),
            continuation_resume_config_sha256(v6),
        )

    def test_v8_refines_v6_with_bias_sensitive_tracking_only(self):
        config = load_experiment_config(CONFIG_V8)

        self.assertEqual(
            config.name, "mlp_sac_pid_angular_acceleration_tracking_v8"
        )
        self.assertEqual(config.checkpoint.resume_mode, "policy")
        self.assertEqual(config.checkpoint.resume_from.name, "step_8388608.pt")
        self.assertEqual(
            tuple(config.reward.calculator.params["error_scale_rad_s2"]),
            (0.25, 0.25, 0.125),
        )
        self.assertEqual(
            config.reward.calculator.params["aggregation"], "worst_axis"
        )
        self.assertAlmostEqual(config.sac.actor_learning_rate, 0.000002)
        self.assertEqual(config.sac.policy_anchor_weight, 5.0)
        self.assertEqual(config.sac.policy_anchor_max_action_deviation, 0.05)
        self.assertEqual(config.checkpoint.interval_control_steps, 1_048_576)
        self.assertEqual(config.checkpoint.keep_last, 6)
        self.assertEqual(
            tuple(field["name"] for field in config.reward.context_fields),
            (
                "desired_angular_acceleration_b",
                "actual_angular_acceleration_b",
            ),
        )

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
