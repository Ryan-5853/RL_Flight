from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from flight_train.algorithms import RecurrentPPO, TorchRLPPO, TorchRLSAC
from flight_train.collector import TensorDictRolloutCollector
from flight_train.commands import VirtualPilotCommandSource
from flight_train.config import (
    ModelConfig,
    PPOConfig,
    SACConfig,
    TaskConfig,
    load_experiment_config,
    override_run_config,
)
from flight_train.core import EnvSpec
from flight_train.envs import SimEnvAdapter
from flight_train.math import quaternion_geodesic_angle
from flight_train.models import build_actor_critic, build_sac_actor_critic
from flight_train.randomization import StaticParameterSpec, StaticRandomizer
from flight_train.rewards import AttitudeRewardCalculator, RewardOutput
from flight_train.tasks import AttitudeTrackingTask


class TensorEnv:
    def __init__(self, batch: int = 4, device: torch.device | None = None) -> None:
        device = device or torch.device("cpu")
        self.spec = EnvSpec(batch, 21, 4, device, torch.float32, 5000, 500, ())
        self._step = torch.zeros(batch, dtype=torch.int64, device=device)

    def reset(self, mask=None):
        return TensorDict(
            {
                "observation": torch.zeros(self.spec.parallel_count, 21, device=self.spec.device),
                "is_init": torch.ones(self.spec.parallel_count, 1, dtype=torch.bool, device=self.spec.device),
            },
            [self.spec.parallel_count],
            device=self.spec.device,
        )

    def step(self, action):
        self._step += 1
        batch = self.spec.parallel_count
        reset = (self._step % 5 == 0)[:, None]
        return TensorDict(
            {
                "observation": torch.randn(batch, 21, device=self.spec.device),
                "reward": -action.square().mean(-1, keepdim=True),
                "terminated": torch.zeros(batch, 1, dtype=torch.bool, device=self.spec.device),
                "truncated": reset,
                "done": reset,
                "valid": torch.ones(batch, 1, dtype=torch.bool, device=self.spec.device),
                "is_init": reset,
            },
            [batch],
            device=self.spec.device,
        )

    def close(self):
        pass


class TensorTrainingTests(unittest.TestCase):
    def test_v2_smoke_config_parses(self):
        path = Path(__file__).parents[1] / "configs/experiments/gru_ppo_smoke.json"
        config = load_experiment_config(path)
        self.assertEqual(config.schema_version, 2)
        self.assertEqual(config.entrypoint.type, "flight_train.runner:run_experiment")
        self.assertNotEqual(config.static_randomization.seed, config.dynamic_randomization.seed)

    def test_mlp_ppo_smoke_config_parses(self):
        path = Path(__file__).parents[1] / "configs/experiments/mlp_ppo_smoke.json"
        config = load_experiment_config(path)
        self.assertEqual(config.model.architecture, "mlp")
        self.assertEqual(config.model.encoder_sizes, (256, 256, 128))
        self.assertEqual(config.ppo.sequence_length, 1)

    def test_mlp_sac_smoke_config_parses(self):
        path = Path(__file__).parents[1] / "configs/experiments/mlp_sac_smoke.json"
        config = load_experiment_config(path)
        self.assertEqual(config.algorithm_name, "sac")
        self.assertIsNone(config.ppo)
        self.assertEqual(config.sac.replay_batch_size, 32)
        self.assertEqual(config.run.rollout_steps, 8)
        self.assertEqual(config.sac.n_step_return, 4)
        self.assertEqual(
            config.model.sac_initial_action_std,
            (0.05, 0.08, 0.08, 0.08),
        )
        self.assertFalse(config.model.sac_learnable_action_std)
        self.assertEqual(config.sac.critic_pretraining_updates, 2)

    def test_sac_policy_starts_at_zero_mean_with_bounded_configured_std(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/mlp_sac_nominal_baseline_1.yaml"
        )
        model = build_sac_actor_critic(
            21, 4, config.model, torch.device("cpu")
        )
        td = TensorDict(
            {"observation": torch.randn(32, 21)}, batch_size=[32]
        )
        model.policy_module(td)
        expected = torch.tensor([0.05, 0.08, 0.08, 0.08]).expand(32, -1)
        self.assertTrue(torch.allclose(td["loc"], torch.zeros_like(td["loc"])))
        self.assertTrue(torch.allclose(td["scale"], expected, atol=1e-6))
        minimum = torch.tensor([0.01, 0.02, 0.02, 0.02])
        maximum = torch.tensor([0.10, 0.15, 0.15, 0.15])
        self.assertTrue(torch.all(td["scale"] > minimum))
        self.assertTrue(torch.all(td["scale"] < maximum))
        reported = model.exploration_std_for(td["observation"])
        self.assertTrue(torch.allclose(reported, expected[0], atol=1e-6))
        with torch.no_grad():
            for parameter in model.actor.parameters():
                parameter.add_(0.01 * torch.randn_like(parameter))
        changed_td = TensorDict(
            {"observation": torch.randn(32, 21)}, batch_size=[32]
        )
        model.policy_module(changed_td)
        self.assertTrue(
            torch.allclose(changed_td["scale"], expected, atol=1e-6)
        )
        self.assertEqual(config.run.total_control_steps, 8_388_608)
        self.assertEqual(
            config.control_contract.policy_action_trim,
            (0.53523, 0.0, 0.0, 0.0),
        )
        self.assertFalse(config.model.sac_learnable_action_std)
        self.assertEqual(config.sac.critic_pretraining_updates, 1024)
        self.assertEqual(config.sac.n_step_return, 64)
        self.assertEqual(config.sac.updates_per_collection, 16)
        self.assertEqual(config.run.rollout_steps, 128)
        self.assertEqual(config.sac.initial_alpha, 0.0001)
        self.assertEqual(config.sac.actor_learning_rate, 0.00003)
        self.assertEqual(config.sac.alpha_learning_rate, 0.0001)
        self.assertEqual(config.evaluation.interval_control_steps, 2_097_152)

    def test_sac_collect_update_and_replay_state(self):
        device = torch.device("cpu")
        model_config = ModelConfig(
            (32, 32),
            32,
            32,
            architecture="mlp",
            sac_initial_action_std=(0.08, 0.15, 0.15, 0.15),
            sac_minimum_action_std=(0.01, 0.02, 0.02, 0.02),
            sac_maximum_action_std=(0.20, 0.35, 0.35, 0.35),
        )
        model = build_sac_actor_critic(
            21,
            4,
            model_config,
            device,
        )
        sac = TorchRLSAC(
            model,
            SACConfig(
                gamma=0.99,
                n_step_return=1,
                replay_capacity=1024,
                replay_batch_size=16,
                warmup_transitions=16,
                updates_per_collection=2,
                actor_learning_rate=3e-4,
                critic_learning_rate=3e-4,
                alpha_learning_rate=1e-4,
                actor_max_grad_norm=1.0,
                critic_max_grad_norm=5.0,
                target_tau=0.005,
                target_update_interval=1,
                initial_alpha=0.1,
                target_entropy=-4.0,
                min_alpha=1e-4,
                max_alpha=1.0,
                critic_pretraining_updates=2,
            ),
            device,
        )
        rollout = TensorDictRolloutCollector(
            TensorEnv(batch=4, device=device), model, 4
        ).collect()
        actor_before = {
            key: value.detach().clone()
            for key, value in model.actor.state_dict().items()
        }
        metrics = sac.update(rollout)
        self.assertEqual(sac.replay_size, 16)
        self.assertEqual(float(metrics["sac_updates"]), 2.0)
        self.assertEqual(float(metrics["sac_critic_pretraining"]), 1.0)
        self.assertEqual(float(metrics["sac_actor_updates_total"]), 0.0)
        for key, value in model.actor.state_dict().items():
            self.assertTrue(torch.equal(value, actor_before[key]))
        self.assertTrue(torch.isfinite(metrics["loss_qvalue"]))
        state = sac.state_dict()
        self.assertIn("replay", state)

        restored_model = build_sac_actor_critic(
            21,
            4,
            model_config,
            device,
        )
        restored = TorchRLSAC(restored_model, sac.config, device)
        restored.load_state_dict(state)
        self.assertEqual(restored.replay_size, 16)
        restored_metrics = restored.update(rollout)
        self.assertEqual(restored.replay_size, 32)
        self.assertEqual(float(restored_metrics["sac_critic_pretraining"]), 0.0)
        self.assertEqual(float(restored_metrics["sac_actor_updates_total"]), 2.0)
        self.assertTrue(torch.isfinite(restored_metrics["loss_actor"]))

    def test_sac_n_step_return_crosses_rollout_boundary_and_shifts_terminal(self):
        device = torch.device("cpu")
        model_config = ModelConfig(
            (16, 16),
            16,
            16,
            architecture="mlp",
            sac_initial_action_std=(0.05, 0.05, 0.05, 0.05),
            sac_minimum_action_std=(0.01, 0.01, 0.01, 0.01),
            sac_maximum_action_std=(0.10, 0.10, 0.10, 0.10),
        )
        model = build_sac_actor_critic(21, 4, model_config, device)
        sac = TorchRLSAC(
            model,
            SACConfig(
                gamma=0.9,
                n_step_return=3,
                replay_capacity=32,
                replay_batch_size=2,
                warmup_transitions=2,
                updates_per_collection=1,
                actor_learning_rate=3e-4,
                critic_learning_rate=3e-4,
                alpha_learning_rate=1e-4,
                actor_max_grad_norm=1.0,
                critic_max_grad_norm=5.0,
                target_tau=0.005,
                target_update_interval=1,
                initial_alpha=0.1,
                target_entropy=-4.0,
                min_alpha=1e-4,
                max_alpha=1.0,
            ),
            device,
        )

        def chunk(start: int, *, terminate_first: bool = False) -> TensorDict:
            observation = torch.zeros(1, 2, 21)
            next_observation = torch.zeros(1, 2, 21)
            observation[..., 0] = torch.tensor(
                [[float(start), float(start + 1)]]
            )
            next_observation[..., 0] = torch.tensor(
                [[float(start + 1), float(start + 2)]]
            )
            terminated = torch.tensor(
                [[[terminate_first], [False]]], dtype=torch.bool
            )
            return TensorDict(
                {
                    "observation": observation,
                    "action": torch.zeros(1, 2, 4),
                    "next": TensorDict(
                        {
                            "observation": next_observation,
                            "reward": torch.tensor(
                                [[[float(start + 1)], [float(start + 2)]]]
                            ),
                            "done": terminated.clone(),
                            "terminated": terminated,
                            "truncated": torch.zeros(
                                1, 2, 1, dtype=torch.bool
                            ),
                            "valid": torch.ones(
                                1, 2, 1, dtype=torch.bool
                            ),
                        },
                        batch_size=[1, 2],
                    ),
                },
                batch_size=[1, 2],
            )

        # 第一批只有 2 步，不足以形成 3-step target，必须只进入 pending。
        self.assertEqual(sac.add(chunk(0)), 0)
        self.assertEqual(sac.replay_size, 0)
        self.assertEqual(sac.n_step_pending.batch_size, torch.Size([1, 2]))

        # 第二批第一步（全局 t=2）终止。前两个起点都应在终止处停止：
        # G0 = 1 + .9*2 + .9²*3；G1 = 2 + .9*3。
        self.assertEqual(sac.add(chunk(2, terminate_first=True)), 2)
        stored = sac.replay._storage[:2]
        self.assertTrue(
            torch.allclose(
                stored[("next", "reward")].squeeze(-1),
                torch.tensor([5.23, 4.70]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                stored["steps_to_next_obs"],
                torch.tensor([3, 2]),
            )
        )
        self.assertTrue(stored[("next", "terminated")].all())
        self.assertTrue(
            torch.equal(
                stored[("next", "observation")][:, 0],
                torch.tensor([3.0, 3.0]),
            )
        )

        # TorchRL SAC 的 TD0Estimator 必须读取 steps_to_next_obs；否则虽然
        # replay 保存了多步奖励，bootstrap 仍会错误地只乘一次 gamma。
        estimator_input = TensorDict(
            {
                "steps_to_next_obs": torch.tensor([2, 3]),
                "next": TensorDict(
                    {
                        "reward": torch.ones(2, 1),
                        "done": torch.zeros(2, 1, dtype=torch.bool),
                        "terminated": torch.zeros(2, 1, dtype=torch.bool),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[2],
        )
        target = sac.loss.value_estimator.value_estimate(
            estimator_input,
            next_value=torch.full((2, 1), 10.0),
        )
        self.assertTrue(
            torch.allclose(
                target,
                torch.tensor([[1.0 + 0.9**2 * 10.0], [1.0 + 0.9**3 * 10.0]]),
                atol=1e-6,
            )
        )

        # exact checkpoint 必须带上尚未形成 target 的原始尾部。
        state = sac.state_dict()
        restored_model = build_sac_actor_critic(
            21, 4, model_config, device
        )
        restored = TorchRLSAC(restored_model, sac.config, device)
        restored.load_state_dict(state)
        self.assertTrue(
            torch.equal(
                restored.n_step_pending["observation"],
                sac.n_step_pending["observation"],
            )
        )

    def test_curriculum_can_be_observed_without_promotion(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/mlp_sac_nominal_baseline_1.yaml"
        )
        env = object.__new__(SimEnvAdapter)
        env.batch_size = 256
        env.device = torch.device("cpu")
        env.task_config = config.task
        env.curriculum_stage = 0
        env.curriculum_successes = 0
        env.curriculum_failures = 0
        env.curriculum_consecutive_passes = 0
        env.curriculum_last_success_fraction = 0.0
        env.max_episode_steps = 1000
        env._spec = SimpleNamespace(control_hz=500)
        env.command_source = SimpleNamespace(
            set_curriculum_scale=lambda scale: None
        )
        terminated = torch.zeros(256, 1, dtype=torch.bool)
        truncated = torch.ones(256, 1, dtype=torch.bool)

        for _ in range(3):
            metrics = env.update_episode_curriculum(
                terminated, truncated, allow_promotion=False
            )
        self.assertEqual(env.curriculum_stage, 0)
        self.assertEqual(env.curriculum_consecutive_passes, 0)
        self.assertEqual(float(metrics["curriculum_last_success_fraction"]), 1.0)
        self.assertEqual(float(metrics["curriculum_promotion_enabled"]), 0.0)

        env.update_episode_curriculum(
            terminated, truncated, allow_promotion=True
        )
        metrics = env.update_episode_curriculum(
            terminated, truncated, allow_promotion=True
        )
        self.assertEqual(env.curriculum_stage, 1)
        self.assertEqual(float(metrics["curriculum_promoted"]), 1.0)

    def test_commented_training_entry_template_parses(self):
        path = (
            Path(__file__).parents[1]
            / "configs/experiments/training_entry_template.yaml"
        )
        config = load_experiment_config(path)
        self.assertEqual(config.schema_version, 2)
        self.assertEqual(config.model.architecture, "mlp")
        self.assertEqual(config.model.encoder_sizes, (256, 256, 128))
        self.assertEqual(config.run.parallel_count, 128)

    def test_nominal_mlp_training_entry_parses(self):
        path = (
            Path(__file__).parents[1]
            / "configs/experiments/mlp_nominal_baseline_1.yaml"
        )
        config = load_experiment_config(path)
        self.assertEqual(config.command_source.version, "2")
        self.assertEqual(config.command_source.height_target_m, 0.0)
        self.assertEqual(config.command_source.height_initial_throttle_range, (0.55, 0.58))
        self.assertEqual(config.command_source.spool_target_range, (0.54, 0.58))
        self.assertEqual(config.command_source.throttle_rise_rate_per_s, 4.0)
        self.assertEqual(config.run.parallel_count, 256)
        self.assertEqual(config.run.rollout_steps, 256)
        self.assertEqual(config.run.total_control_steps, 33_554_432)
        self.assertEqual(
            config.run.total_control_steps
            // (config.run.parallel_count * config.run.rollout_steps),
            512,
        )
        self.assertEqual(config.command_source.max_roll_rad, 0.10)
        self.assertEqual(config.command_source.max_pitch_rad, 0.10)
        self.assertEqual(config.command_source.max_yaw_rate_rad_s, 0.35)
        self.assertEqual(config.dynamic_randomization.parameters, {})
        self.assertEqual(config.ppo.gamma, 0.9995)
        self.assertEqual(config.ppo.gae_lambda, 0.9985)
        self.assertEqual(config.ppo.epochs, 4)
        self.assertEqual(config.ppo.sequences_per_minibatch, 16_384)
        self.assertEqual(
            config.control_contract.policy_action_trim,
            (0.5539, 0.0, 0.0, 0.0),
        )
        self.assertEqual(config.task.terminate_tilt_rad, 1.0)
        self.assertEqual(config.task.terminate_angular_rate_rad_s, 8.0)

    def test_cli_run_overrides_update_resolved_fingerprint_input(self):
        path = Path(__file__).parents[1] / "configs/experiments/mlp_ppo_smoke.json"
        config = override_run_config(
            load_experiment_config(path),
            device="cuda:0",
            output_root="/tmp/rl-flight-test-runs",
        )
        self.assertEqual(config.run.device, "cuda:0")
        self.assertEqual(config.raw["run"]["device"], "cuda:0")
        self.assertEqual(config.run.output_root, Path("/tmp/rl-flight-test-runs"))
        self.assertEqual(config.raw["run"]["output_root"], "/tmp/rl-flight-test-runs")

    def test_nominal_virtual_pilot_reaches_hover_band_after_spool(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/mlp_nominal_baseline_1.yaml"
        )
        pilot = VirtualPilotCommandSource(
            config.command_source,
            batch_size=32,
            device=torch.device("cpu"),
            dtype=torch.float32,
            control_hz=500,
        )
        pilot.reset(torch.ones(32, dtype=torch.bool))
        height = torch.zeros(32, 1)
        for _ in range(50):
            pilot.step(height)
        self.assertGreaterEqual(float(pilot.upper_throttle.min()), 0.54)
        self.assertLessEqual(float(pilot.upper_throttle.max()), 0.58)

    def test_action_transform_exact_ranges(self):
        standard = torch.tensor([[1.0, -1.0, 0.0, 1.0]])
        command = SimEnvAdapter.action_to_command(standard, torch.tensor([[0.37]]))
        torch.testing.assert_close(command, torch.tensor([[0.37, 1.0, -1.0, 0.0, 1.0]]))
        changed = SimEnvAdapter.action_to_command(-standard, torch.tensor([[0.37]]))
        self.assertEqual(changed[0, 0], 0.37)

    def test_virtual_pilot_first_order_filter_and_stick_bounds(self):
        config = load_experiment_config(
            Path(__file__).parents[1] / "configs/experiments/mlp_ppo_smoke.json"
        ).command_source
        pilot = VirtualPilotCommandSource(
            config, 64, torch.device("cpu"), torch.float32, control_hz=500
        )
        pilot.reset(torch.ones(64, dtype=torch.bool))
        pilot.stick_target[:] = torch.tensor([0.2, -0.1, 0.5])
        pilot.filtered_stick.zero_()
        pilot.hold_remaining.fill_(1.0)
        pilot.step(torch.full((64, 1), 5.0))
        tau = torch.tensor([
            config.roll_time_constant_s,
            config.pitch_time_constant_s,
            config.yaw_time_constant_s,
        ])
        expected = (1.0 - torch.exp(-torch.tensor(1.0 / 500.0) / tau)) * torch.tensor(
            [0.2, -0.1, 0.5]
        )
        torch.testing.assert_close(pilot.filtered_stick[0], expected)

        pilot.hold_remaining.zero_()
        pilot.step(torch.full((64, 1), 5.0))
        normalized_radius = torch.sqrt(
            (pilot.stick_target[:, 0] / config.max_roll_rad).square()
            + (pilot.stick_target[:, 1] / config.max_pitch_rad).square()
        )
        self.assertTrue((normalized_radius <= 1.0 + 1e-6).all())
        self.assertTrue((pilot.stick_target[:, 2].abs() <= 1.0).all())

    def test_virtual_pilot_incremental_height_control_slew_and_masked_reset(self):
        config = load_experiment_config(
            Path(__file__).parents[1] / "configs/experiments/mlp_ppo_smoke.json"
        ).command_source
        pilot = VirtualPilotCommandSource(
            config, 3, torch.device("cpu"), torch.float32, control_hz=500
        )
        pilot.reset(torch.ones(3, dtype=torch.bool))
        pilot.spool_remaining.zero_()
        pilot.upper_throttle.fill_(0.5)
        pilot.height_controller_output.fill_(0.5)
        pilot.step(torch.tensor([[-1.0], [0.0], [1.0]]))
        torch.testing.assert_close(pilot.height_error[:, 0], torch.tensor([1.0, 0.0, -1.0]))
        expected_increment = config.height_proportional_gain + config.height_integral_gain / 500.0
        torch.testing.assert_close(
            pilot.height_controller_output[:, 0],
            torch.tensor([0.5 + expected_increment, 0.5, 0.5 - expected_increment]),
        )
        first_upper = pilot.upper_throttle.clone()
        per_step_slew = max(
            config.throttle_rise_rate_per_s,
            config.throttle_fall_rate_per_s,
        ) / 500.0
        self.assertLessEqual(float((first_upper - 0.5).abs().max()), per_step_slew + 1e-7)
        previous_output = pilot.height_controller_output.clone()
        pilot.step(torch.tensor([[-1.0], [0.0], [1.0]]))
        torch.testing.assert_close(
            pilot.height_controller_output[:, 0],
            previous_output[:, 0]
            + torch.tensor([
                config.height_integral_gain / 500.0,
                0.0,
                -config.height_integral_gain / 500.0,
            ]),
        )
        self.assertLessEqual(
            float((pilot.upper_throttle - first_upper).abs().max()),
            per_step_slew + 1e-7,
        )

        preserved = pilot.filtered_stick[1:].clone()
        preserved_height_output = pilot.height_controller_output[1:].clone()
        pilot.filtered_stick[0].fill_(0.8)
        pilot.height_previous_error.fill_(2.0)
        pilot.reset(torch.tensor([True, False, False]))
        torch.testing.assert_close(pilot.filtered_stick[0], torch.zeros(3))
        torch.testing.assert_close(pilot.filtered_stick[1:], preserved)
        torch.testing.assert_close(pilot.height_previous_error[0], torch.zeros(1))
        torch.testing.assert_close(
            pilot.height_controller_output[1:], preserved_height_output
        )

    def test_quaternion_sign_has_same_reward_distance(self):
        q = torch.tensor([[0.9238795, 0.3826834, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(
            quaternion_geodesic_angle(q, target),
            quaternion_geodesic_angle(-q, target),
        )

    def test_reward_kernel_stays_batched_and_on_device(self):
        cfg = TaskConfig(30.0, 1.3, 20.0, "truth")
        task = AttitudeTrackingTask(
            cfg, 8, torch.device("cpu"), torch.float32,
            AttitudeRewardCalculator({"attitude_weight": 4.0}),
        )
        result = task.transition(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(8, -1),
            torch.zeros(8, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(8, -1),
            torch.zeros(8, 4),
            torch.zeros(8, 4),
        )
        self.assertEqual(result.reward.shape, (8, 1))
        self.assertEqual(result.terminated.shape, (8, 1))
        self.assertEqual(result.reward.device, torch.device("cpu"))

    def test_reward_calculator_can_read_declared_environment_tensor(self):
        class DomainReward:
            def __call__(self, context):
                value = context["truth.altitude"]
                return RewardOutput(value, TensorDict({}, context.batch_size, device=context.device))

            def state_dict(self):
                return {}

            def load_state_dict(self, state):
                del state

        cfg = TaskConfig(30.0, 1.3, 20.0, "truth")
        task = AttitudeTrackingTask(cfg, 3, torch.device("cpu"), torch.float32, DomainReward())
        altitude = torch.tensor([[1.0], [2.0], [3.0]])
        result = task.transition(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, -1),
            torch.zeros(3, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, -1),
            torch.zeros(3, 4), torch.zeros(3, 4),
            TensorDict({"truth.altitude": altitude}, [3]),
        )
        torch.testing.assert_close(result.reward, altitude)

    def test_static_randomization_is_instance_episode_deterministic(self):
        spec = StaticParameterSpec(
            "body.mass", -0.1, 0.1, "uniform", "relative", "ratio", "mass"
        )
        randomizer = StaticRandomizer((spec,), 31, torch.device("cpu"), torch.float32)
        randomizer.bind_baselines({"body.mass": torch.full((4,), 2.4)})
        episodes = torch.tensor([1, 1, 1, 1])
        first = randomizer.sample(torch.tensor([True, False, False, False]), episodes)
        second = randomizer.sample(torch.tensor([False, True, True, True]), episodes)
        torch.testing.assert_close(first["body.mass"], second["body.mass"])
        changed = randomizer.sample(torch.ones(4, dtype=torch.bool), episodes + 1)
        self.assertFalse(torch.equal(first["body.mass"], changed["body.mass"]))

    def test_torchrl_rollout_and_recurrent_ppo_update(self):
        env = TensorEnv()
        model = build_actor_critic(21, 4, ModelConfig((16, 16), 8, 8), torch.device("cpu"))
        rollout = TensorDictRolloutCollector(env, model, 8).collect()
        self.assertEqual(rollout.batch_size, torch.Size([4, 8]))
        self.assertEqual(rollout.device, torch.device("cpu"))
        ppo = RecurrentPPO(
            model,
            PPOConfig(0.99, 0.95, 0.2, 0.001, 0.5, 1.0, 3e-4, 1, 4, 2),
            torch.device("cpu"),
        )
        metrics = ppo.update(rollout)
        self.assertIn("loss_objective", metrics)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertTrue(all(value.device == torch.device("cpu") for value in metrics.values()))

    def test_torchrl_mlp_ppo_rollout_and_update(self):
        env = TensorEnv()
        config = ModelConfig((256, 256, 128), 128, 128, architecture="mlp")
        model = build_actor_critic(21, 4, config, torch.device("cpu"))
        self.assertFalse(model.is_recurrent)
        rollout = TensorDictRolloutCollector(env, model, 8).collect()
        self.assertEqual(rollout.batch_size, torch.Size([4, 8]))
        self.assertNotIn("recurrent_state", rollout.keys())
        ppo = TorchRLPPO(
            model,
            PPOConfig(0.99, 0.95, 0.2, 0.001, 0.5, 1.0, 3e-4, 2, 1, 8),
            torch.device("cpu"),
        )
        metrics = ppo.update(rollout)
        self.assertIn("loss_objective", metrics)
        self.assertIn("actor_grad_norm", metrics)
        self.assertIn("critic_grad_norm", metrics)
        self.assertIsNot(ppo.actor_optimizer, ppo.critic_optimizer)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertTrue(all(value.device == torch.device("cpu") for value in metrics.values()))

    def test_mlp_controller_forward_step_is_deterministic_and_bounded(self):
        model = build_actor_critic(
            21,
            4,
            ModelConfig((256, 256, 128), 128, 128, architecture="mlp"),
            torch.device("cpu"),
        )
        observation = torch.randn(3, 21)
        first, first_state = model.forward_step(observation)
        second, second_state = model.forward_step(observation)
        torch.testing.assert_close(first, second)
        self.assertIsNone(first_state)
        self.assertIsNone(second_state)
        self.assertEqual(first.shape, (3, 4))
        self.assertTrue(((first >= -1.0) & (first <= 1.0)).all())
        centered, _ = model.forward_step(torch.zeros(3, 21))
        self.assertLess(centered.abs().max().item(), 0.01)

    def test_residual_action_zero_maps_to_nominal_trim(self):
        mapped = SimEnvAdapter.action_to_command(
            torch.zeros(2, 4),
            torch.full((2, 1), 0.565),
            (0.5539, 0.0, 0.0, 0.0),
            (0.30, 1.0, 1.0, 1.0),
        )
        expected = torch.tensor(
            [[0.565, 0.5539, 0.0, 0.0, 0.0]]
        ).expand(2, -1)
        torch.testing.assert_close(mapped, expected)

    def test_mlp_exploration_std_is_per_action_and_scheduled(self):
        config = ModelConfig(
            (32, 32),
            32,
            32,
            architecture="mlp",
            initial_action_std=(0.12, 0.25, 0.25, 0.25),
            final_action_std=(0.03, 0.05, 0.05, 0.05),
            exploration_decay_control_steps=100,
        )
        model = build_actor_critic(21, 4, config, torch.device("cpu"))
        torch.testing.assert_close(
            model.exploration_std(), torch.tensor([0.12, 0.25, 0.25, 0.25])
        )
        model.set_exploration_progress(0.5)
        torch.testing.assert_close(
            model.exploration_std(), torch.tensor([0.075, 0.15, 0.15, 0.15])
        )
        model.set_exploration_progress(2.0)
        torch.testing.assert_close(
            model.exploration_std(), torch.tensor([0.03, 0.05, 0.05, 0.05])
        )

    def test_reward_v2_penalizes_early_termination_more(self):
        calculator = AttitudeRewardCalculator(
            {
                "termination_penalty": 50.0,
                "early_termination_penalty": 150.0,
            }
        )
        common = {
            "attitude_geodesic_rad": torch.zeros(2, 1),
            "angular_velocity_b": torch.zeros(2, 3),
            "action": torch.zeros(2, 4),
            "previous_action": torch.zeros(2, 4),
            "terminated": torch.ones(2, 1, dtype=torch.bool),
            "tilt_ratio": torch.zeros(2, 1),
            "rate_ratio": torch.zeros(2, 1),
            "episode_age_fraction": torch.tensor([[0.1], [0.9]]),
        }
        output = calculator(TensorDict(common, [2]))
        self.assertLess(output.reward[0].item(), output.reward[1].item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_torchrl_mlp_ppo_rollout_and_update_cuda(self):
        device = torch.device("cuda:0")
        env = TensorEnv(device=device)
        model = build_actor_critic(
            21,
            4,
            ModelConfig((256, 256, 128), 128, 128, architecture="mlp"),
            device,
        )
        rollout = TensorDictRolloutCollector(env, model, 8).collect()
        ppo = TorchRLPPO(
            model,
            PPOConfig(0.99, 0.95, 0.2, 0.001, 0.5, 1.0, 3e-4, 2, 1, 8),
            device,
        )
        metrics = ppo.update(rollout)
        self.assertEqual(rollout.device, device)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertTrue(all(value.device == device for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
