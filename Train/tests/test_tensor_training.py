from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torchrl.modules import set_recurrent_mode

from flight_train.algorithms import RecurrentPPO, TorchRLPPO, TorchRLSAC
from flight_train.collector import TensorDictRolloutCollector
from flight_train.commands import VirtualPilotCommandSource
from flight_train.config import (
    ModelConfig,
    PPOConfig,
    SACConfig,
    TaskConfig,
    continuation_resume_config_sha256,
    exact_resume_config_sha256,
    load_experiment_config,
    override_run_config,
)
from flight_train.core import EnvSpec
from flight_train.envs import SimEnvAdapter, _load_timing
from flight_train.evaluation import load_fixed_evaluation_suite
from flight_train.math import quaternion_geodesic_angle
from flight_train.models import (
    BoundedNormalParameters,
    build_actor_critic,
    build_sac_actor_critic,
)
from flight_train.randomization import StaticParameterSpec, StaticRandomizer
from flight_train.rewards import AttitudeRewardCalculator, RewardOutput
from flight_train.runner import (
    _apply_sac_continuation_overrides,
    _restore_policy,
)
from flight_train.tasks import AttitudeTrackingTask
from simenv.config import load_and_materialize


class TensorEnv:
    def __init__(self, batch: int = 4, device: torch.device | None = None) -> None:
        device = device or torch.device("cpu")
        self.spec = EnvSpec(batch, 21, 4, device, torch.float32, 500, 500, ())
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
    def test_source_simenv_configs_use_fixed_single_step_timebase(self):
        environment_root = Path(__file__).parents[1] / "configs/environment"
        expected_strides = {
            "sim_smoke.json": 1,
            "gru_sac_upright_height_only_small_tip.yaml": 20,
            "gru_sac_truth_nominal_no_randomization.yaml": 20,
            "mlp_nominal_baseline_1.yaml": 20,
        }
        for name, expected_stride in expected_strides.items():
            with self.subTest(config=name):
                materialized = load_and_materialize(
                    environment_root / name,
                    parallel_count=2,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                self.assertEqual(materialized.timing.physics_hz, 500)
                self.assertEqual(materialized.timing.control_hz, 500)
                self.assertEqual(materialized.timing.substeps, 1)
                self.assertEqual(
                    materialized.logging.physics_step_stride,
                    expected_stride,
                )
                for sensor in ("gyro", "accelerometer", "motor_speed"):
                    torch.testing.assert_close(
                        materialized.parameters[f"sensors.{sensor}.sample_hz"],
                        torch.full((2,), 500.0),
                    )
                self.assertEqual(
                    materialized.sensor_interpolation["gyro"], "linear"
                )
                self.assertEqual(
                    materialized.sensor_interpolation["accelerometer"],
                    "linear",
                )

    def test_train_timing_preflight_rejects_legacy_multirate_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "timing": {
                            "physics_hz": {"value": 5000},
                            "control_hz": {"value": 500},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must both equal 500"):
                _load_timing(path)

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

    def test_gru_sac_nominal_config_parses_without_curriculum(self):
        path = (
            Path(__file__).parents[1]
            / "configs/experiments/gru_sac_truth_nominal_no_curriculum.yaml"
        )
        config = load_experiment_config(path)
        self.assertEqual(config.algorithm_name, "sac")
        self.assertEqual(config.task.attitude_source, "truth")
        self.assertEqual(config.task.curriculum_durations_s, (30.0,))
        self.assertEqual(config.task.curriculum_target_scales, (1.0,))
        self.assertEqual(config.model.architecture, "gru")
        self.assertEqual(config.model.encoder_sizes, (128, 128))
        self.assertEqual(config.model.hidden_size, 128)
        self.assertEqual(config.model.recurrent_layers, 2)
        self.assertEqual(config.sac.replay_burn_in_steps, 128)
        self.assertEqual(config.sac.replay_sequence_length, 128)
        self.assertEqual(config.sac.replay_sample_length, 256)
        self.assertEqual(config.sac.replay_batch_size, 4096)
        self.assertEqual(config.run.rollout_steps, 256)
        self.assertEqual(config.static_randomization.parameters, {})
        self.assertEqual(config.dynamic_randomization.parameters, {})

    def test_gru_sac_upright_height_only_config_and_initial_tip(self):
        path = (
            Path(__file__).parents[1]
            / "configs/experiments/gru_sac_upright_height_only_small_tip.yaml"
        )
        config = load_experiment_config(path)
        self.assertEqual(
            config.name, "gru_sac_upright_height_only_small_tip"
        )
        self.assertEqual(config.command_source.max_roll_rad, 0.0)
        self.assertEqual(config.command_source.max_pitch_rad, 0.0)
        self.assertEqual(config.command_source.max_yaw_rate_rad_s, 0.0)
        self.assertEqual(config.command_source.initial_target_scale, 0.0)
        self.assertEqual(config.sac.updates_per_collection, 32)
        self.assertEqual(config.sac.replay_burn_in_steps, 128)
        self.assertEqual(config.sac.replay_sequence_length, 128)
        self.assertEqual(config.static_randomization.parameters, {})
        self.assertEqual(config.dynamic_randomization.parameters, {})

        materialized = load_and_materialize(
            config.simulator_config,
            parallel_count=256,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        attitude = materialized.initial_state["attitude_q_wb"]
        angular_velocity = materialized.initial_state[
            "angular_velocity_b"
        ]
        torch.testing.assert_close(
            attitude.norm(dim=-1), torch.ones(256)
        )
        self.assertTrue((angular_velocity[:, 0] >= 0.02).all())
        self.assertTrue((angular_velocity[:, 0] <= 0.04).all())
        self.assertTrue((angular_velocity[:, 1] >= -0.03).all())
        self.assertTrue((angular_velocity[:, 1] <= -0.01).all())
        self.assertTrue((angular_velocity[:, 2].abs() <= 0.005).all())

        pilot = VirtualPilotCommandSource(
            config.command_source,
            batch_size=256,
            device=torch.device("cpu"),
            dtype=torch.float32,
            control_hz=500,
        )
        pilot.reset(torch.ones(256, dtype=torch.bool))
        for _ in range(100):
            pilot.step(torch.zeros(256, 1))
        snapshot = pilot.snapshot()
        self.assertEqual(torch.count_nonzero(snapshot.stick_target), 0)
        self.assertEqual(torch.count_nonzero(snapshot.filtered_stick), 0)
        self.assertEqual(torch.count_nonzero(pilot.desired_yaw_rate), 0)
        expected_upright = torch.zeros(256, 4)
        expected_upright[:, 0] = 1.0
        torch.testing.assert_close(
            snapshot.target_attitude_q_wb, expected_upright
        )

    def test_gru_sac_upright_credit_fixed_config(self):
        path = (
            Path(__file__).parents[1]
            / "configs/experiments/"
            "gru_sac_upright_height_only_small_tip_credit_fixed.yaml"
        )
        config = load_experiment_config(path)
        self.assertEqual(
            config.name,
            "gru_sac_upright_height_only_small_tip_credit_fixed",
        )
        self.assertEqual(config.model.architecture, "gru")
        self.assertEqual(config.model.encoder_sizes, (128, 128))
        self.assertEqual(config.model.hidden_size, 128)
        self.assertEqual(config.model.recurrent_layers, 2)
        self.assertEqual(config.run.rollout_steps, 512)
        self.assertEqual(config.sac.replay_burn_in_steps, 128)
        self.assertEqual(config.sac.replay_sequence_length, 128)
        self.assertEqual(config.sac.replay_sample_length, 256)
        self.assertGreater(
            config.run.rollout_steps,
            config.sac.replay_sample_length,
        )
        self.assertEqual(config.sac.n_step_return, 16)
        self.assertEqual(config.sac.critic_pretraining_updates, 512)
        self.assertEqual(config.sac.updates_per_collection, 64)
        self.assertEqual(config.sac.warmup_transitions, 262_144)
        self.assertEqual(
            config.model.sac_initial_action_std,
            (0.06, 0.05, 0.05, 0.05),
        )
        self.assertEqual(
            config.model.sac_maximum_action_std,
            (0.10, 0.08, 0.08, 0.08),
        )
        self.assertEqual(
            config.evaluation.interval_control_steps,
            1_048_576,
        )
        self.assertEqual(
            config.checkpoint.interval_control_steps,
            524_288,
        )
        self.assertEqual(
            config.evaluation.interval_control_steps
            % config.checkpoint.interval_control_steps,
            0,
        )
        reward_params = config.reward.calculator.params
        self.assertEqual(reward_params["roll_pitch_weight"], 0)
        self.assertEqual(reward_params["tilt_weight"], 0.08)
        self.assertEqual(
            reward_params["angular_rate_weight"],
            0.01 / 8.0**2,
        )
        self.assertEqual(
            reward_params["yaw_rate_weight"],
            0.01 / 8.0**2,
        )
        self.assertEqual(reward_params["alive_bonus"], 0.02)
        self.assertEqual(reward_params["termination_penalty"], 10.0)
        self.assertEqual(reward_params["tilt_barrier_weight"], 0)
        self.assertEqual(reward_params["rate_barrier_weight"], 0)
        # 64 updates × (4096 / 256 sequences) × 128 loss steps
        # 与 256 env × 512 collector steps严格相等。
        effective_training_steps = (
            config.sac.updates_per_collection
            * (
                config.sac.replay_batch_size
                // config.sac.replay_sample_length
            )
            * config.sac.replay_sequence_length
        )
        self.assertEqual(
            effective_training_steps,
            config.run.parallel_count * config.run.rollout_steps,
        )

    def test_mlp_sac_upright_credit_fixed_uses_gentle_actor_updates(self):
        path = (
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_upright_height_only_small_tip_credit_fixed.yaml"
        )
        config = load_experiment_config(path)
        self.assertEqual(
            config.name,
            "mlp_sac_upright_height_only_small_tip_credit_fixed",
        )
        self.assertEqual(config.model.architecture, "mlp")
        self.assertEqual(config.model.encoder_sizes, (128, 128))
        self.assertEqual(
            config.control_contract.observation_history_mode,
            "multirate_actuator",
        )
        self.assertEqual(
            config.control_contract.observation_history_dense_action_steps,
            32,
        )
        self.assertEqual(
            config.control_contract.observation_history_sparse_physical_frames,
            15,
        )
        self.assertEqual(
            config.control_contract
            .observation_history_sparse_physical_stride_steps,
            4,
        )
        self.assertEqual(
            config.control_contract.observation_history_span_steps, 60
        )
        self.assertEqual(config.control_contract.observation_dim, 314)
        self.assertEqual(
            config.control_contract.current_observation_offset, 0
        )
        self.assertEqual(config.sac.replay_burn_in_steps, 0)
        self.assertEqual(config.sac.replay_sequence_length, 1)
        self.assertEqual(config.sac.critic_pretraining_updates, 512)
        self.assertEqual(config.sac.actor_update_interval, 4)
        self.assertEqual(config.sac.actor_learning_rate, 0.00003)
        self.assertEqual(config.sac.critic_learning_rate, 0.0003)
        self.assertEqual(config.sac.updates_per_collection, 64)

    def test_observation_history_uses_strided_oldest_to_current_frames(self):
        env = object.__new__(SimEnvAdapter)
        env.batch_size = 2
        env.base_observation_dim = 21
        env.observation_history_mode = "uniform"
        env.observation_history_frames = 3
        env.observation_history_stride_steps = 2
        env.observation_history_capacity = 5
        env.observation_history = torch.zeros(2, 5, 21)
        env.observation_history_index = 4
        env._observation_history_offsets = torch.tensor([4, 2, 0])
        env._spec = SimpleNamespace(observation_dim=63)

        initial = torch.zeros(2, 21)
        env._reset_observation_history(
            torch.ones(2, dtype=torch.bool), initial
        )
        for value in range(1, 5):
            base = torch.full((2, 21), float(value))
            env._append_observation_history(
                base, torch.zeros(2, dtype=torch.bool)
            )

        stacked = env._history_observation().reshape(2, 3, 21)
        torch.testing.assert_close(
            stacked[:, :, 0],
            torch.tensor([[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]]),
        )

        reset_base = torch.stack(
            (torch.full((21,), 9.0), torch.full((21,), 5.0))
        )
        env._append_observation_history(
            reset_base, torch.tensor([True, False])
        )
        reset_stacked = env._history_observation().reshape(2, 3, 21)
        torch.testing.assert_close(
            reset_stacked[0],
            torch.full((3, 21), 9.0),
        )
        torch.testing.assert_close(
            reset_stacked[1, :, 0],
            torch.tensor([1.0, 3.0, 5.0]),
        )

    def test_multirate_history_keeps_dense_actions_and_sparse_physics(self):
        env = object.__new__(SimEnvAdapter)
        env.batch_size = 1
        env.base_observation_dim = 21
        env.previous_action_start = 17
        env.previous_action_dim = 4
        env.physical_response_dim = 11
        env.observation_history_mode = "multirate_actuator"
        env.observation_history_dense_action_steps = 4
        env.observation_history_sparse_physical_frames = 2
        env.observation_history_capacity = 7
        env.observation_history = torch.zeros(1, 7, 21)
        env.observation_history_index = 6
        env._dense_action_history_offsets = torch.tensor([3, 2, 1, 0])
        env._sparse_physical_history_offsets = torch.tensor([6, 3])
        env._physical_response_indices = torch.arange(4, 15)
        env._spec = SimpleNamespace(observation_dim=59)

        def frame(step: int) -> torch.Tensor:
            value = torch.zeros(1, 21)
            value[:, 0] = float(step)
            value[:, 4:15] = (
                float(step) * 100.0 + torch.arange(11)
            )
            value[:, 17:21] = (
                float(step) * 10.0 + torch.arange(4)
            )
            return value

        env._reset_observation_history(
            torch.ones(1, dtype=torch.bool), frame(0)
        )
        for step in range(1, 7):
            env._append_observation_history(
                frame(step), torch.zeros(1, dtype=torch.bool)
            )

        observation = env._history_observation()
        torch.testing.assert_close(observation[:, :21], frame(6))
        torch.testing.assert_close(
            observation[:, 21:37].reshape(1, 4, 4),
            torch.stack(
                [frame(step)[0, 17:21] for step in range(3, 7)]
            )[None],
        )
        torch.testing.assert_close(
            observation[:, 37:].reshape(1, 2, 11),
            torch.stack(
                [frame(step)[0, 4:15] for step in (0, 3)]
            )[None],
        )

    def test_gru_sac_long_collector_samples_sliding_windows_and_terminal(self):
        device = torch.device("cpu")
        model_config = ModelConfig(
            (8, 8),
            8,
            8,
            architecture="gru",
            sac_initial_action_std=(0.05, 0.05, 0.05, 0.05),
            sac_minimum_action_std=(0.01, 0.01, 0.01, 0.01),
            sac_maximum_action_std=(0.10, 0.10, 0.10, 0.10),
            recurrent_layers=2,
        )
        model = build_sac_actor_critic(21, 4, model_config, device)
        sac = TorchRLSAC(
            model,
            SACConfig(
                gamma=0.99,
                n_step_return=1,
                replay_capacity=128,
                replay_batch_size=16,
                warmup_transitions=16,
                updates_per_collection=1,
                actor_learning_rate=3e-4,
                critic_learning_rate=3e-4,
                alpha_learning_rate=3e-4,
                actor_max_grad_norm=1.0,
                critic_max_grad_norm=5.0,
                target_tau=0.005,
                target_update_interval=1,
                initial_alpha=0.1,
                target_entropy="auto",
                min_alpha=None,
                max_alpha=None,
                replay_sequence_length=2,
                replay_burn_in_steps=2,
            ),
            device,
        )
        batch, time = 2, 8
        terminated = torch.zeros(batch, time, 1, dtype=torch.bool)
        terminated[:, 4] = True
        rollout = TensorDict(
            {
                "observation": torch.randn(batch, time, 21),
                "action": torch.zeros(batch, time, 4),
                "is_init": torch.zeros(batch, time, 1, dtype=torch.bool),
                "next": TensorDict(
                    {
                        "observation": torch.randn(batch, time, 21),
                        "reward": torch.zeros(batch, time, 1),
                        "done": terminated.clone(),
                        "terminated": terminated,
                        "truncated": torch.zeros(
                            batch, time, 1, dtype=torch.bool
                        ),
                        "valid": torch.ones(
                            batch, time, 1, dtype=torch.bool
                        ),
                        "is_init": terminated.clone(),
                    },
                    batch_size=[batch, time],
                ),
            },
            batch_size=[batch, time],
        )
        self.assertEqual(sac.add(rollout), batch * time)
        # 固定 sampler RNG，避免概率性回归测试。
        sac.replay._sampler._rng = torch.Generator().manual_seed(20260726)
        start_offsets: set[int] = set()
        terminal_in_loss = 0
        for _ in range(32):
            sampled = sac.replay.sample().reshape(-1, 4)
            start_offsets.update(
                int(value)
                for value in (
                    sampled["index"][:, 0].squeeze(-1) % time
                ).tolist()
            )
            terminal_in_loss += int(
                sampled[("next", "terminated")][:, 2:].sum()
            )
        # 4-step 样本可以在 5-step 真实 episode 片段的 offset 0 或 1
        # 开始；旧的 collector==sample_length 配置只能从 offset 0 开始。
        self.assertEqual(start_offsets, {0, 1})
        # offset 1 的后两步包含真实 terminal，证明终止样本能进入 loss 段。
        self.assertGreater(terminal_in_loss, 0)

    def test_gru_sac_upright_minimal_reward_exact_terms(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "gru_sac_upright_height_only_small_tip_credit_fixed.yaml"
        )
        calculator = AttitudeRewardCalculator(
            config.reward.calculator.params
        )
        context = TensorDict(
            {
                "attitude_geodesic_rad": torch.zeros(2, 1),
                # 故意提供非零 roll/pitch error，验证重复姿态项已关闭。
                "roll_pitch_error_rad": torch.ones(2, 2),
                "yaw_rate_error_rad_s": torch.tensor([[4.0], [0.0]]),
                "tilt_rad": torch.tensor([[0.5], [0.0]]),
                "angular_velocity_b": torch.tensor(
                    [[2.0, 3.0, 4.0], [0.0, 0.0, 0.0]]
                ),
                "action": torch.ones(2, 4),
                "previous_action": torch.zeros(2, 4),
                "terminated": torch.tensor([[False], [True]]),
                "tilt_ratio": torch.tensor([[0.5], [0.0]]),
                "rate_ratio": torch.tensor([[0.5], [0.0]]),
                "episode_age_fraction": torch.zeros(2, 1),
                "episode_remaining_fraction": torch.ones(2, 1),
            },
            batch_size=[2],
        )
        output = calculator(context)
        normalized_rate_weight = 0.01 / 8.0**2
        expected_first = (
            0.02
            - 0.08 * 0.5**2
            - normalized_rate_weight * (2.0**2 + 3.0**2)
            - normalized_rate_weight * 4.0**2
        )
        torch.testing.assert_close(
            output.reward[:, 0],
            torch.tensor([expected_first, 0.02 - 10.0]),
        )
        for disabled_term in (
            "reward.attitude",
            "reward.action_rate",
            "reward.saturation",
            "reward.risk",
            "reward.survival",
        ):
            self.assertEqual(
                torch.count_nonzero(output.terms[disabled_term]),
                0,
            )
        torch.testing.assert_close(
            output.terms["reward.tilt"][:, 0],
            torch.tensor([-0.08 * 0.5**2, 0.0]),
        )
        torch.testing.assert_close(
            output.terms["reward.angular_rate"][:, 0],
            torch.tensor(
                [
                    -normalized_rate_weight * (2.0**2 + 3.0**2),
                    0.0,
                ]
            ),
        )
        torch.testing.assert_close(
            output.terms["reward.yaw_rate"][:, 0],
            torch.tensor(
                [-normalized_rate_weight * 4.0**2, 0.0]
            ),
        )
        torch.testing.assert_close(
            output.terms["reward.termination"][:, 0],
            torch.tensor([0.0, -10.0]),
        )

    def test_gru_sac_collects_sequences_and_updates(self):
        device = torch.device("cpu")
        model_config = ModelConfig(
            (16, 16),
            16,
            16,
            architecture="gru",
            sac_initial_action_std=(0.10, 0.10, 0.10, 0.10),
            sac_minimum_action_std=(0.01, 0.01, 0.01, 0.01),
            sac_maximum_action_std=(0.50, 0.50, 0.50, 0.50),
            recurrent_layers=2,
        )
        model = build_sac_actor_critic(21, 4, model_config, device)
        rollout = TensorDictRolloutCollector(
            TensorEnv(batch=4, device=device), model, 8
        ).collect()
        self.assertEqual(
            rollout["recurrent_state"].shape, torch.Size([4, 8, 2, 16])
        )
        sac = TorchRLSAC(
            model,
            SACConfig(
                gamma=0.99,
                n_step_return=1,
                replay_capacity=1024,
                replay_batch_size=16,
                warmup_transitions=16,
                updates_per_collection=1,
                actor_learning_rate=3e-4,
                critic_learning_rate=3e-4,
                alpha_learning_rate=3e-4,
                actor_max_grad_norm=1.0,
                critic_max_grad_norm=5.0,
                target_tau=0.005,
                target_update_interval=1,
                initial_alpha=0.1,
                target_entropy="auto",
                min_alpha=None,
                max_alpha=None,
                replay_sequence_length=2,
                replay_burn_in_steps=2,
            ),
            device,
        )
        metrics = sac.update(rollout)
        self.assertEqual(sac.replay_size, 32)
        self.assertEqual(float(metrics["sac_updates"]), 1.0)
        self.assertEqual(float(metrics["sac_actor_updates_total"]), 1.0)
        self.assertTrue(torch.isfinite(metrics["loss_actor"]))
        self.assertTrue(torch.isfinite(metrics["loss_qvalue"]))
        state = sac.state_dict()
        restored_model = build_sac_actor_critic(
            21, 4, model_config, device
        )
        restored = TorchRLSAC(restored_model, sac.config, device)
        restored.load_state_dict(state)
        restored_metrics = restored.update(rollout)
        self.assertEqual(restored.replay_size, 64)
        self.assertEqual(
            float(restored_metrics["sac_actor_updates_total"]), 2.0
        )
        self.assertTrue(torch.isfinite(restored_metrics["loss_actor"]))

    def test_gru_sac_burn_in_rebuilds_detached_initial_states(self):
        device = torch.device("cpu")
        model_config = ModelConfig(
            (8, 8),
            8,
            8,
            architecture="gru",
            sac_initial_action_std=(0.10, 0.10, 0.10, 0.10),
            sac_minimum_action_std=(0.01, 0.01, 0.01, 0.01),
            sac_maximum_action_std=(0.50, 0.50, 0.50, 0.50),
            recurrent_layers=2,
        )
        model = build_sac_actor_critic(21, 4, model_config, device)
        sac = TorchRLSAC(
            model,
            SACConfig(
                gamma=0.99,
                n_step_return=1,
                replay_capacity=128,
                replay_batch_size=8,
                warmup_transitions=8,
                updates_per_collection=1,
                actor_learning_rate=3e-4,
                critic_learning_rate=3e-4,
                alpha_learning_rate=3e-4,
                actor_max_grad_norm=1.0,
                critic_max_grad_norm=5.0,
                target_tau=0.005,
                target_update_interval=1,
                initial_alpha=0.1,
                target_entropy="auto",
                min_alpha=None,
                max_alpha=None,
                replay_sequence_length=2,
                replay_burn_in_steps=2,
            ),
            device,
        )
        observation = torch.randn(2, 4, 21)
        next_observation = torch.randn(2, 4, 21)
        batch = TensorDict(
            {
                "observation": observation,
                "action": torch.zeros(2, 4, 4),
                "is_init": torch.zeros(2, 4, 1, dtype=torch.bool),
                "next": TensorDict(
                    {
                        "observation": next_observation,
                        "reward": torch.zeros(2, 4, 1),
                        "done": torch.zeros(2, 4, 1, dtype=torch.bool),
                        "terminated": torch.zeros(
                            2, 4, 1, dtype=torch.bool
                        ),
                        "truncated": torch.zeros(
                            2, 4, 1, dtype=torch.bool
                        ),
                        "valid": torch.ones(2, 4, 1, dtype=torch.bool),
                        "is_init": torch.zeros(
                            2, 4, 1, dtype=torch.bool
                        ),
                    },
                    batch_size=[2, 4],
                ),
            },
            batch_size=[2, 4],
        )
        with set_recurrent_mode(True):
            train = sac._prepare_recurrent_batch(batch)
        self.assertEqual(train.batch_size, torch.Size([2, 2]))
        root_state = train["recurrent_state"]
        next_state = train[("next", "recurrent_state")]
        self.assertEqual(root_state.shape, torch.Size([2, 2, 2, 8]))
        self.assertFalse(root_state.requires_grad)
        self.assertFalse(next_state.requires_grad)
        self.assertTrue(torch.count_nonzero(root_state[:, 0]) > 0)
        self.assertTrue(torch.count_nonzero(next_state[:, 0]) > 0)
        self.assertEqual(torch.count_nonzero(root_state[:, 1]), 0)
        self.assertEqual(torch.count_nonzero(next_state[:, 1]), 0)

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
                actor_update_interval=2,
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
        self.assertEqual(float(restored_metrics["sac_actor_updates"]), 1.0)
        self.assertEqual(float(restored_metrics["sac_actor_updates_total"]), 1.0)
        self.assertTrue(torch.isfinite(restored_metrics["loss_actor"]))

    def test_sac_policy_anchor_is_training_only_and_penalizes_drift(self):
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
        actor_keys = tuple(model.actor.state_dict())
        sac = TorchRLSAC(
            model,
            SACConfig(
                gamma=0.99,
                n_step_return=1,
                replay_capacity=32,
                replay_batch_size=4,
                warmup_transitions=4,
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
                policy_anchor_weight=10.0,
                policy_anchor_max_action_deviation=0.0,
            ),
            device,
        )
        sac.initialize_policy_anchor()
        self.assertEqual(tuple(model.actor.state_dict()), actor_keys)
        batch = TensorDict(
            {"observation": torch.randn(8, 21)},
            batch_size=[8],
        )
        torch.testing.assert_close(
            sac._policy_anchor_loss(batch), torch.zeros(())
        )
        trainable = [
            parameter
            for parameter in model.policy_module.parameters()
            if parameter.requires_grad
        ]
        with torch.no_grad():
            trainable[-1].add_(0.5)
        anchor_loss = sac._policy_anchor_loss(batch)
        self.assertGreater(anchor_loss.item(), 0.0)
        anchor_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in trainable))
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in sac.policy_anchor_module.parameters()
            )
        )
        self.assertIsNotNone(sac.state_dict()["policy_anchor"])

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

    def test_curriculum_quality_gate_rejects_low_quality_survival(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_upright_height_only_small_tip_stability_v3.yaml"
        )
        self.assertEqual(
            config.task.curriculum_durations_s,
            (2.5, 3.5, 5.0, 10.0, 30.0),
        )
        self.assertAlmostEqual(
            config.task.curriculum_max_roll_pitch_rmse_rad,
            torch.deg2rad(torch.tensor(10.0)).item(),
        )
        self.assertEqual(config.sac.replay_capacity, 2097152)
        self.assertEqual(config.sac.critic_pretraining_updates, 1024)
        self.assertEqual(config.sac.actor_update_interval, 8)
        self.assertEqual(config.sac.actor_learning_rate, 0.000015)
        self.assertEqual(
            config.reward.calculator.params["yaw_rate_weight"], 0.0015625
        )
        self.assertEqual(
            config.task.curriculum_max_yaw_rate_rmse_rad_s, 0.5
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
        env.max_episode_steps = 1250
        env._spec = SimpleNamespace(control_hz=500)
        env.command_source = SimpleNamespace(
            set_curriculum_scale=lambda scale: None
        )
        terminated = torch.zeros(256, 1, dtype=torch.bool)
        truncated = torch.ones(256, 1, dtype=torch.bool)

        metrics = env.update_episode_curriculum(
            terminated,
            truncated,
            torch.zeros_like(truncated),
            allow_promotion=True,
        )
        self.assertEqual(env.curriculum_stage, 0)
        self.assertEqual(float(metrics["curriculum_last_success_fraction"]), 0.0)
        self.assertEqual(
            float(metrics["curriculum_rollout_survival_fraction"]), 1.0
        )
        self.assertEqual(
            float(metrics["curriculum_rollout_quality_fraction"]), 0.0
        )

        quality_success = torch.ones_like(truncated)
        env.update_episode_curriculum(
            terminated,
            truncated,
            quality_success,
            allow_promotion=True,
        )
        metrics = env.update_episode_curriculum(
            terminated,
            truncated,
            quality_success,
            allow_promotion=True,
        )
        self.assertEqual(env.curriculum_stage, 1)
        self.assertEqual(float(metrics["curriculum_promoted"]), 1.0)

    def test_v6_focuses_curriculum_and_reward_on_upright_survival(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_upright_height_only_small_tip_stability_v6.yaml"
        )
        params = config.reward.calculator.params
        self.assertEqual(params["yaw_rate_weight"], 0.0)
        self.assertEqual(params["rate_barrier_weight"], 0.0)
        self.assertAlmostEqual(
            params["roll_pitch_cost_cap_rad"],
            torch.deg2rad(torch.tensor(20.0)).item(),
        )
        self.assertAlmostEqual(
            config.task.curriculum_max_roll_pitch_rmse_rad,
            torch.deg2rad(torch.tensor(10.0)).item(),
        )
        self.assertIsNone(config.task.curriculum_max_yaw_rate_rmse_rad_s)
        self.assertIsNone(config.task.curriculum_max_angular_rate_rms_rad_s)
        suite = load_fixed_evaluation_suite(config.evaluation.suite_path)
        self.assertEqual(suite.schema_version, 3)
        self.assertEqual(suite.yaw_rate_tracking_weight, 0.0)
        self.assertIsNone(
            suite.checkpoint_selection.maximum_hover_yaw_rate_rmse_rad_s
        )

    def test_v7_policy_warm_start_and_yaw_control_config(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_upright_height_only_small_tip_stability_v7.yaml"
        )
        params = config.reward.calculator.params
        self.assertEqual(config.checkpoint.resume_mode, "policy")
        self.assertEqual(
            config.checkpoint.resume_from.name, "step_3145728.pt"
        )
        self.assertEqual(
            config.task.terminate_angular_rate_axes, "roll_pitch"
        )
        self.assertEqual(
            config.task.curriculum_max_yaw_rate_rmse_rad_s, 1.0
        )
        self.assertEqual(params["yaw_rate_weight"], 0.0015625)
        self.assertEqual(params["yaw_rate_huber_delta_rad_s"], 1.0)
        self.assertEqual(params["yaw_rate_cost_cap"], 16.0)
        self.assertNotIn("roll_pitch_cost_cap_rad", params)
        self.assertAlmostEqual(
            params["roll_pitch_huber_delta_rad"],
            torch.deg2rad(torch.tensor(20.0)).item(),
        )
        self.assertEqual(params["rate_barrier_weight"], 0.002)
        self.assertEqual(config.sac.policy_anchor_weight, 10.0)
        self.assertEqual(
            config.sac.policy_anchor_max_action_deviation, 0.08
        )
        self.assertEqual(config.sac.actor_update_interval, 8)
        suite = load_fixed_evaluation_suite(config.evaluation.suite_path)
        self.assertEqual(suite.schema_version, 2)
        self.assertEqual(suite.yaw_rate_tracking_weight, 0.25)
        self.assertEqual(
            suite.checkpoint_selection.maximum_hover_yaw_rate_rmse_rad_s,
            0.5,
        )

    def test_v8_uses_dense_full_history_and_worst_axis_reward(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_upright_height_only_small_tip_stability_v8.yaml"
        )
        params = config.reward.calculator.params
        self.assertEqual(
            config.name,
            "mlp_sac_upright_height_only_small_tip_stability_v8",
        )
        self.assertEqual(
            config.control_contract.observation_history_mode,
            "uniform",
        )
        self.assertEqual(
            config.control_contract.observation_history_frames,
            61,
        )
        self.assertEqual(
            config.control_contract.observation_history_stride_steps,
            1,
        )
        self.assertEqual(
            config.control_contract.observation_history_span_steps,
            60,
        )
        self.assertEqual(config.control_contract.observation_dim, 1281)
        self.assertIsNone(config.checkpoint.resume_from)
        self.assertEqual(config.sac.policy_anchor_weight, 0.0)
        self.assertEqual(config.sac.replay_capacity, 524_288)
        self.assertTrue(config.model.sac_learnable_action_std)
        self.assertEqual(
            config.model.sac_initial_action_std,
            (0.10, 0.10, 0.10, 0.10),
        )
        self.assertEqual(config.sac.initial_alpha, 0.01)
        self.assertGreater(params["joint_tracking_weight"], 0.0)
        self.assertEqual(params["roll_pitch_weight"], 0.0)
        self.assertEqual(params["tilt_weight"], 0.0)
        self.assertEqual(params["yaw_rate_weight"], 0.0)

    def test_v9_continues_v8_final_with_moderate_actor_updates(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_upright_height_only_small_tip_stability_v9.yaml"
        )
        params = config.reward.calculator.params
        self.assertEqual(
            config.name,
            "mlp_sac_upright_height_only_small_tip_stability_v9",
        )
        self.assertEqual(config.control_contract.observation_dim, 1281)
        self.assertEqual(config.run.total_control_steps, 33_554_432)
        self.assertEqual(config.checkpoint.resume_mode, "continuation")
        self.assertEqual(
            config.checkpoint.resume_from.name,
            "step_16777216.pt",
        )
        self.assertEqual(config.sac.replay_capacity, 524_288)
        self.assertEqual(config.sac.critic_pretraining_updates, 2048)
        self.assertEqual(config.sac.actor_update_interval, 4)
        self.assertEqual(config.sac.actor_learning_rate, 0.00003)
        self.assertEqual(config.sac.policy_anchor_weight, 0.0)
        self.assertTrue(config.model.sac_learnable_action_std)
        self.assertEqual(
            config.model.sac_initial_action_std,
            (0.06, 0.06, 0.06, 0.06),
        )
        self.assertEqual(
            config.model.sac_minimum_action_std,
            (0.01, 0.01, 0.01, 0.01),
        )
        self.assertEqual(
            config.model.sac_maximum_action_std,
            (0.12, 0.12, 0.12, 0.12),
        )
        self.assertEqual(config.sac.initial_alpha, 0.003)
        self.assertEqual(config.sac.min_alpha, 0.0003)
        self.assertEqual(config.sac.max_alpha, 0.01)
        self.assertEqual(config.evaluation.execution.max_in_flight, 2)
        self.assertGreater(params["joint_tracking_weight"], 0.0)
        self.assertEqual(params["roll_pitch_weight"], 0.0)
        self.assertEqual(params["tilt_weight"], 0.0)
        self.assertEqual(params["yaw_rate_weight"], 0.0)

        v8 = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_upright_height_only_small_tip_stability_v8.yaml"
        )
        self.assertNotEqual(
            exact_resume_config_sha256(v8),
            exact_resume_config_sha256(config),
        )
        self.assertEqual(
            continuation_resume_config_sha256(v8),
            continuation_resume_config_sha256(config),
        )
        incompatible = copy.deepcopy(config.raw)
        incompatible["algorithm"]["entropy"]["max_alpha"] = None
        self.assertNotEqual(
            continuation_resume_config_sha256(v8),
            continuation_resume_config_sha256(incompatible),
        )

    def test_command_tracking_v1_policy_warm_starts_with_small_envelope(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_attitude_command_tracking_v1.yaml"
        )
        params = config.reward.calculator.params
        self.assertEqual(
            config.name,
            "mlp_sac_attitude_command_tracking_v1",
        )
        self.assertEqual(config.checkpoint.resume_mode, "policy")
        self.assertEqual(
            config.checkpoint.resume_from.name,
            "best_fixed_evaluation.pt",
        )
        self.assertEqual(config.run.total_control_steps, 16_777_216)
        self.assertEqual(config.control_contract.observation_dim, 1281)
        self.assertEqual(config.command_source.max_roll_rad, 0.10)
        self.assertEqual(config.command_source.max_pitch_rad, 0.10)
        self.assertEqual(config.command_source.max_yaw_rate_rad_s, 0.35)
        self.assertEqual(
            config.task.curriculum_target_scales,
            (0.0, 0.0, 0.10, 0.25, 0.50, 1.0),
        )
        self.assertAlmostEqual(
            config.task.curriculum_max_roll_pitch_rmse_rad,
            torch.deg2rad(torch.tensor(2.5)).item(),
        )
        self.assertEqual(
            config.task.curriculum_max_yaw_rate_rmse_rad_s,
            0.12,
        )
        self.assertAlmostEqual(
            params["joint_roll_pitch_scale_rad"],
            torch.deg2rad(torch.tensor(5.0)).item(),
        )
        self.assertEqual(params["joint_yaw_rate_scale_rad_s"], 0.35)
        self.assertEqual(
            config.model.sac_initial_action_std,
            (0.04, 0.05, 0.05, 0.05),
        )
        self.assertTrue(
            all(value < 0.06 for value in config.model.sac_initial_action_std)
        )
        self.assertEqual(
            config.model.sac_maximum_action_std,
            (0.06, 0.08, 0.08, 0.08),
        )
        suite = load_fixed_evaluation_suite(config.evaluation.suite_path)
        self.assertEqual(suite.schema_version, 4)
        self.assertIn(
            "yaw_rate_full_step",
            {item.name for item in suite.scenarios},
        )

    def test_command_tracking_v2_protects_warm_started_policy(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_attitude_command_tracking_v2.yaml"
        )
        self.assertEqual(
            config.name,
            "mlp_sac_attitude_command_tracking_v2",
        )
        self.assertEqual(config.checkpoint.resume_mode, "policy")
        self.assertEqual(
            config.checkpoint.resume_from.name,
            "best_fixed_evaluation.pt",
        )
        self.assertEqual(
            config.task.curriculum_target_scales,
            (0.05, 0.10, 0.25, 0.50, 0.75, 1.0),
        )
        self.assertEqual(config.sac.critic_pretraining_updates, 2048)
        self.assertEqual(config.sac.actor_update_interval, 8)
        self.assertEqual(config.sac.policy_anchor_weight, 10.0)
        self.assertEqual(
            config.sac.policy_anchor_max_action_deviation,
            0.05,
        )
        self.assertEqual(config.sac.actor_learning_rate, 0.000015)
        self.assertEqual(
            config.model.sac_initial_action_std,
            (0.04, 0.05, 0.05, 0.05),
        )
        self.assertEqual(
            config.model.sac_maximum_action_std,
            (0.06, 0.08, 0.08, 0.08),
        )
        self.assertEqual(
            config.evaluation.interval_control_steps,
            524_288,
        )
        self.assertEqual(
            config.checkpoint.interval_control_steps,
            524_288,
        )

    def test_command_tracking_v3_continues_v2_to_full_envelope(self):
        root = Path(__file__).parents[1] / "configs/experiments"
        v2 = load_experiment_config(
            root / "mlp_sac_attitude_command_tracking_v2.yaml"
        )
        v3 = load_experiment_config(
            root / "mlp_sac_attitude_command_tracking_v3.yaml"
        )
        self.assertEqual(
            v3.name,
            "mlp_sac_attitude_command_tracking_v3",
        )
        self.assertEqual(v3.checkpoint.resume_mode, "continuation")
        self.assertEqual(
            v3.checkpoint.resume_from.name,
            "step_16777216.pt",
        )
        self.assertEqual(v3.run.total_control_steps, 67_108_864)
        self.assertEqual(
            v3.task.curriculum_target_scales,
            (0.05, 0.10, 0.25, 0.50, 0.75, 1.0),
        )
        self.assertEqual(v3.sac.actor_update_interval, 16)
        self.assertEqual(v3.sac.actor_learning_rate, 0.0000075)
        self.assertEqual(
            v3.model.sac_maximum_action_std,
            (0.05, 0.07, 0.07, 0.07),
        )
        self.assertEqual(
            v3.evaluation.interval_control_steps,
            4_194_304,
        )
        self.assertEqual(
            v3.checkpoint.interval_control_steps,
            2_097_152,
        )
        self.assertEqual(v3.evaluation.execution.max_in_flight, 1)
        self.assertEqual(
            continuation_resume_config_sha256(v3),
            continuation_resume_config_sha256(v2),
        )
        source_run_config = json.loads(
            (
                v3.checkpoint.resume_from.parents[1]
                / "config.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            continuation_resume_config_sha256(v3),
            continuation_resume_config_sha256(source_run_config),
        )

    def test_sac_continuation_applies_runtime_bounds_and_learning_rates(self):
        config = load_experiment_config(
            Path(__file__).parents[1]
            / "configs/experiments/"
            "mlp_sac_upright_height_only_small_tip_stability_v9.yaml"
        )
        device = torch.device("cpu")
        model = build_sac_actor_critic(
            config.control_contract.observation_dim,
            4,
            config.model,
            device,
        )
        algorithm = TorchRLSAC(model, config.sac, device)
        distribution = next(
            module
            for module in model.actor.modules()
            if hasattr(module, "maximum_std")
        )
        distribution.initial_std.fill_(0.10)
        distribution.minimum_std.fill_(0.025)
        distribution.maximum_std.fill_(0.20)
        algorithm.loss.alpha_init.fill_(0.01)
        algorithm.loss.min_log_alpha.fill_(torch.log(torch.tensor(0.001)))
        algorithm.loss.max_log_alpha.fill_(torch.log(torch.tensor(0.05)))
        algorithm.loss.log_alpha.data.fill_(torch.log(torch.tensor(0.02)))
        algorithm.actor_optimizer.param_groups[0]["lr"] = 0.5

        _apply_sac_continuation_overrides(config, model, algorithm)

        torch.testing.assert_close(
            distribution.initial_std,
            torch.full((4,), 0.06),
        )
        torch.testing.assert_close(
            distribution.minimum_std,
            torch.full((4,), 0.01),
        )
        torch.testing.assert_close(
            distribution.maximum_std,
            torch.full((4,), 0.12),
        )
        self.assertAlmostEqual(
            float(algorithm.loss.log_alpha.detach().exp()),
            0.01,
        )
        self.assertEqual(
            algorithm.actor_optimizer.param_groups[0]["lr"],
            0.00003,
        )
        self.assertEqual(
            algorithm.critic_optimizer.param_groups[0]["lr"],
            0.0002,
        )
        self.assertEqual(
            algorithm.alpha_optimizer.param_groups[0]["lr"],
            0.0001,
        )
        self.assertAlmostEqual(
            float(algorithm.loss.min_log_alpha.detach().exp()),
            0.0003,
        )
        self.assertAlmostEqual(
            float(algorithm.loss.max_log_alpha.detach().exp()),
            0.01,
        )

    def test_joint_tracking_reward_is_noncompensatory_and_follows_worst_axis(self):
        calculator = AttitudeRewardCalculator(
            {
                "roll_pitch_weight": 0.0,
                "tilt_weight": 0.0,
                "yaw_rate_weight": 0.0,
                "joint_tracking_weight": 1.0,
                "joint_roll_pitch_scale_rad": 1.0,
                "joint_yaw_rate_scale_rad_s": 1.0,
                "joint_tracking_huber_delta": 1.0,
                "angular_rate_weight": 0.0,
                "action_rate_weight": 0.0,
                "saturation_weight": 0.0,
                "alive_bonus": 0.0,
                "tilt_barrier_weight": 0.0,
                "rate_barrier_weight": 0.0,
            }
        )
        roll_pitch = torch.tensor(
            [[0.5, 0.0], [0.1, 0.0], [2.0, 0.0], [0.5, 0.0]],
            requires_grad=True,
        )
        yaw_rate = torch.tensor(
            [[2.0], [2.0], [0.5], [0.5]],
            requires_grad=True,
        )
        context = TensorDict(
            {
                "roll_pitch_error_rad": roll_pitch,
                "yaw_rate_error_rad_s": yaw_rate,
                "tilt_rad": torch.zeros(4, 1),
                "angular_velocity_b": torch.zeros(4, 3),
                "action": torch.zeros(4, 4),
                "previous_action": torch.zeros(4, 4),
                "terminated": torch.zeros(4, 1, dtype=torch.bool),
                "tilt_ratio": torch.zeros(4, 1),
                "rate_ratio": torch.zeros(4, 1),
                "episode_age_fraction": torch.zeros(4, 1),
            },
            batch_size=[4],
        )
        output = calculator(context)
        # 前两项的 yaw 都是较差轴；继续改善已经较好的 RP 不增加奖励。
        torch.testing.assert_close(
            output.terms["reward.joint_tracking"][:, 0],
            torch.tensor([-3.0, -3.0, -3.0, -0.25]),
        )
        torch.testing.assert_close(
            output.diagnostics["joint_roll_pitch_cost"][:, 0],
            torch.tensor([0.25, 0.01, 3.0, 0.25]),
        )
        torch.testing.assert_close(
            output.diagnostics["joint_yaw_rate_cost"][:, 0],
            torch.tensor([3.0, 3.0, 0.25, 0.25]),
        )
        torch.testing.assert_close(
            output.diagnostics["joint_roll_pitch_dominant"][:, 0],
            torch.tensor([0.0, 0.0, 1.0, 0.5]),
        )
        torch.testing.assert_close(
            output.diagnostics["joint_yaw_rate_dominant"][:, 0],
            torch.tensor([1.0, 1.0, 0.0, 0.5]),
        )
        output.reward.sum().backward()
        # yaw 很差时只有 yaw 获得主跟踪梯度；RP 很差时则相反。
        self.assertEqual(float(roll_pitch.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(yaw_rate.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(roll_pitch.grad[2].abs().sum()), 0.0)
        self.assertEqual(float(yaw_rate.grad[2].abs().sum()), 0.0)

    def test_joint_tracking_rejects_legacy_linear_axis_rewards(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            AttitudeRewardCalculator(
                {
                    "joint_tracking_weight": 0.01,
                    "roll_pitch_weight": 0.1,
                    "tilt_weight": 0.0,
                    "yaw_rate_weight": 0.0,
                }
            )

    def test_roll_pitch_reward_cost_can_be_capped(self):
        cap_rad = torch.deg2rad(torch.tensor(20.0)).item()
        calculator = AttitudeRewardCalculator(
            {
                "roll_pitch_weight": 0.55,
                "roll_pitch_cost_cap_rad": cap_rad,
                "tilt_weight": 0.0,
                "yaw_rate_weight": 0.0,
                "angular_rate_weight": 0.0,
                "action_rate_weight": 0.0,
                "saturation_weight": 0.0,
                "alive_bonus": 0.0,
                "tilt_barrier_weight": 0.0,
                "rate_barrier_weight": 0.0,
            }
        )
        angles = torch.deg2rad(torch.tensor([10.0, 20.0, 40.0]))
        context = TensorDict(
            {
                "roll_pitch_error_rad": torch.stack(
                    (angles, torch.zeros_like(angles)), dim=-1
                ),
                "tilt_rad": torch.zeros(3, 1),
                "yaw_rate_error_rad_s": torch.zeros(3, 1),
                "angular_velocity_b": torch.zeros(3, 3),
                "action": torch.zeros(3, 4),
                "previous_action": torch.zeros(3, 4),
                "terminated": torch.zeros(3, 1, dtype=torch.bool),
                "tilt_ratio": torch.zeros(3, 1),
                "rate_ratio": torch.zeros(3, 1),
                "episode_age_fraction": torch.zeros(3, 1),
            },
            batch_size=[3],
        )
        attitude_reward = calculator(context).terms["reward.attitude"].squeeze(-1)
        self.assertAlmostEqual(
            attitude_reward[0].item(),
            -0.55 * torch.deg2rad(torch.tensor(10.0)).square().item(),
        )
        torch.testing.assert_close(attitude_reward[1], attitude_reward[2])

    def test_yaw_rate_reward_can_transition_from_quadratic_to_linear(self):
        calculator = AttitudeRewardCalculator(
            {
                "roll_pitch_weight": 0.0,
                "tilt_weight": 0.0,
                "yaw_rate_weight": 0.0015625,
                "yaw_rate_huber_delta_rad_s": 1.0,
                "angular_rate_weight": 0.0,
                "action_rate_weight": 0.0,
                "saturation_weight": 0.0,
                "tilt_barrier_weight": 0.0,
                "rate_barrier_weight": 0.0,
            }
        )
        context = TensorDict(
            {
                "roll_pitch_error_rad": torch.zeros(3, 2),
                "tilt_rad": torch.zeros(3, 1),
                "yaw_rate_error_rad_s": torch.tensor([[0.5], [1.0], [4.0]]),
                "angular_velocity_b": torch.zeros(3, 3),
                "action": torch.zeros(3, 4),
                "previous_action": torch.zeros(3, 4),
                "terminated": torch.zeros(3, 1, dtype=torch.bool),
                "tilt_ratio": torch.zeros(3, 1),
                "rate_ratio": torch.zeros(3, 1),
                "episode_age_fraction": torch.zeros(3, 1),
            },
            batch_size=[3],
        )
        yaw_reward = calculator(context).terms["reward.yaw_rate"].squeeze(-1)
        torch.testing.assert_close(
            yaw_reward,
            -0.0015625 * torch.tensor([0.25, 1.0, 7.0]),
        )

    def test_roll_pitch_huber_and_yaw_cost_cap_keep_safety_dominant(self):
        delta = torch.deg2rad(torch.tensor(20.0)).item()
        calculator = AttitudeRewardCalculator(
            {
                "roll_pitch_weight": 0.55,
                "roll_pitch_huber_delta_rad": delta,
                "tilt_weight": 0.0,
                "yaw_rate_weight": 0.0015625,
                "yaw_rate_huber_delta_rad_s": 1.0,
                "yaw_rate_cost_cap": 16.0,
                "angular_rate_weight": 0.0,
                "action_rate_weight": 0.0,
                "saturation_weight": 0.0,
                "tilt_barrier_weight": 0.0,
                "rate_barrier_weight": 0.0,
            }
        )
        angles = torch.deg2rad(torch.tensor([10.0, 20.0, 40.0]))
        yaw_errors = torch.tensor(
            [[0.5], [4.0], [20.0]], requires_grad=True
        )
        context = TensorDict(
            {
                "roll_pitch_error_rad": torch.stack(
                    (angles, torch.zeros_like(angles)), dim=-1
                ),
                "tilt_rad": torch.zeros(3, 1),
                "yaw_rate_error_rad_s": yaw_errors,
                "angular_velocity_b": torch.zeros(3, 3),
                "action": torch.zeros(3, 4),
                "previous_action": torch.zeros(3, 4),
                "terminated": torch.zeros(3, 1, dtype=torch.bool),
                "tilt_ratio": torch.zeros(3, 1),
                "rate_ratio": torch.zeros(3, 1),
                "episode_age_fraction": torch.zeros(3, 1),
            },
            batch_size=[3],
        )
        terms = calculator(context).terms
        expected_roll_pitch_cost = torch.tensor(
            [
                angles[0].square().item(),
                angles[1].square().item(),
                2.0 * delta * angles[2].item() - delta**2,
            ]
        )
        torch.testing.assert_close(
            terms["reward.attitude"].squeeze(-1),
            -0.55 * expected_roll_pitch_cost,
        )
        torch.testing.assert_close(
            terms["reward.yaw_rate"].squeeze(-1),
            -0.0015625
            * (
                16.0
                * torch.tensor([0.25, 7.0, 39.0])
                / (16.0 + torch.tensor([0.25, 7.0, 39.0]))
            ),
        )
        terms["reward.yaw_rate"].sum().backward()
        self.assertNotEqual(yaw_errors.grad[-1].item(), 0.0)

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

    def test_roll_pitch_rate_termination_does_not_terminate_on_yaw(self):
        cfg = TaskConfig(
            30.0,
            1.3,
            8.0,
            "truth",
            terminate_angular_rate_axes="roll_pitch",
        )
        calculator = AttitudeRewardCalculator(
            {
                "roll_pitch_weight": 0.0,
                "tilt_weight": 0.0,
                "yaw_rate_weight": 0.0,
                "angular_rate_weight": 0.0,
            }
        )
        task = AttitudeTrackingTask(
            cfg, 2, torch.device("cpu"), torch.float32, calculator
        )
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1)
        result = task.transition(
            identity,
            torch.tensor([[0.0, 0.0, 9.0], [9.0, 0.0, 0.0]]),
            identity,
            torch.zeros(2, 4),
            torch.zeros(2, 4),
        )
        self.assertFalse(bool(result.terminated[0]))
        self.assertTrue(bool(result.terminated[1]))

    def test_policy_resume_restores_mean_actor_and_resets_target_std(self):
        source_config = ModelConfig(
            (32, 32),
            32,
            32,
            architecture="mlp",
            sac_initial_action_std=(0.06, 0.06, 0.06, 0.06),
            sac_minimum_action_std=(0.01, 0.01, 0.01, 0.01),
            sac_maximum_action_std=(0.12, 0.12, 0.12, 0.12),
        )
        target_config = ModelConfig(
            (32, 32),
            32,
            32,
            architecture="mlp",
            sac_initial_action_std=(0.04, 0.05, 0.05, 0.05),
            sac_minimum_action_std=(0.01, 0.015, 0.015, 0.015),
            sac_maximum_action_std=(0.06, 0.08, 0.08, 0.08),
        )
        source = build_sac_actor_critic(
            21, 4, source_config, torch.device("cpu")
        )
        target = build_sac_actor_critic(
            21, 4, target_config, torch.device("cpu")
        )
        with torch.no_grad():
            for parameter in source.actor.parameters():
                parameter.fill_(0.125)
            for parameter in source.qvalue.parameters():
                parameter.fill_(0.75)
        observation = torch.linspace(-1.0, 1.0, 42).reshape(2, 21)
        source_td = TensorDict(
            {"observation": observation.clone()},
            batch_size=[2],
        )
        source.policy_module(source_td)
        qvalue_before = {
            name: value.clone() for name, value in target.qvalue.state_dict().items()
        }
        _restore_policy({"actor": source.actor.state_dict()}, model=target)

        target_td = TensorDict(
            {"observation": observation.clone()},
            batch_size=[2],
        )
        target.policy_module(target_td)
        torch.testing.assert_close(target_td["loc"], source_td["loc"])
        torch.testing.assert_close(
            target_td["scale"],
            torch.tensor(
                target_config.sac_initial_action_std,
            ).expand(2, -1),
        )
        distribution = next(
            module
            for module in target.actor.modules()
            if isinstance(module, BoundedNormalParameters)
        )
        torch.testing.assert_close(
            distribution.minimum_std,
            torch.tensor(target_config.sac_minimum_action_std),
        )
        torch.testing.assert_close(
            distribution.maximum_std,
            torch.tensor(target_config.sac_maximum_action_std),
        )
        for name, value in qvalue_before.items():
            torch.testing.assert_close(target.qvalue.state_dict()[name], value)

    def test_reward_terms_log_alive_and_sum_to_total_reward(self):
        calculator = AttitudeRewardCalculator(
            {
                "roll_pitch_weight": 0.0,
                "tilt_weight": 0.0,
                "yaw_rate_weight": 0.0,
                "angular_rate_weight": 0.0,
                "action_rate_weight": 0.0,
                "saturation_weight": 0.0,
                "alive_bonus": 0.05,
                "survival_progress_weight": 0.4,
                "termination_penalty": 2.0,
                "early_termination_penalty": 1.0,
            }
        )
        context = TensorDict(
            {
                "attitude_geodesic_rad": torch.zeros(2, 1),
                "roll_pitch_error_rad": torch.zeros(2, 2),
                "yaw_rate_error_rad_s": torch.zeros(2, 1),
                "tilt_rad": torch.zeros(2, 1),
                "angular_velocity_b": torch.zeros(2, 3),
                "action": torch.zeros(2, 4),
                "previous_action": torch.zeros(2, 4),
                "terminated": torch.tensor([[False], [True]]),
                "tilt_ratio": torch.zeros(2, 1),
                "rate_ratio": torch.zeros(2, 1),
                "episode_age_fraction": torch.tensor([[0.25], [0.75]]),
                "episode_remaining_fraction": torch.tensor(
                    [[0.75], [0.25]]
                ),
            },
            batch_size=[2],
        )
        output = calculator(context)
        torch.testing.assert_close(
            output.terms["reward.alive"],
            torch.full((2, 1), 0.05),
        )
        torch.testing.assert_close(
            output.terms["reward.survival"],
            torch.tensor([[0.10], [0.30]]),
        )
        term_sum = torch.stack(
            list(output.terms.values()), dim=0
        ).sum(dim=0)
        torch.testing.assert_close(term_sum, output.reward)

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
