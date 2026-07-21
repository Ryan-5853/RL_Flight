from __future__ import annotations

import unittest

import torch
from tensordict import TensorDict

from flight_train.algorithms import RecurrentPPO
from flight_train.collector import TensorDictRolloutCollector
from flight_train.config import ModelConfig, PPOConfig, RewardConfig, TaskConfig
from flight_train.core import EnvSpec
from flight_train.envs import SimEnvAdapter
from flight_train.math import quaternion_geodesic_angle
from flight_train.models import build_actor_critic
from flight_train.tasks import AttitudeTrackingTask


class TensorEnv:
    def __init__(self, batch: int = 4, device: torch.device | None = None) -> None:
        device = device or torch.device("cpu")
        self.spec = EnvSpec(batch, 21, 5, device, torch.float32, 5000, 500, ())
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
    def test_action_transform_exact_ranges(self):
        standard = torch.tensor([[-1.0, 1.0, -1.0, 0.0, 1.0]])
        command = SimEnvAdapter.action_to_command(standard)
        torch.testing.assert_close(command, torch.tensor([[0.0, 1.0, -1.0, 0.0, 1.0]]))

    def test_quaternion_sign_has_same_reward_distance(self):
        q = torch.tensor([[0.9238795, 0.3826834, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(
            quaternion_geodesic_angle(q, target),
            quaternion_geodesic_angle(-q, target),
        )

    def test_reward_kernel_stays_batched_and_on_device(self):
        reward = RewardConfig(4.0, 0.1, 0.01, 0.02, 0.1)
        cfg = TaskConfig(30.0, 0.3, 3.14, 1.3, 20.0, "truth", reward)
        task = AttitudeTrackingTask(cfg, 8, torch.device("cpu"), torch.float32, 1)
        task.reset(torch.ones(8, dtype=torch.bool))
        result = task.transition(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(8, -1),
            torch.zeros(8, 3),
            torch.zeros(8, 5),
            torch.zeros(8, 5),
        )
        self.assertEqual(result.reward.shape, (8, 1))
        self.assertEqual(result.terminated.shape, (8, 1))
        self.assertEqual(result.reward.device, torch.device("cpu"))

    def test_torchrl_rollout_and_recurrent_ppo_update(self):
        env = TensorEnv()
        model = build_actor_critic(21, 5, ModelConfig((16, 16), 8, 8), torch.device("cpu"))
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


if __name__ == "__main__":
    unittest.main()
