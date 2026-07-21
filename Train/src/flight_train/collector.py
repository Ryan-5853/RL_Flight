from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase

from .core import BatchedControlEnv
from .models import ActorCritic


class TensorDictRolloutCollector:
    """Synchronous GPU collector producing TorchRL's canonical transition keys."""

    def __init__(self, env: BatchedControlEnv, model: ActorCritic, rollout_steps: int) -> None:
        self.env = env
        self.model = model
        self.rollout_steps = rollout_steps
        self.device = env.spec.device
        self.batch_size = env.spec.parallel_count
        self.current = env.reset()
        self.current.set(
            "recurrent_state",
            torch.zeros(
                (self.batch_size, 1, model.hidden_size),
                device=self.device,
                dtype=env.spec.dtype,
            ),
        )

    @torch.no_grad()
    def collect(self) -> TensorDictBase:
        steps: list[TensorDictBase] = []
        current = self.current
        for _ in range(self.rollout_steps):
            policy_td = current.select(
                "observation", "recurrent_state", "is_init", strict=True
            ).clone(False)
            self.model.actor(policy_td)
            self.model.critic(policy_td)
            next_env = self.env.step(policy_td["action"])
            transition = TensorDict(
                {
                    "observation": current["observation"],
                    "recurrent_state": current["recurrent_state"],
                    "is_init": current["is_init"],
                    "action": policy_td["action"],
                    "action_log_prob": policy_td["action_log_prob"],
                    "state_value": policy_td["state_value"],
                    "next": TensorDict(
                        {
                            "observation": next_env["observation"],
                            "reward": next_env["reward"],
                            "terminated": next_env["terminated"],
                            "truncated": next_env["truncated"],
                            "done": next_env["done"],
                            "valid": next_env["valid"],
                            "is_init": next_env["is_init"],
                        },
                        batch_size=[self.batch_size],
                        device=self.device,
                    ),
                },
                batch_size=[self.batch_size],
                device=self.device,
            )
            steps.append(transition)
            current = TensorDict(
                {
                    "observation": next_env["observation"],
                    "is_init": next_env["is_init"],
                    "recurrent_state": policy_td[("next", "recurrent_state")],
                },
                batch_size=[self.batch_size],
                device=self.device,
            )

        # TorchRL recurrent modules consume [B,T,...], with time last in batch_size.
        rollout = torch.stack(steps, dim=1)
        bootstrap_td = current.clone(False)
        self.model.policy_module(bootstrap_td)
        self.model.critic(bootstrap_td)
        values = rollout["state_value"]
        next_values = torch.cat((values[:, 1:], bootstrap_td["state_value"][:, None]), dim=1)
        rollout[("next", "state_value")] = next_values
        rollout[("next", "reward")] = torch.where(
            rollout[("next", "valid")], rollout[("next", "reward")], torch.zeros_like(rollout[("next", "reward")])
        )
        self.current = current
        return rollout
