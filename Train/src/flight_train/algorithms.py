from __future__ import annotations

from collections.abc import Mapping

import torch
from tensordict import TensorDictBase
from torchrl.modules import set_recurrent_mode
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from .config import PPOConfig
from .models import ActorCritic


class RecurrentPPO:
    """TorchRL PPO objective over GPU-resident recurrent TensorDict sequences."""

    def __init__(self, model: ActorCritic, config: PPOConfig, device: torch.device) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.advantage = GAE(
            gamma=config.gamma,
            lmbda=config.gae_lambda,
            value_network=None,
            average_gae=True,
            vectorized=True,
            time_dim=1,
            device=device,
        )
        self.loss = ClipPPOLoss(
            actor_network=model.actor,
            critic_network=model.critic,
            clip_epsilon=config.clip_epsilon,
            entropy_bonus=config.entropy_coefficient > 0,
            entropy_coeff=config.entropy_coefficient,
            critic_coeff=config.value_coefficient,
            normalize_advantage=False,
            loss_critic_type="smooth_l1",
            reduction="mean",
            device=device,
        )
        parameters = list(self.loss.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    def update(self, rollout: TensorDictBase) -> Mapping[str, torch.Tensor]:
        if rollout.device != self.device:
            raise ValueError(f"rollout must stay on {self.device}, got {rollout.device}")
        with torch.no_grad():
            self.advantage(rollout)
            # ClipPPOLoss natively applies ``shifted_valid`` to every loss term.
            rollout["shifted_valid"] = rollout[("next", "valid")]

        batch, time = rollout.batch_size
        length = self.config.sequence_length
        chunks = time // length
        # Keep complete temporal chunks so GRU burn/reset semantics are preserved.
        sequences = rollout.reshape(batch, chunks, length).transpose(0, 1).reshape(batch * chunks, length)
        sequence_count = sequences.batch_size[0]
        aggregate: dict[str, torch.Tensor] = {}
        updates = 0
        with set_recurrent_mode(True):
            for _ in range(self.config.epochs):
                permutation = torch.randperm(sequence_count, device=self.device)
                for start in range(0, sequence_count, self.config.sequences_per_minibatch):
                    index = permutation[start : start + self.config.sequences_per_minibatch]
                    minibatch = sequences[index]
                    loss_td = self.loss(minibatch)
                    objective = (
                        loss_td["loss_objective"]
                        + loss_td.get("loss_entropy", 0.0)
                        + loss_td["loss_critic"]
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.loss.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    for key, value in loss_td.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            aggregate[key] = aggregate.get(key, torch.zeros_like(value)) + value.detach()
                    aggregate["grad_norm"] = aggregate.get(
                        "grad_norm", torch.zeros((), device=self.device)
                    ) + torch.as_tensor(grad_norm, device=self.device)
                    updates += 1
        return {key: value / updates for key, value in aggregate.items()}
