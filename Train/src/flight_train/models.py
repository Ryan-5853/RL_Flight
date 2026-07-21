from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data import Bounded
from torchrl.modules import (
    GRUModule,
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

from .config import ModelConfig


@dataclass(frozen=True)
class ActorCritic:
    actor: ProbabilisticActor
    critic: ValueOperator
    recurrent: GRUModule
    policy_module: TensorDictSequential
    hidden_size: int


def build_actor_critic(
    observation_dim: int,
    action_dim: int,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> ActorCritic:
    encoder = TensorDictModule(
        MLP(
            in_features=observation_dim,
            out_features=config.encoder_sizes[-1],
            num_cells=config.encoder_sizes[:-1],
            activation_class=nn.SiLU,
            activate_last_layer=True,
            device=device,
        ),
        in_keys=["observation"],
        out_keys=["embedding"],
    )
    recurrent = GRUModule(
        input_size=config.encoder_sizes[-1],
        hidden_size=config.hidden_size,
        num_layers=1,
        batch_first=True,
        in_keys=["embedding", "recurrent_state", "is_init"],
        out_keys=["recurrent_features", ("next", "recurrent_state")],
        device=device,
    )
    params = TensorDictModule(
        nn.Sequential(
            MLP(
                in_features=config.hidden_size,
                out_features=2 * action_dim,
                num_cells=[config.actor_head_size],
                activation_class=nn.SiLU,
                device=device,
            ),
            NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4),
        ),
        in_keys=["recurrent_features"],
        out_keys=["loc", "scale"],
    )
    policy_module = TensorDictSequential(encoder, recurrent, params)
    spec = Bounded(
        low=-1.0, high=1.0, shape=(action_dim,), device=device, dtype=dtype
    )
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        spec=spec,
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
        log_prob_key="action_log_prob",
        default_interaction_type="random",
    ).to(device).to(dtype)
    critic = ValueOperator(
        MLP(
            in_features=config.hidden_size,
            out_features=1,
            num_cells=[config.actor_head_size],
            activation_class=nn.SiLU,
            device=device,
        ),
        in_keys=["recurrent_features"],
        out_keys=["state_value"],
    ).to(device).to(dtype)
    return ActorCritic(
        actor=actor,
        critic=critic,
        recurrent=recurrent,
        policy_module=policy_module,
        hidden_size=config.hidden_size,
    )
