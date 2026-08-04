from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict
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
from torchrl.envs.utils import ExplorationType, set_exploration_type

from .config import ModelConfig


@dataclass(frozen=True)
class ActorCritic:
    """TorchRL 策略/价值网络集合，支持无状态 MLP 与有状态 GRU。

    ``policy_module`` 只计算编码、循环特征和分布参数，供 rollout 末端的
    bootstrap 使用；``actor`` 在此基础上采样动作并计算旧策略对数概率。
    """

    actor: ProbabilisticActor
    critic: ValueOperator
    recurrent: GRUModule | None
    policy_module: TensorDictSequential
    hidden_size: int
    architecture: str
    exploration_module: "ScheduledNormalParameters | None" = None
    recurrent_layers: int = 1

    @property
    def is_recurrent(self) -> bool:
        return self.recurrent is not None

    @torch.no_grad()
    def set_exploration_progress(self, progress: float) -> None:
        if self.exploration_module is not None:
            self.exploration_module.set_progress(progress)

    def exploration_std(self) -> torch.Tensor | None:
        if self.exploration_module is None:
            return None
        return self.exploration_module.current_std.detach()

    @torch.no_grad()
    def forward_step(
        self,
        observation: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        is_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """确定性控制单步；GRU 额外返回下一 hidden。"""

        if observation.ndim != 2:
            raise ValueError("observation must have shape [B, observation_dim]")
        batch = observation.shape[0]
        values: dict[str, torch.Tensor] = {"observation": observation}
        if self.is_recurrent:
            if recurrent_state is None:
                recurrent_state = torch.zeros(
                    batch,
                    self.recurrent_layers,
                    self.hidden_size,
                    dtype=observation.dtype,
                    device=observation.device,
                )
            if is_init is None:
                is_init = torch.zeros(
                    batch, 1, dtype=torch.bool, device=observation.device
                )
            values["recurrent_state"] = recurrent_state
            values["is_init"] = is_init
        td = TensorDict(values, batch_size=[batch], device=observation.device)
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            self.actor(td)
        next_state = (
            td[("next", "recurrent_state")] if self.is_recurrent else None
        )
        return td["action"], next_state


@dataclass(frozen=True)
class SACActorCritic:
    """SAC 模型：可选循环 actor 与供 SACLoss 复制的 Q 网络模板。"""

    actor: ProbabilisticActor
    qvalue: ValueOperator
    policy_module: TensorDictSequential
    architecture: str = "mlp"
    hidden_size: int = 0
    recurrent: GRUModule | None = None
    recurrent_layers: int = 1
    critic: None = None
    exploration_module: None = None

    @property
    def is_recurrent(self) -> bool:
        return self.recurrent is not None

    @torch.no_grad()
    def set_exploration_progress(self, progress: float) -> None:
        # SAC 的探索由可学习策略方差和自动温度 alpha 控制，不做时间退火。
        del progress

    def exploration_std(self) -> None:
        return None

    @torch.no_grad()
    def exploration_std_for(self, observation: torch.Tensor) -> torch.Tensor:
        """返回给定观测批次上的逐动作平均策略标准差。"""

        flat = observation.reshape(-1, observation.shape[-1])
        values: dict[str, torch.Tensor] = {"observation": flat}
        if self.is_recurrent:
            values["is_init"] = torch.ones(
                flat.shape[0], 1, device=flat.device, dtype=torch.bool
            )
        td = TensorDict(values, batch_size=[flat.shape[0]], device=flat.device)
        self.policy_module(td)
        return td["scale"].mean(dim=0)

    @torch.no_grad()
    def forward_step(
        self,
        observation: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        is_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if observation.ndim != 2:
            raise ValueError("observation must have shape [B, observation_dim]")
        batch = observation.shape[0]
        values: dict[str, torch.Tensor] = {"observation": observation}
        if self.is_recurrent:
            if recurrent_state is None:
                recurrent_state = torch.zeros(
                    batch,
                    self.recurrent_layers,
                    self.hidden_size,
                    dtype=observation.dtype,
                    device=observation.device,
                )
            if is_init is None:
                is_init = torch.zeros(
                    batch, 1, dtype=torch.bool, device=observation.device
                )
            values["recurrent_state"] = recurrent_state
            values["is_init"] = is_init
        td = TensorDict(values, batch_size=[batch], device=observation.device)
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            self.actor(td)
        next_state = (
            td[("next", "recurrent_state")] if self.is_recurrent else None
        )
        return td["action"], next_state


class BoundedNormalParameters(nn.Module):
    """生成 SAC actor 的均值和有硬上下界的标准差。"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        initial_std: tuple[float, ...],
        minimum_std: tuple[float, ...],
        maximum_std: tuple[float, ...],
        learnable_std: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        if not (
            len(initial_std)
            == len(minimum_std)
            == len(maximum_std)
            == action_dim
        ):
            raise ValueError("SAC std vectors must match action_dim")
        self.learnable_std = learnable_std
        self.network = MLP(
            in_features=observation_dim,
            out_features=(2 * action_dim if learnable_std else action_dim),
            num_cells=list(hidden_sizes),
            activation_class=nn.SiLU,
            device=device,
        )
        # 零初始化输出层：初始均值严格为零，方差不依赖随机观测。
        final_linear = [
            module
            for module in self.network.modules()
            if isinstance(module, nn.Linear)
        ][-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        minimum = torch.tensor(minimum_std, device=device)
        maximum = torch.tensor(maximum_std, device=device)
        initial = torch.tensor(initial_std, device=device)
        if learnable_std:
            raw_initial = torch.logit((initial - minimum) / (maximum - minimum))
            with torch.no_grad():
                final_linear.bias[action_dim:].copy_(raw_initial)
        self.register_buffer("initial_std", initial)
        self.register_buffer("minimum_std", minimum)
        self.register_buffer("maximum_std", maximum)

    @torch.no_grad()
    def reset_std_configuration(
        self,
        initial_std: torch.Tensor,
        minimum_std: torch.Tensor,
        maximum_std: torch.Tensor,
    ) -> None:
        """应用目标实验的方差边界，并把方差输出头重置到初始值。"""

        initial = initial_std.to(
            device=self.initial_std.device,
            dtype=self.initial_std.dtype,
        )
        minimum = minimum_std.to(
            device=self.minimum_std.device,
            dtype=self.minimum_std.dtype,
        )
        maximum = maximum_std.to(
            device=self.maximum_std.device,
            dtype=self.maximum_std.dtype,
        )
        expected_shape = self.initial_std.shape
        if (
            initial.shape != expected_shape
            or minimum.shape != expected_shape
            or maximum.shape != expected_shape
        ):
            raise ValueError("SAC std configuration shape is incompatible")
        if not bool(((minimum < initial) & (initial < maximum)).all().item()):
            raise ValueError(
                "each SAC action std must satisfy minimum < initial < maximum"
            )

        self.initial_std.copy_(initial)
        self.minimum_std.copy_(minimum)
        self.maximum_std.copy_(maximum)
        if not self.learnable_std:
            return

        final_linear = [
            module
            for module in self.network.modules()
            if isinstance(module, nn.Linear)
        ][-1]
        action_dim = self.initial_std.numel()
        raw_initial = torch.logit(
            (self.initial_std - self.minimum_std)
            / (self.maximum_std - self.minimum_std)
        )
        # 输出层前半部分是动作均值，必须保留 checkpoint 权重；只重置
        # 后半部分的 raw scale，避免旧任务的方差状态污染新任务。
        final_linear.weight[action_dim:].zero_()
        final_linear.bias[action_dim:].copy_(raw_initial)

    @torch.no_grad()
    def reset_mean_output(self) -> None:
        """Reset only the action-mean head while retaining hidden features."""

        final_linear = [
            module
            for module in self.network.modules()
            if isinstance(module, nn.Linear)
        ][-1]
        action_dim = self.initial_std.numel()
        final_linear.weight[:action_dim].zero_()
        final_linear.bias[:action_dim].zero_()

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(observation)
        if not self.learnable_std:
            return output, self.initial_std.expand_as(output)
        loc, raw_scale = output.chunk(2, dim=-1)
        scale = self.minimum_std + (
            self.maximum_std - self.minimum_std
        ) * raw_scale.sigmoid()
        return loc, scale


class ScheduledNormalParameters(nn.Module):
    """MLP 均值网络与按动作维度线性退火的固定标准差。"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        initial_std: tuple[float, ...],
        final_std: tuple[float, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if len(initial_std) != action_dim or len(final_std) != action_dim:
            raise ValueError("exploration std size must equal action_dim")
        self.loc_network = MLP(
            in_features=observation_dim,
            out_features=action_dim,
            num_cells=list(hidden_sizes),
            activation_class=nn.SiLU,
            device=device,
        ).to(dtype)
        # 残差策略初始均值接近 0，因此执行器命令从已辨识配平点附近开始。
        output_layers = [
            module
            for module in self.loc_network.modules()
            if isinstance(module, nn.Linear)
        ]
        if not output_layers:
            raise RuntimeError("MLP policy contains no linear output layer")
        nn.init.uniform_(output_layers[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(output_layers[-1].bias)
        self.register_buffer(
            "initial_std", torch.tensor(initial_std, device=device, dtype=dtype)
        )
        self.register_buffer(
            "final_std", torch.tensor(final_std, device=device, dtype=dtype)
        )
        self.register_buffer("schedule_progress", torch.zeros((), device=device, dtype=dtype))
        self.register_buffer("current_std", self.initial_std.clone())

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.loc_network(observation)
        return loc, self.current_std.expand_as(loc)

    @torch.no_grad()
    def set_progress(self, progress: float) -> None:
        clipped = min(max(float(progress), 0.0), 1.0)
        self.schedule_progress.fill_(clipped)
        self.current_std.copy_(
            self.initial_std
            + self.schedule_progress * (self.final_std - self.initial_std)
        )


def build_actor_critic(
    observation_dim: int,
    action_dim: int,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> ActorCritic:
    """按配置构建 TorchRL MLP 或 GRU actor-critic。"""

    if config.architecture == "mlp":
        return _build_mlp_actor_critic(
            observation_dim, action_dim, config, device, dtype
        )
    if config.architecture != "gru":
        raise ValueError(f"unsupported model architecture: {config.architecture}")
    return _build_gru_actor_critic(
        observation_dim, action_dim, config, device, dtype
    )


def build_sac_actor_critic(
    observation_dim: int,
    action_dim: int,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SACActorCritic:
    """构建 SAC 的有界随机 actor 和单个 Q 模板。

    TorchRL ``SACLoss`` 会把 Q 模板复制为两个独立 Q 网络并维护目标参数。
    GRU 只用于策略时序特征；双 Q 读取包含上一动作的完整真值观测和当前动作。
    replay 训练时由算法按连续序列执行截断 BPTT。
    """

    if config.architecture == "mlp":
        params = TensorDictModule(
            BoundedNormalParameters(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_sizes=config.encoder_sizes,
                initial_std=config.sac_initial_action_std,
                minimum_std=config.sac_minimum_action_std,
                maximum_std=config.sac_maximum_action_std,
                learnable_std=config.sac_learnable_action_std,
                device=device,
            ),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        policy_module = TensorDictSequential(params)
        recurrent = None
    elif config.architecture == "gru":
        encoder = TensorDictModule(
            MLP(
                in_features=observation_dim,
                out_features=config.encoder_sizes[-1],
                num_cells=list(config.encoder_sizes[:-1]),
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
            num_layers=config.recurrent_layers,
            batch_first=True,
            in_keys=["embedding", "recurrent_state", "is_init"],
            out_keys=["recurrent_features", ("next", "recurrent_state")],
            device=device,
        )
        params = TensorDictModule(
            BoundedNormalParameters(
                observation_dim=config.hidden_size,
                action_dim=action_dim,
                hidden_sizes=(config.actor_head_size,),
                initial_std=config.sac_initial_action_std,
                minimum_std=config.sac_minimum_action_std,
                maximum_std=config.sac_maximum_action_std,
                learnable_std=config.sac_learnable_action_std,
                device=device,
            ),
            in_keys=["recurrent_features"],
            out_keys=["loc", "scale"],
        )
        policy_module = TensorDictSequential(encoder, recurrent, params)
    else:
        raise ValueError(f"unsupported SAC model architecture: {config.architecture}")
    spec = Bounded(
        low=-1.0,
        high=1.0,
        shape=(action_dim,),
        device=device,
        dtype=dtype,
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
    qvalue = ValueOperator(
        MLP(
            in_features=observation_dim + action_dim,
            out_features=1,
            num_cells=list(config.encoder_sizes),
            activation_class=nn.SiLU,
            device=device,
        ),
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    ).to(device).to(dtype)
    return SACActorCritic(
        actor=actor,
        qvalue=qvalue,
        policy_module=policy_module,
        architecture=config.architecture,
        hidden_size=(config.hidden_size if recurrent is not None else 0),
        recurrent=recurrent,
        recurrent_layers=config.recurrent_layers,
    )


def _build_gru_actor_critic(
    observation_dim: int,
    action_dim: int,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> ActorCritic:
    """构建循环 actor-critic，并一次性放到目标设备和精度。

    TensorDict 键流为：
    ``observation -> embedding -> recurrent_features -> loc/scale -> action``。
    GRU 同时读取 ``recurrent_state`` 与 ``is_init``，并把新隐状态写到
    ``("next", "recurrent_state")``，从而和环境的 episode 边界解耦。
    """

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
        num_layers=config.recurrent_layers,
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
    # 策略统一输出标准动作域，具体执行器量纲由环境适配器负责转换。
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
    # critic 复用本次策略前向产生的 recurrent_features，不重复执行 GRU。
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
        architecture="gru",
        recurrent_layers=config.recurrent_layers,
    )


def _build_mlp_actor_critic(
    observation_dim: int,
    action_dim: int,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> ActorCritic:
    """构建文档基线 MLP：展平观测直接映射到动作分布与价值。"""

    exploration = ScheduledNormalParameters(
        observation_dim,
        action_dim,
        config.encoder_sizes,
        config.initial_action_std or tuple(0.2 for _ in range(action_dim)),
        config.final_action_std or tuple(0.05 for _ in range(action_dim)),
        device,
        dtype,
    )
    params = TensorDictModule(
        exploration,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    policy_module = TensorDictSequential(params)
    spec = Bounded(
        low=-1.0,
        high=1.0,
        shape=(action_dim,),
        device=device,
        dtype=dtype,
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
            in_features=observation_dim,
            out_features=1,
            num_cells=list(config.encoder_sizes),
            activation_class=nn.SiLU,
            device=device,
        ),
        in_keys=["observation"],
        out_keys=["state_value"],
    ).to(device).to(dtype)
    return ActorCritic(
        actor=actor,
        critic=critic,
        recurrent=None,
        policy_module=policy_module,
        hidden_size=0,
        architecture="mlp",
        exploration_module=exploration,
    )
