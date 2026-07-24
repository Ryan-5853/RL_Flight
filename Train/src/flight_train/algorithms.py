from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Callable
from contextlib import nullcontext

import torch
from tensordict import TensorDictBase
from torchrl.data import LazyTensorStorage, MultiStep, TensorDictReplayBuffer
from torchrl.modules import set_recurrent_mode
from torchrl.objectives import ClipPPOLoss, SACLoss, SoftUpdate
from torchrl.objectives.utils import ValueEstimators
from torchrl.objectives.value import GAE

from .config import PPOConfig, SACConfig
from .core import tensordict_to_device
from .models import ActorCritic, SACActorCritic


class TorchRLPPO:
    """基于 TorchRL 的 PPO 更新器，同时调度 MLP 与 recurrent policy。

    输入 rollout 必须完整驻留在指定设备，形状为 ``[B,T]``。GAE、损失计算、
    序列重排、反向传播和优化器更新均直接处理 TensorDict/GPU 张量。
    """

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
        self.actor_parameters = list(model.actor.parameters())
        self.critic_parameters = list(model.critic.parameters())
        actor_ids = {id(parameter) for parameter in self.actor_parameters}
        if any(id(parameter) in actor_ids for parameter in self.critic_parameters):
            raise ValueError("actor and critic must not share trainable parameters")
        self.actor_optimizer = torch.optim.Adam(
            self.actor_parameters, lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters,
            lr=(
                config.critic_learning_rate
                if config.critic_learning_rate is not None
                else config.learning_rate
            ),
        )

    def update(
        self,
        rollout: TensorDictBase,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Mapping[str, torch.Tensor]:
        """对一批完整 rollout 计算 GAE，并执行多轮 PPO 更新。

        返回值仍是设备上的标量张量；只有记录器会在低频边界调用 ``item``。
        """

        if rollout.device != self.device:
            raise ValueError(f"rollout must stay on {self.device}, got {rollout.device}")
        with torch.no_grad():
            self.advantage(rollout)
            # ClipPPOLoss 会把 shifted_valid 原生应用到每一个损失项，屏蔽仿真无效转移。
            rollout["shifted_valid"] = rollout[("next", "valid")]

        batch, time = rollout.batch_size
        length = self.config.sequence_length
        chunks = time // length
        # minibatch 的基本单位是完整时间块，不能打散单个时间步，否则会破坏 GRU
        # 的状态传播以及 is_init 对 episode 边界的复位语义。
        sequences = rollout.reshape(batch, chunks, length).transpose(0, 1).reshape(batch * chunks, length)
        sequence_count = sequences.batch_size[0]
        aggregate: dict[str, torch.Tensor] = {}
        updates = 0
        updates_per_epoch = (
            sequence_count + self.config.sequences_per_minibatch - 1
        ) // self.config.sequences_per_minibatch
        total_updates = self.config.epochs * updates_per_epoch
        recurrent_context = (
            set_recurrent_mode(True) if self.model.is_recurrent else nullcontext()
        )
        # GRU 按完整序列更新；MLP 使用 length=1 的同一张量化 minibatch 路径。
        with recurrent_context:
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
                    if not bool(torch.isfinite(objective).all()):
                        raise FloatingPointError("PPO objective is non-finite")
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.actor_parameters, self.config.max_grad_norm
                    )
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.critic_parameters,
                        (
                            self.config.critic_max_grad_norm
                            if self.config.critic_max_grad_norm is not None
                            else self.config.max_grad_norm
                        ),
                    )
                    if not bool(
                        torch.isfinite(torch.as_tensor(actor_grad_norm)).all()
                        and torch.isfinite(torch.as_tensor(critic_grad_norm)).all()
                    ):
                        raise FloatingPointError("PPO gradient norm is non-finite")
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    for key, value in loss_td.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            aggregate[key] = aggregate.get(key, torch.zeros_like(value)) + value.detach()
                    aggregate["actor_grad_norm"] = aggregate.get(
                        "actor_grad_norm", torch.zeros((), device=self.device)
                    ) + torch.as_tensor(actor_grad_norm, device=self.device)
                    aggregate["critic_grad_norm"] = aggregate.get(
                        "critic_grad_norm", torch.zeros((), device=self.device)
                    ) + torch.as_tensor(critic_grad_norm, device=self.device)
                    updates += 1
                    if progress_callback is not None:
                        progress_callback(updates, total_updates)
        return {key: value / updates for key, value in aggregate.items()}


# 保留旧公开名称，已有 GRU 配置和外部调用无需迁移。
RecurrentPPO = TorchRLPPO


class TorchRLSAC:
    """TorchRL MLP-SAC：GPU replay、双 Q、自动温度和软目标网络更新。"""

    def __init__(
        self,
        model: SACActorCritic,
        config: SACConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.loss = SACLoss(
            actor_network=model.actor,
            qvalue_network=model.qvalue,
            num_qvalue_nets=2,
            loss_function="smooth_l1",
            alpha_init=config.initial_alpha,
            min_alpha=config.min_alpha,
            max_alpha=config.max_alpha,
            action_spec=model.actor.spec,
            fixed_alpha=False,
            target_entropy=config.target_entropy,
            delay_qvalue=True,
            reduction="mean",
        )
        self.loss.make_value_estimator(ValueEstimators.TD0, gamma=config.gamma)
        self.multi_step = MultiStep(
            gamma=config.gamma,
            n_steps=config.n_step_return,
        )
        self.target_updater = SoftUpdate(self.loss, tau=config.target_tau)
        self.actor_parameters = list(
            self.loss.actor_network_params.values(True, True)
        )
        self.critic_parameters = list(
            self.loss.qvalue_network_params.values(True, True)
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor_parameters, lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters, lr=config.critic_learning_rate
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.loss.log_alpha], lr=config.alpha_learning_rate
        )
        self.replay = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                config.replay_capacity,
                device=device,
            ),
            batch_size=config.replay_batch_size,
        )
        self.gradient_updates = 0
        self.actor_updates = 0
        # 多步回报必须跨 collector batch 延续，不能把每个 rollout 的尾部当作
        # episode 截断。这里按并行环境保存尚缺未来转移的 n-1 步原始上下文。
        self.n_step_pending: TensorDictBase | None = None

    @property
    def replay_size(self) -> int:
        return len(self.replay)

    def add(self, rollout: TensorDictBase) -> int:
        """生成无偏 n-step 转移，并只把 SAC 必需字段写入 replay。

        collector 的时间维边界不是 episode 边界。当前 rollout 会先与上次保留的
        ``n-1`` 步拼接；只有已经拥有完整未来窗口（或窗口内真实结束）的起点才会
        写入 replay。这样增大 ``n_step_return`` 不依赖
        ``control_steps_per_rollout``，也不会在每次采集边界系统性缩短回报。
        """

        if rollout.ndim != 2:
            raise ValueError(
                f"SAC rollout must have [B,T] batch dimensions, got {rollout.batch_size}"
            )
        raw = rollout.select(
            "observation",
            "action",
            ("next", "observation"),
            ("next", "reward"),
            ("next", "done"),
            ("next", "terminated"),
            ("next", "truncated"),
            ("next", "valid"),
            strict=True,
        ).clone(False)
        # 数值无效的仿真转移必须终止多步累计，并禁止从无效 next observation
        # bootstrap。正常的 time-limit truncated 仍保留 terminated=False。
        invalid = ~raw[("next", "valid")]
        raw[("next", "done")] = raw[("next", "done")] | invalid
        raw[("next", "terminated")] = (
            raw[("next", "terminated")] | invalid
        )

        if self.n_step_pending is not None:
            if self.n_step_pending.batch_size[0] != raw.batch_size[0]:
                raise ValueError(
                    "SAC n-step pending batch does not match rollout parallel count"
                )
            raw = torch.cat((self.n_step_pending, raw), dim=1)

        context = self.config.n_step_return - 1
        if context:
            # clone 张量而非只复制 TensorDict 容器，避免下面 MultiStep 的键替换
            # 改写 exact-resume 所需的原始尾部。
            self.n_step_pending = raw[:, -context:].clone()
        else:
            self.n_step_pending = None

        complete_count = raw.batch_size[1] - context
        if complete_count <= 0:
            return 0

        # TorchRL 负责折扣奖励求和、移动 next observation，并写入
        # steps_to_next_obs。TD0Estimator 随后会自动使用 gamma ** steps。
        transitions = self.multi_step(raw)
        steps = transitions["steps_to_next_obs"]

        # TorchRL MultiStep 有意保留原始一步 done keys；SAC 的 TD target 则需要
        # n-step 终点的 terminated 标志。按 steps_to_next_obs 把所有终点状态键
        # 从原始序列显式 gather 到相同的 next observation 时刻。
        time = raw.batch_size[1]
        endpoint = (
            torch.arange(time, device=self.device)[None, :]
            + steps.to(torch.long)
            - 1
        ).clamp_max(time - 1)
        for key in ("done", "terminated", "truncated", "valid"):
            value = raw[("next", key)]
            gather_index = endpoint
            while gather_index.ndim < value.ndim:
                gather_index = gather_index.unsqueeze(-1)
            gather_index = gather_index.expand_as(value)
            transitions[("next", key)] = torch.gather(
                value, dim=1, index=gather_index
            )

        transitions = transitions[:, :complete_count].select(
            "observation",
            "action",
            "steps_to_next_obs",
            ("next", "observation"),
            ("next", "reward"),
            ("next", "done"),
            ("next", "terminated"),
            ("next", "truncated"),
            ("next", "valid"),
            strict=True,
        ).reshape(-1)
        valid = transitions[("next", "valid")].squeeze(-1)
        transitions = transitions[valid]
        if transitions.batch_size[0]:
            self.replay.extend(transitions)
        return transitions.batch_size[0]

    def update(
        self,
        rollout: TensorDictBase,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Mapping[str, torch.Tensor]:
        """写入新经验；warm-up 后按配置进行若干次 off-policy 更新。"""

        added_transitions = self.add(rollout)
        zero = torch.zeros((), device=self.device)
        if self.replay_size < self.config.warmup_transitions:
            return {
                "loss_actor": zero,
                "loss_qvalue": zero,
                "loss_alpha": zero,
                "alpha": self.loss._alpha.detach(),
                "entropy": zero,
                "actor_grad_norm": zero,
                "critic_grad_norm": zero,
                "replay_size": torch.tensor(
                    self.replay_size, device=self.device, dtype=torch.float32
                ),
                "sac_updates": zero,
                "sac_warmup": torch.ones((), device=self.device),
                "sac_critic_pretraining": zero,
                "sac_critic_updates_total": torch.tensor(
                    self.gradient_updates, device=self.device, dtype=torch.float32
                ),
                "sac_actor_updates_total": torch.tensor(
                    self.actor_updates, device=self.device, dtype=torch.float32
                ),
                "sac_replay_added": torch.tensor(
                    added_transitions,
                    device=self.device,
                    dtype=torch.float32,
                ),
                "sac_n_step_return": torch.tensor(
                    self.config.n_step_return,
                    device=self.device,
                    dtype=torch.float32,
                ),
            }

        aggregate: dict[str, torch.Tensor] = {}
        updates = self.config.updates_per_collection
        for update_index in range(updates):
            batch = self.replay.sample().to(self.device)
            # actor 只能读取已经完成当前 critic 更新后的 Q；warm-up 数据达到
            # 阈值并不代表随机初始化的 critic 已经学会了动力学。
            actor_enabled = (
                self.gradient_updates
                >= self.config.critic_pretraining_updates
            )
            critic_loss_td = self.loss(batch)
            critic_objective = critic_loss_td["loss_qvalue"]
            if not bool(torch.isfinite(critic_objective).all()):
                raise FloatingPointError("SAC critic objective is non-finite")
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.alpha_optimizer.zero_grad(set_to_none=True)
            critic_objective.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_parameters, self.config.critic_max_grad_norm
            )
            if not bool(torch.isfinite(torch.as_tensor(critic_grad_norm)).all()):
                raise FloatingPointError("SAC critic gradient norm is non-finite")
            self.critic_optimizer.step()
            self.gradient_updates += 1
            if self.gradient_updates % self.config.target_update_interval == 0:
                self.target_updater.step()

            actor_grad_norm = zero.clone()
            actor_loss = zero.clone()
            alpha_loss = zero.clone()
            entropy = critic_loss_td["entropy"].detach()
            if actor_enabled:
                # critic 已先完成本轮更新；重新前向，避免 actor 沿旧 Q 梯度移动。
                actor_loss_td = self.loss(batch)
                actor_objective = (
                    actor_loss_td["loss_actor"]
                    + actor_loss_td["loss_alpha"]
                )
                if not bool(torch.isfinite(actor_objective).all()):
                    raise FloatingPointError("SAC actor objective is non-finite")
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.alpha_optimizer.zero_grad(set_to_none=True)
                actor_objective.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_parameters, self.config.actor_max_grad_norm
                )
                if not bool(
                    torch.isfinite(torch.as_tensor(actor_grad_norm)).all()
                ):
                    raise FloatingPointError(
                        "SAC actor gradient norm is non-finite"
                    )
                self.actor_optimizer.step()
                self.alpha_optimizer.step()
                self.actor_updates += 1
                actor_loss = actor_loss_td["loss_actor"].detach()
                alpha_loss = actor_loss_td["loss_alpha"].detach()
                entropy = actor_loss_td["entropy"].detach()

            scalar_metrics = {
                "loss_actor": actor_loss,
                "loss_qvalue": critic_objective.detach(),
                "loss_alpha": alpha_loss,
                "alpha": self.loss._alpha.detach(),
                "entropy": entropy,
            }
            for key, value in scalar_metrics.items():
                aggregate[key] = (
                    aggregate.get(key, torch.zeros_like(value)) + value
                )
            aggregate["actor_grad_norm"] = aggregate.get(
                "actor_grad_norm", zero.clone()
            ) + torch.as_tensor(actor_grad_norm, device=self.device)
            aggregate["critic_grad_norm"] = aggregate.get(
                "critic_grad_norm", zero.clone()
            ) + torch.as_tensor(critic_grad_norm, device=self.device)
            if progress_callback is not None:
                progress_callback(update_index + 1, updates)
        metrics = {key: value / updates for key, value in aggregate.items()}
        metrics.update(
            {
                "replay_size": torch.tensor(
                    self.replay_size, device=self.device, dtype=torch.float32
                ),
                "sac_updates": torch.tensor(
                    updates, device=self.device, dtype=torch.float32
                ),
                "sac_warmup": zero,
                "sac_critic_pretraining": torch.tensor(
                    float(
                        self.config.critic_pretraining_updates > 0
                        and self.actor_updates == 0
                    ),
                    device=self.device,
                ),
                "sac_critic_updates_total": torch.tensor(
                    self.gradient_updates, device=self.device, dtype=torch.float32
                ),
                "sac_actor_updates_total": torch.tensor(
                    self.actor_updates, device=self.device, dtype=torch.float32
                ),
                "sac_replay_added": torch.tensor(
                    added_transitions,
                    device=self.device,
                    dtype=torch.float32,
                ),
                "sac_n_step_return": torch.tensor(
                    self.config.n_step_return,
                    device=self.device,
                    dtype=torch.float32,
                ),
            }
        )
        return metrics

    def state_dict(self) -> Mapping[str, object]:
        """返回 exact-resume 所需的完整 SAC 状态，包括 replay 内容。"""

        return {
            "loss": self.loss.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "gradient_updates": self.gradient_updates,
            "actor_updates": self.actor_updates,
            "n_step_pending": self.n_step_pending,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.loss.load_state_dict(state["loss"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        # TorchRL 0.13 的 LazyTensorStorage state_dict 会把 TensorDict 嵌套键
        # 展平成带点字符串，并在 load 后丢失 batch_size/device 元数据。显式重建
        # 容量维和 next 子 TensorDict，才能在 CUDA 恢复后继续 extend/sample。
        self.replay.load_state_dict(state["replay"])
        storage = self.replay._storage
        replay_tensor = storage._storage.unflatten_keys(".")
        replay_tensor.batch_size = torch.Size([self.config.replay_capacity])
        storage._storage = replay_tensor.to(self.device)
        storage.device = self.device
        self.gradient_updates = int(state["gradient_updates"])
        self.actor_updates = int(state.get("actor_updates", 0))
        pending = state.get("n_step_pending")
        if pending is not None and not isinstance(pending, TensorDictBase):
            raise ValueError("checkpoint SAC n_step_pending must be a TensorDict")
        self.n_step_pending = (
            None
            if pending is None
            else tensordict_to_device(pending, self.device)
        )
