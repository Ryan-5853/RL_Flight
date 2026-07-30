from __future__ import annotations

import torch
from collections.abc import Callable
from tensordict import TensorDict, TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from .core import BatchedControlEnv
from .models import ActorCritic, SACActorCritic


class TensorDictRolloutCollector:
    """同步采集 GPU rollout，并生成 TorchRL 标准转移键。

    采集器持有跨 rollout 的当前观测；GRU 策略额外持有隐状态，MLP 不创建
    占位 hidden。输出 batch 维为 ``[B,T]``，根节点保存当前时刻策略输入/输出，
    ``next`` 节点保存环境转移结果。
    """

    def __init__(
        self,
        env: BatchedControlEnv,
        model: ActorCritic | SACActorCritic,
        rollout_steps: int,
        *,
        deterministic: bool = False,
    ) -> None:
        self.env = env
        self.model = model
        self.rollout_steps = rollout_steps
        self.device = env.spec.device
        self.batch_size = env.spec.parallel_count
        self.deterministic = deterministic
        self.current = env.reset()
        if model.is_recurrent:
            # GRUModule 的单步隐状态约定为 [B, num_layers, hidden_size]。
            self.current.set(
                "recurrent_state",
                torch.zeros(
                    (
                        self.batch_size,
                        model.recurrent_layers,
                        model.hidden_size,
                    ),
                    device=self.device,
                    dtype=env.spec.dtype,
                ),
            )

    @torch.no_grad()
    def collect(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> TensorDictBase:
        """采集固定长度序列，全程禁用 autograd 并保持数据在环境设备上。"""

        steps: list[TensorDictBase] = []
        current = self.current
        for step_index in range(self.rollout_steps):
            policy_keys = ["observation"]
            if self.model.is_recurrent:
                policy_keys.extend(("recurrent_state", "is_init"))
            policy_td = current.select(*policy_keys, strict=True).clone(False)
            # actor 写入 action/log_prob/next recurrent_state；critic 复用循环特征。
            interaction = (
                ExplorationType.DETERMINISTIC
                if self.deterministic
                else ExplorationType.RANDOM
            )
            with set_exploration_type(interaction):
                self.model.actor(policy_td)
            if self.model.critic is not None:
                self.model.critic(policy_td)
            next_env = self.env.step(policy_td["action"])
            next_values = {
                "observation": next_env["observation"],
                "reward": next_env["reward"],
                "terminated": next_env["terminated"],
                "truncated": next_env["truncated"],
                "done": next_env["done"],
                "valid": next_env["valid"],
                "is_init": next_env["is_init"],
            }
            # 只把训练监控需要的低维诊断带入 rollout；完整 command/pilot/info
            # 仍由环境/SimEnv 日志保存，避免显著增加 PPO 批次显存。
            if "info" in next_env.keys():
                info = next_env["info"]
                next_values.update(
                    {
                        "attitude_error_rad": info["attitude_error_rad"],
                        "angular_rate_norm_rad_s": info["angular_rate_norm"],
                        "height_error_m": info["pilot.height_error_m"],
                        "episode_length_steps": info["episode.length_steps"],
                        "curriculum_quality_success": info.get(
                            "episode.curriculum_quality_success",
                            torch.ones_like(next_env["truncated"]),
                        ),
                    }
                )
                for reward_key in (
                    "reward.alive",
                    "reward.attitude",
                    "reward.tilt",
                    "reward.yaw_rate",
                    "reward.angular_rate",
                    "reward.action_rate",
                    "reward.saturation",
                    "reward.risk",
                    "reward.survival",
                    "reward.termination",
                ):
                    if reward_key in info.keys():
                        next_values[reward_key] = info[reward_key]
            transition_values = {
                "observation": current["observation"],
                "is_init": current["is_init"],
                "action": policy_td["action"],
                "action_log_prob": policy_td["action_log_prob"],
                "policy_scale": policy_td["scale"],
                "next": TensorDict(
                    next_values,
                    batch_size=[self.batch_size],
                    device=self.device,
                ),
            }
            if "state_value" in policy_td.keys():
                transition_values["state_value"] = policy_td["state_value"]
            if self.model.is_recurrent:
                transition_values["recurrent_state"] = current["recurrent_state"]
            transition = TensorDict(
                transition_values,
                batch_size=[self.batch_size],
                device=self.device,
            )
            steps.append(transition)
            current_values = {
                "observation": next_env["observation"],
                "is_init": next_env["is_init"],
            }
            if self.model.is_recurrent:
                current_values["recurrent_state"] = policy_td[
                    ("next", "recurrent_state")
                ]
            current = TensorDict(
                current_values,
                batch_size=[self.batch_size],
                device=self.device,
            )
            if progress_callback is not None:
                progress_callback(step_index + 1, self.rollout_steps)

        # TorchRL 循环模块消费 [B,T,...]，时间维是 TensorDict batch_size 的末维。
        rollout = torch.stack(steps, dim=1)
        if self.model.critic is not None:
            # PPO 的最后一个 next observation 不在根节点序列中，需要额外估值
            # 作为 GAE bootstrap。SAC 的 TD0 target 在 replay 采样时由目标 Q 计算。
            bootstrap_td = current.clone(False)
            self.model.policy_module(bootstrap_td)
            self.model.critic(bootstrap_td)
            values = rollout["state_value"]
            next_values = torch.cat(
                (values[:, 1:], bootstrap_td["state_value"][:, None]), dim=1
            )
            rollout[("next", "state_value")] = next_values
        # 仿真失败的 transition 由 shifted_valid 排除；先把奖励清零避免污染统计量。
        rollout[("next", "reward")] = torch.where(
            rollout[("next", "valid")], rollout[("next", "reward")], torch.zeros_like(rollout[("next", "reward")])
        )
        self.current = current
        return rollout
