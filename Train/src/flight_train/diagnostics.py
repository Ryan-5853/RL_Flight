from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from tensordict import TensorDict, TensorDictBase

from .algorithms import TorchRLSAC
from .config import (
    ExperimentConfig,
    exact_resume_config_sha256,
    load_experiment_config,
    override_run_config,
)
from .envs import SimEnvAdapter, _load_timing
from .models import SACActorCritic, build_sac_actor_critic
from .randomization import StaticRandomizer
from .recording import load_checkpoint
from .registry import ComponentRegistry


@dataclass
class SACDiagnosticContext:
    """已恢复 checkpoint 策略、双 Q 和训练环境的只读诊断上下文。"""

    config: ExperimentConfig
    checkpoint: Mapping[str, Any]
    env: SimEnvAdapter
    model: SACActorCritic
    algorithm: TorchRLSAC
    static_randomizer: StaticRandomizer
    reward_calculator: Any
    current_observation: torch.Tensor

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "SACDiagnosticContext":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass
class _EnvironmentSnapshot:
    simulator: Mapping[str, Any]
    previous_action: torch.Tensor
    observation_history: torch.Tensor
    observation_history_index: int
    episode_id: torch.Tensor
    episode_step: torch.Tensor
    episode_roll_pitch_squared_sum: torch.Tensor
    episode_yaw_rate_squared_sum: torch.Tensor
    episode_angular_rate_squared_sum: torch.Tensor
    episode_quality_steps: torch.Tensor
    current_static_parameters: TensorDictBase
    command_source: Mapping[str, Any]
    static_randomizer: Mapping[str, Any]
    curriculum_stage: int
    curriculum_successes: int
    curriculum_failures: int
    curriculum_consecutive_passes: int
    curriculum_last_success_fraction: float


def load_checkpoint_experiment_config(
    reference_config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
) -> ExperimentConfig:
    """用 checkpoint 归档的原始配置重建实验，引用路径取自当前工程。

    运行目录中的 ``config.json`` 保留了原始相对路径，但它离开
    ``configs/experiments`` 后不能直接解析。这里仅把环境、评测和输出路径替换
    为 reference config 已解析的绝对路径；所有数值训练字段仍严格来自 checkpoint。
    在设备覆盖前必须通过 exact-resume 指纹校验。
    """

    reference = load_experiment_config(reference_config_path)
    state = load_checkpoint(Path(checkpoint_path).expanduser().resolve())
    archived = copy.deepcopy(dict(state.get("config", {})))
    if not archived:
        raise ValueError("checkpoint archived configuration is missing")
    environment = archived.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("checkpoint environment configuration is invalid")
    environment["config_path"] = str(reference.simulator_config)
    run = archived.get("run")
    if not isinstance(run, dict):
        raise ValueError("checkpoint run configuration is invalid")
    run["output_root"] = str(reference.run.output_root)
    evaluation = archived.get("evaluation")
    if (
        isinstance(evaluation, dict)
        and reference.evaluation.suite_path is not None
    ):
        evaluation["suite_path"] = str(reference.evaluation.suite_path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="rl-flight-checkpoint-config-",
        dir="/tmp",
        encoding="utf-8",
    ) as temporary:
        json.dump(archived, temporary, ensure_ascii=False)
        temporary.flush()
        config = load_experiment_config(temporary.name)
    expected = state.get("exact_resume_config_sha256")
    actual = exact_resume_config_sha256(state["config"])
    if actual != expected:
        raise ValueError(
            "reconstructed checkpoint configuration fingerprint mismatch"
        )
    if device is not None:
        config = override_run_config(config, device=device)
    return config


def load_sac_diagnostic_context(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
) -> SACDiagnosticContext:
    """恢复环境、actor 和双 Q，但不加载 4M replay 到 TorchRL buffer。"""

    if config.algorithm_name != "sac" or config.sac is None:
        raise ValueError("SAC diagnostics require an SAC experiment configuration")
    device = torch.device(config.run.device)
    state = load_checkpoint(Path(checkpoint_path).expanduser().resolve())
    if state.get("algorithm_name") != "sac":
        raise ValueError("checkpoint is not an SAC checkpoint")

    reward_calculator = ComponentRegistry().build_reward(
        {
            "type": config.reward.calculator.type,
            "version": config.reward.calculator.version,
            "params": config.reward.calculator.params,
        }
    )
    static_randomizer = StaticRandomizer(
        config.static_parameter_specs(),
        config.static_randomization.seed,
        device,
        config.torch_dtype,
    )
    env = SimEnvAdapter.create(
        config.simulator_config,
        config.run.parallel_count,
        device,
        config.torch_dtype,
        config.task,
        reward_calculator=reward_calculator,
        static_randomizer=static_randomizer,
        dynamic_randomization=dict(config.dynamic_randomization.parameters),
        dynamic_seed=config.dynamic_randomization.seed,
        reward_context_fields=tuple(
            dict(field) for field in config.reward.context_fields
        ),
        command_source_config=config.command_source,
        control_contract_config=config.control_contract,
    )
    try:
        _restore_environment_only(
            env,
            state,
            config,
            static_randomizer,
            reward_calculator,
            device,
        )
        model = build_sac_actor_critic(
            env.spec.observation_dim,
            env.spec.action_dim,
            config.model,
            device,
            config.torch_dtype,
        )
        algorithm = TorchRLSAC(model, config.sac, device)
        model.actor.load_state_dict(state["actor"], strict=True)
        algorithm.loss.load_state_dict(state["sac"]["loss"], strict=True)
        model.actor.eval()

        stored_current = state.get("collector_current")
        if not isinstance(stored_current, TensorDictBase):
            raise ValueError("checkpoint collector_current is missing")
        current_observation = stored_current["observation"].to(
            device=device, dtype=config.torch_dtype
        )
        observed, *_ = env._observation()
        observation_error = (observed - current_observation).abs().amax()
        if float(observation_error.item()) > 1e-6:
            raise ValueError(
                "restored environment observation does not match collector_current: "
                f"max_abs_error={float(observation_error.item()):.9g}"
            )
        return SACDiagnosticContext(
            config=config,
            checkpoint=state,
            env=env,
            model=model,
            algorithm=algorithm,
            static_randomizer=static_randomizer,
            reward_calculator=reward_calculator,
            current_observation=current_observation,
        )
    except BaseException:
        env.close()
        raise


def replay_action_gradient_report(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    sample_count: int = 100_000,
    seed: int = 72001,
    minimum_yaw_rate_rad_s: float = 0.5,
) -> dict[str, Any]:
    """比较 replay 的真实局部动作导数和 learned-Q 动作梯度。

    固定方差 SAC 的 ``stored_action - deterministic_action`` 是与状态近似独立
    的随机激励。用它回归下一步角加速度，可以避免直接拿相关的策略均值做因果
    推断。报告只选择正偏航且 yaw 指令近零的有效 transition。
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    state = load_checkpoint(Path(checkpoint_path).expanduser().resolve())
    replay = state["sac"]["replay"]["_storage"]
    length = int(replay["_len"])
    storage = replay["_storage"]
    count = min(sample_count, length)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    index = torch.randint(length, (count,), generator=generator)
    observation = storage["observation"][index]
    action = storage["action"][index]
    next_observation = storage["next.observation"][index]
    done = storage["next.done"][index, 0]

    # uniform 模式的当前帧在末尾，multirate 模式的当前完整帧在开头。
    current_frame_offset = (
        config.control_contract.current_observation_offset
    )
    yaw_rate_index = current_frame_offset + 6
    yaw_command_index = current_frame_offset + 15
    angular_scale = config.task.terminate_angular_rate_rad_s
    _, control_hz = _load_timing(config.simulator_config)
    yaw_rate = observation[:, yaw_rate_index] * angular_scale
    selected = (
        ~done
        & (observation[:, yaw_command_index].abs() < 0.05)
        & (yaw_rate > minimum_yaw_rate_rad_s)
    )
    observation = observation[selected]
    action = action[selected]
    next_observation = next_observation[selected]
    yaw_rate = yaw_rate[selected]
    minimum_selected = min(128, max(16, count // 10))
    if observation.shape[0] < minimum_selected:
        raise RuntimeError("too few positive-yaw replay transitions for diagnosis")

    device = torch.device(config.run.device)
    model = build_sac_actor_critic(
        observation.shape[-1], 4, config.model, device, config.torch_dtype
    )
    algorithm = TorchRLSAC(model, config.sac, device)
    model.actor.load_state_dict(state["actor"], strict=True)
    algorithm.loss.load_state_dict(state["sac"]["loss"], strict=True)
    observation_device = observation.to(device)
    action_device = action.to(device)
    with torch.no_grad():
        deterministic_action, _ = model.forward_step(observation_device)
    exploration_residual = action_device - deterministic_action
    yaw_acceleration = (
        (
            next_observation[:, yaw_rate_index].to(device)
            - observation_device[:, yaw_rate_index]
        )
        * angular_scale
        * control_hz
    )
    centered_action = exploration_residual - exploration_residual.mean(0)
    centered_acceleration = yaw_acceleration - yaw_acceleration.mean()
    dynamics_slope = torch.linalg.lstsq(
        centered_action,
        centered_acceleration[:, None],
    ).solution[:, 0]

    gradient_count = min(4096, observation_device.shape[0])
    gradient_observation = observation_device[:gradient_count]
    gradient_action = deterministic_action[:gradient_count].detach().requires_grad_(True)
    q_values = _q_values(
        algorithm, gradient_observation, gradient_action
    )
    q_gradients = []
    for q_index in range(2):
        gradient = torch.autograd.grad(
            q_values[q_index].mean(),
            gradient_action,
            retain_graph=True,
        )[0]
        # mean(Q) 的逐样本梯度含 1/B；sum 恢复平均局部导数。
        q_gradients.append(gradient.sum(0))
    minimum_q = q_values.min(0).values
    minimum_gradient = torch.autograd.grad(
        minimum_q.mean(), gradient_action
    )[0].sum(0)

    return {
        "checkpoint_global_control_steps": int(state["global_control_steps"]),
        "sampled_transitions": count,
        "selected_positive_yaw_transitions": int(observation.shape[0]),
        "yaw_rate_rad_s": {
            "mean": float(yaw_rate.mean().item()),
            "p95": float(torch.quantile(yaw_rate, 0.95).item()),
        },
        "deterministic_action_mean": _vector(deterministic_action.mean(0)),
        "exploration_residual_std": _vector(exploration_residual.std(0)),
        "empirical_yaw_acceleration_per_action": _vector(dynamics_slope),
        "q_mean": _vector(q_values.mean(1)),
        "q_action_gradient": [_vector(value) for value in q_gradients],
        "minimum_q_action_gradient": _vector(minimum_gradient),
        "interpretation": {
            "positive_yaw_correction_gradient_expected": (
                "-sign(empirical_yaw_acceleration_per_action)"
            ),
            "action_fields": list(config.control_contract.policy_action_fields),
        },
    }


def finite_horizon_action_sweep(
    context: SACDiagnosticContext,
    *,
    action_axis: int = 0,
    deltas: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10),
    horizon_steps: int = 250,
    minimum_yaw_rate_rad_s: float = 0.5,
) -> dict[str, Any]:
    """在完全相同的 checkpoint 环境状态上比较真实 return 和双 Q。

    候选动作只用于第一控制步，之后统一使用 checkpoint 的确定性 actor，因此
    真实有限时域回报与 ``Q(s,a)`` 的定义一致。每个候选动作前恢复完整仿真、
    VirtualPilot、随机计数器和训练环境状态。
    """

    if not 0 <= action_axis < context.env.spec.action_dim:
        raise ValueError("action_axis is outside the policy action dimension")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if not deltas:
        raise ValueError("at least one action delta is required")

    env = context.env
    device = env.spec.device
    observation = context.current_observation
    with torch.no_grad():
        base_action, _ = context.model.forward_step(observation)
    truth = env.simulator.observe(
        "truth", ("angular_velocity_b",)
    ).values["angular_velocity_b"]
    selected = (
        (truth[:, 2] > minimum_yaw_rate_rad_s)
        & (
            observation[
                :,
                context.config.control_contract.current_observation_offset
                + 15,
            ].abs()
            < 0.05
        )
    )
    if not bool(selected.any().item()):
        raise RuntimeError("checkpoint state contains no selected positive-yaw instances")

    snapshot = _snapshot_environment(context)
    candidates = []
    for delta in deltas:
        candidate = base_action.clone()
        candidate[:, action_axis] = (
            candidate[:, action_axis] + float(delta)
        ).clamp(-1.0, 1.0)
        with torch.no_grad():
            q_value = _q_values(context.algorithm, observation, candidate)
        _restore_snapshot(context, snapshot)
        alive = selected.clone()
        discounted_return = torch.zeros(env.spec.parallel_count, device=device)
        discount = 1.0
        current_observation = observation
        terminal_count = 0
        first_reward_mean = 0.0
        reward_min = float("inf")
        reward_max = float("-inf")
        for step in range(horizon_steps):
            if step == 0:
                action = candidate
            else:
                with torch.no_grad():
                    action, _ = context.model.forward_step(current_observation)
            transition = env.step(action)
            reward = transition["reward"].squeeze(-1)
            selected_reward = reward[selected]
            if step == 0:
                first_reward_mean = float(selected_reward.mean().item())
            reward_min = min(reward_min, float(selected_reward.amin().item()))
            reward_max = max(reward_max, float(selected_reward.amax().item()))
            discounted_return += discount * reward * alive.to(reward.dtype)
            done = transition["done"].squeeze(-1)
            terminal_count += int((alive & done).sum().item())
            alive &= ~done
            current_observation = transition["observation"]
            discount *= context.config.sac.gamma
            if not bool(alive.any().item()):
                break
        candidates.append(
            {
                "delta": float(delta),
                "candidate_action_mean": _vector(candidate[selected].mean(0)),
                "q1_mean": float(q_value[0, selected, 0].mean().item()),
                "q2_mean": float(q_value[1, selected, 0].mean().item()),
                "minimum_q_mean": float(
                    q_value[:, selected, 0].min(0).values.mean().item()
                ),
                "finite_horizon_return_mean": float(
                    discounted_return[selected].mean().item()
                ),
                "finite_horizon_return_p50": float(
                    discounted_return[selected].median().item()
                ),
                "terminal_count": terminal_count,
                "first_reward_mean": first_reward_mean,
                "reward_min": reward_min,
                "reward_max": reward_max,
            }
        )
    _restore_snapshot(context, snapshot)

    gradient_action = base_action.detach().requires_grad_(True)
    q_value = _q_values(context.algorithm, observation, gradient_action)
    minimum_q = q_value.min(0).values
    gradient = torch.autograd.grad(
        minimum_q[selected].mean(), gradient_action
    )[0]
    return {
        "checkpoint_global_control_steps": int(
            context.checkpoint["global_control_steps"]
        ),
        "action_axis": action_axis,
        "action_field": context.config.control_contract.policy_action_fields[
            action_axis
        ],
        "selected_instances": int(selected.sum().item()),
        "selected_yaw_rate_rad_s": {
            "mean": float(truth[selected, 2].mean().item()),
            "p95": float(torch.quantile(truth[selected, 2], 0.95).item()),
        },
        "horizon_steps": horizon_steps,
        "horizon_s": horizon_steps / context.env.spec.control_hz,
        "minimum_q_gradient_on_axis": float(
            gradient[selected, action_axis].mean().item()
        ),
        "candidates": candidates,
    }


def transition_contract_report(
    context: SACDiagnosticContext,
) -> dict[str, Any]:
    """验证标准动作、仿真 command、reward 和 next observation 的一拍因果关系。"""

    snapshot = _snapshot_environment(context)
    observation = context.current_observation
    with torch.no_grad():
        action, _ = context.model.forward_step(observation)
    pilot = context.env.command_source.snapshot()
    expected_command = context.env.action_to_command(
        action,
        pilot.upper_throttle,
        context.config.control_contract.policy_action_trim,
        context.config.control_contract.policy_action_residual_scale,
    )

    first = context.env.step(action)
    first_command = first["info", "action.command"]
    first_simulator_control = context.env.simulator.state_dict()["control"]
    _restore_snapshot(context, snapshot)
    second = context.env.step(action)
    _restore_snapshot(context, snapshot)

    alternative = action.clone()
    alternative[:, 0] = (alternative[:, 0] + 0.05).clamp(-1.0, 1.0)
    different = context.env.step(alternative)
    _restore_snapshot(context, snapshot)

    non_reset = ~first["done"].squeeze(-1)
    return {
        "checkpoint_global_control_steps": int(
            context.checkpoint["global_control_steps"]
        ),
        "restored_observation_max_abs_error": float(
            (
                context.env._observation()[0] - context.current_observation
            ).abs().amax().item()
        ),
        "command_transform_max_abs_error": float(
            (first_command - expected_command).abs().amax().item()
        ),
        "simulator_control_max_abs_error_non_reset": float(
            (
                first_simulator_control[non_reset]
                - expected_command[non_reset]
            ).abs().amax().item()
            if bool(non_reset.any().item())
            else 0.0
        ),
        "exact_restore_reward_equal": bool(
            torch.equal(first["reward"], second["reward"])
        ),
        "exact_restore_next_observation_equal": bool(
            torch.equal(first["observation"], second["observation"])
        ),
        "alternative_action_next_observation_max_abs_delta": float(
            (
                different["observation"] - first["observation"]
            ).abs().amax().item()
        ),
        "alternative_action_reward_max_abs_delta": float(
            (different["reward"] - first["reward"]).abs().amax().item()
        ),
    }


def _q_values(
    algorithm: TorchRLSAC,
    observation: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    values = []
    for index in range(2):
        td = TensorDict(
            {"observation": observation, "action": action},
            batch_size=[observation.shape[0]],
            device=observation.device,
        )
        with algorithm.loss.qvalue_network_params[index].to_module(
            algorithm.loss.qvalue_network
        ):
            output = algorithm.loss.qvalue_network(td)
        values.append(output["state_action_value"])
    return torch.stack(values, dim=0)


def _snapshot_environment(
    context: SACDiagnosticContext,
) -> _EnvironmentSnapshot:
    env = context.env
    return _EnvironmentSnapshot(
        simulator=_clone_state(env.simulator.state_dict()),
        previous_action=env.previous_action.clone(),
        observation_history=env.observation_history.clone(),
        observation_history_index=env.observation_history_index,
        episode_id=env.episode_id.clone(),
        episode_step=env.episode_step.clone(),
        episode_roll_pitch_squared_sum=(
            env.episode_roll_pitch_squared_sum.clone()
        ),
        episode_yaw_rate_squared_sum=env.episode_yaw_rate_squared_sum.clone(),
        episode_angular_rate_squared_sum=(
            env.episode_angular_rate_squared_sum.clone()
        ),
        episode_quality_steps=env.episode_quality_steps.clone(),
        current_static_parameters=env.current_static_parameters.clone(),
        command_source=_clone_state(env.command_source.state_dict()),
        static_randomizer=_clone_state(context.static_randomizer.state_dict()),
        curriculum_stage=env.curriculum_stage,
        curriculum_successes=env.curriculum_successes,
        curriculum_failures=env.curriculum_failures,
        curriculum_consecutive_passes=env.curriculum_consecutive_passes,
        curriculum_last_success_fraction=env.curriculum_last_success_fraction,
    )


def _restore_snapshot(
    context: SACDiagnosticContext,
    snapshot: _EnvironmentSnapshot,
) -> None:
    env = context.env
    env.simulator.load_state_dict(snapshot.simulator)
    env.previous_action.copy_(snapshot.previous_action)
    env.observation_history.copy_(snapshot.observation_history)
    env.observation_history_index = snapshot.observation_history_index
    env.episode_id.copy_(snapshot.episode_id)
    env.episode_step.copy_(snapshot.episode_step)
    env.episode_roll_pitch_squared_sum.copy_(
        snapshot.episode_roll_pitch_squared_sum
    )
    env.episode_yaw_rate_squared_sum.copy_(
        snapshot.episode_yaw_rate_squared_sum
    )
    env.episode_angular_rate_squared_sum.copy_(
        snapshot.episode_angular_rate_squared_sum
    )
    env.episode_quality_steps.copy_(snapshot.episode_quality_steps)
    env.current_static_parameters = snapshot.current_static_parameters.clone()
    env.command_source.load_state_dict(snapshot.command_source)
    context.static_randomizer.load_state_dict(snapshot.static_randomizer)
    env.curriculum_stage = snapshot.curriculum_stage
    env.curriculum_successes = snapshot.curriculum_successes
    env.curriculum_failures = snapshot.curriculum_failures
    env.curriculum_consecutive_passes = snapshot.curriculum_consecutive_passes
    env.curriculum_last_success_fraction = (
        snapshot.curriculum_last_success_fraction
    )
    env.max_episode_steps = env._duration_to_steps(
        context.config.task.curriculum_durations_s[env.curriculum_stage]
    )
    env.command_source.set_curriculum_scale(
        context.config.task.curriculum_target_scales[env.curriculum_stage]
    )


def _restore_environment_only(
    env: SimEnvAdapter,
    state: Mapping[str, Any],
    config: ExperimentConfig,
    static_randomizer: StaticRandomizer,
    reward_calculator: Any,
    device: torch.device,
) -> None:
    training = state.get("training_environment")
    if not isinstance(training, Mapping):
        raise ValueError("checkpoint training_environment state is missing")
    env.simulator.load_state_dict(state["simulator_state"])
    env.previous_action.copy_(
        training["previous_policy_action"].to(device)
    )
    env.episode_id.copy_(training["episode_id"].to(device))
    env.episode_step.copy_(training["episode_step"].to(device))
    for target, name in (
        (
            env.episode_roll_pitch_squared_sum,
            "episode_roll_pitch_squared_sum",
        ),
        (env.episode_yaw_rate_squared_sum, "episode_yaw_rate_squared_sum"),
        (
            env.episode_angular_rate_squared_sum,
            "episode_angular_rate_squared_sum",
        ),
        (env.episode_quality_steps, "episode_quality_steps"),
    ):
        stored = training.get(name)
        if stored is None:
            target.zero_()
        else:
            target.copy_(stored.to(device))
    env.command_source.load_state_dict(training["command_source"])
    curriculum = training["episode_curriculum"]
    env.curriculum_stage = int(curriculum["stage"])
    env.curriculum_successes = int(curriculum["successes"])
    env.curriculum_failures = int(curriculum["failures"])
    env.curriculum_consecutive_passes = int(curriculum["consecutive_passes"])
    env.curriculum_last_success_fraction = float(
        curriculum["last_success_fraction"]
    )
    env.max_episode_steps = env._duration_to_steps(
        config.task.curriculum_durations_s[env.curriculum_stage]
    )
    env.command_source.set_curriculum_scale(
        config.task.curriculum_target_scales[env.curriculum_stage]
    )
    static_randomizer.load_state_dict(training["static_randomizer"])
    reward_calculator.load_state_dict(training["reward_calculator"])
    current_static = training["static_parameters"]
    if not isinstance(current_static, TensorDictBase):
        raise ValueError("checkpoint static_parameters is missing")
    env.current_static_parameters = current_static.to(device)
    stored_history = training.get("observation_history")
    if stored_history is None:
        if env.observation_history_capacity != 1:
            raise ValueError(
                "checkpoint observation history is missing for diagnostics"
            )
        base_observation, *_ = env._base_observation()
        env.observation_history[:, 0].copy_(base_observation)
        env.observation_history_index = 0
    else:
        if not isinstance(stored_history, torch.Tensor):
            raise ValueError("checkpoint observation_history must be a tensor")
        if (
            stored_history.shape != env.observation_history.shape
            or stored_history.dtype != env.observation_history.dtype
        ):
            raise ValueError("checkpoint observation_history is incompatible")
        env.observation_history.copy_(stored_history.to(device))
        history_index = int(training.get("observation_history_index", -1))
        if not 0 <= history_index < env.observation_history_capacity:
            raise ValueError(
                "checkpoint observation_history_index is incompatible"
            )
        env.observation_history_index = history_index


def _clone_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, TensorDictBase):
        return value.clone()
    if isinstance(value, Mapping):
        return type(value)((key, _clone_state(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    return copy.deepcopy(value)


def _vector(value: torch.Tensor) -> list[float]:
    return [
        float(item)
        for item in value.detach().reshape(-1).cpu().tolist()
    ]
