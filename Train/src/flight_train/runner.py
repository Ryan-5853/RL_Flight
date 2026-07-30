from __future__ import annotations

import signal
import threading
from typing import Any, Mapping

import torch
from tensordict import TensorDictBase
from simenv import InsufficientDiskSpaceError as SimulatorDiskSpaceError

from .algorithms import TorchRLPPO, TorchRLSAC
from .collector import TensorDictRolloutCollector
from .config import ExperimentConfig, exact_resume_config_sha256
from .core import tensordict_to_device
from .envs import SimEnvAdapter
from .models import build_actor_critic, build_sac_actor_critic
from .progress import LiveTrainingProgress
from .randomization import StaticRandomizer
from .recording import (
    InsufficientDiskSpaceError as CheckpointDiskSpaceError,
    RunRecorder,
    load_checkpoint,
)
from .registry import ComponentRegistry


def _rollout_diagnostic_metrics(
    rollout: TensorDictBase, control_hz: int
) -> dict[str, torch.Tensor]:
    """在 update 边界汇总控制质量；不在逐控制步执行 CPU 同步。"""

    terminated = rollout[("next", "terminated")]
    truncated = rollout[("next", "truncated")]
    done = rollout[("next", "done")]
    valid = rollout[("next", "valid")]
    action = rollout["action"]
    done_float = done.to(torch.float32)
    done_count = done_float.sum()
    metrics = {
        "terminated_fraction": terminated.to(torch.float32).mean(),
        "truncated_fraction": truncated.to(torch.float32).mean(),
        "done_fraction": done_float.mean(),
        "invalid_fraction": (~valid).to(torch.float32).mean(),
        "episode_reset_count": done_count,
        "action_rms": action.square().mean().sqrt(),
        "action_peak_abs": action.abs().amax(),
        "action_saturation_fraction": (action.abs() >= 0.95).to(torch.float32).mean(),
    }
    diagnostic_keys = {
        "attitude_error_rad",
        "angular_rate_norm_rad_s",
        "height_error_m",
        "episode_length_steps",
    }
    if not diagnostic_keys.issubset(set(rollout["next"].keys())):
        return metrics

    valid_float = valid.to(torch.float32)
    valid_count = valid_float.sum().clamp_min(1.0)
    attitude_error = rollout[("next", "attitude_error_rad")]
    angular_rate = rollout[("next", "angular_rate_norm_rad_s")]
    height_error = rollout[("next", "height_error_m")]
    episode_length = rollout[("next", "episode_length_steps")].to(torch.float32)
    safe_attitude = torch.where(valid, attitude_error, torch.zeros_like(attitude_error))
    metrics.update(
        {
            "attitude_error_mean_deg": (
                safe_attitude.sum() / valid_count * (180.0 / torch.pi)
            ),
            "attitude_error_p95_deg": (
                torch.quantile(safe_attitude.flatten(), 0.95) * (180.0 / torch.pi)
            ),
            "angular_rate_mean_rad_s": (
                torch.where(valid, angular_rate, torch.zeros_like(angular_rate)).sum()
                / valid_count
            ),
            "height_error_abs_mean_m": (
                torch.where(valid, height_error.abs(), torch.zeros_like(height_error)).sum()
                / valid_count
            ),
            "completed_episode_length_mean_steps": (
                (episode_length * done_float).sum() / done_count.clamp_min(1.0)
            ),
            "completed_episode_survival_mean_s": (
                (episode_length * done_float).sum()
                / done_count.clamp_min(1.0)
                / float(control_hz)
            ),
            "episode_age_mean_s": episode_length.mean() / float(control_hz),
            "episode_age_p50_s": torch.quantile(episode_length, 0.50)
            / float(control_hz),
            "episode_age_p95_s": torch.quantile(episode_length, 0.95)
            / float(control_hz),
            "episode_age_max_s": episode_length.amax() / float(control_hz),
        }
    )
    for seconds in (1, 2, 5, 10, 30):
        metrics[f"episode_age_ge_{seconds}s_fraction"] = (
            episode_length >= seconds * control_hz
        ).to(torch.float32).mean()
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
        if reward_key in rollout["next"].keys():
            metrics[f"{reward_key}_mean"] = rollout[("next", reward_key)].mean()
    return metrics


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """组装环境、模型、采集器和算法，执行一次完整配置驱动实验。

    该函数是训练编排层：数值计算由各张量模块完成，这里只管理生命周期、
    全局控制步计数、低频记录和最终 checkpoint。
    """

    device = torch.device(config.run.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    torch.manual_seed(config.run.seed)
    torch.use_deterministic_algorithms(config.seeds.deterministic_algorithms)
    torch.backends.cudnn.benchmark = config.seeds.cudnn_benchmark
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.run.seed)

    env = None
    recorder = None
    progress = None
    global_steps = 0
    status = "failed"
    stop_signal: int | None = None
    stop_reason: str | None = None
    stop_detail: str | None = None
    last_checkpoint_step: int | None = None
    last_checkpoint_kind: str | None = None
    previous_signal_handlers: dict[int, Any] = {}

    def request_safe_stop(signum: int, _frame: Any) -> None:
        """第一次信号请求在更新边界保存；第二次信号立即中断。"""

        nonlocal stop_signal
        if stop_signal is not None:
            raise KeyboardInterrupt
        stop_signal = signum
    try:
        recorder = RunRecorder(config)
        progress = LiveTrainingProgress(config, recorder.run_id)
        # A policy/exact resume checkpoint gives a useful estimate before the
        # first rollout. Fresh runs are checked again with their concrete state
        # immediately before the first checkpoint.
        recorder.ensure_checkpoint_capacity()
        registry = ComponentRegistry()
        reward_calculator = registry.build_reward(
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
            reward_context_fields=tuple(dict(field) for field in config.reward.context_fields),
            command_source_config=config.command_source,
            control_contract_config=config.control_contract,
        )
        if not config.run.allow_unimplemented_simulator:
            # 正式实验默认拒绝占位 kernel，防止接口冒烟结果被误认为控制学习结果。
            missing = []
            if not env.simulator.dynamics_implemented:
                missing.append("dynamics")
            if not env.simulator.sensors_implemented:
                missing.append("sensors")
            if missing:
                raise RuntimeError(
                    "SimEnv kernels are not implemented: " + ", ".join(missing)
                    + "; set run.allow_unimplemented_simulator=true only for interface smoke tests"
                )

        model = (
            build_sac_actor_critic(
                env.spec.observation_dim,
                env.spec.action_dim,
                config.model,
                device,
                config.torch_dtype,
            )
            if config.algorithm_name == "sac"
            else build_actor_critic(
                env.spec.observation_dim,
                env.spec.action_dim,
                config.model,
                device,
                config.torch_dtype,
            )
        )
        collector = TensorDictRolloutCollector(env, model, config.run.rollout_steps)
        if config.algorithm_name == "sac":
            if config.sac is None:
                raise RuntimeError("parsed SAC configuration is missing")
            algorithm = TorchRLSAC(model, config.sac, device)
        else:
            if config.ppo is None:
                raise RuntimeError("parsed PPO configuration is missing")
            algorithm = TorchRLPPO(model, config.ppo, device)
        resume_parent_run_id: str | None = None
        resume_source_global_control_steps: int | None = None
        if config.checkpoint.resume_from is not None:
            resume_state = load_checkpoint(config.checkpoint.resume_from)
            resume_parent_run_id = (
                str(resume_state.get("run_id") or "") or None
            )
            resume_source_global_control_steps = int(
                resume_state.get("global_control_steps", -1)
            )
            if config.checkpoint.resume_mode == "exact":
                global_steps = _restore_exact(
                    config,
                    resume_state,
                    env=env,
                    model=model,
                    algorithm=algorithm,
                    collector=collector,
                    static_randomizer=static_randomizer,
                    reward_calculator=reward_calculator,
                    device=device,
                )
            else:
                _restore_policy(resume_state, model=model)
                if isinstance(algorithm, TorchRLSAC):
                    algorithm.initialize_policy_anchor()
            # policy 热启动只需 actor；exact restore 也已把所有状态复制到对应
            # 组件。及时释放包含完整 replay 的 CPU checkpoint，避免训练全程
            # 额外占用约 5 GB 主存。
            del resume_state
        recorder.bind_environment(
            batch_id=env.simulator.batch_id,
            instance_ids=env.simulator.instance_ids,
            simulator_log_directory=env.simulator.log_directory,
        )
        if config.checkpoint.resume_from is not None:
            recorder.bind_resume(
                checkpoint=config.checkpoint.resume_from,
                parent_run_id=resume_parent_run_id,
                mode=config.checkpoint.resume_mode,
                source_global_control_steps=resume_source_global_control_steps,
            )
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_safe_stop)
        checkpoint_interval = config.checkpoint.interval_control_steps
        next_checkpoint_step = (
            None
            if checkpoint_interval is None
            else (global_steps // checkpoint_interval + 1) * checkpoint_interval
        )
        next_evaluation_step = (
            None
            if not config.evaluation.enabled
            else (
                global_steps // config.evaluation.interval_control_steps + 1
            )
            * config.evaluation.interval_control_steps
        )
        rollout_index = global_steps // (
            config.run.parallel_count * config.run.rollout_steps
        )
        while global_steps < config.run.total_control_steps:
            # Stop at a rollout boundary before subsequent logs can consume the
            # capacity reserved for the next atomic checkpoint.
            recorder.ensure_checkpoint_capacity()
            rollout_index += 1
            model.set_exploration_progress(
                global_steps / float(config.model.exploration_decay_control_steps)
            )
            progress.begin_rollout(global_steps, rollout_index)
            rollout = collector.collect(progress.collect_step)
            progress.begin_update()
            metrics = algorithm.update(rollout, progress.update_step)
            sac_learning_locked = (
                config.algorithm_name == "sac"
                and (
                    float(metrics["sac_warmup"].detach().item()) > 0.5
                    or float(
                        metrics["sac_critic_pretraining"].detach().item()
                    )
                    > 0.5
                )
            )
            curriculum_quality_success = rollout.get(
                ("next", "curriculum_quality_success"),
                None,
            )
            curriculum_metrics = env.update_episode_curriculum(
                rollout[("next", "terminated")],
                rollout[("next", "truncated")],
                curriculum_quality_success,
                allow_promotion=not sac_learning_locked,
            )
            # TensorDict 的 [B,T] batch 元素数就是本轮新增的控制决策总数。
            global_steps += rollout.batch_size.numel()
            metrics = {
                **metrics,
                "rollout_reward_mean": rollout[("next", "reward")].mean(),
                "valid_fraction": rollout[("next", "valid")].to(torch.float32).mean(),
                **curriculum_metrics,
                **_rollout_diagnostic_metrics(rollout, env.spec.control_hz),
            }
            exploration_std_for = getattr(model, "exploration_std_for", None)
            if "policy_scale" in rollout.keys():
                exploration_std = rollout["policy_scale"].reshape(
                    -1, rollout["policy_scale"].shape[-1]
                ).mean(dim=0)
            else:
                exploration_std = (
                    exploration_std_for(rollout["observation"])
                    if exploration_std_for is not None
                    else model.exploration_std()
                )
            if exploration_std is not None:
                for action_index, value in enumerate(exploration_std):
                    metrics[f"exploration_std_action_{action_index}"] = value
            metric_record = recorder.metrics(global_steps, metrics)
            progress.complete_update(global_steps, metric_record)
            if next_checkpoint_step is not None and global_steps >= next_checkpoint_step:
                checkpoint_kind = (
                    "final"
                    if global_steps >= config.run.total_control_steps
                    else "periodic"
                )
                checkpoint_path = recorder.checkpoint(
                    global_steps,
                    _checkpoint_state(
                        config,
                        recorder.run_id,
                        global_steps,
                        env,
                        model,
                        algorithm,
                        collector,
                        static_randomizer,
                        reward_calculator,
                        device,
                    ),
                    kind=checkpoint_kind,
                )
                last_checkpoint_step = global_steps
                last_checkpoint_kind = checkpoint_kind
                progress.checkpoint(global_steps, checkpoint_kind)
                if (
                    next_evaluation_step is not None
                    and global_steps >= next_evaluation_step
                ):
                    from .evaluation import (
                        load_fixed_evaluation_suite,
                        run_fixed_evaluation,
                    )

                    suite = load_fixed_evaluation_suite(
                        config.evaluation.suite_path
                    )
                    cuda_devices = (
                        []
                        if device.type != "cuda"
                        else [
                            device.index
                            if device.index is not None
                            else torch.cuda.current_device()
                        ]
                    )
                    # 固定评测会设置自己的 seed；fork_rng 保证它不改变续训轨迹。
                    with torch.random.fork_rng(devices=cuda_devices):
                        evaluation_result = run_fixed_evaluation(
                            config,
                            checkpoint_path,
                            suite,
                            output_root=recorder.directory / "evaluations",
                        )
                    recorder.evaluation(global_steps, evaluation_result)
                    hover = evaluation_result["report"]["scenarios"]["hover"]
                    hover_survival = float(
                        hover["metrics"]["survival_time_s"]["mean"]
                    )
                    hover_roll_pitch_rmse = float(
                        hover["metrics"]["roll_pitch_rmse_deg"]["mean"]
                    )
                    hover_yaw_rate_rmse = float(
                        hover["metrics"]["yaw_rate_rmse_rad_s"]["mean"]
                    )
                    selection = suite.checkpoint_selection
                    best_quality_passed = (
                        hover_survival
                        >= selection.minimum_hover_survival_s
                        and hover_roll_pitch_rmse
                        <= selection.maximum_hover_roll_pitch_rmse_deg
                        and (
                            selection.maximum_hover_yaw_rate_rmse_rad_s is None
                            or hover_yaw_rate_rmse
                            <= selection.maximum_hover_yaw_rate_rmse_rad_s
                        )
                    )
                    promoted = recorder.promote_evaluation_checkpoints(
                        checkpoint_path,
                        control_steps=global_steps,
                        hover_survival_s=hover_survival,
                        hover_roll_pitch_rmse_deg=hover_roll_pitch_rmse,
                        hover_yaw_rate_rmse_rad_s=hover_yaw_rate_rmse,
                        total_score=float(
                            evaluation_result["report"]["total_score"]
                        ),
                        minimum_hover_survival_s=(
                            selection.minimum_hover_survival_s
                        ),
                        quality_passed=best_quality_passed,
                    )
                    print(
                        "固定评测完成 | "
                        f"控制步={global_steps} | "
                        f"总分={evaluation_result['report']['total_score']:.2f} | "
                        f"悬停生存={hover_survival:.2f}s | "
                        f"横滚俯仰RMSE={hover_roll_pitch_rmse:.2f}deg | "
                        f"偏航角速度RMSE={hover_yaw_rate_rmse:.2f}rad/s"
                        + (
                            ""
                            if best_quality_passed
                            else " | 未通过 best 质量门槛"
                        )
                        + (
                            f" | 已更新 best={','.join(promoted)}"
                            if promoted
                            else ""
                        ),
                        flush=True,
                    )
                    while next_evaluation_step <= global_steps:
                        next_evaluation_step += (
                            config.evaluation.interval_control_steps
                        )
                while next_checkpoint_step <= global_steps:
                    next_checkpoint_step += checkpoint_interval
            if stop_signal is not None:
                if last_checkpoint_step == global_steps:
                    recorder.relabel_checkpoint(global_steps, kind="interrupt")
                else:
                    recorder.checkpoint(
                        global_steps,
                        _checkpoint_state(
                            config,
                            recorder.run_id,
                            global_steps,
                            env,
                            model,
                            algorithm,
                            collector,
                            static_randomizer,
                            reward_calculator,
                            device,
                        ),
                        kind="interrupt",
                    )
                progress.checkpoint(global_steps, "interrupt")
                last_checkpoint_step = global_steps
                last_checkpoint_kind = "interrupt"
                status = "interrupted"
                break
        if status != "interrupted" and last_checkpoint_step != global_steps:
            recorder.checkpoint(
                global_steps,
                _checkpoint_state(
                    config,
                    recorder.run_id,
                    global_steps,
                    env,
                    model,
                    algorithm,
                    collector,
                    static_randomizer,
                    reward_calculator,
                    device,
                ),
                kind="final",
            )
            progress.checkpoint(global_steps, "final")
            last_checkpoint_kind = "final"
        elif status != "interrupted" and last_checkpoint_kind != "final":
            recorder.relabel_checkpoint(global_steps, kind="final")
            progress.checkpoint(global_steps, "final")
            last_checkpoint_kind = "final"
        if status != "interrupted":
            status = "completed"
    except (CheckpointDiskSpaceError, SimulatorDiskSpaceError) as exc:
        status = "interrupted"
        stop_reason = "insufficient_disk_space"
        stop_detail = str(exc)
        print(
            "磁盘空间不足，已在安全边界停止训练"
            f" | 控制步={global_steps:,}"
            f" | 最近完整 checkpoint={last_checkpoint_step}"
            f" | {exc}",
            flush=True,
        )
    finally:
        # 即使创建或训练中途抛错，也释放仿真资源并留下状态记录。SimEnv
        # 可能只在关闭时等待到后台写入结果，因此这里仍识别空间保护异常。
        close_error: BaseException | None = None
        if env is not None:
            try:
                env.close()
            except SimulatorDiskSpaceError as exc:
                status = "interrupted"
                stop_reason = "insufficient_disk_space"
                stop_detail = str(exc)
            except BaseException as exc:
                close_error = exc
        if recorder is not None:
            recorder.close(
                status,
                global_steps,
                reason=stop_reason,
                detail=stop_detail,
                last_checkpoint_step=last_checkpoint_step,
            )
        if progress is not None:
            progress.finish(status, global_steps)
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)
        if close_error is not None:
            raise close_error
    return {
        "run_directory": recorder.directory,
        "global_control_steps": global_steps,
        "status": status,
        "reason": stop_reason,
        "last_checkpoint_step": last_checkpoint_step,
    }


def _checkpoint_state(
    config: ExperimentConfig,
    run_id: str,
    global_steps: int,
    env: SimEnvAdapter,
    model,
    algorithm: TorchRLPPO | TorchRLSAC,
    collector: TensorDictRolloutCollector,
    static_randomizer: StaticRandomizer,
    reward_calculator,
    device: torch.device,
) -> Mapping[str, Any]:
    """在完整采集/更新边界构造精确续训状态。"""

    state: dict[str, Any] = {
        "checkpoint_schema_version": 5 if config.algorithm_name == "sac" else 4,
        "algorithm_name": config.algorithm_name,
        "run_id": run_id,
        "exact_resume_config_sha256": exact_resume_config_sha256(config),
        "global_control_steps": global_steps,
        "actor": model.actor.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        ),
        "collector_current": collector.current,
        "simulator_state": env.simulator.state_dict(),
        "training_environment": {
            "previous_policy_action": env.previous_action,
            "observation_history": env.observation_history,
            "observation_history_index": env.observation_history_index,
            "episode_id": env.episode_id,
            "episode_step": env.episode_step,
            "episode_roll_pitch_squared_sum": (
                env.episode_roll_pitch_squared_sum
            ),
            "episode_yaw_rate_squared_sum": env.episode_yaw_rate_squared_sum,
            "episode_angular_rate_squared_sum": (
                env.episode_angular_rate_squared_sum
            ),
            "episode_quality_steps": env.episode_quality_steps,
            "command_source": env.command_source.state_dict(),
            "static_parameters": env.current_static_parameters,
            "dynamic_parameters": {
                name: value.clone() for name, value in env.simulator.parameters.items()
            },
            "static_randomizer": static_randomizer.state_dict(),
            "reward_calculator": reward_calculator.state_dict(),
            "episode_curriculum": {
                "stage": env.curriculum_stage,
                "successes": env.curriculum_successes,
                "failures": env.curriculum_failures,
                "consecutive_passes": env.curriculum_consecutive_passes,
                "last_success_fraction": env.curriculum_last_success_fraction,
            },
        },
        "config": config.raw,
    }
    if isinstance(algorithm, TorchRLSAC):
        state["sac"] = algorithm.state_dict()
    else:
        state.update(
            {
                "critic": model.critic.state_dict(),
                "actor_optimizer": algorithm.actor_optimizer.state_dict(),
                "critic_optimizer": algorithm.critic_optimizer.state_dict(),
            }
        )
    return state


def _restore_exact(
    config: ExperimentConfig,
    state: Mapping[str, Any],
    *,
    env: SimEnvAdapter,
    model,
    algorithm: TorchRLPPO | TorchRLSAC,
    collector: TensorDictRolloutCollector,
    static_randomizer: StaticRandomizer,
    reward_calculator,
    device: torch.device,
) -> int:
    """按依赖顺序恢复 rollout 边界上的完整训练状态。"""

    expected_schema = 5 if config.algorithm_name == "sac" else 4
    if int(state.get("checkpoint_schema_version", -1)) != expected_schema:
        raise ValueError(
            f"checkpoint does not contain exact-resume schema v{expected_schema} state"
        )
    if str(state.get("algorithm_name", config.algorithm_name)) != config.algorithm_name:
        raise ValueError("checkpoint algorithm is incompatible")
    expected = exact_resume_config_sha256(config)
    if state.get("exact_resume_config_sha256") != expected:
        raise ValueError("checkpoint configuration is incompatible with exact resume")
    global_steps = int(state.get("global_control_steps", -1))
    if global_steps < 0 or global_steps > config.run.total_control_steps:
        raise ValueError("checkpoint global_control_steps is outside the requested run budget")

    training = state.get("training_environment")
    if not isinstance(training, Mapping):
        raise ValueError("checkpoint training_environment state is missing")
    simulator_state = state.get("simulator_state")
    if not isinstance(simulator_state, Mapping):
        raise ValueError("checkpoint simulator_state is missing")
    env.simulator.load_state_dict(simulator_state)

    _copy_training_tensor(env.previous_action, training, "previous_policy_action")
    _copy_training_tensor(env.episode_id, training, "episode_id")
    _copy_training_tensor(env.episode_step, training, "episode_step")
    for tensor, name in (
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
            tensor.zero_()
        else:
            _copy_training_tensor(tensor, training, name)
    env.command_source.load_state_dict(training["command_source"])
    curriculum = training.get("episode_curriculum")
    if not isinstance(curriculum, Mapping):
        raise ValueError("checkpoint episode_curriculum state is missing")
    env.curriculum_stage = int(curriculum["stage"])
    if not 0 <= env.curriculum_stage < len(config.task.curriculum_durations_s):
        raise ValueError("checkpoint episode curriculum stage is incompatible")
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
    current_static = training.get("static_parameters")
    if not isinstance(current_static, TensorDictBase):
        raise ValueError("checkpoint static_parameters is missing")
    env.current_static_parameters = tensordict_to_device(current_static, device)
    stored_history = training.get("observation_history")
    if stored_history is None:
        if env.observation_history_capacity != 1:
            raise ValueError(
                "checkpoint observation history is missing for a stacked observation"
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

    model.actor.load_state_dict(state["actor"])
    if isinstance(algorithm, TorchRLSAC):
        sac_state = state.get("sac")
        if not isinstance(sac_state, Mapping):
            raise ValueError("checkpoint SAC state is missing")
        algorithm.load_state_dict(sac_state)
    else:
        model.critic.load_state_dict(state["critic"])
        algorithm.actor_optimizer.load_state_dict(state["actor_optimizer"])
        algorithm.critic_optimizer.load_state_dict(state["critic_optimizer"])
    current = state.get("collector_current")
    if not isinstance(current, TensorDictBase):
        raise ValueError("checkpoint collector_current is missing")
    collector.current = tensordict_to_device(current, device)

    # RNG 最后恢复，确保环境/模型构造和 state load 的任何实现细节都不能消耗续训流。
    torch_rng = state.get("torch_rng_state")
    if not isinstance(torch_rng, torch.Tensor):
        raise ValueError("checkpoint torch_rng_state is missing")
    torch.set_rng_state(torch_rng.cpu())
    if device.type == "cuda":
        cuda_rng = state.get("cuda_rng_state_all")
        if not isinstance(cuda_rng, list) or len(cuda_rng) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state is incompatible with visible devices")
        torch.cuda.set_rng_state_all(cuda_rng)
    return global_steps


def _restore_policy(state: Mapping[str, Any], *, model) -> None:
    """只恢复策略权重；新目标下的 critic、replay 与优化器从头初始化。"""

    actor_state = state.get("actor")
    if not isinstance(actor_state, Mapping):
        raise ValueError("checkpoint actor state is missing")
    model.actor.load_state_dict(actor_state, strict=True)


def _copy_training_tensor(
    destination: torch.Tensor, state: Mapping[str, Any], name: str
) -> None:
    value = state.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"checkpoint training tensor is missing: {name}")
    if value.shape != destination.shape or value.dtype != destination.dtype:
        raise ValueError(f"checkpoint training tensor is incompatible: {name}")
    destination.copy_(value.to(destination.device))
