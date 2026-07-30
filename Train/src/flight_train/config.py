from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from .randomization import StaticParameterSpec


class ConfigError(ValueError):
    """实验配置不符合 schema 或跨字段约束。"""


@dataclass(frozen=True)
class ComponentConfig:
    type: str
    version: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class SeedConfig:
    base: int
    deterministic_algorithms: bool
    cudnn_benchmark: bool


@dataclass(frozen=True)
class RandomizationConfig:
    seed: int
    scope: str
    parameters: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class RewardCalculatorConfig:
    calculator: ComponentConfig
    context_fields: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TaskConfig:
    episode_duration_s: float
    terminate_tilt_rad: float
    terminate_angular_rate_rad_s: float
    attitude_source: str
    terminate_angular_rate_axes: str = "all"
    curriculum_durations_s: tuple[float, ...] = ()
    curriculum_target_scales: tuple[float, ...] = ()
    curriculum_success_fraction: float = 1.0
    curriculum_evaluation_episodes_per_env: float = 1.0
    curriculum_consecutive_passes: int = 1
    curriculum_max_roll_pitch_rmse_rad: float | None = None
    curriculum_max_yaw_rate_rmse_rad_s: float | None = None
    curriculum_max_angular_rate_rms_rad_s: float | None = None


@dataclass(frozen=True)
class VirtualPilotConfig:
    type: str
    version: str
    seed: int
    throttle_minimum: float
    throttle_maximum: float
    spool_duration_s: float
    spool_target_range: tuple[float, float]
    height_target_m: float
    height_initial_throttle_range: tuple[float, float]
    height_proportional_gain: float
    height_integral_gain: float
    height_error_limit_m: float
    throttle_rise_rate_per_s: float
    throttle_fall_rate_per_s: float
    max_roll_rad: float
    max_pitch_rad: float
    max_yaw_rate_rad_s: float
    roll_time_constant_s: float
    pitch_time_constant_s: float
    yaw_time_constant_s: float
    center_exponent: float
    hold_duration_range_s: tuple[float, float]
    initial_target_scale: float


@dataclass(frozen=True)
class ControlContractConfig:
    version: str
    observation_profile: str
    observation_history_mode: str
    observation_history_frames: int
    observation_history_stride_steps: int
    observation_history_dense_action_steps: int
    observation_history_sparse_physical_frames: int
    observation_history_sparse_physical_stride_steps: int
    policy_action_fields: tuple[str, ...]
    external_action_fields: tuple[str, ...]
    simulator_command_fields: tuple[str, ...]
    policy_action_trim: tuple[float, ...]
    policy_action_residual_scale: tuple[float, ...]

    @property
    def observation_dim(self) -> int:
        if self.observation_history_mode == "uniform":
            return 21 * self.observation_history_frames
        if self.observation_history_mode == "multirate_actuator":
            return (
                21
                + 4 * self.observation_history_dense_action_steps
                + 11 * self.observation_history_sparse_physical_frames
            )
        raise RuntimeError(
            f"unsupported observation history mode "
            f"{self.observation_history_mode!r}"
        )

    @property
    def observation_history_span_steps(self) -> int:
        if self.observation_history_mode == "uniform":
            return (
                self.observation_history_frames - 1
            ) * self.observation_history_stride_steps
        if self.observation_history_mode == "multirate_actuator":
            return max(
                self.observation_history_dense_action_steps,
                self.observation_history_sparse_physical_frames
                * self.observation_history_sparse_physical_stride_steps,
            )
        raise RuntimeError(
            f"unsupported observation history mode "
            f"{self.observation_history_mode!r}"
        )

    @property
    def current_observation_offset(self) -> int:
        if self.observation_history_mode == "uniform":
            return (self.observation_history_frames - 1) * 21
        if self.observation_history_mode == "multirate_actuator":
            return 0
        raise RuntimeError(
            f"unsupported observation history mode "
            f"{self.observation_history_mode!r}"
        )


@dataclass(frozen=True)
class CheckpointConfig:
    resume_from: Path | None
    resume_mode: str
    interval_control_steps: int | None
    keep_last: int
    minimum_free_space_bytes: int = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool
    interval_control_steps: int | None
    suite_path: Path | None


@dataclass(frozen=True)
class ModelConfig:
    encoder_sizes: tuple[int, ...]
    hidden_size: int
    actor_head_size: int
    architecture: str = "gru"
    initial_action_std: tuple[float, ...] = ()
    final_action_std: tuple[float, ...] = ()
    exploration_decay_control_steps: int = 1
    sac_initial_action_std: tuple[float, ...] = ()
    sac_minimum_action_std: tuple[float, ...] = ()
    sac_maximum_action_std: tuple[float, ...] = ()
    sac_learnable_action_std: bool = True
    recurrent_layers: int = 1


@dataclass(frozen=True)
class PPOConfig:
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    entropy_coefficient: float
    value_coefficient: float
    # max_grad_norm/learning_rate 保留为 actor 的公开字段，兼容已有调用方。
    max_grad_norm: float
    learning_rate: float
    epochs: int
    sequence_length: int
    sequences_per_minibatch: int
    critic_max_grad_norm: float | None = None
    critic_learning_rate: float | None = None


@dataclass(frozen=True)
class SACConfig:
    """连续动作 SAC 的训练和 GPU 经验回放配置。"""

    gamma: float
    n_step_return: int
    replay_capacity: int
    replay_batch_size: int
    warmup_transitions: int
    updates_per_collection: int
    actor_learning_rate: float
    critic_learning_rate: float
    alpha_learning_rate: float
    actor_max_grad_norm: float
    critic_max_grad_norm: float
    target_tau: float
    target_update_interval: int
    initial_alpha: float
    target_entropy: float | str
    min_alpha: float | None
    max_alpha: float | None
    replay_sequence_length: int = 1
    replay_burn_in_steps: int = 0
    critic_pretraining_updates: int = 0
    actor_update_interval: int = 1
    policy_anchor_weight: float = 0.0
    policy_anchor_max_action_deviation: float = 0.0

    @property
    def replay_sample_length(self) -> int:
        """每条 replay 样本的总步数：隐状态预热段加有效训练段。"""

        return self.replay_burn_in_steps + self.replay_sequence_length


@dataclass(frozen=True)
class RunConfig:
    device: str
    dtype: str
    parallel_count: int
    rollout_steps: int
    total_control_steps: int
    seed: int
    output_root: Path
    allow_unimplemented_simulator: bool


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    entrypoint: ComponentConfig
    seeds: SeedConfig
    simulator_factory: str
    simulator_config: Path
    static_randomization: RandomizationConfig
    dynamic_randomization: RandomizationConfig
    reward: RewardCalculatorConfig
    run: RunConfig
    task: TaskConfig
    command_source: VirtualPilotConfig
    control_contract: ControlContractConfig
    checkpoint: CheckpointConfig
    evaluation: EvaluationConfig
    model: ModelConfig
    algorithm_name: str
    ppo: PPOConfig | None
    sac: SACConfig | None
    source_path: Path
    raw: Mapping[str, Any]

    @property
    def torch_dtype(self) -> torch.dtype:
        values = {"float32": torch.float32, "float64": torch.float64}
        try:
            return values[self.run.dtype]
        except KeyError as exc:
            raise ConfigError("run.dtype must be float32 or float64") from exc

    def static_parameter_specs(self) -> tuple[StaticParameterSpec, ...]:
        specs = []
        for name, node in self.static_randomization.parameters.items():
            distribution = str(node["distribution"])
            low, high = _parameter_range(node)
            specs.append(
                StaticParameterSpec(
                    name=name,
                    low=low,
                    high=high,
                    distribution=distribution,
                    mode=str(node.get("mode", "absolute")),
                    unit=str(node.get("unit", "unspecified")),
                    seed_stream=str(node["seed_stream"]),
                )
            )
        return tuple(specs)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    raw = _load(source)
    _keys(raw, {
        "schema_version", "experiment", "seed", "run", "environment",
        "randomization", "task", "reward", "model", "algorithm", "collector",
        "command_source", "control_contract", "evaluation", "checkpoint", "recording",
    }, "root")
    if raw.get("schema_version") != 2:
        raise ConfigError("schema_version must equal 2; v1 reward templates require explicit migration")

    experiment = _map(raw, "experiment")
    entrypoint = _component(_map(experiment, "entrypoint"), "experiment.entrypoint")
    name = str(experiment.get("name", "")).strip()
    if not name:
        raise ConfigError("experiment.name must not be empty")

    seed_node = _map(raw, "seed")
    seeds = SeedConfig(
        base=_nonnegative_int(seed_node, "base"),
        deterministic_algorithms=bool(seed_node.get("deterministic_algorithms", True)),
        cudnn_benchmark=bool(seed_node.get("cudnn_benchmark", False)),
    )

    run_node = _map(raw, "run")
    collector = _map(raw, "collector")
    run = RunConfig(
        device=str(run_node.get("device", "cuda:0")),
        dtype=str(run_node.get("dtype", "float32")),
        parallel_count=_positive_int(run_node, "parallel_count"),
        rollout_steps=_positive_int(collector, "control_steps_per_rollout"),
        total_control_steps=_positive_int(run_node, "total_control_steps"),
        seed=seeds.base,
        output_root=(source.parent / str(run_node.get("output_root", "../../runs"))).resolve(),
        allow_unimplemented_simulator=bool(run_node.get("allow_unimplemented_simulator", False)),
    )

    environment = _map(raw, "environment")
    _keys(
        environment,
        {"factory", "config_path", "observation_source", "observation_profile"},
        "environment",
    )
    simulator = (source.parent / str(environment.get("config_path", ""))).resolve()
    if not simulator.is_file():
        raise ConfigError(f"simulator config does not exist: {simulator}")

    randomization = _map(raw, "randomization")
    static = _randomization(_map(randomization, "static"), "randomization.static", static=True)
    dynamic = _randomization(_map(randomization, "dynamic"), "randomization.dynamic", static=False)
    if static.seed == dynamic.seed:
        raise ConfigError("static and dynamic randomization seeds must differ")
    overlap = set(static.parameters) & set(dynamic.parameters)
    if overlap:
        raise ConfigError(
            f"randomization parameters must have one owner; duplicated: {sorted(overlap)}"
        )
    streams = [
        str(node["seed_stream"])
        for group in (static.parameters, dynamic.parameters)
        for node in group.values()
    ]
    if len(streams) != len(set(streams)):
        raise ConfigError("randomization seed_stream values must be globally unique")

    task_node = _map(raw, "task")
    _keys(
        task_node,
        {"episode_duration_s", "termination", "episode_curriculum"},
        "task",
    )
    termination = _map(task_node, "termination")
    _keys(
        termination,
        {"max_tilt_rad", "max_angular_rate_rad_s", "angular_rate_axes"},
        "task.termination",
    )
    angular_rate_axes = str(termination.get("angular_rate_axes", "all"))
    if angular_rate_axes not in {"all", "roll_pitch"}:
        raise ConfigError(
            "task.termination.angular_rate_axes must be all or roll_pitch"
        )
    task = TaskConfig(
        episode_duration_s=_positive_float(task_node, "episode_duration_s"),
        terminate_tilt_rad=_positive_float(termination, "max_tilt_rad"),
        terminate_angular_rate_rad_s=_positive_float(termination, "max_angular_rate_rad_s"),
        attitude_source=str(environment.get("observation_source", "sensor")),
        terminate_angular_rate_axes=angular_rate_axes,
    )
    curriculum_node = task_node.get("episode_curriculum")
    if curriculum_node is not None:
        if not isinstance(curriculum_node, Mapping):
            raise ConfigError("task.episode_curriculum must be a mapping")
        _keys(
            curriculum_node,
            {
                "durations_s",
                "target_scales",
                "success_fraction",
                "evaluation_episodes_per_env",
                "consecutive_passes",
                "quality_gate",
            },
            "task.episode_curriculum",
        )
        durations = tuple(
            _positive_float_value(v, "task.episode_curriculum.durations_s")
            for v in _sequence(curriculum_node, "durations_s")
        )
        scales = tuple(
            _unit_float_value(v, "task.episode_curriculum.target_scales")
            for v in _sequence(curriculum_node, "target_scales")
        )
        if not durations or len(durations) != len(scales):
            raise ConfigError(
                "episode curriculum durations_s and target_scales must have equal nonzero length"
            )
        if any(right < left for left, right in zip(durations, durations[1:])):
            raise ConfigError("episode curriculum durations_s must be nondecreasing")
        if any(right < left for left, right in zip(scales, scales[1:])):
            raise ConfigError("episode curriculum target_scales must be nondecreasing")
        if any(
            right_duration == left_duration and right_scale == left_scale
            for left_duration, right_duration, left_scale, right_scale in zip(
                durations,
                durations[1:],
                scales,
                scales[1:],
            )
        ):
            raise ConfigError(
                "each curriculum stage must increase duration_s or target_scale"
            )
        if abs(durations[-1] - task.episode_duration_s) > 1e-9:
            raise ConfigError(
                "last curriculum duration must equal task.episode_duration_s"
            )
        quality_gate = curriculum_node.get("quality_gate", {})
        if not isinstance(quality_gate, Mapping):
            raise ConfigError(
                "task.episode_curriculum.quality_gate must be a mapping"
            )
        _keys(
            quality_gate,
            {
                "max_roll_pitch_rmse_rad",
                "max_yaw_rate_rmse_rad_s",
                "max_angular_rate_rms_rad_s",
            },
            "task.episode_curriculum.quality_gate",
        )

        def optional_positive_quality_limit(name: str) -> float | None:
            value = quality_gate.get(name)
            if value is None:
                return None
            return _positive_float_value(
                value, f"task.episode_curriculum.quality_gate.{name}"
            )

        task = replace(
            task,
            curriculum_durations_s=durations,
            curriculum_target_scales=scales,
            curriculum_success_fraction=_unit_float(
                curriculum_node, "success_fraction"
            ),
            curriculum_evaluation_episodes_per_env=_positive_float(
                curriculum_node, "evaluation_episodes_per_env"
            ),
            curriculum_consecutive_passes=_positive_int(
                curriculum_node, "consecutive_passes"
            ),
            curriculum_max_roll_pitch_rmse_rad=(
                optional_positive_quality_limit("max_roll_pitch_rmse_rad")
            ),
            curriculum_max_yaw_rate_rmse_rad_s=(
                optional_positive_quality_limit("max_yaw_rate_rmse_rad_s")
            ),
            curriculum_max_angular_rate_rms_rad_s=(
                optional_positive_quality_limit("max_angular_rate_rms_rad_s")
            ),
        )
    else:
        task = replace(
            task,
            curriculum_durations_s=(task.episode_duration_s,),
            curriculum_target_scales=(1.0,),
        )
    if task.attitude_source not in {"truth", "sensor"}:
        raise ConfigError("environment.observation_source must be truth or sensor")

    command_source = _virtual_pilot(_map(raw, "command_source"))
    if command_source.seed in {static.seed, dynamic.seed}:
        raise ConfigError("command_source.seed must differ from static/dynamic randomization seeds")
    control_contract = _control_contract(_map(raw, "control_contract"))
    checkpoint = _checkpoint_config(_map(raw, "checkpoint"), source)
    evaluation = _evaluation_config(_map(raw, "evaluation"), source)
    if evaluation.enabled:
        if checkpoint.interval_control_steps is None:
            raise ConfigError("periodic evaluation requires checkpoint.interval_control_steps")
        if (
            evaluation.interval_control_steps % checkpoint.interval_control_steps
            != 0
        ):
            raise ConfigError(
                "evaluation.interval_control_steps must be a multiple of checkpoint interval"
            )

    reward_node = _map(raw, "reward")
    context_fields = tuple(_sequence(_map(reward_node, "context"), "fields"))
    _validate_reward_context(context_fields)
    reward = RewardCalculatorConfig(
        calculator=_component(_map(reward_node, "calculator"), "reward.calculator"),
        context_fields=context_fields,
    )

    requested_algorithm_name = str(_map(raw, "algorithm").get("name", ""))
    model_node = _map(raw, "model")
    model_type = str(model_node.get("type", "gru_actor_critic"))
    if model_type == "gru_actor_critic":
        encoder = _map(model_node, "encoder")
        recurrent = _map(model_node, "recurrent")
        actor_head = _map(model_node, "actor_head")
        encoder_sizes = tuple(
            _positive_int_value(v, "model.encoder.hidden_sizes")
            for v in _sequence(encoder, "hidden_sizes")
        )
        recurrent_sizes_raw = recurrent.get("hidden_sizes")
        if recurrent_sizes_raw is None:
            recurrent_sizes = (_positive_int(recurrent, "hidden_size"),)
        else:
            if not isinstance(recurrent_sizes_raw, (list, tuple)):
                raise ConfigError("model.recurrent.hidden_sizes must be a sequence")
            recurrent_sizes = tuple(
                _positive_int_value(v, "model.recurrent.hidden_sizes")
                for v in recurrent_sizes_raw
            )
            if not recurrent_sizes:
                raise ConfigError("model.recurrent.hidden_sizes must not be empty")
            if len(set(recurrent_sizes)) != 1:
                raise ConfigError(
                    "stacked GRU layers must currently use one common hidden size"
                )
        model_kwargs: dict[str, Any] = {}
        if requested_algorithm_name == "sac":
            policy_distribution = _map(model_node, "policy_distribution")
            learnable_action_std = policy_distribution.get(
                "learnable_action_std", True
            )
            if not isinstance(learnable_action_std, bool):
                raise ConfigError(
                    "model.policy_distribution.learnable_action_std must be boolean"
                )
            model_kwargs = {
                "sac_initial_action_std": tuple(
                    _positive_float_value(
                        value, "model.policy_distribution.initial_action_std"
                    )
                    for value in _sequence(
                        policy_distribution, "initial_action_std"
                    )
                ),
                "sac_minimum_action_std": tuple(
                    _positive_float_value(
                        value, "model.policy_distribution.minimum_action_std"
                    )
                    for value in _sequence(
                        policy_distribution, "minimum_action_std"
                    )
                ),
                "sac_maximum_action_std": tuple(
                    _positive_float_value(
                        value, "model.policy_distribution.maximum_action_std"
                    )
                    for value in _sequence(
                        policy_distribution, "maximum_action_std"
                    )
                ),
                "sac_learnable_action_std": learnable_action_std,
            }
        model = ModelConfig(
            encoder_sizes=encoder_sizes,
            hidden_size=recurrent_sizes[0],
            actor_head_size=_positive_int_value(
                _sequence(actor_head, "hidden_sizes")[0],
                "model.actor_head.hidden_sizes",
            ),
            architecture="gru",
            recurrent_layers=len(recurrent_sizes),
            **model_kwargs,
        )
        if requested_algorithm_name == "sac":
            if not (
                len(model.sac_initial_action_std)
                == len(model.sac_minimum_action_std)
                == len(model.sac_maximum_action_std)
                == 4
            ):
                raise ConfigError(
                    "SAC policy distribution std lists must each contain 4 actions"
                )
            if any(
                not minimum < initial < maximum
                for minimum, initial, maximum in zip(
                    model.sac_minimum_action_std,
                    model.sac_initial_action_std,
                    model.sac_maximum_action_std,
                    strict=True,
                )
            ):
                raise ConfigError(
                    "each SAC action std must satisfy minimum < initial < maximum"
                )
    elif model_type == "mlp_actor_critic":
        hidden_sizes = tuple(
            _positive_int_value(v, "model.hidden_sizes")
            for v in _sequence(model_node, "hidden_sizes")
        )
        if requested_algorithm_name == "sac":
            policy_distribution = _map(model_node, "policy_distribution")
            learnable_action_std = policy_distribution.get(
                "learnable_action_std", True
            )
            if not isinstance(learnable_action_std, bool):
                raise ConfigError(
                    "model.policy_distribution.learnable_action_std must be boolean"
                )
            model = ModelConfig(
                encoder_sizes=hidden_sizes,
                hidden_size=hidden_sizes[-1],
                actor_head_size=hidden_sizes[-1],
                architecture="mlp",
                sac_initial_action_std=tuple(
                    _positive_float_value(
                        value, "model.policy_distribution.initial_action_std"
                    )
                    for value in _sequence(
                        policy_distribution, "initial_action_std"
                    )
                ),
                sac_minimum_action_std=tuple(
                    _positive_float_value(
                        value, "model.policy_distribution.minimum_action_std"
                    )
                    for value in _sequence(
                        policy_distribution, "minimum_action_std"
                    )
                ),
                sac_maximum_action_std=tuple(
                    _positive_float_value(
                        value, "model.policy_distribution.maximum_action_std"
                    )
                    for value in _sequence(
                        policy_distribution, "maximum_action_std"
                    )
                ),
                sac_learnable_action_std=learnable_action_std,
            )
            if not (
                len(model.sac_initial_action_std)
                == len(model.sac_minimum_action_std)
                == len(model.sac_maximum_action_std)
                == 4
            ):
                raise ConfigError(
                    "SAC policy distribution std lists must each contain 4 actions"
                )
            if any(
                not minimum < initial < maximum
                for minimum, initial, maximum in zip(
                    model.sac_minimum_action_std,
                    model.sac_initial_action_std,
                    model.sac_maximum_action_std,
                    strict=True,
                )
            ):
                raise ConfigError(
                    "each SAC action std must satisfy minimum < initial < maximum"
                )
        else:
            exploration_node = _map(model_node, "exploration")
            model = ModelConfig(
                encoder_sizes=hidden_sizes,
                hidden_size=hidden_sizes[-1],
                actor_head_size=hidden_sizes[-1],
                architecture="mlp",
                initial_action_std=tuple(
                    _positive_float_value(v, "model.exploration.initial_action_std")
                    for v in _sequence(exploration_node, "initial_action_std")
                ),
                final_action_std=tuple(
                    _positive_float_value(v, "model.exploration.final_action_std")
                    for v in _sequence(exploration_node, "final_action_std")
                ),
                exploration_decay_control_steps=_positive_int(
                    exploration_node, "decay_control_steps"
                ),
            )
            if (
                len(model.initial_action_std) != 4
                or len(model.final_action_std) != 4
            ):
                raise ConfigError("MLP exploration std lists must each contain 4 actions")
            if any(
                final > initial
                for initial, final in zip(
                    model.initial_action_std, model.final_action_std, strict=True
                )
            ):
                raise ConfigError("final_action_std must not exceed initial_action_std")
    else:
        raise ConfigError(
            "model.type must be gru_actor_critic or mlp_actor_critic"
        )

    algorithm = _map(raw, "algorithm")
    algorithm_name = str(algorithm.get("name", ""))
    ppo: PPOConfig | None = None
    sac: SACConfig | None = None
    if algorithm_name in {"ppo", "recurrent_ppo"}:
        expected_algorithm = "recurrent_ppo" if model.architecture == "gru" else "ppo"
        if algorithm_name != expected_algorithm:
            raise ConfigError(
                f"algorithm.name must be {expected_algorithm} for {model.architecture} policy"
            )
        optimizer = _map(algorithm, "optimizer")
        minibatches = _positive_int(algorithm, "minibatches")
        sequence_length = _positive_int(algorithm, "sequence_length")
        chunks = run.parallel_count * (run.rollout_steps // sequence_length)
        ppo = PPOConfig(
            gamma=_unit_float(algorithm, "gamma"),
            gae_lambda=_unit_float(algorithm, "gae_lambda"),
            clip_epsilon=_positive_float(algorithm, "clip_epsilon"),
            entropy_coefficient=_nonnegative_float(algorithm, "entropy_coefficient"),
            value_coefficient=_nonnegative_float(algorithm, "value_coefficient"),
            max_grad_norm=_positive_float(algorithm, "actor_max_grad_norm"),
            learning_rate=_positive_float(optimizer, "actor_learning_rate"),
            epochs=_positive_int(algorithm, "epochs_per_rollout"),
            sequence_length=sequence_length,
            sequences_per_minibatch=max(1, chunks // minibatches),
            critic_max_grad_norm=_positive_float(algorithm, "critic_max_grad_norm"),
            critic_learning_rate=_positive_float(optimizer, "critic_learning_rate"),
        )
        if run.rollout_steps % ppo.sequence_length:
            raise ConfigError(
                "collector.control_steps_per_rollout must be divisible by algorithm.sequence_length"
            )
        if model.architecture == "mlp" and ppo.sequence_length != 1:
            raise ConfigError("MLP PPO requires algorithm.sequence_length=1")
    elif algorithm_name == "sac":
        optimizer = _map(algorithm, "optimizer")
        replay = _map(algorithm, "replay")
        target_update = _map(algorithm, "target_update")
        entropy = _map(algorithm, "entropy")
        policy_anchor_raw = algorithm.get("policy_anchor", {})
        if not isinstance(policy_anchor_raw, Mapping):
            raise ConfigError("algorithm.policy_anchor must be a mapping")
        policy_anchor_weight = _nonnegative_float(
            policy_anchor_raw, "weight"
        )
        policy_anchor_max_action_deviation = _nonnegative_float(
            policy_anchor_raw, "max_action_deviation"
        )
        if policy_anchor_weight > 0:
            if model.architecture != "mlp":
                raise ConfigError(
                    "algorithm.policy_anchor currently requires an MLP SAC policy"
                )
            if checkpoint.resume_from is None:
                raise ConfigError(
                    "algorithm.policy_anchor requires checkpoint.resume.from"
                )
        if not bool(entropy.get("automatic", True)):
            raise ConfigError("SAC currently requires automatic entropy tuning")
        target_entropy_raw = entropy.get("target_entropy", "auto")
        target_entropy: float | str
        if isinstance(target_entropy_raw, str):
            if target_entropy_raw != "auto":
                raise ConfigError("algorithm.entropy.target_entropy must be auto or a number")
            target_entropy = "auto"
        else:
            target_entropy = _finite_float_value(
                target_entropy_raw, "algorithm.entropy.target_entropy"
            )
        min_alpha_raw = entropy.get("min_alpha")
        max_alpha_raw = entropy.get("max_alpha")
        min_alpha = (
            None
            if min_alpha_raw is None
            else _positive_float_value(min_alpha_raw, "algorithm.entropy.min_alpha")
        )
        max_alpha = (
            None
            if max_alpha_raw is None
            else _positive_float_value(max_alpha_raw, "algorithm.entropy.max_alpha")
        )
        if min_alpha is not None and max_alpha is not None and min_alpha >= max_alpha:
            raise ConfigError("algorithm.entropy.min_alpha must be below max_alpha")
        replay_capacity = _positive_int(replay, "capacity_transitions")
        replay_batch_size = _positive_int(replay, "batch_size")
        warmup_transitions = _positive_int(replay, "warmup_transitions")
        if replay_batch_size > replay_capacity:
            raise ConfigError("SAC replay batch_size must not exceed capacity")
        if warmup_transitions > replay_capacity:
            raise ConfigError("SAC warmup_transitions must not exceed replay capacity")
        replay_sequence_length = _positive_int_value(
            replay.get("sequence_length", 1),
            "algorithm.replay.sequence_length",
        )
        replay_burn_in_steps = _nonnegative_int_value(
            replay.get("burn_in_steps", 0),
            "algorithm.replay.burn_in_steps",
        )
        replay_sample_length = replay_burn_in_steps + replay_sequence_length
        if model.architecture == "mlp":
            if replay_sequence_length != 1:
                raise ConfigError("MLP SAC requires replay.sequence_length=1")
            if replay_burn_in_steps:
                raise ConfigError("MLP SAC requires replay.burn_in_steps=0")
        if model.architecture == "gru":
            if replay_sequence_length < 2:
                raise ConfigError(
                    "GRU SAC requires replay.sequence_length of at least 2"
                )
            if replay_batch_size % replay_sample_length:
                raise ConfigError(
                    "GRU SAC replay.batch_size must be divisible by "
                    "burn_in_steps + sequence_length"
                )
            if run.rollout_steps < replay_sample_length:
                raise ConfigError(
                    "GRU SAC collector rollout must be at least "
                    "burn_in_steps + sequence_length"
                )
        sac = SACConfig(
            gamma=_unit_float(algorithm, "gamma"),
            n_step_return=_positive_int_value(
                algorithm.get("n_step_return", 1),
                "algorithm.n_step_return",
            ),
            replay_capacity=replay_capacity,
            replay_batch_size=replay_batch_size,
            warmup_transitions=warmup_transitions,
            updates_per_collection=_positive_int(algorithm, "updates_per_collection"),
            actor_learning_rate=_positive_float(optimizer, "actor_learning_rate"),
            critic_learning_rate=_positive_float(optimizer, "critic_learning_rate"),
            alpha_learning_rate=_positive_float(optimizer, "alpha_learning_rate"),
            actor_max_grad_norm=_positive_float(algorithm, "actor_max_grad_norm"),
            critic_max_grad_norm=_positive_float(algorithm, "critic_max_grad_norm"),
            target_tau=_unit_float(target_update, "tau"),
            target_update_interval=_positive_int(target_update, "interval_updates"),
            initial_alpha=_positive_float(entropy, "initial_alpha"),
            target_entropy=target_entropy,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            replay_sequence_length=replay_sequence_length,
            replay_burn_in_steps=replay_burn_in_steps,
            critic_pretraining_updates=_nonnegative_int(
                algorithm, "critic_pretraining_updates"
            ),
            actor_update_interval=_positive_int_value(
                algorithm.get("actor_update_interval", 1),
                "algorithm.actor_update_interval",
            ),
            policy_anchor_weight=policy_anchor_weight,
            policy_anchor_max_action_deviation=(
                policy_anchor_max_action_deviation
            ),
        )
    else:
        raise ConfigError("algorithm.name must be ppo, recurrent_ppo, or sac")

    return ExperimentConfig(
        schema_version=2,
        name=name,
        entrypoint=entrypoint,
        seeds=seeds,
        simulator_factory=str(environment.get("factory", "simenv:SimulationEnvironment")),
        simulator_config=simulator,
        static_randomization=static,
        dynamic_randomization=dynamic,
        reward=reward,
        run=run,
        task=task,
        command_source=command_source,
        control_contract=control_contract,
        checkpoint=checkpoint,
        evaluation=evaluation,
        model=model,
        algorithm_name=algorithm_name,
        ppo=ppo,
        sac=sac,
        source_path=source,
        raw=raw,
    )


def override_run_config(
    config: ExperimentConfig,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> ExperimentConfig:
    """只应用 CLI 白名单运行覆盖，并同步 resolved config/fingerprint 输入。"""

    run = config.run
    raw = copy.deepcopy(dict(config.raw))
    raw_run = raw.get("run")
    if not isinstance(raw_run, dict):
        raise ConfigError("resolved run configuration must be mutable mapping data")
    if device is not None:
        try:
            torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ConfigError(f"invalid run device override: {device}") from exc
        run = replace(run, device=device)
        raw_run["device"] = device
    if output_root is not None:
        resolved_output = Path(output_root).expanduser().resolve()
        run = replace(run, output_root=resolved_output)
        raw_run["output_root"] = str(resolved_output)
    checkpoint = config.checkpoint
    if resume_from is not None:
        resolved_resume = Path(resume_from).expanduser().resolve()
        if not resolved_resume.is_file():
            raise ConfigError(f"resume checkpoint does not exist: {resolved_resume}")
        checkpoint = replace(checkpoint, resume_from=resolved_resume, resume_mode="exact")
        raw_checkpoint = raw.setdefault("checkpoint", {})
        if not isinstance(raw_checkpoint, dict):
            raise ConfigError("resolved checkpoint configuration must be a mapping")
        raw_checkpoint["resume"] = {"from": str(resolved_resume), "mode": "exact"}
    return replace(config, run=run, checkpoint=checkpoint, raw=raw)


def exact_resume_config_sha256(config: ExperimentConfig | Mapping[str, Any]) -> str:
    """计算精确续训兼容摘要，忽略只影响运行长度/产物位置的字段。"""

    raw = config.raw if isinstance(config, ExperimentConfig) else config
    normalized = copy.deepcopy(dict(raw))
    run = normalized.get("run", {})
    if isinstance(run, dict):
        run.pop("total_control_steps", None)
        run.pop("output_root", None)
    checkpoint = normalized.get("checkpoint")
    if isinstance(checkpoint, dict):
        # checkpoint 调度和来源只影响产物，不改变数值训练语义。
        normalized.pop("checkpoint", None)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _component(node: Mapping[str, Any], path: str) -> ComponentConfig:
    type_name = str(node.get("type", "")).strip()
    if not type_name:
        raise ConfigError(f"{path}.type must not be empty")
    params = node.get("params", {})
    if not isinstance(params, Mapping):
        raise ConfigError(f"{path}.params must be a mapping")
    return ComponentConfig(type_name, str(node.get("version", "1")), params)


def _checkpoint_config(node: Mapping[str, Any], source: Path) -> CheckpointConfig:
    _keys(
        node,
        {
            "resume",
            "interval_control_steps",
            "keep_last",
            "minimum_free_space_bytes",
        },
        "checkpoint",
    )
    interval_value = node.get("interval_control_steps")
    interval = (
        None
        if interval_value is None
        else _positive_int_value(interval_value, "checkpoint.interval_control_steps")
    )
    keep_last = _positive_int_value(node.get("keep_last", 3), "checkpoint.keep_last")
    minimum_free_space_bytes = _nonnegative_int_value(
        node.get("minimum_free_space_bytes", 2 * 1024 * 1024 * 1024),
        "checkpoint.minimum_free_space_bytes",
    )
    resume = node.get("resume")
    if resume is None:
        return CheckpointConfig(
            None,
            "exact",
            interval,
            keep_last,
            minimum_free_space_bytes,
        )
    if not isinstance(resume, Mapping):
        raise ConfigError("checkpoint.resume must be a mapping")
    _keys(resume, {"from", "mode"}, "checkpoint.resume")
    mode = str(resume.get("mode", "exact"))
    if mode not in {"exact", "policy"}:
        raise ConfigError(
            "checkpoint.resume.mode must be exact or policy"
        )
    value = resume.get("from")
    if value is None or value == "":
        return CheckpointConfig(
            None,
            mode,
            interval,
            keep_last,
            minimum_free_space_bytes,
        )
    path = (source.parent / str(value)).resolve()
    if not path.is_file():
        raise ConfigError(f"resume checkpoint does not exist: {path}")
    return CheckpointConfig(
        path,
        mode,
        interval,
        keep_last,
        minimum_free_space_bytes,
    )


def _evaluation_config(
    node: Mapping[str, Any], source: Path
) -> EvaluationConfig:
    if not node:
        return EvaluationConfig(False, None, None)
    _keys(node, {"enabled", "interval_control_steps", "suite_path"}, "evaluation")
    enabled = bool(node.get("enabled", False))
    if not enabled:
        return EvaluationConfig(False, None, None)
    interval = _positive_int(node, "interval_control_steps")
    suite_path = (source.parent / str(node.get("suite_path", ""))).resolve()
    if not suite_path.is_file():
        raise ConfigError(f"evaluation suite does not exist: {suite_path}")
    return EvaluationConfig(True, interval, suite_path)


def _virtual_pilot(node: Mapping[str, Any]) -> VirtualPilotConfig:
    _keys(node, {"type", "version", "seed", "params"}, "command_source")
    type_name = str(node.get("type", ""))
    if type_name != "flight_train.commands:VirtualPilotCommandSource":
        raise ConfigError("command_source.type must be the registered VirtualPilotCommandSource")
    version = str(node.get("version", ""))
    if version != "2":
        raise ConfigError("command_source.version must be 2 for incremental height PI")
    params = _map(node, "params")
    _keys(params, {"throttle", "sticks"}, "command_source.params")
    throttle = _map(params, "throttle")
    sticks = _map(params, "sticks")
    _keys(throttle, {"minimum", "maximum", "spool", "height_controller", "slew_rate"}, "command_source.params.throttle")
    _keys(
        sticks,
        {"roll", "pitch", "yaw", "target_sampling", "reset"},
        "command_source.params.sticks",
    )
    spool = _map(throttle, "spool")
    height = _map(throttle, "height_controller")
    slew = _map(throttle, "slew_rate")
    roll = _map(sticks, "roll")
    pitch = _map(sticks, "pitch")
    yaw = _map(sticks, "yaw")
    sampling = _map(sticks, "target_sampling")
    hold = _map(sampling, "hold_duration_s")
    reset = _map(sticks, "reset")
    _keys(spool, {"duration_s", "target_range"}, "command_source.params.throttle.spool")
    _keys(height, {"observation_source", "target_m", "initial_throttle_range", "proportional_gain", "integral_gain", "error_limit_m"}, "command_source.params.throttle.height_controller")
    _keys(slew, {"rise_per_s", "fall_per_s"}, "command_source.params.throttle.slew_rate")
    _keys(roll, {"mode", "limit_rad", "time_constant_s"}, "command_source.params.sticks.roll")
    _keys(pitch, {"mode", "limit_rad", "time_constant_s"}, "command_source.params.sticks.pitch")
    _keys(yaw, {"mode", "limit_rad_s", "time_constant_s"}, "command_source.params.sticks.yaw")
    _keys(sampling, {"distribution", "center_exponent", "hold_duration_s"}, "command_source.params.sticks.target_sampling")
    _keys(hold, {"range"}, "command_source.params.sticks.target_sampling.hold_duration_s")
    _keys(reset, {"filtered_stick", "initial_target_scale"}, "command_source.params.sticks.reset")
    if str(roll.get("mode")) != "angle" or str(pitch.get("mode")) != "angle":
        raise ConfigError("virtual pilot roll/pitch modes must be angle")
    if str(yaw.get("mode")) != "rate":
        raise ConfigError("virtual pilot yaw mode must be rate")
    if str(sampling.get("distribution")) != "centered":
        raise ConfigError("virtual pilot target sampling distribution must be centered")
    if str(reset.get("filtered_stick")) != "zero":
        raise ConfigError("virtual pilot reset.filtered_stick must be zero")
    if str(height.get("observation_source")) != "truth":
        raise ConfigError("virtual pilot height_controller.observation_source must be truth")

    minimum = _unit_float(throttle, "minimum")
    maximum = _unit_float(throttle, "maximum")
    if minimum >= maximum:
        raise ConfigError("virtual pilot throttle minimum must be below maximum")
    spool_range = _bounded_pair(spool, "target_range", minimum, maximum)
    initial_throttle_range = _bounded_pair(
        height, "initial_throttle_range", minimum, maximum
    )
    hold_range = _bounded_pair(hold, "range", 0.0, float("inf"), strict_low=True)
    initial_scale = _unit_float(reset, "initial_target_scale")
    return VirtualPilotConfig(
        type=type_name,
        version=version,
        seed=_nonnegative_int(node, "seed"),
        throttle_minimum=minimum,
        throttle_maximum=maximum,
        spool_duration_s=_nonnegative_float(spool, "duration_s"),
        spool_target_range=spool_range,
        height_target_m=_finite_float(height, "target_m"),
        height_initial_throttle_range=initial_throttle_range,
        height_proportional_gain=_nonnegative_float(height, "proportional_gain"),
        height_integral_gain=_nonnegative_float(height, "integral_gain"),
        height_error_limit_m=_positive_float(height, "error_limit_m"),
        throttle_rise_rate_per_s=_positive_float(slew, "rise_per_s"),
        throttle_fall_rate_per_s=_positive_float(slew, "fall_per_s"),
        max_roll_rad=_nonnegative_float(roll, "limit_rad"),
        max_pitch_rad=_nonnegative_float(pitch, "limit_rad"),
        max_yaw_rate_rad_s=_nonnegative_float(yaw, "limit_rad_s"),
        roll_time_constant_s=_positive_float(roll, "time_constant_s"),
        pitch_time_constant_s=_positive_float(pitch, "time_constant_s"),
        yaw_time_constant_s=_positive_float(yaw, "time_constant_s"),
        center_exponent=_positive_float(sampling, "center_exponent"),
        hold_duration_range_s=hold_range,
        initial_target_scale=initial_scale,
    )


def _control_contract(node: Mapping[str, Any]) -> ControlContractConfig:
    _keys(
        node,
        {
            "version",
            "observation_profile",
            "observation_history",
            "policy_action",
            "external_action",
            "simulator_command",
            "action_transform",
        },
        "control_contract",
    )
    policy = tuple(str(v) for v in _sequence(_map(node, "policy_action"), "fields"))
    external = tuple(str(v) for v in _sequence(_map(node, "external_action"), "fields"))
    simulator = tuple(str(v) for v in _sequence(_map(node, "simulator_command"), "fields"))
    expected_policy = ("lower_motor", "servo_1", "servo_2", "servo_3")
    expected_external = ("upper_motor",)
    expected_simulator = ("upper_motor", *expected_policy)
    if policy != expected_policy or external != expected_external or simulator != expected_simulator:
        raise ConfigError("control_contract fields must match self_stabilize_v1 ownership and order")
    version = str(node.get("version", ""))
    profile = str(node.get("observation_profile", ""))
    if version != "self_stabilize_v1" or profile != "attitude_self_stabilize_21d_v3":
        raise ConfigError("unsupported self-stabilize control/observation contract")
    history_raw = node.get("observation_history", {})
    if not isinstance(history_raw, Mapping):
        raise ConfigError("control_contract.observation_history must be a mapping")
    _keys(
        history_raw,
        {
            "mode",
            "frames",
            "stride_steps",
            "dense_action_steps",
            "sparse_physical_frames",
            "sparse_physical_stride_steps",
        },
        "control_contract.observation_history",
    )
    history_mode = str(history_raw.get("mode", "uniform"))
    if history_mode == "uniform":
        unexpected = {
            "dense_action_steps",
            "sparse_physical_frames",
            "sparse_physical_stride_steps",
        }.intersection(history_raw)
        if unexpected:
            raise ConfigError(
                "uniform observation history does not accept "
                f"{sorted(unexpected)}"
            )
        history_frames = _positive_int_value(
            history_raw.get("frames", 1),
            "control_contract.observation_history.frames",
        )
        history_stride_steps = _positive_int_value(
            history_raw.get("stride_steps", 1),
            "control_contract.observation_history.stride_steps",
        )
        if history_frames == 1 and history_stride_steps != 1:
            raise ConfigError(
                "single-frame observation history requires stride_steps=1"
            )
        dense_action_steps = 0
        sparse_physical_frames = 0
        sparse_physical_stride_steps = 1
    elif history_mode == "multirate_actuator":
        unexpected = {"frames", "stride_steps"}.intersection(history_raw)
        if unexpected:
            raise ConfigError(
                "multirate_actuator observation history does not accept "
                f"{sorted(unexpected)}"
            )
        dense_action_steps = _positive_int_value(
            history_raw.get("dense_action_steps"),
            "control_contract.observation_history.dense_action_steps",
        )
        sparse_physical_frames = _positive_int_value(
            history_raw.get("sparse_physical_frames"),
            "control_contract.observation_history.sparse_physical_frames",
        )
        sparse_physical_stride_steps = _positive_int_value(
            history_raw.get("sparse_physical_stride_steps"),
            "control_contract.observation_history."
            "sparse_physical_stride_steps",
        )
        history_frames = 1
        history_stride_steps = 1
    else:
        raise ConfigError(
            "control_contract.observation_history.mode must be uniform or "
            "multirate_actuator"
        )
    transform = _map(node, "action_transform")
    _keys(
        transform,
        {"type", "trim_command", "residual_scale"},
        "control_contract.action_transform",
    )
    if str(transform.get("type")) != "residual_around_trim":
        raise ConfigError(
            "control_contract.action_transform.type must be residual_around_trim"
        )
    trim = tuple(
        float(value) for value in _sequence(transform, "trim_command")
    )
    scale = tuple(
        _positive_float_value(
            value, "control_contract.action_transform.residual_scale"
        )
        for value in _sequence(transform, "residual_scale")
    )
    if len(trim) != 4 or len(scale) != 4:
        raise ConfigError("trim_command and residual_scale must each contain 4 values")
    lower = (0.0, -1.0, -1.0, -1.0)
    upper = (1.0, 1.0, 1.0, 1.0)
    if any(
        not math.isfinite(center)
        or center - radius < low
        or center + radius > high
        for center, radius, low, high in zip(
            trim, scale, lower, upper, strict=True
        )
    ):
        raise ConfigError(
            "trim_command ± residual_scale must stay inside simulator action bounds"
        )
    return ControlContractConfig(
        version=version,
        observation_profile=profile,
        observation_history_mode=history_mode,
        observation_history_frames=history_frames,
        observation_history_stride_steps=history_stride_steps,
        observation_history_dense_action_steps=dense_action_steps,
        observation_history_sparse_physical_frames=sparse_physical_frames,
        observation_history_sparse_physical_stride_steps=(
            sparse_physical_stride_steps
        ),
        policy_action_fields=policy,
        external_action_fields=external,
        simulator_command_fields=simulator,
        policy_action_trim=trim,
        policy_action_residual_scale=scale,
    )


def _bounded_pair(
    node: Mapping[str, Any],
    key: str,
    minimum: float,
    maximum: float,
    *,
    strict_low: bool = False,
) -> tuple[float, float]:
    value = node.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{key} must be [low, high]")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low) or not math.isfinite(high):
        raise ConfigError(f"{key} bounds must be finite")
    low_valid = low > minimum if strict_low else low >= minimum
    if not low_valid or high < low or high > maximum:
        raise ConfigError(f"{key} must be ordered inside the configured bounds")
    return low, high


def _randomization(node: Mapping[str, Any], path: str, *, static: bool) -> RandomizationConfig:
    seed = _nonnegative_int(node, "seed")
    scope = str(node.get("scope", "episode" if static else "simulator"))
    parameters = _map(node, "parameters")
    checked: dict[str, Mapping[str, Any]] = {}
    for name, value in parameters.items():
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}.parameters.{name} must be a mapping")
        allowed_keys = (
            {"distribution", "mode", "range", "unit", "seed_stream"}
            if static
            else {
                "distribution", "range", "stddev", "valid_range",
                "on_out_of_range", "unit", "seed_stream",
            }
        )
        _keys(value, allowed_keys, f"{path}.parameters.{name}")
        if "distribution" not in value or "seed_stream" not in value:
            raise ConfigError(
                f"{path}.parameters.{name} requires distribution and seed_stream"
            )
        distribution = str(value["distribution"])
        allowed = {"uniform"} if static else {"normal", "truncated_normal", "uniform"}
        if distribution not in allowed:
            raise ConfigError(f"unsupported distribution at {path}.parameters.{name}: {distribution}")
        if static and str(value.get("mode", "absolute")) not in {"absolute", "relative"}:
            raise ConfigError(f"{path}.parameters.{name}.mode must be absolute or relative")
        if distribution == "uniform" and "range" not in value:
            raise ConfigError(f"{path}.parameters.{name}.range is required for uniform")
        if distribution in {"normal", "truncated_normal"} and "stddev" not in value:
            raise ConfigError(f"{path}.parameters.{name}.stddev is required for normal")
        if distribution == "uniform":
            bounds = torch.as_tensor(value["range"])
            if bounds.ndim >= 1 and bounds.shape[-1] == 2:
                low, high = bounds[..., 0], bounds[..., 1]
            else:
                raise ConfigError(
                    f"{path}.parameters.{name}.range must end with [low, high]"
                )
            if not bool(torch.isfinite(bounds).all()) or bool((high < low).any()):
                raise ConfigError(f"{path}.parameters.{name}.range must be finite and ordered")
        if distribution in {"normal", "truncated_normal"}:
            stddev = torch.as_tensor(value["stddev"])
            if not bool(torch.isfinite(stddev).all()) or bool((stddev < 0).any()):
                raise ConfigError(f"{path}.parameters.{name}.stddev must be finite and nonnegative")
        checked[str(name)] = value
    return RandomizationConfig(seed, scope, checked)


def _validate_reward_context(fields: tuple[Any, ...]) -> None:
    allowed_sources = {"task", "policy", "adapter", "truth", "sensor", "train_randomizer", "simenv"}
    names: set[str] = set()
    for index, field in enumerate(fields):
        path = f"reward.context.fields[{index}]"
        if not isinstance(field, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        unknown = set(field) - {"name", "source", "alias", "dtype", "shape", "unit", "visibility"}
        if unknown:
            raise ConfigError(f"unknown fields at {path}: {sorted(unknown)}")
        name = str(field.get("name", "")).strip()
        source = str(field.get("source", "")).strip()
        if not name or source not in allowed_sources:
            raise ConfigError(f"{path} requires a name and a supported source")
        alias = str(field.get("alias", name))
        if alias in names:
            raise ConfigError(f"duplicate reward context output name: {alias}")
        names.add(alias)
        if source == "truth" and field.get("visibility", "reward_only") != "reward_only":
            raise ConfigError(f"{path}: truth fields must use visibility=reward_only")


def _parameter_range(node: Mapping[str, Any]) -> tuple[Any, Any]:
    """把标量或逐元素 ``[...,2]`` 随机化范围拆成上下界。"""

    value = node["range"]
    bounds = torch.as_tensor(value)
    if bounds.shape == (2,):
        return value[0], value[1]
    if bounds.ndim >= 2 and bounds.shape[-1] == 2:
        return bounds[..., 0].tolist(), bounds[..., 1].tolist()
    raise ConfigError("parameter range must end with [low, high]")


def _load(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"experiment config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ConfigError("configuration root must be a mapping")
    return value


def _map(node: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = node.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _sequence(node: Mapping[str, Any], key: str) -> list[Any]:
    value = node.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{key} must be a non-empty list")
    return value


def _keys(node: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(node) - allowed
    if unknown:
        raise ConfigError(f"unknown fields at {path}: {sorted(unknown)}")


def _positive_int(node: Mapping[str, Any], key: str) -> int:
    return _positive_int_value(node.get(key), key)


def _positive_int_value(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(node: Mapping[str, Any], key: str) -> int:
    return _nonnegative_int_value(node.get(key), key)


def _nonnegative_int_value(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{key} must be a nonnegative integer")
    return value


def _positive_float(node: Mapping[str, Any], key: str) -> float:
    return _positive_float_value(node.get(key, 0.0), key)


def _positive_float_value(raw: Any, key: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _finite_float(node: Mapping[str, Any], key: str) -> float:
    return _finite_float_value(node.get(key, float("nan")), key)


def _finite_float_value(raw: Any, key: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ConfigError(f"{key} must be finite")
    return value


def _nonnegative_float(node: Mapping[str, Any], key: str) -> float:
    value = float(node.get(key, 0.0))
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"{key} must be nonnegative")
    return value


def _unit_float(node: Mapping[str, Any], key: str) -> float:
    return _unit_float_value(node.get(key, -1.0), key)


def _unit_float_value(raw: Any, key: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ConfigError(f"{key} must be in [0, 1]")
    return value
