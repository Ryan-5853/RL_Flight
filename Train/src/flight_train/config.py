from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RewardConfig:
    attitude_weight: float
    angular_rate_weight: float
    action_rate_weight: float
    saturation_weight: float
    alive_bonus: float


@dataclass(frozen=True)
class TaskConfig:
    episode_duration_s: float
    max_target_tilt_rad: float
    max_target_yaw_rad: float
    terminate_tilt_rad: float
    terminate_angular_rate_rad_s: float
    attitude_source: str
    reward: RewardConfig


@dataclass(frozen=True)
class ModelConfig:
    encoder_sizes: tuple[int, ...]
    hidden_size: int
    actor_head_size: int


@dataclass(frozen=True)
class PPOConfig:
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    entropy_coefficient: float
    value_coefficient: float
    max_grad_norm: float
    learning_rate: float
    epochs: int
    sequence_length: int
    sequences_per_minibatch: int


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
    simulator_config: Path
    run: RunConfig
    task: TaskConfig
    model: ModelConfig
    ppo: PPOConfig
    source_path: Path
    raw: Mapping[str, Any]

    @property
    def torch_dtype(self) -> torch.dtype:
        values = {"float32": torch.float32, "float64": torch.float64}
        try:
            return values[self.run.dtype]
        except KeyError as exc:
            raise ConfigError("run.dtype must be float32 or float64") from exc


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    raw = _load(source)
    _keys(raw, {"schema_version", "experiment", "run", "environment", "task", "model", "algorithm"}, "root")
    if raw.get("schema_version") != 1:
        raise ConfigError("schema_version must equal 1")
    experiment = _map(raw, "experiment")
    run_node = _map(raw, "run")
    environment = _map(raw, "environment")
    task_node = _map(raw, "task")
    reward_node = _map(task_node, "reward")
    model_node = _map(raw, "model")
    algorithm = _map(raw, "algorithm")
    _keys(experiment, {"name"}, "experiment")
    _keys(
        run_node,
        {
            "device", "dtype", "parallel_count", "rollout_steps",
            "total_control_steps", "seed", "output_root",
            "allow_unimplemented_simulator",
        },
        "run",
    )
    _keys(environment, {"config_path"}, "environment")
    _keys(
        task_node,
        {
            "episode_duration_s", "max_target_tilt_rad", "max_target_yaw_rad",
            "terminate_tilt_rad", "terminate_angular_rate_rad_s",
            "attitude_source", "reward",
        },
        "task",
    )
    _keys(
        reward_node,
        {
            "attitude_weight", "angular_rate_weight", "action_rate_weight",
            "saturation_weight", "alive_bonus",
        },
        "task.reward",
    )
    _keys(model_node, {"encoder_sizes", "hidden_size", "actor_head_size"}, "model")
    _keys(
        algorithm,
        {
            "name", "gamma", "gae_lambda", "clip_epsilon",
            "entropy_coefficient", "value_coefficient", "max_grad_norm",
            "learning_rate", "epochs", "sequence_length",
            "sequences_per_minibatch",
        },
        "algorithm",
    )
    if algorithm.get("name") != "recurrent_ppo":
        raise ConfigError("algorithm.name must be recurrent_ppo in framework version 1")

    run = RunConfig(
        device=str(run_node.get("device", "cuda:0")),
        dtype=str(run_node.get("dtype", "float32")),
        parallel_count=_positive_int(run_node, "parallel_count"),
        rollout_steps=_positive_int(run_node, "rollout_steps"),
        total_control_steps=_positive_int(run_node, "total_control_steps"),
        seed=_nonnegative_int(run_node, "seed"),
        output_root=(source.parent / str(run_node.get("output_root", "../../runs"))).resolve(),
        allow_unimplemented_simulator=bool(run_node.get("allow_unimplemented_simulator", False)),
    )
    task = TaskConfig(
        episode_duration_s=_positive_float(task_node, "episode_duration_s"),
        max_target_tilt_rad=_nonnegative_float(task_node, "max_target_tilt_rad"),
        max_target_yaw_rad=_nonnegative_float(task_node, "max_target_yaw_rad"),
        terminate_tilt_rad=_positive_float(task_node, "terminate_tilt_rad"),
        terminate_angular_rate_rad_s=_positive_float(task_node, "terminate_angular_rate_rad_s"),
        attitude_source=str(task_node.get("attitude_source", "truth")),
        reward=RewardConfig(
            attitude_weight=_nonnegative_float(reward_node, "attitude_weight"),
            angular_rate_weight=_nonnegative_float(reward_node, "angular_rate_weight"),
            action_rate_weight=_nonnegative_float(reward_node, "action_rate_weight"),
            saturation_weight=_nonnegative_float(reward_node, "saturation_weight"),
            alive_bonus=float(reward_node.get("alive_bonus", 0.0)),
        ),
    )
    encoder_sizes = tuple(
        _positive_int_value(v, "model.encoder_sizes")
        for v in model_node.get("encoder_sizes", [128, 128])
    )
    if not encoder_sizes:
        raise ConfigError("model.encoder_sizes must not be empty")
    model = ModelConfig(
        encoder_sizes=encoder_sizes,
        hidden_size=_positive_int(model_node, "hidden_size"),
        actor_head_size=_positive_int(model_node, "actor_head_size"),
    )
    ppo = PPOConfig(
        gamma=_unit_float(algorithm, "gamma"),
        gae_lambda=_unit_float(algorithm, "gae_lambda"),
        clip_epsilon=_positive_float(algorithm, "clip_epsilon"),
        entropy_coefficient=_nonnegative_float(algorithm, "entropy_coefficient"),
        value_coefficient=_nonnegative_float(algorithm, "value_coefficient"),
        max_grad_norm=_positive_float(algorithm, "max_grad_norm"),
        learning_rate=_positive_float(algorithm, "learning_rate"),
        epochs=_positive_int(algorithm, "epochs"),
        sequence_length=_positive_int(algorithm, "sequence_length"),
        sequences_per_minibatch=_positive_int(algorithm, "sequences_per_minibatch"),
    )
    if run.rollout_steps % ppo.sequence_length:
        raise ConfigError("run.rollout_steps must be divisible by algorithm.sequence_length")
    simulator = (source.parent / str(environment.get("config_path", ""))).resolve()
    if not simulator.is_file():
        raise ConfigError(f"simulator config does not exist: {simulator}")
    if task.attitude_source not in {"truth", "sensor"}:
        raise ConfigError("task.attitude_source must be truth or sensor")
    name = str(experiment.get("name", "")).strip()
    if not name:
        raise ConfigError("experiment.name must not be empty")
    return ExperimentConfig(
        schema_version=1,
        name=name,
        simulator_config=simulator,
        run=run,
        task=task,
        model=model,
        ppo=ppo,
        source_path=source,
        raw=raw,
    )


def _load(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"experiment config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError("YAML requires PyYAML; JSON configuration remains supported") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ConfigError("configuration root must be a mapping")
    return value


def _map(node: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = node.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
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
    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{key} must be a nonnegative integer")
    return value


def _positive_float(node: Mapping[str, Any], key: str) -> float:
    value = float(node.get(key, 0.0))
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _nonnegative_float(node: Mapping[str, Any], key: str) -> float:
    value = float(node.get(key, 0.0))
    if value < 0:
        raise ConfigError(f"{key} must be nonnegative")
    return value


def _unit_float(node: Mapping[str, Any], key: str) -> float:
    value = float(node.get(key, -1.0))
    if not 0 <= value <= 1:
        raise ConfigError(f"{key} must be in [0, 1]")
    return value
