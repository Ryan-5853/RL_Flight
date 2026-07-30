from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from tensordict import TensorDict

from .collector import TensorDictRolloutCollector
from .core import BatchedControlEnv
from .models import ActorCritic
from .commands import VirtualPilotCommandSource, VirtualPilotSnapshot
from .config import ExperimentConfig
from .envs import SimEnvAdapter
from .math import euler_to_quaternion, normalize_quaternion, quaternion_geodesic_angle
from .models import build_actor_critic, build_sac_actor_critic
from .randomization import StaticRandomizer
from .recording import load_checkpoint
from .registry import ComponentRegistry


@contextmanager
def isolated_torch_rng(device: torch.device) -> Iterator[None]:
    """隔离评估消耗的 CPU/CUDA 全局 RNG，不改变后续训练随机序列。"""

    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices, enabled=True):
        yield


@torch.no_grad()
def collect_evaluation_rollout(
    env: BatchedControlEnv,
    model,
    steps: int,
):
    """在独立评估环境中采样，并恢复模型模式与训练侧全局 RNG。"""

    if steps <= 0:
        raise ValueError("evaluation steps must be positive")
    actor_training = model.actor.training
    critic_training = (
        model.critic.training if model.critic is not None else None
    )
    try:
        model.actor.eval()
        if model.critic is not None:
            model.critic.eval()
        with isolated_torch_rng(env.spec.device):
            return TensorDictRolloutCollector(
                env, model, steps, deterministic=True
            ).collect()
    finally:
        model.actor.train(actor_training)
        if model.critic is not None:
            model.critic.train(critic_training)


@dataclass(frozen=True)
class FixedScenario:
    """一个完全确定的评测科目。"""

    name: str
    type: str
    duration_s: float
    fixed_euler_rad: tuple[float, float, float]
    target_velocity_n_m_s: tuple[float, float, float]
    circle_radius_m: float
    circle_period_s: float
    roll_amplitude_rad: float
    pitch_amplitude_rad: float


@dataclass(frozen=True)
class ScoreLimits:
    attitude_rmse_bad_deg: float
    position_rmse_bad_m: float
    velocity_rmse_bad_m_s: float
    action_rms_bad: float
    action_delta_rms_bad: float
    saturation_fraction_bad: float
    response_time_bad_s: float
    settling_error_deg: float
    settling_window_s: float
    phase_lag_max_s: float
    yaw_rate_rmse_bad_rad_s: float = 2.0


@dataclass(frozen=True)
class CheckpointSelectionCriteria:
    minimum_hover_survival_s: float = 0.0
    maximum_hover_roll_pitch_rmse_deg: float = float("inf")
    maximum_hover_yaw_rate_rmse_rad_s: float | None = float("inf")


@dataclass(frozen=True)
class FixedEvaluationSuite:
    schema_version: int
    name: str
    seed: int
    parallel_count: int
    scenarios: tuple[FixedScenario, ...]
    limits: ScoreLimits
    score_weights: Mapping[str, float]
    self_stabilize_tracking: bool
    yaw_rate_tracking_weight: float
    checkpoint_selection: CheckpointSelectionCriteria
    output_root: Path
    source_path: Path
    raw: Mapping[str, Any]


class ScriptedEvaluationCommandSource:
    """保留高度增量式 PI，只把随机摇杆替换成固定可复现科目命令。"""

    def __init__(
        self,
        base: VirtualPilotCommandSource,
        scenario: FixedScenario,
        control_hz: int,
    ) -> None:
        self.base = base
        self.scenario = scenario
        self.control_hz = control_hz
        self.elapsed_steps = torch.zeros(
            base.batch_size, dtype=torch.int64, device=base.device
        )
        self._apply_target()

    @property
    def upper_throttle(self) -> torch.Tensor:
        return self.base.upper_throttle

    @property
    def config(self):
        return self.base.config

    @property
    def target_attitude(self) -> torch.Tensor:
        return self.base.target_attitude

    @property
    def desired_yaw_rate(self) -> torch.Tensor:
        return self.base.desired_yaw_rate

    def set_curriculum_scale(self, scale: float) -> None:
        self.base.set_curriculum_scale(scale)

    def reset(self, mask: torch.Tensor) -> None:
        self.base.reset(mask)
        self.elapsed_steps = torch.where(
            mask, torch.zeros_like(self.elapsed_steps), self.elapsed_steps
        )
        self._apply_target()

    def step(
        self, height_m: torch.Tensor, active_mask: torch.Tensor | None = None
    ) -> None:
        self.base.step(height_m, active_mask)
        if active_mask is None:
            active_mask = torch.ones_like(self.elapsed_steps, dtype=torch.bool)
        self.elapsed_steps = self.elapsed_steps + active_mask.to(torch.int64)
        self._apply_target()

    def snapshot(self) -> VirtualPilotSnapshot:
        snapshot = self.base.snapshot()
        return VirtualPilotSnapshot(
            upper_throttle=snapshot.upper_throttle,
            target_attitude_q_wb=self.base.target_attitude,
            stick_target=self.base.stick_target,
            filtered_stick=self.base.filtered_stick,
            throttle_target=snapshot.throttle_target,
            height_target=snapshot.height_target,
            height_error=snapshot.height_error,
        )

    def info(self) -> TensorDict:
        return self.base.info()

    def _apply_target(self) -> None:
        time_s = self.elapsed_steps.to(self.base.dtype) / float(self.control_hz)
        roll, pitch, yaw = _scenario_euler(self.scenario, time_s)
        scripted = torch.stack((roll, pitch, yaw), dim=-1)
        self.base.stick_target.copy_(scripted)
        self.base.filtered_stick.copy_(scripted)
        self.base.target_yaw.copy_(yaw)
        self.base.target_attitude = euler_to_quaternion(roll, pitch, yaw)


def load_fixed_evaluation_suite(path: str | Path) -> FixedEvaluationSuite:
    """读取固定评测 YAML/JSON，并执行严格字段和数值校验。"""

    source = Path(path).expanduser().resolve()
    raw = _load_mapping(source)
    _only_keys(
        raw,
        {
            "schema_version",
            "name",
            "seed",
            "parallel_count",
            "output_root",
            "scenarios",
            "scoring",
            "checkpoint_selection",
        },
        "evaluation suite",
    )
    schema_version = int(raw.get("schema_version", -1))
    if schema_version not in {1, 2, 3}:
        raise ValueError("evaluation schema_version must equal 1, 2, or 3")
    scenarios_node = raw.get("scenarios")
    if not isinstance(scenarios_node, list) or not scenarios_node:
        raise ValueError("evaluation scenarios must be a non-empty list")
    scenarios = tuple(_parse_scenario(item, index) for index, item in enumerate(scenarios_node))
    names = [item.name for item in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("evaluation scenario names must be unique")
    required_types = {"hover", "constant_translation", "circle"}
    if {item.type for item in scenarios} != required_types:
        raise ValueError(f"evaluation scenarios must contain exactly {sorted(required_types)}")

    scoring = _mapping(raw.get("scoring"), "scoring")
    scoring_fields = {"limits", "weights"}
    if schema_version >= 3:
        scoring_fields.add("self_stabilize_tracking")
    _only_keys(scoring, scoring_fields, "scoring")
    limits_node = _mapping(scoring.get("limits"), "scoring.limits")
    legacy_limit_names = tuple(ScoreLimits.__dataclass_fields__)[:-1]
    limit_names = (
        tuple(ScoreLimits.__dataclass_fields__)
        if schema_version >= 2
        else legacy_limit_names
    )
    _only_keys(limits_node, set(limit_names), "scoring.limits")
    limits_values = {name: _positive_float(limits_node.get(name), f"scoring.limits.{name}") for name in limit_names}
    limits = ScoreLimits(**limits_values)
    weights = _mapping(scoring.get("weights"), "scoring.weights")
    expected_weights = {"survival", "tracking", "action", "response"}
    _only_keys(weights, expected_weights, "scoring.weights")
    parsed_weights = {name: _nonnegative_float(weights.get(name), f"scoring.weights.{name}") for name in expected_weights}
    total_weight = sum(parsed_weights.values())
    if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
        raise ValueError("scoring.weights must sum to 1")
    if schema_version >= 3:
        stabilization_scoring = _mapping(
            scoring.get("self_stabilize_tracking"),
            "scoring.self_stabilize_tracking",
        )
        _only_keys(
            stabilization_scoring,
            {"yaw_rate_weight"},
            "scoring.self_stabilize_tracking",
        )
        yaw_rate_tracking_weight = _nonnegative_float(
            stabilization_scoring.get("yaw_rate_weight"),
            "scoring.self_stabilize_tracking.yaw_rate_weight",
        )
        if yaw_rate_tracking_weight > 1.0:
            raise ValueError(
                "scoring.self_stabilize_tracking.yaw_rate_weight must not exceed 1"
            )
    else:
        yaw_rate_tracking_weight = 0.25
    output_root = Path(str(raw.get("output_root", "evaluation_runs")))
    if not output_root.is_absolute():
        output_root = (source.parent / output_root).resolve()
    parallel_count = _positive_int(raw.get("parallel_count"), "parallel_count")
    seed = _nonnegative_int(raw.get("seed"), "seed")
    selection_node = raw.get("checkpoint_selection", {})
    if not isinstance(selection_node, Mapping):
        raise ValueError("checkpoint_selection must be a mapping")
    selection_names = {
        "minimum_hover_survival_s",
        "maximum_hover_roll_pitch_rmse_deg",
        "maximum_hover_yaw_rate_rmse_rad_s",
    }
    _only_keys(selection_node, selection_names, "checkpoint_selection")
    required_selection_names = (
        selection_names
        if schema_version <= 2
        else selection_names - {"maximum_hover_yaw_rate_rmse_rad_s"}
    )
    if (schema_version >= 2 or selection_node) and not (
        required_selection_names.issubset(selection_node)
    ):
        missing = sorted(required_selection_names - set(selection_node))
        raise ValueError(
            f"checkpoint_selection is missing required fields: {missing}"
        )
    checkpoint_selection = (
        CheckpointSelectionCriteria()
        if not selection_node
        else CheckpointSelectionCriteria(
            minimum_hover_survival_s=_nonnegative_float(
                selection_node["minimum_hover_survival_s"],
                "checkpoint_selection.minimum_hover_survival_s",
            ),
            maximum_hover_roll_pitch_rmse_deg=_positive_float(
                selection_node["maximum_hover_roll_pitch_rmse_deg"],
                "checkpoint_selection.maximum_hover_roll_pitch_rmse_deg",
            ),
            maximum_hover_yaw_rate_rmse_rad_s=(
                _positive_float(
                    selection_node["maximum_hover_yaw_rate_rmse_rad_s"],
                    "checkpoint_selection.maximum_hover_yaw_rate_rmse_rad_s",
                )
                if "maximum_hover_yaw_rate_rmse_rad_s" in selection_node
                else None
            ),
        )
    )
    return FixedEvaluationSuite(
        schema_version=schema_version,
        name=str(raw.get("name") or "fixed_attitude_v1"),
        seed=seed,
        parallel_count=parallel_count,
        scenarios=scenarios,
        limits=limits,
        score_weights=parsed_weights,
        self_stabilize_tracking=schema_version >= 2,
        yaw_rate_tracking_weight=yaw_rate_tracking_weight,
        checkpoint_selection=checkpoint_selection,
        output_root=output_root,
        source_path=source,
        raw=raw,
    )


@torch.no_grad()
def run_fixed_evaluation(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    suite: FixedEvaluationSuite,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """加载策略权重，在三个固定科目上运行确定性评测并保存报告。"""

    device = torch.device(config.run.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    checkpoint_source = Path(checkpoint_path).expanduser().resolve()
    state = load_checkpoint(checkpoint_source)
    torch.manual_seed(suite.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(suite.seed)

    # 先用一个短生命周期环境确定观测/动作契约；每个科目再创建全新环境，
    # 防止前一科目的动力学、传感器历史或高度 PI 状态泄漏。
    observation_dim = config.control_contract.observation_dim
    model = (
        build_sac_actor_critic(
            observation_dim, 4, config.model, device, config.torch_dtype
        )
        if config.algorithm_name == "sac"
        else build_actor_critic(
            observation_dim, 4, config.model, device, config.torch_dtype
        )
    )
    model.actor.load_state_dict(state["actor"], strict=True)
    model.actor.eval()
    if model.critic is not None:
        model.critic.eval()

    root = Path(output_root).expanduser().resolve() if output_root else suite.output_root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evaluation_id = f"{stamp}_{checkpoint_source.stem}_{uuid.uuid4().hex[:8]}"
    directory = root / suite.name / evaluation_id
    directory.mkdir(parents=True, exist_ok=False)
    archived_suite = directory / f"suite{suite.source_path.suffix or '.yaml'}"
    shutil.copyfile(suite.source_path, archived_suite)
    report: dict[str, Any] = {
        "schema_version": suite.schema_version,
        "evaluation_id": evaluation_id,
        "suite": suite.name,
        "suite_config": archived_suite.name,
        "suite_config_sha256": _sha256(suite.source_path),
        "checkpoint": str(checkpoint_source),
        "checkpoint_sha256": _sha256(checkpoint_source),
        "checkpoint_global_control_steps": int(state.get("global_control_steps", -1)),
        "experiment": config.name,
        "device": str(device),
        "dtype": str(config.torch_dtype),
        "parallel_count": suite.parallel_count,
        "seed": suite.seed,
        "termination": {
            "max_tilt_rad": config.task.terminate_tilt_rad,
            "max_angular_rate_rad_s": (
                config.task.terminate_angular_rate_rad_s
            ),
            "angular_rate_axes": config.task.terminate_angular_rate_axes,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "score_weights": dict(suite.score_weights),
        "yaw_rate_tracking_weight": suite.yaw_rate_tracking_weight,
        "scenarios": {},
    }
    scenario_scores: list[float] = []
    try:
        for scenario in suite.scenarios:
            env = _create_evaluation_environment(config, suite, scenario, device)
            try:
                result, trajectory = _run_scenario(
                    env,
                    model,
                    scenario,
                    suite.limits,
                    suite.score_weights,
                    self_stabilize_tracking=suite.self_stabilize_tracking,
                    yaw_rate_tracking_weight=suite.yaw_rate_tracking_weight,
                )
                report["scenarios"][scenario.name] = result
                scenario_scores.append(float(result["total_score"]))
                torch.save(trajectory, directory / f"trajectory_{scenario.name}.pt")
            finally:
                env.close()
        report["total_score"] = sum(scenario_scores) / len(scenario_scores)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        (directory / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return {"evaluation_directory": directory, "report": report}


def _create_evaluation_environment(
    config: ExperimentConfig,
    suite: FixedEvaluationSuite,
    scenario: FixedScenario,
    device: torch.device,
) -> SimEnvAdapter:
    reward_calculator = ComponentRegistry().build_reward(
        {
            "type": config.reward.calculator.type,
            "version": config.reward.calculator.version,
            "params": config.reward.calculator.params,
        }
    )
    # 固定基准评测禁用 Train 层静态/动态随机化；SimEnv 配置中的标称参数和
    # 确定性传感器随机流仍然生效。
    randomizer = StaticRandomizer((), suite.seed, device, config.torch_dtype)
    evaluation_duration = max(
        config.task.episode_duration_s, scenario.duration_s + 1.0
    )
    task = replace(
        config.task,
        episode_duration_s=evaluation_duration,
        curriculum_durations_s=(evaluation_duration,),
        curriculum_target_scales=(1.0,),
    )
    pilot_config = replace(config.command_source, seed=suite.seed)
    env = SimEnvAdapter.create(
        config.simulator_config,
        suite.parallel_count,
        device,
        config.torch_dtype,
        task,
        reward_calculator=reward_calculator,
        static_randomizer=randomizer,
        dynamic_randomization=None,
        dynamic_seed=None,
        reward_context_fields=(),
        command_source_config=pilot_config,
        control_contract_config=config.control_contract,
    )
    env.command_source = ScriptedEvaluationCommandSource(
        env.command_source, scenario, env.spec.control_hz
    )
    return env


def _run_scenario(
    env: SimEnvAdapter,
    model: ActorCritic,
    scenario: FixedScenario,
    limits: ScoreLimits,
    score_weights: Mapping[str, float],
    *,
    self_stabilize_tracking: bool = False,
    yaw_rate_tracking_weight: float = 0.25,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    batch = env.spec.parallel_count
    device = env.spec.device
    dtype = env.spec.dtype
    steps = round(scenario.duration_s * env.spec.control_hz)
    initial = env.reset()
    observation = initial["observation"]
    is_init = initial["is_init"]
    hidden = None
    alive = torch.ones(batch, dtype=torch.bool, device=device)
    survival_s = torch.full((batch,), scenario.duration_s, dtype=dtype, device=device)
    initial_position = env.simulator.observe("truth", ("position_n",)).values["position_n"]
    traces: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "time_s", "alive", "action", "position_n", "velocity_n",
            "attitude_q_wb", "target_position_n", "target_velocity_n",
            "target_attitude_q_wb", "target_euler_rad", "actual_euler_rad",
            "yaw_rate_error_rad_s",
        )
    }

    for step in range(steps):
        time_s = torch.full((batch,), step / env.spec.control_hz, device=device, dtype=dtype)
        target_euler = torch.stack(_scenario_euler(scenario, time_s), dim=-1)
        target_attitude = euler_to_quaternion(
            target_euler[:, 0], target_euler[:, 1], target_euler[:, 2]
        )
        target_position, target_velocity = _scenario_trajectory(
            scenario, time_s + 1.0 / env.spec.control_hz, initial_position
        )
        action, hidden = model.forward_step(observation, hidden, is_init)
        transition = env.step(action)
        truth = env.simulator.observe(
            "truth", ("position_n", "velocity_n", "attitude_q_wb")
        ).values
        done = transition["done"].squeeze(-1)
        newly_done = alive & done
        survival_s = torch.where(
            newly_done,
            torch.full_like(survival_s, (step + 1) / env.spec.control_hz),
            survival_s,
        )
        sample_alive = alive & ~done
        actual_euler = _quaternion_to_euler(truth["attitude_q_wb"])
        values = {
            "time_s": time_s,
            "alive": sample_alive,
            "action": action,
            "position_n": truth["position_n"],
            "velocity_n": truth["velocity_n"],
            "attitude_q_wb": truth["attitude_q_wb"],
            "target_position_n": target_position,
            "target_velocity_n": target_velocity,
            "target_attitude_q_wb": target_attitude,
            "target_euler_rad": target_euler,
            "actual_euler_rad": actual_euler,
            "yaw_rate_error_rad_s": transition[
                ("info", "yaw_rate_error_rad_s")
            ].squeeze(-1),
        }
        for name, value in values.items():
            traces[name].append(value.detach().clone())
        alive = sample_alive
        observation = transition["observation"]
        is_init = transition["is_init"]
        if not bool(alive.any().item()):
            break

    trajectory = {name: torch.stack(values).cpu() for name, values in traces.items()}
    metrics = _score_trajectory(
        trajectory,
        survival_s.cpu(),
        scenario,
        limits,
        env.spec.control_hz,
        score_weights,
        self_stabilize_tracking=self_stabilize_tracking,
        yaw_rate_tracking_weight=yaw_rate_tracking_weight,
    )
    return metrics, trajectory


def _score_trajectory(
    trajectory: Mapping[str, torch.Tensor],
    survival_s: torch.Tensor,
    scenario: FixedScenario,
    limits: ScoreLimits,
    control_hz: int,
    score_weights: Mapping[str, float],
    *,
    self_stabilize_tracking: bool = False,
    yaw_rate_tracking_weight: float = 0.25,
) -> dict[str, Any]:
    alive = trajectory["alive"].bool()
    attitude_error_deg = torch.rad2deg(
        quaternion_geodesic_angle(
            trajectory["attitude_q_wb"], trajectory["target_attitude_q_wb"]
        ).squeeze(-1)
    )
    roll_pitch_error_rad = torch.atan2(
        torch.sin(
            trajectory["actual_euler_rad"][..., :2]
            - trajectory["target_euler_rad"][..., :2]
        ),
        torch.cos(
            trajectory["actual_euler_rad"][..., :2]
            - trajectory["target_euler_rad"][..., :2]
        ),
    )
    roll_pitch_error_deg = torch.rad2deg(
        torch.linalg.vector_norm(roll_pitch_error_rad, dim=-1)
    )
    yaw_rate_error = trajectory.get("yaw_rate_error_rad_s")
    if yaw_rate_error is None:
        yaw_rate_error = torch.zeros_like(attitude_error_deg)
    elif yaw_rate_error.ndim == alive.ndim + 1:
        yaw_rate_error = yaw_rate_error.squeeze(-1)
    yaw_rate_error = yaw_rate_error.abs()
    position_error = torch.linalg.vector_norm(
        trajectory["position_n"] - trajectory["target_position_n"], dim=-1
    )
    velocity_error = torch.linalg.vector_norm(
        trajectory["velocity_n"] - trajectory["target_velocity_n"], dim=-1
    )
    action_norm = torch.linalg.vector_norm(trajectory["action"], dim=-1) / math.sqrt(4.0)
    saturation = (trajectory["action"].abs() >= 0.95).to(torch.float32).mean(dim=-1)
    action_delta = torch.zeros_like(action_norm)
    if action_delta.shape[0] > 1:
        action_delta[1:] = torch.linalg.vector_norm(
            trajectory["action"][1:] - trajectory["action"][:-1], dim=-1
        ) / math.sqrt(4.0)
    attitude_rmse = _masked_rmse(attitude_error_deg, alive)
    attitude_p95 = _masked_quantile(attitude_error_deg, alive, 0.95)
    roll_pitch_rmse = _masked_rmse(roll_pitch_error_deg, alive)
    roll_pitch_p95 = _masked_quantile(roll_pitch_error_deg, alive, 0.95)
    yaw_rate_rmse = _masked_rmse(yaw_rate_error, alive)
    yaw_rate_p95 = _masked_quantile(yaw_rate_error, alive, 0.95)
    position_rmse = _masked_rmse(position_error, alive)
    position_p95 = _masked_quantile(position_error, alive, 0.95)
    velocity_rmse = _masked_rmse(velocity_error, alive)
    action_rms = _masked_rmse(action_norm, alive)
    action_peak = _masked_quantile(trajectory["action"].abs().amax(dim=-1), alive, 1.0)
    action_delta_rms = _masked_rmse(action_delta, alive)
    saturation_fraction = _masked_mean(saturation, alive)
    response_error_deg = (
        roll_pitch_error_deg if self_stabilize_tracking else attitude_error_deg
    )
    if scenario.type == "circle":
        response_s = _circle_phase_lag(
            trajectory["target_euler_rad"], trajectory["actual_euler_rad"], alive,
            control_hz, limits.phase_lag_max_s,
        )
    elif scenario.type == "hover":
        response_s = _recovery_time_after_peak(
            response_error_deg, alive, control_hz,
            limits.settling_error_deg, limits.settling_window_s,
            scenario.duration_s,
        )
    else:
        response_s = _settling_time(
            response_error_deg, alive, control_hz,
            limits.settling_error_deg, limits.settling_window_s,
            scenario.duration_s,
        )

    survival_score = 100.0 * (survival_s / scenario.duration_s).clamp(0.0, 1.0)
    attitude_score = _lower_is_better(attitude_rmse, limits.attitude_rmse_bad_deg)
    roll_pitch_score = _lower_is_better(
        roll_pitch_rmse, limits.attitude_rmse_bad_deg
    )
    yaw_rate_score = _lower_is_better(
        yaw_rate_rmse, limits.yaw_rate_rmse_bad_rad_s
    )
    inner_attitude_score = (
        (1.0 - yaw_rate_tracking_weight) * roll_pitch_score
        + yaw_rate_tracking_weight * yaw_rate_score
        if self_stabilize_tracking
        else attitude_score
    )
    position_score = _lower_is_better(position_rmse, limits.position_rmse_bad_m)
    velocity_score = _lower_is_better(velocity_rmse, limits.velocity_rmse_bad_m_s)
    if scenario.type == "hover":
        tracking_score = 0.6 * inner_attitude_score + 0.4 * position_score
    else:
        tracking_score = (
            0.5 * inner_attitude_score
            + 0.3 * position_score
            + 0.2 * velocity_score
        )
    action_score = (
        0.5 * _lower_is_better(action_rms, limits.action_rms_bad)
        + 0.3 * _lower_is_better(action_delta_rms, limits.action_delta_rms_bad)
        + 0.2 * _lower_is_better(saturation_fraction, limits.saturation_fraction_bad)
    )
    response_score = _lower_is_better(response_s, limits.response_time_bad_s)
    total_score = (
        score_weights["survival"] * survival_score
        + score_weights["tracking"] * tracking_score
        + score_weights["action"] * action_score
        + score_weights["response"] * response_score
    )
    # 每个指标保留逐实例值，汇总同时报告 mean/p95，便于观察批量噪声离散度。
    raw = {
        "survival_time_s": survival_s,
        "attitude_rmse_deg": attitude_rmse,
        "attitude_error_p95_deg": attitude_p95,
        "roll_pitch_rmse_deg": roll_pitch_rmse,
        "roll_pitch_error_p95_deg": roll_pitch_p95,
        "yaw_rate_rmse_rad_s": yaw_rate_rmse,
        "yaw_rate_error_p95_rad_s": yaw_rate_p95,
        "position_rmse_m": position_rmse,
        "position_error_p95_m": position_p95,
        "velocity_rmse_m_s": velocity_rmse,
        "action_rms": action_rms,
        "action_peak_abs": action_peak,
        "action_delta_rms": action_delta_rms,
        "action_saturation_fraction": saturation_fraction,
        "response_time_s": response_s,
    }
    scores = {
        "survival": survival_score,
        "tracking": tracking_score,
        "action": action_score,
        "response": response_score,
        "total": total_score,
    }
    return {
        "type": scenario.type,
        "duration_s": scenario.duration_s,
        "metrics": {name: _summarize(value) for name, value in raw.items()},
        "scores": {name: _summarize(value) for name, value in scores.items()},
        "total_score": float(total_score.mean().item()),
    }


def _scenario_euler(
    scenario: FixedScenario, time_s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scenario.type in {"hover", "constant_translation"}:
        return tuple(torch.full_like(time_s, value) for value in scenario.fixed_euler_rad)  # type: ignore[return-value]
    omega = 2.0 * math.pi / scenario.circle_period_s
    return (
        scenario.roll_amplitude_rad * torch.sin(omega * time_s),
        scenario.pitch_amplitude_rad * torch.cos(omega * time_s),
        torch.full_like(time_s, scenario.fixed_euler_rad[2]),
    )


def _scenario_trajectory(
    scenario: FixedScenario,
    time_s: torch.Tensor,
    origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scenario.type == "hover":
        return origin, torch.zeros_like(origin)
    if scenario.type == "constant_translation":
        velocity = torch.tensor(
            scenario.target_velocity_n_m_s, device=time_s.device, dtype=time_s.dtype
        ).expand_as(origin)
        return origin + velocity * time_s[:, None], velocity
    omega = 2.0 * math.pi / scenario.circle_period_s
    radius = scenario.circle_radius_m
    displacement = torch.stack(
        (
            radius * torch.sin(omega * time_s),
            radius * (1.0 - torch.cos(omega * time_s)),
            torch.zeros_like(time_s),
        ),
        dim=-1,
    )
    velocity = torch.stack(
        (
            radius * omega * torch.cos(omega * time_s),
            radius * omega * torch.sin(omega * time_s),
            torch.zeros_like(time_s),
        ),
        dim=-1,
    )
    return origin + displacement, velocity


def _quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square()))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    return torch.stack((roll, pitch, yaw), dim=-1)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=0)
    total = torch.where(mask, value, torch.zeros_like(value)).sum(dim=0)
    mean = total / count.clamp_min(1)
    return torch.where(count > 0, mean, torch.full_like(mean, float("inf")))


def _masked_rmse(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_masked_mean(value.square(), mask))


def _masked_quantile(
    value: torch.Tensor, mask: torch.Tensor, quantile: float
) -> torch.Tensor:
    result = torch.full((value.shape[1],), float("inf"), dtype=value.dtype)
    for batch_index in range(value.shape[1]):
        selected = value[:, batch_index][mask[:, batch_index]]
        if selected.numel() > 0:
            result[batch_index] = torch.quantile(selected, quantile)
    return result


def _lower_is_better(value: torch.Tensor, bad: float) -> torch.Tensor:
    return 100.0 * (1.0 - value / bad).clamp(0.0, 1.0)


def _settling_time(
    error_deg: torch.Tensor,
    alive: torch.Tensor,
    control_hz: int,
    threshold_deg: float,
    window_s: float,
    fallback_s: float,
) -> torch.Tensor:
    window = max(1, round(window_s * control_hz))
    result = torch.full((error_deg.shape[1],), fallback_s, dtype=error_deg.dtype)
    for batch_index in range(error_deg.shape[1]):
        good = (error_deg[:, batch_index] <= threshold_deg) & alive[:, batch_index]
        for start in range(max(0, good.numel() - window + 1)):
            if bool(good[start : start + window].all().item()):
                result[batch_index] = start / control_hz
                break
    return result


def _recovery_time_after_peak(
    error_deg: torch.Tensor,
    alive: torch.Tensor,
    control_hz: int,
    threshold_deg: float,
    window_s: float,
    fallback_s: float,
) -> torch.Tensor:
    """悬停从本次最大姿态扰动起算，避免起始水平状态虚假得到零响应时间。"""

    window = max(1, round(window_s * control_hz))
    result = torch.full((error_deg.shape[1],), fallback_s, dtype=error_deg.dtype)
    for batch_index in range(error_deg.shape[1]):
        valid_indices = torch.nonzero(alive[:, batch_index], as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            continue
        peak_relative = torch.argmax(error_deg[valid_indices, batch_index])
        peak = int(valid_indices[peak_relative].item())
        good = (error_deg[:, batch_index] <= threshold_deg) & alive[:, batch_index]
        last_start = good.numel() - window
        for start in range(peak, last_start + 1):
            if bool(good[start : start + window].all().item()):
                result[batch_index] = (start - peak) / control_hz
                break
    return result


def _circle_phase_lag(
    target: torch.Tensor,
    actual: torch.Tensor,
    alive: torch.Tensor,
    control_hz: int,
    max_lag_s: float,
) -> torch.Tensor:
    max_lag = min(round(max_lag_s * control_hz), target.shape[0] - 1)
    result = torch.full((target.shape[1],), max_lag_s, dtype=target.dtype)
    for batch_index in range(target.shape[1]):
        best_error = float("inf")
        best_lag = max_lag
        for lag in range(max_lag + 1):
            mask = alive[lag:, batch_index]
            if not bool(mask.any().item()):
                continue
            error = actual[lag:, batch_index, :2] - target[: target.shape[0] - lag, batch_index, :2]
            mse = float(error[mask].square().mean().item())
            if mse < best_error:
                best_error = mse
                best_lag = lag
        result[batch_index] = best_lag / control_hz
    return result


def _summarize(value: torch.Tensor) -> dict[str, Any]:
    flat = value.to(torch.float64).flatten()
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.5).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "per_instance": [float(item) for item in flat.tolist()],
    }


def _parse_scenario(node: Any, index: int) -> FixedScenario:
    value = _mapping(node, f"scenarios[{index}]")
    allowed = {
        "name", "type", "duration_s", "fixed_euler_deg", "target_velocity_n_m_s",
        "circle_radius_m", "circle_period_s", "roll_amplitude_deg", "pitch_amplitude_deg",
    }
    _only_keys(value, allowed, f"scenarios[{index}]")
    type_name = str(value.get("type"))
    if type_name not in {"hover", "constant_translation", "circle"}:
        raise ValueError(f"unsupported scenario type: {type_name}")
    euler_deg = _float_triplet(value.get("fixed_euler_deg", [0, 0, 0]), f"scenarios[{index}].fixed_euler_deg")
    velocity = _float_triplet(value.get("target_velocity_n_m_s", [0, 0, 0]), f"scenarios[{index}].target_velocity_n_m_s")
    circle_radius = _positive_float(value.get("circle_radius_m", 1.0), f"scenarios[{index}].circle_radius_m")
    circle_period = _positive_float(value.get("circle_period_s", 8.0), f"scenarios[{index}].circle_period_s")
    return FixedScenario(
        name=str(value.get("name") or type_name),
        type=type_name,
        duration_s=_positive_float(value.get("duration_s"), f"scenarios[{index}].duration_s"),
        fixed_euler_rad=tuple(math.radians(item) for item in euler_deg),
        target_velocity_n_m_s=velocity,
        circle_radius_m=circle_radius,
        circle_period_s=circle_period,
        roll_amplitude_rad=math.radians(_nonnegative_float(value.get("roll_amplitude_deg", 0.0), f"scenarios[{index}].roll_amplitude_deg")),
        pitch_amplitude_rad=math.radians(_nonnegative_float(value.get("pitch_amplitude_deg", 0.0), f"scenarios[{index}].pitch_amplitude_deg")),
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"evaluation configuration does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        value = yaml.safe_load(text)
    return _mapping(value, "evaluation root")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _only_keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _float_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
