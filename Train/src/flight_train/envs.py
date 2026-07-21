from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from .config import TaskConfig
from .core import EnvSpec, assert_tensor_on
from .tasks import AttitudeTrackingTask


class SimEnvAdapter:
    """TensorDict adapter around SimEnv. Control-step tensors stay on device."""

    observation_fields = (
        "attitude_q_wb",
        "angular_velocity_b",
        "acceleration_b",
        "motor_speed",
        "target_attitude_q_wb",
        "previous_action",
    )

    def __init__(
        self,
        simulator: Any,
        simulator_config: str | Path,
        task_config: TaskConfig,
        device: torch.device,
        dtype: torch.dtype,
        *,
        seed: int,
    ) -> None:
        self.simulator = simulator
        self.simulator_config = Path(simulator_config).resolve()
        self.device = device
        self.dtype = dtype
        self.batch_size = simulator.parallel_count
        physics_hz, control_hz = _load_timing(self.simulator_config)
        self._spec = EnvSpec(
            parallel_count=self.batch_size,
            observation_dim=21,
            action_dim=5,
            device=self.device,
            dtype=self.dtype,
            physics_hz=physics_hz,
            control_hz=control_hz,
            observation_fields=self.observation_fields,
        )
        self.task_config = task_config
        self.task = AttitudeTrackingTask(task_config, self.batch_size, self.device, self.dtype, seed)
        self.previous_action = torch.zeros((self.batch_size, 5), device=self.device, dtype=self.dtype)
        self.episode_id = torch.zeros(self.batch_size, device=self.device, dtype=torch.int64)
        self.episode_step = torch.zeros(self.batch_size, device=self.device, dtype=torch.int64)
        episode_steps = task_config.episode_duration_s * self._spec.control_hz
        rounded = round(episode_steps)
        if abs(episode_steps - rounded) > 1e-6:
            raise ValueError("task episode duration must be an integer number of control steps")
        self.max_episode_steps = rounded
        self._closed = False

    @classmethod
    def create(
        cls,
        simulator_config: str | Path,
        parallel_count: int,
        device: torch.device | str,
        dtype: torch.dtype,
        task_config: TaskConfig,
        *,
        seed: int,
    ) -> "SimEnvAdapter":
        from simenv import SimulationEnvironment

        resolved_device = torch.device(device)
        simulator = SimulationEnvironment.create(
            simulator_config, parallel_count, resolved_device, dtype
        )
        return cls(
            simulator,
            simulator_config,
            task_config,
            resolved_device,
            dtype,
            seed=seed,
        )

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    def reset(self, mask: torch.Tensor | None = None) -> TensorDict:
        if mask is None:
            mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        assert_tensor_on(mask, device=self.device, dtype=torch.bool, shape=(self.batch_size,), name="reset mask")
        self.simulator.reset(mask, self.simulator_config)
        self.task.reset(mask)
        self.previous_action = torch.where(mask[:, None], torch.zeros_like(self.previous_action), self.previous_action)
        self.episode_step = torch.where(mask, torch.zeros_like(self.episode_step), self.episode_step)
        self.episode_id = self.episode_id + mask.to(torch.int64)
        observation, _attitude, _rate = self._observation()
        return TensorDict(
            {
                "observation": observation,
                "is_init": mask[:, None],
                "episode_id": self.episode_id.clone(),
                "episode_step": self.episode_step.clone(),
            },
            batch_size=[self.batch_size],
            device=self.device,
        )

    @torch.no_grad()
    def step(self, standard_action: torch.Tensor) -> TensorDict:
        assert_tensor_on(
            standard_action,
            device=self.device,
            dtype=self.dtype,
            shape=(self.batch_size, 5),
            name="standard_action",
        )
        command = self.action_to_command(standard_action)
        result = self.simulator.advance(command)
        observation_before_reset, attitude, angular_velocity = self._observation()
        transition = self.task.transition(
            attitude, angular_velocity, standard_action, self.previous_action
        )
        self.episode_step = self.episode_step + 1
        valid = result.valid[:, None]
        truncated = self.episode_step[:, None] >= self.max_episode_steps
        terminated = transition.terminated
        reset_mask = (terminated | truncated | ~valid).squeeze(-1)

        # SimEnv performs the sparse GPU-mask reset. Its metadata/logging path may
        # synchronize on reset boundaries; rollout and learning tensors do not move.
        self.simulator.reset(reset_mask, self.simulator_config)
        self.task.reset(reset_mask)
        self.episode_id = self.episode_id + reset_mask.to(torch.int64)
        self.episode_step = torch.where(reset_mask, torch.zeros_like(self.episode_step), self.episode_step)
        self.previous_action = torch.where(reset_mask[:, None], torch.zeros_like(standard_action), standard_action)
        observation_after_reset, _attitude, _rate = self._observation()
        observation = torch.where(reset_mask[:, None], observation_after_reset, observation_before_reset)

        info = dict(transition.info)
        info["sim.error_code"] = result.error_code[:, None]
        info["action.command"] = command
        return TensorDict(
            {
                "observation": observation,
                "reward": transition.reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": terminated | truncated | ~valid,
                "valid": valid,
                "is_init": reset_mask[:, None],
                "episode_id": self.episode_id.clone(),
                "episode_step": self.episode_step.clone(),
                "info": TensorDict(info, batch_size=[self.batch_size], device=self.device),
            },
            batch_size=[self.batch_size],
            device=self.device,
        )

    def _observation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        truth = self.simulator.observe(
            "truth",
            ("attitude_q_wb", "angular_velocity_b", "linear_acceleration_n", "motor_speed"),
        )
        sensor = self.simulator.observe("sensor")
        if self.task_config.attitude_source == "truth":
            attitude = truth.values["attitude_q_wb"]
        else:
            try:
                attitude = sensor.values["attitude_q_wb"]
            except KeyError as exc:
                raise KeyError("sensor attitude requested but SimEnv exposes no attitude_q_wb sensor") from exc
        angular_velocity = sensor.values.get("gyro", truth.values["angular_velocity_b"])
        acceleration = sensor.values.get("accelerometer", truth.values["linear_acceleration_n"])
        motor_speed = sensor.values.get("motor_speed", truth.values["motor_speed"])
        observation = torch.cat(
            (
                attitude,
                angular_velocity,
                acceleration,
                motor_speed,
                self.task.target_attitude,
                self.previous_action,
            ),
            dim=-1,
        )
        if observation.shape != (self.batch_size, 21):
            raise RuntimeError(f"observation contract produced {tuple(observation.shape)}, expected {(self.batch_size, 21)}")
        return observation, attitude, angular_velocity

    @staticmethod
    def action_to_command(standard_action: torch.Tensor) -> torch.Tensor:
        motors = (standard_action[..., :2].clamp(-1.0, 1.0) + 1.0) * 0.5
        servos = standard_action[..., 2:].clamp(-1.0, 1.0)
        return torch.cat((motors, servos), dim=-1)

    def close(self) -> None:
        if not self._closed:
            self.simulator.close()
            self._closed = True


def _load_timing(path: Path) -> tuple[int, int]:
    """Read public structural timing fields without touching SimEnv internals."""
    text = path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML simulator config requires PyYAML") from exc
        config = yaml.safe_load(text)
    timing = config["timing"]
    physics_hz = timing["physics_hz"]["value"]
    control_hz = timing["control_hz"]["value"]
    if (
        isinstance(physics_hz, bool)
        or not isinstance(physics_hz, int)
        or isinstance(control_hz, bool)
        or not isinstance(control_hz, int)
        or physics_hz <= 0
        or control_hz <= 0
        or physics_hz % control_hz
    ):
        raise ValueError("simulator timing must contain positive integral physics/control frequencies")
    return physics_hz, control_hz
