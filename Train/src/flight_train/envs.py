from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from .config import ControlContractConfig, TaskConfig, VirtualPilotConfig
from .core import EnvSpec, assert_tensor_on
from .math import euler_to_quaternion, quaternion_to_euler
from .randomization import StaticRandomizer
from .rewards import RewardCalculator
from .registry import ComponentRegistry
from .tasks import AttitudeTrackingTask


class SimEnvAdapter:
    """将 SimEnv 适配为训练层约定的批量 TensorDict 环境。

    适配器负责观测拼接、动作域映射、任务奖励和 episode 生命周期；SimEnv
    只负责动力学、执行器与传感器推进。控制周期内的张量始终留在目标设备。

    观测固定为 21 维：目标相对当前姿态四元数 4、角速度 3、加速度 3、
    电机转速 2、舵机实际角 3、目标偏航角速度 1、上桨油门 1、
    上一策略动作 4。
    """

    observation_fields = (
        "relative_attitude_q_error",
        "angular_velocity_b",
        "acceleration_b",
        "motor_speed",
        "servo_angle",
        "desired_yaw_rate",
        "upper_throttle_command",
        "previous_policy_action",
    )
    acceleration_scale_m_s2 = 9.80665
    motor_speed_scale_rad_s = 1800.0

    def __init__(
        self,
        simulator: Any,
        simulator_config: str | Path,
        task_config: TaskConfig,
        device: torch.device,
        dtype: torch.dtype,
        *,
        reward_calculator: RewardCalculator,
        static_randomizer: StaticRandomizer,
        reward_context_fields: tuple[dict[str, Any], ...],
        command_source_config: VirtualPilotConfig,
        control_contract_config: ControlContractConfig,
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
            action_dim=4,
            device=self.device,
            dtype=self.dtype,
            physics_hz=physics_hz,
            control_hz=control_hz,
            observation_fields=self.observation_fields,
        )
        self.task_config = task_config
        self.task = AttitudeTrackingTask(
            task_config,
            self.batch_size,
            self.device,
            self.dtype,
            reward_calculator,
        )
        self.command_source = ComponentRegistry().build_command_source(
            command_source_config,
            self.batch_size,
            self.device,
            self.dtype,
            control_hz,
        )
        self.control_contract_config = control_contract_config
        self.curriculum_stage = 0
        self.curriculum_successes = 0
        self.curriculum_failures = 0
        self.curriculum_consecutive_passes = 0
        self.curriculum_last_success_fraction = 0.0
        self.command_source.set_curriculum_scale(
            task_config.curriculum_target_scales[0]
        )
        self.static_randomizer = static_randomizer
        # SimEnv 是所有物理标称值的唯一来源；Train 只持有随机化规则。
        self.static_randomizer.bind_baselines(self.simulator.parameters)
        self.reward_context_fields = reward_context_fields
        self.current_static_parameters = TensorDict(
            {}, batch_size=[self.batch_size], device=self.device
        )
        self.previous_action = torch.zeros((self.batch_size, 4), device=self.device, dtype=self.dtype)
        self.episode_id = torch.zeros(self.batch_size, device=self.device, dtype=torch.int64)
        self.episode_step = torch.zeros(self.batch_size, device=self.device, dtype=torch.int64)
        self.max_episode_steps = self._duration_to_steps(
            task_config.curriculum_durations_s[0]
        )
        self._closed = False

    def _duration_to_steps(self, duration_s: float) -> int:
        episode_steps = duration_s * self._spec.control_hz
        rounded = round(episode_steps)
        if abs(episode_steps - rounded) > 1e-6:
            raise ValueError(
                "task episode duration must be an integer number of control steps"
            )
        return rounded

    def update_episode_curriculum(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        *,
        allow_promotion: bool = True,
    ) -> dict[str, torch.Tensor]:
        """累计完整 episode 结果，并在允许时按连续达标次数晋级。"""

        self.curriculum_failures += int(terminated.sum().item())
        self.curriculum_successes += int(
            (truncated & ~terminated).sum().item()
        )
        completed = self.curriculum_successes + self.curriculum_failures
        required = max(
            1,
            round(
                self.batch_size
                * self.task_config.curriculum_evaluation_episodes_per_env
            ),
        )
        promoted = False
        if completed >= required:
            success_fraction = self.curriculum_successes / float(completed)
            self.curriculum_last_success_fraction = success_fraction
            if (
                allow_promotion
                and success_fraction
                >= self.task_config.curriculum_success_fraction
            ):
                self.curriculum_consecutive_passes += 1
            else:
                self.curriculum_consecutive_passes = 0
            self.curriculum_successes = 0
            self.curriculum_failures = 0
            if (
                self.curriculum_consecutive_passes
                >= self.task_config.curriculum_consecutive_passes
                and self.curriculum_stage + 1
                < len(self.task_config.curriculum_durations_s)
            ):
                self.curriculum_stage += 1
                self.curriculum_consecutive_passes = 0
                self.max_episode_steps = self._duration_to_steps(
                    self.task_config.curriculum_durations_s[
                        self.curriculum_stage
                    ]
                )
                self.command_source.set_curriculum_scale(
                    self.task_config.curriculum_target_scales[
                        self.curriculum_stage
                    ]
                )
                promoted = True
        scalar = lambda value: torch.tensor(
            value, device=self.device, dtype=torch.float32
        )
        return {
            "curriculum_stage": scalar(self.curriculum_stage),
            "curriculum_episode_duration_s": scalar(
                self.task_config.curriculum_durations_s[self.curriculum_stage]
            ),
            "curriculum_target_scale": scalar(
                self.task_config.curriculum_target_scales[self.curriculum_stage]
            ),
            "curriculum_last_success_fraction": scalar(
                self.curriculum_last_success_fraction
            ),
            "curriculum_consecutive_passes": scalar(
                self.curriculum_consecutive_passes
            ),
            "curriculum_promoted": scalar(float(promoted)),
            "curriculum_promotion_enabled": scalar(float(allow_promotion)),
        }

    @classmethod
    def create(
        cls,
        simulator_config: str | Path,
        parallel_count: int,
        device: torch.device | str,
        dtype: torch.dtype,
        task_config: TaskConfig,
        *,
        reward_calculator: RewardCalculator,
        static_randomizer: StaticRandomizer,
        dynamic_randomization: dict[str, Any] | None = None,
        dynamic_seed: int | None = None,
        reward_context_fields: tuple[dict[str, Any], ...] = (),
        command_source_config: VirtualPilotConfig,
        control_contract_config: ControlContractConfig,
    ) -> "SimEnvAdapter":
        """按公开工厂接口创建 SimEnv，并绑定训练所需的设备和精度。"""

        from simenv import SimulationEnvironment

        resolved_device = torch.device(device)
        simulator = SimulationEnvironment.create(
            simulator_config,
            parallel_count,
            resolved_device,
            dtype,
            dynamic_randomization=dynamic_randomization,
            dynamic_seed=dynamic_seed,
        )
        return cls(
            simulator,
            simulator_config,
            task_config,
            resolved_device,
            dtype,
            reward_calculator=reward_calculator,
            static_randomizer=static_randomizer,
            reward_context_fields=reward_context_fields,
            command_source_config=command_source_config,
            control_contract_config=control_contract_config,
        )

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    def reset(
        self,
        mask: torch.Tensor | None = None,
        *,
        static_parameters: TensorDict | None = None,
    ) -> TensorDict:
        """稀疏重置指定环境；未传 mask 时重置整个 batch。

        返回的 ``is_init`` 直接供 GRU 清零对应环境的隐状态。
        """

        if mask is None:
            mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        assert_tensor_on(mask, device=self.device, dtype=torch.bool, shape=(self.batch_size,), name="reset mask")
        candidates = (
            static_parameters
            if static_parameters is not None
            else self.static_randomizer.sample(mask, self.episode_id + 1)
        )
        self.simulator.reset(mask, self.simulator_config, static_parameters=candidates)
        if len(self.current_static_parameters.keys()) == 0:
            self.current_static_parameters = candidates.clone()
        else:
            for key in candidates.keys():
                current = self.current_static_parameters[key]
                replacement = candidates[key]
                expanded = mask.reshape(self.batch_size, *([1] * (replacement.ndim - 1)))
                self.current_static_parameters[key] = torch.where(expanded, replacement, current)
        self.command_source.reset(mask)
        self.previous_action = torch.where(mask[:, None], torch.zeros_like(self.previous_action), self.previous_action)
        self.episode_step = torch.where(mask, torch.zeros_like(self.episode_step), self.episode_step)
        self.episode_id = self.episode_id + mask.to(torch.int64)
        observation, _attitude, _rate, _height = self._observation()
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
        """推进一个控制周期并返回 TorchRL 风格的下一状态字段。

        输入策略动作域统一为 ``[-1,1]^4``。终止、超时或仿真无效的环境会立即
        稀疏重置，返回的 observation 已是新 episode 首帧，同时 ``is_init``
        标记出需要重置循环状态的位置。
        """

        assert_tensor_on(
            standard_action,
            device=self.device,
            dtype=self.dtype,
            shape=(self.batch_size, 4),
            name="standard_action",
        )
        pilot = self.command_source.snapshot()
        pilot_info = self.command_source.info()
        command = self.action_to_command(
            standard_action,
            pilot.upper_throttle,
            self.control_contract_config.policy_action_trim,
            self.control_contract_config.policy_action_residual_scale,
        )
        result = self.simulator.advance(command)
        _observation_before_reset, attitude, angular_velocity, height = self._observation()
        transition = self.task.transition(
            attitude,
            angular_velocity,
            pilot.target_attitude_q_wb,
            standard_action,
            self.previous_action,
            episode_step=self.episode_step,
            max_episode_steps=self.max_episode_steps,
            env_context=self._reward_environment_context(),
            desired_yaw_rate=self.command_source.desired_yaw_rate,
        )
        self.episode_step = self.episode_step + 1
        # 保存本次转移所属 episode 的实际长度；下面 masked reset 会把内部
        # episode_step 清零，但训练诊断仍需知道刚结束 episode 活了多久。
        transition_episode_step = self.episode_step.clone()
        valid = result.valid[:, None] & transition.valid
        truncated = self.episode_step[:, None] >= self.max_episode_steps
        terminated = transition.terminated
        reset_mask = (terminated | truncated | ~valid).squeeze(-1)

        # SimEnv 根据 GPU 布尔 mask 执行稀疏重置。其元数据/日志路径可能在重置边界
        # 发生一次同步，但 rollout 与学习张量不会因此搬到 CPU。
        static_candidates = self.static_randomizer.sample(reset_mask, self.episode_id + 1)
        self.simulator.reset(
            reset_mask,
            self.simulator_config,
            static_parameters=static_candidates,
        )
        for key in static_candidates.keys():
            replacement = static_candidates[key]
            expanded = reset_mask.reshape(self.batch_size, *([1] * (replacement.ndim - 1)))
            self.current_static_parameters[key] = torch.where(
                expanded, replacement, self.current_static_parameters[key]
            )
        self.command_source.step(height, active_mask=~reset_mask)
        self.command_source.reset(reset_mask)
        self.episode_id = self.episode_id + reset_mask.to(torch.int64)
        self.episode_step = torch.where(reset_mask, torch.zeros_like(self.episode_step), self.episode_step)
        self.previous_action = torch.where(reset_mask[:, None], torch.zeros_like(standard_action), standard_action)
        # 飞手先更新下一控制周期的油门和姿态目标，再构造交给下一次策略推理的观测。
        # 对 reset 实例这里读取重置后的 SimEnv 状态；其他实例保持刚推进后的状态。
        observation, _attitude, _rate, _height = self._observation()

        info = dict(transition.info)
        info["sim.error_code"] = result.error_code[:, None]
        info["action.command"] = command
        info["flight.height_m"] = height
        info["episode.length_steps"] = transition_episode_step[:, None]
        info.update(dict(pilot_info.items()))
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

    def _observation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """读取真值/传感器视图并按稳定字段顺序拼接策略观测。"""

        truth = self.simulator.observe(
            "truth",
            (
                "position_n",
                "attitude_q_wb",
                "angular_velocity_b",
                "linear_acceleration_n",
                "motor_speed",
                "servo_angle",
            ),
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
        servo_angle = truth.values["servo_angle"]
        height = -truth.values["position_n"][..., 2:3]
        # 固定尺度属于部署特征契约，训练和 MCU 推理必须使用相同数值。
        normalized_angular_velocity = (
            angular_velocity / self.task_config.terminate_angular_rate_rad_s
        )
        normalized_acceleration = acceleration / self.acceleration_scale_m_s2
        normalized_motor_speed = motor_speed / self.motor_speed_scale_rad_s
        normalized_servo_angle = servo_angle / (0.5 * torch.pi)
        current_euler = quaternion_to_euler(attitude)
        target_euler = quaternion_to_euler(self.command_source.target_attitude)
        roll_pitch_error = torch.atan2(
            torch.sin(target_euler[:, :2] - current_euler[:, :2]),
            torch.cos(target_euler[:, :2] - current_euler[:, :2]),
        )
        relative_attitude = euler_to_quaternion(
            roll_pitch_error[:, 0],
            roll_pitch_error[:, 1],
            torch.zeros_like(roll_pitch_error[:, 0]),
        )
        yaw_scale = max(self.command_source.config.max_yaw_rate_rad_s, 1e-6)
        normalized_desired_yaw_rate = (
            self.command_source.desired_yaw_rate / yaw_scale
        )
        observation = torch.cat(
            (
                relative_attitude,
                normalized_angular_velocity,
                normalized_acceleration,
                normalized_motor_speed,
                normalized_servo_angle,
                normalized_desired_yaw_rate,
                self.command_source.upper_throttle * 2.0 - 1.0,
                self.previous_action,
            ),
            dim=-1,
        )
        if observation.shape != (self.batch_size, 21):
            raise RuntimeError(f"observation contract produced {tuple(observation.shape)}, expected {(self.batch_size, 21)}")
        return observation, attitude, angular_velocity, height

    def _reward_environment_context(self) -> TensorDict:
        """只发布配置显式声明的环境张量，不改变 student observation。"""

        values: dict[str, torch.Tensor] = {}
        task_fields = {
            "attitude_geodesic_rad", "attitude_q_wb", "target_attitude_q_wb",
            "angular_velocity_b", "action", "previous_action", "terminated",
        }
        for field in self.reward_context_fields:
            name = str(field["name"])
            source = str(field["source"])
            key = str(field.get("alias", name))
            if name in task_fields or source in {"task", "policy", "adapter"}:
                continue
            if source in {"truth", "sensor"}:
                values[key] = self.simulator.observe(source, (name,)).values[name]
            elif source == "train_randomizer":
                if name == "static_domain":
                    values.update(
                        {f"static_domain.{param}": value for param, value in self.current_static_parameters.items()}
                    )
                else:
                    values[key] = self.current_static_parameters[name]
            elif source == "simenv":
                values[key] = self.simulator.parameters[name]
            else:
                raise ValueError(f"unsupported reward context source: {source}")
        return TensorDict(values, batch_size=[self.batch_size], device=self.device)

    @staticmethod
    def action_to_command(
        standard_action: torch.Tensor,
        upper_throttle: torch.Tensor,
        trim_command: tuple[float, ...] = (0.5, 0.0, 0.0, 0.0),
        residual_scale: tuple[float, ...] = (0.5, 1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        """把 4 维策略动作和外部油门合成为 SimEnv 5 维命令。

        上桨 PWM 完全来自 VirtualPilot；策略四维均表示相对配平命令的残差，
        依次映射到下桨 PWM 和三个舵面。该所有权边界防止 PPO 学习高度油门。
        """

        trim = standard_action.new_tensor(trim_command)
        scale = standard_action.new_tensor(residual_scale)
        physical = trim + scale * standard_action.clamp(-1.0, 1.0)
        physical_lower = physical[..., :1].clamp(0.0, 1.0)
        physical_servos = physical[..., 1:4].clamp(-1.0, 1.0)
        return torch.cat(
            (
                upper_throttle.clamp(0.0, 1.0),
                physical_lower,
                physical_servos,
            ),
            dim=-1,
        )

    def close(self) -> None:
        if not self._closed:
            self.simulator.close()
            self._closed = True


def _load_timing(path: Path) -> tuple[int, int]:
    """从公开配置读取结构性频率，不访问 SimEnv 私有成员。"""
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
