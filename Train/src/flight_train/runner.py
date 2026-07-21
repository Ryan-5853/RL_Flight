from __future__ import annotations

from typing import Any

import torch

from .algorithms import RecurrentPPO
from .collector import TensorDictRolloutCollector
from .config import ExperimentConfig
from .envs import SimEnvAdapter
from .models import build_actor_critic
from .recording import RunRecorder


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    device = torch.device(config.run.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    torch.manual_seed(config.run.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.run.seed)

    env = None
    recorder = None
    global_steps = 0
    status = "failed"
    try:
        env = SimEnvAdapter.create(
            config.simulator_config,
            config.run.parallel_count,
            device,
            config.torch_dtype,
            config.task,
            seed=config.run.seed + 1,
        )
        if not config.run.allow_unimplemented_simulator:
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

        model = build_actor_critic(
            env.spec.observation_dim,
            env.spec.action_dim,
            config.model,
            device,
            config.torch_dtype,
        )
        collector = TensorDictRolloutCollector(env, model, config.run.rollout_steps)
        algorithm = RecurrentPPO(model, config.ppo, device)
        recorder = RunRecorder(config)
        while global_steps < config.run.total_control_steps:
            rollout = collector.collect()
            metrics = algorithm.update(rollout)
            global_steps += rollout.batch_size.numel()
            metrics = {
                **metrics,
                "rollout_reward_mean": rollout[("next", "reward")].mean(),
                "valid_fraction": rollout[("next", "valid")].to(torch.float32).mean(),
            }
            recorder.metrics(global_steps, metrics)
        recorder.checkpoint(
            global_steps,
            {
                "global_control_steps": global_steps,
                "actor": model.actor.state_dict(),
                "critic": model.critic.state_dict(),
                "optimizer": algorithm.optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
                "collector_current": collector.current,
                "training_environment": {
                    "previous_action": env.previous_action,
                    "episode_id": env.episode_id,
                    "episode_step": env.episode_step,
                    "target_attitude": env.task.target_attitude,
                    "task_rng_state": env.task.generator.get_state(),
                },
                "config": config.raw,
            },
        )
        status = "completed"
        return {"run_directory": recorder.directory, "global_control_steps": global_steps}
    finally:
        if env is not None:
            env.close()
        if recorder is not None:
            recorder.close(status, global_steps)
