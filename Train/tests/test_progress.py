from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest

from flight_train.config import load_experiment_config
from flight_train.progress import LiveTrainingProgress


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _config_for_progress(
    *,
    total_steps: int,
    parallel_count: int = 256,
    rollout_steps: int = 512,
    algorithm_name: str = "sac",
):
    return SimpleNamespace(
        run=SimpleNamespace(
            total_control_steps=total_steps,
            parallel_count=parallel_count,
            rollout_steps=rollout_steps,
            device="cuda:0",
        ),
        algorithm_name=algorithm_name,
    )


class TrainingProgressTests(unittest.TestCase):
    def test_non_tty_progress_reports_stages_metrics_and_finish(self):
        config = load_experiment_config(
            Path(__file__).parents[1] / "configs/experiments/mlp_ppo_smoke.json"
        )
        stream = StringIO()
        progress = LiveTrainingProgress(
            config, "12345678-test", stream=stream, refresh_seconds=0.0
        )
        progress.begin_rollout(0, 1)
        progress.collect_step(config.run.rollout_steps, config.run.rollout_steps)
        progress.begin_update()
        progress.update_step(1, 2)
        progress.update_step(2, 2)
        progress.complete_update(
            config.run.parallel_count * config.run.rollout_steps,
            {
                "rollout_reward_mean": 0.5,
                "attitude_error_p95_deg": 1.25,
                "terminated_fraction": 0.01,
                "action_saturation_fraction": 0.02,
                "actuator_energy_proxy_mean": 0.01234,
                "actuator_effort_proxy_mean": 0.05678,
                "sampling_steps_per_second": 2048.0,
            },
        )
        progress.checkpoint(128, "final")
        progress.finish("completed", 128)
        output = stream.getvalue()
        self.assertIn("训练开始", output)
        self.assertIn("PPO 更新", output)
        self.assertIn("PPO 更新 2/2", output)
        self.assertIn("姿态P95=1.25°", output)
        self.assertIn("执行器能耗代理=0.01234/step", output)
        self.assertIn("执行器持续负载=0.05678", output)
        self.assertIn("采样=2,048 sample/s", output)
        self.assertIn("端到端(近8轮)", output)
        self.assertIn("checkpoint 已保存", output)
        self.assertIn("status=completed", output)
        # 非 TTY 日志不输出每个采样刷新，避免重定向日志刷屏。
        self.assertNotIn("采样 16/16", output)

    def test_continuation_rate_uses_only_steps_added_in_this_session(self):
        config = _config_for_progress(
            total_steps=33_554_432,
        )
        stream = StringIO()
        clock = FakeClock(100.0)
        progress = LiveTrainingProgress(
            config,
            "12345678-test",
            stream=stream,
            refresh_seconds=0.0,
            clock=clock,
        )
        resumed_steps = 16_777_216
        rollout_steps = config.run.parallel_count * config.run.rollout_steps
        progress.start_session(resumed_steps)
        progress.begin_rollout(resumed_steps, 129)
        clock.advance(10.0)
        progress.complete_update(
            resumed_steps + rollout_steps,
            {
                "rollout_reward_mean": 0.5,
                "attitude_error_p95_deg": 1.25,
                "terminated_fraction": 0.01,
                "action_saturation_fraction": 0.02,
                "sampling_steps_per_second": 20_000.0,
            },
        )
        output = stream.getvalue()
        # 131,072 个本次新增控制步 / 10 秒，而不是用恢复后的 16M 全局步数。
        self.assertIn("端到端(近8轮)=13,107 step/s", output)
        self.assertNotIn("1,690,829", output)
        self.assertIn("采样=20,000 sample/s", output)

    def test_rolling_rate_uses_at_most_the_latest_eight_rollouts(self):
        config = _config_for_progress(
            total_steps=9 * 8 * 16,
            parallel_count=8,
            rollout_steps=16,
            algorithm_name="ppo",
        )
        stream = StringIO()
        clock = FakeClock()
        progress = LiveTrainingProgress(
            config,
            "12345678-test",
            stream=stream,
            refresh_seconds=0.0,
            clock=clock,
        )
        steps_per_rollout = config.run.parallel_count * config.run.rollout_steps
        progress.start_session(0)
        for rollout_index in range(1, 10):
            progress.begin_rollout(
                (rollout_index - 1) * steps_per_rollout, rollout_index
            )
            # 第一轮很慢；第 2~9 轮均为 1 秒。第 9 轮显示时应已排除第一轮。
            clock.advance(100.0 if rollout_index == 1 else 1.0)
            progress.complete_update(
                rollout_index * steps_per_rollout,
                {
                    "rollout_reward_mean": 0.5,
                    "attitude_error_p95_deg": 1.25,
                    "terminated_fraction": 0.01,
                    "action_saturation_fraction": 0.02,
                },
            )
        expected = f"端到端(近8轮)={steps_per_rollout:,.0f} step/s"
        self.assertIn(expected, stream.getvalue().splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
