from __future__ import annotations

from io import StringIO
from pathlib import Path
import unittest

from flight_train.config import load_experiment_config
from flight_train.progress import LiveTrainingProgress


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
            },
        )
        progress.checkpoint(128, "final")
        progress.finish("completed", 128)
        output = stream.getvalue()
        self.assertIn("训练开始", output)
        self.assertIn("PPO 更新", output)
        self.assertIn("PPO 更新 2/2", output)
        self.assertIn("姿态P95=1.25°", output)
        self.assertIn("checkpoint 已保存", output)
        self.assertIn("status=completed", output)
        # 非 TTY 日志不输出每个采样刷新，避免重定向日志刷屏。
        self.assertNotIn("采样 16/16", output)


if __name__ == "__main__":
    unittest.main()
