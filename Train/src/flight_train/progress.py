from __future__ import annotations

import math
import sys
import time
from typing import Any, Mapping, TextIO

from .config import ExperimentConfig


class LiveTrainingProgress:
    """训练终端进度：TTY 原位刷新，重定向日志时输出低频完整行。"""

    def __init__(
        self,
        config: ExperimentConfig,
        run_id: str,
        *,
        stream: TextIO | None = None,
        refresh_seconds: float = 2.0,
    ) -> None:
        self.stream = stream or sys.stderr
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.refresh_seconds = refresh_seconds
        self.total_steps = config.run.total_control_steps
        self.batch_size = config.run.parallel_count
        self.rollout_steps = config.run.rollout_steps
        self.algorithm_name = config.algorithm_name.upper()
        self.total_rollouts = math.ceil(
            self.total_steps / (self.batch_size * self.rollout_steps)
        )
        self.started = time.monotonic()
        self.last_rendered = 0.0
        self.global_steps = 0
        self.rollout_index = 0
        self._line_active = False
        self._write_line(
            "训练开始"
            f" | run={run_id[:8]} | device={config.run.device}"
            f" | B={self.batch_size} | rollout={self.rollout_steps}"
            f" | 总样本={self.total_steps:,} | 预计 {self.total_rollouts} 轮"
        )

    def begin_rollout(self, global_steps: int, rollout_index: int) -> None:
        self.global_steps = global_steps
        self.rollout_index = rollout_index
        self._render("准备采样", global_steps, force=True)

    def collect_step(self, local_step: int, total_local_steps: int) -> None:
        now = time.monotonic()
        if local_step != total_local_steps and now - self.last_rendered < self.refresh_seconds:
            return
        projected = min(
            self.total_steps,
            self.global_steps + local_step * self.batch_size,
        )
        self._render(
            f"采样 {local_step}/{total_local_steps}",
            projected,
            force=local_step == total_local_steps,
        )

    def begin_update(self) -> None:
        projected = min(
            self.total_steps,
            self.global_steps + self.rollout_steps * self.batch_size,
        )
        self._render(f"{self.algorithm_name} 更新", projected, force=True)

    def update_step(self, completed: int, total: int) -> None:
        projected = min(
            self.total_steps,
            self.global_steps + self.rollout_steps * self.batch_size,
        )
        self._render(
            f"{self.algorithm_name} 更新 {completed}/{total}",
            projected,
            force=completed == total,
        )

    def complete_update(
        self, global_steps: int, metrics: Mapping[str, Any]
    ) -> None:
        self.global_steps = global_steps
        elapsed = max(time.monotonic() - self.started, 1e-9)
        samples_per_second = global_steps / elapsed
        fields = [
            f"第 {self.rollout_index}/{self.total_rollouts} 轮完成",
            f"reward={_metric(metrics, 'rollout_reward_mean'):.3f}",
            f"姿态P95={_metric(metrics, 'attitude_error_p95_deg'):.2f}°",
            (
                f"课程={_metric(metrics, 'curriculum_episode_duration_s'):.3g}s"
                f"/成功率={100.0 * _metric(metrics, 'curriculum_last_success_fraction'):.1f}%"
            ),
            f"终止={100.0 * _metric(metrics, 'terminated_fraction'):.2f}%",
            f"动作饱和={100.0 * _metric(metrics, 'action_saturation_fraction'):.2f}%",
            f"{samples_per_second:,.0f} sample/s",
        ]
        if self.algorithm_name == "SAC":
            std_values = "/".join(
                f"{_metric(metrics, f'exploration_std_action_{index}'):.3f}"
                for index in range(4)
            )
            if _metric(metrics, "sac_warmup") > 0.5:
                sac_phase = "warmup"
            elif _metric(metrics, "sac_critic_pretraining") > 0.5:
                sac_phase = "critic-only"
            else:
                sac_phase = "actor"
            fields.insert(
                2,
                (
                    f"replay={_metric(metrics, 'replay_size'):,.0f}"
                    f"/alpha={_metric(metrics, 'alpha'):.4f}"
                    f"/std={std_values}"
                    f"/phase={sac_phase}"
                ),
            )
        self._write_line(self._prefix(global_steps) + " | " + " | ".join(fields))

    def checkpoint(self, control_steps: int, kind: str) -> None:
        self._write_line(
            f"checkpoint 已保存 | 类型={kind} | 控制步={control_steps:,}"
        )

    def finish(self, status: str, global_steps: int) -> None:
        elapsed = time.monotonic() - self.started
        self._write_line(
            f"训练结束 | status={status} | 控制步={global_steps:,}/{self.total_steps:,}"
            f" | 用时={_duration(elapsed)}"
        )

    def _render(self, stage: str, steps: int, *, force: bool) -> None:
        now = time.monotonic()
        # 非交互日志不输出采样中的临时行，只保留阶段切换和每轮汇总，避免刷屏。
        if not self.is_tty and stage.startswith("采样"):
            return
        if not force and now - self.last_rendered < self.refresh_seconds:
            return
        self.last_rendered = now
        text = self._prefix(steps) + f" | 第 {self.rollout_index}/{self.total_rollouts} 轮 | {stage}"
        if self.is_tty:
            self.stream.write("\r\033[K" + text)
            self.stream.flush()
            self._line_active = True
        else:
            self._write_line(text)

    def _prefix(self, steps: int) -> str:
        completed = min(max(steps, 0), self.total_steps)
        fraction = completed / self.total_steps
        filled = round(24 * fraction)
        bar = (
            "=" * 24
            if filled >= 24
            else "=" * filled + ">" + "." * (23 - filled)
        )
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = completed / elapsed
        eta = (self.total_steps - completed) / rate if rate > 0 else float("inf")
        return (
            f"[{bar}] {100.0 * fraction:6.2f}%"
            f" | {completed:,}/{self.total_steps:,}"
            f" | 已用 {_duration(elapsed)} | ETA {_duration(eta)}"
        )

    def _write_line(self, text: str) -> None:
        if self._line_active:
            self.stream.write("\r\033[K")
            self._line_active = False
        self.stream.write(text + "\n")
        self.stream.flush()


def _metric(values: Mapping[str, Any], name: str) -> float:
    value = values.get(name, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
