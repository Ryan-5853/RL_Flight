from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from simenv import RealtimeSimulationEnvironment
from test_environment import _config


class RealtimeSimulationEnvironmentTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        path = root / "config.json"
        path.write_text(
            json.dumps(_config(str(root / "logs"))), encoding="utf-8"
        )
        return path

    def test_create_disables_disk_logging_and_reuses_observation_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with RealtimeSimulationEnvironment.create(
                self._write_config(root), compile_kernels=False
            ) as realtime:
                first = realtime.observation_vector("sensor", ("gyro",))
                second = realtime.observation_vector("sensor", ("gyro",))

                self.assertEqual(first.shape, (1, 3))
                self.assertEqual(first.data_ptr(), second.data_ptr())
                self.assertEqual(realtime.observation_layout["sensor"], ("gyro",))
                self.assertFalse((root / "logs").exists())

    def test_state_views_are_uncloned_and_refresh_after_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with RealtimeSimulationEnvironment.create(
                self._write_config(root), compile_kernels=False
            ) as realtime:
                first = realtime.state_views("truth", ("position_n",))[0]
                internal = realtime.environment._truth["position_n"]
                self.assertEqual(first.data_ptr(), internal.data_ptr())

                realtime.advance(torch.zeros((1, 5), dtype=torch.float32))
                second = realtime.state_views("truth", ("position_n",))[0]
                self.assertEqual(
                    second.data_ptr(),
                    realtime.environment._truth["position_n"].data_ptr(),
                )
                self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_warmup_restores_exact_numerical_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with RealtimeSimulationEnvironment.create(
                self._write_config(root), compile_kernels=False
            ) as realtime:
                before = realtime.environment.state_dict()
                elapsed = realtime.warmup(steps=2)
                after = realtime.environment.state_dict()

                self.assertGreaterEqual(elapsed, 0.0)
                for group in ("parameters", "truth", "sensors", "random_counters"):
                    for name, expected in before[group].items():
                        torch.testing.assert_close(after[group][name], expected)
                for name in (
                    "physics_step",
                    "control_step",
                    "valid",
                    "error_code",
                    "generation",
                    "instance_seeds",
                    "control",
                ):
                    torch.testing.assert_close(after[name], before[name])

    def test_policy_loop_advances_and_reports_compute_latency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with RealtimeSimulationEnvironment.create(
                self._write_config(root), compile_kernels=False
            ) as realtime:
                control = torch.zeros((1, 5), dtype=torch.float32)

                def policy(observation: torch.Tensor) -> torch.Tensor:
                    self.assertEqual(observation.shape, (1, 3))
                    return control

                stats = realtime.run_policy(policy, steps=4, realtime=False)

                self.assertEqual(stats.steps, 4)
                self.assertEqual(stats.deadline_misses, 0)
                self.assertGreater(stats.achieved_hz, 0.0)
                self.assertGreater(stats.mean_compute_us, 0.0)
                self.assertEqual(
                    int(realtime.environment.state_dict()["physics_step"][0]), 4
                )

    def test_policy_loop_rejects_cross_device_or_wrong_shape_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with RealtimeSimulationEnvironment.create(
                self._write_config(root), compile_kernels=False
            ) as realtime:
                with self.assertRaisesRegex(ValueError, "shape"):
                    realtime.run_policy(
                        lambda _: torch.zeros(4), steps=1, realtime=False
                    )


if __name__ == "__main__":
    unittest.main()
