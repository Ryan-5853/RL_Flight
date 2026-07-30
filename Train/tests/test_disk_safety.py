from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from flight_train.config import load_experiment_config
from flight_train.recording import (
    InsufficientDiskSpaceError,
    RunRecorder,
    load_checkpoint,
)
from flight_train.runner import run_experiment


ROOT = Path(__file__).parents[1]
SIM_CONFIG = ROOT / "configs/environment/sim_smoke.json"
EXPERIMENT_CONFIG = ROOT / "configs/experiments/gru_ppo_smoke.json"


def _write_config(root: Path, *, name: str) -> Path:
    simulator = json.loads(SIM_CONFIG.read_text(encoding="utf-8"))
    simulator["logging"] = {
        "directory": str(root / "simlogs"),
        "chunk_steps": 32,
        "queue_chunks": 2,
        "overflow": "block",
        "minimum_free_space_bytes": 0,
    }
    simulator_path = root / "sim.json"
    simulator_path.write_text(json.dumps(simulator), encoding="utf-8")

    experiment = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    experiment["experiment"]["name"] = name
    experiment["environment"]["config_path"] = str(simulator_path)
    experiment["run"].update(
        {
            "device": "cpu",
            "parallel_count": 2,
            "total_control_steps": 8,
            "output_root": str(root / "runs"),
        }
    )
    experiment["task"]["episode_duration_s"] = 0.008
    experiment["collector"]["control_steps_per_rollout"] = 4
    experiment["algorithm"].update(
        {"sequence_length": 2, "minibatches": 2, "epochs_per_rollout": 1}
    )
    experiment["checkpoint"]["minimum_free_space_bytes"] = 1024
    path = root / "experiment.json"
    path.write_text(json.dumps(experiment), encoding="utf-8")
    return path


class TrainingDiskSafetyTests(unittest.TestCase):
    def test_checkpoint_preflight_preserves_last_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config = load_experiment_config(
                _write_config(Path(raw), name="checkpoint-space")
            )
            recorder = RunRecorder(config)
            try:
                previous = recorder.checkpoint(
                    4,
                    {"value": torch.tensor([1.0])},
                    kind="periodic",
                )
                with mock.patch(
                    "flight_train.recording.shutil.disk_usage",
                    return_value=SimpleNamespace(free=1024),
                ), mock.patch(
                    "flight_train.recording.torch.save"
                ) as save:
                    with self.assertRaises(InsufficientDiskSpaceError):
                        recorder.checkpoint(
                            8,
                            {"value": torch.tensor([2.0])},
                            kind="periodic",
                        )
                    save.assert_not_called()

                restored = load_checkpoint(previous)
                torch.testing.assert_close(
                    restored["value"], torch.tensor([1.0])
                )
                self.assertFalse(
                    (
                        recorder.directory
                        / "checkpoints"
                        / "step_8.tmp"
                    ).exists()
                )
                with mock.patch(
                    "flight_train.recording.torch.save",
                    side_effect=RuntimeError(
                        "basic_ios::clear: iostream error"
                    ),
                ):
                    with self.assertRaises(InsufficientDiskSpaceError):
                        recorder.checkpoint(
                            12,
                            {"value": torch.tensor([3.0])},
                            kind="periodic",
                        )
                self.assertFalse(
                    (
                        recorder.directory
                        / "checkpoints"
                        / "step_12.tmp"
                    ).exists()
                )
            finally:
                recorder.close("interrupted", 8)

    def test_runner_records_space_guard_as_recoverable_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config = load_experiment_config(
                _write_config(Path(raw), name="runner-space")
            )
            error = InsufficientDiskSpaceError(
                config.run.output_root,
                available_bytes=1024,
                required_bytes=4096,
                reserve_bytes=1024,
                estimated_checkpoint_bytes=3072,
                operation="test checkpoint preflight",
            )
            with mock.patch.object(
                RunRecorder,
                "ensure_checkpoint_capacity",
                side_effect=error,
            ):
                result = run_experiment(config)

            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["reason"], "insufficient_disk_space")
            status = json.loads(
                (result["run_directory"] / "status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["status"], "interrupted")
            self.assertEqual(status["reason"], "insufficient_disk_space")
            self.assertIn("test checkpoint preflight", status["detail"])


if __name__ == "__main__":
    unittest.main()
