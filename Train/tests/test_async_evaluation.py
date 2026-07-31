from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from flight_train.async_evaluation import (
    AsyncEvaluationManager,
    _atomic_write_json,
)
from flight_train.config import (
    ConfigError,
    exact_resume_config_sha256,
    load_experiment_config,
)
from flight_train.evaluation import load_fixed_evaluation_suite
from flight_train.recording import load_checkpoint
from flight_train.runner import run_experiment


ROOT = Path(__file__).parents[1]


def _successful_controlled_worker(
    config,
    suite,
    policy_checkpoint,
    output_root,
    result_path,
    source_checkpoint,
    source_checkpoint_sha256,
    log_path,
):
    del config, suite, source_checkpoint, source_checkpoint_sha256, log_path
    state = load_checkpoint(policy_checkpoint)
    control_steps = int(state["global_control_steps"])
    run_directory = result_path.parents[2]
    release = run_directory / f"release_{control_steps}"
    deadline = time.monotonic() + 15.0
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"test worker was not released: {control_steps}")
        time.sleep(0.01)
    evaluation_directory = output_root / "fake_suite" / f"step_{control_steps}"
    evaluation_directory.mkdir(parents=True)
    (evaluation_directory / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint_global_control_steps": control_steps,
            }
        ),
        encoding="utf-8",
    )
    _atomic_write_json(
        result_path,
        {
            "schema_version": 1,
            "status": "completed",
            "global_control_steps": control_steps,
            "evaluation_directory": str(evaluation_directory),
        },
    )


def _failing_worker(
    config,
    suite,
    policy_checkpoint,
    output_root,
    result_path,
    source_checkpoint,
    source_checkpoint_sha256,
    log_path,
):
    del (
        config,
        suite,
        policy_checkpoint,
        output_root,
        source_checkpoint,
        source_checkpoint_sha256,
        log_path,
    )
    _atomic_write_json(
        result_path,
        {
            "schema_version": 1,
            "status": "failed",
            "error": "synthetic evaluation failure",
            "traceback": "synthetic traceback",
        },
    )


def _write_checkpoint(path: Path, value: float) -> None:
    torch.save({"value": torch.tensor([value])}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        digest + "\n", encoding="ascii"
    )


class AsyncEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_experiment_config(
            ROOT / "configs/experiments/mlp_sac_smoke.json"
        )
        self.suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_self_stabilize_v2.yaml"
        )

    def test_v7_enables_four_latest_only_subprocess_workers(self):
        config = load_experiment_config(
            ROOT
            / "configs/experiments/"
            "mlp_sac_upright_height_only_small_tip_stability_v7.yaml"
        )
        execution = config.evaluation.execution
        self.assertEqual(execution.mode, "subprocess")
        self.assertEqual(execution.max_in_flight, 4)
        self.assertEqual(execution.pending_policy, "latest")
        self.assertEqual(execution.pin_checkpoint, "hardlink")
        self.assertTrue(execution.wait_for_final)
        self.assertEqual(execution.failure_policy, "continue_training")
        legacy_raw = json.loads(json.dumps(config.raw))
        legacy_raw["evaluation"].pop("execution")
        self.assertEqual(
            exact_resume_config_sha256(config),
            exact_resume_config_sha256(legacy_raw),
        )

    def test_invalid_parallel_worker_count_is_rejected(self):
        raw = json.loads(
            (ROOT / "configs/experiments/mlp_sac_smoke.json").read_text()
        )
        raw["environment"]["config_path"] = str(
            ROOT / "configs/environment/sim_smoke.json"
        )
        raw["evaluation"] = {
            "enabled": True,
            "interval_control_steps": 128,
            "suite_path": str(
                ROOT / "configs/evaluation/fixed_self_stabilize_v2.yaml"
            ),
            "execution": {
                "mode": "subprocess",
                "max_in_flight": 9,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "must not exceed 8"):
                load_experiment_config(path)

    def test_latest_pending_replaces_older_job_and_pins_without_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "run"
            run_directory.mkdir()
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            paths = []
            for step in (10, 20, 30):
                path = checkpoints / f"step_{step}.pt"
                _write_checkpoint(path, float(step))
                paths.append(path)
            manager = AsyncEvaluationManager(
                self.config,
                self.suite,
                run_directory,
                worker_target=_successful_controlled_worker,
            )
            try:
                torch.manual_seed(1234)
                rng_before_submit = torch.get_rng_state().clone()
                self.assertEqual(
                    manager.submit(paths[0], 10, {"weight": torch.ones(2)}),
                    "started",
                )
                torch.testing.assert_close(
                    torch.get_rng_state(), rng_before_submit
                )
                first_job = next(iter(manager._running.values())).job
                self.assertEqual(
                    first_job.pinned_checkpoint.stat().st_ino,
                    paths[0].stat().st_ino,
                )
                snapshot = load_checkpoint(first_job.policy_checkpoint)
                self.assertEqual(snapshot["global_control_steps"], 10)
                self.assertEqual(set(snapshot), {
                    "checkpoint_schema_version",
                    "global_control_steps",
                    "actor",
                })

                self.assertEqual(
                    manager.submit(paths[1], 20, {"weight": torch.ones(2) * 2}),
                    "queued",
                )
                old_pending_directory = manager._pending.directory
                self.assertEqual(
                    manager.submit(paths[2], 30, {"weight": torch.ones(2) * 3}),
                    "replaced_pending",
                )
                self.assertFalse(old_pending_directory.exists())
                self.assertEqual(manager.running_steps, (10,))
                self.assertEqual(manager.pending_step, 30)

                (run_directory / "release_10").touch()
                first = manager.wait()
                self.assertEqual(first.status, "completed")
                self.assertEqual(first.job.control_steps, 10)
                self.assertTrue(first.job.pinned_checkpoint.exists())
                manager.acknowledge(first)
                self.assertEqual(manager.running_steps, (30,))
                self.assertIsNone(manager.pending_step)

                (run_directory / "release_30").touch()
                latest = manager.wait()
                self.assertEqual(latest.status, "completed")
                self.assertEqual(latest.job.control_steps, 30)
                manager.acknowledge(latest)
                self.assertFalse(manager.has_work)
                self.assertTrue(all(path.exists() for path in paths))
                self.assertEqual(list(manager.jobs_root.iterdir()), [])
            finally:
                manager.shutdown(cancel=True)

    def test_multiple_workers_finish_out_of_order_and_share_one_latest_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "run"
            run_directory.mkdir()
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            paths = {}
            for step in (10, 20, 30, 40):
                path = checkpoints / f"step_{step}.pt"
                _write_checkpoint(path, float(step))
                paths[step] = path
            execution = replace(
                self.config.evaluation.execution,
                max_in_flight=2,
            )
            config = replace(
                self.config,
                evaluation=replace(
                    self.config.evaluation,
                    execution=execution,
                ),
            )
            manager = AsyncEvaluationManager(
                config,
                self.suite,
                run_directory,
                worker_target=_successful_controlled_worker,
            )
            try:
                self.assertEqual(
                    manager.submit(
                        paths[10], 10, {"weight": torch.ones(2)}
                    ),
                    "started",
                )
                self.assertEqual(
                    manager.submit(
                        paths[20], 20, {"weight": torch.ones(2) * 2}
                    ),
                    "started",
                )
                self.assertEqual(manager.running_steps, (10, 20))
                self.assertEqual(manager.running_count, 2)

                self.assertEqual(
                    manager.submit(
                        paths[30], 30, {"weight": torch.ones(2) * 3}
                    ),
                    "queued",
                )
                superseded_directory = manager._pending.directory
                self.assertEqual(
                    manager.submit(
                        paths[40], 40, {"weight": torch.ones(2) * 4}
                    ),
                    "replaced_pending",
                )
                self.assertFalse(superseded_directory.exists())
                self.assertEqual(manager.pending_step, 40)

                # Worker 20 completes before worker 10. Its acknowledgement opens
                # one slot and immediately promotes the sole latest pending job.
                (run_directory / "release_20").touch()
                second = manager.wait()
                self.assertEqual(second.job.control_steps, 20)
                manager.acknowledge(second)
                self.assertEqual(manager.running_steps, (10, 40))
                self.assertIsNone(manager.pending_step)

                (run_directory / "release_40").touch()
                latest = manager.wait()
                self.assertEqual(latest.job.control_steps, 40)
                manager.acknowledge(latest)
                self.assertEqual(manager.running_steps, (10,))

                (run_directory / "release_10").touch()
                first = manager.wait()
                self.assertEqual(first.job.control_steps, 10)
                manager.acknowledge(first)
                self.assertFalse(manager.has_work)
                self.assertEqual(list(manager.jobs_root.iterdir()), [])
            finally:
                manager.shutdown(cancel=True)

    def test_failed_worker_releases_large_pin_but_keeps_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "run"
            run_directory.mkdir()
            checkpoint = root / "step_10.pt"
            _write_checkpoint(checkpoint, 10.0)
            manager = AsyncEvaluationManager(
                self.config,
                self.suite,
                run_directory,
                worker_target=_failing_worker,
            )
            try:
                manager.submit(
                    checkpoint,
                    10,
                    {"weight": torch.ones(2)},
                )
                outcome = manager.wait()
                self.assertEqual(outcome.status, "failed")
                self.assertIn("synthetic", outcome.error)
                job_directory = outcome.job.directory
                manager.acknowledge(outcome)
                self.assertFalse(outcome.job.pinned_checkpoint.exists())
                self.assertFalse(outcome.job.policy_checkpoint.exists())
                self.assertTrue((job_directory / "result.json").is_file())
                self.assertTrue((job_directory / "job.json").is_file())
                self.assertTrue(checkpoint.exists())
            finally:
                manager.shutdown(cancel=True)

    def test_training_continues_with_spawned_cpu_evaluation_and_records_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulator = json.loads(
                (ROOT / "configs/environment/sim_smoke.json").read_text()
            )
            simulator["logging"]["directory"] = str(root / "simlogs")
            simulator_path = root / "simulator.json"
            simulator_path.write_text(json.dumps(simulator), encoding="utf-8")

            suite = yaml.safe_load(
                (
                    ROOT / "configs/evaluation/fixed_self_stabilize_v2.yaml"
                ).read_text(encoding="utf-8")
            )
            suite["name"] = "async_smoke"
            suite["parallel_count"] = 2
            suite["output_root"] = str(root / "standalone_evaluations")
            for scenario in suite["scenarios"]:
                scenario["duration_s"] = 0.002
            suite_path = root / "suite.yaml"
            suite_path.write_text(
                yaml.safe_dump(suite, sort_keys=False),
                encoding="utf-8",
            )

            experiment = json.loads(
                (
                    ROOT / "configs/experiments/mlp_sac_smoke.json"
                ).read_text()
            )
            experiment["experiment"]["name"] = "async-evaluation-smoke"
            experiment["environment"]["config_path"] = str(simulator_path)
            experiment["run"].update(
                {
                    "device": "cpu",
                    "parallel_count": 2,
                    "total_control_steps": 32,
                    "output_root": str(root / "runs"),
                }
            )
            experiment["evaluation"] = {
                "enabled": True,
                "interval_control_steps": 16,
                "suite_path": str(suite_path),
                "execution": {
                    "mode": "subprocess",
                    "max_in_flight": 2,
                    "pending_policy": "latest",
                    "pin_checkpoint": "hardlink",
                    "wait_for_final": True,
                    "failure_policy": "stop_training",
                },
            }
            experiment["checkpoint"] = {
                "interval_control_steps": 16,
                # step 16 rotates out before its worker finishes; its in-flight
                # hard link must still support best-checkpoint promotion.
                "keep_last": 1,
                "minimum_free_space_bytes": 0,
            }
            experiment_path = root / "experiment.json"
            experiment_path.write_text(
                json.dumps(experiment),
                encoding="utf-8",
            )

            result = run_experiment(load_experiment_config(experiment_path))
            self.assertEqual(result["status"], "completed")
            records = [
                json.loads(line)
                for line in (
                    result["run_directory"] / "evaluations.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sorted(record["global_control_steps"] for record in records),
                [16, 32],
            )
            self.assertTrue(
                all(record["status"] == "completed" for record in records)
            )
            checkpoint_directory = result["run_directory"] / "checkpoints"
            self.assertFalse((checkpoint_directory / "step_16.pt").exists())
            self.assertTrue(
                (checkpoint_directory / "best_total_evaluation.pt").is_file()
            )
            self.assertEqual(
                list(
                    (
                        result["run_directory"] / "evaluation_jobs"
                    ).iterdir()
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
