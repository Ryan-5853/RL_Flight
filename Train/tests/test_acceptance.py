from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from tensordict import TensorDict

from flight_train.algorithms import RecurrentPPO
from flight_train.collector import TensorDictRolloutCollector
from flight_train.commands import VirtualPilotCommandSource
from flight_train.config import (
    ConfigError,
    ModelConfig,
    PPOConfig,
    TaskConfig,
    load_experiment_config,
)
from flight_train.core import EnvSpec
from flight_train.envs import SimEnvAdapter
from flight_train.evaluation import collect_evaluation_rollout
from flight_train.math import quaternion_geodesic_angle
from flight_train.models import build_actor_critic
from flight_train.randomization import StaticParameterSpec, StaticRandomizer
from flight_train.recording import RunRecorder, load_checkpoint, tensor_state_sha256
from flight_train.rewards import AttitudeRewardCalculator, RewardOutput
from flight_train.runner import run_experiment
from flight_train.tasks import AttitudeTrackingTask


ROOT = Path(__file__).parents[1]
SIM_CONFIG = ROOT / "configs/environment/sim_smoke.json"
EXPERIMENT_CONFIG = ROOT / "configs/experiments/gru_ppo_smoke.json"


class DeterministicTensorEnv:
    """小型验收环境：确定性状态转移，可指定单个实例失效。"""

    def __init__(self, batch: int, invalid_at: tuple[int, int] | None = None) -> None:
        self.spec = EnvSpec(batch, 21, 4, torch.device("cpu"), torch.float32, 10, 10, ())
        self.invalid_at = invalid_at
        self.step_index = 0
        self.state = torch.zeros(batch, 21)

    def reset(self, mask=None, *, static_parameters=None):
        del static_parameters
        if mask is None:
            mask = torch.ones(self.spec.parallel_count, dtype=torch.bool)
        self.state = torch.where(mask[:, None], torch.zeros_like(self.state), self.state)
        return TensorDict(
            {"observation": self.state.clone(), "is_init": mask[:, None]},
            [self.spec.parallel_count],
        )

    def step(self, action):
        self.step_index += 1
        valid = torch.ones(self.spec.parallel_count, 1, dtype=torch.bool)
        if self.invalid_at is not None and self.step_index == self.invalid_at[0]:
            valid[self.invalid_at[1]] = False
        delta = torch.zeros_like(self.state)
        delta[:, :4] = action
        self.state = self.state + delta
        return TensorDict(
            {
                "observation": self.state.clone(),
                "reward": -action.square().sum(-1, keepdim=True),
                "terminated": torch.zeros_like(valid),
                "truncated": torch.zeros_like(valid),
                "done": ~valid,
                "valid": valid,
                "is_init": ~valid,
            },
            [self.spec.parallel_count],
        )

    def close(self):
        pass


def _model(batch: int = 2):
    del batch
    return build_actor_critic(
        21, 4, ModelConfig((16, 16), 8, 8), torch.device("cpu")
    )


def _write_run_config(directory: Path, *, name: str = "acceptance") -> Path:
    sim = json.loads(SIM_CONFIG.read_text(encoding="utf-8"))
    sim["logging"] = {
        "directory": str(directory / "simlogs"),
        "chunk_steps": 32,
        "queue_chunks": 2,
        "overflow": "block",
    }
    sim_path = directory / "sim.json"
    sim_path.write_text(json.dumps(sim), encoding="utf-8")

    experiment = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    experiment["experiment"]["name"] = name
    experiment["environment"]["config_path"] = str(sim_path)
    experiment["run"].update(
        {
            "device": "cpu",
            "parallel_count": 2,
            "total_control_steps": 8,
            "output_root": str(directory / "runs"),
        }
    )
    experiment["task"]["episode_duration_s"] = 0.008
    experiment["collector"]["control_steps_per_rollout"] = 4
    experiment["algorithm"].update(
        {"sequence_length": 2, "minibatches": 2, "epochs_per_rollout": 1}
    )
    path = directory / "experiment.json"
    path.write_text(json.dumps(experiment), encoding="utf-8")
    return path


class FrameworkAcceptanceTests(unittest.TestCase):
    def test_01_interface_shapes_for_single_and_batched_environments(self):
        for batch in (1, 4):
            torch.manual_seed(10)
            env = DeterministicTensorEnv(batch)
            model = _model()
            rollout = TensorDictRolloutCollector(env, model, 3).collect()
            self.assertEqual(rollout.batch_size, torch.Size([batch, 3]))
            self.assertEqual(rollout["observation"].shape, (batch, 3, 21))
            self.assertEqual(rollout["action"].shape, (batch, 3, 4))
            self.assertEqual(rollout[("next", "reward")].shape, (batch, 3, 1))
            self.assertEqual(rollout[("next", "valid")].shape, (batch, 3, 1))
            self.assertEqual(rollout["recurrent_state"].shape, (batch, 3, 1, 8))

    def test_02_action_boundaries_and_nonfinite_behavior_are_explicit(self):
        action = torch.tensor([[2.0, -2.0, 0.0, 2.0]])
        torch.testing.assert_close(
            SimEnvAdapter.action_to_command(action, torch.tensor([[0.42]])),
            torch.tensor([[0.42, 1.0, -1.0, 0.0, 1.0]]),
        )
        nonfinite = torch.tensor([[float("nan"), float("inf"), 0.0, 0.0]])
        mapped = SimEnvAdapter.action_to_command(nonfinite, torch.tensor([[0.42]]))
        self.assertEqual(mapped[0, 0], 0.42)
        self.assertTrue(torch.isnan(mapped[0, 1]))
        from simenv import SimulationEnvironment
        with tempfile.TemporaryDirectory() as raw:
            config = load_experiment_config(_write_run_config(Path(raw)))
            env = SimulationEnvironment.create(config.simulator_config, 1, "cpu")
            try:
                result = env.advance(mapped)
                self.assertFalse(result.valid[0])
                self.assertNotEqual(int(result.error_code[0]), 0)
            finally:
                env.close()

    def test_03_quaternion_sign_equivalence_in_error_and_reward(self):
        q = torch.tensor([[0.9238795, 0.3826834, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        positive = quaternion_geodesic_angle(q, target)
        negative = quaternion_geodesic_angle(-q, target)
        torch.testing.assert_close(positive, negative)
        calculator = AttitudeRewardCalculator()
        base = {
            "angular_velocity_b": torch.zeros(1, 3),
            "action": torch.zeros(1, 4),
            "previous_action": torch.zeros(1, 4),
        }
        first = calculator(TensorDict({**base, "attitude_geodesic_rad": positive}, [1]))
        second = calculator(TensorDict({**base, "attitude_geodesic_rad": negative}, [1]))
        torch.testing.assert_close(first.reward, second.reward)

    def test_04_rnn_reset_is_per_instance_and_chunking_keeps_hidden(self):
        torch.manual_seed(11)
        model = _model()
        observation = torch.randn(2, 21)
        state = torch.ones(2, 1, 8)
        td = TensorDict(
            {
                "observation": observation,
                "recurrent_state": state,
                "is_init": torch.tensor([[True], [False]]),
            },
            [2],
        )
        model.policy_module(td)
        zero = TensorDict(
            {
                "observation": observation[:1],
                "recurrent_state": torch.zeros(1, 1, 8),
                "is_init": torch.zeros(1, 1, dtype=torch.bool),
            },
            [1],
        )
        model.policy_module(zero)
        torch.testing.assert_close(td[("next", "recurrent_state")][:1], zero[("next", "recurrent_state")])
        self.assertFalse(torch.equal(td[("next", "recurrent_state")][1], zero[("next", "recurrent_state")][0]))

        rollout = TensorDictRolloutCollector(DeterministicTensorEnv(2), model, 4).collect()
        self.assertFalse(rollout["is_init"][:, 2].any())
        self.assertTrue(torch.count_nonzero(rollout["recurrent_state"][:, 2]) > 0)

    def test_05_invalid_instance_is_excluded_without_masking_peers(self):
        def collect(invalid_at):
            torch.manual_seed(12)
            local_model = _model()
            local_rollout = TensorDictRolloutCollector(
                DeterministicTensorEnv(3, invalid_at=invalid_at), local_model, 4
            ).collect()
            return local_model, local_rollout

        model, rollout = collect((2, 1))
        _reference_model, reference = collect(None)
        self.assertFalse(rollout[("next", "valid")][1, 1, 0])
        self.assertTrue(rollout[("next", "valid")][0].all())
        self.assertTrue(rollout[("next", "valid")][2].all())
        torch.testing.assert_close(rollout["action"][[0, 2]], reference["action"][[0, 2]])
        torch.testing.assert_close(
            rollout[("next", "observation")][[0, 2]],
            reference[("next", "observation")][[0, 2]],
        )
        algorithm = RecurrentPPO(
            model,
            PPOConfig(0.99, 0.95, 0.2, 0.0, 0.5, 1.0, 3e-4, 1, 2, 2),
            torch.device("cpu"),
        )
        algorithm.update(rollout)
        torch.testing.assert_close(rollout["shifted_valid"], rollout[("next", "valid")])

    def test_06_deterministic_short_training_has_identical_tensor_digest(self):
        def tensor_trace():
            torch.manual_seed(20260721)
            model = _model()
            initial = tensor_state_sha256(model.actor.state_dict())
            rollout = TensorDictRolloutCollector(DeterministicTensorEnv(2), model, 4).collect()
            algorithm = RecurrentPPO(
                model,
                PPOConfig(0.99, 0.95, 0.2, 0.001, 0.5, 1.0, 3e-4, 1, 2, 2),
                torch.device("cpu"),
            )
            metrics = algorithm.update(rollout)
            return {
                "initial": initial,
                "actions": rollout["action"].clone(),
                "rewards": rollout[("next", "reward")].clone(),
                "losses": {key: value.clone() for key, value in metrics.items()},
                "final": tensor_state_sha256(model.actor.state_dict()),
            }

        trace_a = tensor_trace()
        trace_b = tensor_trace()
        self.assertEqual(trace_a["initial"], trace_b["initial"])
        torch.testing.assert_close(trace_a["actions"], trace_b["actions"])
        torch.testing.assert_close(trace_a["rewards"], trace_b["rewards"])
        for key in trace_a["losses"]:
            torch.testing.assert_close(trace_a["losses"][key], trace_b["losses"][key])
        self.assertEqual(trace_a["final"], trace_b["final"])

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config_path = _write_run_config(root)
            first = run_experiment(load_experiment_config(config_path))
            second = run_experiment(load_experiment_config(config_path))
            first_state = load_checkpoint(next((first["run_directory"] / "checkpoints").glob("*.pt")))
            second_state = load_checkpoint(next((second["run_directory"] / "checkpoints").glob("*.pt")))
            selected = lambda state: {
                "actor": state["actor"],
                "critic": state["critic"],
                "actor_optimizer": state["actor_optimizer"],
                "critic_optimizer": state["critic_optimizer"],
                "steps": state["global_control_steps"],
            }
            self.assertEqual(
                tensor_state_sha256(selected(first_state)),
                tensor_state_sha256(selected(second_state)),
            )
            first_metrics = json.loads((first["run_directory"] / "metrics.jsonl").read_text().splitlines()[0])
            second_metrics = json.loads((second["run_directory"] / "metrics.jsonl").read_text().splitlines()[0])
            first_metrics.pop("utc")
            second_metrics.pop("utc")
            self.assertEqual(first_metrics, second_metrics)

    def test_07_exact_restore_tensor_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base_path = _write_run_config(root, name="exact_resume")
            base = json.loads(base_path.read_text())

            continuous_raw = copy.deepcopy(base)
            continuous_raw["run"]["total_control_steps"] = 16
            continuous_path = root / "continuous.json"
            continuous_path.write_text(json.dumps(continuous_raw))
            continuous = run_experiment(load_experiment_config(continuous_path))

            prefix_raw = copy.deepcopy(base)
            prefix_raw["run"]["total_control_steps"] = 8
            prefix_path = root / "prefix.json"
            prefix_path.write_text(json.dumps(prefix_raw))
            prefix = run_experiment(load_experiment_config(prefix_path))
            prefix_checkpoint = next((prefix["run_directory"] / "checkpoints").glob("*.pt"))

            resumed_raw = copy.deepcopy(base)
            resumed_raw["run"]["total_control_steps"] = 16
            resumed_raw["checkpoint"] = {
                "resume": {"from": str(prefix_checkpoint), "mode": "exact"}
            }
            resumed_path = root / "resumed.json"
            resumed_path.write_text(json.dumps(resumed_raw))
            resumed = run_experiment(load_experiment_config(resumed_path))

            continuous_state = load_checkpoint(
                next((continuous["run_directory"] / "checkpoints").glob("*.pt"))
            )
            resumed_state = load_checkpoint(
                next((resumed["run_directory"] / "checkpoints").glob("*.pt"))
            )

            def numerical_state(state):
                simulator = dict(state["simulator_state"])
                simulator.pop("source_identity")
                return {
                    "global_control_steps": state["global_control_steps"],
                    "actor": state["actor"],
                    "critic": state["critic"],
                    "actor_optimizer": state["actor_optimizer"],
                    "critic_optimizer": state["critic_optimizer"],
                    "torch_rng_state": state["torch_rng_state"],
                    "collector_current": state["collector_current"],
                    "simulator_state": simulator,
                    "training_environment": state["training_environment"],
                }

            self.assertEqual(
                tensor_state_sha256(numerical_state(continuous_state)),
                tensor_state_sha256(numerical_state(resumed_state)),
            )
            continuous_metrics = json.loads(
                (continuous["run_directory"] / "metrics.jsonl").read_text().splitlines()[-1]
            )
            resumed_metrics = json.loads(
                (resumed["run_directory"] / "metrics.jsonl").read_text().splitlines()[-1]
            )
            continuous_metrics.pop("utc")
            resumed_metrics.pop("utc")
            self.assertEqual(continuous_metrics, resumed_metrics)
            manifest = json.loads((resumed["run_directory"] / "manifest.json").read_text())
            self.assertTrue(manifest["exact_resume_supported"])
            self.assertEqual(manifest["resume_mode"], "exact")
            prefix_state = load_checkpoint(prefix_checkpoint)
            self.assertEqual(manifest["parent_run_id"], prefix_state["run_id"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable in this acceptance environment")
    def test_07_exact_restore_tensor_identity_cuda(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base_path = _write_run_config(root, name="exact_resume_cuda")
            base = json.loads(base_path.read_text())
            base["run"]["device"] = "cuda:0"

            def run_with(total_steps, name, resume_from=None):
                value = copy.deepcopy(base)
                value["run"]["total_control_steps"] = total_steps
                if resume_from is not None:
                    value["checkpoint"] = {
                        "resume": {"from": str(resume_from), "mode": "exact"}
                    }
                path = root / f"{name}.json"
                path.write_text(json.dumps(value))
                return run_experiment(load_experiment_config(path))

            continuous = run_with(16, "continuous_cuda")
            prefix = run_with(8, "prefix_cuda")
            prefix_checkpoint = next((prefix["run_directory"] / "checkpoints").glob("*.pt"))
            resumed = run_with(16, "resumed_cuda", prefix_checkpoint)
            continuous_state = load_checkpoint(
                next((continuous["run_directory"] / "checkpoints").glob("*.pt"))
            )
            resumed_state = load_checkpoint(
                next((resumed["run_directory"] / "checkpoints").glob("*.pt"))
            )
            for key in (
                "actor", "critic", "actor_optimizer", "critic_optimizer",
                "collector_current",
                "training_environment", "torch_rng_state", "cuda_rng_state_all",
            ):
                self.assertEqual(
                    tensor_state_sha256(continuous_state[key]),
                    tensor_state_sha256(resumed_state[key]),
                )
            simulator_a = dict(continuous_state["simulator_state"])
            simulator_b = dict(resumed_state["simulator_state"])
            simulator_a.pop("source_identity")
            simulator_b.pop("source_identity")
            self.assertEqual(
                tensor_state_sha256(simulator_a), tensor_state_sha256(simulator_b)
            )

    def test_08_records_link_instances_logs_updates_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = run_experiment(load_experiment_config(_write_run_config(root)))
            run_dir = result["run_directory"]
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertTrue(manifest["exact_resume_supported"])
            self.assertEqual(len(manifest["simulator_instance_ids"]), 2)
            sim_dir = Path(manifest["simulator_log_directory"])
            metadata = json.loads((sim_dir / "metadata.json").read_text())
            self.assertEqual(metadata["batch_id"], manifest["simulator_batch_id"])
            self.assertTrue((sim_dir / "parameters.pt").is_file())
            timelines = list(sim_dir.glob("timeline_*.pt"))
            self.assertTrue(timelines)
            timeline = torch.load(timelines[0], weights_only=False)
            self.assertIn("control", timeline)
            self.assertIn("generation", timeline)
            self.assertTrue((sim_dir / "resets.jsonl").is_file())
            resets = [json.loads(line) for line in (sim_dir / "resets.jsonl").read_text().splitlines()]
            linked_ids = {
                instance["instance_id"]
                for event in resets
                for instance in event["instances"]
            }
            self.assertTrue(set(manifest["simulator_instance_ids"]).issubset(linked_ids))
            reset_snapshot = torch.load(
                sim_dir / "resets/000000/parameters.pt", weights_only=False
            )
            self.assertIn("body.mass", reset_snapshot["parameters"])
            metric_lines = (run_dir / "metrics.jsonl").read_text().splitlines()
            self.assertTrue(metric_lines)
            metrics = json.loads(metric_lines[-1])
            for key in (
                "terminated_fraction",
                "truncated_fraction",
                "done_fraction",
                "invalid_fraction",
                "episode_reset_count",
                "attitude_error_mean_deg",
                "attitude_error_p95_deg",
                "angular_rate_mean_rad_s",
                "height_error_abs_mean_m",
                "action_rms",
                "action_peak_abs",
                "action_saturation_fraction",
                "completed_episode_length_mean_steps",
                "completed_episode_survival_mean_s",
            ):
                self.assertIn(key, metrics)
            checkpoint = next((run_dir / "checkpoints").glob("*.pt"))
            self.assertTrue(checkpoint.with_suffix(".pt.sha256").is_file())

    def test_09_bad_configs_fail_before_sampling(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base_path = _write_run_config(root)
            base = json.loads(base_path.read_text())

            cases = []
            unknown = copy.deepcopy(base)
            unknown["unknown"] = 1
            cases.append((unknown, ConfigError))
            leak = copy.deepcopy(base)
            leak["reward"]["context"]["fields"].append(
                {"name": "position_n", "source": "truth", "visibility": "student"}
            )
            cases.append((leak, ConfigError))
            sequence = copy.deepcopy(base)
            sequence["collector"]["control_steps_per_rollout"] = 3
            cases.append((sequence, ConfigError))
            duplicate_baseline = copy.deepcopy(base)
            duplicate_baseline["randomization"]["static"]["parameters"]["body.mass"][
                "baseline"
            ] = 2.4
            cases.append((duplicate_baseline, ConfigError))
            stale_task_command = copy.deepcopy(base)
            stale_task_command["task"]["command"] = {"max_tilt_rad": 0.2}
            cases.append((stale_task_command, ConfigError))
            bad_pilot = copy.deepcopy(base)
            bad_pilot["command_source"]["params"]["throttle"]["height_controller"]["observation_source"] = "sensor"
            cases.append((bad_pilot, ConfigError))

            for index, (value, error) in enumerate(cases):
                path = root / f"bad_{index}.json"
                path.write_text(json.dumps(value))
                with self.assertRaises(error):
                    config = load_experiment_config(path)
                    run_experiment(config)

            illegal_sim = json.loads(Path(base["environment"]["config_path"]).read_text())
            illegal_sim["timing"]["control_hz"]["value"] = 333
            illegal_path = root / "illegal_timing.json"
            illegal_path.write_text(json.dumps(illegal_sim))
            illegal = copy.deepcopy(base)
            illegal["environment"]["config_path"] = str(illegal_path)
            illegal_experiment = root / "bad_timing_experiment.json"
            illegal_experiment.write_text(json.dumps(illegal))
            with self.assertRaises(Exception):
                run_experiment(load_experiment_config(illegal_experiment))

    def test_10_evaluation_does_not_change_training_rng_or_model_mode(self):
        torch.manual_seed(13)
        model = _model()
        model.actor.train(True)
        before = torch.get_rng_state().clone()
        collect_evaluation_rollout(DeterministicTensorEnv(2), model, 3)
        after = torch.get_rng_state()
        torch.testing.assert_close(before, after)
        self.assertTrue(model.actor.training)

        def train_once(insert_evaluation: bool):
            torch.manual_seed(1300)
            local_model = _model()
            if insert_evaluation:
                collect_evaluation_rollout(DeterministicTensorEnv(2), local_model, 2)
            rollout = TensorDictRolloutCollector(
                DeterministicTensorEnv(2), local_model, 4
            ).collect()
            algorithm = RecurrentPPO(
                local_model,
                PPOConfig(0.99, 0.95, 0.2, 0.001, 0.5, 1.0, 3e-4, 1, 2, 2),
                torch.device("cpu"),
            )
            metrics = algorithm.update(rollout)
            return rollout["action"], metrics, tensor_state_sha256(local_model.actor.state_dict())

        actions_a, metrics_a, digest_a = train_once(False)
        actions_b, metrics_b, digest_b = train_once(True)
        torch.testing.assert_close(actions_a, actions_b)
        for key in metrics_a:
            torch.testing.assert_close(metrics_a[key], metrics_b[key])
        self.assertEqual(digest_a, digest_b)

    def test_11_checkpoint_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            result = run_experiment(
                load_experiment_config(_write_run_config(Path(raw)))
            )
            checkpoint = next((result["run_directory"] / "checkpoints").glob("*.pt"))
            with checkpoint.open("r+b") as stream:
                stream.seek(32)
                original = stream.read(1)
                stream.seek(32)
                stream.write(bytes([original[0] ^ 0xFF]))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_checkpoint(checkpoint)

        with tempfile.TemporaryDirectory() as raw:
            config = load_experiment_config(_write_run_config(Path(raw), name="atomic"))
            recorder = RunRecorder(config)
            try:
                previous = recorder.checkpoint(
                    4, {"value": torch.tensor([1.0])}, kind="periodic"
                )
                with mock.patch(
                    "flight_train.recording.torch.save",
                    side_effect=RuntimeError("simulated interruption"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        recorder.checkpoint(
                            8, {"value": torch.tensor([2.0])}, kind="periodic"
                        )
                restored = load_checkpoint(previous)
                torch.testing.assert_close(restored["value"], torch.tensor([1.0]))
                for step in (8, 12, 16):
                    recorder.checkpoint(
                        step, {"value": torch.tensor([float(step)])}, kind="periodic"
                    )
                self.assertFalse(previous.exists())
                index = json.loads(
                    (recorder.directory / "checkpoints" / "index.json").read_text()
                )
                self.assertEqual(
                    [item["global_control_steps"] for item in index["checkpoints"]],
                    [8, 12, 16],
                )
            finally:
                recorder.close("failed", 4)

    def test_11a_best_checkpoint_requires_quality_and_uses_composite_score(self):
        with tempfile.TemporaryDirectory() as raw:
            config = load_experiment_config(
                _write_run_config(Path(raw), name="best-selection")
            )
            recorder = RunRecorder(config)
            try:
                first = recorder.checkpoint(
                    4, {"value": torch.tensor([1.0])}, kind="periodic"
                )
                second = recorder.checkpoint(
                    8, {"value": torch.tensor([2.0])}, kind="periodic"
                )
                self.assertFalse(
                    recorder.promote_best_evaluation_checkpoint(
                        first,
                        control_steps=4,
                        hover_survival_s=5.0,
                        hover_roll_pitch_rmse_deg=20.0,
                        hover_yaw_rate_rmse_rad_s=1.0,
                        total_score=90.0,
                        quality_passed=False,
                    )
                )
                self.assertTrue(
                    recorder.promote_best_evaluation_checkpoint(
                        first,
                        control_steps=4,
                        hover_survival_s=4.0,
                        hover_roll_pitch_rmse_deg=5.0,
                        hover_yaw_rate_rmse_rad_s=0.2,
                        total_score=80.0,
                    )
                )
                # 更长生存不能覆盖综合得分更高的已选模型。
                self.assertFalse(
                    recorder.promote_best_evaluation_checkpoint(
                        second,
                        control_steps=8,
                        hover_survival_s=8.0,
                        hover_roll_pitch_rmse_deg=5.0,
                        hover_yaw_rate_rmse_rad_s=0.2,
                        total_score=70.0,
                    )
                )
                self.assertTrue(
                    recorder.promote_best_evaluation_checkpoint(
                        second,
                        control_steps=8,
                        hover_survival_s=3.0,
                        hover_roll_pitch_rmse_deg=5.0,
                        hover_yaw_rate_rmse_rad_s=0.2,
                        total_score=85.0,
                    )
                )
                metadata = json.loads(
                    (
                        recorder.directory
                        / "checkpoints"
                        / "best_fixed_evaluation.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["global_control_steps"], 8)
                self.assertEqual(metadata["total_score"], 85.0)
                self.assertEqual(metadata["source_checkpoint"], second.name)
            finally:
                recorder.close("completed", 8)

    def test_11b_evaluation_checkpoints_preserve_independent_objectives(self):
        with tempfile.TemporaryDirectory() as raw:
            config = load_experiment_config(
                _write_run_config(Path(raw), name="multi-best-selection")
            )
            recorder = RunRecorder(config)
            try:
                first = recorder.checkpoint(
                    4, {"value": torch.tensor([1.0])}, kind="periodic"
                )
                second = recorder.checkpoint(
                    8, {"value": torch.tensor([2.0])}, kind="periodic"
                )
                third = recorder.checkpoint(
                    12, {"value": torch.tensor([3.0])}, kind="periodic"
                )
                self.assertEqual(
                    recorder.promote_evaluation_checkpoints(
                        first,
                        control_steps=4,
                        hover_survival_s=5.0,
                        hover_roll_pitch_rmse_deg=20.0,
                        hover_yaw_rate_rmse_rad_s=0.1,
                        total_score=90.0,
                        minimum_hover_survival_s=8.0,
                        quality_passed=False,
                    ),
                    ("total", "upright", "yaw"),
                )
                self.assertEqual(
                    recorder.promote_evaluation_checkpoints(
                        second,
                        control_steps=8,
                        hover_survival_s=10.0,
                        hover_roll_pitch_rmse_deg=5.0,
                        hover_yaw_rate_rmse_rad_s=2.0,
                        total_score=80.0,
                        minimum_hover_survival_s=8.0,
                        quality_passed=False,
                    ),
                    ("upright",),
                )
                self.assertEqual(
                    recorder.promote_evaluation_checkpoints(
                        third,
                        control_steps=12,
                        hover_survival_s=10.0,
                        hover_roll_pitch_rmse_deg=6.0,
                        hover_yaw_rate_rmse_rad_s=0.05,
                        total_score=70.0,
                        minimum_hover_survival_s=8.0,
                        quality_passed=True,
                    ),
                    ("yaw", "fixed"),
                )
                checkpoint_dir = recorder.directory / "checkpoints"
                selections = {
                    name: json.loads(
                        (
                            checkpoint_dir / f"best_{name}_evaluation.json"
                        ).read_text(encoding="utf-8")
                    )
                    for name in ("total", "upright", "yaw", "fixed")
                }
                self.assertEqual(
                    selections["total"]["global_control_steps"], 4
                )
                self.assertEqual(
                    selections["upright"]["global_control_steps"], 8
                )
                self.assertEqual(
                    selections["yaw"]["global_control_steps"], 12
                )
                self.assertEqual(
                    selections["fixed"]["global_control_steps"], 12
                )
                self.assertFalse(
                    selections["total"]["quality_gate_passed"]
                )
                self.assertTrue(
                    selections["fixed"]["quality_gate_passed"]
                )
                # 同一次保存的 best 使用硬链接，不重复占用 checkpoint 数据块。
                self.assertEqual(
                    first.stat().st_ino,
                    (checkpoint_dir / "best_total_evaluation.pt").stat().st_ino,
                )
            finally:
                recorder.close("completed", 12)

    def test_12_minimal_learning_smoke_updates_evaluates_saves_and_loads(self):
        torch.manual_seed(14)
        model = _model()
        rollout = TensorDictRolloutCollector(DeterministicTensorEnv(2), model, 4).collect()
        algorithm = RecurrentPPO(
            model,
            PPOConfig(0.99, 0.95, 0.2, 0.001, 0.5, 1.0, 3e-4, 1, 2, 2),
            torch.device("cpu"),
        )
        metrics = algorithm.update(rollout)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        evaluation = collect_evaluation_rollout(DeterministicTensorEnv(2), model, 2)
        self.assertTrue(torch.isfinite(evaluation[("next", "reward")]).all())
        with tempfile.TemporaryDirectory() as raw:
            result = run_experiment(load_experiment_config(_write_run_config(Path(raw))))
            state = load_checkpoint(next((result["run_directory"] / "checkpoints").glob("*.pt")))
            restored = build_actor_critic(
                21, 4, ModelConfig((64, 64), 64, 64), torch.device("cpu")
            )
            restored.actor.load_state_dict(state["actor"])
            restored.critic.load_state_dict(state["critic"])
            self.assertEqual(
                tensor_state_sha256(restored.actor.state_dict()),
                tensor_state_sha256(state["actor"]),
            )

    def test_13_static_and_dynamic_seed_streams_are_isolated(self):
        spec = StaticParameterSpec(
            "body.mass", -0.1, 0.1, "uniform", "relative", "ratio", "mass"
        )
        episodes = torch.tensor([1, 1, 1])
        mask = torch.ones(3, dtype=torch.bool)

        def sample(seed):
            randomizer = StaticRandomizer(
                (spec,), seed, torch.device("cpu"), torch.float32
            )
            randomizer.bind_baselines({"body.mass": torch.full((3,), 2.4)})
            return randomizer.sample(mask, episodes)

        first = sample(101)
        same = sample(101)
        changed = sample(102)
        torch.testing.assert_close(first["body.mass"], same["body.mass"])
        self.assertFalse(torch.equal(first["body.mass"], changed["body.mass"]))

        with tempfile.TemporaryDirectory() as raw:
            config_path = _write_run_config(Path(raw))
            config = load_experiment_config(config_path)
            from simenv import SimulationEnvironment

            kwargs = dict(
                config_path=config.simulator_config,
                parallel_count=3,
                device="cpu",
                dynamic_randomization=dict(config.dynamic_randomization.parameters),
            )
            env_a = SimulationEnvironment.create(**kwargs, dynamic_seed=201)
            env_b = SimulationEnvironment.create(**kwargs, dynamic_seed=201)
            env_c = SimulationEnvironment.create(**kwargs, dynamic_seed=202)
            try:
                key = "sensors.gyro.noise.stddev"
                torch.testing.assert_close(env_a.parameters[key], env_b.parameters[key])
                self.assertFalse(torch.equal(env_a.parameters[key], env_c.parameters[key]))
            finally:
                env_a.close(); env_b.close(); env_c.close()

    def test_14_static_parameters_change_only_on_masked_reset(self):
        from simenv import SimulationEnvironment

        with tempfile.TemporaryDirectory() as raw:
            config = load_experiment_config(_write_run_config(Path(raw)))
            env = SimulationEnvironment.create(config.simulator_config, 4, "cpu")
            mask = torch.tensor([False, True, False, True])
            initial = torch.tensor([2.2, 2.3, 2.4, 2.5])
            replacement = torch.tensor([2.6, 2.7, 2.8, 2.9])
            try:
                env.reset(
                    torch.ones(4, dtype=torch.bool),
                    config.simulator_config,
                    static_parameters={"body.mass": initial},
                )
                before = env.parameters["body.mass"]
                env.advance(torch.zeros(4, 5))
                torch.testing.assert_close(env.parameters["body.mass"], before)
                env.reset(
                    mask,
                    config.simulator_config,
                    static_parameters={"body.mass": replacement},
                )
                after = env.parameters["body.mass"]
                torch.testing.assert_close(after[~mask], before[~mask])
                torch.testing.assert_close(after[mask], replacement[mask])
                env.advance(torch.zeros(4, 5))
                torch.testing.assert_close(env.parameters["body.mass"], after)
            finally:
                env.close()

    def test_15_simenv_owns_and_replays_dynamic_processes(self):
        from simenv import SimulationEnvironment

        with tempfile.TemporaryDirectory() as raw:
            config_path = _write_run_config(Path(raw))
            config = load_experiment_config(config_path)
            kwargs = dict(
                config_path=config.simulator_config,
                parallel_count=2,
                device="cpu",
                dynamic_randomization=dict(config.dynamic_randomization.parameters),
                dynamic_seed=401,
            )
            env_a = SimulationEnvironment.create(**kwargs)
            env_b = SimulationEnvironment.create(**kwargs)
            control = torch.tensor([[0.6, 0.6, 0.0, 0.0, 0.0]]).expand(2, -1)
            try:
                for _ in range(3):
                    env_a.advance(control)
                    env_b.advance(control)
                    torch.testing.assert_close(
                        env_a.observe("sensor").values["gyro"],
                        env_b.observe("sensor").values["gyro"],
                    )
                self.assertTrue((env_a.log_directory / "parameters.pt").is_file())
            finally:
                env_a.close(); env_b.close()

    def test_16_reward_calculator_is_replaceable_without_schema_changes(self):
        class ConstantReward:
            def __init__(self, value): self.value = value
            def __call__(self, context):
                reward = torch.full((context.batch_size[0], 1), self.value, device=context.device)
                return RewardOutput(reward, TensorDict({}, context.batch_size, device=context.device))
            def state_dict(self): return {}
            def load_state_dict(self, state): del state

        cfg = TaskConfig(1.0, 2.0, 30.0, "truth")
        inputs = (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
            torch.zeros(2, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
            torch.zeros(2, 4), torch.zeros(2, 4),
        )
        first = AttitudeTrackingTask(cfg, 2, torch.device("cpu"), torch.float32, ConstantReward(1.0))
        second = AttitudeTrackingTask(cfg, 2, torch.device("cpu"), torch.float32, ConstantReward(2.0))
        result_a = first.transition(*inputs)
        result_b = second.transition(*inputs)
        self.assertEqual(result_a.reward.shape, result_b.reward.shape)
        self.assertFalse(torch.equal(result_a.reward, result_b.reward))
        self.assertEqual(set(result_a.info), set(result_b.info))

    def test_17_reward_tensor_boundary_cpu_and_hot_path_source(self):
        source = Path(ROOT / "src/flight_train/rewards.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {"cpu", "numpy", "item"}
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(forbidden.isdisjoint(calls))
        reward_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "__call__"
        )
        self.assertFalse(any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(reward_call)))
        calculator = AttitudeRewardCalculator()
        for batch in (1, 5):
            context = TensorDict(
                {
                    "attitude_geodesic_rad": torch.zeros(batch, 1),
                    "angular_velocity_b": torch.zeros(batch, 3),
                    "action": torch.zeros(batch, 4),
                },
                [batch], device=torch.device("cpu"),
            )
            output = calculator(context)
            self.assertEqual(output.reward.shape, (batch, 1))
            self.assertEqual(output.reward.device, context.device)
            self.assertTrue(torch.isfinite(output.reward).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable in this acceptance environment")
    def test_17_reward_tensor_boundary_cuda(self):
        device = torch.device("cuda:0")
        calculator = AttitudeRewardCalculator()
        for batch in (1, 5):
            context = TensorDict(
                {
                    "attitude_geodesic_rad": torch.zeros(batch, 1, device=device),
                    "angular_velocity_b": torch.zeros(batch, 3, device=device),
                    "action": torch.zeros(batch, 4, device=device),
                },
                [batch], device=device,
            )
            output = calculator(context)
            self.assertEqual(output.reward.shape, (batch, 1))
            self.assertEqual(output.reward.device, device)
            self.assertTrue(torch.isfinite(output.reward).all())

    def test_18_virtual_pilot_owns_upper_motor_and_stick_limits(self):
        config = load_experiment_config(EXPERIMENT_CONFIG).command_source
        pilot = VirtualPilotCommandSource(
            config, 32, torch.device("cpu"), torch.float32, control_hz=500
        )
        pilot.reset(torch.ones(32, dtype=torch.bool))
        first = SimEnvAdapter.action_to_command(
            torch.full((32, 4), -1.0), pilot.upper_throttle
        )
        second = SimEnvAdapter.action_to_command(
            torch.full((32, 4), 1.0), pilot.upper_throttle
        )
        torch.testing.assert_close(first[:, 0], second[:, 0])
        self.assertFalse(torch.equal(first[:, 1:], second[:, 1:]))

        pilot.hold_remaining.zero_()
        pilot.step(torch.full((32, 1), 5.0))
        normalized_tilt = torch.sqrt(
            (pilot.stick_target[:, 0] / config.max_roll_rad).square()
            + (pilot.stick_target[:, 1] / config.max_pitch_rad).square()
        )
        self.assertTrue((normalized_tilt <= 1.0 + 1e-6).all())
        self.assertTrue((pilot.target_yaw >= -torch.pi).all())
        self.assertTrue((pilot.target_yaw < torch.pi).all())

    def test_19_virtual_pilot_incremental_height_control_and_reset_isolation(self):
        config = load_experiment_config(EXPERIMENT_CONFIG).command_source
        pilot = VirtualPilotCommandSource(
            config, 3, torch.device("cpu"), torch.float32, control_hz=500
        )
        pilot.reset(torch.ones(3, dtype=torch.bool))
        pilot.spool_remaining.zero_()
        pilot.stick_target[:] = torch.tensor([0.2, -0.1, 0.5])
        pilot.filtered_stick.zero_()
        pilot.hold_remaining.fill_(1.0)
        pilot.height_controller_output.fill_(0.5)
        pilot.upper_throttle.fill_(0.5)
        pilot.step(torch.tensor([[-1.0], [0.0], [1.0]]))
        tau = torch.tensor([
            config.roll_time_constant_s,
            config.pitch_time_constant_s,
            config.yaw_time_constant_s,
        ])
        expected = (1.0 - torch.exp(-torch.tensor(1.0 / 500.0) / tau)) * torch.tensor(
            [0.2, -0.1, 0.5]
        )
        torch.testing.assert_close(pilot.filtered_stick[0], expected)
        torch.testing.assert_close(pilot.height_error[:, 0], torch.tensor([1.0, 0.0, -1.0]))
        expected_increment = config.height_proportional_gain + config.height_integral_gain / 500.0
        torch.testing.assert_close(
            pilot.height_controller_output[:, 0],
            torch.tensor([0.5 + expected_increment, 0.5, 0.5 - expected_increment]),
        )

        preserved = pilot.filtered_stick[1:].clone()
        pilot.reset(torch.tensor([True, False, False]))
        torch.testing.assert_close(pilot.filtered_stick[0], torch.zeros(3))
        torch.testing.assert_close(pilot.filtered_stick[1:], preserved)

        source = Path(ROOT / "src/flight_train/commands.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        step = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "step"
        )
        self.assertFalse(any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(step)))
        forbidden = {"cpu", "numpy", "item"}
        calls = {
            node.func.attr
            for node in ast.walk(step)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(forbidden.isdisjoint(calls))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable in this acceptance environment")
    def test_19_virtual_pilot_cuda_residency(self):
        device = torch.device("cuda:0")
        config = load_experiment_config(EXPERIMENT_CONFIG).command_source
        pilot = VirtualPilotCommandSource(config, 8, device, torch.float32, control_hz=500)
        pilot.reset(torch.ones(8, dtype=torch.bool, device=device))
        pilot.step(torch.full((8, 1), 5.0, device=device))
        for name in (
            "upper_throttle", "throttle_target", "spool_remaining",
            "height_controller_output", "height_previous_error", "height_error", "height_target",
            "stick_target", "filtered_stick", "hold_remaining", "target_yaw", "target_attitude",
        ):
            self.assertEqual(getattr(pilot, name).device, device)
        command = SimEnvAdapter.action_to_command(
            torch.zeros(8, 4, device=device), pilot.upper_throttle
        )
        self.assertEqual(command.device, device)
        self.assertEqual(command.shape, (8, 5))


if __name__ == "__main__":
    unittest.main()
