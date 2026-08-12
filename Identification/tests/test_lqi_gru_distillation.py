from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
for source in ("Identification/src", "Controller/src", "SimEnv/src", "Train/src"):
    path = str(ROOT / source)
    if path not in sys.path:
        sys.path.insert(0, path)

from flight_identification.lqi_gru_distillation import (  # noqa: E402
    CausalLQIStudent,
    _masked_loss,
    fit_observation_normalization,
    load_sequences,
    select_observations,
)
from flight_identification.lqi_gru_experiment import (  # noqa: E402
    ACTION_NAMES,
    EXTERNAL_UPPER_ACTION_NAMES,
    OBSERVATION_NAMES,
    _oracle_gains,
)


class LQIGRUDistillationTests(unittest.TestCase):
    def _write_shard(self, root: Path, split: str = "train") -> None:
        directory = root / split
        directory.mkdir(parents=True)
        torch.save(
            {
                "schema_version": 1,
                "observations": torch.randn(3, 7, len(OBSERVATION_NAMES)),
                "teacher_actions": torch.randn(3, 7, 5).clamp(-1, 1),
                "valid_mask": torch.ones(3, 7, dtype=torch.bool),
                "parameter_labels_audit_only": torch.randn(3, 4),
                "oracle_lqi_gain_audit_only": torch.randn(3, 5, 13),
                "group_id": torch.arange(3),
                "observation_names": OBSERVATION_NAMES,
                "action_names": ACTION_NAMES,
                "label_names": ("p0", "p1", "p2", "p3"),
                "leakage_contract": "audit tensors are never student inputs",
            },
            directory / "shard_000000.pt",
        )

    def test_loader_keeps_audit_parameters_out_of_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_shard(root)
            values = load_sequences(root, "train")
        self.assertEqual(values.observations.shape[-1], len(OBSERVATION_NAMES))
        self.assertEqual(values.labels_audit_only.shape[-1], 4)
        self.assertFalse(any("parameter" in name for name in values.observation_names))
        self.assertEqual(
            tuple(values.observation_names[-5:]),
            tuple(f"previous_command.{name.split('.', 1)[1]}" for name in ACTION_NAMES),
        )
        mean, std = fit_observation_normalization(values)
        self.assertEqual(tuple(mean.shape), (len(OBSERVATION_NAMES),))
        self.assertTrue((std > 0).all())

    def test_loader_rejects_ambiguous_non_audit_parameter_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_shard(root)
            path = root / "train" / "shard_000000.pt"
            shard = torch.load(path, weights_only=False)
            shard["parameter_labels"] = shard["parameter_labels_audit_only"]
            torch.save(shard, path)
            with self.assertRaisesRegex(ValueError, "leakage contract"):
                load_sequences(root, "train")

    def test_twenty_dimensional_subset_removes_only_previous_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_shard(root)
            values = load_sequences(root, "train")
            selected_names = tuple(
                name
                for name in values.observation_names
                if not name.startswith("previous_command.")
            )
            subset = select_observations(values, selected_names)
        self.assertEqual(subset.observations.shape[-1], 20)
        self.assertEqual(subset.observation_names, OBSERVATION_NAMES[:20])
        self.assertTrue(torch.equal(subset.actions, values.actions))
        self.assertTrue(torch.equal(subset.group_id, values.group_id))

    def test_causal_student_outputs_bounded_five_axis_commands(self) -> None:
        torch.manual_seed(3)
        model = CausalLQIStudent(len(OBSERVATION_NAMES), 16, 2, (12,), 0.0)
        observation = torch.randn(4, 9, len(OBSERVATION_NAMES))
        prediction, hidden, encoded = model(observation)
        self.assertEqual(tuple(prediction.shape), (4, 9, 5))
        self.assertEqual(tuple(hidden.shape), (2, 4, 16))
        self.assertEqual(tuple(encoded.shape), (4, 9, 16))
        self.assertTrue((prediction[..., :2] >= 0).all())
        self.assertTrue((prediction[..., :2] <= 1).all())
        self.assertTrue((prediction[..., 2:].abs() <= 1).all())
        loss = _masked_loss(prediction, torch.zeros_like(prediction), torch.ones(4, 9, dtype=torch.bool), 0.2)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_causal_student_supports_native_four_output_contract(self) -> None:
        model = CausalLQIStudent(
            len(OBSERVATION_NAMES),
            16,
            2,
            (12,),
            0.0,
            action_size=len(EXTERNAL_UPPER_ACTION_NAMES),
            motor_action_count=1,
        )
        prediction, _, _ = model(torch.randn(3, 11, len(OBSERVATION_NAMES)))
        self.assertEqual(tuple(prediction.shape), (3, 11, 4))
        self.assertTrue((prediction[..., :1] >= 0.0).all())
        self.assertTrue((prediction[..., :1] <= 1.0).all())
        self.assertTrue((prediction[..., 1:].abs() <= 1.0).all())

    def test_oracle_designs_one_augmented_gain_per_parameter_row(self) -> None:
        from flight_controller import load_controller_config
        from simenv.config import load_and_materialize

        simulator = ROOT / "SimEnv/configs/sim2real_micro_coaxial.yaml"
        controller = ROOT / "Controller/configs/lqi_oracle_distillation_manual.yaml"
        materialized = load_and_materialize(simulator, 2, torch.device("cpu"), torch.float64)
        gains = _oracle_gains(
            load_controller_config(controller), materialized.parameters, 1.0 / 500.0
        )
        self.assertEqual(tuple(gains.shape), (2, 5, 13))
        self.assertTrue(torch.isfinite(gains).all())

    def test_external_upper_oracle_designs_four_input_gain(self) -> None:
        from flight_controller import load_controller_config
        from simenv.config import load_and_materialize

        simulator = ROOT / "SimEnv/configs/sim2real_micro_coaxial.yaml"
        controller = ROOT / "Controller/configs/lqi_sim2real_micro_coaxial_4out.yaml"
        materialized = load_and_materialize(
            simulator, 2, torch.device("cpu"), torch.float64
        )
        gains = _oracle_gains(
            load_controller_config(controller), materialized.parameters, 1.0 / 500.0
        )
        self.assertEqual(tuple(gains.shape), (2, 4, 13))
        self.assertTrue(torch.isfinite(gains).all())


if __name__ == "__main__":
    unittest.main()
