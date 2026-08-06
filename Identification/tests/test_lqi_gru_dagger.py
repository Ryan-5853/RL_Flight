from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
for source in ("Identification/src", "Controller/src", "SimEnv/src", "Train/src"):
    path = str(ROOT / source)
    if path not in sys.path:
        sys.path.insert(0, path)

from flight_identification.lqi_gru_dagger import (  # noqa: E402
    _save_dagger_shard,
    prepare_dagger_dataset,
)
from flight_identification.lqi_gru_distillation import (  # noqa: E402
    CausalLQIStudent,
    StudentStepState,
    step_student,
)
from flight_identification.lqi_gru_experiment import (  # noqa: E402
    ACTION_NAMES,
    OBSERVATION_NAMES,
)


class DaggerArchitectureTests(unittest.TestCase):
    def test_all_architectures_emit_bounded_five_axis_commands(self) -> None:
        for arch, context in (
            ("gru", None),
            ("lstm", None),
            ("tcn", 128),
            ("transformer", 128),
        ):
            model = CausalLQIStudent(
                25, 64, 2, (64, 32), 0.0, arch=arch, context_steps=context
            )
            length = context or 32
            observations = torch.randn(3, length, 25)
            commands, _, _ = model(observations)
            self.assertEqual(commands.shape, (3, length, 5))
            self.assertTrue((commands[..., :2] >= 0.0).all())
            self.assertTrue((commands[..., :2] <= 1.0).all())
            self.assertTrue((commands[..., 2:].abs() <= 1.0).all())

    def test_step_student_recurrent_and_windowed(self) -> None:
        for arch, context in (("gru", None), ("transformer", 8)):
            model = CausalLQIStudent(
                25, 64, 2, (64, 32), 0.0, arch=arch, context_steps=context
            )
            checkpoint = {"arch": arch, "context_steps": context}
            state = StudentStepState()
            observation = torch.randn(4, 25)
            for _ in range(3):
                command, state = step_student(
                    model, checkpoint, observation, state, full_context=True
                )
                self.assertEqual(command.shape, (4, 5))

    def test_prepare_dagger_dataset_restricts_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            output = Path(directory) / "merged"
            for split in ("train", "validation", "test"):
                (base / split).mkdir(parents=True)
                torch.save(
                    {
                        "schema_version": 1,
                        "observations": torch.randn(2, 3, len(OBSERVATION_NAMES)),
                        "teacher_actions": torch.randn(2, 3, 5),
                        "valid_mask": torch.ones(2, 3, dtype=torch.bool),
                        "parameter_labels_audit_only": torch.randn(2, 4),
                        "oracle_lqi_gain_audit_only": torch.randn(2, 5, 13),
                        "group_id": torch.arange(2),
                        "observation_names": OBSERVATION_NAMES,
                        "action_names": ACTION_NAMES,
                        "label_names": ("a", "b", "c", "d"),
                        "leakage_contract": "audit tensors are never student inputs",
                    },
                    base / split / "shard_000000.pt",
                )
            torch.save(
                {
                    "labels_audit_only": torch.randn(6, 4),
                    "split_assignment": torch.tensor([0, 1, 2, 0, 1, 2]),
                    "label_names": ("a", "b", "c", "d"),
                },
                base / "parameter_groups_audit_only.pt",
            )
            (base / "manifest.json").write_text(
                json.dumps({"episode_steps": 3, "totals": {"groups": 6}}),
                encoding="utf-8",
            )
            names = tuple(
                name
                for name in OBSERVATION_NAMES
                if not name.startswith("previous_command.")
            )
            manifest = prepare_dagger_dataset(base, output, names)
            self.assertEqual(tuple(manifest["observation_names"]), names)
            shard = torch.load(
                output / "train" / "shard_000000.pt", weights_only=False
            )
            self.assertEqual(shard["observations"].shape[-1], len(names))
            self.assertEqual(tuple(shard["observation_names"]), names)

    def test_dagger_shard_keeps_partial_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = torch.randn(3, 4, len(OBSERVATION_NAMES))
            actions = torch.randn(3, 4, 5)
            valid = torch.ones(3, 4, dtype=torch.bool)
            valid[1, 2:] = False
            counts = _save_dagger_shard(
                root,
                7,
                2,
                torch.tensor([0, 1, 0]),
                torch.tensor([0, 0, 1]),
                observations,
                actions,
                valid,
                torch.randn(3, 4),
                torch.randn(3, 5, 13),
                ("a", "b", "c", "d"),
                OBSERVATION_NAMES,
            )
            self.assertEqual(counts["train"], 2)
            self.assertEqual(counts["validation"], 1)
            saved = torch.load(root / "train" / "shard_000007.pt", weights_only=False)
            self.assertEqual(saved["dagger_iteration"], 2)
            self.assertEqual(saved["valid_mask"][1, 2:].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
