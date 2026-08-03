from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from flight_identification.config import (
    ConvergenceConfig,
    IdentificationExperimentConfig,
    InitialStateConfig,
    WindowConfig,
    load_experiment_config,
)
from flight_identification.experiment import (
    FEATURE_NAMES,
    LABEL_NAMES,
    _group_split_assignments,
    _sample_group_labels,
    _yaw_free_attitude,
    generate_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "Identification" / "configs" / "lqr_zero_attitude_yaw_rate_v1.yaml"
SIM_CONFIG = ROOT / "SimEnv" / "configs" / "example.yaml"
CONTROLLER_CONFIG = ROOT / "Controller" / "configs" / "lqr_identification_nominal.yaml"


class IdentificationExperimentTests(unittest.TestCase):
    def test_default_config_is_one_second_and_has_order_of_magnitude_range(self) -> None:
        config = load_experiment_config(DEFAULT_CONFIG)
        self.assertEqual(config.window_steps, 500)
        self.assertEqual(config.log10_effectiveness_range, (-1.0, 1.0))
        self.assertEqual(config.parallel_count % config.initial_conditions_per_group, 0)

    def test_effectiveness_labels_preserve_rigid_body_inertia_constraints(self) -> None:
        generator = torch.Generator().manual_seed(7)
        nominal = torch.tensor([0.030, 0.028, 0.012], dtype=torch.float64)
        labels = _sample_group_labels(
            2048, nominal, (-1.0, 1.0), generator, torch.float64
        )
        self.assertTrue(((labels >= -1.0) & (labels <= 1.0)).all())
        inertia = nominal * torch.pow(10.0, labels[:, :3])
        self.assertTrue((2.0 * inertia.amax(dim=1) <= inertia.sum(dim=1)).all())

    def test_parameter_group_split_is_disjoint(self) -> None:
        assignments = _group_split_assignments(
            100, (0.7, 0.15, 0.15), torch.Generator().manual_seed(11)
        )
        self.assertEqual(set(assignments.tolist()), {0, 1, 2})
        self.assertEqual(assignments.shape, (100,))

    def test_yaw_is_removed_from_the_observable_attitude(self) -> None:
        half_yaw = 0.5 * math.radians(70.0)
        yaw_only = torch.tensor(
            [[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            _yaw_free_attitude(yaw_only),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )

    def test_smoke_generation_has_no_servo_feedback_and_no_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            config = IdentificationExperimentConfig(
                source_path=DEFAULT_CONFIG,
                simulator_config=SIM_CONFIG,
                controller_config=CONTROLLER_CONFIG,
                output_directory=output,
                seed=19,
                device="cpu",
                dtype="float32",
                measurement_mode="ideal",
                parameter_groups=3,
                initial_conditions_per_group=1,
                parallel_count=3,
                episode_duration_s=0.02,
                log10_effectiveness_range=(-0.01, 0.01),
                initial_state=InitialStateConfig(
                    minimum_tilt_rad=0.0,
                    maximum_tilt_rad=0.0,
                    maximum_angular_rate_rad_s=(0.0, 0.0, 0.0),
                ),
                convergence=ConvergenceConfig(
                    maximum_roll_pitch_error_rad=math.pi,
                    maximum_angular_rate_rad_s=100.0,
                    hold_s=0.002,
                    safety_tilt_rad=math.pi,
                    safety_angular_rate_rad_s=1000.0,
                ),
                window=WindowConfig(
                    length_s=0.01,
                    stride_s=0.01,
                    split_fractions=(1 / 3, 1 / 3, 1 / 3),
                    maximum_start_s=None,
                ),
            )
            manifest = generate_dataset(config)
            self.assertEqual(manifest["controller_semantics"]["yaw_target"], "angular_rate_zero")
            self.assertFalse(manifest["controller_semantics"]["yaw_angle_feedback"])
            self.assertFalse(manifest["controller_semantics"]["servo_angle_feedback"])
            self.assertEqual(
                manifest["controller_semantics"]["measurement_mode"], "ideal"
            )
            self.assertNotIn("servo_angle", " ".join(FEATURE_NAMES))
            self.assertEqual(len(LABEL_NAMES), 13)
            self.assertEqual(manifest["totals"]["converged_episodes"], 3)

            groups_by_split: dict[str, set[int]] = {}
            for split in ("train", "validation", "test"):
                shard = torch.load(
                    next((output / split).glob("shard_*.pt")), weights_only=False
                )
                self.assertEqual(shard["features"].shape[1:], (5, len(FEATURE_NAMES)))
                groups_by_split[split] = set(shard["group_id"].tolist())
                adaptation = torch.load(
                    next((output / "adaptation" / split).glob("shard_*.pt")),
                    weights_only=False,
                )
                self.assertIn("final_success", adaptation)
                self.assertTrue(adaptation["final_success"].all())
            self.assertTrue(groups_by_split["train"].isdisjoint(groups_by_split["validation"]))
            self.assertTrue(groups_by_split["train"].isdisjoint(groups_by_split["test"]))
            self.assertTrue(groups_by_split["validation"].isdisjoint(groups_by_split["test"]))
            stored_manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(stored_manifest["totals"], manifest["totals"])


if __name__ == "__main__":
    unittest.main()
