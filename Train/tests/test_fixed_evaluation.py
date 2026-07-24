from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from flight_train.evaluation import (
    FixedScenario,
    ScoreLimits,
    _scenario_trajectory,
    _score_trajectory,
    load_fixed_evaluation_suite,
)
from flight_train.math import euler_to_quaternion


ROOT = Path(__file__).parents[1]


class FixedEvaluationTests(unittest.TestCase):
    def test_suite_parses_three_required_subjects(self):
        suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_attitude_v1.yaml"
        )
        self.assertEqual(
            [item.type for item in suite.scenarios],
            ["hover", "constant_translation", "circle"],
        )
        self.assertAlmostEqual(sum(suite.score_weights.values()), 1.0)

    def test_circle_reference_starts_at_origin_with_tangent_velocity(self):
        scenario = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_attitude_v1.yaml"
        ).scenarios[2]
        time = torch.zeros(2)
        origin = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        position, velocity = _scenario_trajectory(scenario, time, origin)
        torch.testing.assert_close(position, origin)
        expected_speed = 2.0 * torch.pi * scenario.circle_radius_m / scenario.circle_period_s
        torch.testing.assert_close(velocity[:, 0], torch.full((2,), expected_speed))
        torch.testing.assert_close(velocity[:, 1:], torch.zeros(2, 2))

    def test_perfect_hover_scores_one_hundred(self):
        steps, batch = 20, 2
        zeros_3 = torch.zeros(steps, batch, 3)
        target_q = euler_to_quaternion(
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
        )
        trajectory = {
            "alive": torch.ones(steps, batch, dtype=torch.bool),
            "action": torch.zeros(steps, batch, 4),
            "position_n": zeros_3.clone(),
            "velocity_n": zeros_3.clone(),
            "attitude_q_wb": target_q.clone(),
            "target_position_n": zeros_3.clone(),
            "target_velocity_n": zeros_3.clone(),
            "target_attitude_q_wb": target_q,
            "target_euler_rad": zeros_3.clone(),
            "actual_euler_rad": zeros_3.clone(),
        }
        limits = ScoreLimits(15.0, 5.0, 3.0, 1.0, 0.25, 0.5, 2.0, 2.0, 0.02, 1.0)
        scenario = FixedScenario(
            "hover", "hover", 0.2, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            1.0, 8.0, 0.0, 0.0,
        )
        result = _score_trajectory(
            trajectory,
            torch.full((batch,), 0.2),
            scenario,
            limits,
            100,
            {"survival": 0.35, "tracking": 0.35, "action": 0.15, "response": 0.15},
        )
        self.assertAlmostEqual(result["total_score"], 100.0)

    def test_unknown_suite_field_is_rejected(self):
        source = (ROOT / "configs/evaluation/fixed_attitude_v1.yaml").read_text()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bad.yaml"
            path.write_text(source + "\nunknown_field: true\n")
            with self.assertRaisesRegex(ValueError, "unknown evaluation suite"):
                load_fixed_evaluation_suite(path)


if __name__ == "__main__":
    unittest.main()
