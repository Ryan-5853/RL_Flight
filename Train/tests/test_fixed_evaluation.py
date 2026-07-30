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

    def test_self_stabilize_suite_scores_yaw_rate_not_yaw_heading(self):
        suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_self_stabilize_v2.yaml"
        )
        self.assertTrue(suite.self_stabilize_tracking)
        self.assertEqual(suite.schema_version, 2)
        self.assertEqual(
            suite.checkpoint_selection.minimum_hover_survival_s, 2.5
        )

        steps, batch = 100, 2
        zeros_3 = torch.zeros(steps, batch, 3)
        target_q = euler_to_quaternion(
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
        )
        yaw_offset = torch.full((steps, batch), torch.pi / 2.0)
        actual_euler = zeros_3.clone()
        actual_euler[..., 2] = yaw_offset
        actual_q = euler_to_quaternion(
            actual_euler[..., 0],
            actual_euler[..., 1],
            actual_euler[..., 2],
        )
        trajectory = {
            "alive": torch.ones(steps, batch, dtype=torch.bool),
            "action": torch.zeros(steps, batch, 4),
            "position_n": zeros_3.clone(),
            "velocity_n": zeros_3.clone(),
            "attitude_q_wb": actual_q,
            "target_position_n": zeros_3.clone(),
            "target_velocity_n": zeros_3.clone(),
            "target_attitude_q_wb": target_q,
            "target_euler_rad": zeros_3.clone(),
            "actual_euler_rad": actual_euler,
            "yaw_rate_error_rad_s": torch.zeros(steps, batch),
        }
        result = _score_trajectory(
            trajectory,
            torch.full((batch,), 10.0),
            suite.scenarios[0],
            suite.limits,
            100,
            suite.score_weights,
            self_stabilize_tracking=True,
        )
        self.assertGreater(result["metrics"]["attitude_rmse_deg"]["mean"], 89.0)
        self.assertAlmostEqual(
            result["metrics"]["roll_pitch_rmse_deg"]["mean"], 0.0
        )
        self.assertAlmostEqual(
            result["metrics"]["yaw_rate_rmse_rad_s"]["mean"], 0.0
        )
        self.assertAlmostEqual(result["total_score"], 100.0)

    def test_upright_survival_suite_records_but_does_not_score_yaw_rate(self):
        suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_upright_survival_v3.yaml"
        )
        self.assertEqual(suite.schema_version, 3)
        self.assertTrue(suite.self_stabilize_tracking)
        self.assertEqual(suite.yaw_rate_tracking_weight, 0.0)
        self.assertIsNone(
            suite.checkpoint_selection.maximum_hover_yaw_rate_rmse_rad_s
        )

        steps, batch = 100, 2
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
            "yaw_rate_error_rad_s": torch.full((steps, batch), 10.0),
        }
        result = _score_trajectory(
            trajectory,
            torch.full((batch,), 10.0),
            suite.scenarios[0],
            suite.limits,
            100,
            suite.score_weights,
            self_stabilize_tracking=suite.self_stabilize_tracking,
            yaw_rate_tracking_weight=suite.yaw_rate_tracking_weight,
        )
        self.assertAlmostEqual(
            result["metrics"]["yaw_rate_rmse_rad_s"]["mean"], 10.0
        )
        self.assertAlmostEqual(result["total_score"], 100.0)

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
