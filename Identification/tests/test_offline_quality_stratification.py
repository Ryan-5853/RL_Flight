from __future__ import annotations

import unittest

import torch

from flight_identification.offline_quality_stratification import (
    _group_quality_statistics,
    _quality_stratified_summary,
)


FEATURE_NAMES = (
    "angular_velocity_b.x",
    "angular_velocity_b.y",
    "angular_velocity_b.z",
    "command.motor_upper",
    "command.motor_lower",
    "command.servo_1",
    "command.servo_2",
    "command.servo_3",
)


class OfflineQualityStratificationTests(unittest.TestCase):
    def test_group_quality_statistics_selects_informative_flights(self) -> None:
        groups, trials, steps = 3, 4, 8
        features = torch.zeros(groups, trials, steps, len(FEATURE_NAMES))
        valid = torch.ones(groups, trials, steps, dtype=torch.bool)
        eligible = torch.ones(groups, trials, dtype=torch.bool)
        features[:, 0, :, 0] = 0.6  # high angular-rate trial per group
        features[:, 0, 1:, 3:] = 0.2  # command movement on the same trial
        statistics = _group_quality_statistics(
            features,
            valid,
            eligible,
            FEATURE_NAMES,
            trial_count=2,
            downsampled_features=features,
            downsampled_valid=valid,
            normalization={
                "feature_mean": torch.zeros(len(FEATURE_NAMES)),
                "feature_std": torch.ones(len(FEATURE_NAMES)),
            },
        )
        self.assertEqual(
            statistics["selected_flight_count"].tolist(), [2, 2, 2]
        )
        self.assertTrue(
            torch.all(statistics["log_information_score_mean"] > 0.2)
        )
        self.assertTrue(
            torch.all(statistics["input_z_score_abs_p95"] > 0.1)
        )

    def test_group_quality_statistics_skips_ineligible_flights(self) -> None:
        groups, trials, steps = 2, 4, 8
        features = torch.zeros(groups, trials, steps, len(FEATURE_NAMES))
        valid = torch.ones(groups, trials, steps, dtype=torch.bool)
        eligible = torch.zeros(groups, trials, dtype=torch.bool)
        eligible[:, 0] = True
        features[:, 0, :, 0] = 0.5
        statistics = _group_quality_statistics(
            features,
            valid,
            eligible,
            FEATURE_NAMES,
            trial_count=2,
            downsampled_features=features,
            downsampled_valid=valid,
            normalization={
                "feature_mean": torch.zeros(len(FEATURE_NAMES)),
                "feature_std": torch.ones(len(FEATURE_NAMES)),
            },
        )
        self.assertEqual(
            statistics["selected_flight_count"].tolist(), [1, 1]
        )

    def test_quality_stratified_summary_reports_correlations(self) -> None:
        deltas = [1.0, -1.0, 1.0, -1.0]
        statistics = {
            "log_information_score_mean": torch.tensor(
                [2.0, 1.0, 2.0, 1.0], dtype=torch.float32
            ),
            "input_z_score_abs_p95": torch.tensor(
                [0.1, 2.0, 0.1, 2.0], dtype=torch.float32
            ),
        }
        summary = _quality_stratified_summary(deltas, statistics)
        self.assertEqual(summary["improved_group_fraction"], 0.5)
        self.assertEqual(summary["degraded_group_fraction"], 0.5)
        by_name = {
            item["name"]: item for item in summary["quality_associations"]
        }
        self.assertGreater(
            by_name["log_information_score_mean"][
                "pearson_correlation_with_group_convergence_delta"
            ],
            0.99,
        )
        self.assertLess(
            by_name["input_z_score_abs_p95"][
                "pearson_correlation_with_group_convergence_delta"
            ],
            -0.99,
        )
        self.assertEqual(
            by_name["log_information_score_mean"]["lowest_quartile_mean_delta"],
            -1.0,
        )
        self.assertEqual(
            by_name["log_information_score_mean"]["highest_quartile_mean_delta"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
