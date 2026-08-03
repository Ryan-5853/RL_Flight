from __future__ import annotations

import unittest

import torch

from flight_identification.training import (
    HistoryMLP,
    TemporalConvIdentifier,
    _aggregate_by_group,
    _group_balanced_weights,
    effective_lqr_labels,
    lqr_latent_labels,
    lqr_latent_to_effective,
    regression_metrics,
)


class IdentificationTrainingTests(unittest.TestCase):
    def test_history_mlp_shape(self) -> None:
        model = HistoryMLP(100, 14, 13, (32, 16))
        self.assertEqual(model(torch.zeros(7, 100, 14)).shape, (7, 13))

    def test_temporal_convolution_shape(self) -> None:
        model = TemporalConvIdentifier(14, 11, 32, (24,), dropout=0.0)
        self.assertEqual(model(torch.zeros(7, 100, 14)).shape, (7, 11))

    def test_group_weights_give_each_group_equal_mass(self) -> None:
        groups = torch.tensor([1, 1, 1, 2, 3, 3])
        weights = _group_balanced_weights(groups)
        self.assertAlmostEqual(float(weights[groups == 1].sum()), 1.0)
        self.assertAlmostEqual(float(weights[groups == 2].sum()), 1.0)
        self.assertAlmostEqual(float(weights[groups == 3].sum()), 1.0)

    def test_group_aggregation_averages_windows(self) -> None:
        predictions = torch.tensor([[1.0], [3.0], [8.0]])
        labels = torch.tensor([[2.0], [2.0], [9.0]])
        prediction, label = _aggregate_by_group(
            predictions, labels, torch.tensor([5, 5, 7])
        )
        torch.testing.assert_close(prediction, torch.tensor([[2.0], [8.0]]))
        torch.testing.assert_close(label, torch.tensor([[2.0], [9.0]]))

    def test_metrics_are_exact_for_exact_prediction(self) -> None:
        labels = torch.tensor([[-0.5, 0.2], [0.5, -0.2]])
        metrics = regression_metrics(labels, labels, ("a", "b"))
        self.assertAlmostEqual(metrics["mean_r2"], 1.0)
        self.assertAlmostEqual(metrics["mean_rmse_log10"], 0.0)
        for parameter in metrics["parameters"]:
            self.assertAlmostEqual(parameter["median_factor_error"], 1.0)

    def test_effective_labels_remove_two_unobservable_physical_directions(self) -> None:
        labels = torch.randn(4, 13)
        transformed = effective_lqr_labels(labels)
        common_moment_and_inertia = labels.clone()
        common_moment_and_inertia[:, 0:4] += 0.7
        common_moment_and_inertia[:, 4] += 0.7
        grid_collective_trade = labels.clone()
        grid_collective_trade[:, 3] += 0.4
        grid_collective_trade[:, 7:10] -= 0.4
        torch.testing.assert_close(
            effective_lqr_labels(common_moment_and_inertia), transformed
        )
        torch.testing.assert_close(
            effective_lqr_labels(grid_collective_trade), transformed
        )

    def test_lqr_latent_exactly_reconstructs_effective_matrix_targets(self) -> None:
        labels = torch.randn(32, 13)
        torch.testing.assert_close(
            lqr_latent_to_effective(lqr_latent_labels(labels)),
            effective_lqr_labels(labels),
        )


if __name__ == "__main__":
    unittest.main()
