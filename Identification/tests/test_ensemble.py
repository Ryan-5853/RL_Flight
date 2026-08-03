from __future__ import annotations

import unittest

import torch

from flight_identification.ensemble import calibrated_std, fit_uncertainty_calibration


class EnsembleTests(unittest.TestCase):
    def test_calibration_produces_positive_finite_uncertainty(self) -> None:
        predictions = torch.stack(
            (torch.zeros(20, 3), torch.full((20, 3), 0.2)), dim=0
        )
        labels = torch.linspace(-0.5, 0.5, 20)[:, None].expand(-1, 3)
        calibration = fit_uncertainty_calibration(predictions, labels)
        std = calibrated_std(predictions, calibration)
        self.assertTrue(torch.isfinite(std).all())
        self.assertTrue((std > 0).all())


if __name__ == "__main__":
    unittest.main()
