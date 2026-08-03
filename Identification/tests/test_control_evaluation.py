from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import expm

from flight_identification.control_evaluation import (
    _discrete_model,
    _lqr_gain,
    _nominal_actuator_model,
    _scaled_actuator_model,
)


ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG = ROOT / "SimEnv" / "configs" / "example.yaml"


class ControlValueEvaluationTests(unittest.TestCase):
    def test_structured_discretization_matches_matrix_exponential(self) -> None:
        generator = np.random.default_rng(20260803)
        effectiveness = generator.normal(size=(3, 5))
        slopes = generator.uniform(0.2, 3.0, size=5)
        tau = generator.uniform(0.005, 0.5, size=5)
        dt = 1.0 / 500.0
        actual_a, actual_b = _discrete_model(
            effectiveness, slopes, tau, dt
        )
        continuous_a = np.zeros((10, 10), dtype=np.float64)
        continuous_b = np.zeros((10, 5), dtype=np.float64)
        continuous_a[0, 2] = 1.0
        continuous_a[1, 3] = 1.0
        continuous_a[2:5, 5:10] = effectiveness
        continuous_a[5:10, 5:10] = -np.diag(1.0 / tau)
        continuous_b[5:10] = np.diag(slopes / tau)
        augmented = np.block(
            [[continuous_a, continuous_b], [np.zeros((5, 15))]]
        )
        expected = expm(augmented * dt)
        np.testing.assert_allclose(actual_a, expected[:10, :10], atol=1e-14)
        np.testing.assert_allclose(actual_b, expected[:10, 10:], atol=1e-14)

    def test_nominal_reconstruction_matches_controller_pole_radius(self) -> None:
        effectiveness, slopes, tau = _nominal_actuator_model(SIM_CONFIG)
        scaled_effectiveness, scaled_tau = _scaled_actuator_model(
            np.zeros(14), effectiveness, tau
        )
        np.testing.assert_allclose(scaled_effectiveness, effectiveness)
        np.testing.assert_allclose(scaled_tau, tau)
        a, b = _discrete_model(effectiveness, slopes, tau, 1.0 / 500.0)
        state_scales = np.asarray(
            [0.1745329, 0.1745329, 2.0, 2.0, 1.5, 250.0, 250.0, 0.15, 0.15, 0.15]
        )
        input_scales = np.asarray([0.18, 0.18, 0.45, 0.45, 0.45])
        q = np.diag(1.0 / state_scales**2)
        r = 0.25 * np.diag(1.0 / input_scales**2)
        gain = _lqr_gain(a, b, q, r)
        radius = np.max(np.abs(np.linalg.eigvals(a - b @ gain)))
        self.assertAlmostEqual(
            float(radius), 0.9927257495398903, delta=1e-9
        )


if __name__ == "__main__":
    unittest.main()
