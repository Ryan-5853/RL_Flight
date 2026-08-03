from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from flight_identification.control_evaluation import (
    _discrete_model,
    _lqr_gain,
    _nominal_actuator_model,
    _scaled_actuator_model,
)


ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG = ROOT / "SimEnv" / "configs" / "example.yaml"


class ControlValueEvaluationTests(unittest.TestCase):
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
