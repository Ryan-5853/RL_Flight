from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
for source in ("Identification/src", "Controller/src", "SimEnv/src", "Train/src"):
    path = str(ROOT / source)
    if path not in sys.path:
        sys.path.insert(0, path)

from flight_identification.external_pilot import (  # noqa: E402
    load_external_pilot_profiles,
    make_external_pilot,
)
from flight_train.config import _virtual_pilot  # noqa: E402


class ExternalPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = (
            ROOT
            / "Identification/configs/lqi_gru_dagger_4out_external_pilot_v1.yaml"
        )
        cls.raw = yaml.safe_load(cls.path.read_text(encoding="utf-8"))
        cls.pilot_config = _virtual_pilot(cls.raw["command_source"])

    def test_profiles_include_feedback_and_manual_capabilities(self) -> None:
        profiles = load_external_pilot_profiles(self.raw)
        self.assertEqual(
            tuple(profile.name for profile in profiles),
            ("skilled", "average", "poor", "manual"),
        )
        self.assertEqual(profiles[-1].mode, "manual_sample_hold")
        self.assertGreater(profiles[2].reaction_delay_s[1], 0.0)

    def test_profile_filter_and_upper_throttle_ownership(self) -> None:
        batch = 8
        pilot = make_external_pilot(
            self.pilot_config,
            self.raw,
            batch,
            torch.device("cpu"),
            torch.float32,
            500,
            profile_names=("manual",),
        )
        pilot.generator.manual_seed(7)
        active = torch.ones(batch, dtype=torch.bool)
        pilot.reset(active)
        for _ in range(20):
            pilot.step(torch.zeros(batch, 1), active)
        self.assertTrue(pilot.manual_mode.all())
        self.assertTrue((pilot.snapshot().upper_throttle >= 0.25).all())
        self.assertTrue((pilot.snapshot().upper_throttle <= 0.90).all())
        self.assertEqual(set(pilot.episode_profile_names()), {"manual"})

    def test_poor_feedback_has_delayed_low_rate_updates(self) -> None:
        batch = 32
        pilot = make_external_pilot(
            self.pilot_config,
            self.raw,
            batch,
            torch.device("cpu"),
            torch.float32,
            500,
            profile_names=("poor",),
        )
        pilot.generator.manual_seed(11)
        active = torch.ones(batch, dtype=torch.bool)
        pilot.reset(active)
        self.assertTrue((pilot.reaction_delay_steps >= 60).all())
        self.assertTrue((pilot.update_period_steps >= 50).all())
        self.assertTrue((pilot.feedback_gain_scale <= 0.65).all())


if __name__ == "__main__":
    unittest.main()
