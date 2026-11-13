from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest

import yaml



ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = (
    ROOT
    / "px4_trans/configs/lqi_coaxial_collective_airframe1_yaw_balanced_v2.yaml"
)
MANIFEST = (
    ROOT
    / "px4_trans/generated/airframe-1/yaw-balanced-v2/"
    / "lqi_coaxial_collective_manifest.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoaxialCollectiveFirmwarePayloadTest(unittest.TestCase):
    def test_shared_motor_table_uses_csv_in_controller_domain(self) -> None:
        deployment = yaml.safe_load(DEPLOYMENT.read_text(encoding="utf-8"))
        simulator_path = ROOT / deployment["simulator_config"]
        # Verify the source calibration directly; the firmware reuse path
        # deliberately does not require SimEnv source or Torch.
        simulator = yaml.safe_load(simulator_path.read_text(encoding="utf-8"))
        tables = [motor["pwm_to_rpm_table"]["value"] for motor in simulator["motors"]]
        table = tables[0]
        self.assertEqual(tables[0], tables[1])
        with (
            ROOT / deployment["motor_speed_calibration"]["path"]
        ).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        for index, row in enumerate(rows):
            command = 2.0 * float(row["command_normalized"])
            speed = float(row["rpm_mean"]) * 2.0 * math.pi / 60.0
            self.assertEqual(table[index + 2][0], command)
            self.assertTrue(
                math.isclose(
                    table[index + 2][1], speed, rel_tol=2e-7
                )
            )
        self.assertEqual(table[-1][0], 1.0)
        self.assertEqual(table[-1][1], 1200.0)

    def test_payload_manifest_matches_generated_sources(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["airframe_id"], "airframe-1")
        self.assertEqual(manifest["model_checksum"], '0x30298daf')
        source = manifest["source"]
        for path_name, digest_name in (
            ("simulator_config", "simulator_sha256"),
            ("controller_config", "controller_sha256"),
            ("calibration_path", "calibration_sha256"),
            ("identification_report", "identification_report_sha256"),
        ):
            self.assertEqual(
                sha256(ROOT / source[path_name]), source[digest_name]
            )
        generated = manifest["generated"]
        for path_name, digest_name in (
            ("candidate_payload", "candidate_payload_sha256"),
            ("active_header", "active_header_sha256"),
            ("golden_header", "golden_header_sha256"),
            ("golden_json", "golden_json_sha256"),
        ):
            self.assertEqual(
                sha256(ROOT / generated[path_name]),
                generated[digest_name],
            )
        self.assertEqual(
            generated["active_header_sha256"],
            generated["candidate_payload_sha256"],
        )
        if "firmware_artifact" in manifest:
            firmware = manifest["firmware_artifact"]
            self.assertEqual(
                sha256(ROOT / firmware["path"]), firmware["sha256"]
            )
        payload = (
            ROOT
            / "px4_trans/generated/airframe-1/yaw-balanced-v2/"
            / "LqiCoaxialCollectivePayload.hpp"
        ).read_text(encoding="utf-8")
        self.assertIn("kYawBalance[7]", payload)
        self.assertEqual(manifest["controller_contract"], "pilot_acceleration_shared_yaw_antiwindup")
        self.assertEqual(source["allocation_safety"]["roll_pitch_integral_moment_fraction"], .5)
        self.assertEqual(source["allocation_safety"]["grid_common_preference_fraction"], .5)
        self.assertAlmostEqual(source["yaw_balance_profile"]["grid_common_command"], -0.16, places=6)
        self.assertAlmostEqual(source["yaw_balance_profile"]["maximum_speed_ratio"], 1.15, places=6)


if __name__ == "__main__":
    unittest.main()
