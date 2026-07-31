from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from flight_deploy.bundle import BundleBuilder, DeploymentBundle
from flight_deploy.errors import BundleIntegrityError
from flight_deploy.runtime import PolicyRuntime

from test_adapters import flight_checkpoint


class BundleRuntimeTests(unittest.TestCase):
    def test_checkpoint_to_bundle_to_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "checkpoint.pt"
            bundle_path = root / "bundle"
            torch.save(flight_checkpoint((21, 12, 8)), checkpoint_path)
            bundle = BundleBuilder().build(
                checkpoint_path,
                bundle_path,
                verify_checksum=False,
            )
            self.assertEqual(bundle.manifest["interface"]["input"]["shape"], ["batch", 21])
            self.assertLess(
                bundle.manifest["artifacts"]["model.ts"][
                    "conversion_max_abs_error"
                ],
                1e-6,
            )
            self.assertIn("golden_vectors.pt", bundle.manifest["artifacts"])
            runtime = PolicyRuntime.load(bundle_path)
            output = runtime.infer(torch.randn(3, 21))
            self.assertEqual(output.shape, (3, 4))
            self.assertTrue(bool(torch.isfinite(output).all()))
            export_runtime = PolicyRuntime.load(
                bundle_path, backend="torch_export"
            )
            export_output = export_runtime.infer(torch.randn(3, 21))
            self.assertEqual(export_output.shape, (3, 4))

    def test_integrity_check_detects_modified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "checkpoint.pt"
            bundle_path = root / "bundle"
            torch.save(flight_checkpoint((21, 8, 8)), checkpoint_path)
            BundleBuilder().build(
                checkpoint_path,
                bundle_path,
                verify_checksum=False,
            )
            with (bundle_path / "model.ts").open("ab") as stream:
                stream.write(b"damage")
            with self.assertRaises(BundleIntegrityError):
                DeploymentBundle.load(bundle_path)

    def test_unknown_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(
                json.dumps({"schema_version": 99}),
                encoding="utf-8",
            )
            with self.assertRaises(BundleIntegrityError):
                DeploymentBundle.load(root)
