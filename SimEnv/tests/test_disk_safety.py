from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from simenv.config import LoggingConfig
from simenv.errors import InsufficientDiskSpaceError
from simenv.logging import TensorChunkLogger


class SimulatorDiskSafetyTests(unittest.TestCase):
    def test_timeline_preflight_keeps_partial_file_unpublished(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config_path = root / "sim.json"
            config_path.write_text(
                json.dumps({"schema_version": 1}),
                encoding="utf-8",
            )
            free_bytes = 1024 * 1024 * 1024

            def disk_usage(_path: Path) -> SimpleNamespace:
                return SimpleNamespace(free=free_bytes)

            with mock.patch(
                "simenv.logging.shutil.disk_usage",
                side_effect=disk_usage,
            ):
                logger = TensorChunkLogger(
                    LoggingConfig(
                        directory=root / "logs",
                        chunk_steps=1,
                        queue_chunks=1,
                        overflow="block",
                        mode="full",
                        physics_step_stride=1,
                        fields=None,
                        minimum_free_space_bytes=1024,
                    ),
                    "batch",
                    ("instance",),
                    config_path,
                    {"schema_version": 1},
                    {"mass": torch.ones(1)},
                )
                free_bytes = 512
                logger.append({"value": torch.ones(1)})
                logger._thread.join(timeout=2)
                self.assertFalse(logger._thread.is_alive())
                with self.assertRaises(InsufficientDiskSpaceError):
                    logger.append({"value": torch.ones(1)})
                with self.assertRaises(InsufficientDiskSpaceError):
                    logger.close()

            self.assertFalse(
                (logger.directory / "timeline_000000.pt").exists()
            )
            self.assertFalse(
                (logger.directory / "timeline_000000.pt.tmp").exists()
            )

    def test_pytorch_iostream_failure_is_converted_and_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config_path = root / "sim.json"
            config_path.write_text(
                json.dumps({"schema_version": 1}),
                encoding="utf-8",
            )
            logger = TensorChunkLogger(
                LoggingConfig(
                    directory=root / "logs",
                    chunk_steps=1,
                    queue_chunks=1,
                    overflow="block",
                    mode="full",
                    physics_step_stride=1,
                    fields=None,
                    minimum_free_space_bytes=0,
                ),
                "batch",
                ("instance",),
                config_path,
                {"schema_version": 1},
                {"mass": torch.ones(1)},
            )
            with mock.patch(
                "simenv.logging.torch.save",
                side_effect=RuntimeError(
                    "basic_ios::clear: iostream error"
                ),
            ):
                logger.append({"value": torch.ones(1)})
                logger._thread.join(timeout=2)
                with self.assertRaises(InsufficientDiskSpaceError):
                    logger.close()

            self.assertFalse(
                (logger.directory / "timeline_000000.pt").exists()
            )
            self.assertFalse(
                (logger.directory / "timeline_000000.pt.tmp").exists()
            )


if __name__ == "__main__":
    unittest.main()
