from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import ExperimentConfig


class RunRecorder:
    """CPU/file I/O boundary. It receives only detached scalar summaries."""

    def __init__(self, config: ExperimentConfig) -> None:
        resolved = json.dumps(config.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        fingerprint = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = str(uuid.uuid4())
        self.directory = config.run.output_root / config.name / f"{stamp}_{fingerprint}_{self.run_id[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "checkpoints").mkdir()
        (self.directory / "config.json").write_text(
            json.dumps(config.raw, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest = {
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": fingerprint,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchrl": _torchrl_version(),
            "device": config.run.device,
            "dtype": config.run.dtype,
            "simulator_config": str(config.simulator_config),
            "simulator_config_sha256": _sha256(config.simulator_config),
            "exact_resume_supported": False,
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._metrics = (self.directory / "metrics.jsonl").open("a", encoding="utf-8")

    def metrics(self, control_steps: int, values: Mapping[str, torch.Tensor]) -> None:
        # One synchronization per update, outside collection/backpropagation.
        record: dict[str, Any] = {
            "global_control_steps": control_steps,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in values.items():
            if value.numel() == 1:
                record[key] = float(value.detach().item())
        self._metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._metrics.flush()

    def checkpoint(self, control_steps: int, state: Mapping[str, Any]) -> Path:
        destination = self.directory / "checkpoints" / f"step_{control_steps}.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save(dict(state), temporary)
        temporary.replace(destination)
        return destination

    def close(self, status: str, control_steps: int) -> None:
        if not self._metrics.closed:
            self._metrics.close()
        (self.directory / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "global_control_steps": control_steps,
                    "ended_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torchrl_version() -> str:
    import torchrl

    return torchrl.__version__

