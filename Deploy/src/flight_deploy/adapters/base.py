from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from ..model import ConvertedPolicy


class CheckpointAdapter(ABC):
    """Extension point for a training framework or checkpoint family."""

    name: str
    priority: int = 0

    @abstractmethod
    def probe(self, checkpoint: Mapping[str, Any]) -> bool:
        """Return true only when this adapter can identify the checkpoint."""

    @abstractmethod
    def describe(self, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return cheap, JSON-compatible checkpoint information."""

    @abstractmethod
    def convert(
        self,
        checkpoint: Mapping[str, Any],
        *,
        source_path: Path,
    ) -> ConvertedPolicy:
        """Convert a recognized checkpoint to the canonical policy interface."""
