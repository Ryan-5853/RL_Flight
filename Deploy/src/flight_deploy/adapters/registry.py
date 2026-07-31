from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..errors import UnsupportedCheckpointError
from .base import CheckpointAdapter


class AdapterRegistry:
    def __init__(self, adapters: Iterable[CheckpointAdapter] = ()) -> None:
        self._adapters: dict[str, CheckpointAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: CheckpointAdapter) -> None:
        if not adapter.name or adapter.name in self._adapters:
            raise ValueError(f"duplicate or empty adapter name: {adapter.name!r}")
        self._adapters[adapter.name] = adapter

    def names(self) -> tuple[str, ...]:
        return tuple(
            adapter.name
            for adapter in sorted(
                self._adapters.values(),
                key=lambda item: (-item.priority, item.name),
            )
        )

    def select(
        self,
        checkpoint: Mapping[str, Any],
        *,
        requested: str | None = None,
    ) -> CheckpointAdapter:
        if requested is not None:
            try:
                adapter = self._adapters[requested]
            except KeyError as exc:
                raise UnsupportedCheckpointError(
                    f"unknown adapter {requested!r}; available: {', '.join(self.names())}"
                ) from exc
            if not adapter.probe(checkpoint):
                raise UnsupportedCheckpointError(
                    f"adapter {requested!r} does not recognize this checkpoint"
                )
            return adapter
        for name in self.names():
            adapter = self._adapters[name]
            if adapter.probe(checkpoint):
                return adapter
        raise UnsupportedCheckpointError(
            "no adapter recognizes this checkpoint; install or register a "
            "CheckpointAdapter for its training framework and architecture"
        )
