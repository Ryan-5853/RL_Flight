from __future__ import annotations

import json
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import LoggingConfig
from .errors import LoggingError


class TensorChunkLogger:
    """Lossless [time, B, ...] tensor logger with a background disk writer."""

    def __init__(
        self,
        config: LoggingConfig,
        batch_id: str,
        instance_ids: tuple[str, ...],
        config_path: Path,
        raw_config: Mapping[str, Any],
        parameters: Mapping[str, torch.Tensor],
    ) -> None:
        self.directory = config.directory / batch_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._chunk_steps = config.chunk_steps
        self._overflow = config.overflow
        self._queue: queue.Queue[object] = queue.Queue(maxsize=config.queue_chunks)
        self._sentinel = object()
        self._buffers: dict[str, torch.Tensor] = {}
        self._cursor = 0
        self._chunk_index = 0
        self._reset_index = 0
        self._closed = False
        self._writer_error: BaseException | None = None

        shutil.copyfile(config_path, self.directory / f"config{config_path.suffix or '.yaml'}")
        (self.directory / "metadata.json").write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "instance_ids": instance_ids,
                    "schema_version": raw_config.get("schema_version"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        torch.save(
            {name: value.detach().cpu() for name, value in parameters.items()},
            self.directory / "parameters.pt",
        )

        self._thread = threading.Thread(target=self._writer_loop, name=f"simlog-{batch_id}", daemon=True)
        self._thread.start()

    def record_reset(
        self,
        reset_mask: torch.Tensor,
        generation: torch.Tensor,
        previous_instance_ids: tuple[str, ...],
        instance_ids: tuple[str, ...],
        config_path: Path,
        parameters: Mapping[str, torch.Tensor],
    ) -> None:
        """Persist one masked reset operation and its candidate parameter batch."""
        self._check_open()
        self._check_writer()
        reset_directory = self.directory / "resets" / f"{self._reset_index:06d}"
        reset_directory.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(
            config_path,
            reset_directory / f"config{config_path.suffix or '.yaml'}",
        )
        torch.save(
            {
                "reset_mask": reset_mask.detach().cpu(),
                "generation": generation.detach().cpu(),
                "parameters": {
                    name: value.detach().cpu() for name, value in parameters.items()
                },
            },
            reset_directory / "parameters.pt",
        )
        indices = torch.nonzero(reset_mask, as_tuple=False).flatten().cpu().tolist()
        event = {
            "reset_index": self._reset_index,
            "instance_indices": indices,
            "instances": [
                {
                    "instance_index": index,
                    "generation": int(generation[index].item()),
                    "previous_instance_id": previous_instance_ids[index],
                    "instance_id": instance_ids[index],
                }
                for index in indices
            ],
            "config": str(config_path),
        }
        with (self.directory / "resets.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._reset_index += 1

    def append(self, record: Mapping[str, torch.Tensor]) -> None:
        self._check_open()
        self._check_writer()
        if not self._buffers:
            self._allocate(record)
        if set(record) != set(self._buffers):
            raise LoggingError("timeline record fields changed after logger initialization")

        for name, value in record.items():
            target = self._buffers[name][self._cursor]
            if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
                raise LoggingError(f"timeline field {name!r} changed shape, dtype, or device")
            target.copy_(value.detach())
        self._cursor += 1
        if self._cursor == self._chunk_steps:
            self.flush()

    def flush(self) -> None:
        self._check_open()
        self._check_writer()
        if self._cursor == 0:
            return
        payload, event = self._snapshot(self._cursor)
        item = (self._chunk_index, payload, event)
        self._enqueue(item)
        self._chunk_index += 1
        self._cursor = 0

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
            self._enqueue(self._sentinel, force_block=True)
            self._thread.join()
            self._check_writer()
        finally:
            self._closed = True

    def _allocate(self, record: Mapping[str, torch.Tensor]) -> None:
        for name, value in record.items():
            self._buffers[name] = torch.empty(
                (self._chunk_steps, *value.shape), dtype=value.dtype, device=value.device
            )

    def _snapshot(self, count: int) -> tuple[dict[str, torch.Tensor], torch.cuda.Event | None]:
        first = next(iter(self._buffers.values()))
        if first.device.type != "cuda":
            return ({name: value[:count].detach().cpu().clone() for name, value in self._buffers.items()}, None)

        event = torch.cuda.Event()
        payload: dict[str, torch.Tensor] = {}
        for name, value in self._buffers.items():
            source = value[:count].detach()
            host = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
            host.copy_(source, non_blocking=True)
            payload[name] = host
        event.record(torch.cuda.current_stream(first.device))
        return payload, event

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._sentinel:
                        return
                    chunk_index, payload, event = item
                    if event is not None:
                        event.synchronize()
                    torch.save(payload, self.directory / f"timeline_{chunk_index:06d}.pt")
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._writer_error = exc

    def _enqueue(self, item: object, force_block: bool = False) -> None:
        if not force_block and self._overflow == "error":
            try:
                self._queue.put_nowait(item)
            except queue.Full as exc:
                raise LoggingError("timeline writer queue is full") from exc
            return

        while True:
            self._check_writer()
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _check_open(self) -> None:
        if self._closed:
            raise LoggingError("timeline logger is closed")

    def _check_writer(self) -> None:
        if self._writer_error is not None:
            raise LoggingError("timeline writer failed") from self._writer_error
