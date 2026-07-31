from __future__ import annotations

import errno
import json
import os
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .config import LoggingConfig
from .errors import InsufficientDiskSpaceError, LoggingError


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
        simulation_hz: int = 500,
    ) -> None:
        self.directory = config.directory / batch_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._chunk_steps = config.chunk_steps
        self._overflow = config.overflow
        self._mode = config.mode
        self._physics_step_stride = config.physics_step_stride
        self._fields = config.fields
        self._minimum_free_space_bytes = config.minimum_free_space_bytes
        self._physics_append_count = 0
        self._queue: queue.Queue[object] = queue.Queue(maxsize=config.queue_chunks)
        self._sentinel = object()
        self._buffers: dict[str, torch.Tensor] = {}
        self._cursor = 0
        self._chunk_index = 0
        self._reset_index = 0
        self._closed = False
        self._writer_error: BaseException | None = None

        self._ensure_disk_space(0, "initialize simulator logging")
        shutil.copyfile(
            config_path,
            self.directory / f"config{config_path.suffix or '.yaml'}",
        )
        metadata = json.dumps(
            {
                "batch_id": batch_id,
                "instance_ids": instance_ids,
                "schema_version": raw_config.get("schema_version"),
                "logging_mode": config.mode,
                "simulation_hz": simulation_hz,
                "simulation_step_s": 1.0 / simulation_hz,
                "timeline_step_semantics": (
                    "physics_step and control_step are the same "
                    f"{simulation_hz} Hz single-step clock"
                ),
                "physics_step_stride": config.physics_step_stride,
                "timeline_fields": (
                    list(config.fields) if config.fields is not None else "all"
                ),
                "reset_parameters": (
                    "selected_instances" if config.mode == "compact" else "full_batch"
                ),
                "reset_event_storage": (
                    "sparse_snapshot"
                    if config.mode == "compact"
                    else "full_batch_timeline"
                ),
                "minimum_free_space_bytes": config.minimum_free_space_bytes,
            },
            ensure_ascii=False,
            indent=2,
        )
        self._ensure_disk_space(
            len(metadata.encode("utf-8")), "write simulator log metadata"
        )
        (self.directory / "metadata.json").write_text(metadata, encoding="utf-8")
        self._atomic_torch_save(
            {name: value.detach().cpu() for name, value in parameters.items()},
            self.directory / "parameters.pt",
            operation="write simulator parameters",
        )

        self._thread = threading.Thread(target=self._writer_loop, name=f"simlog-{batch_id}", daemon=True)
        self._thread.start()

    @property
    def uses_sparse_reset_events(self) -> bool:
        """Whether masked reset state belongs in its sparse reset snapshot."""
        return self._mode == "compact"

    def record_reset(
        self,
        reset_mask: torch.Tensor,
        generation: torch.Tensor,
        previous_instance_ids: tuple[str, ...],
        instance_ids: tuple[str, ...],
        config_path: Path,
        parameters: Mapping[str, torch.Tensor],
        timeline_record: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Persist one masked reset and, in compact mode, its selected post-reset state."""
        self._check_open()
        self._check_writer()
        self._ensure_disk_space(0, "write simulator reset")
        reset_directory = self.directory / "resets" / f"{self._reset_index:06d}"
        reset_directory.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(
            config_path,
            reset_directory / f"config{config_path.suffix or '.yaml'}",
        )
        indices = torch.nonzero(reset_mask, as_tuple=False).flatten().cpu().tolist()
        generation_cpu = generation.detach().cpu()
        if self._mode == "compact":
            saved_parameters = {
                name: value[reset_mask].detach().cpu()
                for name, value in parameters.items()
            }
            saved_generation = generation[reset_mask].detach().cpu()
            if timeline_record is None:
                raise LoggingError("compact reset logging requires a post-reset timeline record")
            saved_timeline = self._select_fields(timeline_record)
            saved_timeline = {
                name: value[reset_mask].detach().cpu()
                for name, value in saved_timeline.items()
            }
        else:
            saved_parameters = {
                name: value.detach().cpu() for name, value in parameters.items()
            }
            saved_generation = generation_cpu
            saved_timeline = None
        snapshot = {
            "reset_mask": reset_mask.detach().cpu(),
            "parameter_instance_indices": torch.tensor(indices, dtype=torch.int64),
            "generation": saved_generation,
            "parameters": saved_parameters,
        }
        if saved_timeline is not None:
            snapshot["post_reset_timeline"] = saved_timeline
        self._atomic_torch_save(
            snapshot,
            reset_directory / "parameters.pt",
            operation="write simulator reset parameters",
        )
        event = {
            "reset_index": self._reset_index,
            "instance_indices": indices,
            "instances": [
                {
                    "instance_index": index,
                    "generation": int(generation_cpu[index].item()),
                    "previous_instance_id": previous_instance_ids[index],
                    "instance_id": instance_ids[index],
                }
                for index in indices
            ],
            "config": str(config_path),
        }
        event_line = json.dumps(event, ensure_ascii=False) + "\n"
        self._ensure_disk_space(
            len(event_line.encode("utf-8")), "append simulator reset event"
        )
        with (self.directory / "resets.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(event_line)
        self._reset_index += 1

    def append(self, record: Mapping[str, torch.Tensor], *, force: bool = False) -> None:
        self._check_open()
        self._check_writer()
        if not self._begin_append(force):
            return
        self._append_selected(record)

    def append_lazy(
        self,
        record_factory: Callable[[], Mapping[str, torch.Tensor]],
        *,
        force: bool = False,
    ) -> None:
        """仅在本物理步确实需要保存时才构造时间线记录。"""

        self._check_open()
        self._check_writer()
        if not self._begin_append(force):
            return
        self._append_selected(record_factory())

    def _begin_append(self, force: bool) -> bool:
        if force:
            return True
        self._physics_append_count += 1
        return self._physics_append_count % self._physics_step_stride == 0

    def _append_selected(self, record: Mapping[str, torch.Tensor]) -> None:
        record = self._select_fields(record)
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
        except BaseException:
            self._stop_writer_without_flushing()
            raise
        finally:
            self._closed = True

    def _stop_writer_without_flushing(self) -> None:
        """Release a failed writer without attempting any additional disk writes."""

        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        if self._thread.is_alive():
            try:
                self._queue.put_nowait(self._sentinel)
            except queue.Full:
                # The queue was drained above; this is only a defensive race.
                pass
            self._thread.join()

    def _allocate(self, record: Mapping[str, torch.Tensor]) -> None:
        for name, value in record.items():
            self._buffers[name] = torch.empty(
                (self._chunk_steps, *value.shape), dtype=value.dtype, device=value.device
            )

    def _select_fields(
        self, record: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        if self._fields is None:
            return record
        unknown = sorted(set(self._fields) - set(record))
        if unknown:
            raise LoggingError(f"unknown configured timeline fields: {unknown}")
        return {name: record[name] for name in self._fields}

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
                    self._atomic_torch_save(
                        payload,
                        self.directory / f"timeline_{chunk_index:06d}.pt",
                        operation=f"write simulator timeline chunk {chunk_index}",
                    )
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
            if isinstance(self._writer_error, InsufficientDiskSpaceError):
                raise self._writer_error
            raise LoggingError("timeline writer failed") from self._writer_error

    def _atomic_torch_save(
        self,
        value: Any,
        destination: Path,
        *,
        operation: str,
    ) -> None:
        """Preflight, write to a temporary file, then atomically publish it."""

        estimated_bytes = _tensor_storage_bytes(value)
        self._ensure_disk_space(estimated_bytes, operation)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            torch.save(value, temporary)
            _fsync_file(temporary)
            temporary.replace(destination)
            _fsync_directory(destination.parent)
        except BaseException as exc:
            available = _available_bytes(destination.parent)
            partial_bytes = temporary.stat().st_size if temporary.exists() else 0
            if (
                _is_no_space_error(exc)
                or available < self._minimum_free_space_bytes
            ):
                error = InsufficientDiskSpaceError(
                    str(destination),
                    available_bytes=available,
                    required_bytes=max(
                        estimated_bytes,
                        partial_bytes,
                    )
                    + self._minimum_free_space_bytes,
                    reserve_bytes=self._minimum_free_space_bytes,
                    operation=operation,
                )
                self._writer_error = error
                temporary.unlink(missing_ok=True)
                raise error from exc
            temporary.unlink(missing_ok=True)
            raise

    def _ensure_disk_space(self, write_bytes: int, operation: str) -> None:
        available = _available_bytes(self.directory)
        required = max(0, write_bytes) + self._minimum_free_space_bytes
        if available < required:
            error = InsufficientDiskSpaceError(
                str(self.directory),
                available_bytes=available,
                required_bytes=required,
                reserve_bytes=self._minimum_free_space_bytes,
                operation=operation,
            )
            self._writer_error = error
            raise error


def _tensor_storage_bytes(value: Any) -> int:
    """Conservatively estimate tensor payload bytes without serializing it."""

    seen_objects: set[int] = set()
    seen_storages: set[tuple[str, int, int]] = set()

    def visit(item: Any) -> int:
        object_id = id(item)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key in seen_storages:
                return 0
            seen_storages.add(key)
            return storage.nbytes()
        if isinstance(item, Mapping):
            return sum(visit(child) for child in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(child) for child in item)
        return 0

    return visit(value)


def _available_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _is_no_space_error(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, OSError) and current.errno in {
            errno.ENOSPC,
            errno.EDQUOT,
        }:
            return True
        message = str(current).lower()
        if (
            "no space left on device" in message
            or "disk quota exceeded" in message
            or "iostream error" in message
            or "unexpected pos" in message
        ):
            return True
        current = current.__cause__
    return False


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
