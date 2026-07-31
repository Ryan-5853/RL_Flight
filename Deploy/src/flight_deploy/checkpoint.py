from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .errors import CheckpointError
from .hashing import sha256_file


def verify_checkpoint_checksum(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise CheckpointError(f"checkpoint does not exist: {source}")
    checksum_path = source.with_suffix(source.suffix + ".sha256")
    actual = sha256_file(source)
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
        if expected != actual:
            raise CheckpointError(f"checkpoint checksum mismatch: {source}")
    return actual


def load_checkpoint(
    path: str | Path,
    *,
    verify_checksum: bool = True,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = Path(path).expanduser().resolve()
    if verify_checksum:
        digest = verify_checkpoint_checksum(source)
        checksum_verified = True
    else:
        if not source.is_file():
            raise CheckpointError(f"checkpoint does not exist: {source}")
        checksum_path = source.with_suffix(source.suffix + ".sha256")
        digest = (
            checksum_path.read_text(encoding="utf-8").strip().split()[0]
            if checksum_path.is_file()
            else "unverified"
        )
        checksum_verified = False
    try:
        value = torch.load(
            source,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise CheckpointError(f"failed to load checkpoint {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CheckpointError("checkpoint root must be a mapping")
    metadata: dict[str, Any] = {
        "path": str(source),
        "sha256": digest,
        "checksum_verified": checksum_verified,
        "size_bytes": source.stat().st_size,
    }
    sidecar = source.with_suffix(".json")
    if sidecar.is_file():
        try:
            selection = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            selection = None
        if isinstance(selection, Mapping):
            metadata["selection"] = dict(selection)
    return value, metadata
