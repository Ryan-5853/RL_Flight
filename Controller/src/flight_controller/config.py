from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_controller_config(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    raw = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError("controller configuration root must be a mapping")
    if "type" not in raw:
        raise ValueError("controller configuration requires type")
    params = raw.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("controller.params must be a mapping")
    return raw
