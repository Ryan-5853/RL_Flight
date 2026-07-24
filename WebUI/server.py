#!/usr/bin/env python3
"""RL Flight WebUI static server and restricted server-side config browser."""

from __future__ import annotations

import argparse
import json
import re
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

import yaml


WEBUI_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEBUI_ROOT.parent
DEFAULT_ROOTS = {
    "simenv": PROJECT_ROOT / "SimEnv" / "configs",
    "train": PROJECT_ROOT / "Train" / "configs" / "experiments",
}
DEFAULT_CHECKPOINT_ROOTS = (PROJECT_ROOT / "Train" / "runs",)
DEFAULT_RUNTIME_LOG_ROOT = WEBUI_ROOT / "runtime-runs"
ALLOWED_SUFFIXES = {".yaml", ".yml", ".json"}
MAX_CONFIG_BYTES = 2 * 1024 * 1024


def parse_config_roots(values: list[str]) -> dict[str, Path]:
    roots = {name: path.resolve() for name, path in DEFAULT_ROOTS.items() if path.is_dir()}
    for value in values:
        if "=" not in value:
            raise ValueError(f"配置根必须使用 NAME=/absolute/path 格式：{value}")
        name, raw_path = value.split("=", 1)
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"配置根名称无效：{name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"配置根不是服务器上的目录：{path}")
        roots[name] = path
    return roots


class WebUIHandler(SimpleHTTPRequestHandler):
    server_version = "RLFlightWebUI/0.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBUI_ROOT), **kwargs)

    @property
    def config_roots(self) -> dict[str, Path]:
        return self.server.config_roots  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/config/roots":
            self._list_roots()
            return
        if parsed.path == "/api/config/files":
            self._list_files(parse_qs(parsed.query))
            return
        if parsed.path == "/api/config/file":
            self._read_file(parse_qs(parsed.query))
            return
        if parsed.path == "/api/runtime/capabilities":
            payload = self.server.runtime_registry.capabilities()  # type: ignore[attr-defined]
            payload["checkpoint_roots"] = [str(path) for path in self.server.checkpoint_roots]  # type: ignore[attr-defined]
            self._json(payload)
            return
        if parsed.path == "/api/runtime/checkpoints":
            self._list_checkpoints()
            return
        runtime_match = re.fullmatch(r"/api/runtime/sessions/([0-9a-f-]+)(?:/(status|telemetry))?", parsed.path)
        if runtime_match:
            self._runtime_get(runtime_match.group(1), runtime_match.group(2) or "status", parse_qs(parsed.query))
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/runtime/sessions":
            self._runtime_create()
            return
        runtime_match = re.fullmatch(r"/api/runtime/sessions/([0-9a-f-]+)/(control|start|pause|step|reset)", parsed.path)
        if runtime_match:
            self._runtime_post(runtime_match.group(1), runtime_match.group(2))
            return
        self._error("unknown API endpoint", HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        runtime_match = re.fullmatch(r"/api/runtime/sessions/([0-9a-f-]+)", parsed.path)
        if not runtime_match:
            self._error("unknown API endpoint", HTTPStatus.NOT_FOUND)
            return
        try:
            self.server.runtime_registry.close(runtime_match.group(1))  # type: ignore[attr-defined]
            self._json({"closed": True, "session_id": runtime_match.group(1)})
        except KeyError as error:
            self._error(str(error), HTTPStatus.NOT_FOUND)

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": message}, status)

    def _request_json(self, max_bytes: int = 5 * 1024 * 1024) -> Mapping[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length <= 0 or length > max_bytes:
            raise ValueError(f"request body must be between 1 and {max_bytes} bytes")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON request body must be an object")
        return value

    def _resolve_checkpoint(self, raw_path: str) -> Path:
        if not raw_path:
            raise ValueError("checkpoint_path is required")
        requested = Path(raw_path).expanduser()
        roots = self.server.checkpoint_roots  # type: ignore[attr-defined]
        candidates = [requested.resolve()] if requested.is_absolute() else [(root / requested).resolve() for root in roots]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix != ".pt":
                continue
            if any(_is_relative_to(candidate, root) for root in roots):
                return candidate
        raise ValueError("checkpoint must be an existing .pt file inside an allowed server checkpoint root")

    def _list_checkpoints(self) -> None:
        roots = self.server.checkpoint_roots  # type: ignore[attr-defined]
        entries = []
        for root_index, root in enumerate(roots):
            for path in root.rglob("*.pt"):
                if (
                    not path.is_file()
                    or not _is_relative_to(path.resolve(), root)
                    or not path.with_suffix(path.suffix + ".sha256").is_file()
                ):
                    continue
                stat = path.stat()
                entries.append({
                    "root": root_index,
                    "path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                })
        entries.sort(key=lambda item: item["modified_ns"], reverse=True)
        self._json({
            "roots": [{"id": index, "path": str(path)} for index, path in enumerate(roots)],
            "checkpoints": entries[:500],
            "truncated": len(entries) > 500,
        })

    def _runtime_create(self) -> None:
        try:
            body = self._request_json()
            simenv_yaml = body.get("simenv_yaml")
            test_yaml = body.get("test_yaml")
            if not isinstance(simenv_yaml, str) or not isinstance(test_yaml, str):
                raise ValueError("simenv_yaml and test_yaml are required strings")
            checkpoint = self._resolve_checkpoint(str(body.get("checkpoint_path", "")))
            session = self.server.runtime_registry.create(simenv_yaml, test_yaml, str(checkpoint))  # type: ignore[attr-defined]
            self._json({"session": session.status()}, HTTPStatus.CREATED)
        except Exception as error:
            self._error(f"{type(error).__name__}: {error}")

    def _runtime_get(self, session_id: str, resource: str, query: dict[str, list[str]]) -> None:
        try:
            session = self.server.runtime_registry.get(session_id)  # type: ignore[attr-defined]
            if resource == "status":
                self._json({"session": session.status()})
                return
            after = int(query.get("after", ["-1"])[0])
            timeout = min(1.0, max(0.0, float(query.get("timeout", ["0"])[0])))
            deadline = time.monotonic() + timeout
            telemetry = session.telemetry(after)
            while telemetry is None and time.monotonic() < deadline:
                time.sleep(.02)
                telemetry = session.telemetry(after)
            self._json({"telemetry": telemetry, "session": session.status()})
        except KeyError as error:
            self._error(str(error), HTTPStatus.NOT_FOUND)
        except (TypeError, ValueError) as error:
            self._error(str(error))

    def _runtime_post(self, session_id: str, action: str) -> None:
        try:
            session = self.server.runtime_registry.get(session_id)  # type: ignore[attr-defined]
            accepted = True
            if action == "control":
                body = self._request_json(64 * 1024)
                accepted = session.update_command(int(body.get("sequence", -1)), _require_mapping(body.get("channels"), "channels"))
            elif action == "start":
                session.start()
            elif action == "pause":
                session.pause()
            elif action == "step":
                session.step_once()
            elif action == "reset":
                session.reset()
            self._json({"accepted": accepted, "session": session.status()})
        except KeyError as error:
            self._error(str(error), HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._error(f"{type(error).__name__}: {error}")

    def _resolve(self, root_name: str, relative: str, *, require_file: bool = False) -> tuple[Path, Path]:
        root = self.config_roots.get(root_name)
        if root is None:
            raise ValueError("未知的服务器配置根目录")
        if Path(relative).is_absolute():
            raise ValueError("只允许使用配置根目录内的相对路径")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError("路径越过了允许的服务器配置根目录") from error
        if require_file:
            if not target.is_file() or target.suffix.lower() not in ALLOWED_SUFFIXES:
                raise ValueError("所选路径不是可读取的 YAML/JSON 配置文件")
        elif not target.is_dir():
            raise ValueError("所选路径不是服务器目录")
        return root, target

    def _list_roots(self) -> None:
        self._json({"roots": [{"id": name, "label": name, "path": str(path)} for name, path in self.config_roots.items()]})

    def _list_files(self, query: dict[str, list[str]]) -> None:
        try:
            root_name = query.get("root", [""])[0]
            relative = query.get("path", [""])[0]
            root, directory = self._resolve(root_name, relative)
            entries = []
            for child in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    kind = "directory"
                elif child.is_file() and child.suffix.lower() in ALLOWED_SUFFIXES:
                    kind = "file"
                else:
                    continue
                stat = child.stat()
                entries.append({
                    "name": child.name,
                    "path": child.relative_to(root).as_posix(),
                    "kind": kind,
                    "size": stat.st_size if kind == "file" else None,
                    "modified_ns": stat.st_mtime_ns,
                })
            self._json({"root": root_name, "path": directory.relative_to(root).as_posix(), "entries": entries})
        except (OSError, ValueError) as error:
            self._error(str(error))

    def _read_file(self, query: dict[str, list[str]]) -> None:
        try:
            root_name = query.get("root", [""])[0]
            relative = query.get("path", [""])[0]
            root, path = self._resolve(root_name, relative, require_file=True)
            size = path.stat().st_size
            if size > MAX_CONFIG_BYTES:
                raise ValueError(f"配置文件超过 {MAX_CONFIG_BYTES // 1024 // 1024} MiB 限制")
            text = path.read_text(encoding="utf-8")
            data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
            if not isinstance(data, dict):
                raise ValueError("配置文件顶层必须是映射对象")
            kind = "simenv" if "timing" in data and "body" in data else "test" if "environment" in data or "command_source" in data else "unknown"
            self._json({
                "root": root_name,
                "path": path.relative_to(root).as_posix(),
                "kind": kind,
                "size": size,
                "content": text,
                "config": data,
            })
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
            self._error(str(error))

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve RL Flight WebUI and restricted server config files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--config-root",
        action="append",
        default=[],
        metavar="NAME=/ABSOLUTE/PATH",
        help="add or replace an allowed server-side config root",
    )
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=[],
        metavar="/ABSOLUTE/PATH",
        help="allow runtime checkpoints from this server directory (repeatable)",
    )
    parser.add_argument(
        "--runtime-log-root",
        default=str(DEFAULT_RUNTIME_LOG_ROOT),
        metavar="/ABSOLUTE/PATH",
        help="server-owned directory for interactive SimEnv logs",
    )
    args = parser.parse_args()
    roots = parse_config_roots(args.config_root)
    checkpoint_roots = tuple(
        Path(value).expanduser().resolve() for value in args.checkpoint_root
    ) or tuple(path.resolve() for path in DEFAULT_CHECKPOINT_ROOTS if path.is_dir())
    if any(not path.is_dir() for path in checkpoint_roots):
        raise ValueError("every checkpoint root must be an existing server directory")
    runtime_log_root = Path(args.runtime_log_root).expanduser().resolve()
    runtime_log_root.mkdir(parents=True, exist_ok=True)
    from runtime import RuntimeRegistry
    server = ThreadingHTTPServer((args.host, args.port), WebUIHandler)
    server.config_roots = roots  # type: ignore[attr-defined]
    server.checkpoint_roots = checkpoint_roots  # type: ignore[attr-defined]
    server.runtime_registry = RuntimeRegistry(runtime_log_root)  # type: ignore[attr-defined]
    print(f"RL Flight WebUI: http://{args.host}:{args.port}")
    for name, path in roots.items():
        print(f"  config root {name}: {path}")
    for path in checkpoint_roots:
        print(f"  checkpoint root: {path}")
    print(f"  runtime log root: {runtime_log_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.runtime_registry.close_all()  # type: ignore[attr-defined]
        server.server_close()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


if __name__ == "__main__":
    main()
