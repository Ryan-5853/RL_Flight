#!/usr/bin/env python3
"""RL Flight WebUI static server and restricted server-side config browser."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import socket
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
    "controller": PROJECT_ROOT / "Controller" / "configs",
    "simenv": PROJECT_ROOT / "SimEnv" / "configs",
    "train": PROJECT_ROOT / "Train" / "configs" / "experiments",
}
DEFAULT_CHECKPOINT_ROOTS = (WEBUI_ROOT / "artifacts",)
DEFAULT_RUNTIME_LOG_ROOT = WEBUI_ROOT / "runtime-runs"
ALLOWED_SUFFIXES = {".yaml", ".yml", ".json"}
MAX_CONFIG_BYTES = 2 * 1024 * 1024


class WebUIHTTPServer(ThreadingHTTPServer):
    # Control and telemetry use separate persistent browser connections.  A
    # larger accept queue also protects lifecycle requests during reconnects.
    request_queue_size = 128
    daemon_threads = True
    block_on_close = False


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
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBUI_ROOT), **kwargs)

    def end_headers(self) -> None:
        # The UI and runtime protocol evolve together. Disabling browser cache
        # prevents an old runtime-client.js from talking to a new Python server.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # Synchronously printing every 20 Hz control frame and telemetry
        # long-poll perturbs the latency being measured and can grow terminal
        # logs without bound. Keep errors and lifecycle requests visible.
        status = str(args[1]) if len(args) > 1 else ""
        high_rate = bool(re.fullmatch(
            r"/api/runtime/sessions/[0-9a-f-]+/(?:control|telemetry|telemetry-stream)(?:\?.*)?",
            urlparse(self.path).path + (
                f"?{urlparse(self.path).query}"
                if urlparse(self.path).query else ""
            ),
        ))
        if high_rate and status.startswith("2"):
            return
        super().log_message(format, *args)

    @property
    def config_roots(self) -> dict[str, Path]:
        return self.server.config_roots  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        request_received_ns = time.time_ns()
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
            payload["api_version"] = 2
            payload["features"] = {
                "offline_rollout": True,
                "offline_rollout_api": "/api/runtime/rollouts",
            }
            self._json(payload)
            return
        if parsed.path == "/api/runtime/clock":
            self._json({
                "server_received_ns": request_received_ns,
                "server_response_started_ns": time.time_ns(),
            })
            return
        if parsed.path == "/api/runtime/checkpoints":
            self._list_checkpoints()
            return
        rollout_match = re.fullmatch(
            r"/api/(?:runtime/)?rollouts/([0-9a-f-]+)(?:/(data))?",
            parsed.path,
        )
        if rollout_match:
            self._rollout_get(
                rollout_match.group(1),
                rollout_match.group(2) or "status",
            )
            return
        runtime_match = re.fullmatch(
            r"/api/runtime/sessions/([0-9a-f-]+)(?:/(status|telemetry|telemetry-stream))?",
            parsed.path,
        )
        if runtime_match:
            resource = runtime_match.group(2) or "status"
            if resource == "telemetry-stream":
                self._runtime_stream(
                    runtime_match.group(1),
                    parse_qs(parsed.query),
                )
            else:
                self._runtime_get(
                    runtime_match.group(1),
                    resource,
                    parse_qs(parsed.query),
                    request_received_ns=request_received_ns,
                )
            return
        super().do_GET()

    def do_POST(self) -> None:
        request_received_ns = time.time_ns()
        parsed = urlparse(self.path)
        if parsed.path == "/api/runtime/sessions":
            self._runtime_create()
            return
        if parsed.path in {
            "/api/rollouts",
            "/api/runtime/rollouts",
        }:
            self._rollout_create()
            return
        runtime_match = re.fullmatch(r"/api/runtime/sessions/([0-9a-f-]+)/(control|start|pause|step|reset|close)", parsed.path)
        if runtime_match:
            self._runtime_post(
                runtime_match.group(1),
                runtime_match.group(2),
                request_received_ns=request_received_ns,
            )
            return
        self._error("unknown API endpoint", HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        rollout_match = re.fullmatch(
            r"/api/(?:runtime/)?rollouts/([0-9a-f-]+)",
            parsed.path,
        )
        if rollout_match:
            try:
                job = self.server.rollout_registry.cancel(  # type: ignore[attr-defined]
                    rollout_match.group(1)
                )
                self._json({"job": job.status()})
            except KeyError as error:
                self._error(str(error), HTTPStatus.NOT_FOUND)
            return
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
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A page exit can close an outstanding telemetry long-poll after
            # the response has been built. The session cleanup path owns it.
            pass

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

    def _discard_request_body(self, max_bytes: int = 64 * 1024) -> None:
        """Consume optional lifecycle bodies so HTTP/1.1 stays synchronized."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length < 0 or length > max_bytes:
            raise ValueError(f"request body must not exceed {max_bytes} bytes")
        if length:
            self.rfile.read(length)

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

    def _resolve_inference_package(self, raw_path: str) -> Path:
        if not raw_path:
            raise ValueError("inference package path is required")
        requested = Path(raw_path).expanduser()
        roots = self.server.checkpoint_roots  # type: ignore[attr-defined]
        candidates = (
            [requested.resolve()]
            if requested.is_absolute()
            else [(root / requested).resolve() for root in roots]
        )
        for candidate in candidates:
            if (
                candidate.exists()
                and any(_is_relative_to(candidate, root) for root in roots)
            ):
                return candidate
        raise ValueError(
            "inference package must exist inside an allowed package root; "
            f"received={raw_path!r}, "
            f"allowed_roots={[str(root) for root in roots]}"
        )

    def _list_checkpoints(self) -> None:
        roots = self.server.checkpoint_roots  # type: ignore[attr-defined]
        entries = []
        for root_index, root in enumerate(roots):
            # Deployment inference packages are directories whose integrity and
            # backend files are described by manifest.json.  Keep the response
            # key named "checkpoints" for compatibility with the existing UI.
            for manifest in root.rglob("manifest.json"):
                package = manifest.parent
                resolved = package.resolve()
                if not package.is_dir() or not _is_relative_to(resolved, root):
                    continue
                files = [path for path in package.iterdir() if path.is_file()]
                entries.append({
                    "root": root_index,
                    "path": package.relative_to(root).as_posix(),
                    "size": sum(path.stat().st_size for path in files),
                    "modified_ns": max(
                        (path.stat().st_mtime_ns for path in files),
                        default=manifest.stat().st_mtime_ns,
                    ),
                    "kind": "inference_package",
                })
            # Retain legacy checkpoint discovery for custom loaders that still
            # consume a .pt file with a SHA-256 sidecar.
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
                    "kind": "legacy_checkpoint",
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
            parsed_test = yaml.safe_load(test_yaml)
            if not isinstance(parsed_test, Mapping):
                raise ValueError("test_yaml must contain a mapping")
            controller = parsed_test.get("controller", {"type": "neural"})
            if not isinstance(controller, Mapping):
                raise ValueError("controller must be a mapping")
            controller_type = str(controller.get("type", "neural"))
            replace_existing = body.get("replace_existing", False)
            if not isinstance(replace_existing, bool):
                raise ValueError("replace_existing must be a boolean")
            checkpoint: Path | None = None
            if controller_type == "neural":
                checkpoint = self._resolve_inference_package(
                    str(body.get("checkpoint_path", ""))
                )
            session = self.server.runtime_registry.create(
                simenv_yaml,
                test_yaml,
                None if checkpoint is None else str(checkpoint),
                replace_existing=replace_existing,
            )  # type: ignore[attr-defined]
            self._json({"session": session.status()}, HTTPStatus.CREATED)
        except Exception as error:
            self._error(f"{type(error).__name__}: {error}")

    def _rollout_create(self) -> None:
        try:
            body = self._request_json()
            simenv_yaml = body.get("simenv_yaml")
            test_yaml = body.get("test_yaml")
            if not isinstance(simenv_yaml, str) or not isinstance(
                test_yaml, str
            ):
                raise ValueError(
                    "simenv_yaml and test_yaml are required strings"
                )
            parsed_test = yaml.safe_load(test_yaml)
            if not isinstance(parsed_test, Mapping):
                raise ValueError("test_yaml must contain a mapping")
            controller = parsed_test.get(
                "controller", {"type": "neural"}
            )
            if not isinstance(controller, Mapping):
                raise ValueError("controller must be a mapping")
            checkpoint: Path | None = None
            if str(controller.get("type", "neural")) == "neural":
                checkpoint = self._resolve_inference_package(
                    str(body.get("checkpoint_path", ""))
                )
            fps = float(body.get("fps", 30))
            if not 1 <= fps <= 60:
                raise ValueError("fps must be between 1 and 60")
            job = self.server.rollout_registry.create(  # type: ignore[attr-defined]
                simenv_yaml,
                test_yaml,
                None if checkpoint is None else str(checkpoint),
                fps=fps,
            )
            self._json(
                {"job": job.status()},
                HTTPStatus.ACCEPTED,
            )
        except Exception as error:
            self._error(f"{type(error).__name__}: {error}")

    def _rollout_get(self, job_id: str, resource: str) -> None:
        try:
            job = self.server.rollout_registry.get(job_id)  # type: ignore[attr-defined]
            if resource == "data":
                self._json({"rollout": job.result()})
            else:
                self._json({"job": job.status()})
        except KeyError as error:
            self._error(str(error), HTTPStatus.NOT_FOUND)
        except RuntimeError as error:
            self._error(str(error), HTTPStatus.CONFLICT)

    def _runtime_get(
        self,
        session_id: str,
        resource: str,
        query: dict[str, list[str]],
        *,
        request_received_ns: int,
    ) -> None:
        try:
            session = self.server.runtime_registry.get(session_id)  # type: ignore[attr-defined]
            if resource == "status":
                self._json({"session": session.status()})
                return
            after = int(query.get("after", ["-1"])[0])
            timeout = min(1.0, max(0.0, float(query.get("timeout", ["0"])[0])))
            telemetry = session.wait_telemetry(after, timeout)
            if telemetry is not None:
                trace = telemetry.setdefault("latency_trace", {})
                trace["telemetry_transport"] = "long_poll"
                trace["telemetry_poll_received_ns"] = request_received_ns
                trace["server_response_started_ns"] = time.time_ns()
            self._json({"telemetry": telemetry, "session": session.status()})
        except KeyError as error:
            self._error(str(error), HTTPStatus.NOT_FOUND)
        except (TypeError, ValueError) as error:
            self._error(str(error))

    @staticmethod
    def _compact_webui_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
        """Remove fields the browser does not render while preserving the API."""
        truth = telemetry.get("truth")
        if isinstance(truth, dict):
            visible_truth = {
                "position_n",
                "attitude_q_wb",
                "angular_velocity_b",
                "motor_speed",
                "servo_angle",
                "grid_force_b",
                "force_b",
                "grid_moment_b",
                "moment_b",
            }
            telemetry["truth"] = {
                key: value for key, value in truth.items() if key in visible_truth
            }
        controller = telemetry.get("controller")
        if isinstance(controller, dict):
            diagnostics = controller.get("diagnostics")
            if isinstance(diagnostics, dict):
                controller["diagnostics"] = {
                    key: value
                    for key, value in diagnostics.items()
                    if key == "controller.lqr_blend"
                }
        reference = telemetry.get("reference")
        if isinstance(reference, dict):
            telemetry["reference"] = {
                key: value
                for key, value in reference.items()
                if key == "target_attitude_q_wb"
            }
        return telemetry

    def _write_stream_chunk(self, payload: object) -> None:
        body = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        self.wfile.write(f"{len(body):X}\r\n".encode("ascii"))
        self.wfile.write(body)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _runtime_stream(
        self,
        session_id: str,
        query: dict[str, list[str]],
    ) -> None:
        try:
            session = self.server.runtime_registry.get(session_id)  # type: ignore[attr-defined]
            after = int(query.get("after", ["-1"])[0])
        except KeyError as error:
            self._error(str(error), HTTPStatus.NOT_FOUND)
            return
        except (TypeError, ValueError) as error:
            self._error(str(error))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            "application/x-ndjson; charset=utf-8",
        )
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        last_write = time.monotonic()
        try:
            while True:
                telemetry = session.wait_telemetry(after, 1.0)
                now = time.monotonic()
                if telemetry is not None:
                    after = int(telemetry["sequence"])
                    telemetry = self._compact_webui_telemetry(telemetry)
                    trace = telemetry.setdefault("latency_trace", {})
                    trace["telemetry_transport"] = "stream"
                    trace["server_response_started_ns"] = time.time_ns()
                    status = session.status()
                    self._write_stream_chunk({
                        "telemetry": telemetry,
                        "session": status,
                    })
                    last_write = now
                    if status["state"] in {"closed", "faulted"}:
                        break
                elif now - last_write >= 1.0:
                    status = session.status()
                    self._write_stream_chunk({
                        "heartbeat": True,
                        "session": status,
                    })
                    last_write = now
                    if status["state"] in {"closed", "faulted"}:
                        break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _runtime_post(
        self,
        session_id: str,
        action: str,
        *,
        request_received_ns: int,
    ) -> None:
        try:
            if action == "close":
                self._discard_request_body()
                self.server.runtime_registry.close(session_id)  # type: ignore[attr-defined]
                self._json({"closed": True, "session_id": session_id})
                return
            session = self.server.runtime_registry.get(session_id)  # type: ignore[attr-defined]
            accepted = True
            if action == "control":
                body = self._request_json(64 * 1024)
                accepted = session.update_command(
                    int(body.get("sequence", -1)),
                    _require_mapping(body.get("channels"), "channels"),
                    trace=_require_mapping(body.get("trace", {}), "trace"),
                    server_received_ns=request_received_ns,
                )
                # This endpoint is the high-rate input path. Returning the full
                # runtime status here needlessly couples input latency to
                # simulation/status lock contention.
                self._json({
                    "accepted": accepted,
                    "server_control_received_ns": request_received_ns,
                    "server_control_response_started_ns": time.time_ns(),
                })
                return
            elif action == "start":
                body = self._request_json(64 * 1024)
                accepted = session.start(
                    int(body.get("sequence", -1)),
                    _require_mapping(body.get("channels"), "channels"),
                    trace=_require_mapping(body.get("trace", {}), "trace"),
                    server_received_ns=request_received_ns,
                )
            elif action == "pause":
                self._discard_request_body()
                session.pause()
            elif action == "step":
                body = self._request_json(64 * 1024)
                accepted = session.step_once(
                    int(body.get("sequence", -1)),
                    _require_mapping(body.get("channels"), "channels"),
                    trace=_require_mapping(body.get("trace", {}), "trace"),
                    server_received_ns=request_received_ns,
                )
            elif action == "reset":
                self._discard_request_body()
                session.reset()
            if action in {"start", "step"}:
                # Start/step already accepted the input atomically. Respond
                # before any status serialization so the browser can begin its
                # steady control pump without consuming the watchdog window.
                self._json({
                    "accepted": accepted,
                    "server_control_received_ns": request_received_ns,
                    "server_control_response_started_ns": time.time_ns(),
                })
                return
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
            kind = (
                "simenv"
                if "timing" in data and "body" in data
                else "test"
                if "environment" in data or "command_source" in data
                else "controller"
                if "type" in data and isinstance(data.get("params", {}), Mapping)
                else "unknown"
            )
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
        "--runtime-log-root",
        default=str(DEFAULT_RUNTIME_LOG_ROOT),
        metavar="/ABSOLUTE/PATH",
        help="server-owned directory for interactive SimEnv logs",
    )
    parser.add_argument(
        "--inference-loader",
        default="",
        metavar="MODULE:CALLABLE",
        help=(
            "deployment inference package loader; callable signature is "
            "(path, device, dtype) -> RealtimeInferencePackage"
        ),
    )
    args = parser.parse_args()
    roots = parse_config_roots(args.config_root)
    checkpoint_roots = tuple(path.resolve() for path in DEFAULT_CHECKPOINT_ROOTS)
    if any(not path.is_dir() for path in checkpoint_roots):
        raise ValueError(
            "WebUI inference package directory does not exist: "
            + ", ".join(str(path) for path in checkpoint_roots)
        )
    runtime_log_root = Path(args.runtime_log_root).expanduser().resolve()
    runtime_log_root.mkdir(parents=True, exist_ok=True)
    from inference_package import load_flight_deploy_package

    inference_loader = load_flight_deploy_package
    if args.inference_loader:
        if ":" not in args.inference_loader:
            raise ValueError(
                "--inference-loader must use MODULE:CALLABLE syntax"
            )
        module_name, attribute = args.inference_loader.split(":", 1)
        inference_loader = getattr(
            importlib.import_module(module_name), attribute
        )
        if not callable(inference_loader):
            raise ValueError("--inference-loader target must be callable")
    from runtime import OfflineRolloutRegistry, RuntimeRegistry
    server = WebUIHTTPServer((args.host, args.port), WebUIHandler)
    server.config_roots = roots  # type: ignore[attr-defined]
    server.checkpoint_roots = checkpoint_roots  # type: ignore[attr-defined]
    server.runtime_registry = RuntimeRegistry(  # type: ignore[attr-defined]
        runtime_log_root,
        inference_package_loader=inference_loader,
    )
    server.rollout_registry = OfflineRolloutRegistry(  # type: ignore[attr-defined]
        runtime_log_root,
        inference_package_loader=inference_loader,
    )
    print(f"RL Flight WebUI: http://{args.host}:{args.port}")
    for name, path in roots.items():
        print(f"  config root {name}: {path}")
    for path in checkpoint_roots:
        print(f"  inference package root: {path}")
    print(f"  runtime log root: {runtime_log_root}")
    print(
        "  inference loader: "
        + (args.inference_loader or "flight_deploy default")
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.rollout_registry.close_all()  # type: ignore[attr-defined]
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
