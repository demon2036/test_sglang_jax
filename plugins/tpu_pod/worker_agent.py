#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _read_json(handler: BaseHTTPRequestHandler) -> Any:
    content_length = handler.headers.get("content-length")
    if not content_length:
        return None
    raw = handler.rfile.read(int(content_length))
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def _write_json(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: Any,
    *,
    extra_headers: dict[str, str] | None = None,
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    if extra_headers:
        for key, value in extra_headers.items():
            handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


@dataclass
class _ManagedMockServer:
    backend_id: int
    server: Any
    thread: threading.Thread


@dataclass
class _ManagedSubprocess:
    pid: int
    popen: subprocess.Popen[bytes]
    log_path: str


@dataclass
class _ManagedService:
    kind: str
    ports: list[int]
    started_at: float
    mock_servers: list[_ManagedMockServer]
    processes: list[_ManagedSubprocess]


class _AgentState:
    def __init__(self, *, workdir: Path, quiet: bool):
        self._lock = threading.Lock()
        self._service: _ManagedService | None = None
        self.workdir = workdir
        self.quiet = quiet

    def status(self) -> dict[str, Any]:
        with self._lock:
            service = self._service
            if service is None:
                return {"running": False}
            return {
                "running": True,
                "kind": service.kind,
                "ports": list(service.ports),
                "started_at": service.started_at,
                "pids": [p.pid for p in service.processes],
                "log_paths": [p.log_path for p in service.processes],
            }

    def _stop_locked(self) -> None:
        service = self._service
        self._service = None
        if service is None:
            return
        for entry in service.mock_servers:
            try:
                entry.server.shutdown()
            except Exception:
                pass
            try:
                entry.server.server_close()
            except Exception:
                pass
        for entry in service.processes:
            try:
                entry.popen.terminate()
            except Exception:
                pass
        for entry in service.processes:
            try:
                entry.popen.wait(timeout=30)
            except Exception:
                try:
                    entry.popen.kill()
                except Exception:
                    pass

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def start_mock_openai(
        self,
        *,
        host: str,
        num_servers: int,
        base_port: int,
        model_id: str,
        api_key: str,
        force: bool,
    ) -> dict[str, Any]:
        from plugins.openai_lb.mock_openai_server import make_server, start_server_in_thread

        with self._lock:
            if self._service is not None:
                if not force:
                    raise RuntimeError("service already running")
                self._stop_locked()

            mock_servers: list[_ManagedMockServer] = []
            ports: list[int] = []
            for idx in range(num_servers):
                port = base_port + idx if base_port else 0
                server = make_server(
                    host=host,
                    port=port,
                    backend_id=idx,
                    model_id=model_id,
                    api_key=api_key,
                    quiet=self.quiet,
                )
                ports.append(int(server.server_address[1]))
                thread = start_server_in_thread(server)
                mock_servers.append(_ManagedMockServer(backend_id=idx, server=server, thread=thread))

            self._service = _ManagedService(
                kind="mock-openai",
                ports=ports,
                started_at=time.time(),
                mock_servers=mock_servers,
                processes=[],
            )
            return {"ok": True, "kind": "mock-openai", "ports": ports}

    def start_sglang_multi_openai(
        self,
        *,
        python_exe: str,
        model: str,
        num_servers: int,
        base_port: int,
        host: str,
        load_format: str,
        context_length: int | None,
        max_total_tokens: int | None,
        max_prefill_tokens: int | None,
        api_key: str,
        enable_weight_reload: bool,
        force: bool,
        log_name: str,
    ) -> dict[str, Any]:
        with self._lock:
            if self._service is not None:
                if not force:
                    raise RuntimeError("service already running")
                self._stop_locked()

            log_path = str((self.workdir / log_name).resolve())
            log_file = open(log_path, "ab", buffering=0)
            cmd = [
                python_exe,
                "-u",
                "-m",
                "plugins.sglang_jax.run_multi_openai_servers",
                "--model",
                model,
                "--num-servers",
                str(int(num_servers)),
                "--base-port",
                str(int(base_port)),
                "--host",
                host,
                "--load-format",
                load_format,
                "--keep-running",
            ]
            if context_length is not None and int(context_length) > 0:
                cmd += ["--context-length", str(int(context_length))]
            if max_total_tokens is not None and int(max_total_tokens) > 0:
                cmd += ["--max-total-tokens", str(int(max_total_tokens))]
            if max_prefill_tokens is not None and int(max_prefill_tokens) > 0:
                cmd += ["--max-prefill-tokens", str(int(max_prefill_tokens))]
            if api_key:
                cmd += ["--api-key", api_key]
            if enable_weight_reload:
                cmd += ["--enable-weight-reload"]

            popen = subprocess.Popen(
                cmd,
                cwd=str(self.workdir),
                stdout=log_file,
                stderr=log_file,
            )
            processes = [_ManagedSubprocess(pid=int(popen.pid), popen=popen, log_path=log_path)]
            ports = [int(base_port) + i for i in range(int(num_servers))]
            self._service = _ManagedService(
                kind="sglang-jax-multi-openai",
                ports=ports,
                started_at=time.time(),
                mock_servers=[],
                processes=processes,
            )
            return {
                "ok": True,
                "kind": "sglang-jax-multi-openai",
                "ports": ports,
                "pid": int(popen.pid),
                "log_path": log_path,
            }


class _AgentHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        if getattr(self.server, "quiet", False):
            return
        super().log_message(f"[agent] {fmt}", *args)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return
        if path == "/admin/status":
            state: _AgentState = getattr(self.server, "state")
            _write_json(self, HTTPStatus.OK, {"ok": True, **state.status()})
            return
        _write_json(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown path: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        state: _AgentState = getattr(self.server, "state")
        if path == "/admin/stop":
            state.stop()
            _write_json(self, HTTPStatus.OK, {"ok": True, "stopped": True})
            return
        if path != "/admin/start":
            _write_json(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown path: {path}"})
            return

        try:
            body = _read_json(self) or {}
        except ValueError as exc:
            _write_json(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        kind = str(body.get("kind") or "").strip()
        force = bool(body.get("force") or False)

        try:
            if kind == "mock-openai":
                host = str(body.get("host") or "127.0.0.1")
                num_servers = int(body.get("num_servers") or 4)
                base_port = int(body.get("base_port") or 0)
                model_id = str(body.get("model_id") or "mock-model")
                api_key = str(body.get("api_key") or "")
                result = state.start_mock_openai(
                    host=host,
                    num_servers=num_servers,
                    base_port=base_port,
                    model_id=model_id,
                    api_key=api_key,
                    force=force,
                )
                _write_json(self, HTTPStatus.OK, result)
                return

            if kind == "sglang-jax-multi-openai":
                python_exe = str(body.get("python") or "python")
                model = str(body.get("model") or "").strip()
                if not model:
                    raise ValueError("model is required")
                num_servers = int(body.get("num_servers") or 4)
                base_port = int(body.get("base_port") or 31000)
                host = str(body.get("host") or "0.0.0.0")
                load_format = str(body.get("load_format") or "auto")
                context_length_raw = body.get("context_length")
                max_total_tokens_raw = body.get("max_total_tokens")
                max_prefill_tokens_raw = body.get("max_prefill_tokens")
                context_length = (
                    int(context_length_raw) if context_length_raw is not None else None
                )
                max_total_tokens = (
                    int(max_total_tokens_raw) if max_total_tokens_raw is not None else None
                )
                max_prefill_tokens = (
                    int(max_prefill_tokens_raw)
                    if max_prefill_tokens_raw is not None
                    else None
                )
                api_key = str(body.get("api_key") or "")
                enable_weight_reload = bool(body.get("enable_weight_reload") or False)
                log_name = str(body.get("log_name") or "agent_sglang_openai.log")
                result = state.start_sglang_multi_openai(
                    python_exe=python_exe,
                    model=model,
                    num_servers=num_servers,
                    base_port=base_port,
                    host=host,
                    load_format=load_format,
                    context_length=context_length,
                    max_total_tokens=max_total_tokens,
                    max_prefill_tokens=max_prefill_tokens,
                    api_key=api_key,
                    enable_weight_reload=enable_weight_reload,
                    force=force,
                    log_name=log_name,
                )
                _write_json(self, HTTPStatus.OK, result)
                return

            raise ValueError(f"unsupported kind: {kind!r}")
        except Exception as exc:
            _write_json(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})


def make_server(*, host: str, port: int, workdir: Path, quiet: bool) -> ThreadingHTTPServer:
    state = _AgentState(workdir=workdir, quiet=quiet)
    server = ThreadingHTTPServer((host, port), _AgentHandler)
    server.state = state  # type: ignore[attr-defined]
    server.quiet = quiet  # type: ignore[attr-defined]
    return server


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--workdir", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()
    server = make_server(host=args.host, port=int(args.port), workdir=workdir, quiet=bool(args.quiet))
    print(f"READY worker_agent url=http://{args.host}:{int(args.port)} workdir={workdir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.state.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
