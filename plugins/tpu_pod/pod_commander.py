#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _detect_local_ip() -> str:
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except Exception:
        return "127.0.0.1"
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


def _http_json(
    *,
    method: str,
    url: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", ""):
        raise ValueError(f"unsupported scheme: {parsed.scheme}")
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 80)
    conn = HTTPConnection(host, port, timeout=timeout_sec)
    data = json.dumps(body or {}, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}
    try:
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        if not raw:
            payload: Any = None
        else:
            payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload, dict):
            payload.setdefault("_status", int(resp.status))
            return payload
        return {"_status": int(resp.status), "data": payload}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _wait_ready(*, host: str, port: int, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_error: str | None = None
    while time.time() < deadline:
        conn: HTTPConnection | None = None
        try:
            conn = HTTPConnection(host, int(port), timeout=5.0)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status == 200:
                return
            last_error = f"status={resp.status} body={raw[:200]!r}"
        except Exception as exc:
            last_error = str(exc)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        time.sleep(1)
    raise TimeoutError(f"{host}:{port} not ready: {last_error}")


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


@dataclass(frozen=True)
class Backend:
    host: str
    port: int

    @property
    def label(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class _LocalMockBackend:
    server: Any
    thread: threading.Thread


@dataclass
class _LocalService:
    kind: str
    ports: list[int]
    local_host: str
    mock_backends: list[_LocalMockBackend]
    process: subprocess.Popen[bytes] | None
    log_path: str | None


class PodCommander:
    def __init__(
        self,
        *,
        workdir: Path,
        backend_kind: str,
        model: str,
        num_servers_per_worker: int,
        base_port: int,
        bind_host: str,
        load_format: str,
        context_length: int | None = None,
        max_total_tokens: int | None = None,
        max_prefill_tokens: int | None = None,
        enable_weight_reload: bool,
        python_exe: str,
        remote_agents: list[str],
        startup_timeout_sec: float,
        registry_host: str,
        registry_port: int,
        advertise_host: str,
        quiet: bool,
    ):
        self._workdir = workdir
        self._backend_kind = backend_kind
        self._model = model
        self._num_servers = int(num_servers_per_worker)
        self._base_port = int(base_port)
        self._bind_host = bind_host
        self._load_format = load_format
        self._context_length = int(context_length) if context_length else None
        self._max_total_tokens = int(max_total_tokens) if max_total_tokens else None
        self._max_prefill_tokens = (
            int(max_prefill_tokens) if max_prefill_tokens else None
        )
        self._enable_weight_reload = enable_weight_reload
        self._python_exe = python_exe
        self._remote_agents = list(remote_agents)
        self._startup_timeout_sec = float(startup_timeout_sec)
        self._registry_host = registry_host
        self._registry_port = int(registry_port)
        self._advertise_host = advertise_host
        self._quiet = quiet

        self._local_service: _LocalService | None = None
        self._remote_ports: dict[str, list[int]] = {}
        self._registry: ThreadingHTTPServer | None = None
        self._registry_thread: threading.Thread | None = None

    @property
    def registry_url(self) -> str | None:
        if self._registry is None:
            return None
        host = self._registry.server_address[0]
        port = int(self._registry.server_address[1])
        return f"http://{host}:{port}"

    def backends(self) -> list[Backend]:
        backends: list[Backend] = []
        if self._local_service is not None:
            for port in self._local_service.ports:
                backends.append(Backend(host=self._advertise_host, port=int(port)))
        for agent_url, ports in self._remote_ports.items():
            host = urlparse(agent_url).hostname or agent_url
            for port in ports:
                backends.append(Backend(host=host, port=int(port)))
        return backends

    def start(self) -> None:
        self._start_local()
        self._start_remotes()
        self._wait_all_ready()
        self._start_registry()

    def _start_local(self) -> None:
        if self._backend_kind == "mock-openai":
            from plugins.openai_lb.mock_openai_server import make_server, start_server_in_thread

            mock_backends: list[_LocalMockBackend] = []
            ports: list[int] = []
            bind_host = "127.0.0.1" if self._bind_host == "0.0.0.0" else self._bind_host
            for idx in range(self._num_servers):
                port = self._base_port + idx if self._base_port else 0
                server = make_server(
                    host=bind_host,
                    port=port,
                    backend_id=1000 + idx,
                    model_id=self._model,
                    api_key="",
                    quiet=self._quiet,
                )
                ports.append(int(server.server_address[1]))
                thread = start_server_in_thread(server)
                mock_backends.append(_LocalMockBackend(server=server, thread=thread))
            self._local_service = _LocalService(
                kind="mock-openai",
                ports=ports,
                local_host=self._advertise_host,
                mock_backends=mock_backends,
                process=None,
                log_path=None,
            )
            return

        if self._backend_kind != "sglang-jax-multi-openai":
            raise ValueError(f"unsupported backend_kind: {self._backend_kind!r}")

        log_path = str((self._workdir / "commander_local_sglang_openai.log").resolve())
        log_file = open(log_path, "ab", buffering=0)
        cmd = [
            self._python_exe,
            "-u",
            "-m",
            "plugins.sglang_jax.run_multi_openai_servers",
            "--model",
            self._model,
            "--num-servers",
            str(self._num_servers),
            "--base-port",
            str(self._base_port),
            "--host",
            self._bind_host,
            "--load-format",
            self._load_format,
            "--keep-running",
        ]
        if self._context_length is not None and int(self._context_length) > 0:
            cmd += ["--context-length", str(int(self._context_length))]
        if self._max_total_tokens is not None and int(self._max_total_tokens) > 0:
            cmd += ["--max-total-tokens", str(int(self._max_total_tokens))]
        if self._max_prefill_tokens is not None and int(self._max_prefill_tokens) > 0:
            cmd += ["--max-prefill-tokens", str(int(self._max_prefill_tokens))]
        if self._enable_weight_reload:
            cmd += ["--enable-weight-reload"]
        proc = subprocess.Popen(cmd, cwd=str(self._workdir), stdout=log_file, stderr=log_file)
        ports = [self._base_port + i for i in range(self._num_servers)]
        self._local_service = _LocalService(
            kind="sglang-jax-multi-openai",
            ports=ports,
            local_host=self._advertise_host,
            mock_backends=[],
            process=proc,
            log_path=log_path,
        )

    def _start_remotes(self) -> None:
        for agent_url in self._remote_agents:
            payload: dict[str, Any]
            if self._backend_kind == "mock-openai":
                payload = {
                    "kind": "mock-openai",
                    "host": "127.0.0.1",
                    "num_servers": self._num_servers,
                    "base_port": 0,
                    "model_id": self._model,
                    "force": True,
                }
            else:
                payload = {
                    "kind": "sglang-jax-multi-openai",
                    "python": self._python_exe,
                    "model": self._model,
                    "num_servers": self._num_servers,
                    "base_port": self._base_port,
                    "host": self._bind_host,
                    "load_format": self._load_format,
                    "enable_weight_reload": self._enable_weight_reload,
                    "force": True,
                    "log_name": f"agent_sglang_openai_{int(time.time())}.log",
                }
                if self._context_length is not None and int(self._context_length) > 0:
                    payload["context_length"] = int(self._context_length)
                if self._max_total_tokens is not None and int(self._max_total_tokens) > 0:
                    payload["max_total_tokens"] = int(self._max_total_tokens)
                if (
                    self._max_prefill_tokens is not None
                    and int(self._max_prefill_tokens) > 0
                ):
                    payload["max_prefill_tokens"] = int(self._max_prefill_tokens)
            resp = _http_json(method="POST", url=agent_url, path="/admin/start", body=payload, timeout_sec=30.0)
            if not resp.get("ok"):
                raise RuntimeError(f"remote start failed for {agent_url}: {resp}")
            ports = [int(p) for p in (resp.get("ports") or [])]
            if len(ports) != self._num_servers:
                raise RuntimeError(f"expected {self._num_servers} ports from {agent_url}, got {ports}")
            self._remote_ports[agent_url] = ports

    def _wait_all_ready(self) -> None:
        for backend in self.backends():
            _wait_ready(host=backend.host, port=backend.port, timeout_sec=self._startup_timeout_sec)

    def _start_registry(self) -> None:
        commander = self

        class _RegistryHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
                if commander._quiet:
                    return
                super().log_message(f"[registry] {fmt}", *args)

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/health":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Length", "0")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    return
                if path == "/backends":
                    _write_json(
                        self,
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "model": commander._model,
                            "backends": [b.label for b in commander.backends()],
                        },
                    )
                    return
                _write_json(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown path: {path}"})

        server = ThreadingHTTPServer((self._registry_host, self._registry_port), _RegistryHandler)
        self._registry_port = int(server.server_address[1])
        self._registry = server
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._registry_thread = thread
        thread.start()

    def stop(self) -> None:
        if self._registry is not None:
            try:
                self._registry.shutdown()
            except Exception:
                pass
            try:
                self._registry.server_close()
            except Exception:
                pass
            self._registry = None

        for agent_url in self._remote_agents:
            try:
                _http_json(method="POST", url=agent_url, path="/admin/stop", body={}, timeout_sec=10.0)
            except Exception:
                pass

        local = self._local_service
        self._local_service = None
        if local is None:
            return
        for entry in local.mock_backends:
            try:
                entry.server.shutdown()
            except Exception:
                pass
            try:
                entry.server.server_close()
            except Exception:
                pass
        if local.process is not None:
            try:
                local.process.terminate()
            except Exception:
                pass
            try:
                local.process.wait(timeout=30)
            except Exception:
                try:
                    local.process.kill()
                except Exception:
                    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend-kind",
        type=str,
        default="sglang-jax-multi-openai",
        choices=["sglang-jax-multi-openai", "mock-openai"],
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-servers-per-worker", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=31000)
    parser.add_argument("--bind-host", type=str, default="0.0.0.0")
    parser.add_argument("--load-format", type=str, default="auto")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--max-total-tokens", type=int, default=128)
    parser.add_argument("--max-prefill-tokens", type=int, default=128)
    parser.add_argument("--enable-weight-reload", action="store_true")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument(
        "--remote-agents",
        type=str,
        default="",
        help="Comma-separated agent base URLs, e.g. http://10.0.0.2:9010",
    )
    parser.add_argument("--startup-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--registry-host", type=str, default="0.0.0.0")
    parser.add_argument("--registry-port", type=int, default=9100)
    parser.add_argument(
        "--advertise-host",
        type=str,
        default="",
        help="Host/IP to advertise for local backends (default: auto-detect)",
    )
    parser.add_argument("--workdir", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()
    advertise_host = str(args.advertise_host).strip() or _detect_local_ip()
    remote_agents = [a.strip() for a in str(args.remote_agents).split(",") if a.strip()]
    commander = PodCommander(
        workdir=workdir,
        backend_kind=str(args.backend_kind),
        model=str(args.model),
        num_servers_per_worker=int(args.num_servers_per_worker),
        base_port=int(args.base_port),
        bind_host=str(args.bind_host),
        load_format=str(args.load_format),
        context_length=int(args.context_length),
        max_total_tokens=int(args.max_total_tokens),
        max_prefill_tokens=int(args.max_prefill_tokens),
        enable_weight_reload=bool(args.enable_weight_reload),
        python_exe=str(args.python),
        remote_agents=remote_agents,
        startup_timeout_sec=float(args.startup_timeout_sec),
        registry_host=str(args.registry_host),
        registry_port=int(args.registry_port),
        advertise_host=advertise_host,
        quiet=bool(args.quiet),
    )
    commander.start()
    backends = [b.label for b in commander.backends()]
    print(
        "READY pod_commander "
        f"model={args.model} advertise_host={advertise_host} "
        f"backends={backends} registry=http://{args.registry_host}:{int(args.registry_port)}",
        flush=True,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        commander.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
