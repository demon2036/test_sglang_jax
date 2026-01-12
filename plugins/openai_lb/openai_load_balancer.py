#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    content_length = handler.headers.get("content-length")
    if not content_length:
        return b""
    return handler.rfile.read(int(content_length))


@dataclass(frozen=True)
class Backend:
    host: str
    port: int

    @property
    def label(self) -> str:
        return f"{self.host}:{self.port}"


class RoundRobinPool:
    def __init__(self, backends: list[Backend]):
        if not backends:
            raise ValueError("backends must not be empty")
        self._backends = list(backends)
        self._index = 0
        self._lock = threading.Lock()

    @property
    def backends(self) -> list[Backend]:
        return list(self._backends)

    def choose_order(self) -> list[Backend]:
        with self._lock:
            start = self._index
            self._index = (self._index + 1) % len(self._backends)
        return [self._backends[(start + i) % len(self._backends)] for i in range(len(self._backends))]


class _OpenAILoadBalancerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        if getattr(self.server, "quiet", False):
            return
        super().log_message(f"[lb] {fmt}", *args)

    def _require_api_key(self) -> bool:
        api_key = getattr(self.server, "api_key", "")
        if not api_key:
            return True
        auth = self.headers.get("authorization") or ""
        return auth == f"Bearer {api_key}"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._handle_health()
            return
        if path == "/admin/backends":
            pool: RoundRobinPool = getattr(self.server, "pool")
            _write_json(
                self,
                HTTPStatus.OK,
                {"ok": True, "backends": [b.label for b in pool.backends]},
            )
            return
        self._proxy_request()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy_request()

    def _handle_health(self) -> None:
        pool: RoundRobinPool = getattr(self.server, "pool")
        for backend in pool.choose_order():
            ok = self._probe_backend_health(backend)
            if ok:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                return
        self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def _probe_backend_health(self, backend: Backend) -> bool:
        conn: HTTPConnection | None = None
        timeout: float = float(getattr(self.server, "backend_timeout_sec", 60.0))
        try:
            conn = HTTPConnection(backend.host, backend.port, timeout=timeout)
            conn.request("GET", "/health", headers={"Connection": "close"})
            resp = conn.getresponse()
            resp.read()
            return resp.status == 200
        except Exception:
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _proxy_request(self) -> None:
        if not self._require_api_key():
            _write_json(self, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        body = _read_body(self)
        method = self.command
        path = self.path

        last_error: str | None = None
        pool: RoundRobinPool = getattr(self.server, "pool")
        for backend in pool.choose_order():
            try:
                response = self._forward_to_backend(backend, method=method, path=path, body=body)
                self._write_backend_response(backend, response)
                return
            except Exception as exc:
                last_error = str(exc)
                continue

        _write_json(
            self,
            HTTPStatus.BAD_GATEWAY,
            {"error": "all backends failed", "detail": last_error or "unknown"},
        )

    def _forward_to_backend(self, backend: Backend, *, method: str, path: str, body: bytes) -> HTTPResponse:
        headers = {}
        for key, value in self.headers.items():
            if key.lower() in ("host", "connection", "content-length"):
                continue
            headers[key] = value
        headers["Host"] = backend.label
        headers["Connection"] = "close"
        if body:
            headers["Content-Length"] = str(len(body))

        timeout: float = float(getattr(self.server, "backend_timeout_sec", 60.0))
        conn = HTTPConnection(backend.host, backend.port, timeout=timeout)
        try:
            conn.request(method, path, body=body if body else None, headers=headers)
            resp = conn.getresponse()
            resp._lb_conn = conn  # type: ignore[attr-defined]
            return resp
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

    def _write_backend_response(self, backend: Backend, resp: HTTPResponse) -> None:
        conn = getattr(resp, "_lb_conn", None)
        try:
            self.send_response(resp.status)
            content_length = resp.getheader("Content-Length")
            transfer_encoding = resp.getheader("Transfer-Encoding")

            for key, value in resp.getheaders():
                lower = key.lower()
                if lower in ("connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade"):
                    continue
                if lower == "transfer-encoding":
                    continue
                if lower == "content-length":
                    continue
                self.send_header(key, value)

            self.send_header("X-OpenAI-LB-Backend", backend.label)

            if content_length is not None:
                self.send_header("Content-Length", content_length)
                self.send_header("Connection", "close")
                self.end_headers()
                self._stream_body(resp, chunked=False)
                return

            if transfer_encoding and "chunked" in transfer_encoding.lower():
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                self._stream_body(resp, chunked=True)
                return

            body = resp.read()
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if body:
                self.wfile.write(body)
        finally:
            try:
                resp.close()
            except Exception:
                pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _stream_body(self, resp: HTTPResponse, *, chunked: bool) -> None:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            if chunked:
                header = f"{len(chunk):X}\r\n".encode("ascii")
                self.wfile.write(header)
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            else:
                self.wfile.write(chunk)
                self.wfile.flush()
        if chunked:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()


def make_server(
    *,
    listen_host: str,
    listen_port: int,
    backends: list[Backend],
    api_key: str = "",
    backend_timeout_sec: float = 60.0,
    quiet: bool = False,
) -> ThreadingHTTPServer:
    pool = RoundRobinPool(backends)
    server = ThreadingHTTPServer((listen_host, listen_port), _OpenAILoadBalancerHandler)
    server.pool = pool  # type: ignore[attr-defined]
    server.api_key = api_key  # type: ignore[attr-defined]
    server.backend_timeout_sec = backend_timeout_sec  # type: ignore[attr-defined]
    server.quiet = quiet  # type: ignore[attr-defined]
    return server


def start_server_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _parse_backend(value: str) -> Backend:
    value = value.strip()
    if not value:
        raise ValueError("backend is empty")
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    host = parsed.hostname or ""
    port = parsed.port or 0
    if not host or not port:
        raise ValueError(f"Invalid backend: {value}")
    return Backend(host=host, port=int(port))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", type=str, default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument(
        "--backends",
        type=str,
        required=True,
        help="Comma-separated list, e.g. 127.0.0.1:31000,127.0.0.1:31001",
    )
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--backend-timeout-sec", type=float, default=60.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    backend_list = [_parse_backend(item) for item in args.backends.split(",") if item.strip()]
    server = make_server(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        backends=backend_list,
        api_key=args.api_key,
        backend_timeout_sec=args.backend_timeout_sec,
        quiet=args.quiet,
    )
    print(
        f"READY openai_load_balancer url=http://{args.listen_host}:{args.listen_port} backends={[b.label for b in backend_list]}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
