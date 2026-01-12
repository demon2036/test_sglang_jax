#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


@dataclass
class MockWeights:
    version: int = 0
    checksum: float = 0.0


def _compute_checksum(version: int) -> float:
    return float(version) * 0.1234567


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


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: Any, extra_headers: dict[str, str] | None = None) -> None:
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


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        if getattr(self.server, "quiet", False):
            return
        backend_id = getattr(self.server, "backend_id", "?")
        super().log_message(f"[backend={backend_id}] {fmt}", *args)

    def _require_api_key(self) -> bool:
        api_key = getattr(self.server, "api_key", "")
        if not api_key:
            return True
        auth = self.headers.get("authorization") or ""
        return auth == f"Bearer {api_key}"

    def do_GET(self) -> None:  # noqa: N802
        if not self._require_api_key():
            _write_json(self, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        path = urlparse(self.path).path
        if path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return

        if path == "/v1/models":
            _write_json(
                self,
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [{"id": getattr(self.server, "model_id", ""), "object": "model"}],
                },
                extra_headers={"X-Backend-Id": str(getattr(self.server, "backend_id", ""))},
            )
            return

        if path == "/admin/weights_checksum":
            weights_lock: threading.Lock = getattr(self.server, "weights_lock")
            weights: MockWeights = getattr(self.server, "weights")
            with weights_lock:
                checksum = weights.checksum
                version = weights.version
            _write_json(
                self,
                HTTPStatus.OK,
                {"ok": True, "version": version, "checksum": checksum},
                extra_headers={"X-Backend-Id": str(getattr(self.server, "backend_id", ""))},
            )
            return

        _write_json(self, HTTPStatus.NOT_FOUND, {"error": f"unknown path: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_api_key():
            _write_json(self, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        path = urlparse(self.path).path
        if path == "/v1/chat/completions":
            try:
                body = _read_json(self) or {}
            except ValueError as exc:
                _write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            messages = body.get("messages") or []
            user_content = ""
            if isinstance(messages, list) and messages:
                last = messages[-1] if isinstance(messages[-1], dict) else {}
                user_content = str(last.get("content") or "")

            weights_lock: threading.Lock = getattr(self.server, "weights_lock")
            weights: MockWeights = getattr(self.server, "weights")
            with weights_lock:
                version = weights.version

            response_id = f"mockcmpl-{uuid.uuid4().hex[:12]}"
            now = int(time.time())
            backend_id = getattr(self.server, "backend_id", "")
            model_id = getattr(self.server, "model_id", "")
            content = f"[backend={backend_id} version={version}] {user_content}".strip()
            payload = {
                "id": response_id,
                "object": "chat.completion",
                "created": now,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
            _write_json(
                self,
                HTTPStatus.OK,
                payload,
                extra_headers={
                    "X-Backend-Id": str(backend_id),
                    "X-Weights-Version": str(version),
                },
            )
            return

        if path == "/admin/perturb_weights":
            try:
                body = _read_json(self) or {}
            except ValueError as exc:
                _write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            delta = int(body.get("delta") or 1)
            weights_lock: threading.Lock = getattr(self.server, "weights_lock")
            weights: MockWeights = getattr(self.server, "weights")
            with weights_lock:
                weights.version += delta
                weights.checksum = _compute_checksum(weights.version)
                version = weights.version
                checksum = weights.checksum
            _write_json(
                self,
                HTTPStatus.OK,
                {"ok": True, "version": version, "checksum": checksum},
                extra_headers={"X-Backend-Id": str(getattr(self.server, "backend_id", ""))},
            )
            return

        if path == "/admin/reload_weights":
            try:
                body = _read_json(self) or {}
            except ValueError as exc:
                _write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            version = int(body.get("version") or 0)
            weights_lock: threading.Lock = getattr(self.server, "weights_lock")
            weights: MockWeights = getattr(self.server, "weights")
            with weights_lock:
                weights.version = version
                weights.checksum = _compute_checksum(weights.version)
                checksum = weights.checksum
            _write_json(
                self,
                HTTPStatus.OK,
                {"ok": True, "version": version, "checksum": checksum},
                extra_headers={"X-Backend-Id": str(getattr(self.server, "backend_id", ""))},
            )
            return

        _write_json(self, HTTPStatus.NOT_FOUND, {"error": f"unknown path: {path}"})


def make_server(
    *,
    host: str,
    port: int,
    backend_id: int,
    model_id: str,
    api_key: str = "",
    quiet: bool = False,
) -> ThreadingHTTPServer:
    weights = MockWeights(version=0, checksum=_compute_checksum(0))
    weights_lock = threading.Lock()

    server = ThreadingHTTPServer((host, port), _MockOpenAIHandler)
    server.backend_id = backend_id  # type: ignore[attr-defined]
    server.model_id = model_id  # type: ignore[attr-defined]
    server.weights = weights  # type: ignore[attr-defined]
    server.weights_lock = weights_lock  # type: ignore[attr-defined]
    server.api_key = api_key  # type: ignore[attr-defined]
    server.quiet = quiet  # type: ignore[attr-defined]
    return server


def start_server_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend-id", type=int, required=True)
    parser.add_argument("--model-id", type=str, default="mock-model")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    server = make_server(
        host=args.host,
        port=args.port,
        backend_id=args.backend_id,
        model_id=args.model_id,
        api_key=args.api_key,
        quiet=args.quiet,
    )
    print(f"READY mock_openai_server backend_id={args.backend_id} url=http://{args.host}:{args.port}", flush=True)
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
