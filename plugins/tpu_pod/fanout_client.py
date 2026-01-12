#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.client import HTTPConnection
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class Backend:
    host: str
    port: int

    @property
    def label(self) -> str:
        return f"{self.host}:{self.port}"


def _get_json(url: str, path: str, *, timeout_sec: float) -> Any:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 80)
    conn = HTTPConnection(host, port, timeout=timeout_sec)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"GET {path} -> {resp.status}: {raw[:200]!r}")
        return json.loads(raw.decode("utf-8")) if raw else None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _post_json(url: str, path: str, payload: dict[str, Any], *, timeout_sec: float) -> Any:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 80)
    conn = HTTPConnection(host, port, timeout=timeout_sec)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}
    try:
        conn.request("POST", path, body=data, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        return {
            "status": int(resp.status),
            "body": json.loads(raw.decode("utf-8")) if raw else None,
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _parse_backends(items: list[str]) -> list[Backend]:
    backends: list[Backend] = []
    for item in items:
        if "://" in item:
            parsed = urlparse(item)
            backends.append(Backend(host=parsed.hostname or "127.0.0.1", port=int(parsed.port or 80)))
        else:
            host, port_str = item.rsplit(":", 1)
            backends.append(Backend(host=host, port=int(port_str)))
    return backends


def _extract_content(resp_body: Any) -> str:
    if not isinstance(resp_body, dict):
        return str(resp_body)
    choices = resp_body.get("choices")
    if isinstance(choices, list) and choices:
        message = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    return json.dumps(resp_body, ensure_ascii=False)[:400]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-url", type=str, required=True, help="e.g. http://10.0.0.1:9100")
    parser.add_argument("--prompt", type=str, default="你是谁")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    registry = _get_json(args.registry_url, "/backends", timeout_sec=10.0) or {}
    backends = _parse_backends(list(registry.get("backends") or []))
    model = str(args.model).strip() or str(registry.get("model") or "")
    if not model:
        raise RuntimeError("model is missing (pass --model or ensure registry provides one)")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": str(args.prompt)}],
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }

    start = time.time()
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {
            executor.submit(
                _post_json,
                f"http://{b.host}:{b.port}",
                "/v1/chat/completions",
                payload,
                timeout_sec=float(args.timeout_sec),
            ): b
            for b in backends
        }
        for future in as_completed(futures):
            backend = futures[future]
            resp = future.result()
            content = _extract_content(resp.get("body"))
            print(f"RESP backend={backend.label} status={resp['status']} content={content}", flush=True)

    wall = time.time() - start
    print(f"DONE backends={len(backends)} wall_sec={wall:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

