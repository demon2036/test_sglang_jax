#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from http.client import HTTPConnection


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    args: list[str]
    url: str


def _wait_ready(host: str, port: int, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_error: str | None = None
    while time.time() < deadline:
        try:
            conn = HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health", headers={"Connection": "close"})
            resp = conn.getresponse()
            resp.read()
            if resp.status == 200:
                return
            last_error = f"status={resp.status}"
        except Exception as exc:
            last_error = str(exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        time.sleep(0.2)
    raise RuntimeError(f"Server {host}:{port} not ready: {last_error}")


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Default bind host (used if --backend-host/--lb-host unset).")
    parser.add_argument("--backend-host", type=str, default="", help="Bind host for backend servers (default: --host).")
    parser.add_argument("--lb-host", type=str, default="", help="Bind host for the load balancer (default: --host).")
    parser.add_argument("--num-backends", type=int, default=4)
    parser.add_argument("--backend-base-port", type=int, default=0, help="0 = auto-pick free ports.")
    parser.add_argument("--lb-port", type=int, default=0, help="0 = auto-pick free port.")
    parser.add_argument("--model-id", type=str, default="mock-model")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--startup-timeout-sec", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    backend_bind_host = args.backend_host or args.host
    lb_bind_host = args.lb_host or args.host
    backend_connect_host = "127.0.0.1" if backend_bind_host in ("0.0.0.0", "::") else backend_bind_host
    lb_connect_host = "127.0.0.1" if lb_bind_host in ("0.0.0.0", "::") else lb_bind_host

    if args.backend_base_port:
        backend_ports = [args.backend_base_port + i for i in range(args.num_backends)]
    else:
        backend_ports = [_find_free_port(backend_bind_host) for _ in range(args.num_backends)]

    lb_port = args.lb_port or _find_free_port(lb_bind_host)
    backend_urls = [f"http://{backend_connect_host}:{p}" for p in backend_ports]

    specs: list[ProcessSpec] = []
    for backend_id, port in enumerate(backend_ports):
        specs.append(
            ProcessSpec(
                name=f"backend-{backend_id}",
                args=[
                    sys.executable,
                    "-m",
                    "plugins.openai_lb.mock_openai_server",
                    "--host",
                    backend_bind_host,
                    "--port",
                    str(port),
                    "--backend-id",
                    str(backend_id),
                    "--model-id",
                    args.model_id,
                    "--api-key",
                    args.api_key,
                ]
                + (["--quiet"] if args.quiet else []),
                url=f"http://{args.host}:{port}",
            )
        )

    specs.append(
        ProcessSpec(
            name="lb",
            args=[
                sys.executable,
                "-m",
                "plugins.openai_lb.openai_load_balancer",
                "--listen-host",
                lb_bind_host,
                "--listen-port",
                str(lb_port),
                "--backends",
                ",".join(backend_urls),
                "--api-key",
                args.api_key,
                "--backend-timeout-sec",
                "60",
            ]
            + (["--quiet"] if args.quiet else []),
            url=f"http://{lb_connect_host}:{lb_port}",
        )
    )

    procs: list[subprocess.Popen[str]] = []
    try:
        for spec in specs:
            proc = subprocess.Popen(spec.args, stdout=None, stderr=None, text=True)
            procs.append(proc)

        for port in backend_ports:
            _wait_ready(backend_connect_host, port, args.startup_timeout_sec)
        _wait_ready(lb_connect_host, lb_port, args.startup_timeout_sec)

        print(f"READY cluster lb={specs[-1].url} backends={backend_urls}", flush=True)
        print("Example request:", flush=True)
        print(
            f"  curl -s {specs[-1].url}/v1/chat/completions -H 'Content-Type: application/json' "
            f"-d '{{\"model\":\"{args.model_id}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'",
            flush=True,
        )
        print("Ctrl-C to stop.", flush=True)

        while True:
            time.sleep(1)
            for proc in procs:
                code = proc.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f"Process exited: pid={proc.pid} code={code}")
    except KeyboardInterrupt:
        return 0
    finally:
        for proc in procs[::-1]:
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in procs[::-1]:
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
