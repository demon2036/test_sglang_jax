from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPConnection

from plugins.openai_lb.mock_openai_server import make_server as make_mock_server
from plugins.openai_lb.mock_openai_server import start_server_in_thread as start_mock_thread
from plugins.openai_lb.openai_load_balancer import Backend
from plugins.openai_lb.openai_load_balancer import make_server as make_lb_server
from plugins.openai_lb.openai_load_balancer import start_server_in_thread as start_lb_thread


def _post_json(host: str, port: int, path: str, payload: dict, *, timeout: float = 5.0) -> tuple[int, dict[str, str], dict]:
    body = json.dumps(payload).encode("utf-8")
    conn = HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "Connection": "close",
            },
        )
        resp = conn.getresponse()
        raw = resp.read()
        headers = {k.lower(): v for k, v in resp.getheaders()}
        decoded = json.loads(raw.decode("utf-8")) if raw else {}
        return resp.status, headers, decoded
    finally:
        try:
            conn.close()
        except Exception:
            pass


class OpenAILoadBalancerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._servers: list = []
        self._threads: list[threading.Thread] = []

        backends: list[Backend] = []
        for backend_id in range(4):
            server = make_mock_server(
                host="127.0.0.1",
                port=0,
                backend_id=backend_id,
                model_id="mock-model",
                quiet=True,
            )
            thread = start_mock_thread(server)
            self._servers.append(server)
            self._threads.append(thread)
            host, port = server.server_address
            backends.append(Backend(host=host, port=port))

        lb_server = make_lb_server(
            listen_host="127.0.0.1",
            listen_port=0,
            backends=backends,
            quiet=True,
            backend_timeout_sec=1.0,
        )
        lb_thread = start_lb_thread(lb_server)
        self._servers.append(lb_server)
        self._threads.append(lb_thread)
        self.lb_host, self.lb_port = lb_server.server_address

    def tearDown(self) -> None:
        for server in self._servers[::-1]:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        for thread in self._threads[::-1]:
            try:
                thread.join(timeout=2)
            except Exception:
                pass

    def test_round_robin_distribution(self) -> None:
        backend_ids: list[int] = []
        for _ in range(8):
            status, headers, _ = _post_json(
                self.lb_host,
                self.lb_port,
                "/v1/chat/completions",
                {
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            self.assertEqual(status, 200)
            backend_ids.append(int(headers["x-backend-id"]))

        self.assertEqual(set(backend_ids[:4]), {0, 1, 2, 3})
        self.assertEqual(backend_ids[4:], backend_ids[:4])

    def test_failover_when_one_backend_down(self) -> None:
        down_backend_id = 1
        down_server = self._servers[down_backend_id]
        down_server.shutdown()
        down_server.server_close()

        seen: set[int] = set()
        for _ in range(12):
            status, headers, _ = _post_json(
                self.lb_host,
                self.lb_port,
                "/v1/chat/completions",
                {
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            self.assertEqual(status, 200)
            backend_id = int(headers["x-backend-id"])
            self.assertNotEqual(backend_id, down_backend_id)
            seen.add(backend_id)

        self.assertTrue(seen.issubset({0, 2, 3}))
        self.assertTrue(seen)

    def test_returns_502_when_all_backends_down(self) -> None:
        for server in self._servers[:-1]:
            server.shutdown()
            server.server_close()

        status, _, payload = _post_json(
            self.lb_host,
            self.lb_port,
            "/v1/chat/completions",
            {
                "model": "mock-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        self.assertEqual(status, 502)
        self.assertEqual(payload.get("error"), "all backends failed")

    def test_concurrent_requests(self) -> None:
        def send_one() -> int:
            status, headers, _ = _post_json(
                self.lb_host,
                self.lb_port,
                "/v1/chat/completions",
                {
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            if status != 200:
                raise RuntimeError(f"status={status}")
            return int(headers["x-backend-id"])

        backend_ids: list[int] = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_one) for _ in range(30)]
            for future in as_completed(futures):
                backend_ids.append(future.result())

        self.assertGreaterEqual(len(set(backend_ids)), 2)


if __name__ == "__main__":
    unittest.main()

