from __future__ import annotations

import json
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from plugins.tpu_pod.pod_commander import PodCommander
from plugins.tpu_pod.worker_agent import make_server as make_agent_server


def _get_json(host: str, port: int, path: str, *, timeout: float = 5.0) -> dict:
    conn = HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"GET {path} -> {resp.status}: {raw[:200]!r}")
        return json.loads(raw.decode("utf-8")) if raw else {}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _post_json(host: str, port: int, path: str, payload: dict, *, timeout: float = 5.0) -> tuple[int, dict]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
        decoded = json.loads(raw.decode("utf-8")) if raw else {}
        return resp.status, decoded
    finally:
        try:
            conn.close()
        except Exception:
            pass


class TpuPodOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._servers: list = []
        self._threads: list[threading.Thread] = []
        self._commander: PodCommander | None = None

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self._tmp_path = Path(tmp.name)

        agent_server = make_agent_server(host="127.0.0.1", port=0, workdir=self._tmp_path, quiet=True)
        agent_thread = threading.Thread(target=agent_server.serve_forever, daemon=True)
        agent_thread.start()
        self._servers.append(agent_server)
        self._threads.append(agent_thread)
        agent_host, agent_port = agent_server.server_address
        self._agent_url = f"http://{agent_host}:{agent_port}"

    def tearDown(self) -> None:
        if self._commander is not None:
            try:
                self._commander.stop()
            except Exception:
                pass
            self._commander = None

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

    def test_commander_can_start_remote_mock_backends_and_publish_registry(self) -> None:
        commander = PodCommander(
            workdir=self._tmp_path,
            backend_kind="mock-openai",
            model="mock-model",
            num_servers_per_worker=4,
            base_port=0,
            bind_host="127.0.0.1",
            load_format="auto",
            enable_weight_reload=False,
            python_exe="python",
            remote_agents=[self._agent_url],
            startup_timeout_sec=10.0,
            registry_host="127.0.0.1",
            registry_port=0,
            advertise_host="127.0.0.1",
            quiet=True,
        )
        commander.start()
        self._commander = commander

        registry_url = commander.registry_url
        self.assertIsNotNone(registry_url)
        parsed = urlparse(str(registry_url))
        registry_host = parsed.hostname or "127.0.0.1"
        registry_port = int(parsed.port or 80)

        payload = _get_json(registry_host, registry_port, "/backends")
        self.assertTrue(payload.get("ok"))
        backends = payload.get("backends") or []
        self.assertEqual(len(backends), 8)

        for backend in backends:
            host, port_str = backend.rsplit(":", 1)
            status, resp = _post_json(
                host,
                int(port_str),
                "/v1/chat/completions",
                {
                    "model": payload.get("model") or "mock-model",
                    "messages": [{"role": "user", "content": "你是谁"}],
                    "max_tokens": 8,
                },
                timeout=5.0,
            )
            self.assertEqual(status, 200)
            content = (((resp.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
            self.assertIn("你是谁", content)


if __name__ == "__main__":
    unittest.main()
