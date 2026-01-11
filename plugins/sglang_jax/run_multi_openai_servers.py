#!/usr/bin/env python3
"""Launch multiple OpenAI-compatible servers in one process and probe them concurrently."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SGLANG_JAX_PYTHON = PROJECT_ROOT / "sglang-jax" / "python"
if SGLANG_JAX_PYTHON.exists() and str(SGLANG_JAX_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_JAX_PYTHON))

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, Response
from sgl_jax.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
)
from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sgl_jax.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sgl_jax.srt.utils import add_api_key_middleware


@dataclass
class ServerThread:
    index: int
    port: int
    device_index: int
    thread: threading.Thread | None
    engine: Any | None
    server: Any | None


def _patch_zero_penalty_cache() -> None:
    """Make zero-penalty cache device-aware to avoid cross-device reuse."""
    from sgl_jax.srt.sampling import sampling_batch_info as sbi

    if getattr(sbi, "_sglang_jax_zero_penalty_cache_patched", False):
        return

    def _sharding_cache_key(sharding):
        if sharding is None:
            return ("none",)
        mesh = getattr(sharding, "mesh", None)
        spec = getattr(sharding, "spec", None)
        device_ids = None
        if mesh is not None:
            try:
                device_ids = tuple(int(d.id) for d in mesh.devices.reshape(-1))
            except Exception:
                try:
                    device_ids = tuple(int(d.id) for d in mesh.devices.flat)
                except Exception:
                    device_ids = None
        if device_ids is None:
            device_set = getattr(sharding, "device_set", None)
            if device_set is not None:
                try:
                    device_ids = tuple(sorted(int(d.id) for d in device_set))
                except Exception:
                    device_ids = None
        if device_ids is None:
            return ("sharding_id", id(sharding), str(spec))
        return ("devices", device_ids, str(spec))

    def _patched_get_or_create_zero_penalty_device(shape, sharding):
        key_shape = (int(shape[0]), int(shape[1]))
        key = (key_shape, _sharding_cache_key(sharding))
        with sbi._zero_linear_penalty_lock:
            cached = sbi._zero_linear_penalty_cache.get(key)
        if cached is not None:
            return cached
        zero_penalty = sbi.device_array(
            sbi.np.zeros(key_shape, dtype=sbi.np.float32),
            sharding=sharding,
        )
        with sbi._zero_linear_penalty_lock:
            existing = sbi._zero_linear_penalty_cache.get(key)
            if existing is None:
                sbi._zero_linear_penalty_cache[key] = zero_penalty
                return zero_penalty
            return existing

    sbi._get_or_create_zero_penalty_device = _patched_get_or_create_zero_penalty_device
    sbi._sglang_jax_zero_penalty_cache_patched = True


def _patch_engine_signal_handlers() -> None:
    """Allow Engine init from worker threads by skipping signal handlers there."""
    from sgl_jax.srt.entrypoints import engine as engine_mod

    if getattr(engine_mod, "_sglang_jax_engine_signal_patched", False):
        return

    def _patched_set_envs_and_config(server_args):
        import os
        import signal
        import threading
        import multiprocessing as mp

        engine_mod.set_ulimit()

        def sigchld_handler(signum, frame):
            pid, exitcode = os.waitpid(0, os.WNOHANG)
            if exitcode != 0:
                engine_mod.logger.warning(
                    "Child process unexpectedly failed with exitcode=%s. pid=%s",
                    exitcode,
                    pid,
                )
                engine_mod.logger.warning("Child process pid=%s frame=%s", pid, frame)

        def sigquit_handler(signum, frame):
            engine_mod.logger.error(
                "Received sigquit from a child process. It usually means the child failed."
            )
            engine_mod.kill_process_tree(os.getpid())

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGCHLD, sigchld_handler)
            signal.signal(signal.SIGQUIT, sigquit_handler)

        if not server_args.enable_single_process:
            mp.set_start_method("spawn", force=True)
        else:
            from multiprocessing import resource_tracker

            resource_tracker._resource_tracker._fd = -1

    engine_mod._set_envs_and_config = _patched_set_envs_and_config
    engine_mod._sglang_jax_engine_signal_patched = True


def _parse_int_list(value: str) -> list[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.environ.get("SGLANG_JAX_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM"))
    parser.add_argument("--num-servers", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=31000)
    parser.add_argument("--ports", type=str, default="", help="Comma-separated ports. Overrides --base-port if set.")
    parser.add_argument("--device-indexes", type=str, default="", help="Comma-separated device ids (default: first N local devices).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--load-format",
        type=str,
        default="dummy",
        choices=["auto", "safetensors", "pt", "dummy"],
        help="Use dummy to avoid downloading weights for this validation run.",
    )
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--max-total-tokens", type=int, default=128)
    parser.add_argument("--max-prefill-tokens", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--prompt", type=str, default="1+1=?")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--keep-running", action="store_true", help="Leave servers running after validation.")
    parser.add_argument("--force-exit", action="store_true", help="Force exit after validation to avoid hanging threads.")
    return parser.parse_args()


def _resolve_ports(args: argparse.Namespace) -> list[int]:
    if args.ports:
        ports = _parse_int_list(args.ports)
    else:
        ports = [args.base_port + i for i in range(args.num_servers)]
    if len(ports) != args.num_servers:
        raise ValueError(f"Expected {args.num_servers} ports, got {len(ports)}.")
    return ports


def _resolve_device_indexes(args: argparse.Namespace) -> list[int]:
    if args.device_indexes:
        device_indexes = _parse_int_list(args.device_indexes)
    else:
        try:
            import jax
        except ImportError as exc:
            raise RuntimeError("jax is required to auto-select device indexes.") from exc
        device_indexes = [int(dev.id) for dev in jax.local_devices()[: args.num_servers]]
    if len(device_indexes) < args.num_servers:
        raise ValueError(
            f"Need {args.num_servers} devices, but only got {len(device_indexes)}."
        )
    return device_indexes[: args.num_servers]


def _wait_for_server_ready(
    host: str, port: int, api_key: str, timeout_sec: int
) -> bool:
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests is required for readiness checks.")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout_sec
    last_error: str | None = None
    while time.time() < deadline:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return True
            last_error = f"status={response.status_code} body={response.text[:200]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    if last_error:
        print(f"[wait] port={port} not ready: {last_error}", flush=True)
    return False


def _send_chat_request(
    host: str,
    port: int,
    model: str,
    api_key: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> dict[str, Any]:
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests is required for validation requests.")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    url = f"http://{host}:{port}/v1/chat/completions"
    start = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    elapsed = time.time() - start
    return {
        "port": port,
        "status": response.status_code,
        "elapsed_sec": elapsed,
        "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
    }


def _build_fastapi_app(engine, api_key: str):
    app = FastAPI(openapi_url=None)
    if api_key:
        add_api_key_middleware(app, api_key)

    openai_completion = OpenAIServingCompletion(engine.tokenizer_manager, engine.template_manager)
    openai_chat = OpenAIServingChat(engine.tokenizer_manager, engine.template_manager)

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    @app.post("/v1/completions")
    async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
        return await openai_completion.handle_request(request, raw_request)

    @app.post("/v1/chat/completions")
    async def openai_v1_chat_completions(request: ChatCompletionRequest, raw_request: Request):
        return await openai_chat.handle_request(request, raw_request)

    @app.get("/v1/models", response_class=ORJSONResponse)
    async def available_models():
        served_model_name = engine.tokenizer_manager.served_model_name
        model_cards = [
            ModelCard(
                id=served_model_name,
                root=served_model_name,
                max_model_len=engine.tokenizer_manager.model_config.context_len,
            )
        ]
        return ModelList(data=model_cards)

    return app


def _run_server_thread(state: ServerThread, args: argparse.Namespace) -> None:
    from sgl_jax.srt.entrypoints.engine import Engine
    from sgl_jax.srt.server_args import ServerArgs
    import uvicorn

    server_args = ServerArgs(
        model_path=args.model,
        load_format=args.load_format,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        max_prefill_tokens=args.max_prefill_tokens,
        page_size=args.page_size,
        max_running_requests=args.max_running_requests,
        mem_fraction_static=args.mem_fraction_static,
        device_indexes=[state.device_index],
        tp_size=1,
        dp_size=1,
        host=args.host,
        port=state.port,
        enable_single_process=True,
        log_level="info",
    )

    engine = Engine(server_args=server_args)
    app = _build_fastapi_app(engine, args.api_key)

    config = uvicorn.Config(
        app,
        host=args.host,
        port=state.port,
        log_level="info",
        loop="uvloop",
    )
    server = uvicorn.Server(config)

    state.engine = engine
    state.server = server
    server.run()


def _shutdown_servers(states: list[ServerThread]) -> None:
    for state in states:
        if state.server is not None:
            state.server.should_exit = True
    for state in states:
        if state.thread is not None and state.thread.is_alive():
            state.thread.join(timeout=20)
    for state in states:
        if state.engine is not None:
            try:
                state.engine.shutdown()
            except Exception:
                pass


def main() -> int:
    args = _parse_args()

    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jit_cache_openai_multi")
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")

    _patch_zero_penalty_cache()
    _patch_engine_signal_handlers()

    ports = _resolve_ports(args)
    device_indexes = _resolve_device_indexes(args)

    states: list[ServerThread] = []
    for idx, (port, device_index) in enumerate(zip(ports, device_indexes, strict=True)):
        state = ServerThread(idx, port, device_index, None, None, None)
        thread = threading.Thread(
            target=_run_server_thread,
            args=(state, args),
            daemon=True,
        )
        state.thread = thread
        states.append(state)
        thread.start()

    try:
        for state in states:
            ready = _wait_for_server_ready(args.host, state.port, args.api_key, args.startup_timeout)
            if not ready:
                raise RuntimeError(f"Server on port {state.port} did not become ready.")

        print(
            f"READY: servers={len(states)} ports={ports} devices={device_indexes}",
            flush=True,
        )

        start = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=len(states)) as executor:
            futures = {
                executor.submit(
                    _send_chat_request,
                    args.host,
                    state.port,
                    args.model,
                    args.api_key,
                    args.prompt,
                    args.max_new_tokens,
                    args.temperature,
                    args.request_timeout,
                ): state
                for state in states
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = result["status"]
                elapsed = result["elapsed_sec"]
                print(f"RESP: port={result['port']} status={status} sec={elapsed:.3f}", flush=True)

        wall = time.time() - start
        max_req = max(res["elapsed_sec"] for res in results) if results else 0.0
        print(
            f"DONE: servers={len(states)} wall_sec={wall:.3f} max_req_sec={max_req:.3f}",
            flush=True,
        )
        print(json.dumps(results, indent=2)[:4000], flush=True)

        if args.force_exit:
            os._exit(0)

        if args.keep_running:
            print("KEEP_RUNNING: leaving servers alive.", flush=True)
            return 0
    finally:
        if not args.keep_running and not args.force_exit:
            _shutdown_servers(states)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
