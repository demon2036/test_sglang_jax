#!/usr/bin/env python3
"""Smoke-test running multiple sglang-jax Engines in one process via threads.

Goal: validate whether we can create multiple `Engine(enable_single_process=True)`
instances in a *single* Python process, each pinned to a different TPU device
via `device_indexes=[...]`, and issue concurrent `generate()` calls.

Example (TPU v4-8, 4 local devices):
  cd /root/test_sglang_jax
  JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_multi \
    python -u -m plugins.sglang_jax.run_multi_engine_threads \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --num-engines 4 \
      --load-format dummy
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SGLANG_JAX_PYTHON = PROJECT_ROOT / "sglang-jax" / "python"
if SGLANG_JAX_PYTHON.exists() and str(SGLANG_JAX_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_JAX_PYTHON))


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
    parser.add_argument("--model", type=str, default=os.environ.get("SGLANG_JAX_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--num-engines", type=int, default=4)
    parser.add_argument(
        "--device-indexes",
        type=str,
        default="",
        help="Comma-separated device ids to use (default: first N local device ids).",
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default="dummy",
        choices=["auto", "safetensors", "pt", "dummy"],
        help="Use dummy to avoid downloading weights for this smoke test.",
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-prefill-tokens", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--prompt", type=str, default="Please answer: 1+1=?")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4, help="How many concurrent generate calls.")
    parser.add_argument(
        "--warmup-sequential",
        action="store_true",
        help="Run one generate per engine sequentially before concurrent run.",
    )
    parser.add_argument(
        "--engine-per-thread",
        action="store_true",
        help="Create each Engine inside its own worker thread.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jit_cache_multi")
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")

    import jax
    from flax import nnx
    from sgl_jax.srt.entrypoints.engine import Engine

    _patch_zero_penalty_cache()
    _patch_engine_signal_handlers()

    local_devices = list(jax.local_devices())
    if not local_devices:
        raise RuntimeError("No local JAX devices found.")

    if args.device_indexes:
        device_indexes = _parse_int_list(args.device_indexes)
    else:
        device_indexes = [d.id for d in local_devices[: args.num_engines]]

    if len(device_indexes) != args.num_engines:
        raise ValueError(f"Need {args.num_engines} device ids, got {len(device_indexes)}: {device_indexes}")

    if len(set(device_indexes)) != len(device_indexes):
        raise ValueError(f"Duplicate device ids: {device_indexes}")

    print(
        "JAX:",
        {
            "backend": jax.default_backend(),
            "process_index": jax.process_index(),
            "process_count": jax.process_count(),
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "local_device_ids": [d.id for d in local_devices],
        },
        flush=True,
    )
    if jax.process_count() != 1:
        print(
            f"ERROR: This smoke test must run on a single host (process_count==1). Got process_count={jax.process_count()}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    devices_by_id = {d.id: d for d in jax.devices()}
    try:
        def _get_model_runner(engine: Engine):
            scheduler = engine.scheduler_info.get("scheduler")
            if scheduler is None:
                raise RuntimeError("engine.scheduler_info['scheduler'] is missing")
            tp_worker = getattr(scheduler, "tp_worker", None)
            if tp_worker is None:
                raise RuntimeError("scheduler.tp_worker is missing")
            worker = getattr(tp_worker, "worker", None)
            if worker is not None and hasattr(worker, "model_runner"):
                return worker.model_runner
            if hasattr(tp_worker, "model_runner"):
                return tp_worker.model_runner
            raise RuntimeError("Unable to locate model_runner on tp_worker")

        def _fix_sampler_state_to_engine_device(engine: Engine):
            model_runner = _get_model_runner(engine)
            target_device = model_runner.mesh.devices.reshape(-1)[0]
            sampler_def, sampler_state = nnx.split(model_runner.sampler)
            sampler_state = jax.tree.map(
                lambda x: jax.device_put(x, target_device) if isinstance(x, jax.Array) else x,
                sampler_state,
            )
            model_runner.sampler = nnx.merge(sampler_def, sampler_state)
            model_runner.initialize_jit()

        def _build_engine(engine_index: int, device_id: int) -> Engine:
            print(f"[engine {engine_index}] init device_indexes={[device_id]}", flush=True)
            default_device = devices_by_id.get(device_id)
            if default_device is None:
                raise ValueError(f"Unknown device id {device_id}. Known: {sorted(devices_by_id)}")
            with jax.default_device(default_device):
                engine = Engine(
                    model_path=args.model,
                    trust_remote_code=True,
                    tp_size=1,
                    device="tpu",
                    device_indexes=[device_id],
                    enable_single_process=True,
                    disable_overlap_schedule=True,
                    disable_precompile=True,
                    load_format=args.load_format,
                    context_length=args.context_length,
                    max_total_tokens=args.max_total_tokens,
                    max_prefill_tokens=args.max_prefill_tokens,
                    page_size=args.page_size,
                    max_running_requests=args.max_running_requests,
                    mem_fraction_static=args.mem_fraction_static,
                    download_dir="/tmp",
                    dtype="bfloat16",
                    skip_server_warmup=True,
                )
            scheduler = engine.scheduler_info.get("scheduler")
            mesh = getattr(scheduler, "mesh", None)
            mesh_device_ids = None
            if mesh is not None:
                mesh_device_ids = mesh.device_ids.flatten().tolist()
            print(
                f"[engine {engine_index}] OK mesh_device_ids={mesh_device_ids} server_args.device_indexes={engine.server_args.device_indexes}",
                flush=True,
            )
            _fix_sampler_state_to_engine_device(engine)
            return engine

        sampling_params = {"max_new_tokens": args.max_new_tokens, "temperature": 0}

        def _run_one(engine_index: int, engine: Engine, prompt: str, params: dict):
            t0 = time.perf_counter()
            # Ensure per-engine ops/materialized arrays land on the intended device.
            target_device = _get_model_runner(engine).mesh.devices.reshape(-1)[0]
            with jax.default_device(target_device):
                out = engine.generate(prompt=prompt, sampling_params=params)
            dt = time.perf_counter() - t0
            item = out[0] if isinstance(out, list) else out
            text = item.get("text", "")
            return engine_index, dt, text

        class EngineWorker:
            def __init__(self, engine_index: int, device_id: int):
                self.engine_index = engine_index
                self.device_id = device_id
                self._ready = threading.Event()
                self._error: Exception | None = None
                self._engine: Engine | None = None
                self._tasks: queue.Queue[tuple[str, dict, Future] | None] = queue.Queue()
                self._thread = threading.Thread(
                    target=self._run, name=f"engine-worker-{engine_index}", daemon=True
                )

            def start(self):
                self._thread.start()
                self._ready.wait()
                if self._error is not None:
                    raise self._error

            def submit(self, prompt: str, params: dict) -> Future:
                fut: Future = Future()
                self._tasks.put((prompt, params, fut))
                return fut

            def shutdown(self):
                self._tasks.put(None)
                self._thread.join(timeout=30)

            def _run(self):
                try:
                    engine = _build_engine(self.engine_index, self.device_id)
                    self._engine = engine
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    self._error = exc
                    self._ready.set()
                    return
                self._ready.set()
                while True:
                    task = self._tasks.get()
                    if task is None:
                        break
                    prompt, params, fut = task
                    try:
                        fut.set_result(_run_one(self.engine_index, engine, prompt, params))
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        fut.set_exception(exc)
                try:
                    engine.shutdown()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(f"[engine {self.engine_index}] shutdown failed: {exc}", file=sys.stderr, flush=True)

        use_workers = args.engine_per_thread or args.concurrency > 1
        if use_workers:
            concurrency = max(1, min(args.concurrency, len(device_indexes)))
            print(f"Running concurrent generate (engine-workers): {concurrency=}", flush=True)
            workers = [EngineWorker(i, device_indexes[i]) for i in range(len(device_indexes))]
            try:
                for worker in workers:
                    worker.start()
                if args.warmup_sequential:
                    print("Warmup: sequential generate", flush=True)
                    for worker in workers:
                        worker.submit(args.prompt, sampling_params).result()
                t0 = time.perf_counter()
                results = []
                active_workers = workers[:concurrency]
                futs = [w.submit(args.prompt, sampling_params) for w in active_workers]
                for fut in as_completed(futs):
                    results.append(fut.result())
                wall = time.perf_counter() - t0
                for engine_index, dt, text in sorted(results, key=lambda x: x[0]):
                    text = text.replace("\n", "\\n")
                    print(f"[engine {engine_index}] generate_sec={dt:.3f} text={text[:160]}", flush=True)
                print(f"DONE: engines={len(active_workers)} wall_sec={wall:.3f}", flush=True)
                return 0
            finally:
                for worker in workers:
                    worker.shutdown()

        engines: list[Engine] = []
        for i, device_id in enumerate(device_indexes):
            engines.append(_build_engine(i, device_id))

        concurrency = max(1, min(args.concurrency, len(engines)))
        if args.warmup_sequential:
            print("Warmup: sequential generate", flush=True)
            for i in range(len(engines)):
                _run_one(i, engines[i], args.prompt, sampling_params)
        print(f"Running concurrent generate: {concurrency=}", flush=True)
        t0 = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [
                ex.submit(_run_one, i, engines[i], args.prompt, sampling_params)
                for i in range(len(engines))
            ]
            for fut in as_completed(futs):
                results.append(fut.result())
        wall = time.perf_counter() - t0

        for engine_index, dt, text in sorted(results, key=lambda x: x[0]):
            text = text.replace("\n", "\\n")
            print(f"[engine {engine_index}] generate_sec={dt:.3f} text={text[:160]}", flush=True)
        print(f"DONE: engines={len(engines)} wall_sec={wall:.3f}", flush=True)
        return 0
    finally:
        if "engines" in locals():
            for engine in engines:
                try:
                    engine.shutdown()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"engine.shutdown failed: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
