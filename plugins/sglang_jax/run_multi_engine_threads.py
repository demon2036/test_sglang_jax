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
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SGLANG_JAX_PYTHON = PROJECT_ROOT / "sglang-jax" / "python"
if SGLANG_JAX_PYTHON.exists() and str(SGLANG_JAX_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_JAX_PYTHON))


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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jit_cache_multi")
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")

    import jax
    from sgl_jax.srt.entrypoints.engine import Engine

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

    engines: list[Engine] = []
    try:
        for i, device_id in enumerate(device_indexes):
            print(f"[engine {i}] init device_indexes={[device_id]}", flush=True)
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
                f"[engine {i}] OK mesh_device_ids={mesh_device_ids} server_args.device_indexes={engine.server_args.device_indexes}",
                flush=True,
            )
            engines.append(engine)

        sampling_params = {"max_new_tokens": args.max_new_tokens, "temperature": 0}

        def _run_one(engine_index: int):
            t0 = time.perf_counter()
            out = engines[engine_index].generate(prompt=args.prompt, sampling_params=sampling_params)
            dt = time.perf_counter() - t0
            item = out[0] if isinstance(out, list) else out
            text = item.get("text", "")
            return engine_index, dt, text

        concurrency = max(1, min(args.concurrency, len(engines)))
        print(f"Running concurrent generate: {concurrency=}", flush=True)
        t0 = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_run_one, i) for i in range(len(engines))]
            for fut in as_completed(futs):
                results.append(fut.result())
        wall = time.perf_counter() - t0

        for engine_index, dt, text in sorted(results, key=lambda x: x[0]):
            text = text.replace("\n", "\\n")
            print(f"[engine {engine_index}] generate_sec={dt:.3f} text={text[:160]}", flush=True)
        print(f"DONE: engines={len(engines)} wall_sec={wall:.3f}", flush=True)
        return 0
    finally:
        for engine in engines:
            try:
                engine.shutdown()
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"engine.shutdown failed: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

