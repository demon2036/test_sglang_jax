#!/usr/bin/env python3
"""Demo: update sglang-jax weights in-process by swapping model_state_leaves.

This validates the Tunix-style mechanism: overwrite `model_runner.model_state_leaves`
without restarting the Python process.

Example (TPU v5litepod-4, 1 device):
  cd /root/test_sglang_jax
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate sglang-jax
  export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_hot_update
  python -u -m plugins.sglang_jax.run_engine_hot_update_demo \
    --model hf-internal-testing/tiny-random-LlamaForCausalLM \
    --load-format dummy \
    --device-index 0
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SGLANG_JAX_PYTHON = PROJECT_ROOT / "sglang-jax" / "python"
if SGLANG_JAX_PYTHON.exists() and str(SGLANG_JAX_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_JAX_PYTHON))

from plugins.sglang_jax.weight_hot_update import (
    add_noise_to_model_state_leaves,
    async_reload_weights_from_path,
    async_flush_cache_best_effort,
    compute_model_state_checksum,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.environ.get("SGLANG_JAX_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM"))
    parser.add_argument("--load-format", type=str, default="dummy", choices=["auto", "safetensors", "pt", "dummy"])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--max-total-tokens", type=int, default=128)
    parser.add_argument("--max-prefill-tokens", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--prompt", type=str, default="1+1=?")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--run-generate",
        action="store_true",
        help="Call engine.generate() after hot update (can take minutes due to JIT compile).",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=1e-3)
    parser.add_argument("--noise-leaves", type=int, default=4, help="Only perturb the first N leaves to reduce work.")
    parser.add_argument("--reload-model-path", type=str, default="", help="If set, reload from this path before perturbing (local dir or HF repo id).")
    parser.add_argument("--reload-revision", type=str, default="", help="Optional HF revision for --reload-model-path.")
    return parser.parse_args()


def main() -> int:
    try:
        import jax
        from sgl_jax.srt.entrypoints.engine import Engine
    except ImportError as exc:
        print(f"ERROR: missing deps: {exc}", file=sys.stderr, flush=True)
        return 2

    args = _parse_args()
    device = None
    for dev in jax.local_devices():
        if int(dev.id) == int(args.device_index):
            device = dev
            break
    if device is None:
        print(f"ERROR: device_index={args.device_index} not in local_devices={[int(d.id) for d in jax.local_devices()]}", file=sys.stderr, flush=True)
        return 2

    print(
        {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [int(d.id) for d in jax.local_devices()],
            "target_device": int(device.id),
        },
        flush=True,
    )

    with jax.default_device(device):
        engine = Engine(
            model_path=args.model,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            device_indexes=[int(device.id)],
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

    if args.reload_model_path:
        t0 = time.time()
        result = asyncio.run(
            async_reload_weights_from_path(
                engine,
                args.reload_model_path,
                revision=(args.reload_revision or None),
                load_format=args.load_format,
            )
        )
        print(
            {
                "reload": {
                    "elapsed_sec": time.time() - t0,
                    "resolved_path": result.model_path,
                    "leaf_count": result.leaf_count,
                }
            },
            flush=True,
        )

    checksum_before = compute_model_state_checksum(engine)
    print({"checksum_before": checksum_before}, flush=True)

    num_perturbed = add_noise_to_model_state_leaves(
        engine,
        seed=args.noise_seed,
        scale=args.noise_scale,
        num_leaves=args.noise_leaves,
    )
    asyncio.run(async_flush_cache_best_effort(engine))
    checksum_after = compute_model_state_checksum(engine)
    print(
        {
            "perturb": {
                "num_leaves": num_perturbed,
                "noise_seed": args.noise_seed,
                "noise_scale": args.noise_scale,
            },
            "checksum_after": checksum_after,
            "checksum_delta": checksum_after - checksum_before,
        },
        flush=True,
    )

    if args.run_generate:
        try:
            sampling_params = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature}
            out = engine.generate(prompt=args.prompt, sampling_params=sampling_params)
            item = out[0] if isinstance(out, list) else out
            print({"generate_text": item.get("text", "")}, flush=True)
        except Exception as exc:
            print(f"WARN: generate failed: {exc}", file=sys.stderr, flush=True)
    else:
        print({"generate_text": None, "skipped": True}, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
