#!/usr/bin/env python3

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one SGLang-JAX Engine on a single TPU device.")
    parser.add_argument(
        "--device-index",
        type=int,
        required=True,
        help="TPU device id (from jax.devices()) to use for this Engine.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face model id (default: Qwen/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (default: 1).")
    parser.add_argument(
        "--context-length",
        type=int,
        default=8192,
        help="Override model context length (default: 8192).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1,
        help="KV cache page_size (default: 1). Increase if backend asserts.",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.5,
        help="mem_fraction_static for Engine (default: 0.5).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generation max_new_tokens (default: 16).",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help="Sleep N seconds before shutdown (default: 0). Useful for concurrency tests.",
    )
    parser.add_argument(
        "--download-dir",
        default="/tmp",
        help="Hugging Face download dir for Engine (default: /tmp).",
    )
    args = parser.parse_args()

    try:
        from sgl_jax.srt.entrypoints.engine import Engine
        from sgl_jax.srt.hf_transformers_utils import get_tokenizer

        engine = Engine(
            model_path=args.model,
            trust_remote_code=True,
            tp_size=args.tp_size,
            device="tpu",
            device_indexes=[args.device_index],
            context_length=args.context_length,
            page_size=args.page_size,
            mem_fraction_static=args.mem_fraction,
            max_prefill_tokens=min(args.context_length, 16384),
            download_dir=args.download_dir,
            dtype="bfloat16",
            skip_server_warmup=True,
            disable_precompile=True,
            log_level="error",
        )
        try:
            tokenizer = get_tokenizer(args.model)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "用一句话介绍你自己。"}],
                add_generation_prompt=True,
                tokenize=False,
            )
            output = engine.generate(
                prompt=prompt,
                sampling_params={"max_new_tokens": args.max_new_tokens, "temperature": 0},
            )
            item = output[0] if isinstance(output, list) else output
            text = (item.get("text", "") or "").strip()
            if not text:
                raise RuntimeError("Empty model output")
            print(f"device_index={args.device_index} OK: {text[:160]!r}")
            if args.hold_seconds > 0:
                import time

                time.sleep(args.hold_seconds)
            return 0
        finally:
            engine.shutdown()
    except Exception as exc:
        print(f"FAILED device_index={args.device_index}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
