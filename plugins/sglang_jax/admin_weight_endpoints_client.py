#!/usr/bin/env python3
"""Client for this repo's sglang-jax admin weight endpoints.

This talks to endpoints exposed by `plugins.sglang_jax.run_multi_openai_servers`
when started with `--enable-weight-reload`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:31000",
        help="Example: http://127.0.0.1:31000",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("checksum", help="GET /admin/weights_checksum")

    perturb = subparsers.add_parser("perturb", help="POST /admin/perturb_weights (demo)")
    perturb.add_argument("--seed", type=int, default=0)
    perturb.add_argument("--scale", type=float, default=1e-3)
    perturb.add_argument("--num-leaves", type=int, default=4)

    reload_weights = subparsers.add_parser("reload", help="POST /admin/reload_weights")
    reload_weights.add_argument("--model-path", type=str, required=True)
    reload_weights.add_argument("--revision", type=str, default="")
    reload_weights.add_argument(
        "--load-format",
        type=str,
        default="",
        help='Optional: "dummy" to force dummy weights.',
    )

    return parser.parse_args()


def _print_response(response) -> None:
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        print(json.dumps(response.json(), indent=2, ensure_ascii=False), flush=True)
        return
    print(response.text, flush=True)


def main() -> int:
    args = _parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        import requests
    except ImportError as exc:
        print(f"ERROR: requests is required: {exc}", file=sys.stderr, flush=True)
        return 2

    if args.command == "checksum":
        response = requests.get(f"{base_url}/admin/weights_checksum", timeout=60)
        _print_response(response)
        return 0 if response.ok else 1

    if args.command == "perturb":
        payload: dict[str, Any] = {
            "seed": args.seed,
            "scale": args.scale,
            "num_leaves": args.num_leaves,
        }
        response = requests.post(
            f"{base_url}/admin/perturb_weights",
            json=payload,
            timeout=600,
        )
        _print_response(response)
        return 0 if response.ok else 1

    if args.command == "reload":
        payload = {
            "model_path": args.model_path,
        }
        if args.revision:
            payload["revision"] = args.revision
        if args.load_format:
            payload["load_format"] = args.load_format

        response = requests.post(
            f"{base_url}/admin/reload_weights",
            json=payload,
            timeout=3600,
        )
        _print_response(response)
        return 0 if response.ok else 1

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

