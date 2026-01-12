from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ReloadResult:
    model_path: str
    elapsed_sec: float
    leaf_count: int


def get_model_runner(engine: Any) -> Any:
    scheduler_info = getattr(engine, "scheduler_info", None)
    if not isinstance(scheduler_info, dict):
        raise RuntimeError("engine.scheduler_info is missing or not a dict")

    scheduler = scheduler_info.get("scheduler")
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

    raise RuntimeError("Unable to locate model_runner on scheduler.tp_worker")


async def async_pause_generation_best_effort(engine: Any) -> None:
    pause_fn = getattr(engine, "async_pause_generation", None)
    if pause_fn is None:
        pause_fn = getattr(engine, "pause_generation", None)
    if pause_fn is None:
        return
    try:
        result = pause_fn(mode="abort")
    except TypeError:
        result = pause_fn()
    if asyncio.iscoroutine(result):
        await result


async def async_flush_cache_best_effort(engine: Any) -> None:
    flush_fn = getattr(engine, "async_flush_cache", None)
    if flush_fn is None:
        flush_fn = getattr(engine, "flush_cache", None)
    if flush_fn is None:
        return
    result = flush_fn()
    if asyncio.iscoroutine(result):
        await result


def _maybe_set_dummy_mode(model_config: Any, load_format: str | None) -> None:
    if not load_format:
        return

    if load_format == "dummy":
        setattr(model_config, "_dummy_mode", True)
        return

    if hasattr(model_config, "_dummy_mode"):
        setattr(model_config, "_dummy_mode", False)


def refresh_model_state_leaves(model_runner: Any) -> int:
    try:
        import jax
        from flax import nnx
    except ImportError as exc:
        raise RuntimeError("jax+flax are required to refresh model_state_leaves") from exc

    _, model_state = nnx.split(model_runner.model)
    model_state_leaves, _ = jax.tree_util.tree_flatten(model_state)
    model_runner.model_state_leaves = model_state_leaves
    return len(model_state_leaves)


def hot_swap_model_state_leaves(
    engine: Any,
    new_model_state_leaves: Sequence[Any],
    *,
    validate_length: bool = True,
) -> int:
    model_runner = get_model_runner(engine)
    if validate_length:
        old_leaves = getattr(model_runner, "model_state_leaves", None)
        if old_leaves is not None and len(new_model_state_leaves) != len(old_leaves):
            raise ValueError(
                f"new_model_state_leaves has len={len(new_model_state_leaves)} but expected len={len(old_leaves)}"
            )

    model_runner.model_state_leaves = list(new_model_state_leaves)
    return len(new_model_state_leaves)


def _resolve_model_path(model_path: str, revision: str | None) -> str:
    if os.path.isdir(model_path):
        return model_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for non-local model_path") from exc

    return snapshot_download(
        repo_id=model_path,
        revision=revision,
        local_dir_use_symlinks=False,
        cache_dir=os.environ.get("HF_HOME") or None,
        resume_download=True,
    )


async def async_reload_weights_from_path(
    engine: Any,
    model_path: str,
    *,
    revision: str | None = None,
    load_format: str | None = None,
    pause_generation: bool = True,
    flush_cache: bool = True,
) -> ReloadResult:
    if pause_generation:
        await async_pause_generation_best_effort(engine)
    if flush_cache:
        await async_flush_cache_best_effort(engine)

    start = time.time()
    resolved_path = _resolve_model_path(model_path, revision)

    model_runner = get_model_runner(engine)
    model_config = model_runner.model_config
    model_config.model_path = resolved_path
    if revision is not None:
        model_config.revision = revision
    _maybe_set_dummy_mode(model_config, load_format)

    try:
        import jax
    except ImportError as exc:
        raise RuntimeError("jax is required to reload weights") from exc

    with jax.set_mesh(model_runner.mesh):
        model_runner.model.load_weights(model_config)

    leaf_count = refresh_model_state_leaves(model_runner)

    if flush_cache:
        await async_flush_cache_best_effort(engine)

    elapsed = time.time() - start
    return ReloadResult(model_path=resolved_path, elapsed_sec=elapsed, leaf_count=leaf_count)


def compute_model_state_checksum(
    engine: Any,
    *,
    num_leaves: int = 4,
    num_elems_per_leaf: int = 1024,
) -> float:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError("jax is required to compute checksum") from exc

    model_runner = get_model_runner(engine)
    leaves: list[Any] = list(getattr(model_runner, "model_state_leaves", []))
    if not leaves:
        raise RuntimeError("model_runner.model_state_leaves is empty")

    num_leaves = max(1, min(num_leaves, len(leaves)))
    checksum: Any = jnp.array(0.0, dtype=jnp.float32)
    for leaf in leaves[:num_leaves]:
        if not isinstance(leaf, jax.Array):
            continue
        flat = jnp.ravel(leaf)
        if num_elems_per_leaf > 0:
            flat = flat[:num_elems_per_leaf]
        checksum = checksum + jnp.sum(flat.astype(jnp.float32))

    return float(jax.device_get(checksum))


def add_noise_to_model_state_leaves(
    engine: Any,
    *,
    seed: int = 0,
    scale: float = 1e-3,
    num_leaves: int | None = None,
) -> int:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError("jax is required to perturb weights") from exc

    model_runner = get_model_runner(engine)
    leaves: list[Any] = list(getattr(model_runner, "model_state_leaves", []))
    if not leaves:
        raise RuntimeError("model_runner.model_state_leaves is empty")

    if num_leaves is None:
        num_leaves = len(leaves)
    num_leaves = max(1, min(num_leaves, len(leaves)))

    base_key = jax.random.key(seed)
    new_leaves: list[Any] = []
    for index, leaf in enumerate(leaves):
        if index >= num_leaves or not isinstance(leaf, jax.Array):
            new_leaves.append(leaf)
            continue

        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            new_leaves.append(leaf)
            continue

        leaf_key = jax.random.fold_in(base_key, index)
        scale_value = jnp.asarray(scale, dtype=leaf.dtype)
        noise = jax.random.normal(leaf_key, leaf.shape, dtype=leaf.dtype) * scale_value
        new_leaves.append(leaf + noise)

    model_runner.model_state_leaves = new_leaves
    return num_leaves
