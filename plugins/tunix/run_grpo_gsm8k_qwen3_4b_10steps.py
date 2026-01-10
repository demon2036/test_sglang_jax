#!/usr/bin/env python3
"""Smoke-run Tunix GRPO (GSM8K) with sglang-jax rollout using Qwen3-4B.

This is intended to validate wiring + basic training loop, not to benchmark.

Example:
  JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache \
    python -u plugins/tunix/run_grpo_gsm8k_qwen3_4b_10steps.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _add_repo_deps_to_sys_path():
    project_root = Path(__file__).resolve().parents[2]

    tunix_root = project_root / "tunix"
    if tunix_root.exists() and str(tunix_root) not in sys.path:
        sys.path.insert(0, str(tunix_root))

    sglang_jax_python = project_root / "sglang-jax" / "python"
    if sglang_jax_python.exists() and str(sglang_jax_python) not in sys.path:
        sys.path.insert(0, str(sglang_jax_python))


def _download_hf_repo_snapshot(repo_id: str, local_dir: Path) -> Path:
    import huggingface_hub

    local_dir.mkdir(parents=True, exist_ok=True)
    huggingface_hub.snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_dir


def _make_fsdp_tp_mesh(devices, fsdp: int, tp: int):
    import jax

    if fsdp * tp != len(devices):
        raise ValueError(f"mesh size mismatch: {fsdp=} * {tp=} != {len(devices)=}")

    return jax.make_mesh(
        (fsdp, tp),
        ("fsdp", "tp"),
        devices=devices,
        axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )


def _split_devices_for_rollout(all_devices, rollout_n: int):
    """Split devices into (rollout_devices, train_devices).

    On multi-host TPU, JAX requires running on all hosts. To avoid hanging due
    to a mesh that only includes devices from a subset of hosts, we pick devices
    for rollout in a round-robin over `process_index` when possible.
    """
    by_process: dict[int, list] = {}
    for device in all_devices:
        by_process.setdefault(device.process_index, []).append(device)

    process_indices = sorted(by_process)
    for devices in by_process.values():
        devices.sort(key=lambda d: d.id)

    rollout_devices = []
    while len(rollout_devices) < rollout_n:
        made_progress = False
        for process_index in process_indices:
            devices = by_process[process_index]
            if devices:
                rollout_devices.append(devices.pop(0))
                made_progress = True
                if len(rollout_devices) == rollout_n:
                    break
        if not made_progress:
            break

    train_devices = []
    for process_index in process_indices:
        train_devices.extend(by_process[process_index])

    if not train_devices:
        train_devices = list(all_devices)

    return rollout_devices, train_devices


def _patch_tunix_sglang_jax_engine_args(
    *,
    disable_precompile: bool,
    max_total_tokens: int,
    max_prefill_tokens: int,
    max_running_requests: int,
):
    """Monkeypatch Tunix's SglangJaxSampler to add extra Engine args.

    Tunix's OSS SglangJaxSampler currently doesn't expose these knobs, but they
    are important to avoid huge TPU precompile shapes for small models.
    """
    from tunix.generate import sglang_jax_sampler as sglang_jax_sampler_lib

    original_fn = sglang_jax_sampler_lib.SglangJaxSampler._sglang_jax_config

    def patched_fn(self, config):
        engine_args = original_fn(self, config)
        engine_args["disable_precompile"] = disable_precompile
        engine_args["max_total_tokens"] = max_total_tokens
        engine_args["max_prefill_tokens"] = max_prefill_tokens
        engine_args["max_running_requests"] = max_running_requests
        return engine_args

    sglang_jax_sampler_lib.SglangJaxSampler._sglang_jax_config = patched_fn


def _patch_tunix_shard_input_for_multihost():
    """Monkeypatch Tunix's shard_input to support multi-controller JAX.

    Tunix's `tunix.sft.sharding_utils.shard_input` uses
    `jax.make_array_from_process_local_data(...)` for all leaves. In multi-host
    (multi-controller) JAX, some leaves can be global `jax.Array`s that are not
    fully addressable, and feeding them back into `make_array_from_process_local_data`
    triggers `device_put` addressability errors.

    This patch keeps the original fast path for host-local numpy inputs, but
    reshard global `jax.Array` leaves via `jax.device_put(...)`.
    """
    from tunix.sft import sharding_utils as sharding_utils_lib

    import jax
    from jax.interpreters import pxla
    import jax.sharding as shd

    original_fn = sharding_utils_lib.shard_input

    def patched_fn(input_data, data_sharding_axis):
        mesh = pxla.thread_resources.env.physical_mesh
        if mesh.empty:
            return input_data

        pspec = shd.PartitionSpec(*data_sharding_axis)

        def _already_sharded(x):
            return (
                isinstance(x, jax.Array)
                and hasattr(x, "sharding")
                and hasattr(x.sharding, "mesh")
                and x.sharding.mesh == mesh
                and hasattr(x.sharding, "spec")
                and x.sharding.spec == pspec
            )

        is_sharded = jax.tree.map(_already_sharded, input_data)
        if all(jax.tree.leaves(is_sharded)):
            return input_data

        def _reshard_leaf(x):
            target_sharding = sharding_utils_lib.get_sharding(x, mesh=mesh, pspec=pspec)
            if isinstance(x, jax.Array):
                # `jax.device_put` resharding can hit edge-cases on TPU when the
                # source is e.g. a `SingleDeviceSharding` array. Keep Tunix's
                # original "treat as process-local data" path for fully
                # addressable arrays, and only use `device_put` for global
                # arrays that span multiple hosts.
                if x.is_fully_addressable:
                    return jax.make_array_from_process_local_data(target_sharding, x)
                return jax.device_put(x, target_sharding)
            return jax.make_array_from_process_local_data(target_sharding, x)

        with jax.transfer_guard("allow"):
            return jax.tree.map(_reshard_leaf, input_data)

    sharding_utils_lib.shard_input = patched_fn
    return original_fn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace repo id for Qwen3-4B.",
    )
    parser.add_argument(
        "--local-model-dir",
        type=str,
        default="",
        help="Local path containing model safetensors; if empty, download to /tmp/models/<repo_id>.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sglang-context-length", type=int, default=2048)
    parser.add_argument("--sglang-page-size", type=int, default=64)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.8)
    parser.add_argument(
        "--rollout-devices",
        type=int,
        default=2,
        help="How many local TPU devices to allocate to sglang-jax rollout (rest go to trainer/reference).",
    )
    args = parser.parse_args()

    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jit_cache")
    os.environ.setdefault("TFDS_DATA_DIR", "/tmp/tfds")

    _add_repo_deps_to_sys_path()

    import optax
    from absl import logging
    import jax
    from tunix.examples.data import math_dataset
    from tunix.generate import tokenizer_adapter
    from tunix.generate import mappings as mappings_lib
    from tunix.models import automodel
    from tunix.models import naming
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.rl.grpo import grpo_learner
    from tunix.rl.rollout import base_rollout
    from tunix.sft import metrics_logger

    from tunix.cli.reward_fn import gsm8k as gsm8k_reward
    from tunix.cli.utils import data as data_lib

    logging.set_verbosity(logging.INFO)
    logging.info("JAX devices: %s", jax.devices())

    if args.local_model_dir:
        local_model_dir = Path(args.local_model_dir).resolve()
    else:
        local_model_dir = Path("/tmp/models") / args.model_id.replace("/", "__")
        _download_hf_repo_snapshot(args.model_id, local_model_dir)

    logging.info("Using local_model_dir=%s", local_model_dir)

    all_devices = list(jax.devices())
    if len(all_devices) < 1:
        raise RuntimeError("No JAX devices found.")

    rollout_n = max(1, min(args.rollout_devices, len(all_devices)))
    rollout_devices, train_devices = _split_devices_for_rollout(all_devices, rollout_n)

    # Training/reference: shard across remaining devices (FSDP) to save memory.
    train_mesh = _make_fsdp_tp_mesh(train_devices, fsdp=len(train_devices), tp=1)
    rollout_mesh = _make_fsdp_tp_mesh(rollout_devices, fsdp=1, tp=len(rollout_devices))

    logging.info("train_mesh.shape=%s train_mesh.devices=%s", train_mesh.shape, train_mesh.devices)
    logging.info(
        "rollout_mesh.shape=%s rollout_mesh.devices=%s", rollout_mesh.shape, rollout_mesh.devices
    )

    # Prevent sglang-jax from precompiling massive shapes (e.g. bs=4096/tokens=16384)
    # which can OOM for small models on TPU; cap KV + request pool sizes.
    desired_max_total_tokens = args.max_prompt_length + args.max_new_tokens + 256
    max_total_tokens = (
        (desired_max_total_tokens + args.sglang_page_size - 1) // args.sglang_page_size
    ) * args.sglang_page_size
    max_prefill_tokens = max_total_tokens
    max_running_requests = max(8, args.batch_size * args.num_generations)
    _patch_tunix_sglang_jax_engine_args(
        disable_precompile=True,
        max_total_tokens=max_total_tokens,
        max_prefill_tokens=max_prefill_tokens,
        max_running_requests=max_running_requests,
    )
    _patch_tunix_shard_input_for_multihost()

    naming_info = naming.ModelNaming(model_id=args.model_id)
    model_name = naming_info.model_name
    model_config = automodel.call_model_config(model_name)

    # Load reference model from local safetensors (avoid AutoModel.from_pretrained(GCS),
    # which only supports Gemma from GCS in OSS Tunix).
    with train_mesh:
        reference_model = automodel.create_model_from_safe_tensors(
            model_name=model_name,
            file_dir=str(local_model_dir),
            model_config=model_config,
            mesh=train_mesh,
        )

    actor_model = reference_model

    # Qwen3 models in Tunix don't yet expose BackendMappingMixin; provide an
    # explicit weight mapping for sglang-jax weight sync.
    rollout_mapping_config = mappings_lib.MappingConfig(
        to_hf_mappings={
            "lm_head.w": ("lm_head.embedding", (None, "model")),
            "embedder.input_embedding": ("model.embed_tokens.embedding", ("model", None)),
            "layers.*.input_layernorm.w": (
                "model.layers.*.input_layernorm.scale",
                (None,),
            ),
            "layers.*.post_attention_layernorm.w": (
                "model.layers.*.post_attention_layernorm.scale",
                (None,),
            ),
            "layers.*.attn.q_proj.w": (
                "model.layers.*.self_attn.q_proj.weight",
                (None, "model", None),
            ),
            "layers.*.attn.k_proj.w": (
                "model.layers.*.self_attn.k_proj.weight",
                (None, "model", None),
            ),
            "layers.*.attn.v_proj.w": (
                "model.layers.*.self_attn.v_proj.weight",
                (None, "model", None),
            ),
            "layers.*.attn.o_proj.w": (
                "model.layers.*.self_attn.o_proj.weight",
                ("model", None, None),
            ),
            "layers.*.attn.q_norm.w": (
                "model.layers.*.self_attn.q_norm.scale",
                (None,),
            ),
            "layers.*.attn.k_norm.w": (
                "model.layers.*.self_attn.k_norm.scale",
                (None,),
            ),
            "layers.*.mlp.gate_proj.kernel": (
                "model.layers.*.mlp.gate_proj.weight",
                (None, "model"),
            ),
            "layers.*.mlp.up_proj.kernel": (
                "model.layers.*.mlp.up_proj.weight",
                (None, "model"),
            ),
            "layers.*.mlp.down_proj.kernel": (
                "model.layers.*.mlp.down_proj.weight",
                ("model", None),
            ),
            "final_norm.w": ("model.norm.scale", (None,)),
        },
        lora_to_hf_mappings=None,
        to_hf_hook_fns=None,
        to_hf_transpose_keys={"lm_head.w": (1, 0)},
    )

    tokenizer = tokenizer_adapter.Tokenizer(
        tokenizer_type="huggingface",
        tokenizer_path=str(local_model_dir),
        add_bos=True,
        add_eos=True,
        hf_access_token=None,
    )

    rollout_config = base_rollout.RolloutConfig(
        max_tokens_to_generate=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        kv_cache_size=args.max_prompt_length + args.max_new_tokens + 256,
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        rollout_mapping_config=rollout_mapping_config,
        rollout_sglang_jax_model_version=str(local_model_dir),
        rollout_sglang_jax_context_length=args.sglang_context_length,
        rollout_sglang_jax_mem_fraction_static=args.sglang_mem_fraction_static,
        rollout_sglang_jax_init_with_random_weights=True,
        rollout_sglang_jax_disable_radix_cache=True,
        rollout_sglang_jax_enable_deterministic_sampling=False,
        rollout_sglang_jax_precompile_bs_paddings=[args.batch_size * args.num_generations],
        rollout_sglang_jax_precompile_token_paddings=[args.max_prompt_length + args.max_new_tokens],
        rollout_sglang_jax_chunked_prefill_size=-1,
        rollout_sglang_jax_page_size=args.sglang_page_size,
    )

    optimizer = optax.adamw(
        learning_rate=3e-6,
        b1=0.9,
        b2=0.99,
        weight_decay=0.1,
    )

    metrics_options = metrics_logger.MetricsLoggerOptions(
        log_dir="/tmp/tensorboard/grpo_gsm8k_qwen3_4b_sglang_jax",
        flush_every_n_steps=20,
    )
    training_config = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=10_000_000,
        max_steps=args.steps,
        mini_batch_size=None,
        train_micro_batch_size=None,
        rollout_micro_batch_size=None,
        compute_logps_micro_batch_size=None,
        metrics_logging_options=metrics_options,
        checkpoint_root_directory=None,
        checkpointing_options=None,
        profiler_options=None,
    )

    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: train_mesh,
            rl_cluster_lib.Role.REFERENCE: train_mesh,
            rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
        },
        rollout_engine="sglang_jax",
        offload_to_cpu=False,
        training_config=training_config,
        rollout_config=rollout_config,
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    algo_config = grpo_learner.GRPOConfig(
        num_generations=args.num_generations,
        num_iterations=1,
        beta=0.08,
        epsilon=0.2,
    )

    learner = grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=[
            gsm8k_reward.match_format_exactly,
            gsm8k_reward.check_answer,
        ],
    )

    dataset = math_dataset.create_dataset(
        data_source="tfds",
        dataset="gsm8k",
        tokenizer=tokenizer,
        tfds_download=True,
        split="train",
    )
    dataset, _ = data_lib.post_init_dataset(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        max_prompt_length=args.max_prompt_length,
    )

    start = time.perf_counter()
    learner.train(dataset, skip_jit=False)
    elapsed = time.perf_counter() - start
    logging.info("DONE: steps=%s elapsed_sec=%.3f", args.steps, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
