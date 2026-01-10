# Scaling and Rollouts SOPs

- **Title**: SOP: Multi-host RL rollout with full TPU utilization (tp=8 replicas)
  **Prereqs**: Cloud TPU v4; 1 process per TPU host (PJRT/libtpu lock); `sglang-jax` installed; model can run with `--tp-size 8`
  **Steps**:
  - Prefer **many independent `v4-8` TPU VMs/slices**, each running **one** sglang-jax server with `--tp-size 8`, instead of one large multi-host slice. (This is the practical way to get N-way scaling with current sglang-jax.)
  - Start one server per TPU VM (template; fill in real values):
    - `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m sgl_jax.launch_server --model-path <MODEL_OR_CHECKPOINT_DIR> --device tpu --tp-size 8 --host 0.0.0.0 --port <PORT> --max-seq-len <ROLLOUT_MAX_LEN> --page-size <PAGE_SIZE>`
  - In rollout workers, send requests to the server pool and keep **high concurrency** so each server stays near `max_running_requests`:
    - Use `GET /get_load` to pick the least-loaded server (or do simple round-robin).
    - Use `POST /v1/chat/completions` (OpenAI-compatible) or `POST /generate` for generation.
  - Weight update strategy (pick one):
    - Low-friction: keep rollout policy stale for a while, then restart servers on new checkpoints (jit cache reduces recompilation cost).
    - Faster hot-swap: train as LoRA and pass `lora_path` per request (engine supports per-request LoRA).
  - If you must use a **single multi-host slice** (e.g., `v4-32`) and still want `tp=8` replicas: current `sglang-jax` does not implement `--dp-size > 1` (engine code has `pass`), so you cannot reliably run 4 independent engines inside one slice; either re-provision as multiple `v4-8`, or run one engine with `--tp-size <TOTAL_DEVICES>` on that slice.
  - Observed: **in-process** sglang-jax rollouts inside a **multi-host** JAX program can crash with `FAILED_PRECONDITION: The program continuator has halted unexpectedly.` (see `docs/sops/tunix-integration.md`). Treat multi-host sglang-jax rollouts as **external service** for now.
  **Expected Result**: Rollout throughput scales ~linearly with number of `v4-8` replicas; TPU stays busy once request concurrency exceeds the engine's `max_running_requests`
  **Troubleshooting**:
  - If you try multiple engines on one TPU VM, the second engine typically fails with `The TPU is already in use by process ...` (libtpu multi-process lock).
  - If a multi-host slice is underutilized with `--tp-size 8`, check mesh shape: `create_device_mesh(ici_parallelism=[-1,tp])` will create a non-1 `data` axis, but sglang-jax does not yet schedule inference across multiple `data` replicas.
  **References**: `python/sgl_jax/launch_server.py` ; `python/sgl_jax/srt/entrypoints/http_server.py` ; `python/sgl_jax/srt/entrypoints/engine.py` ; `python/sgl_jax/srt/managers/scheduler.py`

- **Title**: SOP: Single-process RL training + rollout on one TPU host (Tunix-style)
  **Prereqs**: Cloud TPU v4; Tunix + sglang-jax; you can run everything inside **one** Python process
  **Steps**:
  - Put trainer and rollout in the same process and **split devices** into disjoint meshes (e.g., 2 devices for rollout, rest for actor/reference).
  - Ensure sglang-jax is constructed with `enable_single_process=True` and `device_indexes=<rollout_mesh.device_ids>` so it uses threads and only touches the intended device subset.
  - Use `plugins/tunix/run_grpo_gsm8k_qwen3_4b_10steps.py` as a working example (it patches Tunix to avoid sglang-jax precompile OOMs).
  **Expected Result**: TPU stays single-process (no PJRT lock conflicts), but both training and rollout execute concurrently on different device subsets
  **Troubleshooting**: If you attempt separate rollout processes on the same TPU host, you will hit `/tmp/libtpu_lockfile` conflicts; keep 1 process per host and scale out with more TPU VMs
  **References**: `tunix/tunix/generate/sglang_jax_sampler.py` ; `sglang-jax/python/sgl_jax/srt/entrypoints/engine.py` ; `plugins/tunix/run_grpo_gsm8k_qwen3_4b_10steps.py`
