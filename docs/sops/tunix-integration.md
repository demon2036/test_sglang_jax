# Tunix Integration SOPs

- **Title**: SOP: Inspect Tunix's SGLang-JAX integration (single-process engine + weight sync)
  **Prereqs**: Ubuntu; `git` and `rg` available; network access to GitHub
  **Steps**:
  - `cd /home/john/test_sglang_jax`
  - `rm -rf tunix && git clone https://github.com/google/tunix.git`
  - `rg -n "sglang|sgl_jax|sglang-jax" -S tunix | head`
  - Review key files:
    - `tunix/tunix/generate/sglang_jax_sampler.py`
    - `tunix/tunix/rl/rollout/sglang_jax_rollout.py`
    - `tunix/tests/generate/sglang_jax_sampler_test.py`
    - `tunix/scripts/grpo_demo_llama3_qwen2.py`
  **Expected Result**:
  - Tunix uses `Engine` as a **library in the same Python process** with `enable_single_process=True` (threads, not subprocesses), so it does **not** contradict the TPU \"single-process\" lock behavior.
  - Tunix updates sglang-jax weights **in-memory** by overwriting `model_runner.model_state_leaves` (see `SglangJaxSampler.update_params`), instead of starting a separate sglang-jax server process.
  **Troubleshooting**: N/A
  **References**: https://github.com/google/tunix ; `tunix/tunix/generate/sglang_jax_sampler.py` ; `tunix/scripts/grpo_demo_llama3_qwen2.py`

- **Title**: SOP: Inspect Tunix Pathways hooks (optional) vs OSS multi-host PJRT
  **Prereqs**: Ubuntu; `rg` available
  **Steps**:
  - `cd /home/john/test_sglang_jax`
  - `rg -n "Pathways|pathways_bns|jax_xla_backend\" -S tunix | head`
  - Review key files:
    - `tunix/README.md` (mentions Pathways for large-scale multi-host)
    - `tunix/tunix/cli/grpo_main.py` (flag `--pathways_bns` toggles JAX backend to `pathways`)
    - `tunix/tunix/rl/reshard.py` (tries `pathwaysutils` first; falls back to `jax.device_put`)
    - `tunix/tunix/oss/utils.py` (`pathways_available()` checks `JAX_PLATFORMS` contains `proxy` + `pathwaysutils` import)
  **Expected Result**:
  - Pathways is **not required** for basic multi-host TPU slices in OSS: Tunix uses normal JAX sharding (`jax.make_mesh`) as long as `jax.process_count()>1` / `jax.device_count()` reflects the multi-host runtime.
  - Pathways mode is **optional** and requires a Pathways runtime: `--pathways_bns` + a JAX backend that supports `pathways`/`proxy` + `pathwaysutils`. Without it, code paths fall back to OSS behavior.
  **Troubleshooting**:
  - If you don't have Pathways, do not set `JAX_PLATFORMS=proxy` and do not pass `--pathways_bns`; run multi-host via standard TPU VM slice launch (`--worker=all`).
  **References**: `tunix/README.md` ; `tunix/tunix/cli/grpo_main.py` ; `tunix/tunix/rl/reshard.py` ; `tunix/tunix/oss/utils.py`

- **Title**: SOP: Run Tunix GRPO GSM8K (Qwen3-4B) with sglang-jax rollout on TPU (10 steps)
  **Prereqs**: Cloud TPU v4+ (TPU v3 is unsupported by sglang-jax paged attention); TPU VM already bootstrapped with conda env `tunix` (Python 3.12 + JAX TPU); repos at `/root/sglang-jax` and `/root/tunix`
  **Steps**:
  - `TPU_NAME=tunix-grpo-qwen3-4b-v4-8-spot-20260110-170909; ZONE=us-central2-b`
  - Ensure repos exist under `/root`:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; if [ ! -d /root/sglang-jax/.git ]; then git clone https://github.com/sgl-project/sglang-jax.git /root/sglang-jax; fi; if [ ! -d /root/tunix/.git ]; then git clone https://github.com/google/tunix.git /root/tunix; fi'`
  - Sanity-check JAX backend:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate tunix; python -c "import jax; print(\"jax\", jax.__version__, \"backend\", jax.default_backend(), \"devices\", len(jax.devices()))"'`
  - Sync local plugin code to TPU root (non-invasive overlays):
    - `gcloud alpha compute tpus tpu-vm scp --recurse /home/john/test_sglang_jax/plugins root@"$TPU_NAME":/root/ --zone="$ZONE" --quiet`
  - Run the Tunix GRPO smoke (writes a full log to `/tmp/grpo_qwen3_4b_10steps.log`):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate tunix; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_grpo; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u /root/plugins/tunix/run_grpo_gsm8k_qwen3_4b_10steps.py --steps 10 --rollout-devices 2 --sglang-mem-fraction-static 0.8 2>&1 | tee /tmp/grpo_qwen3_4b_10steps.log'`
  **Expected Result**: Log contains `DONE: steps=10 elapsed_sec=622.413` and per-step losses like `Train step 1 training loss: 0.000044` (see `/tmp/grpo_qwen3_4b_10steps.log`)
  **Troubleshooting**:
  - If you see `NotImplementedError: TPU version must be 4 or higher`, use a v4+ TPU VM/slice.
  - If you hit TPU HBM OOM during sglang-jax startup, reduce rollout limits (the plugin runner forces `disable_precompile=True` and caps `max_total_tokens`, but you may still need a smaller context length).
  **References**: `plugins/tunix/run_grpo_gsm8k_qwen3_4b_10steps.py` ; `tunix/tunix/generate/sglang_jax_sampler.py` ; `sglang-jax/python/sgl_jax/srt/entrypoints/engine.py`

- **Title**: SOP: Run Tunix GRPO GSM8K (Qwen3-4B) on multi-host TPU (v4-16 / 2 workers) using vanilla rollout (10 steps)
  **Prereqs**: Cloud TPU v4-16 with 2 workers (e.g., `tunix-grpo-qwen3-4b-v4-16-spot-20260110-180240` in `us-central2-b`); conda env `tunix` (Python 3.12.12, `jax==0.8.1`, `jaxlib==0.8.1`); repo cloned to `/root/test_sglang_jax`; model weights at `/tmp/models/Qwen__Qwen3-4B-Instruct-2507`
  **Steps**:
  - `TPU_NAME=tunix-grpo-qwen3-4b-v4-16-spot-20260110-180240; ZONE=us-central2-b`
  - Pull latest plugins on all workers:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; cd /root/test_sglang_jax; git pull --ff-only'`
  - Confirm JAX multi-host env (prints on both workers):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate tunix; python -c "import jax, jaxlib, sys; print(\"python\", sys.version.split()[0]); print(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__, \"backend\", jax.default_backend(), \"processes\", jax.process_count(), \"process_index\", jax.process_index(), \"devices\", len(jax.devices()), \"local\", jax.local_device_count())"'`
  - Run 10-step GRPO smoke with vanilla rollout (writes per-worker logs):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate tunix; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_grpo; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u /root/test_sglang_jax/plugins/tunix/run_grpo_gsm8k_qwen3_4b_10steps.py --steps 10 --rollout-devices 8 --rollout-engine vanilla --local-model-dir /tmp/models/Qwen__Qwen3-4B-Instruct-2507 2>&1 | tee /tmp/grpo_qwen3_4b_10steps_multihost_vanilla_$(hostname).log'`
  **Expected Result**:
  - Each worker log contains `DONE: steps=10 elapsed_sec=262.053` (process 0) / `DONE: steps=10 elapsed_sec=261.941` (process 1) and per-step losses like `Train step 10 training loss: 0.000123`.
  **Troubleshooting**:
  - If TPU reports "already in use", ensure no leftover TPU processes; remove stale `/tmp/libtpu_lockfile`.
  **References**: `plugins/tunix/run_grpo_gsm8k_qwen3_4b_10steps.py` ; `tunix/tunix/rl/rl_cluster.py` ; `tunix/tunix/rl/rollout/vanilla_rollout.py`

- **Title**: SOP: Multi-host Tunix + in-process sglang-jax rollout currently fails (TPU runtime continuator halted)
  **Prereqs**: Same as the multi-host SOP above
  **Steps**:
  - Run the same program but with default `rollout_engine=sglang_jax`:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate tunix; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_grpo; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u /root/test_sglang_jax/plugins/tunix/run_grpo_gsm8k_qwen3_4b_10steps.py --steps 10 --rollout-devices 8 --local-model-dir /tmp/models/Qwen__Qwen3-4B-Instruct-2507 --sglang-mem-fraction-static 0.8 2>&1 | tee /tmp/grpo_qwen3_4b_10steps_multihost_$(hostname).log'`
  **Expected Result**:
  - Fails with `jax.errors.JaxRuntimeError: FAILED_PRECONDITION: The program continuator has halted unexpectedly.` from `sglang-jax/python/sgl_jax/srt/managers/scheduler.py` during `jax.device_get(...)`.
  **Troubleshooting**:
  - Likely root cause (code-level): Tunix instantiates `Engine(enable_single_process=True)` with `device_indexes` derived from a mesh that can span multiple hosts, but does **not** set `nnodes/node_rank/dist_init_addr`. This keeps sglang-jax in "single-node" scheduling mode, so **each host schedules independently** (different requests / early-EOS timing), while the underlying JAX program runs on a **cross-host device mesh**. That breaks the "all hosts execute the same collectives in the same order" requirement and can trip TPU runtime's "program continuator halted" failure (it surfaces at `Scheduler.run_batch -> jax.device_get`).
  - Pointers:
    - Tunix config path: `tunix/tunix/generate/sglang_jax_sampler.py` (`device_indexes=mesh.device_ids...`, no `nnodes/node_rank`).
    - sglang-jax multi-node sync is gated on `nnodes>1`: `sglang-jax/python/sgl_jax/srt/managers/scheduler.py` (`recv_requests` broadcasts only when `self.nnodes > 1`).
    - Failure surface: `sglang-jax/python/sgl_jax/srt/managers/scheduler.py` (`run_batch` does `jax.device_get(next_token_ids_device)`).
  - Another warning sign: many sglang-jax tensors are only explicitly sharded when `jax.process_count()==1` (e.g. `ForwardBatch`, `SamplingMetadata`, `LogitsMetadata`), so multi-host PJRT behavior relies on implicit placement and is less battle-tested.
  - Workarounds:
    - Prefer `--rollout-engine vanilla` for multi-host training, or run sglang-jax as external single-host servers and call them over HTTP (our "single controller" design).
    - If you want to experiment with in-process multi-host anyway, the two plausible directions are: (1) make rollout mesh **host-local** so each host runs an independent sglang-jax engine (no cross-host collectives inside the engine), or (2) wire sglang-jax "multi-node serving" mode (`nnodes/node_rank/dist_init_addr`) and ensure only `node_rank=0` accepts requests (others follow).
  **References**: `sglang-jax/python/sgl_jax/srt/managers/scheduler.py` ; `tunix/tunix/rl/rollout/sglang_jax_rollout.py`
