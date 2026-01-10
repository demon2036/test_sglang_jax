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
  - This appears to be an sglang-jax multi-host limitation for in-process usage; prefer `--rollout-engine vanilla` (single program) or run sglang-jax as external single-host servers and call them over HTTP.
  **References**: `sglang-jax/python/sgl_jax/srt/managers/scheduler.py` ; `tunix/tunix/rl/rollout/sglang_jax_rollout.py`
