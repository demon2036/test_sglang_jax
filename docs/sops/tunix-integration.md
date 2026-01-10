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
