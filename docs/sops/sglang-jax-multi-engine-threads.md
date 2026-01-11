# SGLang-JAX Multi-Engine Threading SOPs

- **Title**: SOP: Attempt v4-8 single-process multi-engine thread test (blocked)
  **Prereqs**: Ubuntu host; gcloud configured for project `civil-rarity-482610-s5`; network access to GitHub/Anaconda
  **Steps**:
  - Install alpha commands (needed for `tpu-vm`):
    - `gcloud components install alpha --quiet`
  - Try to create a v4-8 spot TPU VM (failed due to quota):
    - `TPU_NAME=sglang-jax-v4-8-thread-20260111-085411; ZONE=us-central2-b`
    - `gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type=v4-8 --version=tpu-ubuntu2204-base --spot --quiet`
  - Create a v4-8 on-demand TPU VM:
    - `TPU_NAME=sglang-jax-v4-8-thread-ondemand-20260111-085633; ZONE=us-central2-b`
    - `gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type=v4-8 --version=tpu-ubuntu2204-base --quiet`
    - `gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format='value(state,acceleratorType)'`
  - Verify SSH + OS:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'whoami; lsb_release -a || cat /etc/os-release; python3 --version || true' --quiet`
  - Install conda + create `sglang-jax` env:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; if [ ! -d "/root/miniconda3" ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | awk "{print \\$1}" | grep -qx sglang-jax; then conda create -y -n sglang-jax python=3.12; fi; conda activate sglang-jax; pip install -U pip' --quiet`
  - Attempt to clone repos on the TPU VM (failed because VM entered `DELETING`):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; if [ ! -d /root/test_sglang_jax/.git ]; then git clone https://github.com/demon2036/test_sglang_jax.git /root/test_sglang_jax; fi; if [ ! -d /root/test_sglang_jax/sglang-jax/.git ]; then git clone https://github.com/sgl-project/sglang-jax.git /root/test_sglang_jax/sglang-jax; fi'`
  **Expected Result**: TPU VM reaches `READY` and accepts SSH; conda env `sglang-jax` created. In this attempt, the TPU VM transitioned to `DELETING` before repo clone could run.
  **Troubleshooting**:
  - Spot create failed with quota exhaustion: `Quota limit 'TPUV4sPreemptiblePodPerProjectPerRegionForTPUAPI' has been exceeded`.
  - On-demand VM entered `DELETING` unexpectedly; check for org policy or automation that removes on-demand TPU VMs.
  **References**: `plugins/sglang_jax/run_multi_engine_threads.py`

- **Title**: SOP: v4-8 multi-engine thread test (device mismatch failure)
  **Prereqs**: TPU VM runtime `tpu-ubuntu2204-base`; gcloud project `civil-rarity-482610-s5`; network access to GitHub/Anaconda
  **Steps**:
  - Create a spot v4-8 TPU VM and wait for READY:
    - `TPU_NAME=sglang-jax-v4-8-thread-20260111-091651; ZONE=us-central2-b`
    - `gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type=v4-8 --version=tpu-ubuntu2204-base --spot --quiet`
    - `gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format='value(state,acceleratorType)'`
  - Verify SSH + OS:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'whoami; lsb_release -a || cat /etc/os-release; python3 --version || true' --quiet`
  - Install conda + create `sglang-jax` env:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; if [ ! -d "/root/miniconda3" ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | awk "{print \\$1}" | grep -qx sglang-jax; then conda create -y -n sglang-jax python=3.12; fi; conda activate sglang-jax; pip install -U pip' --quiet`
  - Clone repos on the TPU VM:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; if [ ! -d /root/test_sglang_jax/.git ]; then git clone https://github.com/demon2036/test_sglang_jax.git /root/test_sglang_jax; fi; if [ ! -d /root/test_sglang_jax/sglang-jax/.git ]; then git clone https://github.com/sgl-project/sglang-jax.git /root/test_sglang_jax/sglang-jax; fi'`
  - Install sglang-jax:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax/sglang-jax; pip install -e "python[all]"'`
  - Sync the updated thread runner:
    - `gcloud alpha compute tpus tpu-vm scp /home/john/github/test_sglang_jax/plugins/sglang_jax/run_multi_engine_threads.py root@"$TPU_NAME":/root/test_sglang_jax/plugins/sglang_jax/run_multi_engine_threads.py --zone="$ZONE" --quiet`
  - Run 4-engine thread test (dummy weights, smaller context for speed):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_multi; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u -m plugins.sglang_jax.run_multi_engine_threads --model Qwen/Qwen3-4B-Instruct-2507 --num-engines 4 --load-format dummy --context-length 1024 --max-total-tokens 1024 --max-prefill-tokens 1024 --page-size 16 --max-running-requests 1 --prompt "1+1=?" 2>&1 | tee /tmp/multi_engine_threads_after_fix.log'`
  - (Optional) Run with `--concurrency 1` for sequential generation:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_multi; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u -m plugins.sglang_jax.run_multi_engine_threads --model Qwen/Qwen3-4B-Instruct-2507 --num-engines 4 --load-format dummy --context-length 1024 --max-total-tokens 1024 --max-prefill-tokens 1024 --page-size 16 --max-running-requests 1 --prompt "1+1=?" --concurrency 1 2>&1 | tee /tmp/multi_engine_threads_concurrency1.log'`
  **Expected Result**: All 4 engines initialize on device ids `[0,1,2,3]`, but generation fails with `ValueError: Received incompatible devices for jitted computation` (sampler state and args on different TPU devices).
  **Troubleshooting**: If TPU reports it is already in use, run `rm -f /tmp/libtpu_lockfile` and ensure no leftover `run_multi_engine_threads` process is running.
  **References**: `plugins/sglang_jax/run_multi_engine_threads.py`

- **Title**: SOP: Diagnose multi-engine device mismatch from cached linear_penalty
  **Prereqs**: Local `sglang-jax` clone; error shows `args[1][8]` vs `sampler_state_leaves[0]` device ids
  **Steps**:
  - Map `args[1][8]` to `SamplingMetadata.linear_penalty` via `tree_flatten` order:
    - `sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py`
  - Confirm `linear_penalty` is created via `_get_or_create_zero_penalty_device` when no penalties are applied:
    - `sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py`
  - Verify `_zero_linear_penalty_cache` is global and keyed only by shape (not device/mesh):
    - `sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py`
  - Check `device_array` uses `jax.device_put` with `device=sharding`, which pins arrays to the mesh used for the first engine:
    - `sglang-jax/python/sgl_jax/srt/utils/jax_utils.py`
  - Confirm `jitted_sampler` expects all args on the same device and is invoked from `ModelRunner.sample`:
    - `sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py`
  **Expected Result**: Root cause attributed to cross-engine reuse of a cached `linear_penalty` array created on the first engine's device, which conflicts with later engines' devices.
  **Troubleshooting**: If the mismatch index is not `args[1][8]`, re-check the pytree order for the failing argument.
  **References**: `sglang-jax/python/sgl_jax/srt/sampling/sampling_batch_info.py`; `sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py`

- **Title**: SOP: Patch zero-penalty cache + verify 4 engines sequentially (tiny model)
  **Prereqs**: TPU VM `sglang-jax-v4-8-thread-20260111-091651` in `us-central2-b`; repo at `/root/test_sglang_jax`; conda env `sglang-jax`
  **Steps**:
  - Patch plugins to make zero-penalty cache device-aware:
    - `plugins/sglang_jax/run_multi_engine_threads.py` (see `_patch_zero_penalty_cache`)
  - Sync updated plugin to TPU VM:
    - `gcloud alpha compute tpus tpu-vm scp /home/john/github/test_sglang_jax/plugins/sglang_jax/run_multi_engine_threads.py root@"$TPU_NAME":/root/test_sglang_jax/plugins/sglang_jax/run_multi_engine_threads.py --zone="$ZONE" --quiet`
  - Run 4 engines sequentially (`--concurrency 1`) with a tiny Llama model:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_multi; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u -m plugins.sglang_jax.run_multi_engine_threads --model hf-internal-testing/tiny-random-LlamaForCausalLM --num-engines 4 --load-format dummy --context-length 128 --max-total-tokens 128 --max-prefill-tokens 128 --page-size 8 --max-running-requests 1 --max-new-tokens 1 --prompt "1+1=?" --concurrency 1 2>&1 | tee /tmp/multi_engine_threads_multi_concurrency1_tiny.log'`
  - (Optional) inspect output:
    - `tail -n 40 /tmp/multi_engine_threads_multi_concurrency1_tiny.log`
  **Expected Result**: All 4 engines generate sequentially with outputs like `generate_sec=... text=...` and `DONE: engines=4 wall_sec=...`.
  **Troubleshooting**: If a multi-engine run hangs with `--concurrency 4`, try `--concurrency 1` to confirm per-engine generation works; the concurrent run can still stall.
  **References**: `plugins/sglang_jax/run_multi_engine_threads.py`

- **Title**: SOP: Attempt engine-per-thread concurrent run (still long-running)
  **Prereqs**: Same as above; plugin patch to skip signal handlers in non-main threads
  **Steps**:
  - Enable engine-per-thread mode:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_multi; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u -m plugins.sglang_jax.run_multi_engine_threads --model hf-internal-testing/tiny-random-LlamaForCausalLM --num-engines 4 --load-format dummy --context-length 128 --max-total-tokens 128 --max-prefill-tokens 128 --page-size 8 --max-running-requests 1 --max-new-tokens 1 --prompt "1+1=?" --concurrency 4 --engine-per-thread 2>&1 | tee /tmp/multi_engine_threads_engine_per_thread_tiny.log'`
  **Expected Result**: Engines initialize, but concurrent generation did not complete within 5 minutes; process remained active with high CPU.
  **Troubleshooting**: If needed, terminate with `kill -9 <pid>` and remove `/tmp/libtpu_lockfile`.
  **References**: `plugins/sglang_jax/run_multi_engine_threads.py`

- **Title**: SOP: v4-8 concurrent run succeeds with engine workers (warmup + concurrency 4)
  **Prereqs**: TPU VM `sglang-jax-v4-8-thread-20260111-091651` in `us-central2-b`; repo at `/root/test_sglang_jax`; conda env `sglang-jax`
  **Steps**:
  - Sync the updated runner:
    - `TPU_NAME=sglang-jax-v4-8-thread-20260111-091651; ZONE=us-central2-b`
    - `gcloud alpha compute tpus tpu-vm scp /home/john/github/test_sglang_jax/plugins/sglang_jax/run_multi_engine_threads.py root@"$TPU_NAME":/root/test_sglang_jax/plugins/sglang_jax/run_multi_engine_threads.py --zone="$ZONE" --quiet`
  - Run 4 engines concurrently with warmup (tiny model, dummy weights):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_multi; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u -m plugins.sglang_jax.run_multi_engine_threads --model hf-internal-testing/tiny-random-LlamaForCausalLM --num-engines 4 --load-format dummy --context-length 128 --max-total-tokens 128 --max-prefill-tokens 128 --page-size 8 --max-running-requests 1 --max-new-tokens 1 --prompt "1+1=?" --concurrency 4 --warmup-sequential 2>&1 | tee /tmp/multi_engine_threads_workers_concurrency4_tiny.log'`
  - If the process stays alive after `DONE`, stop it:
    - `ps -ef | grep -v grep | grep -n "run_multi_engine_threads" || true`
    - `kill -9 <python-pid> <bash-pid>`
  **Expected Result**: All 4 engines initialize on device ids `[0,1,2,3]`, warmup sequentially, then concurrent generate completes with `DONE: engines=4 wall_sec=...` and per-engine output lines; `wall_sec` is close to the max of per-engine `generate_sec`, confirming concurrency.
  **Troubleshooting**: If the SSH command hangs after `DONE`, terminate the lingering python process (Engine shutdown does not stop all background threads in single-process mode).
  **References**: `plugins/sglang_jax/run_multi_engine_threads.py`
