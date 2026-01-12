# SOP: sglang-jax in-process hot weight update (model_state_leaves)

- **Title**: SOP: sglang-jax in-process hot weight update (swap `model_state_leaves`)
- **Prereqs**:
  - Cloud TPU VM with >=1 device (this run used `v5litepod-4` / 4 devices)
  - Conda env `sglang-jax` with JAX TPU working (this run: Python `3.12.12`, `jax==0.8.1`, backend `tpu`)
  - Repo on TPU VM: `/root/test_sglang_jax` with upstream `sglang-jax` at `/root/test_sglang_jax/sglang-jax`
  - (Windows) OpenSSH client + key at `$HOME\.ssh\google_compute_engine`

## What “in-process hot update” means here

- **Engine/library mode**: overwrite `model_runner.model_state_leaves` in the same Python process, then flush cache.
- **HTTP server mode**: keep the same uvicorn process alive; update weights via `/admin/*` endpoints (a short maintenance window is still recommended).

## Steps

### 1) Pick a READY TPU VM and get its external IP (PowerShell)

- List TPU VMs in the zone:
  - `gcloud compute tpus tpu-vm list --zone=europe-west4-b --format='table(name,zone,state,acceleratorType)'`
- Get external IP:
  - `gcloud compute tpus tpu-vm describe sglang-jax-v5litepod-4-openai-spot-20260112-114521 --zone=europe-west4-b --format='value(networkEndpoints[0].accessConfig.externalIp)'`

### 2) Sync plugin files to the TPU VM (PowerShell)

- `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/sglang_jax/weight_hot_update.py root@34.6.84.124:/root/test_sglang_jax/plugins/sglang_jax/weight_hot_update.py`
- `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/sglang_jax/run_engine_hot_update_demo.py root@34.6.84.124:/root/test_sglang_jax/plugins/sglang_jax/run_engine_hot_update_demo.py`
- `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/sglang_jax/run_multi_openai_servers.py root@34.6.84.124:/root/test_sglang_jax/plugins/sglang_jax/run_multi_openai_servers.py`
- `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/sglang_jax/admin_weight_endpoints_client.py root@34.6.84.124:/root/test_sglang_jax/plugins/sglang_jax/admin_weight_endpoints_client.py`

### 3) Validate Engine “hot swap” (no HTTP)

- Run the demo on device 0 (prints checksum before/after weight perturb):
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@34.6.84.124 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_hot_update; mkdir -p $JAX_COMPILATION_CACHE_DIR; python -u -m plugins.sglang_jax.run_engine_hot_update_demo --model hf-internal-testing/tiny-random-LlamaForCausalLM --load-format dummy --device-index 0 --noise-leaves 4'`

### 4) Validate HTTP in-place hot update (4 ports) + admin client

- Start 4 servers and keep them running:
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@34.6.84.124 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_openai_multi; mkdir -p $JAX_COMPILATION_CACHE_DIR; python -u -m plugins.sglang_jax.run_multi_openai_servers --model hf-internal-testing/tiny-random-LlamaForCausalLM --num-servers 4 --base-port 31000 --load-format dummy --context-length 128 --max-total-tokens 128 --max-prefill-tokens 128 --page-size 8 --max-running-requests 1 --prompt ''1+1=?'' --max-new-tokens 1 --request-timeout 300 --enable-weight-reload --keep-running'`

- In a second terminal, call admin endpoints on port 31000:
  - Check checksum:
    - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@34.6.84.124 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; python -u -m plugins.sglang_jax.admin_weight_endpoints_client --base-url http://127.0.0.1:31000 checksum'`
  - Perturb weights in-memory (demo hot update):
    - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@34.6.84.124 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; python -u -m plugins.sglang_jax.admin_weight_endpoints_client --base-url http://127.0.0.1:31000 perturb --seed 0 --scale 0.001 --num-leaves 4'`
  - Check checksum again (should match `checksum_after`):
    - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@34.6.84.124 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; python -u -m plugins.sglang_jax.admin_weight_endpoints_client --base-url http://127.0.0.1:31000 checksum'`

## Expected Result

- `run_engine_hot_update_demo` prints `checksum_before`, then `checksum_after` with a non-zero `checksum_delta`.
- `admin_weight_endpoints_client perturb` returns `ok: true` and changes the checksum (verified on port `31000`).

## Troubleshooting

- If TPU is “already in use”, remove stale lock: `rm -f /tmp/libtpu_lockfile`.
- If the server exits when your SSH session drops, run it under `nohup`/`tmux` (this run kept the SSH session open while testing).
- Windows OpenSSH may print `close - IO is still pending on closed socket.`; retry commands if the session closes unexpectedly.

## References

- `plugins/sglang_jax/weight_hot_update.py`
- `plugins/sglang_jax/run_engine_hot_update_demo.py`
- `plugins/sglang_jax/run_multi_openai_servers.py` (`/admin/weights_checksum`, `/admin/perturb_weights`, `/admin/reload_weights`)
- `plugins/sglang_jax/admin_weight_endpoints_client.py`

