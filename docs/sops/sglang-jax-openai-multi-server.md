# SGLang-JAX OpenAI Multi-Server SOPs

- **Title**: SOP: Run 4 OpenAI-compatible servers (single-process, 4 TPU devices)
  **Prereqs**: Ubuntu 22.04 TPU VM runtime `tpu-ubuntu2204-base`; gcloud project `civil-rarity-482610-s5`; spot capacity for `v4-8` in `us-central2-b`
  **Steps**:
  - Create a v4-8 spot TPU VM:
    - `TPU_NAME=sglang-jax-v4-8-openai-20260111-215024; ZONE=us-central2-b`
    - `gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type=v4-8 --version=tpu-ubuntu2204-base --spot --quiet`
    - `gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format='value(state)'`
  - Verify SSH + OS:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'whoami; lsb_release -a || cat /etc/os-release; python3 --version || true' --quiet`
  - Install conda + create `sglang-jax` env:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; if [ ! -d "/root/miniconda3" ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | awk "{print \\$1}" | grep -qx sglang-jax; then conda create -y -n sglang-jax python=3.12; fi; conda activate sglang-jax; pip install -U pip' --quiet`
  - Clone repos on the TPU VM:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; if [ ! -d /root/test_sglang_jax/.git ]; then git clone https://github.com/demon2036/test_sglang_jax.git /root/test_sglang_jax; fi; if [ ! -d /root/test_sglang_jax/sglang-jax/.git ]; then git clone https://github.com/sgl-project/sglang-jax.git /root/test_sglang_jax/sglang-jax; fi'`
  - Install sglang-jax:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax/sglang-jax; pip install -e "python[all]"' --quiet`
  - Sync the multi-server plugin:
    - `gcloud alpha compute tpus tpu-vm scp /home/john/github/test_sglang_jax/plugins/sglang_jax/run_multi_openai_servers.py root@"$TPU_NAME":/root/test_sglang_jax/plugins/sglang_jax/run_multi_openai_servers.py --zone="$ZONE" --quiet`
  - Run 4 OpenAI-compatible servers (single process, one TPU device each) and send concurrent requests:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_openai_multi; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u -m plugins.sglang_jax.run_multi_openai_servers --model hf-internal-testing/tiny-random-LlamaForCausalLM --num-servers 4 --base-port 31000 --load-format dummy --context-length 128 --max-total-tokens 128 --max-prefill-tokens 128 --page-size 8 --max-running-requests 1 --prompt "1+1=?" --max-new-tokens 1 --request-timeout 300 --force-exit 2>&1 | tee /tmp/multi_openai_servers_4ports.log'`
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'grep -n "READY\\|DONE\\|RESP" /tmp/multi_openai_servers_4ports.log || true'`
  - Delete the TPU VM:
    - `gcloud alpha compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --quiet`
  **Expected Result**: `/tmp/multi_openai_servers_4ports.log` shows `READY` with ports `[31000-31003]`, four `RESP` lines with status `200`, and `DONE: servers=4 wall_sec=...` confirming concurrent requests to four OpenAI-compatible ports in a single process.
  **Troubleshooting**:
  - If you see `TPU is already in use by process ...` or `/dev/accel*` permission errors, kill leftover Python processes and remove `/tmp/libtpu_lockfile` before retrying.
  - If the SSH command hangs after completion, re-run with `--force-exit` (already in the command above) to avoid lingering engine threads.
  - If requests return 422 for missing `request/raw_request`, ensure the plugin uses module-level FastAPI imports (needed with `from __future__ import annotations`).
  **References**: `plugins/sglang_jax/run_multi_openai_servers.py`
