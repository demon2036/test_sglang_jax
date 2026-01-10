# TPU Tests SOPs

## Unittest runs

- **Title**: SOP: Sync `/home/john/test_sglang_jax/plugins` to TPU root + run Qwen3 chat template unittest
  **Prereqs**: Cloud TPU v4+; repos under `/root/`; conda env with JAX TPU + sglang-jax deps (e.g., `tunix`)
  **Steps**:
  - `TPU_NAME=tunix-grpo-qwen3-4b-v4-8-spot-20260110-170909; ZONE=us-central2-b`
  - Ensure `sglang-jax` exists in `/root`:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; if [ ! -d /root/sglang-jax/.git ]; then git clone https://github.com/sgl-project/sglang-jax.git /root/sglang-jax; fi'`
  - Sync plugin code (non-invasive overlays):
    - `gcloud alpha compute tpus tpu-vm scp --recurse /home/john/test_sglang_jax/plugins root@"$TPU_NAME":/root/ --zone="$ZONE" --quiet`
  - Run the unittest from `/root` (Qwen3-14B by default; override via `SGLANG_JAX_MODEL`):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate tunix; cd /root; mkdir -p /tmp/jit_cache; JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m unittest plugins.sglang_jax.test_engine_chat_template_qwen3_14b.TestEngineChatTemplateQwen3'`
  **Expected Result**: Prints `Model response: ...` and ends with `OK`
  **Troubleshooting**: If TPU reports "already in use", ensure no other TPU processes are running; `rm -f /tmp/libtpu_lockfile` can clear a stale lockfile
  **References**: `plugins/sglang_jax/test_engine_chat_template_qwen3_14b.py`

- **Title**: SOP: Run plugin unittest after SSH (avoid `Ran 0 tests`)
  **Prereqs**: SSH into TPU VM as root; conda env `tunix` available; plugins synced to `/root/plugins`
  **Steps**:
  - `source /root/miniconda3/etc/profile.d/conda.sh && conda activate tunix`
  - `cd /root`
  - `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m unittest plugins.sglang_jax.test_engine_chat_template_qwen3_14b.TestEngineChatTemplateQwen3`
  **Expected Result**: `OK` and a printed `Model response: ...`
  **Troubleshooting**: If `python -m unittest` and the test name are split across 2 shell lines, you may see `Ran 0 tests` then `...: command not found`; keep them on the same line (or use `\` line continuation). If TPU is busy, check `ps -ef | grep sglang` and stop leftover processes
  **References**: N/A

- **Title**: SOP: Run test as root user on TPU VM
  **Prereqs**: SSH into TPU VM; conda env at `/root/miniconda3`; repo at `/root/sglang-jax`; plugins at `/root/plugins`
  **Steps**:
  - `rm -f /tmp/libtpu_lockfile`
  - `source /root/miniconda3/etc/profile.d/conda.sh && conda activate tunix`
  - `cd /root && JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m unittest plugins.sglang_jax.test_engine_chat_template_qwen3_14b.TestEngineChatTemplateQwen3`
  **Expected Result**: `OK` and a printed `Model response: ...`
  **Troubleshooting**: If TPU is busy, run `lsof -w /dev/accel0`; if empty but still blocked, `rm -f /tmp/libtpu_lockfile`
  **References**: N/A

## Mesh validation

- **Title**: SOP: Verify device-indexes works for mesh (TPU)
  **Prereqs**: SSH into TPU VM as root; conda env `sglang-jax` installed; `sglang-jax` installed in env
  **Steps**:
  - `TPU_NAME=sglang-jax-spot-qwen14b-20260110-114742; ZONE=us-central2-b`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; python - <<"PY"\nfrom sgl_jax.srt.server_args import ServerArgs\nfrom sgl_jax.srt.utils.mesh_utils import create_device_mesh\nargs = ServerArgs(model_path="Qwen/Qwen3-14B", device="tpu", tp_size=1, device_indexes=[0])\nmesh = create_device_mesh(ici_parallelism=[-1, args.tp_size], dcn_parallelism=[1, 1], device_indexes=args.device_indexes)\nprint("mesh.shape=", mesh.shape)\nprint("mesh.devices=", mesh.devices)\nPY'`
  **Expected Result**: `mesh.devices` only contains `TpuDevice(id=0, ...)` and `mesh.shape` is `{'data': 1, 'tensor': 1}`
  **Troubleshooting**: If `device {self.device} is not consistent with 'JAX_PLATFORMS' ...`, unset `JAX_PLATFORMS` or set it to `tpu`
  **References**: `python/sgl_jax/srt/server_args.py` ; `python/sgl_jax/srt/utils/mesh_utils.py`

## Concurrency behavior

- **Title**: SOP: Test multi-engine concurrency on one TPU VM (Qwen3-4B)
  **Prereqs**: TPU VM `sglang-jax-spot-qwen14b-20260110-114742` in `us-central2-b`; repo at `/root/sglang-jax`; conda env `sglang-jax` (python 3.12 + jax 0.8.1); local helper script at `/home/john/test_sglang_jax/scripts/run_engine_once_qwen3_4b.py`
  **Steps**:
  - `TPU_NAME=sglang-jax-spot-qwen14b-20260110-114742; ZONE=us-central2-b`
  - (Optional) confirm Qwen3-4B context length is very large:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; python - <<"PY"\nfrom sgl_jax.srt.hf_transformers_utils import get_config, get_hf_text_config, get_context_length\nm="Qwen/Qwen3-4B-Instruct-2507"\nconf=get_config(m, trust_remote_code=True)\ntext=get_hf_text_config(conf)\nprint(getattr(text,"max_position_embeddings",None))\nprint(get_context_length(text))\nPY'`
  - Copy the single-engine runner onto the TPU VM:
    - `gcloud alpha compute tpus tpu-vm scp /home/john/test_sglang_jax/scripts/run_engine_once_qwen3_4b.py root@"$TPU_NAME":/root/sglang-jax/test/srt/run_engine_once_qwen3_4b.py --zone="$ZONE" --quiet`
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'chmod +x /root/sglang-jax/test/srt/run_engine_once_qwen3_4b.py'`
  - Run 1 Engine (works):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_single; mkdir -p "$JAX_COMPILATION_CACHE_DIR"; python -u /root/sglang-jax/test/srt/run_engine_once_qwen3_4b.py --device-index 0 --model Qwen/Qwen3-4B-Instruct-2507'`
  - Try to run 2 Engines as separate processes (expected to fail: TPU backend is single-process):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; rm -f /tmp/libtpu_lockfile || true; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/sglang-jax/test/srt; (JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_dev0 python -u run_engine_once_qwen3_4b.py --device-index 0 --hold-seconds 60 >/tmp/engine_dev0.log 2>&1) & sleep 5; (JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache_dev1 python -u run_engine_once_qwen3_4b.py --device-index 1 >/tmp/engine_dev1.log 2>&1) || true; tail -n 5 /tmp/engine_dev1.log || true'`
  **Expected Result**: Single Engine prints `device_index=0 OK: ...`. Second process fails with `RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by process with pid ...` (libtpu multi-process lock).
  **Troubleshooting**: If you see `Internal error when accessing libtpu multi-process lockfile`, run `rm -f /tmp/libtpu_lockfile` and ensure no leftover sglang processes are running.
  **References**: `python/sgl_jax/srt/utils/mesh_utils.py` ; `python/sgl_jax/srt/configs/model_config.py` ; `python/sgl_jax/srt/layers/attention/flashattention_backend.py`
