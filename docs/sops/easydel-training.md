# EasyDeL Training SOPs

- **Title**: SOP: Bootstrap EasyDeL repo + TPU deps on TPU VM
  **Prereqs**: TPU VM `easydel-grpo-gsm8k-v5e-4-20260110-090113-eu` in `europe-west4-b`; conda env `easydel` exists
  **Steps**:
  - `scripts/bootstrap_tpu_easydel_repo.sh easydel-grpo-gsm8k-v5e-4-20260110-090113-eu europe-west4-b easydel /root/easydel`
  **Expected Result**: EasyDeL installed in editable mode with TPU deps; output prints `jax 0.8.2` and `backend tpu`
  **Troubleshooting**:
  - This install pulls the CUDA-enabled `torch==2.8.0` wheel (large download); expect long install time
  - On `v6e-8` in `us-east1-d`, JAX init failed with `Failed to get global TPU topology` (libtpu 0.0.32)
  **References**: https://github.com/erfanzar/EasyDeL ; https://storage.googleapis.com/jax-releases/libtpu_releases.html

- **Title**: SOP: Run EasyDeL GRPO GSM8K 10-step smoke test on TPU
  **Prereqs**: EasyDeL installed on TPU VM; conda env `easydel`; plugin script at `plugins/easydel/run_grpo_gsm8k_10steps.py`
  **Steps**:
  - `scripts/run_easydel_grpo_gsm8k_10steps_tpu.sh easydel-grpo-gsm8k-v5e-4-20260110-090113-eu europe-west4-b`
  **Expected Result**: Training prints `train_step` metrics up to `10` (see `TrainerMetrics` logs)
  **Troubleshooting**:
  - Warnings like `Scheduler thread did not stop gracefully` appear during sampling
  - `WatchJobStateAsync failed` after step 10; safe to interrupt after confirming `train_step: 10`
  **References**: `plugins/easydel/run_grpo_gsm8k_10steps.py`

- **Title**: SOP: Run EasyDeL GRPO GSM8K 10-step multi-host (v4-32)
  **Prereqs**: gcloud configured; TPU VM `easydel-grpo-gsm8k-v4-32-20260110-194536` in `us-central2-b`; conda env `easydel` on all workers; plugin script `plugins/easydel/run_grpo_gsm8k_10steps.py` (multi-host init + `use_esurge_generation=False`)
  **Steps**:
  - `TPU_NAME=easydel-grpo-gsm8k-v4-32-20260110-194536; ZONE=us-central2-b; gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type=v4-32 --version=tpu-ubuntu2204-base --spot --quiet`
  - `TPU_NAME=easydel-grpo-gsm8k-v4-32-20260110-194536; ZONE=us-central2-b; gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --command 'set -euo pipefail; if [ ! -d "/root/miniconda3" ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | awk "{print \\$1}" | grep -qx easydel; then conda create -y -n easydel python=3.12; fi; conda config --set auto_activate_base false; conda activate easydel; python --version; conda --version' --quiet`
  - `TPU_NAME=easydel-grpo-gsm8k-v4-32-20260110-194536; ZONE=us-central2-b; gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --command 'set -euo pipefail; if [ ! -d "/root/easydel/.git" ]; then git clone https://github.com/erfanzar/EasyDeL.git /root/easydel; fi; source /root/miniconda3/etc/profile.d/conda.sh; conda activate easydel; pip install -U pip; pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; pip install -e "/root/easydel[tpu,torch]"; python - <<\"PY\"\nimport jax\nprint(\"jax\", jax.__version__)\nprint(\"backend\", jax.default_backend())\nPY' --quiet`
  - `TPU_NAME=easydel-grpo-gsm8k-v4-32-20260110-194536; ZONE=us-central2-b; gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=all --command 'mkdir -p /root/easydel_overrides' --quiet && gcloud alpha compute tpus tpu-vm scp --recurse plugins/easydel/overrides/* root@"$TPU_NAME":/root/easydel_overrides/ --zone="$ZONE" --worker=all --quiet`
  - `scripts/run_easydel_grpo_gsm8k_10steps_tpu_multihost.sh easydel-grpo-gsm8k-v4-32-20260110-194536 us-central2-b`
  **Expected Result**: `TrainerMetrics` logs reach `train_step: 10` with `process_count=4` and JAX backend `tpu`.
  **Troubleshooting**:
  - If you see `device_put` mismatches from `eSurgeEngine` on multi-host, ensure `use_esurge_generation=False` in `plugins/easydel/run_grpo_gsm8k_10steps.py`.
  **References**: `scripts/run_easydel_grpo_gsm8k_10steps_tpu_multihost.sh`, `plugins/easydel/run_grpo_gsm8k_10steps.py`
