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
