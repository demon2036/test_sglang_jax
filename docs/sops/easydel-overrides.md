# EasyDeL Override SOPs

- **Title**: SOP: Upload EasyDeL overrides to TPU VM (non-invasive)
  **Prereqs**: TPU VM `easydel-grpo-gsm8k-v5e-4-20260110-090113-eu` in `europe-west4-b`; local overrides in `plugins/easydel/overrides`
  **Steps**:
  - `scripts/overlay_easydel_overrides_to_tpu.sh easydel-grpo-gsm8k-v5e-4-20260110-090113-eu europe-west4-b`
  - `TPU_NAME=easydel-grpo-gsm8k-v5e-4-20260110-090113-eu; ZONE=europe-west4-b; gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command "source /root/miniconda3/etc/profile.d/conda.sh; conda activate easydel; PYTHONPATH=/root/easydel_overrides EASYDEL_OVERRIDES_VERBOSE=1 python -c 'import sys; print(\"override sys.path head:\", sys.path[0])'" --quiet`
  **Expected Result**: Upload completes; Python prints `easydel overrides patch module loaded` and `easydel overrides loaded: easydel_overrides_patch`
  **Troubleshooting**: If the override log does not appear, confirm `PYTHONPATH=/root/easydel_overrides` and `EASYDEL_OVERRIDES_VERBOSE=1`
  **References**: N/A
