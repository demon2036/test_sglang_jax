# TPU VM Runtime SOPs

## Runtime inspection

- **Title**: SOP: Inspect TPU VM runtime image and tpu-runtime container (v6e vs v5e)
  **Prereqs**: TPU VMs `easydel-grpo-gsm8k-v6e-8-20260110-084616` (us-east1-d) and `easydel-grpo-gsm8k-v5e-4-20260110-090113-eu` (europe-west4-b) exist
  **Steps**:
  - `gcloud alpha compute tpus tpu-vm describe easydel-grpo-gsm8k-v6e-8-20260110-084616 --zone=us-east1-d --format=json`
  - `gcloud alpha compute tpus tpu-vm describe easydel-grpo-gsm8k-v5e-4-20260110-090113-eu --zone=europe-west4-b --format=json`
  - `TPU_NAME=easydel-grpo-gsm8k-v6e-8-20260110-084616; ZONE=us-east1-d; gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; echo "==os-release=="; cat /etc/os-release; echo "==uname=="; uname -r; echo "==tpu-runtime status=="; systemctl status tpu-runtime --no-pager | sed -n "1,18p"; echo "==tpu-runtime image=="; systemctl status tpu-runtime --no-pager | sed -n "18,26p"; echo "==docker images=="; docker image ls --digests | head -n 8; echo "==/etc tpu=="; ls /etc | grep -i tpu || true' --quiet`
  - `TPU_NAME=easydel-grpo-gsm8k-v5e-4-20260110-090113-eu; ZONE=europe-west4-b; gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; echo "==os-release=="; cat /etc/os-release; echo "==uname=="; uname -r; echo "==tpu-runtime status=="; systemctl status tpu-runtime --no-pager | sed -n "1,18p"; echo "==tpu-runtime image=="; systemctl status tpu-runtime --no-pager | sed -n "18,26p"; echo "==docker images=="; docker image ls --digests | head -n 8; echo "==/etc tpu=="; ls /etc | grep -i tpu || true' --quiet`
  **Expected Result**:
  - Both VMs report `runtimeVersion: tpu-ubuntu2204-base` in the describe output
  - `/etc/os-release` shows Ubuntu 22.04.2 LTS with kernel `5.19.0-1022-gcp`
  - `tpu-runtime` runs `gcr.io/cloud-tpu-v2-images/fake_tensorflow:latest` (digest `sha256:1d04195c7c24dc3564f11cc05ae037639d625ec8838debd95267b461b3d3113e`)
  **Troubleshooting**: N/A
  **References**: N/A

## Runtime version mapping

- **Title**: SOP: List TPU VM runtime versions and confirm v6e alpha (us-east1-d)
  **Prereqs**: gcloud configured; project `civil-rarity-482610-s5`
  **Steps**:
  - `gcloud compute tpus tpu-vm versions list --zone=us-east1-d --format='table(name,version,releaseDate)'`
  **Expected Result**:
  - Output includes `v2-alpha-tpuv6e` and `v6e-ubuntu-2404` in the runtime version list
  - Output includes `tpu-ubuntu2204-base` and other standard TPU VM runtimes
  **Troubleshooting**: If `gcloud alpha compute tpus tpu-vm runtime-versions list` returns `Invalid choice`, use `gcloud compute tpus tpu-vm versions list` instead
  **References**: N/A

- **Title**: SOP: Map TPU generation to runtime images (v4/v5e/v6e)
  **Prereqs**: gcloud configured; project `civil-rarity-482610-s5`; access to `us-central2-b`, `europe-west4-b`, `us-east1-d`
  **Steps**:
  - `gcloud compute tpus tpu-vm versions list --zone=us-central2-b --format='table(name,version,releaseDate)'`
  - `gcloud compute tpus tpu-vm versions list --zone=europe-west4-b --format='table(name,version,releaseDate)'`
  - `gcloud compute tpus tpu-vm versions list --zone=us-east1-d --format='table(name,version,releaseDate)'`
  **Expected Result**:
  - v4-related entries are present (for example, `tpu-vm-v4-base`, `tpu-vm-tf-2.10.0-v4`)
  - v5e-related entries are present (for example, `v5e-rocky9`)
  - v6e-related entries are present (for example, `v6e-ubuntu-2404`, `v2-alpha-tpuv6e`)
  - Generic Ubuntu base images are present (for example, `tpu-ubuntu2204-base`, `tpu-ubuntu2004-base`)
  **Notes**:
  - v4: `tpu-ubuntu2204-base` worked for the sglang-jax v4-8 unittest run (see earlier SOP); `tpu-vm-v4-base` is the generation-specific base runtime in the list
  - v5e: `tpu-ubuntu2204-base` worked for the EasyDeL GRPO GSM8K 10-step run; `v5e-rocky9` is the generation-specific base runtime in the list
  - v6e: `tpu-ubuntu2204-base` + JAX 0.8.2/libtpu 0.0.32 failed with `Failed to get global TPU topology`; try `v6e-ubuntu-2404` or `v2-alpha-tpuv6e` next
  **Troubleshooting**: If a zone returns fewer runtime versions, re-run after ensuring the TPU API is enabled and the project is set correctly
  **References**: N/A
