# TPU VM Lifecycle SOPs

## Discovery and access

- **Title**: SOP: Locate existing TPU VM in project
  **Prereqs**: gcloud configured; project `civil-rarity-482610-s5`
  **Steps**:
  - `gcloud alpha compute tpus locations list --format='value(locationId)' | xargs -P6 -I{} bash -lc 'gcloud alpha compute tpus tpu-vm list --zone="{}" --format="value(name,acceleratorType,state)" 2>/dev/null'`
  - `gcloud alpha compute tpus locations list --format='value(locationId)' | xargs -P6 -I{} bash -lc 'if gcloud alpha compute tpus tpu-vm list --zone="{}" --format="value(name)" 2>/dev/null | rg -q "^qwen3-1-7b-grpo-20260103-124738$"; then echo "{}"; fi'`
  - `gcloud alpha compute tpus tpu-vm describe qwen3-1-7b-grpo-20260103-124738 --zone=us-central2-b --format='yaml(name,acceleratorType,state,networkConfig,serviceAccount,health,apiVersion,version,labels,provisioningModel)'`
  **Expected Result**: Existing TPU VM `qwen3-1-7b-grpo-20260103-124738` found in `us-central2-b` with `v4-8`
  **Troubleshooting**: Cloud Asset API is disabled by default; `gcloud asset search-all-resources` will fail unless enabled
  **References**: N/A

- **Title**: SOP: SSH as root via gcloud (TPU VM)
  **Prereqs**: gcloud configured; TPU VM `qwen3-1-7b-grpo-20260103-124738` in `us-central2-b`
  **Steps**:
  - `gcloud alpha compute tpus tpu-vm ssh root@qwen3-1-7b-grpo-20260103-124738 --zone=us-central2-b --command 'whoami' --quiet`
  - `gcloud alpha compute tpus tpu-vm ssh qwen3-1-7b-grpo-20260103-124738 --zone=us-central2-b --command 'sudo -n whoami' --quiet`
  **Expected Result**: First command prints `root` (may show "Propagating SSH public key..."); second command also prints `root`
  **Troubleshooting**: If `root@...` fails, use the sudo command and/or check `sudo -n true` on the VM
  **References**: N/A

## Provisioning

- **Title**: SOP: Create spot TPU VM and run Qwen3-14B unittest
  **Prereqs**: Ubuntu 22.04 TPU VM runtime `tpu-ubuntu2204-base`; gcloud project `civil-rarity-482610-s5`; local repo at `/home/john/test_sglang_jax/sglang-jax`; spot capacity for `v4-8` in `us-central2-b`
  **Steps**:
  - `TPU_NAME=sglang-jax-spot-qwen14b-20260110-114742; ZONE=us-central2-b`
  - `gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type=v4-8 --version=tpu-ubuntu2204-base --spot --quiet`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'whoami; lsb_release -a || cat /etc/os-release; python3 --version || true' --quiet`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; if [ ! -d "/root/miniconda3" ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; if ! conda env list | awk "{print \\$1}" | grep -qx sglang-jax; then conda create -y -n sglang-jax python=3.12; fi; conda activate sglang-jax; pip install -U pip' --quiet`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; if [ ! -d /root/sglang-jax/.git ]; then git clone https://github.com/sgl-project/sglang-jax.git /root/sglang-jax; fi' --quiet`
  - `gcloud alpha compute tpus tpu-vm scp --recurse /home/john/test_sglang_jax/sglang-jax/plugins root@"$TPU_NAME":/root/sglang-jax/ --zone="$ZONE" --quiet`
  - `gcloud alpha compute tpus tpu-vm scp /home/john/test_sglang_jax/sglang-jax/test/srt/test_engine_chat_template_qwen3_14b.py root@"$TPU_NAME":/root/sglang-jax/test/srt/test_engine_chat_template_qwen3_14b.py --zone="$ZONE" --quiet`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/sglang-jax; pip install -e "python[all]"' --quiet`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/sglang-jax/test/srt; mkdir -p /tmp/jit_cache; JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -m unittest test_engine_chat_template_qwen3_14b.TestEngineChatTemplateQwen3' --quiet`
  **Expected Result**: `OK` printed; model response printed; env shows `python 3.12.12`, `jax 0.8.1`, `jaxlib 0.8.1`, `libtpu 0.0.30`, `jax.device_count=4`
  **Troubleshooting**: If TPU is `PREEMPTED`, create a new spot TPU VM and rerun; if you see TPU lock issues, try `rm -f /tmp/libtpu_lockfile` before rerunning
  **References**: https://github.com/sgl-project/sglang-jax ; https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

- **Title**: SOP: Create spot TPU VM for EasyDeL test (v4-8)
  **Prereqs**: gcloud configured; project `civil-rarity-482610-s5`; TRC spot quota in `us-central2-b`; TPU runtime `tpu-ubuntu2204-base`
  **Steps**:
  - `TPU_NAME=easydel-test-v4-8-20260110-154515; ZONE=us-central2-b`
  - `gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type=v4-8 --version=tpu-ubuntu2204-base --spot --quiet`
  - `gcloud alpha compute tpus tpu-vm list --zone=us-central2-b --filter='name:easydel-test-v4-8-20260110-154515'`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'whoami; lsb_release -a || cat /etc/os-release; python3 --version || true' --quiet`
  **Expected Result**: TPU VM exists in `READY` or `CREATING`; SSH prints `root`, Ubuntu 22.04.2 LTS, and `Python 3.10.6`
  **Troubleshooting**: If the create step hangs or fails for capacity, retry in another TRC-eligible zone
  **References**: N/A

- **Title**: SOP: Create spot TPU VM (v6e-8) in us-east1-d
  **Prereqs**: gcloud configured; project `civil-rarity-482610-s5`; TRC spot quota in `us-east1-d`; TPU runtime `tpu-ubuntu2204-base`
  **Steps**:
  - `TPU_NAME=easydel-grpo-gsm8k-v6e-8-20260110-084616; ZONE=us-east1-d`
  - `gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type=v6e-8 --version=tpu-ubuntu2204-base --spot --quiet`
  - `gcloud alpha compute tpus tpu-vm list --zone=us-east1-d`
  **Expected Result**: TPU VM shows `READY` in `us-east1-d` with accelerator type `v6e-8`
  **Troubleshooting**: JAX 0.8.2 + libtpu 0.0.32 failed with `Failed to get global TPU topology`; use a v5e VM in `europe-west4-b` if needed
  **References**: N/A

- **Title**: SOP: Create spot TPU VM (v5litepod-4) in europe-west4-b
  **Prereqs**: gcloud configured; project `civil-rarity-482610-s5`; TRC spot quota in `europe-west4-b`; TPU runtime `tpu-ubuntu2204-base`
  **Steps**:
  - `TPU_NAME=easydel-grpo-gsm8k-v5e-4-20260110-090113-eu; ZONE=europe-west4-b`
  - `gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type=v5litepod-4 --version=tpu-ubuntu2204-base --spot --quiet`
  - `gcloud alpha compute tpus tpu-vm list --zone=europe-west4-b`
  **Expected Result**: TPU VM shows `READY` in `europe-west4-b` with accelerator type `v5litepod-4`
  **Troubleshooting**:
  - `us-central2-b` returned `There is no more capacity` for `v4-8`
  - `us-central1-a` returned quota/permission errors for `v5litepod-8` and `v5litepod-4`
  **References**: N/A

## Cleanup

- **Title**: SOP: Delete PREEMPTED TPU VMs in us-central2-b
  **Prereqs**: gcloud configured; project `civil-rarity-482610-s5`
  **Steps**:
  - `gcloud alpha compute tpus tpu-vm list --zone=us-central2-b --filter='state=PREEMPTED' --format='value(name,zone)'`
  - `for name in sglang-jax-spot-qwen14b-20260110-114742 tunix-grpo-qwen3-4b-20260110-153427 qwen3-1-7b-grpo-20260103-124738 tunix-grpo-qwen3-4b-20260110-160644 easydel-test-v4-8-20260110-154515; do gcloud alpha compute tpus tpu-vm delete "$name" --zone=us-central2-b --quiet; done`
  - `gcloud alpha compute tpus tpu-vm list --zone=us-central2-b`
  **Expected Result**: PREEMPTED nodes are removed from `us-central2-b` listings
  **Troubleshooting**: If a delete fails, re-run the loop for the remaining node name(s)
  **References**: N/A
