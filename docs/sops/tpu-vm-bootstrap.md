# TPU VM Bootstrap SOPs

- **Title**: SOP: Bootstrap conda + sglang-jax on TPU VM
  **Prereqs**: TPU VM `qwen3-1-7b-grpo-20260103-124738` in `us-central2-b`; Ubuntu on TPU VM
  **Steps**:
  - `gcloud alpha compute tpus tpu-vm ssh qwen3-1-7b-grpo-20260103-124738 --zone=us-central2-b --command 'set -euo pipefail; if [ ! -d "$HOME/miniconda3" ]; then curl -fsSL -o "$HOME/miniconda.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"; rm "$HOME/miniconda.sh"; fi; $HOME/miniconda3/bin/conda --version'`
  - `gcloud alpha compute tpus tpu-vm ssh qwen3-1-7b-grpo-20260103-124738 --zone=us-central2-b --command 'set -euo pipefail; source $HOME/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r; if ! conda env list | grep -q "^sglang-jax "; then conda create -y -n sglang-jax python=3.12; fi; conda activate sglang-jax; python --version; pip install -U pip; cd $HOME/sglang-jax; pip install -e "python[all]"'`
  **Expected Result**: `conda` available, `sglang-jax` env created, dependencies installed
  **Troubleshooting**: If conda prompts for Terms of Service, run the `conda tos accept` commands above
  **References**: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

- **Title**: SOP: Install default conda env for EasyDeL (TPU VM)
  **Prereqs**: TPU VM `easydel-grpo-gsm8k-v5e-4-20260110-090113-eu` in `europe-west4-b`; root SSH; network access for Miniconda download
  **Steps**:
  - `scripts/bootstrap_tpu_conda_default.sh easydel-grpo-gsm8k-v5e-4-20260110-090113-eu europe-west4-b easydel`
  **Expected Result**: `/root/miniconda3` installed, `easydel` env created with Python 3.12.12, and `conda` shows version `25.11.1`
  **Troubleshooting**: `auto_activate_base` alias warning is expected; ensure `/root/.bashrc` contains `conda activate easydel`
  **References**: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
