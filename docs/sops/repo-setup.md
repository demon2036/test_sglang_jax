# Repo Setup SOPs

- **Title**: SOP: Clone sglang-jax repo (placeholder)
  **Prereqs**: TBD (record OS, Python/JAX versions, hardware)
  **Steps**:
  - `git clone <sglang-jax-repo-url>`
  **Expected Result**: Repo cloned into the current directory
  **Troubleshooting**: N/A
  **References**: N/A

- **Title**: SOP: Verify existing sglang-jax clone and origin
  **Prereqs**: Ubuntu Linux; `git` available
  **Steps**:
  - `curl -s 'https://api.github.com/search/repositories?q=sglang-jax' | head -n 40`
  - `ls -la sglang-jax`
  - `git -C sglang-jax rev-parse --is-inside-work-tree`
  - `git -C sglang-jax remote -v`
  **Expected Result**: `sglang-jax` exists as a git repo and `origin` points to `https://github.com/sgl-project/sglang-jax.git`
  **Troubleshooting**: If the repo is missing or not a git repo, confirm the path before re-cloning
  **References**: https://api.github.com/search/repositories?q=sglang-jax

- **Title**: SOP: Clone `demon2036/test_sglang_jax` onto a TPU VM (root) and clone upstream deps
  **Prereqs**: gcloud configured; TPU VM reachable via `gcloud alpha compute tpus tpu-vm ssh`; network access to GitHub
  **Steps**:
  - `TPU_NAME=tunix-grpo-qwen3-4b-v4-16-spot-20260110-180240; ZONE=us-central2-b`
  - Clone this repo onto the TPU VM root:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=0 --quiet --command 'set -euo pipefail; rm -rf /root/test_sglang_jax_clone_test; git clone https://github.com/demon2036/test_sglang_jax.git /root/test_sglang_jax_clone_test; ls -la /root/test_sglang_jax_clone_test | head; ls -la /root/test_sglang_jax_clone_test/plugins | head'`
  - Clone upstream deps into the same folder (depth-1 for speed):
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --worker=0 --quiet --command 'set -euo pipefail; cd /root/test_sglang_jax_clone_test; if [ ! -d tunix/.git ]; then git clone --depth 1 https://github.com/google/tunix.git tunix; fi; if [ ! -d sglang-jax/.git ]; then git clone --depth 1 https://github.com/sgl-project/sglang-jax.git sglang-jax; fi'`
  **Expected Result**: `/root/test_sglang_jax_clone_test` exists with `plugins/`, and contains `tunix/` + `sglang-jax/` as git clones
  **Troubleshooting**: If `git clone` hangs, verify network on the TPU VM (`curl -I https://github.com`)
  **References**: https://github.com/demon2036/test_sglang_jax ; https://github.com/google/tunix ; https://github.com/sgl-project/sglang-jax

- **Title**: SOP: Clone EasyDeL repo
  **Prereqs**: Ubuntu 25.04; Python 3.12.2; JAX not installed (import failed); CPU: AMD Ryzen 7 H 255 w/ Radeon 780M Graphics; GPU unknown (`nvidia-smi` not found)
  **Steps**:
  - `curl -s "https://api.github.com/search/repositories?q=easydel&per_page=5"`
  - `git clone https://github.com/erfanzar/EasyDeL.git easydel`
  **Expected Result**: `easydel/` exists as a git clone of `erfanzar/EasyDeL`
  **Troubleshooting**: If `easydel/` already exists, move it aside or remove it before re-cloning
  **References**: https://api.github.com/search/repositories?q=easydel&per_page=5 ; https://github.com/erfanzar/EasyDeL
