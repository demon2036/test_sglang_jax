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

- **Title**: SOP: Clone EasyDeL repo
  **Prereqs**: Ubuntu 25.04; Python 3.12.2; JAX not installed (import failed); CPU: AMD Ryzen 7 H 255 w/ Radeon 780M Graphics; GPU unknown (`nvidia-smi` not found)
  **Steps**:
  - `curl -s "https://api.github.com/search/repositories?q=easydel&per_page=5"`
  - `git clone https://github.com/erfanzar/EasyDeL.git easydel`
  **Expected Result**: `easydel/` exists as a git clone of `erfanzar/EasyDeL`
  **Troubleshooting**: If `easydel/` already exists, move it aside or remove it before re-cloning
  **References**: https://api.github.com/search/repositories?q=easydel&per_page=5 ; https://github.com/erfanzar/EasyDeL
