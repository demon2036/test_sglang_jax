# Docs Maintenance SOPs

- **Title**: SOP: Update AGENTS assistant execution policy
  **Prereqs**: N/A
  **Steps**:
  - Edit `AGENTS.md` and keep the policy short and action-focused
  - Prefer repo/SOP search and reasonable assumptions; ask only when blocked
  **Expected Result**: `AGENTS.md` guides assistants to implement end-to-end with minimal questions
  **Troubleshooting**: N/A
  **References**: N/A

- **Title**: SOP: Reorganize SOP index and split large TPU VM docs
  **Prereqs**: Ubuntu 25.04; Python 3.12.2; JAX not installed (import failed); CPU: AMD Ryzen 7 H 255 w/ Radeon 780M Graphics; GPU: `nvidia-smi` not found; repo at `/home/john/test_sglang_jax`
  **Steps**:
  - `ls -la docs`
  - `rg --files docs`
  - `sed -n '1,200p' docs/sops.md`
  - `wc -l docs/sops/*.md`
  - Split `docs/sops/tpu-vm-infra.md` into `docs/sops/tpu-vm-lifecycle.md` and `docs/sops/tpu-vm-runtime.md`, and add headings for quick scanning
  - Update `docs/sops.md` with grouped sections and new file paths
  **Expected Result**: SOP index is grouped by topic; TPU VM runtime notes live in `docs/sops/tpu-vm-runtime.md`
  **Troubleshooting**: If references still point to `docs/sops/tpu-vm-infra.md`, run `rg -n \"tpu-vm-infra\"` and update remaining mentions
  **References**: N/A

- **Title**: SOP: Require plan tool usage and fallback plan listing in AGENTS policy
  **Prereqs**: repo at `/home/john/test_sglang_jax`
  **Steps**:
  - `sed -n '1,200p' docs/sops/docs-maintenance.md`
  - `sed -n '1,200p' AGENTS.md`
  - Update the Assistant Execution Policy to require plan/update_plan usage when available, and list a plan before execution when tools are unavailable
  **Expected Result**: `AGENTS.md` states the plan-tool requirement and the no-confirmation fallback behavior
  **Troubleshooting**: N/A
  **References**: N/A
