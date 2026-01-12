# GitHub Push SOPs

- **Title**: SOP: Commit and push local changes to GitHub
  **Prereqs**: `git` and (optional) `gh` configured to authenticate with GitHub (HTTPS or SSH)
  **Environment (verified)**: Ubuntu 6.14; Python 3.12.2; git 2.48.1; gh 2.83.2
  **Steps**:
  - `cd /home/john/test_sglang_jax`
  - (Optional) Confirm GitHub auth + remote:
    - `gh auth status`
    - `git remote -v`
  - Check working tree status:
    - `git status -sb`
  - Run local tests:
    - `python -m pytest -q`
  - Stage changes (example from this repo):
    - `git add docs/sops.md docs/sops/docs-maintenance.md docs/sops/tunix-integration.md docs/sops/git-pull-update.md docs/sops/single-controller-vs-tunix.md`
  - Commit with a Conventional Commit message:
    - `git commit -m "docs: add single-controller vs tunix analysis"`
  - Push to GitHub:
    - `git push origin main`
  **Expected Result**: `git push` completes without errors and the `origin/main` branch shows the new commit.
  **Troubleshooting**: If push fails with auth errors, ensure your GitHub auth is configured for the remote URL scheme.
  **References**: N/A
