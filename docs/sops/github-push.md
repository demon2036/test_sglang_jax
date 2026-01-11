# GitHub Push SOPs

- **Title**: SOP: Commit and push local changes to GitHub
  **Prereqs**: `git` configured with SSH access to `origin`; repo at `/home/john/github/test_sglang_jax`
  **Steps**:
  - Check working tree status:
    - `git status -sb`
  - Stage docs and plugins:
    - `git add docs plugins`
  - Commit with a Conventional Commit message:
    - `git commit -m "feat: add sglang-jax multi-server validation"`
  - Push to GitHub:
    - `git push origin main`
  **Expected Result**: `git push` completes without errors and the `origin/main` branch shows the new commit.
  **Troubleshooting**: If push fails with auth errors, re-run `ssh -T git@github.com` to validate SSH keys.
  **References**: N/A
