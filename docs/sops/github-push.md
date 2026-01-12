# GitHub Push SOPs

- **Title**: SOP: Commit and push local changes to GitHub
  **Prereqs**: `git` configured to authenticate with GitHub (HTTPS or SSH); repo at `<repo-path>`
  **Steps**:
  - Check working tree status:
    - `git status -sb`
  - Run local tests:
    - `python -m pytest -q`
  - Stage changes:
    - `git add docs plugins tests pytest.ini`
  - Commit with a Conventional Commit message:
    - `git commit -m "feat: add openai lb and tpu pod orchestration"`
  - Push to GitHub:
    - `git push origin main`
  **Expected Result**: `git push` completes without errors and the `origin/main` branch shows the new commit.
  **Troubleshooting**: If push fails with auth errors, ensure your GitHub auth is configured for the remote URL scheme.
  **References**: N/A
