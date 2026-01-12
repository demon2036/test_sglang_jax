# SOP Index

SOPs live under `docs/sops/`. This index is organized for fast lookup in 3 ways:

1) **Quick tasks**: if you know what you want to do.
2) **Browse by area**: if you’re exploring by topic/component.
3) **Search recipes**: if you want to grep for a command, env var, or error.

## Quick tasks (task-based)

### Repo + docs

- Set up / verify repo clones: `docs/sops/repo-setup.md`
- Validate network access (web lookups): `docs/sops/network-checks.md`
- Update AGENTS + doc conventions: `docs/sops/docs-maintenance.md`

### Git workflows

- Use worktrees for parallel work: `docs/sops/git-worktrees.md`
- Pull/update safely: `docs/sops/git-pull-update.md`
- Commit/push workflow: `docs/sops/github-push.md`

### Local dev

- 4 local OpenAI ports + load balancer + tests: `docs/sops/openai-local-4ports-load-balancer.md`

### TPU VM: create / access / cleanup

- Find/SSH/create/delete TPU VMs: `docs/sops/tpu-vm-lifecycle.md`
- Bootstrap conda + deps on TPU VM: `docs/sops/tpu-vm-bootstrap.md`
- Runtime image/version mapping: `docs/sops/tpu-vm-runtime.md`
- Delete ALL TPU VMs across zones: `docs/sops/tpu-vm-delete-all.md`
- Quick live-resource checks (PowerShell): `docs/sops/tpu-alive-check.md`

### TPU validation + scaling

- Unittests, mesh checks, concurrency behavior: `docs/sops/tpu-tests.md`
- Rollout scaling guidance: `docs/sops/scaling-rollouts.md`
- Compare HTTP fanout vs Tunix rollout: `docs/sops/single-controller-vs-tunix.md`

### sglang-jax (TPU)

- Run OpenAI-compatible multi-server (single process): `docs/sops/sglang-jax-openai-multi-server.md`
- Multi-engine threads attempt: `docs/sops/sglang-jax-multi-engine-threads.md`
- 2-worker pod, 8 OpenAI ports: `docs/sops/sglang-jax-tpu-pod-2vm-8-openai-servers.md`
- Weight reload without restart: `docs/sops/sglang-jax-weight-reload.md`
- In-process hot update demo: `docs/sops/sglang-jax-inprocess-hot-update.md`

### GRPO frameworks

- EasyDeL overrides: `docs/sops/easydel-overrides.md`
- EasyDeL training + smoke tests: `docs/sops/easydel-training.md`
- EasyDeL GRPO mechanics notes: `docs/sops/easydel-grpo-mechanics.md`
- Tunix integration notes: `docs/sops/tunix-integration.md`
- Evaluate EasyDeL vs Tunix readiness: `docs/sops/grpo-framework-evaluation.md`

## Browse by area (component-based)

### Getting started

- `docs/sops/repo-setup.md`: cloning and repo verification
- `docs/sops/network-checks.md`: network verification
- `docs/sops/docs-maintenance.md`: AGENTS/docs updates

### Git

- `docs/sops/git-worktrees.md`: work on multiple branches in parallel
- `docs/sops/github-push.md`: commit and push workflow
- `docs/sops/git-pull-update.md`: update local repo safely

### Local dev

- `docs/sops/openai-local-4ports-load-balancer.md`: 4 local ports + load balancer + tests (no deps)

### TPU VM lifecycle and runtime

- `docs/sops/tpu-vm-lifecycle.md`: TPU VM discovery, SSH, provisioning, cleanup
- `docs/sops/tpu-vm-delete-all.md`: delete all TPU VMs across locations
- `docs/sops/tpu-vm-bootstrap.md`: conda + sglang-jax bootstrap on TPU VM
- `docs/sops/tpu-vm-runtime.md`: runtime images and version mapping
- `docs/sops/tpu-alive-check.md`: quick check for live TPU resources (PowerShell)

### TPU validation

- `docs/sops/tpu-tests.md`: TPU test runs, mesh checks, concurrency behavior

### sglang-jax (TPU)

- `docs/sops/sglang-jax-multi-engine-threads.md`: v4-8 single-process multi-engine thread attempt
- `docs/sops/sglang-jax-openai-multi-server.md`: v4-8 single-process multi-server OpenAI validation
- `docs/sops/sglang-jax-tpu-pod-2vm-8-openai-servers.md`: v4-16 2-worker pod, 8 OpenAI ports + commander/agent + fanout
- `docs/sops/sglang-jax-weight-reload.md`: reload weights without restarting HTTP server
- `docs/sops/sglang-jax-inprocess-hot-update.md`: in-process hot weight update (swap `model_state_leaves`)
- `docs/sops/scaling-rollouts.md`: rollout scaling guidance
- `docs/sops/single-controller-vs-tunix.md`: compare HTTP fanout vs Tunix in-process rollout

### EasyDeL

- `docs/sops/easydel-overrides.md`: EasyDeL non-invasive overrides
- `docs/sops/easydel-grpo-mechanics.md`: GRPO rollout + scheduling internals
- `docs/sops/easydel-training.md`: EasyDeL TPU install + GRPO smoke tests (single-host + multi-host)

### Tunix

- `docs/sops/tunix-integration.md`: Tunix integration notes

### Evaluations

- `docs/sops/grpo-framework-evaluation.md`: compare EasyDeL vs Tunix for GRPO readiness

## Search recipes (grep-first)

- List all SOP titles: `rg -n '^- \\*\\*Title\\*\\*:' docs/sops`
- Find TPU VM commands: `rg -n 'gcloud (alpha )?compute tpus tpu-vm' docs/sops`
- Find model selection knobs: `rg -n 'SGLANG_JAX_MODEL=|--model' docs/sops`
- Find common env vars: `rg -n 'TPU_NAME=|ZONE=|JAX_COMPILATION_CACHE_DIR' docs/sops`
- Find prerequisites blocks: `rg -n '\\*\\*Prereqs\\*\\*:' docs/sops`
- Find expected-result blocks: `rg -n '\\*\\*Expected Result\\*\\*:' docs/sops`
- Find troubleshooting notes: `rg -n '\\*\\*Troubleshooting\\*\\*:' docs/sops`
