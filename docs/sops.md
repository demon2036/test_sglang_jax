# SOP Index

SOPs are grouped by area under `docs/sops/` for quick lookup.

## Getting started

- `docs/sops/repo-setup.md`: cloning and repo verification
- `docs/sops/network-checks.md`: network verification
- `docs/sops/docs-maintenance.md`: AGENTS/docs updates
- `docs/sops/git-worktrees.md`: work on multiple branches in parallel
- `docs/sops/github-push.md`: commit and push workflow

## Local dev

- `docs/sops/openai-local-4ports-load-balancer.md`: 4 local ports + load balancer + tests (no deps)

## TPU VM lifecycle and runtime

- `docs/sops/tpu-vm-lifecycle.md`: TPU VM discovery, SSH, provisioning, cleanup
- `docs/sops/tpu-vm-delete-all.md`: delete all TPU VMs across locations
- `docs/sops/tpu-vm-bootstrap.md`: conda + sglang-jax bootstrap on TPU VM
- `docs/sops/tpu-vm-runtime.md`: runtime images and version mapping

- `docs/sops/tpu-alive-check.md`: quick check for live TPU resources (PowerShell)

## TPU validation and scaling

- `docs/sops/tpu-tests.md`: TPU test runs, mesh checks, concurrency behavior    
- `docs/sops/sglang-jax-multi-engine-threads.md`: v4-8 single-process multi-engine thread attempt
- `docs/sops/sglang-jax-openai-multi-server.md`: v4-8 single-process multi-server OpenAI validation
- `docs/sops/sglang-jax-tpu-pod-2vm-8-openai-servers.md`: v4-16 2-worker pod, 8 OpenAI ports + commander/agent + fanout
- `docs/sops/sglang-jax-weight-reload.md`: reload weights without restarting HTTP server
- `docs/sops/sglang-jax-inprocess-hot-update.md`: in-process hot weight update (swap `model_state_leaves`)
- `docs/sops/scaling-rollouts.md`: rollout scaling guidance

## EasyDeL

- `docs/sops/easydel-overrides.md`: EasyDeL non-invasive overrides
- `docs/sops/easydel-grpo-mechanics.md`: GRPO rollout + scheduling internals
- `docs/sops/easydel-training.md`: EasyDeL TPU install + GRPO smoke tests (single-host + multi-host)

## Tunix

- `docs/sops/tunix-integration.md`: Tunix integration notes

## Evaluations

- `docs/sops/grpo-framework-evaluation.md`: compare EasyDeL vs Tunix for GRPO readiness
