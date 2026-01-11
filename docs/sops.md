# SOP Index

SOPs are grouped by area under `docs/sops/` for quick lookup.

## Getting started

- `docs/sops/repo-setup.md`: cloning and repo verification
- `docs/sops/network-checks.md`: network verification
- `docs/sops/docs-maintenance.md`: AGENTS/docs updates
- `docs/sops/git-worktrees.md`: work on multiple branches in parallel
- `docs/sops/github-push.md`: commit and push workflow

## TPU VM lifecycle and runtime

- `docs/sops/tpu-vm-lifecycle.md`: TPU VM discovery, SSH, provisioning, cleanup
- `docs/sops/tpu-vm-delete-all.md`: delete all TPU VMs across locations
- `docs/sops/tpu-vm-bootstrap.md`: conda + sglang-jax bootstrap on TPU VM
- `docs/sops/tpu-vm-runtime.md`: runtime images and version mapping

## TPU validation and scaling

- `docs/sops/tpu-tests.md`: TPU test runs, mesh checks, concurrency behavior
- `docs/sops/sglang-jax-multi-engine-threads.md`: v4-8 single-process multi-engine thread attempt
- `docs/sops/sglang-jax-openai-multi-server.md`: v4-8 single-process multi-server OpenAI validation
- `docs/sops/scaling-rollouts.md`: rollout scaling guidance

## EasyDeL

- `docs/sops/easydel-overrides.md`: EasyDeL non-invasive overrides
- `docs/sops/easydel-grpo-mechanics.md`: GRPO rollout + scheduling internals
- `docs/sops/easydel-training.md`: EasyDeL TPU install + GRPO smoke tests (single-host + multi-host)

## Tunix

- `docs/sops/tunix-integration.md`: Tunix integration notes

## Evaluations

- `docs/sops/grpo-framework-evaluation.md`: compare EasyDeL vs Tunix for GRPO readiness
