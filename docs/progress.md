# Progress Log

Last updated: 2026-01-10

This repo is a non-invasive validation playground for `sglang-jax` + `tunix` + TPU.
Rule: do not modify upstream `tunix/` or `sglang-jax/` code; all new work lives under `plugins/` (and helper SOP/scripts under `docs/` / `scripts/`).

## What’s done (verified)

- TPU VM provisioning + bootstrap SOPs captured under `docs/sops/`.
- `sglang-jax` unittest path fixed (avoid `Ran 0 tests` by keeping `python -m unittest ...` on one shell line).
  - See: `docs/sops/tpu-tests.md` (unittest section).
- Tunix + sglang-jax integration inspected:
  - Tunix uses `Engine` **in the same Python process** (threaded mode via `enable_single_process=True`), then hot-swaps weights in-memory.
  - See: `docs/sops/tunix-integration.md`.
- GRPO smoke runs added as **plugins** (no upstream edits):
  - `plugins/tunix/run_grpo_gsm8k_qwen3_4b_10steps.py` (Tunix GRPO GSM8K, Qwen3-4B).
  - `plugins/easydel/run_grpo_gsm8k_10steps.py` (EasyDeL GRPO GSM8K, multi-host init support).
- Multi-host runner helper added:
  - `scripts/run_easydel_grpo_gsm8k_10steps_tpu_multihost.sh`

## Current TPU inventory (from `gcloud alpha compute tpus tpu-vm list`)

- `us-central2-b`
  - `tunix-grpo-qwen3-4b-v4-16-spot-20260110-180240` (v4-16, READY)
  - `easydel-grpo-gsm8k-v4-32-20260110-194536` (v4-32, READY)
- `europe-west4-b`
  - `easydel-grpo-gsm8k-v5e-4-20260110-090113-eu` (v5litepod-4, READY)

## In progress / to verify next

- **Single-host**: run `plugins/sglang_jax/run_multi_engine_threads.py` to validate “one process, N engines pinned to different `device_indexes`” concurrency.
  - Started once on v5e-4 but interrupted before completion; needs a clean rerun and log capture.
- **Multi-host**: verify sglang-jax rollout behavior:
  - Observed: multi-host + in-process sglang-jax rollout can hit TPU runtime errors (`program continuator halted`).
  - Workaround documented: use `--rollout-engine vanilla` on multi-host for now; or run sglang-jax as external single-host servers and send HTTP requests.
  - See: `docs/sops/scaling-rollouts.md`, `docs/sops/tunix-integration.md`.

## Next actions (high priority)

1. Finish the single-host multi-engine thread test on a clean TPU VM (v4-8 or v5e-4) and record logs + conclusions in `docs/sops/tpu-tests.md`.
2. Create a fresh multi-host spot TPU slice (2 workers) and rerun the Tunix GSM8K GRPO smoke for 10 steps, capturing step time + loss logs.
3. Document “TPU TRC quota zone selection + gcloud/gh workflows” as a dedicated SOP under `docs/sops/` (commands must be from actual runs).

