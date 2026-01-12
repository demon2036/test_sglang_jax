# SOP: Compare "single controller" HTTP rollout vs Tunix in-process rollout

- **Title**: SOP: Compare HTTP fanout rollout (ports + load balancer) vs Tunix in-process rollout (Engine as a library)
- **Prereqs**: Repo checked out; `rg` available

## Steps (what to read / inspect)

- Identify the "single controller" (HTTP) building blocks:
  - `sed -n '1,260p' docs/sops/sglang-jax-tpu-pod-2vm-8-openai-servers.md`
  - `sed -n '1,260p' plugins/tpu_pod/pod_commander.py`
  - `sed -n '1,520p' plugins/tpu_pod/worker_agent.py`
  - `sed -n '1,260p' plugins/openai_lb/openai_load_balancer.py`
  - `sed -n '1,260p' plugins/tpu_pod/fanout_client.py`
  - `sed -n '1,760p' plugins/sglang_jax/run_multi_openai_servers.py`

- Identify Tunix's sglang-jax integration (in-process Engine + weight sync):
  - `sed -n '1,260p' tunix/tunix/generate/sglang_jax_sampler.py`
  - `sed -n '1,240p' tunix/tunix/rl/rollout/sglang_jax_rollout.py`

- Cross-check rollout scaling guidance and known multi-host limitations:
  - `sed -n '1,220p' docs/sops/scaling-rollouts.md`
  - `sed -n '1,120p' docs/sops/tunix-integration.md`

## Architecture summary

### A) "Single controller" HTTP fanout (ports + LB)

- Each TPU host runs one serving process (TPU lockfile means **1 process per host**) and exposes one or many HTTP ports:
  - Single port: `python -m sgl_jax.launch_server ...`
  - Multi-port in one process (threads): `python -m plugins.sglang_jax.run_multi_openai_servers ...`
- A controller (often the trainer host) maintains a backend list (static or via a registry) and routes requests:
  - Registry example: `plugins/tpu_pod/pod_commander.py` + `plugins/tpu_pod/worker_agent.py`
  - LB example: `plugins/openai_lb/openai_load_balancer.py` (round-robin proxy)
  - Fanout example: `plugins/tpu_pod/fanout_client.py` (parallel requests)
- Weight updates: typically **restart servers** on a new checkpoint, or **reload from path** (`/admin/reload_weights` when enabled), or use LoRA-per-request if supported.

### B) Tunix in-process rollout (Engine as a library)

- One Python process owns the TPU (no other TPU processes), and runs **training + rollout** together.
- Rollout uses `Engine(**kwargs)` directly (no HTTP), with `enable_single_process=True`:
  - See `tunix/tunix/generate/sglang_jax_sampler.py::SglangJaxSampler`.
- Weight updates are **in-memory**: Tunix maps trainer weights into sglang-jax format and overwrites `model_runner.model_state_leaves` each update step.

## Efficiency comparison (rule of thumb)

- **If training needs TPU on the same host as rollout**:
  - Prefer **Tunix-style in-process** (HTTP servers in another process would block TPU access due to libtpu/PJRT lock).
- **If you can dedicate separate TPU hosts for rollout** (trainer on one host, rollout on other hosts):
  - The **HTTP fanout** design is the practical scale-out path today, even though each request pays serialization + network overhead.
- Per-request overhead:
  - Tunix is typically lower latency and higher efficiency per token (no HTTP/JSON, fewer copies, shared tokenizer/state).
  - HTTP fanout overhead matters most for short prompts / short completions and when using `Connection: close` (no keep-alive).
- Weight freshness:
  - Tunix can be fully on-policy (fast in-memory updates).
  - HTTP fanout usually uses stale weights for some interval unless you implement fast reload semantics.

## Known limitations / gotchas

- `sglang-jax` does not reliably support **multi-host in-process rollout** today (see `docs/sops/tunix-integration.md`); treat multi-host rollout as an **external service** for now.
- `sglang-jax` does not implement `--dp-size > 1` scheduling yet; for N-way rollout scaling, prefer **many v4-8 replicas** over one big multi-host slice unless you run one giant `--tp-size <total_devices>` engine.

## Expected Result

- You can choose the rollout topology based on:
  - hardware topology (single host vs many hosts),
  - weight update frequency requirements (per-step vs periodic),
  - throughput needs (scale-out vs per-request efficiency).

## References

- HTTP rollout + registry + fanout:
  - `docs/sops/sglang-jax-tpu-pod-2vm-8-openai-servers.md`
  - `plugins/tpu_pod/pod_commander.py`
  - `plugins/tpu_pod/worker_agent.py`
  - `plugins/openai_lb/openai_load_balancer.py`
  - `plugins/tpu_pod/fanout_client.py`
  - `plugins/sglang_jax/run_multi_openai_servers.py`
- Tunix integration:
  - `docs/sops/tunix-integration.md`
  - `tunix/tunix/generate/sglang_jax_sampler.py`
  - `tunix/tunix/rl/rollout/sglang_jax_rollout.py`
- Scaling guidance:
  - `docs/sops/scaling-rollouts.md`
