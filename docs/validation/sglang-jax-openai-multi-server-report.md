# SGLang-JAX Multi-Engine + OpenAI Multi-Server Validation Report

## Scope and constraints
- Non-invasive policy: all custom code lives under `plugins/`; no edits to upstream `sglang-jax/`.
- TPU single-process constraint: libtpu can only be initialized once per process.
- Goal: run 4 OpenAI-compatible servers (4 ports) in one process, each pinned to a different TPU device, and send concurrent requests.
- Validation uses dummy weights to avoid large model downloads.

## Code changes
- Added `plugins/sglang_jax/run_multi_openai_servers.py`.
  - Starts 4 uvicorn servers in threads, each with its own `Engine(enable_single_process=True)` and `device_indexes=[id]`.
  - Builds a minimal FastAPI app that exposes `/v1/models`, `/v1/chat/completions`, `/v1/completions`.
  - Includes `_patch_zero_penalty_cache()` to make the zero-penalty cache device-aware.
  - Includes `_patch_engine_signal_handlers()` to avoid signal registration in non-main threads.
  - Adds `--force-exit` to prevent hanging after validation.
- Updated `plugins/sglang_jax/run_multi_engine_threads.py`.
  - Adds the same device-aware zero-penalty cache patch.
  - Adds the signal handler patch for thread-safe Engine creation.
  - Introduces `EngineWorker` (one thread per Engine) with a queue, so `Engine.generate()` runs on the thread that created the event loop.
  - Adds `--warmup-sequential` and `--engine-per-thread` flags.
- Added/updated SOPs:
  - `docs/sops/sglang-jax-openai-multi-server.md`
  - `docs/sops/sglang-jax-multi-engine-threads.md`
  - `docs/sops/tpu-vm-delete-all.md`
  - `docs/sops/git-worktrees.md`
  - `docs/sops/tpu-vm-lifecycle.md` (new delete-all entry)
  - `docs/sops.md` index update

## Failed attempts and why they failed
1. **Multi-engine concurrency set to 1**  
   - `ThreadPoolExecutor(max_workers=1)` serialized all `generate()` calls, so only one engine worked at a time.

2. **Multi-engine concurrency > 1 with shared event loop**  
   - `Engine.generate()` uses `self.loop.run_until_complete()`; calling it from multiple threads on the same loop hangs.

3. **Device mismatch error during sampling**  
   - `SamplingMetadata.linear_penalty` uses `_zero_linear_penalty_cache` keyed only by shape.  
   - Cache reuse across engines pinned arrays to the first device, triggering:
     - `ValueError: Received incompatible devices for jitted computation`.
   - Fix: make cache key include sharding/device ids in `_patch_zero_penalty_cache()`.

4. **Engine-per-thread attempt still hung**  
   - Even with per-thread engine creation, the initial approach still waited indefinitely (no completion within 5 minutes).
   - Fix: switch to a queue-based `EngineWorker` so each engine runs its own event loop thread.

5. **Multi-server via subprocesses (one process per port)**  
   - Launching `sgl_jax.launch_server` in separate processes failed with:
     - `TPU is already in use by process ...`
   - Root cause: libtpu supports only one process per TPU host.
   - Also saw `os.fork()` warning from JAX multithreading.

6. **HTTP 422 errors on OpenAI endpoints**  
   - FastAPI treated `request` and `raw_request` as query params, returning:
     - `"Field required"` for `request` / `raw_request`.
   - Cause: `from __future__ import annotations` plus local imports led to unresolved annotations at runtime.
   - Fix: move FastAPI/OpenAI imports to module scope and remove the dependency wrapper.

7. **TPU init permission error**  
   - `open(/dev/accel2): Operation not permitted` occurred after a previous run left a process holding the device.
   - Fix: kill leftover processes and remove `/tmp/libtpu_lockfile` before retrying.

## Final solution (what worked)
- Single Python process.
- 4 threads, each creates its own `Engine(enable_single_process=True)` on a unique device index.
- Each thread runs a uvicorn server bound to a distinct port (31000-31003).
- Requests are sent concurrently to all 4 ports.

## Validation evidence
- TPU VM: `sglang-jax-v4-8-openai-20260111-215024` (v4-8, us-central2-b). Deleted after run.
- Log: `/tmp/multi_openai_servers_4ports.log`
- Key lines:
  - `READY: servers=4 ports=[31000, 31001, 31002, 31003] devices=[0, 1, 2, 3]`
  - `RESP: port=31000 status=200 ...`
  - `RESP: port=31001 status=200 ...`
  - `RESP: port=31002 status=200 ...`
  - `RESP: port=31003 status=200 ...`
  - `DONE: servers=4 wall_sec=0.726 max_req_sec=0.609`
- Concurrency confirmed: wall time is close to the slowest request, not the sum.

## Notes and limitations
- Dummy weights were used; this validates concurrency and API wiring, not throughput.
- In single-process threading mode, signal handlers are disabled in worker threads; `--force-exit` avoids lingering background threads.
