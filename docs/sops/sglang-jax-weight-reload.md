# SOP: Reload sglang-jax weights without restarting HTTP server

- **Title**: SOP: Reload sglang-jax weights without restarting the HTTP server process
- **Prereqs**:
  - Single-host `sglang-jax` server (TPU/GPU/CPU all OK as long as the model can load)
  - New checkpoint has the **same architecture + shapes** as the currently served model
  - You accept a short "maintenance window" (requests may return 503 during reload)

## Background (what's actually possible)

- Upstream `sglang-jax` does **not** ship a stable "hot reload checkpoint" API for the default `sgl_jax.launch_server` entrypoint.
- The "true hot update" pattern proven in Tunix is **in-process**: overwrite `model_runner.model_state_leaves` (see `tunix/generate/sglang_jax_sampler.py::SglangJaxSampler.update_params`).
- For an **HTTP server** process, the practical options are:
  - **Blue/green**: start a new server with new weights, switch traffic, then kill old server (zero downtime, costs more TPU).
  - **Reload-in-place (custom)**: keep the HTTP server process alive, call `model.load_weights(...)`, and refresh `model_state_leaves` (short downtime, custom code).

## Steps (reload-in-place via this repo's plugin)

This repo's `plugins/sglang_jax/run_multi_openai_servers.py` can expose admin endpoints when started with `--enable-weight-reload`.

- Start servers with admin endpoints enabled (verified on TPU v5litepod-4 for startup + checksum/perturb endpoints):
  - `python -u -m plugins.sglang_jax.run_multi_openai_servers --num-servers 4 --base-port 31000 --load-format dummy --enable-weight-reload --keep-running ...`

- Reload weights on a specific port (**template; not yet verified end-to-end**):
  - `curl -sS -X POST "http://127.0.0.1:31000/admin/reload_weights" -H "Content-Type: application/json" -d '{"model_path":"/path/to/new_checkpoint_dir"}'`
  - Optional fields:
    - `revision`: for HF repo IDs (when `model_path` is not a local directory)
    - `load_format`: set to `"dummy"` to force dummy weights; otherwise keep current behavior

- Check reload status:
  - `curl -sS "http://127.0.0.1:31000/admin/reload_status"`

- (Optional) use the helper client in this repo:
  - `python -m plugins.sglang_jax.admin_weight_endpoints_client --base-url http://127.0.0.1:31000 reload --model-path /path/to/new_checkpoint_dir`

## Expected Result

- During reload: non-admin endpoints return HTTP 503.
- After reload: `POST /admin/reload_weights` returns `{"ok": true, ...}` and new requests use the new weights.

## Troubleshooting

- If reload fails immediately with `No file found for weight...`: the checkpoint dir is missing `*.safetensors` or the model arch/shapes don't match.
- If reload returns 200 but outputs look "stuck", ensure caches are flushed (the plugin calls `engine.async_flush_cache()` before/after reload).
- If you started with `--load-format dummy`, your `ModelConfig` may be in dummy mode; pass `{"load_format":"safetensors"}` (or restart without dummy) to load real weights.
- For large models, in-place reload can cause temporary memory spikes; prefer blue/green on a separate TPU VM when possible.

## References

- `plugins/sglang_jax/run_multi_openai_servers.py` (`/admin/reload_weights`)
- `plugins/sglang_jax/admin_weight_endpoints_client.py` (helper client)
- `docs/sops/sglang-jax-inprocess-hot-update.md` (verified in-process hot update demo + admin endpoints)
- `docs/sops/scaling-rollouts.md` (weight update strategy notes)
- `tunix/generate/sglang_jax_sampler.py::SglangJaxSampler.update_params` (in-memory weight swap via `model_state_leaves`)

