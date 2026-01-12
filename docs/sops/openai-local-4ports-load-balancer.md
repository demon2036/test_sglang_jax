# SOP: Local 4-port OpenAI servers + load balancer (no deps)

- **Title**: SOP: Run 4 local OpenAI-compatible servers + a load balancer (standard library only)
- **Prereqs**:
  - Windows PowerShell 5.1 (verified) or any OS with Python
  - Python `3.13.9` (verified)

## Steps

### 1) Start 4 local mock OpenAI servers (4 ports) + a load balancer

- (Terminal 1) Start a 4-backend cluster + LB on fixed ports:
  - `python -m plugins.openai_lb.run_local_cluster --host 127.0.0.1 --num-backends 4 --backend-base-port 32100 --lb-port 32000 --quiet`

### 2) Send a request to the load balancer

- (Terminal 2) POST an OpenAI-style chat completion to the LB:
  - `$payload = @{ model = 'mock-model'; messages = @(@{ role = 'user'; content = 'hello' }) } | ConvertTo-Json -Compress`
  - `Invoke-RestMethod -Uri http://127.0.0.1:32000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $payload | ConvertTo-Json -Compress -Depth 10`

### 3) (Optional) Expose the load balancer to external callers

- Start backends on localhost, but bind the LB to `0.0.0.0`:
  - `python -m plugins.openai_lb.run_local_cluster --backend-host 127.0.0.1 --lb-host 0.0.0.0 --num-backends 4 --backend-base-port 32110 --lb-port 32010 --quiet`
- Local verification request:
  - `$payload = @{ model = 'mock-model'; messages = @(@{ role = 'user'; content = 'hello' }) } | ConvertTo-Json -Compress`
  - `Invoke-RestMethod -Uri http://127.0.0.1:32010/v1/chat/completions -Method Post -ContentType 'application/json' -Body $payload | ConvertTo-Json -Compress -Depth 10`

### 4) Run automated tests

- `python -m unittest discover -s tests -v`

## Expected Result

- The cluster prints `READY` lines for 4 backends and the LB.
- The request returns JSON with content like `"[backend=<id> version=0] hello"`.
- `python -m unittest discover -s tests -v` ends with `OK`.

## Troubleshooting

- If you run `python -m unittest -v`, it may fail due to optional sglang-jax tests under `plugins/sglang_jax/` (needs `sgl_jax` installed); use `discover -s tests`.
- If ports are in use, change `--backend-base-port` / `--lb-port`.

## References

- `plugins/openai_lb/mock_openai_server.py`
- `plugins/openai_lb/openai_load_balancer.py`
- `plugins/openai_lb/run_local_cluster.py`
- `tests/test_openai_lb_integration.py`

