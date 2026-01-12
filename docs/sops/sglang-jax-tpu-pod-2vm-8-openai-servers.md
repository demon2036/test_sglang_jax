# SOP: 2-worker TPU Pod (v4-16) run 8 OpenAI servers (Qwen3-1.7B)

- **Title**: SOP: Start 8 OpenAI-compatible `sglang-jax` servers on a 2-worker TPU Pod (4 ports per worker) and fan out requests from the other VM
- **Prereqs**:
  - Windows PowerShell 5.1+
  - `gcloud` authenticated (this run: project `civil-rarity-482610-s5`)
  - Existing 2-worker TPU Pod in `us-central2-b` (this run used `v4-16`)
  - SSH key: `$HOME\.ssh\google_compute_engine`
  - On both TPU workers: conda env `sglang-jax` and repo at `/root/test_sglang_jax`

## Steps

### 1) Discover the pod and its worker IPs (PowerShell)

- Confirm project:
  - `gcloud config get-value project`
  - `gcloud auth list`

- List TPU VMs in zone:
  - `gcloud compute tpus tpu-vm list --zone=us-central2-b --format="table(name,state,acceleratorType,networkEndpoints[0].accessConfig.externalIp)"`

- Get both workers' external/internal IPs:
  - `gcloud compute tpus tpu-vm describe sglang-jax-v4-16-pod2vm-qwen3-1p7b-20260112-155552 --zone=us-central2-b --format='yaml(networkEndpoints)'`

This run:
- worker0 ext `107.167.170.11` / int `10.130.0.21`
- worker1 ext `35.186.2.41` / int `10.130.0.20`

### 2) SSH as root (PowerShell)

- `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@107.167.170.11 "whoami"`
- `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@35.186.2.41 "whoami"`

### 3) Confirm OS + Python/JAX versions (no TPU init)

- OS:
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@35.186.2.41 'cat /etc/os-release | head -n 5'`
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@107.167.170.11 'cat /etc/os-release | head -n 5'`

- Python/JAX (forces CPU backend; avoids libtpu lock while servers are running):
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@35.186.2.41 "source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; JAX_PLATFORMS=cpu python -c 'import sys, jax; print(sys.version.split()[0]); print(jax.__version__); print(jax.default_backend())'"`
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@107.167.170.11 "source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; JAX_PLATFORMS=cpu python -c 'import sys, jax; print(sys.version.split()[0]); print(jax.__version__); print(jax.default_backend())'"`

### 4) Sync plugin updates to both workers (PowerShell)

These updates are required for multi-host stability: `plugins/sglang_jax/run_multi_openai_servers.py` now does multi-host lockstep barriers (auto-enabled when `jax.process_count()>1` and `--num-servers>1`).

- worker1:
  - `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/sglang_jax/run_multi_openai_servers.py root@35.186.2.41:/root/test_sglang_jax/plugins/sglang_jax/run_multi_openai_servers.py`
  - `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/tpu_pod/worker_agent.py root@35.186.2.41:/root/test_sglang_jax/plugins/tpu_pod/worker_agent.py`
  - `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/tpu_pod/pod_commander.py root@35.186.2.41:/root/test_sglang_jax/plugins/tpu_pod/pod_commander.py`

- worker0:
  - `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/sglang_jax/run_multi_openai_servers.py root@107.167.170.11:/root/test_sglang_jax/plugins/sglang_jax/run_multi_openai_servers.py`
  - `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/tpu_pod/worker_agent.py root@107.167.170.11:/root/test_sglang_jax/plugins/tpu_pod/worker_agent.py`
  - `scp -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no plugins/tpu_pod/pod_commander.py root@107.167.170.11:/root/test_sglang_jax/plugins/tpu_pod/pod_commander.py`

### 5) Verify the remote worker agent (worker0, port 9010)

- Check the agent process:
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@107.167.170.11 "ps aux | grep -F 'plugins.tpu_pod.worker_agent' | grep -v grep || true"`

- Check agent health + status:
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@107.167.170.11 'curl -sS -m 2 http://127.0.0.1:9010/health -v || true; echo; curl -sS -m 2 http://127.0.0.1:9010/admin/status || true'`

### 6) (Re)start the pod commander on worker1 (node0)

- (If you have a stuck commander) find and kill it:
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@35.186.2.41 "ps aux | grep -F 'plugins.tpu_pod.pod_commander' | grep -v grep || true"`
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@35.186.2.41 "kill -9 28889 || true; sleep 0.5; ps aux | grep -F 'plugins.tpu_pod.pod_commander' | grep -v grep || true"`

- Start commander (background) on worker1, talking to worker0 agent over internal IP:
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@35.186.2.41 "source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; nohup python -u -m plugins.tpu_pod.pod_commander --backend-kind sglang-jax-multi-openai --model Qwen/Qwen3-1.7B --num-servers-per-worker 4 --base-port 31000 --bind-host 0.0.0.0 --load-format auto --context-length 4096 --max-total-tokens 4096 --max-prefill-tokens 4096 --remote-agents http://10.130.0.21:9010 --registry-host 0.0.0.0 --registry-port 9100 --advertise-host 10.130.0.20 --startup-timeout-sec 3600 --workdir /root/test_sglang_jax > /tmp/pod_commander_9100.log 2>&1 & echo POD_COMMANDER_PID=$!"`

### 7) Verify backends + registry

- Verify local backends (worker1):
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@35.186.2.41 'for p in 31000 31001 31002 31003; do echo PORT=$p; curl -sS -m 2 http://127.0.0.1:$p/v1/models | head -c 120 || true; echo; done'`

- Verify remote backends (worker0):
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@107.167.170.11 'for p in 31000 31001 31002 31003; do echo PORT=$p; curl -sS -m 2 http://127.0.0.1:$p/v1/models | head -c 120 || true; echo; done'`

- Verify registry (worker1, port 9100):
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@35.186.2.41 'curl -sS -m 2 http://127.0.0.1:9100/health -v || true; echo; curl -sS -m 2 http://127.0.0.1:9100/backends || true'`

### 8) Node1 fanout request to all 8 backends (prompt: 你是谁)

- Run the fanout client on worker0 using worker1's internal registry URL:
  - `ssh -i $HOME\.ssh\google_compute_engine -o StrictHostKeyChecking=no root@107.167.170.11 "source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/test_sglang_jax; python -u -m plugins.tpu_pod.fanout_client --registry-url http://10.130.0.20:9100 --model Qwen/Qwen3-1.7B --prompt '你是谁' --max-tokens 2048 --timeout-sec 600 --workers 8"`

### 9) Local regression tests (PowerShell)

- `python -m unittest discover -s tests -v`

## Expected Result

- Registry returns 8 endpoints: 4 ports on worker1 + 4 ports on worker0.
- Fanout prints 8 lines with `status=200`.

## Troubleshooting

- If you see `kex_exchange_identification: read: Connection reset`, retry the SSH command.
- If a second `python -c "import jax; ..."` fails with `The TPU is already in use by process with pid ...`:
  - Use `JAX_PLATFORMS=cpu` for version-only inspection, or stop the running server process.
- If multi-host starts fail with launch-group errors like `unexpected peer ... launch id mismatch`:
  - Re-sync `plugins/sglang_jax/run_multi_openai_servers.py` and confirm it includes the multi-host lockstep barriers.

## References

- `plugins/tpu_pod/pod_commander.py`
- `plugins/tpu_pod/worker_agent.py`
- `plugins/tpu_pod/fanout_client.py`
- `plugins/sglang_jax/run_multi_openai_servers.py`
- `docs/sops/tpu-vm-lifecycle.md`
