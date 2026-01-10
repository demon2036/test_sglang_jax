#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 TPU_NAME ZONE [REMOTE_DIR] [COORDINATOR_PORT]" >&2
    exit 1
fi

TPU_NAME="$1"
ZONE="$2"
REMOTE_DIR="${3:-/root/easydel_plugins}"
COORDINATOR_PORT="${4:-${EASYDEL_COORDINATOR_PORT:-8476}}"

if [ ! -f "plugins/easydel/run_grpo_gsm8k_10steps.py" ]; then
    echo "Missing plugins/easydel/run_grpo_gsm8k_10steps.py; run from repo root." >&2
    exit 1
fi

mapfile -t ENDPOINTS < <(
    gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" \
        --format='value(networkEndpoints.ipAddress)' | tr ';' '\n' | awk 'NF'
)

if [ "${#ENDPOINTS[@]}" -lt 2 ]; then
    echo "TPU $TPU_NAME reports ${#ENDPOINTS[@]} worker(s); need >=2 for multi-host run." >&2
    exit 1
fi

COORDINATOR_IP="${ENDPOINTS[0]}"
COORDINATOR_ADDRESS="${COORDINATOR_IP}:${COORDINATOR_PORT}"
PROCESS_COUNT="${#ENDPOINTS[@]}"

echo "Using coordinator ${COORDINATOR_ADDRESS} with ${PROCESS_COUNT} workers."
echo "Tip: ensure ssh-agent has keys loaded for --worker=all (ssh-add -L)."

gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" \
    --zone="$ZONE" \
    --worker=all \
    --command "mkdir -p '$REMOTE_DIR'" \
    --quiet

gcloud alpha compute tpus tpu-vm scp \
    plugins/easydel/run_grpo_gsm8k_10steps.py \
    root@"$TPU_NAME":"$REMOTE_DIR"/ \
    --zone="$ZONE" \
    --worker=all \
    --quiet

REMOTE_CMD=$(cat <<'CMD'
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate easydel
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
export PYTHONPATH=/root/easydel_overrides:${PYTHONPATH:-}
export EASYDEL_JAX_DISTRIBUTED=1
export EASYDEL_JAX_COORDINATOR_ADDRESS="__COORDINATOR__"
export EASYDEL_JAX_PROCESS_COUNT="__PROCESS_COUNT__"

PROCESS_ID="${EASYDEL_JAX_PROCESS_ID:-${JAX_PROCESS_ID:-}}"
if [ -z "$PROCESS_ID" ] && [ -n "${TPU_WORKER_ID:-}" ]; then
    PROCESS_ID="$TPU_WORKER_ID"
fi
if [ -z "$PROCESS_ID" ]; then
    PROCESS_ID="$(hostname | sed -n 's/.*-w-\([0-9][0-9]*\)$/\1/p')"
fi
if [ -z "$PROCESS_ID" ]; then
    PROCESS_ID="0"
fi
export EASYDEL_JAX_PROCESS_ID="$PROCESS_ID"

python -u "__REMOTE_DIR__/run_grpo_gsm8k_10steps.py"
CMD
)

REMOTE_CMD=${REMOTE_CMD//__REMOTE_DIR__/$REMOTE_DIR}
REMOTE_CMD=${REMOTE_CMD//__COORDINATOR__/$COORDINATOR_ADDRESS}
REMOTE_CMD=${REMOTE_CMD//__PROCESS_COUNT__/$PROCESS_COUNT}

gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" \
    --zone="$ZONE" \
    --worker=all \
    --command "$REMOTE_CMD" \
    --quiet
