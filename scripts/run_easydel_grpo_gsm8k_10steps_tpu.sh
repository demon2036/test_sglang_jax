#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 TPU_NAME ZONE [REMOTE_DIR]" >&2
    exit 1
fi

TPU_NAME="$1"
ZONE="$2"
REMOTE_DIR="${3:-/root/easydel_plugins}"

if [ ! -f "plugins/easydel/run_grpo_gsm8k_10steps.py" ]; then
    echo "Missing plugins/easydel/run_grpo_gsm8k_10steps.py; run from repo root." >&2
    exit 1
fi

gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" \
    --zone="$ZONE" \
    --command "mkdir -p '$REMOTE_DIR'" \
    --quiet

gcloud alpha compute tpus tpu-vm scp \
    plugins/easydel/run_grpo_gsm8k_10steps.py \
    root@"$TPU_NAME":"$REMOTE_DIR"/ \
    --zone="$ZONE" \
    --quiet

REMOTE_CMD=$(cat <<'CMD'
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate easydel
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
export PYTHONPATH=/root/easydel_overrides:${PYTHONPATH:-}
python -u "__REMOTE_DIR__/run_grpo_gsm8k_10steps.py"
CMD
)

REMOTE_CMD=${REMOTE_CMD//__REMOTE_DIR__/$REMOTE_DIR}

gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" \
    --zone="$ZONE" \
    --command "$REMOTE_CMD" \
    --quiet
