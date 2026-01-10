#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 TPU_NAME ZONE [REMOTE_DIR]" >&2
    exit 1
fi

TPU_NAME="$1"
ZONE="$2"
REMOTE_DIR="${3:-/root/easydel_overrides}"

if [ ! -d "plugins/easydel/overrides" ]; then
    echo "Missing plugins/easydel/overrides; run from repo root." >&2
    exit 1
fi

gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" \
    --zone="$ZONE" \
    --command "mkdir -p '$REMOTE_DIR'" \
    --quiet

gcloud alpha compute tpus tpu-vm scp --recurse \
    plugins/easydel/overrides/* \
    root@"$TPU_NAME":"$REMOTE_DIR"/ \
    --zone="$ZONE" \
    --quiet

cat <<'MESSAGE'
Overrides uploaded. Use them like:
  PYTHONPATH=/root/easydel_overrides:$PYTHONPATH python -c 'import easydel'
MESSAGE
