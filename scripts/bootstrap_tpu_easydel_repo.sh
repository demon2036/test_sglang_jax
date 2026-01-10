#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 TPU_NAME ZONE [ENV_NAME] [EASYDEL_DIR]" >&2
    exit 1
fi

TPU_NAME="$1"
ZONE="$2"
ENV_NAME="${3:-easydel}"
EASYDEL_DIR="${4:-/root/easydel}"

REMOTE_CMD=$(cat <<'CMD'
set -euo pipefail
if [ ! -d "__EASYDEL_DIR__/.git" ]; then
    git clone https://github.com/erfanzar/EasyDeL.git "__EASYDEL_DIR__"
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "__ENV_NAME__"

pip install -U pip
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e "__EASYDEL_DIR__[tpu,torch]"

python - <<'PY'
import jax
print("jax", jax.__version__)
print("backend", jax.default_backend())
PY
CMD
)

REMOTE_CMD=${REMOTE_CMD//__ENV_NAME__/$ENV_NAME}
REMOTE_CMD=${REMOTE_CMD//__EASYDEL_DIR__/$EASYDEL_DIR}

gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" \
    --zone="$ZONE" \
    --command "$REMOTE_CMD" \
    --quiet
