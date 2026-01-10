#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 TPU_NAME ZONE [ENV_NAME] [MINICONDA_DIR]" >&2
    exit 1
fi

TPU_NAME="$1"
ZONE="$2"
ENV_NAME="${3:-easydel}"
MINICONDA_DIR="${4:-/root/miniconda3}"

REMOTE_CMD=$(cat <<'CMD'
set -euo pipefail
if [ ! -d "__MINICONDA_DIR__" ]; then
    curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /root/miniconda.sh -b -p "__MINICONDA_DIR__"
    rm -f /root/miniconda.sh
fi
source "__MINICONDA_DIR__/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
if ! conda env list | awk '{print $1}' | grep -qx "__ENV_NAME__"; then
    conda create -y -n "__ENV_NAME__" python=3.12
fi
conda config --set auto_activate_base false
conda init bash >/tmp/conda_init.log 2>&1 || true
if ! grep -q "conda activate __ENV_NAME__" /root/.bashrc; then
    echo "conda activate __ENV_NAME__" >> /root/.bashrc
fi
conda activate "__ENV_NAME__"
python --version
conda --version
CMD
)

REMOTE_CMD=${REMOTE_CMD//__MINICONDA_DIR__/$MINICONDA_DIR}
REMOTE_CMD=${REMOTE_CMD//__ENV_NAME__/$ENV_NAME}

gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" \
    --zone="$ZONE" \
    --command "$REMOTE_CMD" \
    --quiet
