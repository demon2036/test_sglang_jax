# EasyDeL Local Overrides

This folder hosts non-invasive overrides for EasyDeL. We avoid editing the
upstream checkout by placing override files on `PYTHONPATH` when running on
TPU VMs.

## Usage (TPU VM)

1. Copy overrides to the TPU VM:
   - `scripts/overlay_easydel_overrides_to_tpu.sh <TPU_NAME> <ZONE>`
2. Run with overrides enabled:
   - `PYTHONPATH=/root/easydel_overrides:$PYTHONPATH python -c 'import easydel'`

Optional: set `EASYDEL_OVERRIDES_VERBOSE=1` to print activation logs.
